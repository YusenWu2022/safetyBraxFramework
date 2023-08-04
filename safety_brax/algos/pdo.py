"""Primal-dual optimization (PDO) algorithm."""

import functools

import jax
import jax.numpy as jnp
import optax
import flax

from safety_brax import jumpy as jp
from safety_brax.algos import PPO
from safety_brax.envs import wrappers
from safety_brax.components import types, preprocessor, Evaluator, gradients
from safety_brax.algos.utils import compute_gae


@flax.struct.dataclass
class TrainingState:
    """Container for training state."""

    params: types.ConstrainedActorCriticParams
    preprocessor_params: types.PreprocessorParams
    lagrangian_multiplier: jnp.float32
    optimizer_state: types.OptState
    env_step: jnp.ndarray


class PDO(PPO):
    """Primal-dual optimization (PDO) algorithm."""

    def __init__(self, env: types.Env, config: types.Config, algo_key: types.PRNGKey):
        """Initialize PDO algorithm.

        Args:
            env: Environment.
            config: Algorithm configuration.
            algo_key: Algorithm seed.
        """
        self.env = (
            env if isinstance(env, wrappers.EvalWrapper) else wrappers.EvalWrapper(env)
        )
        self.config = config
        (
            self.prng_key,
            actor_init_key,
            critic_init_key,
            c_critic_init_key,
            eval_key,
        ) = jax.random.split(algo_key, 5)

        # *log training parameters
        # number of parallel environments
        self.num_envs = config["num_envs"]
        # number of epochs
        self.num_epochs = config["num_epochs"]
        # number of training steps per epoch to roll out training batch
        self.num_training_steps_per_epoch = config["num_training_steps_per_epoch"]
        # rollout length
        self.rollout_length = config["rollout_length"]
        # number of updates using the same batch
        self.num_updates_per_step = config["num_updates_per_step"]
        # number of minibatches to split the batch into
        self.num_minibatches = config["num_minibatches"]
        # size of each minibatch
        self.minibatch_size = config["minibatch_size"]
        # evaluation frequency
        self.eval_frequency = config["eval_frequency"]

        # learning rate
        self.learning_rate = config["learning_rate"]
        # entropy coefficient
        self.entropy_coefficient = config["entropy_coefficient"]
        # reward scaling
        self.reward_scaling = config["reward_scaling"]
        # cost scaling
        self.cost_scaling = config["cost_scaling"]
        # clip epsilon
        self.clip_epsilon = config["clip_epsilon"]
        # GAE lambda
        self.gae_lambda = config["gae_lambda"]
        # discount gamma
        self.discount_gamma = config["discount_gamma"]

        # Tolerance of constraint violation
        self.threshold = config["lagrange_config"]["threshold"]
        # Learning rate of lagrangian multiplier
        self.multiplier_lr = config["lagrange_config"]["multiplier_lr"]

        # *check parameters
        assert (
            self.num_envs == self.env.num_local_envs
        ), "Number of environments must match."  # ! Currently, only support parallel envs on the same machine.
        assert (
            self.num_minibatches * self.minibatch_size % self.num_envs == 0
        ), "Number of minibatches must be divisible by number of environments."

        # *compute needed parameters
        self.num_rollouts_per_step = (
            self.num_minibatches * self.minibatch_size // self.num_envs
        )
        self.env_step_per_training_step = (
            self.num_minibatches * self.minibatch_size * self.rollout_length
        )

        # *initialize components
        # observation preprocessor
        self.observation_preprocessor = preprocessor.normalize
        # self.observation_preprocessor = types.identity_observation_preprocessor
        self.preprocessor_params = preprocessor.init_state(
            types.Array((self.env.observation_size,), jnp.float32)
        )

        # create actor, critic, and constraint critic
        self.actor = self._create_actor(config["actor_config"], actor_init_key)
        self.critic = self._create_critic(config["critic_config"], critic_init_key)
        self.c_critic = self._create_critic(config["critic_config"], c_critic_init_key)

        # optimizer
        self.learning_params = self._params
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        self.optimizer_state = self.optimizer.init(self.learning_params)
        self.gradient_update_fn = self._set_gradient_update_fn()

        # evaluator
        self.evaluator = Evaluator(
            env, self.actor, self.num_envs, config["max_episode_length"], eval_key
        )

        # initialize lagrangian multiplier
        self.lagrangian_multiplier = config["lagrange_config"]["multiplier_init"]

    @property
    def _params(self):
        """Return parameters."""
        return types.ConstrainedActorCriticParams(
            actor_params=self.actor.parameters,
            critic_params=self.critic.parameters,
            c_critic_params=self.c_critic.parameters,
        )

    def _initialize_training_state(self):
        """Return training state."""
        return TrainingState(
            params=self.learning_params,
            preprocessor_params=self.preprocessor_params,
            lagrangian_multiplier=self.lagrangian_multiplier,
            optimizer_state=self.optimizer_state,
            env_step=0,
        )

    def _load_params(
        self,
        params: types.ConstrainedActorCriticParams,
        preprocessor_params: types.PreprocessorParams,
    ):
        """Load parameters."""
        self.learning_params = params
        self.observation_preprocessor = preprocessor_params
        self.actor.load_state(params.actor_params, preprocessor_params)
        self.critic.load_state(params.critic_params, preprocessor_params)
        self.c_critic.load_state(params.c_critic_params, preprocessor_params)

    def _loss_fn(
        self,
        params: types.ConstrainedActorCriticParams,
        # preprocessor_params: types.PreprocessorParams,
        # lagrangian_multiplier: jnp.float32,
        other_params: dict,
        data: types.Transition,
        loss_key: types.PRNGKey,
    ):
        """Compute loss.

        Args:
            params: Actor and critic parameters.
            other_params: Other parameters is a dictionary with the following keys:
                ['preprocessor_params']: Preprocessor parameters.
                ['lagrangian_multiplier']: Lagrangian multiplier.
            data: Transition data with leading dimension [batch_size, rollout_length, ...].
                extras field requires:
                    ['truncation'], ['raw_action'], ['log_prob']
            loss_key: random key for loss function.

        """
        preprocessor_params = other_params["preprocessor_params"]
        lagrangian_multiplier = other_params["lagrangian_multiplier"]

        # Put the time dimension first
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        # *Compute GAE
        value_baseline = self.critic.value_(
            params.critic_params, preprocessor_params, data.observation
        )
        bootstrap_value = self.critic.value_(
            params.critic_params, preprocessor_params, data.next_observation[-1]
        )
        value_target, advantages = compute_gae(
            truncation=data.extras["truncation"],
            termination=data.done * (1 - data.extras["truncation"]),
            rewards=data.reward * self.reward_scaling,
            values=value_baseline,
            bootstrap_value=bootstrap_value,
            lambda_=self.gae_lambda,
            discount=self.discount_gamma,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # *Compute constraint GAE
        c_value_baseline = self.c_critic.value_(
            params.c_critic_params, preprocessor_params, data.observation
        )
        c_bootstrap_value = self.c_critic.value_(
            params.c_critic_params, preprocessor_params, data.next_observation[-1]
        )
        c_value_target, c_advantages = compute_gae(
            truncation=data.extras["truncation"],
            termination=data.done * (1 - data.extras["truncation"]),
            rewards=data.cost * self.cost_scaling,
            values=c_value_baseline,
            bootstrap_value=c_bootstrap_value,
            lambda_=self.gae_lambda,
            discount=self.discount_gamma,
        )
        c_advantages = (c_advantages - c_advantages.mean()) / (
            c_advantages.std() + 1e-8
        )

        # *Compute loss
        # policy loss
        old_log_prob = data.extras["log_prob"]
        log_prob, entropy = self.actor.log_prob_(
            params.actor_params,
            preprocessor_params,
            data.observation,
            data.extras["raw_action"],
            loss_key,
        )
        ratio = jnp.exp(log_prob - old_log_prob)
        ratio_clip = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        reward_loss = -jnp.mean(
            jnp.minimum(ratio * advantages, ratio_clip * advantages)
        )
        cost_loss = jnp.mean(
            jnp.maximum(ratio * c_advantages, ratio_clip * c_advantages)
        )
        policy_loss = reward_loss + lagrangian_multiplier * cost_loss

        # value function loss
        value_error = value_target - value_baseline
        value_loss = jnp.mean(value_error * value_error) * 0.5 * 0.5

        # constraint value function loss
        c_value_error = c_value_target - c_value_baseline
        c_value_loss = jnp.mean(c_value_error * c_value_error) * 0.5 * 0.5

        # entropy loss
        entropy_loss = -jnp.mean(entropy) * self.entropy_coefficient

        # total loss
        total_loss = policy_loss + value_loss + c_value_loss + entropy_loss

        return total_loss, {
            "total_loss": total_loss,
            "reward_loss": reward_loss,
            "cost_loss": cost_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "c_value_loss": c_value_loss,
            "entropy_loss": entropy_loss,
        }

    def _update(
        self,
        training_state: TrainingState,
        batch: types.Transition,
        update_key: types.PRNGKey,
        repeat_update_times: int = 1,
    ):
        """Perform a gradient update with a batch of data."""

        # *update preprocessor parameters
        preprocessor_params = preprocessor.update(
            training_state.preprocessor_params,
            batch.observation,
        )

        # *update network parameters
        # construct a function to update params with a batch of data using SGD
        def sgd_fn(carry, _):
            """Update training state using SGD."""
            params, optimizer_state, current_key = carry
            sgd_key, perm_key, next_key = jp.random_split(current_key, 3)

            # convert batch to data with leading dimension [num_minibatches, minibatch_size, ...]
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(perm_key, x)
                x = jnp.reshape(x, (self.num_minibatches, -1) + x.shape[1:])
                return x

            shuffled_batch = jax.tree_util.tree_map(convert_data, batch)
            # assert shuffled_batch.discount.shape[1] == self.minibatch_size

            # update with SGD
            other_params = {
                "preprocessor_params": preprocessor_params,
                "lagrangian_multiplier": training_state.lagrangian_multiplier,
            }
            updated_params, updated_optimizer_state, metrics = gradients.sgd(
                self.gradient_update_fn,
                params,
                other_params,
                optimizer_state,
                shuffled_batch,
                self.num_minibatches,
                sgd_key,
            )
            return (updated_params, updated_optimizer_state, next_key), metrics

        # repeat the update for repeat_update_times times
        (updated_params, updated_optimizer_state, _), metrics = jax.lax.scan(
            sgd_fn,
            (training_state.params, training_state.optimizer_state, update_key),
            None,
            length=repeat_update_times,
        )

        new_training_state = TrainingState(
            params=updated_params,
            preprocessor_params=preprocessor_params,
            lagrangian_multiplier=training_state.lagrangian_multiplier,
            optimizer_state=updated_optimizer_state,
            env_step=training_state.env_step + self.env_step_per_training_step,
        )
        # metrics["constraint_violation"] = constraint_violation
        metrics["lagrangian_multiplier"] = new_training_state.lagrangian_multiplier
        return new_training_state, metrics

    def _train_epoch(self, training_state: TrainingState, epoch_key: types.PRNGKey):
        updated_training_state, final_state, metrics = super()._train_epoch(
            training_state, epoch_key
        )
        # *update lagrangian multiplier
        eval_metrics = final_state.info["eval_metrics"]
        # eval_metrics.active_episodes.block_until_ready()
        constraint_violation = jp.minimum(
            jp.mean(eval_metrics.episode_metrics["cost"]) - self.threshold,
            self.threshold,
        )
        # updated_multiplier = jp.maximum(
        #     0.0,
        #     updated_training_state.lagrangian_multiplier
        #     + self.multiplier_lr * constraint_violation,
        # )
        updated_multiplier = training_state.lagrangian_multiplier + self.multiplier_lr * constraint_violation
        updated_training_state = TrainingState(
            params=updated_training_state.params,
            preprocessor_params=updated_training_state.preprocessor_params,
            lagrangian_multiplier=updated_multiplier,
            optimizer_state=updated_training_state.optimizer_state,
            env_step=updated_training_state.env_step,
        )
        metrics["constraint_violation"] = constraint_violation
        metrics["lagrangian_multiplier"] = updated_multiplier
        return updated_training_state, final_state, metrics
