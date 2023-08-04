"""Proximal Policy Optimization (PPO) algorithm."""

import time
from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import optax
import flax

from safety_brax import jumpy as jp
from safety_brax.components import types, gradients, Evaluator, preprocessor
from safety_brax.algos import BaseAlgorithm
from safety_brax.algos.utils import compute_gae
from safety_brax.engine.io import html


@flax.struct.dataclass
class TrainingState:
    """Container for training state."""

    params: types.ActorCriticParams
    preprocessor_params: types.PreprocessorParams
    optimizer_state: types.OptState
    env_step: jnp.ndarray


class PPO(BaseAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm."""

    def __init__(self, env: types.Env, config: types.Config, algo_key: types.PRNGKey):
        """Initialize PPO algorithm.

        Args:
            env: The environment to train on, must be a subclass of 'Env'.
            config: The configuration dictionary.
            prng_key: The PRNG key used for training and rendering.
        """
        self.env = env
        self.config = config
        self.prng_key, actor_init_key, critic_init_key, eval_key = jax.random.split(
            algo_key, 4
        )

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
        # clip epsilon
        self.clip_epsilon = config["clip_epsilon"]
        # GAE lambda
        self.gae_lambda = config["gae_lambda"]
        # discount gamma
        self.discount_gamma = config["discount_gamma"]

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

        # create actor and critic
        self.actor = self._create_actor(config["actor_config"], actor_init_key)
        self.critic = self._create_critic(config["critic_config"], critic_init_key)

        # optimizer
        self.learning_params = self._params
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        # self.optimizer_state = self.optimizer.init(self.learning_params)
        self.gradient_update_fn = self._set_gradient_update_fn()

        # evaluator
        self.evaluator = Evaluator(
            env, self.actor, self.num_envs, config["max_episode_length"], eval_key
        )

    @property
    def _params(self):
        """Return parameters."""
        return types.ActorCriticParams(
            actor_params=self.actor.parameters, critic_params=self.critic.parameters
        )

    def _initialize_training_state(self):
        """Return training state."""
        return TrainingState(
            params=self.learning_params,
            preprocessor_params=self.preprocessor_params,
            optimizer_state=self.optimizer.init(self.learning_params),
            env_step=0,
        )

    def _load_params(
        self,
        params: types.ActorCriticParams,
        preprocessor_params: types.PreprocessorParams,
    ):
        """Load parameters."""
        self.learning_params = params
        self.preprocessor_params = preprocessor_params
        self.actor.load_state(params.actor_params, preprocessor_params)
        self.critic.load_state(params.critic_params, preprocessor_params)

    def _loss_fn(
        self,
        params: types.ActorCriticParams,
        other_params: dict,
        data: types.Transition,
        loss_key: types.PRNGKey,
    ):
        """Compute loss.

        Args:
            params: Actor and critic parameters.
            other_params: Other parameters is a dictionary with the following keys:
                ['preprocessor_params']: Preprocessor parameters.
            data: Transition data with leading dimension [batch_size, rollout_length, ...].
                extras field requires:
                    ['truncation'], ['raw_action'], ['log_prob']
            loss_key: PRNG key used for loss computation.

        """
        preprocessor_params = other_params["preprocessor_params"]
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

        policy_loss = -jnp.mean(
            jnp.minimum(ratio * advantages, ratio_clip * advantages)
        )

        # value function loss
        value_error = value_target - value_baseline
        value_loss = jnp.mean(value_error * value_error) * 0.5 * 0.5
        # jax.debug.print("loss:{}",value_loss)
        # entropy loss
        entropy_loss = -jnp.mean(entropy) * self.entropy_coefficient

        total_loss = policy_loss + value_loss + entropy_loss
        return total_loss, {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        }

    def rollout(
        self,
        training_state: TrainingState,
        rollout_state: types.State,
        rollout_length: int,
        rollout_key: types.PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ):
        """Rolls out the environment for `rollout_length` steps."""

        @jax.jit
        def step_fn(carry, _):
            """One step of rollout."""
            state, current_key = carry
            current_key, next_key = jp.random_split(current_key)
            action, action_info = self.actor.act_(
                training_state.params.actor_params,
                training_state.preprocessor_params,
                state.obs,
                current_key,
            )
            next_state = self.env.step(state, action)
            env_info = {k: next_state.info[k] for k in extra_fields}

            transition = types.Transition(
                observation=state.obs,
                action=action,
                reward=next_state.reward,
                cost=next_state.cost,
                done=next_state.done,
                next_observation=next_state.obs,
                extras={**action_info, **env_info},
            )
            return (next_state, next_key), transition

        (final_state, _), data = jax.lax.scan(
            step_fn, (rollout_state, rollout_key), None, length=rollout_length
        )

        return final_state, data

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
            optimizer_state=updated_optimizer_state,
            env_step=training_state.env_step + self.env_step_per_training_step,
        )
        return new_training_state, metrics

    def _training_step(
        self,
        training_state: TrainingState,
        current_state: types.State,
        step_key: types.PRNGKey,
    ):
        """Perform a training step."""
        rollout_key, update_key = jp.random_split(step_key)

        # *roll out a batch of data
        # construct a function that performs a rollout
        def rollout_step(carry, _):
            """One rollout step."""
            current_state, current_key = carry
            current_key, next_key = jp.random_split(current_key)
            next_state, data = self.rollout(
                training_state,
                current_state,
                self.rollout_length,
                current_key,
            )
            return (next_state, next_key), data

        # rollout num_rollouts_per_step times
        (final_state, _), data = jax.lax.scan(
            rollout_step,
            (current_state, rollout_key),
            None,
            length=self.num_rollouts_per_step,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )

        # *Update parameters
        new_training_state, metrics = self._update(
            training_state=training_state,
            batch=data,
            update_key=update_key,
            repeat_update_times=self.num_updates_per_step,
        )
        return new_training_state, final_state, metrics

    def _train_epoch(self, training_state: TrainingState, epoch_key: types.PRNGKey):
        """Train for one epoch."""
        # split the key for resetting the environment and performing training steps
        reset_key, step_key = jp.random_split(epoch_key)

        # reset the environment
        state = self.env.reset(reset_key)

        # construct a scan function that performs a training step
        def train_step(carry, _):
            """One training step."""
            current_training_state, current_state, current_key = carry
            current_key, next_key = jp.random_split(current_key)
            next_training_state, next_state, loss_metrics = self._training_step(
                current_training_state, current_state, current_key
            )
            return (next_training_state, next_state, next_key), loss_metrics

        # perform num_training_steps_per_epoch training steps
        (updated_training_state, final_state, _), loss_metrics = jax.lax.scan(
            train_step,
            (training_state, state, step_key),
            None,
            length=self.num_training_steps_per_epoch,
        )

        # compute the average loss metrics
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return updated_training_state, final_state, loss_metrics

    def train(self, train_key: types.PRNGKey = None, progress_fn: Callable = None):
        """Train PPO agent."""
        # construct needed parameters
        training_state = self._initialize_training_state()
        training_walltime = 0
        if train_key is not None:
            self.prng_key = train_key

        # JIT the training epoch function
        jit_epoch = jax.jit(self._train_epoch)

        for epoch_idx in range(self.num_epochs):
            t = time.time()

            # *training
            # perform an epoch
            epoch_key, self.prng_key = jp.random_split(self.prng_key)
            training_state, _, loss_metrics = jit_epoch(training_state, epoch_key)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), loss_metrics)
            # ?Is this needed without pmap?

            # calculate the time taken for the epoch
            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                self.num_training_steps_per_epoch * self.env_step_per_training_step
            ) / epoch_training_time
            train_metrics = {
                **{f"training/{k}": v for k, v in loss_metrics.items()},
                "training/sps": sps,
                "training/walltime": training_walltime,
            }

            # *evaluation
            self._load_params(training_state.params, training_state.preprocessor_params)

            if epoch_idx % self.eval_frequency == 0:
                # perform an evaluation
                eval_metrics = self.evaluator.eval()
                metrics = {
                    "training/total_env_steps": training_state.env_step,
                    **train_metrics,
                    **eval_metrics,
                }
                print(f"============ Epoch {epoch_idx} ============")
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                if progress_fn is not None:
                    progress_fn(metrics)

    def render(self, render_env: types.Env, render_length: int, filepath: str):
        """Renders the environment."""
        self.prng_key, actor_key, reset_key = jp.random_split(self.prng_key, 3)
        state = render_env.reset(reset_key)

        jit_act = jax.jit(self.actor.act)
        jit_step = jax.jit(render_env.step)

        rollout = [state.qp]
        for _ in range(render_length):
            act_key, actor_key = jp.random_split(actor_key)
            action, _ = jit_act(state.obs, act_key)
            # print('action', action)
            state = jit_step(state, action)
            rollout.append(state.qp)

        html.save_html(filepath, render_env.sys, rollout, True)
