"""Short Horizon Actor-Critic (SHAC) algorithm."""

import time
import functools
from typing import Tuple, Callable, Sequence

import jax
import jax.numpy as jnp
import optax
import flax
from safety_brax.envs import wrappers
from safety_brax import jumpy as jp
from safety_brax.components import types, gradients, Evaluator, preprocessor
from safety_brax.algos import BaseAlgorithm
from safety_brax.algos.utils import compute_gae
from safety_brax.engine.io import html


@flax.struct.dataclass
class TrainingState:
    """Container for training state."""

    # params: types.ActorCriticParams
    actor_params: types.Params
    critic_params: types.Params
    cost_critic_params: types.Params
    actor_optimizer_state: types.OptState
    critic_optimizer_state: types.OptState
    preprocessor_params: types.PreprocessorParams
    lagrangian_multiplier: jnp.float32
    env_step: jnp.ndarray


class SHAC_Lag(BaseAlgorithm):
    """Short Horizon Actor-Critic (SHAC) algorithm."""

    def __init__(self, env: types.Env, config: dict, algo_key: types.PRNGKey):
        self.env = (
            env if isinstance(env, wrappers.EvalWrapper) else wrappers.EvalWrapper(env)
        )
        self.config = config
        self.prng_key, actor_init_key, critic_init_key, eval_key = jax.random.split(
            algo_key, 4
        )

        # lagrangian
        self.lagrangian_multiplier = config["lagrangian_multiplier"]
        self.lagrangian_learningrate = config["lagrangian_learning"]
        self.multiplier_lr = config["multiplier_lr"]
        self.cost_limit_grad = config["cost_limit_grad"]
        self.threshold = config["threshold"]

        # *log training parameters
        # number of parallel environments
        self.num_envs = config["num_envs"]
        # length of the episode
        self.max_episode_length = config["max_episode_length"]
        # number of epochs
        self.num_epochs = config["num_epochs"]
        # number of training steps per epoch to roll out training batch
        self.num_training_steps_per_epoch = config["num_training_steps_per_epoch"]
        # evaluation frequency
        self.eval_frequency = config["eval_frequency"]

        # learning rate
        self.learning_rate = config["learning_rate"]
        # short horizon to compute the gradient
        self.short_horizon = config["short_horizon"]
        # the maximum norm of the gradient for clipping
        self.max_grad_norm = config["max_grad_norm"]
        # minibatch size
        self.minibatch_size = config["minibatch_size"]
        # the repeat update times for critic update
        self.num_updates_per_step = config["num_updates_per_step"]
        # reward scaling
        self.reward_scaling = config["reward_scaling"]
        self.cost_scaling = config["cost_scaling"]
        # GAE lambda
        self.gae_lambda = config["gae_lambda"]
        # discount gamma
        self.discount_gamma = config["discount_gamma"]

        # *check parameters
        assert (
            self.num_envs == self.env.num_local_envs
        ), "Number of environments must match."  # ! Currently, only support parallel envs on the same machine.
        assert (
            self.num_envs % self.minibatch_size == 0
        ), "Number of environments must be divisible by minibatch size."
        self.num_minibatches = self.num_envs // self.minibatch_size
        self.env_step_per_training_step = self.num_envs * self.short_horizon

        # *initialize components
        # observation preprocessor
        self.observation_preprocessor = preprocessor.identity
        self.preprocessor_params = None
        # create actor
        self.actor = self._create_actor(config["actor_config"], actor_init_key)
        # create critic: shared between cost and reward
        self.critic = self._create_critic(config["critic_config"], critic_init_key)
        # optimizer
        self.actor_optimizer = optax.adam(self.learning_rate)
        self.critic_optimizer = optax.adam(self.learning_rate)
        # set gradient function
        actor_loss_fn = functools.partial(self._actor_loss_fn)
        critic_loss_fn = functools.partial(self._critic_loss_fn)
        cost_critic_loss_fn = functools.partial(self._cost_critic_loss_fn)
        # !not set pmap_axis_name for multi-GPU training
        self.actor_grad_fn = jax.jacfwd(actor_loss_fn, has_aux=True)
        # self.critic_grad_fn = jax.grad(critic_loss_fn, has_aux=True)
        self.critic_update_fn = gradients.gradient_update_fn(
            critic_loss_fn, self.critic_optimizer, pmap_axis_name=None, has_aux=True
        )
        self.cost_critic_update_fn = gradients.gradient_update_fn(
            cost_critic_loss_fn,
            self.critic_optimizer,
            pmap_axis_name=None,
            has_aux=True,
        )
        # create evaluator
        self.evaluator = Evaluator(
            env, self.actor, self.num_envs, self.max_episode_length, eval_key
        )

    def _initialize_training_state(self):
        """Return initial training state."""
        return TrainingState(
            actor_params=self.actor.parameters,
            critic_params=self.critic.parameters,
            cost_critic_params=self.critic.parameters,
            actor_optimizer_state=self.actor_optimizer.init(self.actor.parameters),
            critic_optimizer_state=self.critic_optimizer.init(self.critic.parameters),
            preprocessor_params=self.preprocessor_params,
            lagrangian_multiplier=self.lagrangian_multiplier,
            env_step=0,
        )

    def _load_params(self, training_state: TrainingState):
        """Load actor and critic parameters."""
        self.preprocessor_params = training_state.preprocessor_params
        self.actor.load_state(
            training_state.actor_params, training_state.preprocessor_params
        )
        self.critic.load_state(
            training_state.critic_params, training_state.preprocessor_params
        )

    def _actor_loss_fn(
        self,
        actor_params: types.Params,
        critic_params: types.Params,
        cost_critic_params: types.Params,
        preprocessor_params: types.PreprocessorParams,
        current_state: types.State,
        loss_key: types.PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ):
        """Compute actor loss."""
        # split actor loss key
        reset_key, scan_key = jax.random.split(loss_key)

        # *interact with the environment
        # construct env step function
        def env_step_fn(carry: Tuple[types.State, types.PRNGKey], _):
            """Environment step function."""
            current_state, current_key = carry
            action_key, next_key = jp.random_split(current_key)
            action, _ = self.actor.act_(
                actor_params, preprocessor_params, current_state.obs, action_key
            )
            next_state = self.env.step(current_state, action)
            env_info = {k: next_state.info[k] for k in extra_fields}

            transition = types.Transition(
                observation=current_state.obs,
                action=action,
                reward=next_state.reward,
                cost=next_state.cost,
                done=next_state.done,
                next_observation=next_state.obs,
                extras=env_info,
            )
            return (next_state, next_key), (
                next_state.reward,
                next_state.cost,
                transition,
            )

        # rollout short horizon steps
        (final_state, _), (rewards, costs, transitions) = jax.lax.scan(
            env_step_fn, (current_state, scan_key), None, self.short_horizon
        )
        value = self.critic.value_(critic_params, preprocessor_params, final_state.obs)
        cost_value = self.critic.value_(
            cost_critic_params, preprocessor_params, final_state.obs
        )
        actor_loss = jnp.mean(jnp.sum(rewards, axis=0) + value)
        cost_actor_loss = jnp.mean(jnp.sum(costs, axis=0) + cost_value)
        return [actor_loss, cost_actor_loss], {
            "transitions": transitions,
            "final_state": final_state,
            "actor_loss": actor_loss,
            "costs": costs,
        }

    def _critic_loss_fn(
        self,
        critic_params: types.Params,
        other_params: dict,
        data: types.Transition,
        loss_key: types.PRNGKey,
    ):
        """Compute critic loss."""
        preprocessor_params = other_params["preprocessor_params"]
        # Put the time dimension first
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        # *Compute GAE
        value_baseline = self.critic.value_(
            critic_params, preprocessor_params, data.observation
        )
        bootstrap_value = self.critic.value_(
            critic_params, preprocessor_params, data.next_observation[-1]
        )
        value_target, _ = compute_gae(
            truncation=data.extras["truncation"],
            termination=data.done * (1 - data.extras["truncation"]),
            rewards=data.reward * self.reward_scaling,
            values=value_baseline,
            bootstrap_value=bootstrap_value,
            lambda_=self.gae_lambda,
            discount=self.discount_gamma,
        )

        # *Compute critic loss
        value_error = value_target - value_baseline
        critic_loss = jnp.mean(value_error * value_error) * 0.5 * 0.5

        return critic_loss, {
            "critic_loss": critic_loss,
        }

    def _cost_critic_loss_fn(
        self,
        cost_critic_params: types.Params,
        other_params: dict,
        data: types.Transition,
        loss_key: types.PRNGKey,
    ):
        """Compute critic loss."""
        preprocessor_params = other_params["preprocessor_params"]
        # Put the time dimension first
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        # *Compute GAE
        cost_value_baseline = self.critic.value_(
            cost_critic_params, preprocessor_params, data.observation
        )
        cost_bootstrap_value = self.critic.value_(
            cost_critic_params, preprocessor_params, data.next_observation[-1]
        )
        cost_value_target, _ = compute_gae(
            truncation=data.extras["truncation"],
            termination=data.done * (1 - data.extras["truncation"]),
            rewards=data.cost * self.cost_scaling,
            values=cost_value_baseline,
            bootstrap_value=cost_bootstrap_value,
            lambda_=self.gae_lambda,
            discount=self.discount_gamma,
        )

        # *Compute critic loss
        cost_value_error = cost_value_target - cost_value_baseline
        cost_critic_loss = jnp.mean(cost_value_error * cost_value_error) * 0.5 * 0.5

        return cost_critic_loss, {
            "cost_critic_loss": cost_critic_loss,
        }

    def _update(
        self,
        training_state: TrainingState,
        current_state: types.State,
        update_key: types.PRNGKey,
    ):
        """Update the actor and critic. First, using the Short Horizon Back Propagation
        to update the actor and collect a batch of data. Then, using the collected data
        to update the critic.
        """

        # *update actor
        # compute actor gradient
        [actor_grad, actor_cost_grad], info = self.actor_grad_fn(
            training_state.actor_params,
            training_state.critic_params,
            training_state.cost_critic_params,
            training_state.preprocessor_params,
            current_state,
            update_key,
        )
        # actor_grad = gradients.clip_grads(actor_grad, self.max_grad_norm)
        # actor_cost_grad = gradients.clip_grads(actor_cost_grad, self.max_grad_norm)

        final_grad = jax.tree_map(
            lambda x, y: x - training_state.lagrangian_multiplier * y,
            actor_grad,
            actor_cost_grad,
        )
        final_grad = gradients.clip_grads(final_grad, self.max_grad_norm)

        # update actor parameters: from both reward and cost
        (
            actor_params_update,
            updated_actor_optimizer_state,
        ) = self.actor_optimizer.update(
            final_grad, training_state.actor_optimizer_state
        )
        updated_actor_params = optax.apply_updates(
            training_state.actor_params, actor_params_update
        )

        # *update critic
        batch = info["transitions"]

        # construct a function to update critic parameters using SGD
        def sgd_fn(carry, _):
            """Update training state using SGD."""
            critic_params, critic_optimizer_state, current_key = carry
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
                "preprocessor_params": training_state.preprocessor_params,
                "lagrangian_multiplier": training_state.lagrangian_multiplier,
            }
            (
                updated_critic_params,
                updated_critic_optimizer_state,
                metrics,
            ) = gradients.sgd(
                self.critic_update_fn,
                critic_params,
                other_params,
                critic_optimizer_state,
                shuffled_batch,
                self.num_minibatches,
                sgd_key,
            )
            return (
                updated_critic_params,
                updated_critic_optimizer_state,
                next_key,
            ), metrics

        def cost_sgd_fn(carry, _):
            """Update training state using SGD."""
            cost_critic_params, cost_critic_optimizer_state, current_key = carry
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
                "preprocessor_params": training_state.preprocessor_params,
                "lagrangian_multiplier": training_state.lagrangian_multiplier,
            }
            (
                updated_critic_params,
                updated_critic_optimizer_state,
                metrics,
            ) = gradients.sgd(
                self.cost_critic_update_fn,
                cost_critic_params,
                other_params,
                cost_critic_optimizer_state,
                shuffled_batch,
                self.num_minibatches,
                sgd_key,
            )
            return (
                updated_critic_params,
                updated_critic_optimizer_state,
                next_key,
            ), metrics

        # repeat the update for repeat_update_times times
        (
            updated_critic_params,
            updated_critic_optimizer_state,
            _,
        ), metrics = jax.lax.scan(
            sgd_fn,
            (
                training_state.critic_params,
                training_state.critic_optimizer_state,
                update_key,
            ),
            None,
            length=self.num_updates_per_step,
        )

        (
            updated_cost_critic_params,
            updated_critic_optimizer_state,
            _,
        ), metrics = jax.lax.scan(
            cost_sgd_fn,
            (
                training_state.cost_critic_params,
                training_state.critic_optimizer_state,
                update_key,
            ),
            None,
            length=self.num_updates_per_step,
        )

        # *update training state
        training_state = TrainingState(
            actor_params=updated_actor_params,
            critic_params=updated_critic_params,
            cost_critic_params=updated_cost_critic_params,
            actor_optimizer_state=updated_actor_optimizer_state,
            critic_optimizer_state=updated_critic_optimizer_state,
            preprocessor_params=training_state.preprocessor_params,  # !identity preprocessor
            lagrangian_multiplier=training_state.lagrangian_multiplier,
            env_step=training_state.env_step + self.env_step_per_training_step,
        )
        metrics = {
            "actor_loss": info["actor_loss"],
            "lagrangian_multiplier": training_state.lagrangian_multiplier
            # "critic_loss": metrics["critic_loss"],
        }
        return training_state, info["final_state"], metrics

    def _train_epoch(self, training_state: TrainingState, epoch_key: types.PRNGKey):
        """Run one epoch of training."""
        reset_key, step_key = jp.random_split(epoch_key)
        # reset environment
        current_state = self.env.reset(reset_key)
        # construct a function to update training state
        def train_step(carry, _):
            """Update training state."""
            training_state, current_state, current_key = carry
            update_key, next_key = jp.random_split(current_key)
            next_training_state, next_state, metrics = self._update(
                training_state, current_state, update_key
            )
            return (next_training_state, next_state, next_key), metrics

        # repeat the update for num_training_steps_per_epoch
        (training_state, final_state, _), metrics = jax.lax.scan(
            train_step,
            (training_state, current_state, step_key),
            None,
            length=self.num_training_steps_per_epoch,
        )
        # lagrangian_multiplier update
        constraint_violation = jp.minimum(
            jp.mean(final_state.info["eval_metrics"].episode_metrics["cost"])
            - self.threshold,
            self.threshold,
        )
        updated_multiplier = (
            training_state.lagrangian_multiplier
            + self.multiplier_lr * constraint_violation
        )
        training_state = TrainingState(
            actor_params=training_state.actor_params,
            critic_params=training_state.critic_params,
            cost_critic_params=training_state.cost_critic_params,
            actor_optimizer_state=training_state.actor_optimizer_state,
            critic_optimizer_state=training_state.critic_optimizer_state,
            preprocessor_params=training_state.preprocessor_params,
            lagrangian_multiplier=updated_multiplier,
            env_step=training_state.env_step,
        )

        # *compute metrics
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, current_state, metrics

    def train(self, train_key: types.PRNGKey = None, progress_fn: Callable = None):
        """Train SHAC."""
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
            training_state, _, training_metrics = jit_epoch(training_state, epoch_key)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
            # ?Is this needed without pmap?

            # calculate the time taken for the epoch
            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                self.num_training_steps_per_epoch * self.env_step_per_training_step
            ) / epoch_training_time
            train_metrics = {
                **{f"training/{k}": v for k, v in training_metrics.items()},
                "training/sps": sps,
                "training/walltime": training_walltime,
            }

            # *evaluation
            self._load_params(training_state)

            if epoch_idx % self.eval_frequency == 0:
                # perform an evaluation
                eval_metrics = self.evaluator.eval()
                metrics = {
                    "training/total_env_steps": training_state.env_step,
                    "training/lagrangian_multiplier": training_state.lagrangian_multiplier,
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
