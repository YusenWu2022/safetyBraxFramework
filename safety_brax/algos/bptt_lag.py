"""Back propagation through time (BPTT) algorithm."""

import time
import functools
from typing import Tuple, Callable
from safety_brax.envs import wrappers
import jax
import jax.numpy as jnp
import optax
import flax

from safety_brax import jumpy as jp
from safety_brax.components import types, gradients, Evaluator, preprocessor
from safety_brax.algos import BaseAlgorithm
from safety_brax.engine.io import html


@flax.struct.dataclass
class TrainingState:
    """Container for training state."""

    params: types.Params
    preprocessor_params: types.PreprocessorParams
    lagrangian_multiplier: float
    optimizer_state: types.OptState
    env_step: jnp.ndarray


class BPTT_Lag(BaseAlgorithm):
    """Back propagation through time (BPTT) algorithm."""

    def __init__(self, env: types.Env, config: types.Config, algo_key: types.PRNGKey):
        self.env = (
            env if isinstance(env, wrappers.EvalWrapper) else wrappers.EvalWrapper(env)
        )
        self.config = config
        self.prng_key, actor_init_key, eval_key = jax.random.split(algo_key, 3)

        # lagrangian
        self.lagrangian_multiplier = config["lagrangian_multiplier"]
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
        # evaluation frequency
        self.eval_frequency = config["eval_frequency"]

        # learning rate
        self.learning_rate = config["learning_rate"]
        # the truncation length of computing the gradient
        self.truncation_length = config["truncation_length"]
        # the maximum norm of the gradient for clipping
        self.max_grad_norm = config["max_grad_norm"]

        # *check parameters
        assert (
            self.num_envs == self.env.num_local_envs
        ), "Number of environments must match."  # ! Currently, only support parallel envs on the same machine.

        # *initialize components
        # observation preprocessor
        self.observation_preprocessor = preprocessor.identity
        self.preprocessor_params = None

        # create actor
        self.actor = self._create_actor(config["actor_config"], actor_init_key)

        # optimizer
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        # self.optimizer_state = self.optimizer.init(self.actor.parameters)
        loss_fn = functools.partial(self._loss_fn)
        self.grad_fn = jax.jacfwd(loss_fn, has_aux=True)

        # evaluator
        self.evaluator = Evaluator(
            env, self.actor, self.num_envs, self.max_episode_length, eval_key
        )

    def _initialize_training_state(self):
        """Return training state."""
        return TrainingState(
            params=self.actor.parameters,
            lagrangian_multiplier=self.lagrangian_multiplier,
            preprocessor_params=self.preprocessor_params,
            optimizer_state=self.optimizer.init(self.actor.parameters),
            env_step=0,
        )

    def _load_params(
        self,
        params: types.Params,
        preprocessor_params: types.PreprocessorParams,
    ):
        """Load parameters."""
        self.preprocessor_params = preprocessor_params
        self.actor.load_state(params, preprocessor_params)

    def _loss_fn(
        self,
        params: types.Params,
        preprocessor_params: types.PreprocessorParams,
        loss_key: types.PRNGKey,
    ):
        """Compute loss.

        Args:
            params: Actor parameters.
            preprocessor_params: Preprocessor parameters.
            loss_key: PRNG key used for loss computation.
        """
        # split loss key
        reset_key, scan_key = jax.random.split(loss_key)

        # *interact with environment
        # construct env step function
        def env_step_fn(carry: Tuple[types.State, types.PRNGKey], step_idx: int):
            """Environment step function."""
            # TODO: check if necessary to consider truncation.
            current_state, current_key = carry
            action_key, next_key = jax.random.split(current_key)
            action, _ = self.actor.act_(
                params, preprocessor_params, current_state.obs, action_key
            )
            next_state = self.env.step(current_state, action)
            if self.truncation_length is not None:
                next_state = jax.lax.cond(
                    jnp.mod(step_idx + 1, self.truncation_length) == 0.0,
                    jax.lax.stop_gradient,
                    lambda x: x,
                    next_state,
                )

            return (next_state, next_key), (
                next_state.reward,
                next_state.cost,
                current_state.obs,
            )

        # rollout num_envs episodes
        state = self.env.reset(reset_key)
        _, (rewards, costs, observations) = jax.lax.scan(
            env_step_fn, (state, scan_key), jnp.arange(self.max_episode_length)
        )
        return [-jnp.mean(rewards), -jnp.mean(costs)], observations

    def _training_epoch(self, training_state: TrainingState, epoch_key):
        """Perform a training epoch.

        Args:
            training_state: The current training state.
            epoch_key: The PRNG key used for training a single epoch.
        """
        # compute gradient
        [grad, cost_grad], observations = self.grad_fn(
            training_state.params, training_state.preprocessor_params, epoch_key
        )
        # grad = gradients.clip_grads(grad, self.max_grad_norm)
        # cost_grad = gradients.clip_grads(cost_grad, self.max_grad_norm)

        # grad = jax.lax.pmean(grad, axis_name='i')
        # # ! currently, only support parallel envs on the same machine.
        # assert jnp.isnan(grad).sum() == 0, "Gradient contains NaN."

        final_grad = jax.tree_map(
            lambda x, y: x - training_state.lagrangian_multiplier * y, grad, cost_grad
        )
        final_grad = gradients.clip_grads(final_grad, self.max_grad_norm)

        # update parameters
        params_update, optimizer_state = self.optimizer.update(
            final_grad, training_state.optimizer_state
        )
        updated_params = optax.apply_updates(training_state.params, params_update)
        # preprocessor_params = preprocessor.update(
        #     training_state.preprocessor_params, observations
        # )
        # ! identity preprocessor does not need to be updated.
        preprocessor_params = None

        # choose to update multiplier every eval_frequency

        # metrics
        metrics = {
            "grad_norm": optax.global_norm(grad),
            "cost_grad_norm": optax.global_norm(cost_grad),
            "params_norm": optax.global_norm(updated_params),
        }

        # return updated training state
        return (
            TrainingState(
                params=updated_params,
                preprocessor_params=preprocessor_params,
                lagrangian_multiplier=training_state.lagrangian_multiplier,
                optimizer_state=optimizer_state,
                env_step=training_state.env_step
                + self.num_envs * self.max_episode_length,
            ),
            metrics,
        )

    def train(self, train_key: types.PRNGKey = None, progress_fn: Callable = None):
        """Train BPTT agent."""
        # construct needed parameters
        training_state = self._initialize_training_state()
        training_walltime = 0.0
        if train_key is not None:
            self.prng_key = train_key

        # JIT the training epoch function
        jit_epoch = jax.jit(self._training_epoch)

        for epoch_idx in range(self.num_epochs):
            t = time.time()

            # *training
            # perform an epoch
            epoch_key, self.prng_key = jp.random_split(self.prng_key)
            training_state, metrics = jit_epoch(training_state, epoch_key)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            # ?Is this needed without pmap?

            # calculate the time taken for the epoch
            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (self.num_envs * self.max_episode_length) / epoch_training_time
            train_metrics = {
                **{f"training/{k}": v for k, v in metrics.items()},
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
                    "training/lagrangian_multiplier": training_state.lagrangian_multiplier,
                    **train_metrics,
                    **eval_metrics,
                }
                #  lagrangian_multiplier update  every evaluation
                constraint_violation = jp.minimum(
                    metrics["eval/episode_cost"] / self.max_episode_length
                    - self.threshold,
                    self.threshold,
                )
                # updated_multiplier = jp.maximum(
                #     1.0,
                #     training_state.lagrangian_multiplier
                #     - self.multiplier_lr * constraint_violation,
                # )
                updated_multiplier = (
                    training_state.lagrangian_multiplier
                    + self.multiplier_lr * constraint_violation
                )
                training_state = TrainingState(
                    params=training_state.params,
                    preprocessor_params=training_state.preprocessor_params,
                    lagrangian_multiplier=updated_multiplier,
                    optimizer_state=training_state.optimizer_state,
                    env_step=training_state.env_step,
                )
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
