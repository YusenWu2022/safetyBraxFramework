"""differentiable constraint policy optimization(diffCPO) algorithm."""

from calendar import isleap
from cmath import inf
import time
from jax import random
import functools
from typing import Tuple, Callable, Sequence, final
from flax.core.frozen_dict import unfreeze, freeze
import jax
import jax.numpy as jnp
import optax
import flax
from safety_brax.components.running_statistics import update
from safety_brax.engine.jumpy import random_prngkey
from safety_brax.envs import wrappers
from safety_brax import jumpy as jp
from safety_brax.components import (
    actor,
    critic,
    types,
    gradients,
    Evaluator,
    preprocessor,
    hcb,
)
from collections import deque
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
    cost_critic_optimizer_state: types.OptState
    preprocessor_params: types.PreprocessorParams
    env_step: jnp.ndarray
    ep_costs: jnp.ndarray
    grad_update_set: types.Params


class DiffCPO(BaseAlgorithm):
    """differentiable constrainted policy optimization algorithm."""

    def __init__(self, env: types.Env, config: dict, algo_key: types.PRNGKey):
        self.env = (
            env if isinstance(env, wrappers.EvalWrapper) else wrappers.EvalWrapper(env)
        )
        self.config = config
        (
            self.prng_key,
            actor_init_key,
            critic_init_key,
            cost_critic_init_key,
            eval_key,
        ) = jax.random.split(algo_key, 5)

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
        # GAE lambda
        self.gae_lambda = config["gae_lambda"]
        # discount gamma
        self.discount_gamma = config["discount_gamma"]

        # delta in cpo
        self.delta = config["delta"]

        # threshold
        self.threshold = config["threshold"]

        self.eval_metrics = {"eval/episode_cost": 0.0}

        # ep_cost queue storage for estimation of constraint_violation
        self.ep_costs = deque(maxlen=16)
        self.ep_costs.extend(list([200.0 for i in range(0, 16)]))

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
        self.preprocessor_params = preprocessor.init_state(
            types.Array((self.env.observation_size,), jnp.float32)
        )
        # create actor
        self.actor = self._create_actor(config["actor_config"], actor_init_key)
        # create critic
        self.critic = self._create_critic(config["critic_config"], critic_init_key)
        self.cost_critic = self._create_critic(
            config["critic_config"], cost_critic_init_key
        )  # cost_critic to init cost_critic_params!!! big error..same critic value..
        # optimizer
        self.actor_optimizer = optax.adam(self.learning_rate)
        self.critic_optimizer = optax.adam(self.learning_rate)
        self.cost_critic_optimizer = optax.adam(self.learning_rate)
        # set gradient function
        actor_loss_fn = functools.partial(self._actor_loss_fn)
        critic_loss_fn = functools.partial(self._critic_loss_fn)
        cost_critic_loss_fn = functools.partial(self._cost_critic_loss_fn)
        # !not set pmap_axis_name for multi-GPU training
        self.actor_grad_fn = jax.jacfwd(actor_loss_fn, has_aux=True)
        self.critic_update_fn = gradients.gradient_update_fn(
            critic_loss_fn, self.critic_optimizer, pmap_axis_name=None, has_aux=True
        )
        self.cost_critic_update_fn = gradients.gradient_update_fn(
            cost_critic_loss_fn,
            self.cost_critic_optimizer,
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
            cost_critic_params=self.cost_critic.parameters,
            actor_optimizer_state=self.actor_optimizer.init(self.actor.parameters),
            critic_optimizer_state=self.critic_optimizer.init(self.critic.parameters),
            cost_critic_optimizer_state=self.cost_critic_optimizer.init(
                self.critic.parameters
            ),
            preprocessor_params=self.preprocessor_params,
            env_step=0,
            ep_costs=jp.array(self.ep_costs),
            grad_update_set=jax.tree_map(
                lambda x: x * 0,
                self.actor.parameters,
                is_leaf=lambda x: isinstance(x, jax.ShapedArray),
            )
            # no zeros-like: but can just *0 instead
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
        # current_state = jax.lax.stop_gradient(
        #     current_state
        # )  # TODO: check if it is correct.

        # *interact with the environment
        # construct env step function
        @jax.jit
        def env_step_fn(carry, _):
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
        value_cost = self.cost_critic.value_(
            cost_critic_params, preprocessor_params, final_state.obs
        )
        actor_loss = jnp.mean(jnp.sum(rewards, axis=0) + value)
        # jax.debug.print('reward: {} value: {}', rewards, value)
        actor_cost_loss = jnp.mean(jnp.sum(costs, axis=0) + value_cost)
        # jax.debug.print('cost: {} value:{}', costs, value_cost)

        return [actor_loss, actor_cost_loss], {
            "transitions": transitions,
            "final_state": jax.lax.stop_gradient(final_state),
            "actor_loss": actor_loss,
            "actor_cost_loss": actor_cost_loss,
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
        # jax.debug.print("critic_loss:{}", critic_loss)
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
        value_baseline = self.cost_critic.value_(
            cost_critic_params, preprocessor_params, data.observation
        )
        bootstrap_value = self.cost_critic.value_(
            cost_critic_params, preprocessor_params, data.next_observation[-1]
        )
        value_target, _ = compute_gae(
            truncation=data.extras["truncation"],
            termination=data.done * (1 - data.extras["truncation"]),
            rewards=data.cost,  # * self.reward_scaling,  # cost_scaling?
            values=value_baseline,
            bootstrap_value=bootstrap_value,
            lambda_=self.gae_lambda,
            discount=self.discount_gamma,
        )
        # jax.debug.print(
        #     "value_target:{} value_baseline:{}", value_target, value_baseline
        # )
        # *Compute critic loss
        value_error = value_target - value_baseline
        cost_critic_loss = jnp.mean(value_error * value_error) * 0.5 * 0.5
        # jax.debug.print("cost_critic_loss:{}", cost_critic_loss)
        return cost_critic_loss, {
            "critic_loss": cost_critic_loss,
        }

    def rollout(
        self,
        training_state: TrainingState,
        current_state: types.State,
        constraint_violation: float,
        update_key: types.PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ):
        """Rolls out the environment for `rollout_length` steps."""

        [actor_grad, actor_cost_grad], info = self.actor_grad_fn(
            training_state.actor_params,
            training_state.critic_params,
            training_state.cost_critic_params,
            training_state.preprocessor_params,
            current_state,
            update_key,
        )
        final_state = info["final_state"]
        # jax.debug.print("actor_grad:{}",actor_grad)

        def policy_update(
            _actor_grad, _actor_cost_grad, constraint_violation, update_key
        ):
            """dual problem to compute policy update scale and direction"""
            c_k = constraint_violation
            # jax.debug.print("c_k:{}", c_k)

            flatten_cost_grad = jax.tree_util.tree_leaves(
                _actor_cost_grad, is_leaf=lambda x: isinstance(x, jax.ShapedArray)
            )  # deepest: shadedArray...
            flatten_cost_grad = jnp.concatenate(
                [x.reshape(-1, 1) for x in flatten_cost_grad]
            )

            flatten_grad = jax.tree_util.tree_leaves(
                _actor_grad, is_leaf=lambda x: isinstance(x, jax.ShapedArray)
            )  # deepest: shadedArray...
            flatten_grad = jnp.concatenate([x.reshape(-1, 1) for x in flatten_grad])

            r_k = jp.sum(flatten_grad * flatten_grad)
            s_k = jp.sum(flatten_grad * flatten_cost_grad)
            t_k = jp.sum(flatten_cost_grad * flatten_cost_grad)

            # jax.debug.print("t_k: {} r_k: {} s_k: {}", t_k, r_k, s_k)

            # TODO: detailed explanation of the following code
            condition = c_k**2 / (t_k + 1e-8) - self.delta
            optim_case = jnp.where(
                jnp.logical_and(condition >= 0, c_k > 0),
                0,
                jnp.where(
                    jnp.logical_and(condition >= 0, c_k <= 0),
                    1,
                    jnp.where(jnp.logical_and(condition < 0, c_k != 0), 2, 3),
                ),
            )
            jax.debug.print("optim_case: {}", optim_case)

            def all_infeasible_fn():
                """If c^2/t - δ ≥ 0 and c > 0, then the problem is infeasible."""

                times = jp.sqrt(self.delta) / (jp.norm(flatten_cost_grad) + 1e-8)
                # times = jp.clip(times, -0.1, 0.1)  # limit update times in [-0.1,0.1] range to prevent extreme update
                # maybe time  may be fixed to better range: if normal(flaggen_cost_grad) is low then update a little why cost not going down..?
                # jax.debug.print("times:{}",times)
                return jax.tree_util.tree_map(  # can use lambda here directly(without activation)
                    # lambda x: (x) * jp.sqrt(self.delta) / (jp.norm(flatten_cost_grad) + 1e-8) ,
                    lambda x: -x * times,
                    _actor_cost_grad,
                    is_leaf=lambda x: isinstance(x, jax.ShapedArray),
                )

            def all_feasible_fn():
                """If c^2/t - δ ≥ 0 and c <= 0, then the trust region lies entirely within the constraint-satisfying half space."""
                times = jp.sqrt(self.delta) / (jp.norm(flatten_grad) + 1e-8)
                return jax.tree_util.tree_map(
                    lambda x: x * times,
                    _actor_grad,
                    is_leaf=lambda x: isinstance(x, jax.ShapedArray),
                )

            def infeasible_feasible_fn():
                """only part of trust region is feasible"""

                def proj(x, a, b):
                    return jnp.minimum(jnp.maximum(x, a), b)

                A = r_k - s_k**2 / (t_k + 1e-8)
                B = self.delta - c_k**2 / (t_k + 1e-8)

                lambda_a = jp.sqrt(A / B)
                lambda_b = jp.sqrt(r_k / self.delta)

                c_k_eps = c_k + 1e-8
                lambda_a_star = jnp.where(
                    c_k < 0,
                    proj(lambda_a, 0, r_k / c_k_eps),
                    proj(lambda_a, r_k / c_k_eps, 1e6),
                )
                lambda_b_star = jnp.where(
                    c_k < 0,
                    proj(lambda_b, r_k / c_k_eps, 1e6),
                    proj(lambda_b, 0, r_k / c_k_eps),
                )

                p_a_star = -0.5 * (A / (lambda_a_star + 1e-8)) - 0.5 * lambda_a_star * B
                p_b_star = -0.5 * (
                    r_k / (lambda_b_star + 1e-8) + lambda_b_star * self.delta
                )

                lambda_star = jnp.where(
                    p_a_star > p_b_star, lambda_a_star, lambda_b_star
                )

                nu_star = (lambda_star * c_k + s_k) / (t_k + 1e-8)
                nu_star = jnp.maximum(nu_star, 0)
                # jax.debug.print("nu_star:{}, lambda_star:{}", nu_star, lambda_star)
                return jax.tree_util.tree_map(
                    lambda x, y: (x - nu_star * y) / (lambda_star + 1e-8),
                    _actor_grad,
                    _actor_cost_grad,
                    is_leaf=lambda x: isinstance(x, jax.ShapedArray),
                )

            def zero_case_infeasible_feasible_fn():
                """c_k = 0"""

                lambda_star = jnp.maximum(jp.sqrt(r_k / self.delta), 0)

                nu_star = (lambda_star * c_k + s_k) / (t_k + 1e-8)
                nu_star = jnp.maximum(nu_star, 0)
                # jax.debug.print("nu_star:{}, lambda_star:{}", nu_star, lambda_star)

                return jax.tree_util.tree_map(
                    lambda x, y: (x - nu_star * y) / (lambda_star + 1e-8),
                    _actor_grad,
                    _actor_cost_grad,
                    is_leaf=lambda x: isinstance(x, jax.ShapedArray),
                )

            return jax.lax.switch(
                optim_case,
                [
                    all_infeasible_fn,
                    all_feasible_fn,
                    infeasible_feasible_fn,
                    zero_case_infeasible_feasible_fn,
                ],
            )

        actor_param_updates = policy_update(
            _actor_grad=actor_grad,
            _actor_cost_grad=actor_cost_grad,
            constraint_violation=constraint_violation,
            update_key=update_key,
        )

        # method 1:clip 
        actor_param_updates = gradients.clip_grads(
            actor_param_updates, self.max_grad_norm
        )
        grad_update_set = jax.tree_map(
            lambda x, y: x + y, training_state.grad_update_set, actor_param_updates
        )
        # method 2: ignore
        # flatten_update_grad = jax.tree_util.tree_leaves(
        #     actor_param_updates, is_leaf=lambda x: isinstance(x, jax.ShapedArray)
        # )
        # flatten_update_grad = jnp.concatenate(
        #     [x.reshape(-1, 1) for x in flatten_update_grad]
        # )
        # # ignore explosion update_grad: maybe need to record count

        # def ignore_fn():
        #     return training_state.grad_update_set

        # def count_fn():
        #     return jax.tree_map(
        #         lambda x, y: x + y, training_state.grad_update_set, actor_param_updates
        #     )

        # grad_update_set = jp.cond(
        #     jp.norm(flatten_update_grad) > 1.0, ignore_fn, count_fn
        # )

        # maybe use clip instead of ignore

        new_training_state = TrainingState(
            actor_params=training_state.actor_params,
            critic_params=training_state.critic_params,
            cost_critic_params=training_state.cost_critic_params,
            actor_optimizer_state=training_state.actor_optimizer_state,  # TODO how to compute updated_actor_optimizer_state under our algo?
            critic_optimizer_state=training_state.critic_optimizer_state,
            cost_critic_optimizer_state=training_state.cost_critic_optimizer_state,
            preprocessor_params=training_state.preprocessor_params,  # identity preprocessor: update preprocessor together with critic update
            env_step=training_state.env_step,
            ep_costs=training_state.ep_costs,
            grad_update_set=grad_update_set,
        )

        # not change training_state
        return (
            new_training_state,
            info["final_state"],
            info["transitions"],
        )  # need to return to change training_state params

    def _update(
        self,
        training_state: TrainingState,
        batch: types.Transition,
        update_key: types.PRNGKey,
        ep_costs: jnp.ndarray,
        updated_preprocessor_params: types.Params,
        repeat_update_times: int,
        updated_actor_params: types.Params,
    ):
        # construct a function to update critic parameters using SGD
        def sgd_fn(carry, _):
            """Update training state using SGD."""
            critic_params, critic_optimizer_state, current_key = carry
            sgd_key, perm_key, next_key = jp.random_split(current_key, 3)

            # # add cost value to ep_costs: deque to append/extend, the first epoch c_k is always 0
            # self.ep_costs.extend(list(batch.cost))

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
            }
            (
                updated_critic_params,
                updated_critic_optimizer_state,
                metrics,
            ) = gradients.sgd(
                self.cost_critic_update_fn,
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

        # repeat the update for repeat_update_times times
        (
            updated_cost_critic_params,
            updated_cost_critic_optimizer_state,
            _,
        ), metrics = jax.lax.scan(
            cost_sgd_fn,
            (
                training_state.cost_critic_params,
                training_state.cost_critic_optimizer_state,
                update_key,
            ),
            None,
            length=self.num_updates_per_step,
        )
        # first update cost_sgd and then sgd: return episode cost more precisely
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
        # *update training state
        training_state = TrainingState(
            actor_params=updated_actor_params,
            critic_params=updated_critic_params,
            cost_critic_params=updated_cost_critic_params,
            actor_optimizer_state=training_state.actor_optimizer_state,  # TODO how to compute updated_actor_optimizer_state under our algo?
            critic_optimizer_state=updated_critic_optimizer_state,
            cost_critic_optimizer_state=updated_cost_critic_optimizer_state,
            preprocessor_params=updated_preprocessor_params,  # identity preprocessor: update preprocessor together with critic update
            env_step=training_state.env_step + self.env_step_per_training_step * self.num_training_steps_per_epoch,
            ep_costs=ep_costs,
            grad_update_set=training_state.grad_update_set,
        )
        return training_state, metrics

    def _training_step(
        self,
        training_state: TrainingState,
        current_state: types.State,
        step_key: types.PRNGKey,
    ):
        """Perform a training step."""

        rollout_key, update_key = jp.random_split(step_key)
        ep_costs = training_state.ep_costs
        constraint_violation = jnp.mean(ep_costs) - self.threshold
        jax.debug.print("violation:{}", constraint_violation)
        # *roll out a batch of data

        # construct a function that performs a rollout
        training_state = TrainingState(
            actor_params=training_state.actor_params,
            critic_params=training_state.critic_params,
            cost_critic_params=training_state.cost_critic_params,
            actor_optimizer_state=training_state.actor_optimizer_state,  # TODO how to compute updated_actor_optimizer_state under our algo?
            critic_optimizer_state=training_state.critic_optimizer_state,
            cost_critic_optimizer_state=training_state.cost_critic_optimizer_state,
            preprocessor_params=training_state.preprocessor_params,  # identity preprocessor: update preprocessor together with critic update
            env_step=training_state.env_step,
            ep_costs=training_state.ep_costs,
            grad_update_set=jax.tree_map(
                lambda x: x * 0,
                self.actor.parameters,
                is_leaf=lambda x: isinstance(x, jax.ShapedArray),
            ),
        )

        def rollout_step(carry, _):
            """One rollout step."""
            current_state, current_key, training_state = carry
            current_key, next_key = jp.random_split(current_key)
            new_training_state, next_state, data = self.rollout(
                training_state,
                current_state,
                constraint_violation,
                current_key,
            )
            return (next_state, next_key, new_training_state), data

        # rollout num_rollouts_per_step times
        (final_state, _, training_state), data = jax.lax.scan(
            rollout_step,
            (current_state, rollout_key, training_state),
            None,
            length=self.num_training_steps_per_epoch,
        )

        # average update-gradient
        actor_param_updates = jax.tree_map(
            lambda x: x / self.num_training_steps_per_epoch,
            training_state.grad_update_set,
            is_leaf=lambda x: isinstance(x, jax.ShapedArray),
        )
        # jax.debug.print("actor_params:{}",actor_param_updates) # print to see if actor_param_optimization is not fitted 
        updated_actor_params = optax.apply_updates(
            training_state.actor_params, actor_param_updates
        )

        # extend function returns nonetype, can not be itered
        ep_costs = deque(list(ep_costs), maxlen=16)
        ep_costs.extend(
            list(final_state.info["eval_metrics"].episode_metrics["cost"])
        )  # extend 16 values in(envs num)
        jax.debug.print("ep cost mean: {} ", jnp.mean(jp.array(list(ep_costs))))
        ep_costs = jp.array(list(ep_costs))

        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )

        # *update preprocessor parameters with unconverted data
        updated_preprocessor_params = preprocessor.update(  # not problem with data, but with preprocessor_params itself
            training_state.preprocessor_params,
            data.observation,
        )

        # *Update parameters: only update ep_costs every _update()
        new_training_state, metrics = self._update(
            training_state=training_state,
            batch=data,
            update_key=update_key,
            ep_costs=ep_costs,
            repeat_update_times=self.num_updates_per_step,
            updated_preprocessor_params=updated_preprocessor_params,
            updated_actor_params=updated_actor_params,
        )

        return new_training_state, final_state, metrics

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
            (
                next_training_state,
                next_state,
                metrics,
            ) = self._training_step(training_state, current_state, update_key)
            return (next_training_state, next_state, next_key), metrics

        # repeat the update for num_training_steps_per_epoch
        (training_state, current_state, _), metrics = jax.lax.scan(
            train_step,
            (training_state, current_state, step_key),
            None,
            length=1,
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
                self.eval_metrics = self.evaluator.eval()
                metrics = {
                    "training/total_env_steps": training_state.env_step,
                    **train_metrics,
                    **self.eval_metrics,
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
            state = jit_step(state, action)
            rollout.append(state.qp)

        html.save_html(filepath, render_env.sys, rollout, True)
