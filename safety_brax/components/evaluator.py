"""Evaluator"""

import jax
import time

from safety_brax import jumpy as jp
from safety_brax.components import types
from safety_brax.components.actor import Actor
from safety_brax.envs import wrappers


class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: types.Env,
        eval_actor: Actor,
        num_eval_envs: int,
        episode_length: int,
        eval_key: types.PRNGKey,
    ):
        """Initialize the evaluator.

        Args:
            eval_env: The environment to evaluate on.
            eval_actor: The actor to use for evaluation.
            num_eval_envs: Number of environments to evaluate on.
            episode_length: Maximum length of an episode.
            eval_key: The PRNG key used for evaluation.
        """
        if not isinstance(eval_env, wrappers.EvalWrapper):
            eval_env = wrappers.EvalWrapper(eval_env)
        self._eval_env = eval_env
        self._eval_actor = eval_actor
        self._num_eval_envs = num_eval_envs
        self._episode_length = episode_length
        self._eval_key = eval_key
        self._eval_walltime = 0.0
        self._steps_per_episode = episode_length * self._num_eval_envs

        # JIT the key function.
        self._run_episode = jax.jit(self._run_episode_fn)

    def _run_episode_fn(
        self,
        actor_params: types.Params,
        preprocessor_params: types.PreprocessorParams,
        ep_key: types.PRNGKey,
    ):
        """Run an episode. This is a function to be JITed."""
        reset_key, act_key = jp.random_split(ep_key)

        first_state = self._eval_env.reset(reset_key)

        def _step(carry, unused_t):
            state, key = carry
            key, next_key = jp.random_split(key)
            action, _ = self._eval_actor.act_(
                actor_params, preprocessor_params, state.obs, key
            )
            next_state = self._eval_env.step(state, action)
            return (next_state, next_key), None

        (final_state, _), _ = jax.lax.scan(
            _step, (first_state, act_key), None, self._episode_length
        )
        return final_state

    def eval(self, aggregate_episodes: bool = True):
        """Run an evaluation."""
        ep_key, self._eval_key = jp.random_split(self._eval_key)

        t = time.time()
        eval_state = self._run_episode(
            self._eval_actor.parameters, self._eval_actor.preprocessor_params, ep_key
        )
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {
            f"eval/episode_{name}": jp.mean(value) if aggregate_episodes else value
            for name, value in eval_metrics.episode_metrics.items()
        }
        metrics["eval/avg_episode_length"] = jp.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_episode / epoch_eval_time
        self._eval_walltime += epoch_eval_time
        metrics["eval/walltime"] = self._eval_walltime
        return metrics
