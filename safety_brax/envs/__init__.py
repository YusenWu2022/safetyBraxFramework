from typing import Callable, Optional, Type, Union, overload

from safety_brax.envs import wrappers
from safety_brax.envs.env import Env, State, Wrapper
from safety_brax.envs.builder import Builder


def create(
    task_name: str = "velocity",
    agent_name: str = "ant",
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    num_local_envs: Optional[int] = None,
    eval_metrics: bool = False,
    **kwargs
) -> Env:
    """Creates an Env with a specified brax system."""
    env = Builder(
        task_name=task_name,
        agent_name=agent_name,
        **kwargs
    )
    if episode_length is not None:
        env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if num_local_envs:
        env = wrappers.VectorWrapper(env, num_local_envs)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if eval_metrics:
        env = wrappers.EvalWrapper(env)

    return env
