"""Build the environment."""
from copy import deepcopy
from tokenize import Double

from safety_brax import jumpy as jp
from safety_brax.engine import System, QP, Info
from safety_brax.envs.env import Env, State
from safety_brax.envs.assets import Asset, Ant, Ball, Quadrotor, Car, Swimmer
from safety_brax.envs.tasks import Task, Velocity, Circle, Target
import jax.experimental.host_callback as hcb
def host_cb(args, t):
    print(args)

class Builder(Env):
    _agent_registry = {"ant": Ant, "ball": Ball, 'quadrotor': Quadrotor, "car":Car, "swimmer": Swimmer}
    _task_registry = {"velocity": Velocity, "circle": Circle, "target": Target}

    def __init__(
        self,
        task_name: str = "velocity",
        task_config: dict = {},
        agent_name: str = "ant",
        agent_config: dict = {},
        reset_noise_scale: float = 0.1,
    ):
        """Initialize the environment."""
        self._agent_name = agent_name
        self._task_name = task_name
        self._agent = self._agent_registry[agent_name](**agent_config)
        self._task = self._task_registry[task_name](self._agent, **task_config)
        # print(self._task.config)
        self.sys = System(
            config=self._task.config, resource_paths=None  # TODO: add resource paths
        )

        self._reset_noise_scale = reset_noise_scale

    @property
    def action_size(self) -> int:
        if self._agent.action_size:
            return self._agent.action_size
        else:
            return super().action_size

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng0, rng1, rng2 = jp.random_split(rng, 3)
        self._task.reset(rng0)

        assert len(self.sys.config.defaults) == 1, "Only one default config is allowed."
        qpos = self.sys.default_angle() + self._noise(rng1)
        qvel = self._noise(rng2)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

        obs = self._get_obs(qp, self.sys.info(qp))
        reward, cost, done = jp.zeros(3)
        metrics = deepcopy(self._task.metrics)
        return State(qp, obs, reward, cost, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one time step of the environment's dynamics."""
        action = self._agent.action_adaptor(state, action)
        # action = self._task.action_adaptor(state, action)
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)
        reward = self._task.calculate_reward(state, action, qp, info)
        cost = self._task.calculate_cost(state, action, qp, info)
        done = self._task.is_done(qp)
        state.metrics.update(**self._task.metrics)

        return state.replace(qp=qp, obs=obs, reward=reward, cost=cost, done=done)

    def _get_obs(self, qp: QP, info: Info) -> jp.ndarray:
        """Observe ant body position and velocities."""
        return self._task.get_obs(self.sys, qp, info)

    def _noise(self, rng):
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        return jp.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)


if __name__ == "__main__":
    env = Builder()
    rng = jp.random_prngkey(0)
    print(rng)
    print(jp.random_split(rng, 3))
    state = env.reset(jp.random_prngkey(0))
    print(env.observation_size)
    print(env.action_size)
