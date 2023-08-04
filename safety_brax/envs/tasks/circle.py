from safety_brax import jumpy as jp
from safety_brax.engine import QP, Info, System
from safety_brax.envs.env import Env, State
from safety_brax.envs.tasks import Task
from safety_brax.envs.assets import Asset, Ant, Ground, Obstacle, Obstacle_platform
from flax import linen
from torch import abs_
import jax.experimental.host_callback as hcb
import jax


def host_cb(args, t):
    print(args)


class Circle(Task):
    """
    ### Circle Task

    This task is designed to test the ability of the agent to learn to move around a certain circle.
    The agent is rewarded for its angle velocity and penalized for three different
    settings.

    #### Action Space
    The action space is dependent on which agent is used. Please refer to corresponding
    agent's documentation for more details.

    #### Observation Space
    The observation space includes two parts: the observation needed to complete the
    task and the state of the agent.

    The task observation includes the following:
    | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |


    ### Rewards

    The reward consists of three parts:

    - *reward_velocity*: The reward for the angle velocity of the agent around the circle . This is the main
        reward for the task.
    - *reward_ctrl*: A negative reward for penalizing the agent if it takes actions
      that are too large. It is measured as *coefficient **x**
      sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
      control and has a default value of 0.5.
    - *contact_cost*: A negative reward for penalizing the agent if the external
      contact force is too large. It is calculated *0.5 * 0.001 *
      sum(clip(external contact force to [-1,1])<sup>2</sup>)*.
    - *out-of-bound penalty*: A negative cost for going out of circle bound (either too close or far away from center).
      this cost contains three grades of severality:

    """

    def __init__(
        self,
        agent: Asset,
        # reward parameters
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        ctrl_cost_weight: float = 0.5,
        safety_level: int = 0,
    ) -> None:
        """Initialize a circle task.

        Args:
            agent: The agent to be used in the task.
        """
        super().__init__()
        self.r = 20.0
        self.agent = agent
        self.ground = Ground()
        self.obstacle_limit = Obstacle(
            unique_name=True,
            halfsize=[2, 2, 2],
            frozen_pos=[True, True, True],
        )
        self.obstacle1_wall_1 = Obstacle(
            unique_name=True,
            halfsize=[100, 0.1, 2],
            frozen_pos=[True, True, True],
        )
        self.obstacle1_wall_2 = Obstacle(
            unique_name=True,
            halfsize=[100, 0.1, 2],
            frozen_pos=[True, True, True],
        )
        self.obstacle_platform_1 = Obstacle_platform(
            halfsize=[0.2, 0.01, 0], frozen_pos=[True, True, True]
        )

        self.register_asset(self.agent)
        self.register_asset(self.ground)
        self.register_asset(self.obstacle1_wall_2)
        self.register_asset(self.obstacle1_wall_1)
        # self.register_asset(self.obstacle)   # add one board with certain velocity to limit drone moving speed
        self.register_asset(self.obstacle_platform_1)
        self.register_collision(self.agent, self.ground)

        self.init_pos_set = {
            "ant": jp.array([0.0, 0.0, 0.5]),
            "ball": jp.array([0.0, 0.0, 0.5]),
            "quadrotor": jp.array([0.0, 0.0, 10.0]),
            "car": jp.array([0.0, 0.0, 0.5]),
            "swimmer": jp.array([0.0, 0.0, 0.5]),
        }

        _half_height = 1
        # self.set_defaults(
        #     self.agent.core, qp_pos=[0.0, 0.0, 0.0], qp_vel=[0.0, 0.0, 0.0]
        # )
        self.set_defaults(self.obstacle1_wall_1.core, qp_pos=[0.0, 2, _half_height])
        self.set_defaults(self.obstacle1_wall_2.core, qp_pos=[0.0, -2, _half_height])
        self.set_defaults(self.obstacle_platform_1.core, qp_pos=[0.0, 0.0, -0.5])
        # self.set_defaults(self.obstacle_platform_1.config.bodies[0].name,[0,0,0],[0,0,0])

        self._cost_fn_list = []
        self._cost_related_obs = []
        # *safety level 1: wait for latter
        self.set_defaults(self.agent.core, qp_pos=self.init_pos_set[self.agent.type])

        if safety_level >= 1:
            self._cost_fn_list.append(self._level_1_safety)
        if safety_level >= 2:
            self._cost_fn_list.append(self._level_2_safety)

        # Record the reward parameters
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._ctrl_cost_weight = ctrl_cost_weight

        # Info
        self._done, zero = jp.zeros(2)
        self._metrics = {
            # Detailed Reward
            "reward_circle": zero,
            "reward_rotate": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            # Detailed Cost
            "cost_out_of_track": zero,
            "cost_out_of_upper_track": zero,
            # TODO: Add detailed cost
            # State
            "x_position": zero,
            "y_position": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "distance_from_origin": zero,
        }

    # 1、add cost: two lines  constraint
    # level2: 4 walls: velocity
    # 2、ball: mass up  shac
    # 3、abs healthy_reward:0 or 1 not-diff?
    # 4、overleaf

    def _level_1_safety(self, state: State, action: jp.ndarray, qp: QP) -> jp.ndarray:
        """Check if the agent is too far from center."""
        pos = qp.pos[self.agent.rid]
        self.metrics["cost_out_of_track"] = jp.where(
            pos[0] > 16.0, 1.0, 0.0
        ) + jp.where(pos[0] < -16.0, 1.0, 0.0)
        return self.metrics["cost_out_of_track"]

    def _level_2_safety(self, state: State, action: jp.ndarray, qp: QP) -> jp.ndarray:
        """Check if running too fast"""
        pos = qp.pos[self.agent.rid]
        self.metrics["cost_out_of_upper_track"] = jp.where(
            pos[1] > 16.0, 1.0, 0.0
        ) + jp.where(pos[1] < -16.0, 1.0, 0.0)
        return self.metrics["cost_out_of_upper_track"]

    def action_adaptor(self, state, action):
        return action

    def calculate_reward(
        self, state: State, action: jp.ndarray, qp: QP, info: Info
    ) -> jp.ndarray:
        """Calculate the reward for the Circle task.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            qp: The next qp of the environment returned from the engine.
        """
        velocity = (
            qp.pos[self.agent.rid] - state.qp.pos[self.agent.rid]
        ) / self.config.dt

        vel = qp.vel[self.agent.rid]
        pos = qp.pos[self.agent.rid]
        dist_xy = jp.norm(pos[0:2])
        vel_orthogonal = jp.array([-vel[1], vel[0]])
        abs_dist = jp.sqrt(jp.square(dist_xy - self.r))

        circle_reward = (
            200 * jp.dot(pos[0:2], vel_orthogonal) / (1 + abs_dist) / (1e-6 + dist_xy)
        )

        is_healthy = self.agent.is_healthy(qp, info)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        self._done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        # ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        ctrl_cost = 0.0
        contact_cost = 0.0
        reward = (
            circle_reward
            + healthy_reward  # *10?
            - ctrl_cost
            - contact_cost
            - 1000 * self._done  # + rotate_reward
        )
        self._metrics.update(
            reward_circle=circle_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=qp.pos[self.agent.rid, 0],
            y_position=qp.pos[self.agent.rid, 1],
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            distance_from_origin=jp.norm(qp.pos[self.agent.rid, :2]),
        )
        return reward

    def calculate_cost(
        self, state: State, action: jp.ndarray, qp: QP, info: Info
    ) -> float:
        cost = 0.0
        for cost_fn in self._cost_fn_list:
            cost += cost_fn(state, action, qp)
        return cost

    def is_done(self, qp) -> bool:
        """Check if the task is done."""
        return self._done

    def _special_set(self, qp):
        vel = jp.index_update(qp.vel, self.target_idx, jp.array([0.5, 0, 0]))
        qp = qp.replace(vel=vel)

    def reset(self, rng: jp.ndarray) -> None:
        """Reset the task."""
        self._done = False
        self._metrics = {k: jp.zeros_like(v) for k, v in self._metrics.items()}

    def get_obs(self, sys: System, qp: QP, info: Info) -> jp.ndarray:
        """Observe the agent's position and velocity."""
        agent_related_obs = self.agent.asset_related_observation(sys, qp, info)

        cost_related_obs = [
            cost_fn(sys, qp, info) for cost_fn in self._cost_related_obs
        ]
        task_related_obs = [jp.array(self.r)]
        # hcb.id_tap(host_cb, jp.concatenate(agent_related_obs + cost_related_obs))
        return jp.concatenate(agent_related_obs + cost_related_obs)


if __name__ == "__main__":
    ant = Ant()
    circle = Circle(ant)
    print(circle._config.collide_include)
    # print(velocity._config.bodies[velocity.agent.rid])
