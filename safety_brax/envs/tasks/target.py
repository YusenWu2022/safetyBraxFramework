from safety_brax import jumpy as jp
from safety_brax.engine import QP, Info, System
from safety_brax.envs.env import Env, State
from safety_brax.envs.tasks import Task
from safety_brax.envs.assets import (
    Asset,
    Ant,
    Ground,
    Obstacle,
    Obstacle_platform,
    Gate,
)
from flax import linen


class Target(Task):
    """
    ### Target Task

    This task is designed to test the ability of the agent to learn to move towards one certain target goal.
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

    - *distance_reward*: The reward for the narrow distance between agent and target . This is the main
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
        """Initialize a target task.

        Args:
            agent: The agent to be used in the task.
        """
        super().__init__()
        self.agent = agent
        self.ground = Ground()
        self.obstacle_limit = Obstacle(unique_name=True, frozen_pos=[True, True, True])
        self.gate = Gate(unique_name=True, frozen_pos=[True, True, True])

        self.register_asset(self.agent)
        self.register_asset(self.gate)
        self.register_asset(self.ground)
        # self.register_asset(self.obstacle)   # add one board with certain velocity to limit drone moving speed
        self.register_collision(self.agent, self.ground)

        self.init_pos_set = {
            "ant": jp.array([0.0, 0.0, 0.5]),
            "ball": jp.array([0.0, 0.0, 0.5]),
            "quadrotor": jp.array([0.0, 0.0, 10.0]),
            "car": jp.array([0.0,0.0,0.5]),
            "swimmer": jp.array([0.0, 0.0, 0.5]),
        }

        self.goal_pos = jp.array([20, 20, 0])
        self.set_defaults(self.gate.core, qp_pos=self.goal_pos)
        self.set_defaults(self.agent.core, qp_pos=self.init_pos_set[self.agent.type])
        self._cost_fn_list = []
        self._cost_related_obs = []

        # Record the reward parameters
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._ctrl_cost_weight = ctrl_cost_weight

        # Info
        self._done, zero = jp.zeros(2)
        self._metrics = {
            # Detailed Reward
            "reward_distance": zero,
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            # Detailed Cost
            # TODO: Add detailed cost
            # State
            "x_position": zero,
            "y_position": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "distance_from_origin": zero,
        }
        if safety_level >= 1:
            self._cost_fn_list.append(self._level_1_safety)
        # if safety_level >= 2:
        #     self._cost_fn_list.append(self._level_2_safety)

    def _level_1_safety(self, state: State, action: jp.ndarray, qp: QP) -> jp.ndarray:
        """Check if the agent is on the track."""
        # v1: for ball 
        # pos = qp.pos[self.agent.rid]
        # dist_out_of_track = jp.maximum(
        #     pos[1] - self.track_width / 2, -self.track_width / 2 - pos[1]
        # )
        # self.metrics["cost_out_of_track"] = linen.relu(dist_out_of_track)
        # v2: for double_ball
        dist = jp.norm(qp.pos[self.agent.rid, :2] - self.goal_pos[:2])
        self.metrics["cost_out_of_track"] = jp.where(dist < 12.0, 1.0, 0.0) + jp.where(dist > 35.0, 1.0, 0.0)   # limit distacne in [12,35], original about 28
        return self.metrics["cost_out_of_track"]

    def _level_1_observation(self, sys: System, qp: QP, info: Info) -> jp.ndarray:
        """The position of the track boundary will be additionally observed"""
        target_pos = qp.pos[self.target.rid]
        return jp.concatenate(target_pos)

    def calculate_reward(
        self, state: State, action: jp.ndarray, qp: QP, info: Info
    ) -> jp.ndarray:
        """Calculate the reward for the Target task.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            qp: The next qp of the environment returned from the engine.
        """
        velocity = (
            qp.pos[self.agent.rid] - state.qp.pos[self.agent.rid]
        ) / self.config.dt

        dist_reward = (
            100.0 / (1e-6 + jp.norm(qp.pos[self.agent.rid, :2] - self.goal_pos[:2]))
            - 4.0   
        )
        is_healthy = self.agent.is_healthy(qp, info)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        self._done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action)) * 0.0
        contact_cost = 0.0
        reward = dist_reward + healthy_reward - ctrl_cost - contact_cost
        self._metrics.update(
            reward_distance=dist_reward,
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

        y = qp.pos[self.agent.rid][1]
        z = qp.pos[self.agent.rid][2]
        r = jp.norm(qp.pos[self.agent.rid][:1])
        dist = jp.norm(qp.pos[self.agent.rid] - self.goal_pos)
        cs = dict(
            # Step1: only collision cost
            # cost_collisions=1. if jp.sum(jp.square(jp.clip(info.contact.vel,-1,1))) else 0.,
            # Step2: Agent should not leave valid operation space in 2D circle.
            # V1 for ball agent
            # cost_out_of_range=jp.where(dist > 40.0, 1.0, 0.0),
            # cost_too_close=jp.where(dist < 6.0, 1.0, 0.0)
            # V2 for double_ball agent
            cost_out_of_range=jp.where(dist > 35.0, 1.0, 0.0),  # cost specified for different agents 
            cost_too_close=jp.where(dist < 15.0, 1.0, 0.0)
            # cost_out_of_safe=(1. if z in self.agent.healthy_z_range)
            # Step3: Agent should not go too fast(follow a fixed-speed guide)
            # cost_too_fast = (1. if qp.pos[self.agent.rid][0]>=qp.pos[self.obstacle.rid][0] else 0.)
            # continuous cost definision
            # cost_out_of_range = jp.where(jp.abs(z) > 2, jp.abs(z) - 2. ,0.)
            # cost_out_of_range = jp.where(jp.abs(z) > 2, jp.square(jp.abs(z) - 2). ,0.)
            # Step4: Agent should not go too high(set limitation in z axis)
            # cost_too_high = jp.where(jp.abs(z) > 1.00, 1., 0.)
            # continuous cost definision
            # cost_too_high = jp.clip(jp.square(1.0*(jp.abs(z)-1.00)),0,10.0)
            # Step 5: Agent should not go too far from target
            # cost_too_far = jp.where(jp.norm(qp.pos[self.agent.rid]-self.gate.init_pos)>16.0,1.,0.)
            # Step 6. Agent should not go too close to the target
            # cost_too_close = jp.where(jp.norm(qp.pos[self.agent.rid]-self.gate.init_pos)<2.0,1.,0.)
        )
        cost = sum(v for k, v in cs.items())
        return cost

    def get_obs(self, sys: System, qp: QP, info: Info) -> jp.ndarray:
        """Observe the agent's position and velocity."""
        agent_related_obs = self.agent.asset_related_observation(sys, qp, info)

        cost_related_obs = [
            cost_fn(sys, qp, info) for cost_fn in self._cost_related_obs
        ]
        return jp.concatenate(agent_related_obs + cost_related_obs)

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


if __name__ == "__main__":
    ant = Ant()
    circle = Target(ant)
    print(circle._config.collide_include)
