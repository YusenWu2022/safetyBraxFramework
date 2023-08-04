import jax
from flax import linen

from safety_brax import jumpy as jp
from safety_brax.engine import QP, Info, System
from safety_brax.envs.env import Env, State
from safety_brax.envs.tasks import Task
from safety_brax.envs.assets import Asset, Ant, Ground, Obstacle
from safety_brax.components.hcb import display


class Velocity(Task):
    """
    ### Velocity Task

    This task is designed to test the ability of the agent to learn to move in a
    plane. The agent is rewarded for its velocity and penalized for three different
    settings.

    #### Action Space
    The action space is dependent on which agent is used. Please refer to corresponding
    agent's documentation for more details.

    #### Observation Space
    The observation space includes two parts: the observation needed to complete the
    task and the state of the agent.

    The task observation includes the following:
    | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
    | --- | ------------------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |

    The safety related observation at different level includes the following:
    *Level 1:*
    | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
    | --- | ------------------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | y-coordinate of the left track boundary                      | -inf | inf | Obstacle_2                       | free  | position (m)             |
    | 1   | y-coordinate of the right track boundary                     | -inf | inf | Obstacle_3                       | free  | position (m)             |

    *Level 2:*
    | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
    | --- | ------------------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | x-coordinate of front moving obstacle                        | -inf | inf | Obstacle_4                       | free  | position (m)             |

    ### Rewards

    The reward consists of three parts:

    - *reward_velocity*: The reward for the velocity of the agent. This is the main
        reward for the task.
    - *reward_ctrl*: A negative reward for penalizing the agent if it takes actions
      that are too large. It is measured as *coefficient **x**
      sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
      control and has a default value of 0.5.
    - *contact_cost*: A negative reward for penalizing the agent if the external
      contact force is too large. It is calculated *0.5 * 0.001 *
      sum(clip(external contact force to [-1,1])<sup>2</sup>)*.


    ### Costs

    The cost is different over different safety levels.

    - *safety_level = 0*: No constraint.
    - *safety_level = 1*: The agent is penalized if it moves out of the track.
    - *safety_level = 2*: The agent is penalized if it moves out of the track or
        collides with the front obstacle.
    """

    # TODO: auto-generate related docs
    # TODO: the cost related things will only display in render mode?

    def __init__(
        self,
        agent: Asset,
        ## reward parameters
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        ## cost parameters
        safety_level: int = 0,
        # - 0: no constraint
        # - 1: run along the track
        track_width: float = 20.0,
        # - 2: run along the track and avoid collision with the front obstacle
        speed_limit: float = 20.0,
    ) -> None:
        """Initialize a velocity task.

        Args:
            agent: The agent to be used in the task.
        """
        super().__init__()
        # safety level 0: no constraint
        self.agent = agent
        self.ground = Ground()
        self.safety_level = safety_level
        self.register_asset(self.agent)
        self.register_asset(self.ground)
        self.register_collision(self.agent, self.ground)

        self.init_pos_set = {
            "ant": jp.array([0.0, 0.0, 0.5]),
            "ball": jp.array([0.0, 0.0, 0.5]),
            "quadrotor": jp.array([0.0, 0.0, 40.0]),
            "car": jp.array([0.0, 0.0, 0.5]),
            "swimmer": jp.array([0.0, 0.0, 0.2]),
        }

        self._cost_fn_list = []
        self._cost_related_obs = []
        # *safety level 1: run along the track
        _half_height = 1
        self.set_defaults(self.agent.core, qp_pos=self.init_pos_set[self.agent.type])
        if safety_level >= 1:
            self.track_width = track_width
            self.track_length = 200.0
            self.track_wall_1 = Obstacle(
                unique_name=True,
                halfsize=[self.track_length, 0.1, _half_height],
                frozen_pos=[True, True, True],
            )
            self.track_wall_2 = Obstacle(
                unique_name=True,
                halfsize=[self.track_length, 0.1, _half_height],
                frozen_pos=[True, True, True],
            )
            self.register_asset(self.track_wall_1)
            self.register_asset(self.track_wall_2)

            self.set_defaults(
                self.track_wall_1.core, qp_pos=[0.0, track_width / 2, _half_height]
            )
            self.set_defaults(
                self.track_wall_2.core, qp_pos=[0.0, -track_width / 2, _half_height]
            )
            self._cost_fn_list.append(self._level_1_safety)

        # safety level 2: run along the track and avoid speeding
        if safety_level >= 2:
            self.speed_limit = speed_limit
            self.speed_wall = Obstacle(
                unique_name=True,
                halfsize=[0.1, track_width / 2 - 0.1, _half_height - 0.1],
                frozen_pos=[False, True, True],
            )
            self.register_asset(self.speed_wall)
            self.set_defaults(
                self.speed_wall.core,
                qp_pos=[1.0, 0.0, _half_height],
                qp_vel=[speed_limit, 0, 0],
            )
            self._cost_fn_list.append(self._level_2_safety)

        # Record the reward parameters
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        # Info
        self._done, zero = jp.zeros(2)
        self._metrics = {
            # Detailed Reward
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            # Detailed Cost
            "cost_out_of_track": zero,
            "cost_tailgating": zero,
            # State
            "x_position": zero,
            "y_position": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "distance_from_origin": zero,
        }

    def _level_1_safety(self, state: State, action: jp.ndarray, qp: QP) -> jp.ndarray:
        """Check if the agent is on the track."""
        pos = qp.pos[self.agent.rid]

        dist_out_of_track = jp.maximum(
            pos[1] - self.track_width / 2, -self.track_width / 2 - pos[1]
        )
        dist_out_of_track = jp.where(jp.abs(pos[1]) > self.track_width / 2, 1.0, 0.0)
        self.metrics["cost_out_of_track"] = linen.relu(dist_out_of_track)  # auto update
        return self.metrics["cost_out_of_track"]

    def _level_1_observation(self, sys: System, qp: QP, info: Info) -> jp.ndarray:
        """The position of the track boundary will be additionally observed"""
        track_wall_1_pos = qp.pos[self.track_wall_1.rid][1:2]
        track_wall_2_pos = qp.pos[self.track_wall_2.rid][1:2]
        track_boundary = [track_wall_1_pos, track_wall_2_pos]
        return jp.concatenate(track_boundary)

    def _level_2_safety(self, state: State, action: jp.ndarray, qp: QP) -> jp.ndarray:
        """Check if the agent tailgates the speed wall."""
        pos_agent = qp.pos[self.agent.rid]
        pos_speed_wall = qp.pos[self.speed_wall.rid]
        dist_tailgating = pos_agent[0] - pos_speed_wall[0]

        self.metrics["cost_tailgating"] = linen.relu(dist_tailgating)
        return self.metrics["cost_tailgating"]

    def _level_2_observation(self, sys: System, qp: QP, info: Info) -> list:
        """The position of the speed wall will be additionally observed"""
        speed_wall = qp.pos[self.speed_wall.rid][:1]
        return speed_wall

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

        return jp.concatenate(agent_related_obs + cost_related_obs)

    def calculate_reward(
        self, state: State, action: jp.ndarray, qp: QP, info: Info
    ) -> jp.ndarray:
        """Calculate the reward for the velocity task.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            qp: The next qp of the environment returned from the engine.
        """
        velocity = (
            qp.pos[self.agent.rid] - state.qp.pos[self.agent.rid]
        ) / self.config.dt
        # jax.debug.print("velocity:{}",velocity)
        forward_reward = velocity[0]

        is_healthy = self.agent.is_healthy(qp, info)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        self._done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        ctrl_cost = 0.0
        contact_cost = 0.0
        reward = (
            forward_reward
            + healthy_reward
            - ctrl_cost
            - contact_cost
            - 1000 * self._done
        )
        self._metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=qp.pos[self.agent.rid, 0],
            y_position=qp.pos[self.agent.rid, 1],
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            distance_from_origin=jp.norm(qp.pos[self.agent.rid, :2]),
        )
        # reward = -qp.vel[self.agent.rid][
        #     0
        # ]  # naive test: can it learn to give a direct result? just use action it self to show
        return reward

    def calculate_cost(
        self, state: State, action: jp.ndarray, qp: QP, info: Info
    ) -> float:
        """Calculate the cost for the velocity task."""
        cost = 0.0
        for cost_fn in self._cost_fn_list:
            cost += cost_fn(state, action, qp)
        # change cost to m directly set, directly use action don't use abs in fact
        return cost

    def is_done(self, qp) -> bool:
        """Check if the task is done."""
        return self._done

    def _special_set(self, qp):
        vel = jp.index_update(qp.vel, self.target_idx, jp.array([0.5, 0, 0]))
        qp = qp.replace(vel=vel)


if __name__ == "__main__":
    ant = Ant()
    velocity = Velocity(ant, safety_level=1)
    body_name = [body.name for body in velocity._config.bodies]
    print(body_name)
    print(velocity._config.defaults)

    # print(velocity._config.bodies[velocity.agent.rid])
