import os
from safety_brax import jumpy as jp
from safety_brax.envs.assets import Asset
from safety_brax.engine import QP, Info, System
from safety_brax.components import math
import jax
import jax.experimental.host_callback as hcb
from copy import deepcopy


def host_cb(args, t):
    print(args)


class Quadrotor(Asset):
    """
    ### Drone robot
    The ant is a 3D robot consisting of one torso (free rotational body) with
    four wings attached to it with each one having two links.

    ### Action Space
    The agent take a 4-element vector for actions. These will be reflected to 5-dimension real dynamic inforce.

    The action space is a continuous `(action, action, action, action, action,
    action, action, action)` all in `[-1, 1]`, where `action` represents the
    numerical torques applied at the hinge joints.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding config)   |    Joint    | Unit         |
    |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------------|--------------|
    | 1   | Force applied on the front left motor                              | -1          | 1           | motor_1_thruster                 | /           | force (N)    |
    | 2   | Force applied on the front right motor                             | -1          | 1           | motor_2_thruster                 | /           | force (N)    |
    | 3   | Force applied on the back left motor                               | -1          | 1           | motor_3_thruster                 | /           | force (N)    |
    | 4   | Force applied on the back right motor                              | -1          | 1           | motor_4_thruster                 | /           | force (N)    |
    | 5   | Torque applied on the rotor between the torso and virtual frame    | -1          | 1           | torque                           | frame2torso | torque (N m) |

    """

    _default_config_path = os.path.join(
        os.path.dirname(__file__), "json/quadrotor.json"
    )
    _default_collider_list = ["frame"]

    def __init__(
        self,
        config_path: str = None,
        unique_name: bool = False,
    ) -> None:
        super().__init__(config_path, unique_name)
        # for mesh in self._config.meshGeometries[0]:
        self.k_torque = 1.053
        self.type = "quadrotor"

    @property
    def action_size(self) -> int:
        return 4

    def asset_related_observation(self, sys, qp, info) -> list:
        qpos = [qp.pos[self.rid]]
        qvel = [qp.vel[self.rid]]
        # hcb.id_tap(host_cb, qvel)

        if False:
            cfrc = [jp.clip(info.contact.vel, -1, 1), jp.clip(info.contact.ang, -1, 1)]
            # flatten bottom dimension
            cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
        else:
            cfrc = []
        # hcb.id_tap(host_cb,qpos + qvel + cfrc)
        return qpos + qvel + cfrc

    def is_healthy(self, qp, info) -> bool:
        """Return whether the asset is healthy."""
        up = jp.array([0.0, 0.0, 1.0])
        torso_up = math.rotate(up, qp.rot[self.rid])
        # return (1 - jp.abs(torso_up[2]) < 0.5) * (1 - qp.pos[self.rid, 2] <= 0.3)
        return 1

    def action_adaptor(self, state, action):
        """Adapt the action to the actuator."""
        action = jp.clip(action, -1, 1)
        action = jp.square(action * 0.10 + 1)
        up = jp.array([0.0, 0.0, 1.0])  # new version z-y-x turn placed
        torso_up = math.rotate(up, state.qp.rot[self.rid])
        actions_fake = jp.zeros((12))
        action_sys = jax.lax.concatenate([deepcopy(action), actions_fake], dimension=0)
        damping = -(jp.norm(state.qp.vel[self.rid]) ** 2) * 0.03

        for i in [3, 2, 1, 0]:
            action_sys = action_sys.at[2 + 3 * i].set(action[i] * torso_up[2])
            action_sys = action_sys.at[1 + 3 * i].set(action[i] * torso_up[1])
            action_sys = action_sys.at[0 + 3 * i].set(action[i] * torso_up[0])

        action_sys = action_sys.at[2 + 3 * 4].set(damping * state.qp.vel[self.rid, 2])
        action_sys = action_sys.at[1 + 3 * 4].set(damping * state.qp.vel[self.rid, 1])
        action_sys = action_sys.at[0 + 3 * 4].set(damping * state.qp.vel[self.rid, 0])

        torque = self.k_torque * (action[0] - action[1] + action[2] - action[3])
        action_sys = action_sys.at[15].set(0)
        return action_sys


if __name__ == "__main__":
    ant1 = Quadrotor(unique_name=True)
    print(ant1._uid_registry)
    ant2 = Quadrotor(unique_name=True)
    print(ant2._uid_registry)
    print(Asset._uid_registry)
