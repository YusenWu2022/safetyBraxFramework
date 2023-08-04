import os

from safety_brax import jumpy as jp
from safety_brax.envs.assets import Asset


class Ant(Asset):
    """
    ### Ant robot
    The ant is a 3D robot consisting of one torso (free rotational body) with
    four legs attached to it with each leg having two links, named `Aux` and `Foot`.

    ### Action Space
    The agent take a 8-element vector for actions.

    The action space is a continuous `(action, action, action, action, action,
    action, action, action)` all in `[-1, 1]`, where `action` represents the
    numerical torques applied at the hinge joints.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
    |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the rotor between the torso and front left hip   | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
    | 1   | Torque applied on the rotor between the front left two links       | -1          | 1           | ankle_1 (front_left_leg)         | hinge | torque (N m) |
    | 2   | Torque applied on the rotor between the torso and front right hip  | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between the front right two links      | -1          | 1           | ankle_2 (front_right_leg)        | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between the torso and back left hip    | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between the back left two links        | -1          | 1           | ankle_3 (back_leg)               | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the torso and back right hip   | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between the back right two links       | -1          | 1           | ankle_4 (right_back_leg)         | hinge | torque (N m) |
    """

    _default_config_path = os.path.join(os.path.dirname(__file__), "json/ant.json")
    _default_collider_list = [
        "Torso",
        "Foot 1",
        "Foot 2",
        "Foot 3",
        "Foot 4",
    ]

    def __init__(
        self,
        config_path: str = None,
        unique_name: bool = False,
        healthy_z_range: tuple = (0.2, 1.0),
    ) -> None:
        super().__init__(config_path, unique_name)
        self.healthy_z_range = healthy_z_range
        self.type = "ant"

    """
    def asset_related_observation(self, sys, qp, info) -> list:

        joint_angle, joint_vel = sys.joints[self.rid].angle_vel(qp)
        # qpos: position and orientation of the torso and the joint angles.
        if True:
            qpos = [qp.pos[self.rid, 2:], qp.rot[self.rid], joint_angle]
        else:
            qpos = [qp.pos[self.rid], qp.rot[self.rid], joint_angle]

        # qvel: velocity of the torso and the joint angle velocities.
        qvel = [qp.vel[self.rid], qp.ang[self.rid], joint_vel]

        # external contact forces:
        # delta velocity (3,), delta ang (3,) * 10 bodies in the system
        if False:
            cfrc = [
                jp.clip(info.contact.vel, -1, 1),
                jp.clip(info.contact.ang, -1, 1)
            ]
            # flatten bottom dimension
            cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
        else:
            cfrc = []

        return qpos + qvel + cfrc
    """

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

    # def action_adaptor(
    #     self, state, action
    # ):  # limit action in a certain range
    #     action = jp.clip(action, -1.0, 1.0)
    #     return action

    def is_healthy(self, qp, info) -> bool:
        # return jp.where(jp.norm(qp.pos[self.rid][:2])>26.0,0.0,1.0)
        return jp.where(jp.abs(qp.pos[self.rid, 2] < 0.3), 0.0, 1.0)


if __name__ == "__main__":
    ant1 = Ant(unique_name=True)
    print(ant1._uid_registry)
    ant2 = Ant(unique_name=True)
    print(ant2._uid_registry)
    print(Asset._uid_registry)


# how to get this done for diff-cpo algorithm? s
