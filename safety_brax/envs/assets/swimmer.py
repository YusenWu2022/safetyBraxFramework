"""Swimmer robot."""
import os

from safety_brax import jumpy as jp
from safety_brax.envs.assets import Asset
import jax.experimental.host_callback as hcb


def host_cb(args, t):
    print(args)


class Swimmer(Asset):
    """
    ### Swimmer robot with two capsule legs to twister.
    """

    _default_config_path = os.path.join(os.path.dirname(__file__), "json/swimmer.json")
    _default_collider_list = ["Torso","leg 1", "leg 2"]

    def __init__(self, config_path: str = None, unique_name: bool = False) -> None:
        super().__init__(config_path, unique_name)
        self.type = "swimmer"

    # @property
    # def action_size(self) -> int:   # 3*3(forces)+1(actuators)=10
    #     return 10

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

    def action_adaptor(self, state, action):  # directly return output action
        return jp.clip(action, -1.0, 1.0)
        # return action # for actuators do not clip instead

    def is_healthy(self, qp, info) -> bool:
        # return jp.where(jp.norm(qp.pos[self.rid][:2])>26.0,0.0,1.0)
        return 1
