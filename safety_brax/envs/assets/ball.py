"""Point robot."""
import os

from safety_brax import jumpy as jp
from safety_brax.envs.assets import Asset
import jax.experimental.host_callback as hcb
from flax import linen


def host_cb(args, t):
    print(args)


class Ball(Asset):
    """
    ### Point robot.
    """

    _default_config_path = os.path.join(os.path.dirname(__file__), "json/ball.json")
    _default_collider_list = ["Torso"]

    def __init__(self, config_path: str = None, unique_name: bool = False) -> None:
        super().__init__(config_path, unique_name)
        self.type = "ball"

    @property
    def action_size(self) -> int:
        return 2

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

    def action_adaptor(
        self, state, action
    ):  # only one force so just first three dimensions

        direction = jp.where(
            jp.norm(state.qp.vel[self.rid]) > 0,
            jp.array(state.qp.vel[self.rid] / jp.norm(state.qp.vel[self.rid])),
            jp.array([0.0, 0.0, 0.0]),
        )
        # velocity: no friction
        # circle: friction
        action += -direction[:2] * 0.10
        # action = jp.clip(action, -1.0, 1.0)
        action = linen.sigmoid(action)
        action = jp.concatenate(
            action + jp.zeros((1, 1))
        )  # concate an additional dimension
        return action

    def is_healthy(self, qp, info) -> bool:
        # return jp.where(jp.norm(qp.pos[self.rid][:2])>26.0,0.0,1.0)
        return 1


        # sphere: 0.25
