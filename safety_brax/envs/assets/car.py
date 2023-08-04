"""Double point robot."""
import os
import jax
from safety_brax import jumpy as jp
from safety_brax.envs.assets import Asset
import jax.experimental.host_callback as hcb
from copy import deepcopy
from flax import linen


def host_cb(args, t):
    print(args)


class Car(Asset):
    """
    ### Point robot.
    """

    _default_config_path = os.path.join(
        os.path.dirname(__file__), "json/car.json"
    )
    _default_collider_list = ["Torso"]

    def __init__(self, config_path: str = None, unique_name: bool = False) -> None:
        super().__init__(config_path, unique_name)
        self.type = "car"

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

    def action_adaptor(self, state, action):
        """Adapt the action to the actuator."""
        # action = linen.sigmoid(action)
        action = jp.clip(action, -1.0, 1.0)   # use sigmoid instead of clip to get reward smooth 
        torso_up_1 = state.qp.pos[self.rid] - state.qp.pos[self.rid + 1]
        torso_up_2 = state.qp.pos[self.rid] - state.qp.pos[self.rid + 2]
        x, y = torso_up_1[1], -torso_up_1[0]
        torso_wheel_1 = jp.array([x, y])
        x, y = -torso_up_2[1], torso_up_2[0]
        torso_wheel_2 = jp.array([x, y])
        torso_wheel_1 /= jp.norm(torso_wheel_1)
        torso_wheel_2 /= jp.norm(torso_wheel_2)
        action_sys = jax.lax.concatenate([deepcopy(action), jp.zeros(4)], dimension=0)
        direction = jp.where(
            jp.norm(state.qp.vel[self.rid]) > 0,
            jp.array(state.qp.vel[self.rid] / jp.norm(state.qp.vel[self.rid])),
            jp.array([0.0, 0.0, 0.0]),
        )
        action_sys = action_sys.at[0].set(
            torso_wheel_1[0] * action[0] - direction[0] * 0.05
        )
        action_sys = action_sys.at[1].set(
            torso_wheel_1[1] * action[0] - direction[1] * 0.05
        )
        action_sys = action_sys.at[3].set(
            torso_wheel_2[0] * action[1] - direction[0] * 0.05
        )
        action_sys = action_sys.at[4].set(
            torso_wheel_2[1] * action[1] - direction[1] * 0.05
        )
        action_sys = action_sys.at[2].set(0.0)
        action_sys = action_sys.at[5].set(0.0)
        return action_sys

    def is_healthy(self, qp, info) -> bool:
        # return jp.where(jp.norm(qp.pos[self.rid][:2])>26.0,0.0,1.0)
        return 1
