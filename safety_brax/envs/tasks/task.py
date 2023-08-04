from abc import ABC, abstractmethod

from google.protobuf import text_format

from safety_brax import jumpy as jp
from safety_brax.engine import Config, QP, Info
from safety_brax.envs.assets import Asset
from safety_brax.envs import State


class Task(ABC):
    """Base class for tasks."""

    _default_config_txt = """
    friction: 1.0
    gravity { z: -9.8 }
    angular_damping: -0.05
    dt: 0.05
    substeps: 10
    dynamics_mode: "pbd"
    defaults: {}
    """

    def __init__(self) -> None:
        """Initialize a task."""
        self._config = text_format.Parse(self._default_config_txt, Config())
        self._asset_list = []
        self._metrics = {}

    @property
    def metrics(self) -> dict:
        """Return the metrics of the task."""
        return self._metrics

    @property
    def config(self) -> Config:
        """Return the config of the task."""
        return self._config

    def register_asset(self, asset: Asset) -> None:
        """Register assets in the task."""
        self._asset_list.append(asset)
        asset.register(len(self._config.bodies))
        self._config.MergeFrom(asset.config)

    def register_collision(self, asset_a, asset_b) -> None:
        """Register collision between assets."""
        _collision_context = """collide_include {{
            first: "{}"
            second: "{}"
        }}
        """
        for collider_a in asset_a.collider_list:
            for collider_b in asset_b.collider_list:
                collision = text_format.Parse(
                    _collision_context.format(collider_a, collider_b), Config()
                )
                self._config.MergeFrom(collision)

    def set_defaults(
        self, name: str, qp_pos: list = [0.0, 0.0, 0.0], qp_vel: list = [0.0, 0.0, 0.0], qp_rot: list = [0.0,0.0,0.0]
    ) -> None:
        """Set default configurations."""
        assert len(self._config.defaults) == 1, "Only one default config is allowed."
        qp = self._config.defaults[0].qps.add(name=name)
        qp.pos.x, qp.pos.y, qp.pos.z = qp_pos
        qp.vel.x, qp.vel.y, qp.vel.z = qp_vel
        qp.rot.x, qp.rot.y, qp.rot.z = qp_rot

    @abstractmethod
    def reset(self, rng: jp.ndarray) -> None:
        """Reset the task."""
        pass

    @abstractmethod
    def get_obs(self) -> jp.ndarray:
        """Return the observation of the task."""
        pass

    @abstractmethod
    def calculate_reward(
        self, state: State, action: jp.ndarray, qp: QP, info: Info
    ) -> float:
        """Calculate the reward for the current state."""
        pass

    @abstractmethod
    def calculate_cost(
        self, state: State, action: jp.ndarray, qp: QP, info: Info
    ) -> float:
        """Calculate the cost for the current state."""
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Return whether the task is done."""
        pass


if __name__ == "__main__":
    print(Task._default_config_txt)
    task = Task()
    print(task._config)
