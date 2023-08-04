from abc import ABC
import uuid

from google.protobuf import json_format
from google.protobuf.message import Message

from safety_brax.engine import Config


class Asset(ABC):
    """Base class for all assets."""

    _default_config_path = ""
    _uid_registry = []
    _default_collider_list = []

    def __init__(self, config_path: str = None, unique_name: bool = False) -> None:
        """Initialize an asset.

        Args:
            `config_path`: Path to the config file. If not provided, use the default config.
            `unique_name`: Whether to rename the bodies, joints, and actuators with a uid.
        """
        self._config: Config = None
        self._uid = self._generate_uid()
        self._rid = None  # The registered id in the environment.
        if config_path is None:
            assert self._default_config_path != "", "Default config path is not set."
            self.load_config_from_json(self._default_config_path)
        else:
            self.load_config_from_json(config_path)

        self._collider_list = self._default_collider_list
        self._body_list = [body.name for body in self._config.bodies]
        self._joint_list = [joint.name for joint in self._config.joints]
        self._actuator_list = [actuator.name for actuator in self._config.actuators]

        self.unique_name = unique_name
        if unique_name:
            self._refactor_unique_name()
        self.type = ""

    @property
    def config(self) -> Config:
        """Get the config."""
        return self._config

    @property
    def uid(self) -> str:
        """Return the uid of the asset."""
        return self._uid

    @property
    def rid(self) -> int:
        """Get the registered id."""
        return self._rid

    @property
    def collider_list(self) -> list:
        """Return the collision participants of the asset."""
        if self.unique_name:
            return [collider + "_" + self._uid for collider in self._collider_list]
        else:
            return self._collider_list

    @property
    def core(self) -> str:
        """Return the core of the asset, which is used to determine the position of the asset."""
        return self._config.bodies[0].name

    @property
    def action_size(self) -> int:
        return None

    def register(self, rid: int) -> None:
        """Register the asset in the environment."""
        self._rid = rid

    def _generate_uid(self, method="order") -> str:
        """Generate a unique id.

        Args:
            `method`: The method to generate the uid. Currently only support:
                uuid4 - uuid.uuid4().
                order - the order of the asset in the environment.
        """
        if method == "uuid4":
            uid = str(uuid.uuid4())
            while uid in self._uid_registry:
                uid = str(uuid.uuid4())
        elif method == "order":
            uid = str(len(self._uid_registry))
        else:
            raise NotImplementedError
        self._uid_registry.append(uid)
        return uid

    def _refactor_unique_name(self) -> None:
        """Refactor the name of the bodies, joints, and actuators with a uid."""
        for body in self._config.bodies:
            body.name = body.name + "_" + self._uid
        for joint in self._config.joints:
            joint.name = joint.name + "_" + self._uid
            joint.parent = joint.parent + "_" + self._uid
            joint.child = joint.child + "_" + self._uid
        for force in self._config.forces:
            force.name = force.name + "_" + self._uid
            force.body = force.body + "_" + self._uid
        for actuator in self._config.actuators:
            actuator.name = actuator.name + "_" + self._uid
            actuator.joint = actuator.joint + "_" + self._uid
        for default in self._config.defaults:
            for qp in default.qps:
                qp.name = qp.name + "_" + self._uid

    def load_config_from_json(self, json_path: str) -> None:
        """Load the config from a json file."""
        with open(json_path, "r") as f:
            self._config = json_format.Parse(f.read(), Config())

    def save_config_to_json(self, json_path: str) -> None:
        """Save the config to a json file."""
        assert self._config is not None, "Config is not loaded."
        with open(json_path, "w") as f:
            f.write(json_format.MessageToJson(self._config))

    def asset_related_observation(self, sys, qp, info) -> list:
        """Return the asset related observation."""
        return []

    def is_healthy(self, qp, info) -> bool:
        """Return whether the asset is healthy."""
        return True

    def action_adaptor(self, state, action):
        """Adapt the action to the actuator."""
        return action
