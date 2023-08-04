import os

from safety_brax.envs.assets import Asset


class Ground(Asset):
    """
    ### Ground
    The ground is a frozen body with a plane collider.
    """

    _default_config_path = os.path.join(os.path.dirname(__file__), "json/ground.json")
    _default_collider_list = ["Ground",]

    def __init__(self, config_path: str = None, unique_name: bool = False) -> None:
        super().__init__(config_path, unique_name)


if __name__ == "__main__":
    ground = Ground()
    print(ground._config)
    print(ground.collider_list)
