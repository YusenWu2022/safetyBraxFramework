import os
from tkinter.tix import Tree

from safety_brax.envs.assets import Asset


class Obstacle_platform(Asset):
    """
    ### Obstacle

    One plain board moving towards +x-direction at a certain speed, used to limit agent speed


    """

    _default_config_path = os.path.join(
        os.path.dirname(__file__), "json/obstacle_platform.json")
    _default_collider_list = [
    ]  # no collision required

    def __init__(
        self,
        config_path: str = None,
        unique_name: bool = False,
        halfsize: list = [2, 2, 2],
        frozen_pos: list=[False,False,False]
    ) -> None:
        super().__init__(config_path, unique_name)
        body = self._config.bodies[0]
        body.colliders[0].mesh.scale = halfsize[0]
        (
            body.frozen.position.x,
            body.frozen.position.y,
            body.frozen.position.z,
        ) = frozen_pos



if __name__ == "__main__":
    obstacle1 = Obstacle_platform(unique_name=True)
    print(obstacle1._uid_registry)
    obstacle2 = Obstacle_platform(unique_name=True)
    print(obstacle2._uid_registry) 
    print(Asset._uid_registry)
