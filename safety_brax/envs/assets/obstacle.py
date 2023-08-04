import os

from safety_brax.envs.assets import Asset


class Obstacle(Asset):
    """
    ### Obstacle

    Obstacle is a simple box-shaped obstacle that can be placed in the environment.

    """

    _default_config_path = os.path.join(os.path.dirname(__file__), "json/obstacle.json")
    _default_collider_list = [
        "Obstacle",
    ]
    def __init__(
        self,
        config_path: str = None,
        unique_name: bool = False,
        # body config
        halfsize: list = [2, 2, 2],
        frozen_pos: list = [False, False, False],
    ) -> None:
        super().__init__(config_path, unique_name)

        # update body config
        assert len(self._config.bodies) == 1, "Obstacle should have only one body."
        body = self._config.bodies[0]
        assert len(body.colliders) == 1, "Obstacle should have only one collider."
        collider = body.colliders[0]
        (
            collider.box.halfsize.x,
            collider.box.halfsize.y,
            collider.box.halfsize.z,
        ) = halfsize
        (
            body.frozen.position.x,
            body.frozen.position.y,
            body.frozen.position.z,
        ) = frozen_pos


if __name__ == "__main__":
    obstacle1 = Obstacle(unique_name=True)
    print(obstacle1.core)
