import os

from safety_brax.envs.assets import Asset


class Gate(Asset):
    """
    ### Gate
    
    One goal with gate mesh that agent can go through

   
    """

    _default_config_path = os.path.join(os.path.dirname(__file__), "json/gate.json")
    _default_collider_list = [
    ] # no collision required

    def __init__(
        self,
        config_path: str = None,
        unique_name: bool = False,
        halfsize: list = [2, 2, 2],
        frozen_pos: list=[False,False,False]
    ) -> None:
        super().__init__(config_path, unique_name)
        body = self._config.bodies[0]
        (
            body.frozen.position.x,
            body.frozen.position.y,
            body.frozen.position.z,
        ) = frozen_pos




if __name__ == "__main__":
    obstacle1 = Gate(unique_name=True)
    print(obstacle1._uid_registry)
    obstacle2 = Gate(unique_name=True)
    print(obstacle2._uid_registry)
    print(Asset._uid_registry)
