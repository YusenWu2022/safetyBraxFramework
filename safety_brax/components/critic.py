from typing import Tuple, Any
from abc import ABC, abstractmethod, abstractproperty

import jax


from safety_brax import jumpy as jp
from safety_brax.components import networks, types


class Critic(ABC):
    @abstractmethod
    def value(self, observation, action):
        """Returns the value of an observation and action."""
        pass

    @abstractproperty
    def parameters(self):
        """Returns the critic parameters."""
        pass

    @abstractmethod
    def load_state(self, parameters):
        pass

    def __call__(self, observation, action):
        return self.value(observation, action)


class MLPVCritic(Critic):
    """A MLP value critic."""

    def __init__(
        self,
        observation_size: int,
        observation_preprocessor: types.PreprocessObservationFn,
        preprocessor_params: Any,
        hidden_layer_sizes: Tuple[int, ...],
        activation: str,
        init_key,
    ):
        """Initializes the critic.

        Args:
            observation_size: The size of the observation.
            observation_preprocessor: A function that preprocesses the observation.
            hidden_layer_sizes: The sizes of the hidden layers.
            activation: The activation function.
            init_key: The key used to initialize the network parameters.
        """
        self.observation_size = observation_size
        self.observation_preprocessor = observation_preprocessor

        # Initialize the network.
        layer_sizes = list(hidden_layer_sizes) + [1]
        activation = networks.make_activation_fn(activation)
        self._mlp_network = networks.MLP(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )
        # Initialize the parameters.
        self._preprocessor_params = preprocessor_params
        self._network_params = self._init_network_params(init_key)

    def _init_network_params(self, init_key):
        """Initializes the network parameters."""
        dummy_obs = jp.zeros((1, self.observation_size))
        return self._mlp_network.init(init_key, dummy_obs)

    def value_(self, network_params, preprocessor_params, observation):
        """Returns the value of given observation. This is used for computing gradients
        of network parameters."""
        observation = self.observation_preprocessor(observation, preprocessor_params)
        v = self._mlp_network.apply(network_params, observation)
        return jp.squeeze(v, axis=-1)

    def value(self, observation):
        """Returns the value of an observation and action."""
        return self.value_(self.parameters, self._preprocessor_params, observation)

    @property
    def parameters(self):
        """Returns the critic network parameters."""
        return self._network_params

    @property
    def preprocessor_params(self):
        """Returns the critic preprocessor parameters."""
        return self._preprocessor_params

    def load_state(self, parameters, preprocessor_params):
        """Loads the critic state with a tuple of (network_params, preprocessor_params)."""
        self._network_params = parameters
        self._preprocessor_params = preprocessor_params

    def __call__(self, observation):
        return self.value(observation)
