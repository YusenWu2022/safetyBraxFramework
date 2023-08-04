from typing import Tuple, Any
from flax import linen
import jax

from abc import ABC, abstractmethod, abstractproperty

from safety_brax import jumpy as jp
from safety_brax.components import networks, distribution, preprocessor
from safety_brax.components.types import PreprocessObservationFn


class Actor(ABC):
    @abstractmethod
    def act(self, observation):
        """Returns an action given an observation."""
        pass

    # @abstractmethod
    # def predict(self, observation):
    #     """Returns a deterministic action given an observation."""
    #     pass

    @abstractproperty
    def parameters(self):
        """Returns the actor parameters."""
        pass

    @abstractmethod
    def load_state(self, parameters):
        pass

    def __call__(self, observation):
        return self.act(observation)


class MLPNormalActor(Actor):
    """A normal distribution actor."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        observation_preprocessor: PreprocessObservationFn,
        preprocessor_params: Any,
        policy_hidden_layer_sizes: Tuple[int, ...],
        activation: str,
        init_key,
    ):
        """Initializes the actor.

        Args:
            observation_size: The size of the observation.
            action_size: The size of the action.
            observation_preprocessor: A function that preprocesses the observation.
            policy_hidden_layer_sizes: The sizes of the hidden layers.
            activation: The activation function.
            init_key: The key used to initialize the actor.
        """
        self.observation_size = observation_size
        self.action_size = action_size
        self.observation_preprocessor = observation_preprocessor

        # network initialization
        self._normal_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
        distribution_params_size = self._normal_distribution.param_size
        layer_sizes = list(policy_hidden_layer_sizes) + [distribution_params_size]
        activation = networks.make_activation_fn(activation)
        self._mlp_policy_network = networks.MLP(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )

        # parameter initialization
        self._preprocessor_params = preprocessor_params
        self._policy_network_params = self._init_policy_network_params(init_key)

    def _init_policy_network_params(self, init_key):
        """Initializes the policy network parameters."""
        dummy_obs = jp.zeros((1, self.observation_size))
        return self._mlp_policy_network.init(init_key, dummy_obs)

    def act_(self, network_params, preprocessor_params, observation, sampling_key):
        """Returns an action given an observation. This function is used for jax.jit
        acceleration.

        Args:
            network_params: The parameters of the network.
            observation: The observation.
            sampling_key: The key used for sampling.
        """
        observation = self.observation_preprocessor(observation, preprocessor_params)
        logits = self._mlp_policy_network.apply(network_params, observation)
        raw_action = self._normal_distribution.sample_no_postprocessing(
            logits, sampling_key
        )
        log_prob = self._normal_distribution.log_prob(logits, raw_action)
        postprocessed_action = self._normal_distribution.postprocess(raw_action)
        return postprocessed_action, {
            "log_prob": log_prob,
            "raw_action": raw_action,
        }

    def act(self, observation, sampling_key):
        """Returns an action given an observation."""
        return self.act_(
            self._policy_network_params,
            self._preprocessor_params,
            observation,
            sampling_key,
        )

    # def predict(self, observation):
    #     """Returns a deterministic action given an observation."""
    #     observation = self.observation_preprocessor(
    #         observation, self.preprocessor_params
    #     )
    #     logits = self._mlp_policy_network.apply(
    #         self._policy_network_params, observation
    #     )
    #     return self._normal_distribution.mode(logits)

    def log_prob_(
        self, network_params, preprocessor_params, observation, raw_action, entropy_key
    ):
        """Returns the log probability and entropy of the raw action given an observation.
        This is used for computing th gradient of network parameters.

        Args:
            observation: The observation.
            raw_action: The raw action before postprocessing.
            key_sample: The random key used for the entropy.
        """
        observation = self.observation_preprocessor(observation, preprocessor_params)
        logits = self._mlp_policy_network.apply(network_params, observation)
        log_prob = self._normal_distribution.log_prob(logits, raw_action)
        entropy = self._normal_distribution.entropy(logits, entropy_key)
        return log_prob, entropy

    def log_prob(self, observation, raw_action, entropy_key):
        """Returns the log probability and entropy of the raw action given an observation.

        Args:
            observation: The observation.
            raw_action: The raw action before postprocessing.
            key_entropy: The random key used for the entropy."""
        return self.log_prob_(
            self._policy_network_params,
            self._preprocessor_params,
            observation,
            raw_action,
            entropy_key,
        )

    @property
    def parameters(self):
        """Returns the network parameters"""
        return self._policy_network_params

    @property
    def preprocessor_params(self):
        """Returns the preprocessor parameters"""
        return self._preprocessor_params

    def load_state(self, parameters, preprocessor_params):
        """Loads the actor state"""
        self._policy_network_params = parameters
        self._preprocessor_params = preprocessor_params
