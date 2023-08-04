"""Base Algorithm class."""

from abc import ABC, abstractmethod
from typing import Sequence
import functools

from safety_brax.components import types, gradients
from safety_brax.components.actor import MLPNormalActor
from safety_brax.components.critic import MLPVCritic


class BaseAlgorithm(ABC):
    """Base class for all algorithms."""

    @abstractmethod
    def __init__(self, env: types.Env, config: dict, algo_key: types.PRNGKey):
        """Initialize the algorithm.

        Args:
            env: The environment to train on, must be a subclass of 'Env'.
            config: The configuration dictionary.
            algo_key: The PRNG key used for training and rendering.
        """
        self.env = env
        self.config = config
        self.algo_key = algo_key

        self.observation_preprocessor = None
        self.preprocessor_params = None

        self.optimizer = None

    def _loss_fn(self):
        return

    @abstractmethod
    def train(self):
        """Trains the algorithm."""
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """Renders the environment."""
        raise NotImplementedError

    def _create_actor(self, actor_config: types.Config, actor_init_key: types.PRNGKey):
        """Creates the actor.

        Args:
            actor_config: The configuration dictionary for the actor.
            actor_init_key: The PRNG key used for actor initialization.

        Returns:
            The actor.
        """
        return MLPNormalActor(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            observation_preprocessor=self.observation_preprocessor,
            preprocessor_params=self.preprocessor_params,
            policy_hidden_layer_sizes=actor_config["hidden_layer_sizes"],
            activation=actor_config["activation"],
            init_key=actor_init_key,
        )

    def _create_critic(
        self, critic_config: types.Config, critic_init_key: types.PRNGKey
    ):
        """Creates the critic.

        Args:
            critic_config: The configuration dictionary for the critic.
            critic_init_key: The PRNG key used for critic initialization.

        Returns:
            The critic.
        """
        return MLPVCritic(
            observation_size=self.env.observation_size,
            observation_preprocessor=self.observation_preprocessor,
            preprocessor_params=self.preprocessor_params,
            hidden_layer_sizes=critic_config["hidden_layer_sizes"],
            activation=critic_config["activation"],
            init_key=critic_init_key,
        )

    def _set_gradient_update_fn(self):
        """Sets the gradient update function.

        Returns:
            A function that takes the same argument as the loss function plus the
            optimizer state. The output of this function is the loss, the new
            parameter, and the new optimizer state.
        """
        # self will not leading to leak error, but it is not good practice
        loss_fn = functools.partial(self._loss_fn)

        return gradients.gradient_update_fn(
            loss_fn, self.optimizer, pmap_axis_name=None, has_aux=True
        )  # TODO: pmap_axis_name is None for now, since we don't support multi-GPU training
