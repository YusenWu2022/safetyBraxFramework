# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training types."""

from typing import Any, Mapping, NamedTuple, Tuple, TypeVar, Iterable, Union
from optax import OptState
import flax
import dataclasses
import jax.numpy as jnp


from safety_brax.envs import State, Env
# from safety_brax.components.actor import Actor
# from safety_brax.components.critic import Critic

NestedArray = jnp.ndarray

# Protocol was introduced into typing in Python >=3.8
# via https://www.python.org/dev/peps/pep-0544/
# Before that, its status was DRAFT and available via typing_extensions
try:
    from typing import Protocol  # pylint:disable=g-import-not-at-top
except ImportError:
    from typing_extensions import Protocol  # pylint:disable=g-import-not-at-top

Params = Any
PRNGKey = jnp.ndarray
Metrics = Mapping[str, jnp.ndarray]
Observation = jnp.ndarray
Action = jnp.ndarray
Extra = Mapping[str, Any]
PolicyParams = Any
PreprocessorParams = Any
PolicyParams = Tuple[PreprocessorParams, Params]
NetworkType = TypeVar("NetworkType")
Config = dict

# Define types for nested arrays and tensors.
NestedArray = jnp.ndarray
NestedTensor = Any

@dataclasses.dataclass(frozen=True)
class Array:
  """Describes a numpy array or scalar shape and dtype.

  Similar to dm_env.specs.Array.
  """
  shape: Tuple[int, ...]
  dtype: jnp.dtype

NestedSpec = Union[
    Array,
    Iterable['NestedSpec'],
    Mapping[Any, 'NestedSpec'],
]

Nest = Union[NestedArray, NestedTensor, NestedSpec]


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    cost: NestedArray
    done: NestedArray
    # discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()
    # # Used for training
    # advantage: NestedArray = None
    # value_target: NestedArray = None

class ActorCriticParams(NamedTuple):
    """Container for actor-critic parameters."""

    actor_params: Params
    critic_params: Params

class ConstrainedActorCriticParams(NamedTuple):
    """Container for actor-critic parameters."""

    actor_params: Params
    critic_params: Params
    c_critic_params: Params

@flax.struct.dataclass
class TrainingState:
    """Container for training state."""
    params: ActorCriticParams
    preprocessor_params: PreprocessorParams
    optimizer_state: OptState
    env_step: jnp.ndarray

class PreprocessObservationFn(Protocol):
    def __call__(
        self,
        observation: Observation,
        preprocessor_params: PreprocessorParams,
    ) -> jnp.ndarray:
        pass


def identity_observation_preprocessor(
    observation: Observation, preprocessor_params: PreprocessorParams
):
    del preprocessor_params
    return observation


