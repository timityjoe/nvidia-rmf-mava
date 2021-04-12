# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Example running IDQN on pettinzoo MPE environments."""

import importlib
from typing import Any, Dict, Mapping, Sequence, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl
from absl import app, flags
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.environment_loops.pettingzoo import PettingZooParallelEnvironmentLoop
from mava.systems.tf import executors, idqn
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 100, "Number of training episodes to run for.")

flags.DEFINE_integer(
    "num_episodes_per_eval",
    10,
    "Number of training episodes to run between evaluation " "episodes.",
)


def make_environment(
    env_class: str = "mpe", env_name: str = "simple_v2", **kwargs: int
) -> dm_env.Environment:
    """Creates a MPE environment."""
    env_module = importlib.import_module(f"pettingzoo.{env_class}.{env_name}")
    env = env_module.parallel_env(**kwargs)  # type: ignore
    environment = PettingZooParallelEnvWrapper(env)
    return environment


def make_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (256, 256, 256),
    shared_weights: bool = False
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""
    specs = environment_spec.get_agent_specs()
    if isinstance(networks_layer_sizes, Sequence):
        networks_layer_sizes = {
            key: networks_layer_sizes for key in specs.keys()
        }
    observation_networks = {}
    qnetworks = {}
    policy_networks = {}
    behavior_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(specs[key].actions.shape, dtype=int)

        # Create the shared observation network
        observation_network = tf2_utils.to_sonnet_module(tf.identity)
        
        # Create the Q-network
        qnetwork = snt.Sequential(
            [
                networks.LayerNormMLP(
                    networks_layer_sizes[key], activate_final=True
                ),
                networks.NearZeroInitializedLinear(num_dimensions)
            ]
        )

        # Create the policy network.
        # For now use constant epsilon (since I am not sure how to update eps
        epsilon = tf.Variable(0.05, trainable=False)
        policy_network = snt.Sequential(
            [
                qnetwork,
                lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
            ]
        )

        # Create the behavior policy.
        behavior_network = snt.Sequential(
            [
                observation_network,
                policy_network
            ]
        )

        qnetworks[key] = qnetwork
        policy_networks[key] = policy_network
        observation_networks[key] = observation_network
        behavior_networks[key] = behavior_network

    return {
        "qnetworks": qnetworks,
        "policies": policy_networks,
        "observations": observation_networks,
        "behaviors": behavior_networks,
    }


def main(_: Any) -> None:
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment()
    environment_spec = mava_specs.MAEnvironmentSpec(environment)

    system_networks = make_networks(environment_spec)
    
    # TODO
    # Construct the agent.
    system = idqn.IDQN(
        environment_spec=environment_spec,
        qnetworks=system_networks["qnetworks"],
        policy_networks=system_networks["policies"],
        observation_networks=system_networks[
            "observations"
        ],  # pytype: disable=wrong-arg-types
        behavior_networks=system_networks["behaviors"],
    )

    # Create the environment loop used for training.
    train_loop = PettingZooParallelEnvironmentLoop(
        environment, system, label="train_loop"
    )

    # Create the evaluation policy.
    eval_policies = {
        key: snt.Sequential(
            [
                system_networks["observations"][key],
                system_networks["policies"][key],
            ]
        )
        for key in environment_spec.get_agent_specs().keys()
    }

    # Create the evaluation actor and loop.
    eval_actor = executors.FeedForwardExecutor(policy_networks=eval_policies)
    eval_env = make_environment(remove_on_fall=False)
    eval_loop = PettingZooParallelEnvironmentLoop(
        eval_env, eval_actor, label="eval_loop"
    )

    for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
        train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
        eval_loop.run(num_episodes=1)


if __name__ == "__main__":
    app.run(main)
