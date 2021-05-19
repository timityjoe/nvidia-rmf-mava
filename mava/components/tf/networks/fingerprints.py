# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

# TODO (StJohn):
#   - [] Complete class for monotonic mixing
#   - [] Generalise Qmixing to allow for different activations,
#          hypernetwork structures etc.
#   - [] Decide on whether to accept an 'args' term or receive each arg individually.
#   - [] Think about how to default values.

# NOTE (StJohn): I'm still thinking about the structure in general. One of the biggest
# design choices at the moment is how to structure what is passed to the __init__ when
# instantiating vs what is passed to the forward functions.

# Code inspired by PyMARL framework implementation
# https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py

"""Sonnet module that takes two inputs
[observation, fingerprint]"""

# import launchpad as lp
import sonnet as snt
import tensorflow as tf


class ObservationNetworkWithFingerprint(snt.Module):
    """Sonnet module that takes two inputs
    [observation, fingerprint]."""

    def __init__(
        self,
        observation_network: snt.Module,
    ) -> None:
        """Initializes network.
        Args:
            observation_network: ...
            fingerprint_dim: ...
        """
        super(ObservationNetworkWithFingerprint, self).__init__()
        self._observation_network = observation_network

    def __call__(
        self,
        obs: tf.Tensor,
        fingerprint: tf.Tensor,
    ) -> tf.Tensor:

        hidden = self._observation_network(obs)

        # TODO (Claude) when observation network is a
        # Conv net we will need to do some sort of flattening.
        # I dont think the below is exactly right.
        # hidden = tf.keras.layers.Flatten(hidden)

        hidden_with_fingerprint = tf.concat([hidden, fingerprint], axis=1)

        return hidden_with_fingerprint
