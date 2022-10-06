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

"""Trainer components for Mava systems."""
from mava.components.training.advantage_estimation import GAE
from mava.components.training.base import Batch, TrainingState, Utility
from mava.components.training.losses import MAPGWithTrustRegionClippingLoss
from mava.components.training.model_updating import MAPGEpochUpdate, MAPGMinibatchUpdate
from mava.components.training.step import DefaultTrainerStep, MAPGWithTrustRegionStep
from mava.components.training.trainer import (
    BaseTrainerInit,
    CustomTrainerInit,
    OneTrainerPerNetworkInit,
    SingleTrainerInit,
)
