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

"""Integration test of The Executor for Jax-based Mava systems"""

import functools

import pytest

from mava.systems.jax import System
from mava.types import OLT
from mava.utils.environments import debugging_utils
from tests.jax.integration.mock_systems import mock_system_single_process

# Environment.
environment_factory = functools.partial(
    debugging_utils.make_environment,
    env_name="simple_spread",
    action_space="discrete",
)


@pytest.fixture
def test_system_sp() -> System:
    """A single process built system"""
    return mock_system_single_process()


def test_executor_single_process_with_adder(test_system_sp: System) -> None:
    """Test if the executor instantiates processes as expected."""
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_sp._builder.store.system_build

    # _writer.append needs to be called once to get _writer.history
    # _writer.append called in observe_first and observe
    with pytest.raises(RuntimeError):
        assert executor._executor.store.adder._writer.history

    # Run an episode
    executor.run_episode()

    # Observe first and observe
    assert executor._executor.store.adder._writer.history
    assert list(executor._executor.store.adder._writer.history.keys()) == [
        "observations",
        "start_of_episode",
        "actions",
        "rewards",
        "discounts",
        "extras",
    ]
    assert list(
        executor._executor.store.adder._writer.history["observations"].keys()
    ) == ["agent_0", "agent_1", "agent_2"]
    assert (
        type(executor._executor.store.adder._writer.history["observations"]["agent_0"])
        == OLT
    )

    assert len(executor._executor.store.adder._writer._column_history) != 0

    # Select actions and select action
    assert list(executor._executor.store.actions_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert list(executor._executor.store.policies_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    num_possible_actions = environment_factory().action_spec()["agent_0"].num_values
    assert (
        lambda: x in range(0, num_possible_actions)
        for x in list(executor._executor.store.actions_info.values())
    )
    assert (
        lambda: key == "log_prob"
        for key in executor._executor.store.policies_info.values()
    )


def test_executor_single_process_without_adder(test_system_sp: System) -> None:
    """Test if the executor instantiates processes as expected."""
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_sp._builder.store.system_build

    # Remove adder
    executor._executor.store.adder = None
    # Run an episode
    executor.run_episode()

    # Observe first (without adder)
    assert executor._executor.store.adder is None

    # Select actions and select action
    assert list(executor._executor.store.actions_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert list(executor._executor.store.policies_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    num_possible_actions = environment_factory().action_spec()["agent_0"].num_values
    assert (
        lambda: x in range(0, num_possible_actions)
        for x in list(executor._executor.store.actions_info.values())
    )

    assert (
        lambda: key == "log_prob"
        for key in executor._executor.store.policies_info.values()
    )

    # Observe (without adder)
    assert not hasattr(executor._executor.store.adder, "add")
