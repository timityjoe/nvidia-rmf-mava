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

"""Testing ExecutorInit class for system builders"""

from types import SimpleNamespace

import pytest

from mava.components.jax.executing.base import ExecutorInit, ExecutorInitConfig
from mava.systems.jax.builder import Builder
from mava.systems.jax.executor import Executor


@pytest.fixture
def dummy_config() -> ExecutorInitConfig:
    """Dummy config attribute for ExecutorInit class

    Returns:
        ExecutorInitConfig
    """
    return ExecutorInitConfig(interval={"test": 1})


def network_factory():
    """Function used in builder.store.networ_factory"""
    return "after_network_factory"


@pytest.fixture
def mock_builder() -> Builder:
    """Mock builder component.

    Returns:
        Builder
    """
    builder = Builder(components=[])
    # store
    store = SimpleNamespace(network_factory=network_factory, networks=None)
    builder.store = store
    return builder


@pytest.fixture
def mock_executor() -> Executor:
    """Mock executor component.

    Returns:
        Executor
    """
    store = SimpleNamespace(is_evaluator=None, observations={})
    executor = Executor(store=store)
    executor._interval = None
    return executor


# test on_building_init
def test_on_building_init(mock_builder: Builder) -> None:
    """Test on_building_init method from ExecutorInit

    Args:
        mock_builder:Builder
    """
    executor_init = ExecutorInit()
    executor_init.on_building_init(builder=mock_builder)

    assert mock_builder.store.networks == "after_network_factory"


# test on_execution_init_start
def test_on_execution_init_start(
    mock_executor: Executor, dummy_config: ExecutorInitConfig
) -> None:
    """Test on_execution_init_start method from ExecutorInit

    Args:
        mock_executor: Executor
        dummy_config: ExecutorInitConfig
    """
    executor_init = ExecutorInit(config=dummy_config)
    executor_init.on_execution_init_start(executor=mock_executor)

    assert mock_executor._interval == dummy_config.interval


# test name
def test_name() -> None:
    """Test name method from ExecutorInit"""
    executor_init = ExecutorInit()

    assert ExecutorInit.name() == "executor_init"
    assert executor_init.name() == "executor_init"


# test config_class
def test_config_class() -> None:
    """Test config_class method from ExecutorInit"""
    executor_init = ExecutorInit()

    assert ExecutorInit.config_class() == ExecutorInitConfig
    assert executor_init.config_class() == ExecutorInitConfig
