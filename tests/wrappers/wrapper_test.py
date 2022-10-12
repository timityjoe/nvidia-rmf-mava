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

from typing import Any, Dict
from unittest.mock import patch

import dm_env
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from dm_env.specs import DiscreteArray

from mava import types
from mava.utils.environments.flatland_utils import check_flatland_import
from mava.wrappers.env_preprocess_wrappers import (
    ConcatAgentIdToObservation,
    ConcatPrevActionToObservation,
    StackObservations,
)
from tests.conftest import EnvSpec, EnvType, Helpers
from tests.enums import EnvSource

_has_flatland = check_flatland_import()
"""
TestEnvWrapper is a general purpose test class that runs tests for environment wrappers.
This is meant to flexibily test various environments wrappers.

    It is parametrize by an EnvSpec object:
        env_name: [name of env]
        env_type: [EnvType.Parallel/EnvType.Sequential]
        env_source: [What is source env - e.g. PettingZoo, RLLibMultiEnv or Flatland]
            - Used in confest to determine which envs and wrappers to load.

    For new environments - you might need to update the Helpers class in conftest.py.
"""


# TODO (Kale-ab): Test dying agents.
@pytest.mark.parametrize(
    "env_spec",
    [
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Parallel),
        EnvSpec("pettingzoo.mpe.simple_spread_v2", EnvType.Sequential),
        EnvSpec("pettingzoo.sisl.multiwalker_v8", EnvType.Parallel),
        EnvSpec("pettingzoo.sisl.multiwalker_v8", EnvType.Sequential),
        EnvSpec("flatland", EnvType.Parallel, EnvSource.Flatland)
        if _has_flatland
        else None,
        EnvSpec("tic_tac_toe", EnvType.Sequential, EnvSource.OpenSpiel),
    ],
)
class TestEnvWrapper:
    def test_loadmodule(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        """Test that we can load a env module

        and that it contains agents and possible agents
        """

        if env_spec is None:
            pytest.skip()
        env = helpers.get_env(env_spec)
        props_which_should_not_be_none = [env, env.agents, env.possible_agents]
        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to load module"

    def test_wrapper_initialization(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        """Test initialization of env wrapper,

        which should have a nested environment,
        an observation and action space for each agent.
        """

        if env_spec is None:
            pytest.skip()
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)
        num_agents = len(wrapped_env.agents)

        props_which_should_not_be_none = [
            wrapped_env,
            wrapped_env.environment,
            wrapped_env.observation_spec(),
            wrapped_env.action_spec(),
            wrapped_env.reward_spec(),
            wrapped_env.discount_spec(),
        ]

        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to ini wrapped env."
        assert (
            len(wrapped_env.observation_spec()) == num_agents
        ), "Failed to generate observation specs for all agents."
        assert (
            len(wrapped_env.action_spec()) == num_agents
        ), "Failed to generate action specs for all agents."
        assert (
            len(wrapped_env.reward_spec()) == num_agents
        ), "Failed to generate reward specs for all agents."
        assert (
            len(wrapped_env.discount_spec()) == num_agents
        ), "Failed to generate discount specs for all agents."

    def test_wrapper_env_reset(self, env_spec: EnvSpec, helpers: Helpers) -> None:
        """Test of reset of wrapper and that dm_env_timestep has basic props."""

        if env_spec is None:
            pytest.skip()
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)
        num_agents = len(wrapped_env.agents)

        timestep = wrapped_env.reset()
        if type(timestep) == tuple:
            dm_env_timestep, env_extras = timestep
        else:
            dm_env_timestep = timestep
        props_which_should_not_be_none = [dm_env_timestep, dm_env_timestep.observation]

        assert helpers.verify_all_props_not_none(
            props_which_should_not_be_none
        ), "Failed to ini dm_env_timestep."
        assert (
            dm_env_timestep.step_type == dm_env.StepType.FIRST
        ), "Failed to have correct StepType."
        if (
            env_spec.env_name == "tic_tac_toe"
            and env_spec.env_source == EnvSource.OpenSpiel
            and env_spec.env_type == EnvType.Sequential
        ):
            pytest.skip(
                "This test is only applicable to parralel wrappers and only works "
                "for the provided PZ sequential envs because they have 3 agents, and"
                "an OLT has length of 3 (a bug, i'd say)"
            )
        assert (
            len(dm_env_timestep.observation) == num_agents
        ), "Failed to generate observation for all agents."
        assert wrapped_env._reset_next_step is False, "_reset_next_step not set."

        helpers.assert_env_reset(wrapped_env, dm_env_timestep, env_spec)

    def test_convert_env_to_dm_env_0_no_action_mask(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        """Test that observations from petting zoo get converted correctly.

        This only runs if wrapper has a _convert_observations
        or _convert_observation functions.
        """

        if env_spec is None:
            pytest.skip()
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            #  Get agent names from env and mock out data
            agents = wrapped_env.agents
            test_agents_observations = {}
            for agent in agents:
                observation_spec = wrapped_env.observation_spec()
                if isinstance(observation_spec[agent].observation, tuple):
                    # Using the first dim for mock observation
                    observation_shape = observation_spec[agent].observation[0].shape
                else:
                    observation_shape = observation_spec[agent].observation.shape

                test_agents_observations[agent] = np.random.rand(
                    *observation_shape
                ).astype(np.float32)

            # Parallel env_types
            if env_spec.env_type == EnvType.Parallel:
                dm_env_timestep = wrapped_env._convert_observations(
                    test_agents_observations, dones={agent: False for agent in agents}
                )

                for agent in wrapped_env.agents:
                    np.testing.assert_array_equal(
                        test_agents_observations[agent],
                        dm_env_timestep[agent].observation,
                    )

                    assert (
                        bool(dm_env_timestep[agent].terminal) is False
                    ), "Failed to set terminal."

            # Sequential env_types
            elif env_spec.env_type == EnvType.Sequential:
                for agent in agents:
                    dm_env_timestep = wrapped_env._convert_observation(
                        agent, test_agents_observations[agent], done=False
                    )

                    np.testing.assert_array_equal(
                        test_agents_observations[agent],
                        dm_env_timestep.observation,
                    )
                    assert (
                        bool(dm_env_timestep.terminal) is False
                    ), "Failed to set terminal."

    def test_convert_env_to_dm_env_0_empty_obs(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        """Test that observations from petting zoo get converted

        to dm observations correctly when empty obs are returned.
        """

        if env_spec is None:
            pytest.skip()
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            #  Get agent names from env and mock out data
            agents = wrapped_env.agents
            test_agents_observations: Dict = {}

            # Parallel env_types
            if env_spec.env_type == EnvType.Parallel:
                dm_env_timestep = wrapped_env._convert_observations(
                    test_agents_observations, dones={agent: False for agent in agents}
                )

                # We have empty OLT for all agents
                for agent in wrapped_env.agents:
                    observation_spec = wrapped_env.observation_spec()
                    if isinstance(observation_spec[agent].observation, tuple):
                        observation_spec_list = []
                        for obs_spec in observation_spec[agent].observation:
                            observation_spec_list.append(
                                np.zeros(
                                    obs_spec.shape,
                                    dtype=obs_spec.dtype,
                                )
                            )

                        for i, obs in enumerate(observation_spec_list):
                            np.testing.assert_array_equal(
                                dm_env_timestep[agent].observation[i],
                                obs,
                            )
                    else:
                        observation = np.zeros(
                            observation_spec[agent].observation.shape,
                            dtype=observation_spec[agent].observation.dtype,
                        )

                        np.testing.assert_array_equal(
                            dm_env_timestep[agent].observation,
                            observation,
                        )

                    np.testing.assert_array_equal(
                        dm_env_timestep[agent].legal_actions,
                        np.ones(
                            wrapped_env.action_spaces[agent].shape,
                            dtype=wrapped_env.action_spaces[agent].dtype,
                        ),
                    )

    def test_convert_env_to_dm_env_2_with_action_mask(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        """Test that observations **with actions masked**

        from petting zoo get converted to dm observations correctly.
        This only runs if wrapper has a _convert_observations
        or _convert_observation functions.
        """

        if env_spec is None:
            pytest.skip()

        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Does the wrapper have the functions we want to test
        if hasattr(wrapped_env, "_convert_observations") or hasattr(
            wrapped_env, "_convert_observation"
        ):
            #  Get agent names from env and mock out data
            agents = wrapped_env.agents
            test_agents_observations = {}
            for agent in agents:
                observation_spec = wrapped_env.observation_spec()
                if isinstance(observation_spec[agent].observation, tuple):
                    # Using the first dim for mock observation
                    observation_shape = observation_spec[agent].observation[0].shape
                else:
                    observation_shape = observation_spec[agent].observation.shape

                # TODO If cont action space masking is implemented - Update
                test_agents_observations[agent] = {
                    "observation": np.random.rand(*observation_shape).astype(
                        np.float32
                    ),
                    "action_mask": np.random.randint(
                        2, size=wrapped_env.action_spaces[agent].shape
                    ).astype(int),
                }
            # Parallel env_types
            if env_spec.env_type == EnvType.Parallel:
                dm_env_timestep = wrapped_env._convert_observations(
                    test_agents_observations,
                    dones={agent: False for agent in agents},
                )

                for agent in wrapped_env.agents:
                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("observation"),  # type: ignore # noqa: E501
                        dm_env_timestep[agent].observation,
                    )
                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("action_mask"),  # type: ignore # noqa: E501
                        dm_env_timestep[agent].legal_actions,
                    )
                    assert (
                        bool(dm_env_timestep[agent].terminal) is False
                    ), "Failed to set terminal."

            # Sequential env_types
            elif env_spec.env_type == EnvType.Sequential:
                for agent in agents:
                    dm_env_timestep = wrapped_env._convert_observation(
                        agent, test_agents_observations[agent], done=False
                    )

                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("observation"),  # type: ignore # noqa: E501
                        dm_env_timestep.observation,
                    )

                    np.testing.assert_array_equal(
                        test_agents_observations[agent].get("action_mask"),  # type: ignore # noqa: E501
                        dm_env_timestep.legal_actions,
                    )
                    assert (
                        bool(dm_env_timestep.terminal) is False
                    ), "Failed to set terminal."

    def test_step_0_valid_when_env_not_done(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        """Test we can take a action and it updates observations"""

        if env_spec is None:
            pytest.skip()

        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 84
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        agents = wrapped_env.agents

        timestep = wrapped_env.reset()
        if type(timestep) == tuple:
            initial_dm_env_timestep, env_extras = timestep
        else:
            initial_dm_env_timestep = timestep
        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }
            curr_dm_timestep = wrapped_env.step(test_agents_actions)

            for agent in wrapped_env.agents:
                assert not np.array_equal(
                    initial_dm_env_timestep.observation[agent].observation,
                    curr_dm_timestep.observation[agent].observation,
                ), "Failed to update observations."

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            curr_dm_timestep = initial_dm_env_timestep
            for agent in agents:
                if env_spec.env_source == EnvSource.OpenSpiel:
                    test_agent_actions = np.random.choice(
                        np.where(curr_dm_timestep.observation.legal_actions)[0]
                    )
                else:
                    test_agent_actions = wrapped_env.action_spaces[agent].sample()

                curr_dm_timestep = wrapped_env.step(test_agent_actions)

                assert not np.array_equal(
                    initial_dm_env_timestep.observation.observation,
                    curr_dm_timestep.observation.observation,
                ), "Failed to update observations."

        assert (
            wrapped_env._reset_next_step is False
        ), "Failed to set _reset_next_step correctly."
        assert curr_dm_timestep.reward is not None, "Failed to set rewards."
        assert (
            curr_dm_timestep.step_type is dm_env.StepType.MID
        ), "Failed to update step type."

    def test_step_1_valid_when_env_not_done(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        """Test we only step in our env once."""

        if env_spec is None:
            pytest.skip()
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 42
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        agents = wrapped_env.agents

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }
            with patch.object(wrapped_env, "step") as parallel_step:
                parallel_step.return_value = None, None, None, None
                _ = wrapped_env.step(test_agents_actions)
                parallel_step.assert_called_once_with(test_agents_actions)

        # Sequential env_types
        elif env_spec.env_type == EnvType.Sequential:
            for agent in agents:
                with patch.object(wrapped_env, "step") as seq_step:
                    seq_step.return_value = None
                    test_agent_action = wrapped_env.action_spaces[agent].sample()
                    _ = wrapped_env.step(test_agent_action)
                    seq_step.assert_called_once_with(test_agent_action)

    def test_step_2_invalid_when_env_done(
        self, env_spec: EnvSpec, helpers: Helpers, monkeypatch: MonkeyPatch
    ) -> None:
        """Test if all agents are done, env is set to done"""

        if env_spec is None:
            pytest.skip()
        wrapped_env, _ = helpers.get_wrapped_env(env_spec)

        if env_spec.env_source == EnvSource.OpenSpiel:
            pytest.skip("Open Spiel does not use the .last() method")

        # Seed environment since we are sampling actions.
        # We need to seed env and action space.
        random_seed = 42
        wrapped_env.seed(random_seed)
        helpers.seed_action_space(wrapped_env, random_seed)

        #  Get agent names from env
        _ = wrapped_env.reset()
        agents = wrapped_env.agents

        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }

            monkeypatch.setattr(wrapped_env, "env_done", helpers.mock_done)

            curr_dm_timestep = wrapped_env.step(test_agents_actions)

            helpers.assert_env_reset(wrapped_env, curr_dm_timestep, env_spec)

        # Sequential env_types
        # TODO (Kale-ab): Make this part below less reliant on PZ.
        elif env_spec.env_type == EnvType.Sequential:
            n_agents = wrapped_env.num_agents

            # Mock functions to act like PZ environment is done
            def mock_environment_last() -> Any:
                observe = wrapped_env.observation_spaces[agent].sample()
                reward = 0.0
                done = True
                info: Dict = {}
                return observe, reward, done, info

            def mock_step(action: types.Action) -> None:
                return

            # Mocks certain functions - if functions don't exist, error is not thrown.
            monkeypatch.setattr(
                wrapped_env._environment, "last", mock_environment_last, raising=False
            )
            monkeypatch.setattr(
                wrapped_env._environment, "step", mock_step, raising=False
            )

            for index, (agent) in enumerate(wrapped_env.agent_iter(n_agents)):
                test_agent_actions = wrapped_env.action_spaces[agent].sample()

                # Mock whole env being done when you reach final agent
                if index == n_agents - 1:
                    monkeypatch.setattr(
                        wrapped_env,
                        "env_done",
                        helpers.mock_done,
                    )

                # Mock update has occurred in step
                monkeypatch.setattr(
                    wrapped_env._environment, "_has_updated", True, raising=False
                )

                curr_dm_timestep = wrapped_env.step(test_agent_actions)

                # Check each agent is on last step
                assert (
                    curr_dm_timestep.step_type is dm_env.StepType.LAST
                ), "Failed to update step type."

            helpers.assert_env_reset(wrapped_env, curr_dm_timestep, env_spec)

        assert (
            wrapped_env._reset_next_step is True
        ), "Failed to set _reset_next_step correctly."
        assert (
            curr_dm_timestep.step_type is dm_env.StepType.LAST
        ), "Failed to update step type."

    def test_wrapper_env_obs_stacking(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        """Test observations frame staking wrapper"""

        if env_spec is None:
            pytest.skip()

        if (
            env_spec.env_name == "tic_tac_toe"
            and env_spec.env_source == EnvSource.OpenSpiel
            or env_spec.env_type == EnvType.Sequential
        ):
            pytest.skip(
                "This test is only applicable to parralel wrappers and only works "
                "for the provided PZ sequential envs because they have 3 agents, and"
                "an OLT has length of 3 (a bug, i'd say)"
            )

        wrapped_env, _ = helpers.get_wrapped_env(env_spec)
        stacked_env = StackObservations(wrapped_env, num_frames=4)
        agents = wrapped_env.agents
        num_frames = 4

        # test if the reset is done correctly
        normal_step = wrapped_env.reset()
        if type(normal_step) == tuple:
            normal_step, _ = normal_step

        stacked_step = stacked_env.reset()
        if type(stacked_step) == tuple:
            stacked_step, _ = stacked_step

        olt_type = isinstance(normal_step.observation[agents[0]], types.OLT)

        for agent in agents:
            if olt_type:
                wrap_obs = normal_step.observation[agent].observation
                stacked_obs = stacked_step.observation[agent].observation
            else:
                wrap_obs = normal_step.observation[agent]
                stacked_obs = stacked_step.observation[agent]

            assert wrap_obs.shape[0] * num_frames == stacked_obs.shape[0]

        # test if step is done correctly
        # Parallel env_types
        if env_spec.env_type == EnvType.Parallel:
            test_agents_actions = {
                agent: wrapped_env.action_spaces[agent].sample() for agent in agents
            }
            stacked_step = stacked_env.step(test_agents_actions)
            if type(stacked_step) == tuple:
                stacked_step, _ = stacked_step

            for agent in agents:
                if olt_type:
                    wrap_obs = normal_step.observation[agent].observation
                    stacked_obs = stacked_step.observation[agent].observation
                else:
                    wrap_obs = normal_step.observation[agent]
                    stacked_obs = stacked_step.observation[agent]

                assert wrap_obs.shape[0] * num_frames == stacked_obs.shape[0]

                size = wrap_obs.shape[0]
                old_obs = stacked_obs[:size]

                for k in range(num_frames - 1):
                    assert np.array_equal(
                        old_obs, stacked_obs[k * size : (k + 1) * size]
                    )

                # TODO thinking of a better way to test this.
                # assert not np.array_equal(
                #     old_obs, stacked_obs[(num_frames - 1) * size :]
                # )

    def test_wrapper_env_obs_stacking_and_concate(
        self, env_spec: EnvSpec, helpers: Helpers
    ) -> None:
        """Test observations frame staking wrapper"""

        if env_spec is None:
            pytest.skip()

        if (
            env_spec.env_name == "tic_tac_toe"
            and env_spec.env_source == EnvSource.OpenSpiel
            or env_spec.env_type == EnvType.Sequential
        ):
            pytest.skip(
                "This test is only applicable to parralel wrappers and only works "
                "for the provided PZ sequential envs because they have 3 agents, and"
                "an OLT has length of 3 (a bug, i'd say)"
            )

        wrapped_env, _ = helpers.get_wrapped_env(env_spec)
        stacked_env = StackObservations(wrapped_env, num_frames=4)
        agents = wrapped_env.agents

        stacked_step = stacked_env.reset()
        if type(stacked_step) == tuple:
            stacked_step, _ = stacked_step

        olt_type = isinstance(stacked_step.observation[agents[0]], types.OLT)
        if olt_type:
            concat_id = ConcatAgentIdToObservation(stacked_env)

            action_spec = concat_id.action_spec()
            discrete = isinstance(action_spec[agents[0]], DiscreteArray)
            if discrete:
                concat_id_action = ConcatPrevActionToObservation(concat_id)

                concat_id_step = concat_id.reset()
                concat_id_action_step = concat_id_action.reset()

                if type(concat_id_step) == tuple:
                    concat_id_step, _ = concat_id_step

                if type(concat_id_action_step) == tuple:
                    concat_id_action_step, _ = concat_id_action_step

                for agent in agents:
                    stacked_shape = stacked_step.observation[agent].observation.shape[0]
                    concat_id_shape = concat_id_step.observation[
                        agent
                    ].observation.shape[0]
                    concat_id_action_shape = concat_id_action_step.observation[
                        agent
                    ].observation.shape[0]

                    assert (stacked_shape < concat_id_shape) and (
                        concat_id_shape < concat_id_action_shape
                    )
