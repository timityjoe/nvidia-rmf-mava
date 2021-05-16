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

import dataclasses
import time
from typing import Any, Dict, Iterator, List, Optional, Type

import reverb
import sonnet as snt
from acme import datasets
from acme.tf import variable_utils
from acme.utils import counting

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf.qmix import execution, training
from mava.utils import training_utils as train_utils
from mava.wrappers import DetailedTrainerStatistics

# TODO Clean up documentation


class DetailedTrainerStatisticsWithEpsilon(DetailedTrainerStatistics):
    def __init__(
        self,
        trainer: training.QMIXTrainer,
        metrics: List[str] = ["q_value_loss"],
        summary_stats: List = ["mean", "max", "min", "var", "std"],
    ) -> None:
        super().__init__(trainer, metrics, summary_stats)

    def get_epsilon(self) -> float:
        return self._trainer.get_epsilon()  # type: ignore

    def step(self) -> None:
        # Run the learning step.
        fetches = self._step()

        if self._require_loggers:
            self._create_loggers(list(fetches.keys()))
            self._require_loggers = False

        # compute statistics
        self._compute_statistics(fetches)

        # Compute elapsed time.
        # NOTE (Arnu): getting type issues with the timestamp
        # not sure why. Look into a fix for this.
        timestamp = time.time()
        if self._timestamp:  # type: ignore
            elapsed_time = timestamp - self._timestamp  # type: ignore
        else:
            elapsed_time = 0
        self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        train_utils.checkpoint_networks(self._system_checkpointer)

        fetches["epsilon"] = self.get_epsilon()
        self._trainer._decrement_epsilon()  # type: ignore

        if self._logger:
            self._logger.write(fetches)


@dataclasses.dataclass
class QMIXConfig:
    """Configuration options for the QMIX system.
    Args:
            environment_spec: description of the actions, observations, etc.
            discount: discount to use for TD updates.
            batch_size: batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_update_period: number of learner steps to perform before updating
              the target networks.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take from replay for every insert
              that is made.
            n_step: number of steps to squash into a single transition.
            sigma: standard deviation of zero-mean, Gaussian exploration noise.
            clipping: whether to clip gradients by global norm.
            replay_table_name: string indicating what name to give the replay table."""

    environment_spec: specs.MAEnvironmentSpec
    epsilon_min: float
    epsilon_decay: float
    shared_weights: bool
    target_update_period: int
    executor_variable_update_period: int
    clipping: bool
    min_replay_size: int
    max_replay_size: int
    samples_per_insert: Optional[float]
    prefetch_size: int
    batch_size: int
    n_step: int
    discount: float
    checkpoint: bool
    optimizer: snt.Optimizer
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    checkpoint_subpath: str = "~/mava/"


class QMIXBuilder:
    """Builder for QMIX which constructs individual components of the system."""

    """Defines an interface for defining the components of an MARL system.
      Implementations of this interface contain a complete specification of a
      concrete MARL system. An instance of this class can be used to build an
      MARL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: QMIXConfig,
        trainer_fn: Type[training.QMIXTrainer] = training.QMIXTrainer,
        executor_fn: Type[core.Executor] = execution.QMIXFeedForwardExecutor,
        exploration_scheduler_fn: Type[
            LinearExplorationScheduler
        ] = LinearExplorationScheduler,
    ) -> None:
        """Args:
        _config: Configuration options for the QMIX system.

        self._config = config
        self._trainer_fn = trainer_fn
        """

        self._config = config
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn
        self._exploration_scheduler_fn = exploration_scheduler_fn

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer,
            )

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.ParallelNStepTransitionAdder.signature(
                environment_spec
            ),
        )

        return [replay_table]

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the system.
        Args:
            replay_client: Reverb Client which points to the replay server."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
        )
        return iter(dataset)

    def make_adder(
        self, replay_client: reverb.Client
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server."""
        return reverb_adders.ParallelNStepTransitionAdder(
            client=replay_client,
            priority_fns=None,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

    def make_executor(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, Any],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
        trainer: Optional[training.QMIXTrainer] = None,
    ) -> core.Executor:
        """Create an executor instance.
        Args:
            q_networks: A struct of instance of all
                the different system q networks,
                this should be a callable which takes as input observations
                and returns actions.
            adder: How data is recorded (e.g. added to replay).
            variable_source: collection of (nested) numpy arrays. Contains
                source variables as defined in mava.core.
        """

        shared_weights = self._config.shared_weights

        variable_client = None
        if variable_source:
            agent_keys = self._agent_types if shared_weights else self._agents

            # Create policy variables
            variables = {}
            for agent in agent_keys:
                variables[agent] = q_networks[agent].variables

            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"q_network": variables},
                update_period=self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the executor which coordinates the actors.
        return self._executor_fn(
            q_networks=q_networks,
            action_selectors=action_selectors,
            shared_weights=shared_weights,
            variable_client=variable_client,
            adder=adder,
            trainer=trainer,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
    ) -> core.Trainer:
        """Creates an instance of the trainer.
        Args:
          networks: struct describing the networks needed by the trainer; this can
            be specific to the trainer in question.
          dataset: iterator over samples from replay.
          counter: a Counter which allows for recording of counts (trainer steps,
            executor steps, etc.) distributed throughout the system.
          logger: Logger object for logging metadata.
          checkpoint: bool controlling whether the trainer checkpoints itself.
        """
        q_networks = networks["values"]
        target_q_networks = networks["target_values"]
        mixing_network = networks["mixing"]
        target_mixing_network = networks["target_mixing"]

        agents = self._config.environment_spec.get_agent_ids()
        agent_types = self._config.environment_spec.get_agent_types()

        # Make epsilon scheduler
        exploration_scheduler = self._exploration_scheduler_fn(
            epsilon_min=self._config.epsilon_min,
            epsilon_decay=self._config.epsilon_decay,
        )
        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(
            agents=agents,
            agent_types=agent_types,
            discount=self._config.discount,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            mixing_network=mixing_network,
            target_mixing_network=target_mixing_network,
            shared_weights=self._config.shared_weights,
            optimizer=self._config.optimizer,
            target_update_period=self._config.target_update_period,
            clipping=self._config.clipping,
            exploration_scheduler=exploration_scheduler,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=self._config.checkpoint,
            checkpoint_subpath=self._config.checkpoint_subpath,
        )

        trainer = DetailedTrainerStatisticsWithEpsilon(trainer)  # type:ignore

        return trainer