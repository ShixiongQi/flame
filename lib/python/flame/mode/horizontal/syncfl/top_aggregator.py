# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""horizontal FL top level aggregator."""

import os, json
import logging
import time
from copy import deepcopy

from diskcache import Cache
from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    mlflow_runname,
    valid_frameworks,
    weights_to_device,
    weights_to_model_device,
)
from flame.config import Config
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame.optimizers import optimizer_provider
from flame.plugin import PluginManager, PluginType
from flame.registries import registry_provider

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"


class TopAggregator(Role, metaclass=ABCMeta):
    """Top level Aggregator implements an ML aggregation role."""

    @abstract_attribute
    def config(self) -> Config:
        """Abstract attribute for config object."""

    @abstract_attribute
    def model(self):
        """Abstract attribute for model object."""

    @abstract_attribute
    def dataset(self):
        """
        Abstract attribute for datset.

        dataset's type is Dataset (in flame/dataset.py).
        """

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        # global variable for plugin manager
        self.plugin_manager = PluginManager()

        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config.registry.uri, self.config.job.job_id)

        base_model = self.config.base_model
        if base_model and base_model.name != "" and base_model.version > 0:
            self.model = self.registry_client.load_model(
                base_model.name, base_model.version
            )

        self.registry_client.setup_run(mlflow_runname(self.config))
        self.metrics = dict()

        # disk cache is used for saving memory in case model is large
        self.cache = Cache()
        self.optimizer = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )

        self._round = 1
        self._rounds = 1
        self._rounds = self.config.hyperparameters.rounds
        self._work_done = False

        # Used to track unreceived ends in recv_fifo()
        self.registered_ends = None
        self.received_ends = []

        self.load_states() #NOTE: local states (round no., mdoel version, model name) from local path

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)

    def remaining_ends(self) -> list[str]:
        """Get a list of unreceived ends."""
        return list(set(self.registered_ends) - set(self.received_ends))

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0
        # receive local model parameters from trainers
        #TODO: Take a diff and get the the endpoints (trainers) that haven't uploaded their model updates. This is required for Sync-FL and Semi-sync FL
        # for end, msg in channel.recv_fifo(self.remaining_ends()):
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, _ = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            logger.debug(f"{end}'s parameters trained with {count} samples")

            # logger.info(f"Adding {end} to received list")
            # self.received_ends.append(end)

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                # save training result from trainer in a disk cache
                self.cache[end] = tres
                #TODO: Save cache to DB

        if channel.my_role() == "aggregator":
            logger.info("Aggregator creates a dummy client to keep itself warm")
            channel.run_dummy_client()
        
        # time.sleep(60) # Sleep longer than grace period

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights),
            self.cache,
            total=total,
            num_trainers=len(channel.ends()),
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights
        self.weights = global_weights

        # update model with global weights
        self._update_model()

        if channel.my_role() == "aggregator":
            logger.info("Aggregator terminates the dummy client to disable keep-warm")
            channel.terminate_dummy_client()

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_DISTRIBUTE:
            self.dist_tag = tag
            self._distribute_weights(tag)
        #TODO: Record the endpoints (trainers) for which the aggregator has distributed weights. This is required for Sync-FL and Semi-sync FL

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        # send out global model parameters to trainers
        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                },
            )
            # logger.info(f"Add registered endpoint: {end}")
            # self.registered_ends.append(end) # = channel.ends()

    def inform_end_of_training(self) -> None:
        """Inform all the trainers that the training is finished."""
        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        channel.broadcast({MessageType.EOT: self._work_done})
        logger.debug("done broadcasting end-of-training")

    def run_analysis(self):
        """Run analysis plugins and update results to metrics."""
        logger.debug("running analyzer plugins")

        plugins = self.plugin_manager.get_plugins(PluginType.ANALYZER)
        for plugin in plugins:
            # get callback function and call it
            func = plugin.callback()
            metrics = func(self.model, self.dataset)
            if not metrics:
                continue

            self.update_metrics(metrics)

    def save_metrics(self):
        """Save metrics in a model registry."""
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

    def increment_round(self):
        """Increment the round counter."""
        logger.debug(f"Incrementing current round: {self._round}")
        logger.debug(f"Total rounds: {self._rounds}")
        self._round += 1
        self.save_states() #NOTE: save states (ended round no., model version, model name) to local path
        self._work_done = self._round > self._rounds

        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        logger.debug(f"Incremented round to {self._round}")
        # set necessary properties to help channel decide how to select ends
        channel.set_property("round", self._round)

    def save_params(self):
        """Save hyperparamets in a model registry."""
        if self.config.hyperparameters:
            self.registry_client.save_params(self.config.hyperparameters)

    def save_model(self):
        """Save model in a model registry."""
        if self.model:
            logger.info(f"Top aggregator saves model aggregated in Round-{self._round} to registry.")
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"
            self.registry_client.save_model(model_name, self.model)
            self._version = f"{self._round}"

    def update_metrics(self, metrics: dict[str, float]):
        """Update metrics."""
        self.metrics = self.metrics | metrics

    def _update_model(self):
        if self.framework == MLFramework.PYTORCH:
            self.model.load_state_dict(self.weights)
        elif self.framework == MLFramework.TENSORFLOW:
            self.model.set_weights(self.weights)

    def _update_weights(self):
        if self.framework == MLFramework.PYTORCH:
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()

    def load_states(self):
        """ Load states from local path """

        meta_json = self._load_meta("/flame/work/meta.json")
        if meta_json is not None:
            logger.info(f"Top aggregator loads meta info.")
            self._round   = meta_json["training-round"]
            self._version = meta_json["model-version"]
            self._rounds = self._rounds + self._round #NOTE: This is a trick for debugging purpose.

        if self.model is None and self._round is not 1:
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"

            logger.info(f"Top aggregator loads models from MLflow - Model name: {model_name} - version: {self._version}")
            self.model = self.registry_client.load_model(model_name, self._version)

    def _load_meta(self, meta_path: str) -> dict:
        """ Load metadata from /flame/work/meta.json """
        logger.info(f"Top aggregator loads metadata /flame/work/meta.json")

        path = "/flame/work/" #TODO: move to util.py
        dir_list = os.listdir(path)

        logger.info(f"Files and directories in {path}: {dir_list}")
        if "config" in dir_list:
            logger.info(f"config is under {path}")

        if "meta.json" not in dir_list:
            logger.info(f"meta.json is not under {path}")
            return None
        else:
            with open(meta_path) as f:
                return json.loads(f.read()) # return meta_json

    def save_states(self):
        """ Save states to MLflow """
        if self.model:
            logger.info(f"Top aggregator saves metadata of Round-{self._round} to local path.\n\n")
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"
            self._save_meta(self._round, self._version, model_name)

    def _save_meta(self, _round: int, _version: str, _model_name: str) -> None:
        """Save metadata to /flame/work/meta.json """
        meta_path = "/flame/work/meta.json"

        dictionary = {
            "training-round": _round,
            "model-version": _version,
            "model-name": _model_name
            }

        json_object = json.dumps(dictionary, indent = 4)

        with open(meta_path, "w") as outfile:
            outfile.write(json_object)

        logger.info(f"metadata has been updated to {meta_path}.")

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_init = Tasklet(self.initialize)

            task_load_data = Tasklet(self.load_data)

            task_put = Tasklet(self.put, TAG_DISTRIBUTE)

            task_get = Tasklet(self.get, TAG_AGGREGATE)

            task_train = Tasklet(self.train)

            task_eval = Tasklet(self.evaluate)

            task_analysis = Tasklet(self.run_analysis)

            task_save_metrics = Tasklet(self.save_metrics)

            task_increment_round = Tasklet(self.increment_round)

            task_end_of_training = Tasklet(self.inform_end_of_training)

            task_save_params = Tasklet(self.save_params)

            task_save_model = Tasklet(self.save_model)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_put
                >> task_get
                >> task_train
                >> task_eval
                >> task_analysis
                >> task_save_metrics
                >> task_save_model
                >> task_increment_round
            )
            >> task_end_of_training
            >> task_save_params
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
