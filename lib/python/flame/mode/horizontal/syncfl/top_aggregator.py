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

import tempfile, os
import logging
import time
import psutil
import torch
from copy import deepcopy
from datetime import datetime

from diskcache import Cache
from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
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
from flame.datasamplers import datasampler_provider

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"
PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_END_TIME = "round_end_time"

ENABLE_NOISE = False
num_duplication = 1
custom_temp_dir = "/mydata/tmp"

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
        self.registry_client(self.config)

        base_model = self.config.base_model
        if base_model and base_model.name != "" and base_model.version > 0:
            self.model = self.registry_client.load_model(
                base_model.name, base_model.version
            )

        self.registry_client.setup_run()
        self.metrics = dict()

        # disk cache is used for saving memory in case model is large
        # automatic eviction of disk cache is disabled with cull_limit 0
        if not os.path.exists(custom_temp_dir):
            os.makedirs(custom_temp_dir)
        temp_dir = tempfile.TemporaryDirectory(dir=custom_temp_dir)
        self.cache = Cache(directory=temp_dir.name)
        self.cache.reset("size_limit", 1e15)
        self.cache.reset("cull_limit", 0)

        self.optimizer = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).aggregator_data_sampler

        self._round = 1
        self._rounds = 1
        self._rounds = self.config.hyperparameters.rounds
        self._work_done = False

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        self.cpu_time = None
        self.utilization = None

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)

    def audit_weight_size(self, weights):
        total_size = 0
        for k, w in weights.items():
            total_elements = w.numel()
            element_size = w.element_size()
            total_size += total_elements * element_size
        # print(f"total weights size: {total_size / 1024 / 1024} MB")

    def add_gaussian_noise_to_weight(self, weights):
        noise_std = 0.1  # Standard deviation of the Gaussian noise
        noise_mean = 0
        for k, tensor in weights.items():
            # Generate Gaussian noise with the same shape as the tensor
            noise = torch.randn(tensor.shape) * noise_std + noise_mean

            # Add the noise to the tensor
            noisy_tensor = tensor + noise

            weights[k] = noisy_tensor

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        self.RECV_START_T = time.time()

        total = 0
        self.N_ENDS = len(channel.ends())
        self.msg_from_mid_delays = []
        self.cache_delays = []
        RECV_FIRST_T = time.time() # Init. time to receive first update
        RECV_LAST_T = time.time() # Init. time to receive last update
        start_cpu_time = psutil.cpu_times()
        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            if total == 0:
                RECV_FIRST_T = time.time()
            else:
                RECV_LAST_T = time.time()

            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
                self.audit_weight_size(weights)

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            if MessageType.DATASAMPLER_METADATA in msg:
                self.datasampler.handle_metadata_from_trainer(
                    msg[MessageType.DATASAMPLER_METADATA],
                    end,
                    channel,
                )

            if MessageType.SEND_TIMESTAMP in msg:
                self.MSG_SENT_T = msg[MessageType.SEND_TIMESTAMP]

            logger.debug(f"{end}'s parameters trained with {count} samples")

            self.msg_from_mid_delays.append(time.time() - self.MSG_SENT_T)

            self.CACHE_START_T = time.time()
            if ENABLE_NOISE:
                start_duplication_cpu_time = psutil.cpu_times()

                for i in range(0, num_duplication):
                    tmp_weights = deepcopy(weights)
                    self.add_gaussian_noise_to_weight(tmp_weights)
                    if tmp_weights is not None and count > 0:
                        total += count
                        tres = TrainResult(tmp_weights, count)
                        # save training result from trainer in a disk cache
                        self.cache[str(i)] = tres

                end_duplication_cpu_time = psutil.cpu_times()
            else:
                if weights is not None and count > 0:
                    total += count
                    tres = TrainResult(weights, count)
                    # save training result from trainer in a disk cache
                    self.cache[end] = tres
            self.CACHE_END_T = time.time()
            self.cache_delays.append(self.CACHE_END_T - self.CACHE_START_T)

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        self.RECV_END_T = time.time()
        self.recv_delay = self.RECV_END_T - self.RECV_START_T
        self.queue_delay = RECV_LAST_T - RECV_FIRST_T

        self.AGG_START_T = time.time()
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

        self.AGG_END_T = time.time()
        self.agg_delay = self.AGG_END_T - self.AGG_START_T

        end_cpu_time = psutil.cpu_times() # process.cpu_times()

        if ENABLE_NOISE:
            self.cpu_time = sum(end_cpu_time) - sum(start_cpu_time) - (sum(end_duplication_cpu_time) - sum(start_duplication_cpu_time))
            idle_time_diff = end_cpu_time.idle - start_cpu_time.idle - (end_duplication_cpu_time.idle - start_duplication_cpu_time.idle)
        else:
            self.cpu_time = sum(end_cpu_time) - sum(start_cpu_time)
            idle_time_diff = end_cpu_time.idle - start_cpu_time.idle

        self.utilization = 100.0 * (1.0 - idle_time_diff / self.cpu_time) * psutil.cpu_count(logical=True)
        # logger.info(f"CPU time: {self.cpu_time} || CPU utilization: {self.utilization}")

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_DISTRIBUTE:
            self.dist_tag = tag
            self._distribute_weights(tag)

    def _distribute_weights(self, tag: str) -> None:
        self.DIST_START_T = time.time()

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        selected_ends = channel.ends()
        datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        # send out global model parameters to trainers
        for end in selected_ends:
            logger.debug(f"sending weights to {end}")
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                    MessageType.SEND_TIMESTAMP: time.time(),
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )
        
        self.DIST_END_T = time.time()
        self.dist_delay = self.DIST_END_T - self.DIST_START_T

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
        # update metrics with metrics from metric collector
        self.metrics = self.metrics | self.mc.get()
        self.mc.clear()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

    def increment_round(self):
        """Increment the round counter."""
        logger.debug(f"Incrementing current round: {self._round}")
        logger.debug(f"Total rounds: {self._rounds}")
        self._round += 1
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
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"
            self.registry_client.save_model(model_name, self.model)

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

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init = Tasklet("initialize", self.initialize)

            task_load_data = Tasklet("load_data", self.load_data)

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_analysis = Tasklet("analysis", self.run_analysis)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            task_increment_round = Tasklet("inc_round", self.increment_round)

            task_end_of_training = Tasklet(
                "inform_end_of_training", self.inform_end_of_training
            )

            task_save_params = Tasklet("save_params", self.save_params)

            task_save_model = Tasklet("save_model", self.save_model)

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
                >> task_increment_round
            )
            >> task_end_of_training
            >> task_save_params
            >> task_save_model
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
