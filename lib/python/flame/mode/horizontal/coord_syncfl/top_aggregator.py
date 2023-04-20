# Copyright 2023 Cisco Systems, Inc. and its affiliates
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

import logging
from datetime import datetime

from diskcache import Cache
from flame.channel_manager import ChannelManager
from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.syncfl.top_aggregator import (
    TopAggregator as BaseTopAggregator,
)
from flame.common.constants import DeviceType
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    mlflow_runname,
    weights_to_device,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.plugin import PluginManager
from flame.optimizers import optimizer_provider
from flame.registries import registry_provider
from flame.datasamplers import datasampler_provider

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TAG_COORDINATE = "coordinate"  # coordinate with the coordinator
PROP_ROUND_START_TIME = "round_start_time"

WAIT_TIME_FOR_MID_AGG = 60

class TopAggregator(BaseTopAggregator):
    def internal_init(self) -> None:
        """Initialize internal state for role."""
        # global variable for plugin manager
        self.plugin_manager = PluginManager()

        self.cm = ChannelManager()
        self.cm(self.config)
        # self.cm.join_all()
        self.cm.join_cp()

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

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).aggregator_data_sampler

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

    def get_channel(self, tag: str):
        """Return channel of a given tag when it is ready to use."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        channel.await_join()

        return channel

    def get_coordinated_ends(self):
        """Receive the ends of middle aggregators."""
        logger.debug("calling get_coordinate_ends()")
        channel = self.get_channel(TAG_COORDINATE)

        end = channel.one_end()
        msg, _ = channel.recv(end)
        logger.debug(f"received message = {msg} from {end}")

        self._work_done = msg[MessageType.EOT]
        if self._work_done:
            logger.debug("work is done")
            return

        dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
        while dist_channel == None:
            num_ends = len(msg[MessageType.COORDINATED_ENDS])
            logger.info(f"top aggregator joining dataplane channel .... {num_ends} end(s)")
            self.cm.join_dp(msg[MessageType.MID_AGGS_URL], num_ends)

            dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
        # overide distribute channel's ends method
        dist_channel.ends = lambda: msg[MessageType.COORDINATED_ENDS]

        logger.debug("exited get_coordinate_ends()")

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for all middle aggregator to join this channel
        self.mid_agg_no_show = channel.await_ends_join(WAIT_TIME_FOR_MID_AGG)
        if self.mid_agg_no_show:
            logger.info("channel await join timeouted")
            logger.info(f"top-aggregator expects {channel.num_ends} ends, but only {len(channel._ends)} have joined")

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
                },
            )
            # logger.info(f"Add registered endpoint: {end}")
            # self.registered_ends.append(end) # = channel.ends()
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("", self.internal_init)

            task_init = Tasklet("", self.initialize)

            task_load_data = Tasklet("", self.load_data)

            task_get_coord_ends = Tasklet("", self.get_coordinated_ends)

            task_put = Tasklet("", self.put, TAG_DISTRIBUTE)

            task_get = Tasklet("", self.get, TAG_AGGREGATE)

            task_train = Tasklet("", self.train)

            task_eval = Tasklet("", self.evaluate)

            task_analysis = Tasklet("", self.run_analysis)

            task_save_metrics = Tasklet("", self.save_metrics)

            task_increment_round = Tasklet("", self.increment_round)

            task_save_params = Tasklet("", self.save_params)

            task_save_model = Tasklet("", self.save_model)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_get_coord_ends
                >> task_put
                >> task_get
                >> task_train
                >> task_eval
                >> task_analysis
                >> task_save_metrics
                >> task_save_model
                >> task_increment_round
            )
            >> task_save_params
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_COORDINATE]
