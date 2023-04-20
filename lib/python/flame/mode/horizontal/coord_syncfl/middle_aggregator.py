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
from copy import deepcopy

from diskcache import Cache
from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.middle_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_FETCH,
    TAG_UPLOAD,
)
from flame.mode.horizontal.syncfl.middle_aggregator import (
    MiddleAggregator as BaseMiddleAggregator,
)
from flame.common.util import (
    MLFramework,
    delta_weights_pytorch,
    delta_weights_tensorflow,
    get_ml_framework_in_use,
)
from flame.channel_manager import ChannelManager
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame.optimizers import optimizer_provider
from flame.plugin import PluginManager
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
TAG_COORDINATE = "coordinate"
WAIT_TIME_FOR_TRAINER = 60

class MiddleAggregator(BaseMiddleAggregator):
    def internal_init(self) -> None:
        """Initialize internal state for role."""
        # global variable for plugin manager
        self.plugin_manager = PluginManager()

        self.cm = ChannelManager()
        self.cm(self.config)
        # self.cm.join_all()
        self.cm.join_cp()

        self.optimizer = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )

        self._round = 1
        self._work_done = False

        self.cache = Cache()
        self.dataset_size = 0

        # save distribute tag in an instance variable
        self.dist_tag = TAG_DISTRIBUTE

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        if self.framework == MLFramework.PYTORCH:
            self._delta_weights_fn = delta_weights_pytorch

        elif self.framework == MLFramework.TENSORFLOW:
            self._delta_weights_fn = delta_weights_tensorflow

    def get_channel(self, tag: str):
        """Return channel of a given tag when it is ready to use."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        channel.await_join()

        return channel

    def _get_trainers(self) -> None:
        logger.debug("getting trainers from coordinator")

        channel = self.get_channel(TAG_COORDINATE)

        end = channel.one_end()
        msg, _ = channel.recv(end)

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if self._work_done:
            logger.debug("work is done")
            return

        if MessageType.META_INFO_REQ not in msg:
            raise ValueError("no meta info req message")

        # here middle aggregator can send some useful meta info to coordinator
        # meta information can be overhead during the previous round
        #
        # TODO: implement the logic
        channel.send(end, {MessageType.META_INFO_RES: f"flame-aggregator-{self.cm._task_id}.flame.example.com"})
        logger.debug(f"sent meta info response: flame-aggregator-{self.cm._task_id}.flame.example.com")

        msg, _ = channel.recv(end)
        logger.debug(f"received msg = {msg} from {end}")

        if MessageType.COORDINATED_ENDS not in msg:
            raise ValueError("no coordinated ends message")

        self._work_done = msg[MessageType.EOT]
        if self._work_done:
            logger.debug("work is done")
            return

        dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
        while dist_channel == None:
            logger.info("No dist_channel found... joining data plane channel")
            num_ends = len(msg[MessageType.COORDINATED_ENDS])
            self.cm.join_dp(None, num_ends)

            dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
        # overide distribute channel's ends method
        dist_channel.ends = lambda: msg[MessageType.COORDINATED_ENDS]

        logger.debug("exited _get_trainers")

    # def _release_trainers(self):
    #     logger.debug("calling _release_trainers")
    #     dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
    #     agg_channel = self.cm.get_by_tag(TAG_AGGREGATE)

    #     # logger.info(f"Distribute channel: {dist_channel.name()}")
    #     # logger.info(f"Aggregate channel: {agg_channel.name()}")
    #     # self.cm.channel_leave(agg_channel.name())
    #     logger.info(f"Clean up channel: {agg_channel.name()}")
    #     agg_channel.cleanup()

    #     for end_id, end in agg_channel._ends.items():
    #         del agg_channel._backend._endpoints[end_id]
    #         del end

    #     # del self.cm.get(agg_channel.name())
    #     del agg_channel

    def _handle_no_trainer(self):
        channel = self.cm.get_by_tag(TAG_DISTRIBUTE)

        self.no_trainer = False
        if len(channel.ends()) == 0:
            logger.debug("no trainers found")
            self.no_trainer = True
            return

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for all peers to join this channel
        self.trainer_no_show = channel.await_ends_join(WAIT_TIME_FOR_TRAINER)
        if self.trainer_no_show:
            logger.debug("channel await join timeouted")
            logger.info(f"middle-aggregator expects {channel.num_ends} ends, but only {len(channel._ends)} have joined")
            # send dummy weights to unblock top aggregator
            self._send_dummy_weights(TAG_UPLOAD)
            return

        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(
                end, {MessageType.WEIGHTS: self.weights, MessageType.ROUND: self._round}
            )

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        # this call waits for all peers to join this channel
        self.trainer_no_show = channel.await_ends_join(WAIT_TIME_FOR_TRAINER)
        if self.trainer_no_show:
            logger.debug("channel await join timeouted")
            logger.info(f"middle-aggregator expects {channel.num_ends} ends, but only {len(channel._ends)} have joined")

        total = 0
        # receive local model parameters from trainers
        # TODO: do deepcopy speculatively
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, _ = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            if MessageType.WEIGHTS in msg:
                weights = msg[MessageType.WEIGHTS]

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                # save training result from trainer in a disk cache
                self.cache[end] = tres

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights), self.cache, total=total
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # save global weights before updating it
        self.prev_weights = self.weights

        # set global weights
        self.weights = global_weights
        self.dataset_size = total

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("", self.internal_init)

            task_init = Tasklet("", self.initialize)

            task_load_data = Tasklet("", self.load_data)

            task_get_trainers = Tasklet("", self._get_trainers)

            task_no_trainer = Tasklet("", self._handle_no_trainer)
            task_no_trainer.set_continue_fn(cont_fn=lambda: self.no_trainer)

            task_put_dist = Tasklet("", self.put, TAG_DISTRIBUTE)

            task_put_upload = Tasklet("", self.put, TAG_UPLOAD)

            task_get_aggr = Tasklet("", self.get, TAG_AGGREGATE)

            task_get_fetch = Tasklet("", self.get, TAG_FETCH)

            # task_release_trainers = Tasklet("", self._release_trainers)

            task_eval = Tasklet("", self.evaluate)

            task_update_round = Tasklet("", self.update_round)

            task_end_of_training = Tasklet("", self.inform_end_of_training)

        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_get_trainers
                >> task_no_trainer
                >> task_get_fetch
                >> task_put_dist
                >> task_get_aggr
                >> task_put_upload
                # >> task_release_trainers
                >> task_eval
                >> task_update_round
            )
            >> task_end_of_training
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_FETCH, TAG_UPLOAD, TAG_COORDINATE]
