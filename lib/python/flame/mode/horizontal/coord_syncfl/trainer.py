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
from abc import ABCMeta

from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType, TrainerState
from flame.common.util import (
    MLFramework,
    delta_weights_pytorch,
    delta_weights_tensorflow,
    weights_to_device,
    weights_to_model_device,
    mlflow_runname,
    get_ml_framework_in_use,
)
from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.syncfl.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizers import optimizer_provider
from flame.registries import registry_provider
from flame.datasamplers import datasampler_provider

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
TAG_COORDINATE = "coordinate"
WAIT_TIME_FOR_MID_AGG = 60

class Trainer(BaseTrainer, metaclass=ABCMeta):
    def internal_init(self) -> None:
        """Initialize internal state for role."""
        self.cm = ChannelManager()
        self.cm(self.config)
        # self.cm.join_all()
        self.cm.join_cp()

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config.registry.uri, self.config.job.job_id)

        self.registry_client.setup_run(mlflow_runname(self.config))
        self.metrics = dict()

        # needed for trainer-side optimization algorithms such as fedprox
        temp_opt = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )
        self.regularizer = temp_opt.regularizer

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).trainer_data_sampler

        self._round = 1
        self._work_done = False

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

    def _get_aggregator(self):
        logger.debug("calling _get_aggregator")
        channel = self.cm.get_by_tag(TAG_COORDINATE)
        if not channel:
            logger.debug(f"channel not found with tag {TAG_COORDINATE}")
            return

        channel.await_join()

        end = channel.one_end()
        msg, _ = channel.recv(end)

        self._work_done = msg[MessageType.EOT]
        if self._work_done:
            logger.debug("work is done")
            return

        if MessageType.COORDINATED_ENDS in msg:
            self.aggregator_id = msg[MessageType.COORDINATED_ENDS]

        if MessageType.MID_AGGS_URL in msg:
            mid_agg_url = msg[MessageType.MID_AGGS_URL]
            num_ends = len(msg[MessageType.COORDINATED_ENDS])
            logger.info(f"trainer join {num_ends} end(s): {mid_agg_url}")
            self.cm.join_dp({self.aggregator_id: mid_agg_url}, num_ends)

        logger.debug("exited _get_aggregator")

    # def _release_aggregator(self):
    #     logger.debug("calling _release_aggregator")
    #     fetch_channel = self.cm.get_by_tag(TAG_FETCH)
    #     upload_channel = self.cm.get_by_tag(TAG_UPLOAD)

        # logger.info(f"Fetch channel: {fetch_channel.name()}")
        # logger.info(f"Upload channel: {upload_channel.name()}")
        # logger.info(f"Clean up channel: {upload_channel.name()}")
        # self.cm.leave(upload_channel.name())
        # upload_channel.cleanup()

        # for end_id, end in upload_channel._ends.items():
        #     del upload_channel._backend._endpoints[end_id]
        #     del end

        # del self.cm.get(upload_channel.name())
        # del upload_channel

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        # this call waits for all peers to join this channel
        self.mid_agg_no_show = channel.await_ends_join(WAIT_TIME_FOR_MID_AGG)
        if self.mid_agg_no_show:
            logger.debug("channel await join timeouted")
            logger.info(f"trainer expects {channel.num_ends} ends, but only {len(channel._ends)} have joined")

        msg, _ = channel.recv(self.aggregator_id)

        if MessageType.WEIGHTS in msg:
            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        self.regularizer.save_state(TrainerState.PRE_TRAIN, glob_model=self.model)
        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        self._update_weights()
        self.regularizer.save_state(TrainerState.POST_TRAIN, loc_model=self.model)

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        # send delta_weights to regularizer
        self.regularizer.update()

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
        }
        channel.send(self.aggregator_id, msg)
        logger.debug("sending weights done")

    def compose(self) -> None:
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("", self.internal_init)

            task_load_data = Tasklet("", self.load_data)

            task_init = Tasklet("", self.initialize)

            task_get_aggregator = Tasklet("", self._get_aggregator)

            task_get = Tasklet("", self.get, TAG_FETCH)

            task_train = Tasklet("", self.train)

            task_eval = Tasklet("", self.evaluate)

            task_put = Tasklet("", self.put, TAG_UPLOAD)

            # task_release = Tasklet("", self._release_aggregator)

            task_save_metrics = Tasklet("", self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_load_data
                >> task_init
                >> loop(
                    task_get_aggregator
                    >> task_get
                    >> task_train
                    >> task_eval
                    >> task_put
                    # >> task_release
                    >> task_save_metrics
                )
            )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_COORDINATE]
