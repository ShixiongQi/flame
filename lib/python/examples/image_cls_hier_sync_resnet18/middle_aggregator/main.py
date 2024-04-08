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
"""HIRE_FEMNIST coordinated horizontal hierarchical FL middle level aggregator for PyTorch."""

import logging
import time

from flame.config import Config
from flame.mode.horizontal.coord_syncfl.middle_aggregator import MiddleAggregator

# the following needs to be imported to let the flame know
# this aggregator works on tensorflow model
# from tensorflow import keras
import torch
import torchvision.models as tormodels

logger = logging.getLogger(__name__)


class TorchFemnistMiddleAggregator(MiddleAggregator):
    """Torch Femnist Middle Level Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config

        # Output logs
        log_file = f"/mydata/image_cls-mid_aggregator-{config.task_id}.log"
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)

    def initialize(self):
        """Initialize role."""
        pass

    def load_data(self) -> None:
        """Load a test dataset."""
        pass

    def train(self) -> None:
        """Train a model."""
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        pass

    def update_round(self):
        """Update the round counter."""
        logger.debug(f"Update current round: {self._round}")

        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        # set necessary properties to help channel decide how to select ends
        channel.set_property("round", self._round)

        logger.info(f"Wall-clock time: {time.time()} || "
                    f"Round: {self._round} || "
                    f"#TRs: {self.N_ENDS} || "
                    f"Agg delay (s): {self.agg_delay:.4f} || "
                    f"Fetch task delay: {self.fetch_delay:.4f} || "
                    f"DIST task delay: {self.dist_delay:.4f} || "
                    f"RECV task delay: {self.recv_delay:.4f} || "
                    f"SEND task delay: {self.send_delay:.4f} || "
                    f"Queueing delay: {self.queue_delay:.4f} || "
                    f"MSG (from top) delay: {self.msg_from_top_delay:.4f} || "
                    f"MSG (from trainer) Ave. delay: {sum(self.msg_from_tr_delays)/len(self.msg_from_tr_delays):.4f} || "
                    f"Total cache delay: {sum(self.cache_delays):.4f} || "
                    f"Ave. cache delay: {sum(self.cache_delays)/len(self.cache_delays):.4f}")

        logger.info(f"Mid Agg ({self.config.task_id}) Timestamps: "
                    f"CACHE, CACHE_START_T: {self.CACHE_START_T}, CACHE_END_T: {self.CACHE_END_T} || "
                    f"AGG, AGG_START_T: {self.AGG_START_T}, AGG_END_T: {self.AGG_END_T} || "
                    f"FETCH, FETCH_START_T: {self.FETCH_START_T}, FETCH_END_T: {self.FETCH_END_T} || "
                    f"SEND, SEND_START_T: {self.SEND_START_T}, SEND_END_T: {self.SEND_END_T} || "
                    f"DIST, DIST_START_T: {self.DIST_START_T}, DIST_END_T: {self.DIST_END_T} || "
                    f"RECV, RECV_START_T: {self.RECV_START_T}, RECV_END_T: {self.RECV_END_T} || "
                    f"MSG_ToM, MSG_ToM_START_T: {self.MSG_ToM_START_T}, MSG_ToM_END_T: {self.MSG_ToM_END_T} || "
                    f"MSG_TrM, MSG_TrM_START_T: {self.MSG_TrM_START_T}, MSG_TrM_END_T: {self.MSG_TrM_END_T}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = TorchFemnistMiddleAggregator(config)
    a.compose()
    a.run()
