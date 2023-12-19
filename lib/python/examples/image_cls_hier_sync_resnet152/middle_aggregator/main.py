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
"""HIRE_MNIST horizontal hierarchical FL middle level aggregator for Keras."""

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


class TorchMnistMiddleAggregator(MiddleAggregator):
    """Torch Mnist Middle Level Aggregator."""

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = TorchMnistMiddleAggregator(config)
    a.compose()
    a.run()