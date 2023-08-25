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
"""MNIST horizontal FL trainer for PyTorch.

The example below is implemented based on the following example from pytorch:
https://github.com/pytorch/examples/blob/master/mnist/main.py.
"""

import logging
import random
import time

from flame.mode.composer import Composer
from flame.mode.tasklet import Loop, Tasklet
TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from flame.config import Config
from flame.mode.horizontal.coord_syncfl.trainer import Trainer
from torchvision import datasets, transforms
import torchvision.models as tormodels

from flame.fedscale_utils.femnist import FEMNIST
from flame.fedscale_utils.utils_data import get_data_transform
from torch.autograd import Variable

logger = logging.getLogger(__name__)

def override(method):
    return method

class PyTorchFemnistTrainer(Trainer):
    """PyTorch Femnist Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None

        self.device = None
        self.train_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16

        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/femnist/"
        self.loss_squared = 0
        self.completed_steps = 0
        self.epoch_train_loss = 1e-4
        self.loss_decay = 0.2
        self.local_steps = 30
        self.meta_dir = "/mydata/flame_dataset/femnist/"
        self.partition_id = 1

        # Latency metrics
        self.load_data_delay = 0
        self.local_training_delay = 0

        # Output logs
        log_file = f"/mydata/image_cls-trainer-{config.task_id}.log"
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = tormodels.__dict__["resnet152"](num_classes=62).to(self.device)

    def load_data(self) -> None:
        """Load data."""
        self.LOAD_START_T = time.time()

        # Generate a random parition ID
        self.partition_id = random.randint(1, 3399)

        train_transform, test_transform = get_data_transform('mnist')
        train_dataset = FEMNIST(self.data_dir, self.meta_dir, self.partition_id,
                                dataset='train', transform=train_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True, timeout=0,
                num_workers=0, drop_last=True)

        self.LOAD_END_T = time.time()
        self.load_data_delay = self.LOAD_END_T - self.LOAD_START_T

    def train(self) -> None:
        """Train a model."""
        self.TRAIN_START_T = time.time()

        self.optimizer = optim.Adadelta(self.model.parameters())

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

        self.TRAIN_END_T = time.time()
        self.local_training_delay = self.TRAIN_END_T - self.TRAIN_START_T

    def _train_epoch(self, epoch):
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=self.device)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = criterion(output, target)

            loss_list = loss.tolist()
            loss = loss.mean()

            temp_loss = sum(loss_list) / float(len(loss_list))
            self.loss_squared = sum([l ** 2 for l in loss_list]) / float(len(loss_list))

            # only measure the loss of the first epoch
            if self.completed_steps < len(self.train_loader):
                if self.epoch_train_loss == 1e-4:
                    self.epoch_train_loss = temp_loss
                else:
                    self.epoch_train_loss = (1. - self.loss_decay) * self.epoch_train_loss + self.loss_decay * temp_loss

            # Define the backward loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.completed_steps += 1

            if self.completed_steps == self.local_steps:
                break

        # logger.info(f"loss: {loss.item():.6f} \t moving_loss: {self.epoch_train_loss}")

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass

    @override
    def save_metrics(self):
        """Save metrics in a model registry."""

        self.metrics = self.metrics | self.mc.get()
        self.mc.clear()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

        logger.info(f"Wall-clock time: {time.time()} || "
                    f"Training delay (s): {self.local_training_delay:.4f} || "
                    f"Loading data delay (s): {self.load_data_delay:.4f} || "
                    f"Fetch task delay: {self.fetch_delay:.4f} || "
                    f"MSG delay: {self.msg_delay:.4f} || "
                    f"Send task delay: {self.send_delay:.4f}")

        logger.info(f"Trainer ({self.config.task_id}) Timestamps: "
                    f"TRAIN, TRAIN_START_T: {self.TRAIN_START_T}, TRAIN_END_T: {self.TRAIN_END_T} || "
                    f"LOAD, LOAD_START_T: {self.LOAD_START_T}, LOAD_END_T: {self.LOAD_END_T} || "
                    f"FETCH, FETCH_START_T: {self.FETCH_START_T}, FETCH_END_T: {self.FETCH_END_T} || "
                    f"SEND, SEND_START_T: {self.SEND_START_T}, SEND_END_T: {self.SEND_END_T} || "
                    f"MSG_MTr, MSG_MTr_START_T: {self.MSG_MTr_START_T}, MSG_MTr_END_T: {self.MSG_MTr_END_T}")

    @override
    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)
            task_get.set_continue_fn(cont_fn=lambda: not self.fetch_success)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            task_get_aggregator = Tasklet("get_aggregator", self._get_aggregator)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_init
                >> loop(
                    task_load_data >> task_get_aggregator >> task_get >> task_train >> task_eval >> task_put >> task_save_metrics
                )
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchFemnistTrainer(config)
    t.compose()
    t.run()
