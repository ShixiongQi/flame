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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from flame.config import Config
from flame.mode.horizontal.trainer import Trainer
from torchvision import datasets, transforms
import torchvision.models as tormodels

from flame.fedscale_utils.femnist import FEMNIST
from flame.fedscale_utils.utils_data import get_data_transform

logger = logging.getLogger(__name__)


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

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = tormodels.__dict__["resnet18"](num_classes=62).to(self.device)

    def load_data(self) -> None:
        """Load data."""

        train_transform, test_transform = get_data_transform('mnist')
        train_dataset = FEMNIST(self.data_dir, dataset='train', transform=train_transform)

        # indices = torch.arange(2000)
        # train_dataset = data_utils.Subset(train_dataset, indices)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                shuffle=True, pin_memory=True, timeout=0,
                num_workers=0, drop_last=True)

    def train(self) -> None:
        """Train a model."""
        self.optimizer = optim.Adadelta(self.model.parameters())

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

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

        logger.info(f"loss: {loss.item():.6f} \t moving_loss: {self.epoch_train_loss}")

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchFemnistTrainer(config)
    t.compose()
    t.run()
