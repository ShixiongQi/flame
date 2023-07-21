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
"""Google Speech horizontal FL trainer for PyTorch.

Imported from https://github.com/SymbioticLab/FedScale/blob/master/fedscale/utils/models/specialized/resnet_speech.py
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

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch.utils.model_zoo as model_zoo
from torch import Tensor, nn

# libs from FedScale
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import (AlbertTokenizer, AutoConfig, AutoModelWithLMHead)

from flame.fedscale_utils.nlp import load_and_cache_examples, mask_tokens

logger = logging.getLogger(__name__)

# tokenizer = None
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

def collate(examples):
    if tokenizer._pad_token is None:
        return (pad_sequence(examples, batch_first=True), None)
    return (pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id), None)

def init_model(data_dir, model="albert-base-v2"):
    logger.info("Initializing the model ...")

    config = AutoConfig.from_pretrained(
        os.path.join(data_dir, model + '-config.json'))
    model = AutoModelWithLMHead.from_config(config)

    return model

def init_dataset(data_dir, model, overwrite_cache, block_size, dataset="train"):
    if dataset == "train":
        return load_and_cache_examples(data_dir, model, overwrite_cache, block_size, tokenizer, evaluate=False)
    elif dataset == "test":
        return load_and_cache_examples(data_dir, model, overwrite_cache, block_size, tokenizer, evaluate=True)

class PyTorchNextWordPredictionTrainer(Trainer):
    """PyTorch Next Word Prediction Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None

        self.device = None
        self.train_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16
        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/google_speech"
        self.collate_fn = None

        self.overwrite_cache = False
        self.block_size = 62

        self.loss_squared = 0
        self.completed_steps = 0
        self.epoch_train_loss = 1e-4
        self.loss_decay = 0.2
        self.local_steps = 30

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = init_model(self.data_dir, model="albert-base-v2").to(device=self.device)

    def load_data(self) -> None:
        """Load data."""

        # Loading Testing Reddit Dataset
        train_dataset = init_dataset(self.data_dir, "", overwrite_cache=self.overwrite_cache, block_size=self.block_size, dataset="train")
        self.collate_fn = collate

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=0)
    
    def train(self) -> None:
        """Train a model."""
        self.optimizer = optim.Adadelta(self.model.parameters())

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data, target = mask_tokens(data, tokenizer, conf, device=self.device)

            data = Variable(data).to(device=self.device)
            target = Variable(target).to(device=self.device)

            outputs = self.model(data, labels=target)
            loss = outputs[0]

            loss_list = [loss.item()]

            temp_loss = sum(loss_list) / float(len(loss_list))
            self.loss_squared = sum([l ** 2 for l in loss_list]) / float(len(loss_list))

            # only measure the loss of the first epoch
            if self.completed_steps < len(self.train_loader):
                if self.epoch_train_loss == 1e-4:
                    self.epoch_train_loss = temp_loss
                else:
                    self.epoch_train_loss = (1. - self.loss_decay) * self.epoch_train_loss + self.loss_decay * temp_loss

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

    t = PyTorchNextWordPredictionTrainer(config)
    t.compose()
    t.run()
