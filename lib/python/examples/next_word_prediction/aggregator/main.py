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
"""Next Word Prediction horizontal FL aggregator for PyTorch.

Model and data loader are imported from FedScale
"""

import logging
import random

from flame.mode.composer import Composer
from flame.mode.tasklet import Loop, Tasklet

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"

import torch
import torch.nn as nn
import torch.nn.functional as F
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator

from torch.autograd import Variable

# libs from FedScale
import os
import csv
from torch.nn.utils.rnn import pad_sequence
from transformers import (AlbertTokenizer, AutoConfig, AutoModelWithLMHead)

from flame.fedscale_utils.nlp import load_and_cache_examples, mask_tokens
from flame.fedscale_utils.divide_data import DataPartitioner

logger = logging.getLogger(__name__)

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

def load_partition(partition_file_path):
    with open(partition_file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        data_indexes = [list(map(int, row)) for row in reader][0]

    return data_indexes

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res

def test_pytorch_model(model, test_data, device='cpu'):

    test_loss = 0
    correct = 0
    top_5 = 0

    test_len = 0
    perplexity_loss = 0.
    mlm_probability = 0.15

    model = model.to(device=device) # load by pickle
    model.eval()

    with torch.no_grad():
        for data, target in test_data:
            try:
                # if parser.args.mlm else (data, data)
                data, target = mask_tokens(data, tokenizer, mlm_probability, device=device)
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                # if parser.args.mlm else model(data, labels=target)
                outputs = model(data, labels=target)

                loss = outputs[0]
                test_loss += loss.data.item()
                perplexity_loss += loss.data.item()

                acc = accuracy(
                    outputs[1].reshape(-1, outputs[1].shape[2]), target.reshape(-1), topk=(1, 5))

                correct += acc[0].item()
                top_5 += acc[1].item()

            except Exception as ex:
                logger.info(f"Testing of failed as {ex}")
                break
            test_len += len(target)

    test_len = max(test_len, 1)
    # loss function averages over batch size
    test_loss /= len(test_data)
    perplexity_loss /= len(test_data)

    sum_loss = test_loss * test_len

    # in NLP, we care about the perplexity of the model
    acc = round(correct / test_len, 4)
    acc_5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    # logger.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
    #              .format(rank, test_loss, correct, len(test_data.dataset), acc, acc_5))

    testRes = {'top_1': correct, 'top_5': top_5,
               'test_loss': sum_loss, 'test_len': test_len}

    return test_loss, acc, acc_5, testRes

class PyTorchNextWordPredictionAggregator(TopAggregator):
    """PyTorch Next Word Prediction Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None
        self.batch_size = 16

        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/reddit/"
        self.meta_dir = "/mydata/flame_dataset/reddit/"
        self.collate_fn = None

        self.overwrite_cache = False
        self.block_size = 64
        self.testing_sets = None

    def initialize(self):
        """Initialize role."""
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = init_model(self.data_dir, model="albert-base-v2").to(device=self.device)

        print(f"Loading (Whole) Testing Reddit Dataset from cache")
        test_dataset = init_dataset(self.data_dir, "", overwrite_cache=self.overwrite_cache, block_size=self.block_size, dataset="test")
        self.testing_sets = DataPartitioner(data=test_dataset)

    def load_data(self) -> None:
        """Load a test dataset."""

        # Generate a random parition ID
        self.partition_id = random.randint(0, 2453)

        partition_file_path = os.path.join(self.meta_dir, "test", 'client-'+str(self.partition_id)+'-'+'test.csv')
        testing_data_indexes = load_partition(partition_file_path)

        partitioned_test_dataset = self.testing_sets.use(testing_data_indexes, True)

        # self.collate_fn = collate
        self.test_loader = torch.utils.data.DataLoader(partitioned_test_dataset, 
                                batch_size=self.batch_size, shuffle=True, 
                                pin_memory=True, timeout=0, 
                                num_workers=0, drop_last=False, 
                                collate_fn=collate)

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        test_loss, test_accuray, acc_5, testRes = test_pytorch_model(self.model, self.test_loader, device=self.device)

        logger.info(f"Test loss: {test_loss}")
        logger.info(f"Test accuracy: {test_accuray}")

        # update metrics after each evaluation so that the metrics can be
        # logged in a model registry.
        self.update_metrics({
            'test-loss': test_loss,
            'test-accuracy': test_accuray
        })
        # pass

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
            >> task_init
            >> loop(
                task_load_data
                >> task_put
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchNextWordPredictionAggregator(config)
    a.compose()
    a.run()
