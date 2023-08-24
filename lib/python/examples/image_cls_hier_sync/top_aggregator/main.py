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
"""MNIST horizontal FL aggregator for PyTorch.

The example below is implemented based on the following example from pytorch:
https://github.com/pytorch/examples/blob/master/mnist/main.py.
"""

import logging
import random
import time

from flame.mode.composer import Composer
from flame.mode.tasklet import Loop, Tasklet

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"

import torch
import torch.nn as nn
import torch.nn.functional as F
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.coord_syncfl.top_aggregator import TopAggregator
from torchvision import datasets, transforms
import torchvision.models as tormodels

from flame.fedscale_utils.femnist import FEMNIST
from flame.fedscale_utils.utils_data import get_data_transform
from torch.autograd import Variable

logger = logging.getLogger(__name__)
log_file = "/mydata/image_cls-top_aggregator.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

def override(method):
    return method

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

    model = model.to(device=device)  # load by pickle
    model.eval()

    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    with torch.no_grad():
        for data, target in test_data:
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                output = model(data)

                loss = criterion(output, target)
                test_loss += loss.data.item()  # Variable.data
                acc = accuracy(output, target, topk=(1, 5))

                correct += acc[0].item()
                top_5 += acc[1].item()

            except Exception as ex:
                logging.info(f"Testing of failed as {ex}")
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

    # logging.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
    #              .format(rank, test_loss, correct, len(test_data.dataset), acc, acc_5))

    testRes = {'top_1': correct, 'top_5': top_5,
               'test_loss': sum_loss, 'test_len': test_len}

    return test_loss, acc, acc_5, testRes

class PyTorchFemnistAggregator(TopAggregator):
    """PyTorch Femnist Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/femnist/"
        self.meta_dir = "/mydata/flame_dataset/femnist/"
        self.partition_id = 1

        self.previous_round_time = time.time()
        self.current_round_time = time.time()

    def initialize(self):
        """Initialize role."""
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = tormodels.__dict__["resnet152"](num_classes=62).to(device=self.device)


    def load_data(self) -> None:
        """Load a test dataset."""
        self.LOAD_START_T = time.time()

        # Generate a random parition ID
        self.partition_id = random.randint(1, 11)

        train_transform, test_transform = get_data_transform('mnist')
        test_dataset = FEMNIST(self.data_dir, self.meta_dir, self.partition_id,
                            dataset='test', transform=test_transform)

        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16,
                shuffle=True, pin_memory=True, timeout=0,
                num_workers=0, drop_last=False)

        self.dataset = Dataset(dataloader=self.test_loader)

        self.LOAD_END_T = time.time()
        self.load_data_delay = self.LOAD_END_T - self.LOAD_START_T

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        self.EVAL_START_T = time.time()
        test_loss, test_accuray, acc_5, testRes = test_pytorch_model(self.model, self.test_loader, device='cpu')
        self.EVAL_END_T = time.time()
        self.eval_delay = self.EVAL_END_T - self.EVAL_START_T

        self.current_round_time = time.time()
        logger.info(f"Wall-clock time: {self.current_round_time} ||"
                    f"Test loss: {test_loss} || "
                    f"Test accuracy: {test_accuray} || "
                    f"CPU time: {self.cpu_time:.4f} || "
                    f"CPU utilization: {self.utilization:.4f} || "
                    f"R#{self._round}'s duration (s): {self.current_round_time-self.previous_round_time:.4f} || "
                    f"Loading data delay (s): {self.load_data_delay:.4f} || "
                    f"Eval delay (s): {self.eval_delay:.4f} || "
                    f"Agg delay (s): {self.agg_delay:.4f} || "
                    f"Coordination delay (s): {self.coordination_delay:.4f} || "
                    f"DIST task delay: {self.dist_delay:.4f} || "
                    f"RECV task delay: {self.recv_delay:.4f} || "
                    f"Queueing delay: {self.queue_delay:.4f} || "
                    f"MSG (from mid) Ave. delay: {sum(self.msg_from_mid_delays)/len(self.msg_from_mid_delays):.4f} || "
                    f"Total cache delay: {sum(self.cache_delays):.4f} || "
                    f"Ave. cache delay: {sum(self.cache_delays)/len(self.cache_delays):.4f}")

        logger.info(f"Top Aggregator Timestamps: "
                    f"CACHE, CACHE_START_T: {self.CACHE_START_T}, CACHE_END_T: {self.CACHE_END_T} || "
                    f"AGG, AGG_START_T: {self.AGG_START_T}, AGG_END_T: {self.AGG_END_T} || "
                    f"EVAL, EVAL_START_T: {self.EVAL_START_T}, EVAL_END_T: {self.EVAL_END_T} || "
                    f"RECV, RECV_START_T: {self.RECV_START_T}, RECV_END_T: {self.RECV_END_T} || "
                    f"DIST, DIST_START_T: {self.DIST_START_T}, DIST_END_T: {self.DIST_END_T} || "
                    f"LOAD, LOAD_START_T: {self.LOAD_START_T}, LOAD_END_T: {self.LOAD_END_T}")

        self.previous_round_time = self.current_round_time

        # update metrics after each evaluation so that the metrics can be
        # logged in a model registry.
        self.update_metrics({
            'test-loss': test_loss,
            'test-accuracy': test_accuray
        })

    @override
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

            task_get_coord_ends = Tasklet("get_coord_ends", self.get_coordinated_ends)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_init
            >> loop(
                task_load_data
                >> task_get_coord_ends
                >> task_put
                >> task_get
                >> task_train
                >> task_eval
                >> task_analysis
                >> task_save_metrics
                >> task_increment_round
            )
            # >> task_end_of_training
            >> task_save_params
            >> task_save_model
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchFemnistAggregator(config)
    a.compose()
    a.run()