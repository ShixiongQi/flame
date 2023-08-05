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
from flame.mode.horizontal.top_aggregator import TopAggregator
from torchvision import datasets, transforms

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch.utils.model_zoo as model_zoo
from torch import Tensor, nn
from torch.autograd import Variable

from flame.fedscale_utils.speech import SPEECH
from flame.fedscale_utils.transforms_wav import (FixAudioLength, LoadAudio,
                                                    ToMelSpectrogram,
                                                    ToTensor)

logger = logging.getLogger(__name__)
log_file = "/mydata/google_speech_aggregator.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, in_channels=32):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

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

    criterion = nn.NLLLoss()

    with torch.no_grad():
        for data, target in test_data:
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)
                data = torch.unsqueeze(data, 1)

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

    # logging.info('Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
    #              .format(test_loss, correct, len(test_data.dataset), acc, acc_5))

    testRes = {'top_1': correct, 'top_5': top_5,
               'test_loss': sum_loss, 'test_len': test_len}

    return test_loss, acc, acc_5, testRes

class PyTorchGoogleSpeechAggregator(TopAggregator):
    """PyTorch Google Speech Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16
        self.num_class = 35
        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/google_speech/"
        self.meta_dir = "/mydata/flame_dataset/google_speech/"
        self.partition_id = 1

    def initialize(self):
        """Initialize role."""
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = resnet34(num_classes=35, in_channels=1).to(self.device)

    def load_data(self) -> None:
        """Load a test dataset."""

        # Generate a random parition ID
        self.partition_id = random.randint(0, 9)

        # Loading Testing Speech Dataset
        valid_feature_transform = transforms.Compose(
            [ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
        test_dataset = SPEECH(self.data_dir, self.meta_dir, self.partition_id, dataset='test',
                                transform=transforms.Compose([LoadAudio(),
                                                            FixAudioLength(),
                                                            valid_feature_transform]))

        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""

        test_loss, test_accuray, acc_5, testRes = test_pytorch_model(self.model, self.test_loader, device='cpu')

        logger.info(f"Wall-clock time: {time.time()} || Test loss: {test_loss} || Test accuracy: {acc_5} || CPU time: {self.cpu_time} || CPU utilization: {self.utilization}")

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

    a = PyTorchGoogleSpeechAggregator(config)
    a.compose()
    a.run()
