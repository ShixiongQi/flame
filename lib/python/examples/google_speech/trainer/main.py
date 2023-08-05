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
import random

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
from flame.mode.horizontal.trainer import Trainer
from torchvision import datasets, transforms

import math

import torch.utils.model_zoo as model_zoo
from torch import Tensor, nn

import os
from flame.fedscale_utils.speech import SPEECH, BackgroundNoiseDataset
from flame.fedscale_utils.transforms_stft import (AddBackgroundNoiseOnSTFT,
                                                    DeleteSTFT,
                                                    FixSTFTDimension,
                                                    StretchAudioOnSTFT,
                                                    TimeshiftAudioOnSTFT,
                                                    ToMelSpectrogramFromSTFT,
                                                    ToSTFT)
from flame.fedscale_utils.transforms_wav import (ChangeAmplitude,
                                                    ChangeSpeedAndPitchAudio,
                                                    FixAudioLength, LoadAudio,
                                                    ToMelSpectrogram,
                                                    ToTensor)

logger = logging.getLogger(__name__)

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

class PyTorchGoogleSpeechTrainer(Trainer):
    """PyTorch Google Speech Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None

        self.device = None
        self.train_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16
        self.num_class = 35
        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/google_speech"

        self.loss_squared = 0
        self.completed_steps = 0
        self.epoch_train_loss = 1e-4
        self.loss_decay = 0.2
        self.local_steps = 30
        self.meta_dir = "/mydata/flame_dataset/google_speech/"
        self.partition_id = 1

        self.load_background_noise_data()

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f" torch.cuda.is_available(): {torch.cuda.is_available()}")

        self.model = resnet34(num_classes=35, in_channels=1).to(self.device)

    def load_background_noise_data(self) -> None: 
        """Loading Background Noise Dataset"""

        bkg = '_background_noise_'
        self.data_aug_transform = transforms.Compose(
            [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
                TimeshiftAudioOnSTFT(), FixSTFTDimension()])
        bg_dataset = BackgroundNoiseDataset(
            os.path.join(self.data_dir, bkg), self.data_aug_transform)
        self.add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)

    def load_data(self) -> None:
        """Load data."""

        # Generate a random parition ID
        self.partition_id = random.randint(0, 499)

        # Loading Training Speech Dataset
        train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(
            n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
        train_dataset = SPEECH(self.data_dir, self.meta_dir, self.partition_id, dataset='train',
                                transform=transforms.Compose([LoadAudio(),
                                                            self.data_aug_transform,
                                                            self.add_bg_noise,
                                                            train_feature_transform])) 

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

    def train(self) -> None:
        """Train a model."""
        self.optimizer = optim.Adadelta(self.model.parameters())

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.unsqueeze(1)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=self.device)
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

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_init
                >> loop(
                    task_load_data >> task_get >> task_train >> task_eval >> task_put >> task_save_metrics
                )
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchGoogleSpeechTrainer(config)
    t.compose()
    t.run()
