# -*- coding: utf-8 -*-
import csv
import logging
import random
import time
from collections import defaultdict
from random import Random

import numpy as np
from torch.utils.data import DataLoader

class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, test_ratio=1.0, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.test_ratio = test_ratio
        # self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.numOfLabels = numOfClass
        self.client_label_cnt = defaultdict(set)

    # def getNumOfLabels(self):
    #     return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def getClientLen(self):
        return len(self.partitions)

    def getClientLabel(self):
        return [len(self.client_label_cnt[i]) for i in range(self.getClientLen())]

    def use(self, partition, istest):
        resultIndex = partition # self.partitions[partition % len(self.partitions)]

        exeuteLength = len(resultIndex) if not istest else int(
            len(resultIndex) * self.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}
