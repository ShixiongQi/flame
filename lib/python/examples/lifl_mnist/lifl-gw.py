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
"""
LIFL Gateway
"""

import time, socket, os, sys
import argparse, logging, yaml
from _thread import *
from concurrent import futures
import threading
from threading import Thread

from datetime import datetime
from flame.config import Config
from flame.mode.horizontal.top_aggregator import TopAggregator
from flame.mode.composer import Composer
from flame.mode.tasklet import Loop, Tasklet
from flame.channel_manager import ChannelManager
from flame.optimizer.train_result import TrainResult
from flame.mode.message import MessageType
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    valid_frameworks,
    weights_to_device,
    weights_to_model_device,
)
from flame.common.constants import DeviceType
from diskcache import Cache

from http.server import BaseHTTPRequestHandler, HTTPServer

# PLASMA deps
import pyarrow.plasma as plasma
import numpy as np
import pyarrow as pa

DEFAULT_LOG_LEVEL='info'
logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"
PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_END_TIME = "round_end_time"

import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# class LIFLGateway(object):
class LIFLGateway(TopAggregator):
    def __init__(self, config: Config) -> None:
        logger.info("Initialize LIFL Gateway")

        # Flame's abstract_attribute
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        # Attach to PLASMA object store
        logger.debug('Connecting to PLASMA server...')
        self.plasma_client = plasma.connect("/tmp/plasma")

        self.sockmap_server_ip   = "127.0.0.1"
        self.sockmap_server_port = 8081
        self.rpc_server_ip     = "127.0.0.1"
        self.rpc_server_port   = 8082

        # self.route = route

        self.succ_req = 0

        logger.info('Connecting to sockmap server {}:{}...'.format(self.sockmap_server_ip, self.sockmap_server_port))
        self.sockmap_sock = self._sockmap_client(self.sockmap_server_ip, self.sockmap_server_port)

        self.pid = os.getpid()
        self.sock_fd = self.sockmap_sock.fileno()
        self.fn_id = 0
        logger.info("SKMSG metadata: PID {}; socket FD {}; Fn ID {}".format(self.pid, self.sock_fd, self.fn_id))

        logger.info('Connecting to RPC server {}:{}...'.format(self.rpc_server_ip, self.rpc_server_port))
        self.rpc_sock = self._rpc_client(self.rpc_server_ip, self.rpc_server_port)

        skmsg_md = [self.pid, self.sock_fd, self.fn_id]
        skmsg_md_bytes = b''.join([skmsg_md[0].to_bytes(4, byteorder = 'little'), skmsg_md[1].to_bytes(4, byteorder = 'little'), skmsg_md[2].to_bytes(4, byteorder = 'little')])
        logger.debug("Socket metadata: {}".format(skmsg_md_bytes))

        self.rpc_sock.send(skmsg_md_bytes)
        self.rpc_sock.close()

        logger.info('Gateway is running..')

    def internal_init(self) -> None:
        """Initialize internal state for role."""

        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self._work_done = False
        self._round = 1

        self.cache = Cache("/mydata/tmp/obj_store")
        self.cache.reset("size_limit", 1e15)
        self.cache.reset("cull_limit", 0)

        # Initialize model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)

        self._update_weights()

    def _create_socket(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as msg:
            print('Failed to create socket. Error code: {}, Error message: {}'.format(str(msg[0]), msg[1]))
            sys.exit()

        return sock

    def _rpc_client(self, remote_ip, port):
        sock = self._create_socket()

        sock.connect((remote_ip, port))
        logger.info('Connected to RPC server {}:{}'.format(remote_ip, port))

        return sock

    def _sockmap_client(self, remote_ip, port):
        sock = self._create_socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        sock.connect((remote_ip, port))
        logger.info('Connected to sockmap server {}:{}'.format(remote_ip, port))

        return sock

    """
    def gw_rx(self):
        skmsg_md_bytes = self.sockmap_sock.recv(128)
        logger.debug("Gateway completes #{} request: {}".format(self.succ_req, skmsg_md_bytes))
        self.succ_req = self.succ_req + 1

    def gw_tx(self, next_fn, cur_hop, shm_obj_name):
        logger.debug("Gateway TX thread sends SKMSG to Fn#{}".format(next_fn))
        next_hop = cur_hop + 1
        skmsg_md_bytes = b''.join([next_fn.to_bytes(4, byteorder = 'little'), \
                                   next_hop.to_bytes(4, byteorder = 'little'), \
                                   shm_obj_name])
        self.sockmap_sock.sendall(skmsg_md_bytes)

    def router(self, cur_hop):
        if cur_hop == len(self.route):
            # NOTE: The route ends. Back to Gateway
            next_fn = 0
        else:
            next_fn = self.route[cur_hop]
        logger.debug("Routing result: current hop#{}, next function ID is {}".format(cur_hop, next_fn))
        return next_fn

    # core() is the frontend of LIFL gateway
    # It's used to interact between the http handler and function chains
    def core(self, obj_id):
        cur_hop = 0 # NOTE: The function chain always starts from hop#0
        next_fn = self.router(cur_hop)
        if next_fn == 0:
            logger.debug("No route found. Gateway returns response directly")
            return
        self.gw_tx(next_fn, cur_hop, obj_id)

        # Returned from the function chain
        self.gw_rx()
    """

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_AGGREGATE:
            self._receive_weights(tag)

    def _receive_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0

        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            # if MessageType.DATASAMPLER_METADATA in msg:
            #     self.datasampler.handle_metadata_from_trainer(
            #         msg[MessageType.DATASAMPLER_METADATA],
            #         end,
            #         channel,
            #     )

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                self.cache['end'] = tres # save training result from trainer in a disk cache

                print(self.cache['end'])

                # self.cache_weights(tres)
        
        self.weights = weights

    """
    def cache_weights(self, tres):
        weights = tres.weights

        numpy_weights = {key: value.cpu().numpy() for key, value in weights.items()}

        # Serialize the TrainResult object to Arrow's in-memory format
        train_result_dict = {
            'weights': numpy_weights,
            'count': tres.count,
            'version': tres.version
        }

        serialized_train_result = pa.serialize(train_result_dict)

        # print(f"size of table_w: {len(table_w.to_buffer().to_pybytes())}")
        # print(f"size of serialized_train_result: {len(serialized_train_result.to_buffer().to_pybytes())}")

        # # Get the size of the numpy_weights dictionary
        # total_size_bytes = 0
        # for key, value in numpy_weights.items():
        #     size_bytes = value.nbytes
        #     total_size_bytes += size_bytes

        # # Convert to a more human-readable format (e.g., megabytes)
        # total_size_megabytes = total_size_bytes # / (1024 * 1024)
        # print(f"size of numpy_weights: {total_size_megabytes}")

        # Create the object in Plasma
        object_id_str = np.random.bytes(20)
        logger.debug(f"Write weights into a PLASMA object: {object_id_str}")
        object_id = plasma.ObjectID(object_id_str)

        buf = self.plasma_client.create(object_id, len(serialized_train_result.to_buffer().to_pybytes()))

        # Write the tensor into the Plasma-allocated buffer
        stream = pa.BufferOutputStream(buf)
        # stream = pa.FixedSizeBufferWriter(buf)
        stream.set_memcopy_threads(4)
        # pa.ipc.write(serialized_train_result, stream)
        stream.write(serialized_train_result)

        # Seal the Plasma object
        self.plasma_client.seal(object_id)

    def write_weights_to_plasma(self):
        logger.debug("LIFL Gateway is handling GET request")

        # Write request into a PLASMA object
        # Create a pyarrow.Tensor object from a numpy random 2-dimensional array
        data = np.random.randn(10, 4)
        tensor = pa.Tensor.from_numpy(data)

        # Create the object in Plasma
        object_id_str = np.random.bytes(20)
        logger.debug(f"GET handler::obj_id:{object_id_str}")
        object_id = plasma.ObjectID(object_id_str)
        buf = self.plasma_client.create(object_id, pa.ipc.get_tensor_size(tensor))

        # Write the tensor into the Plasma-allocated buffer
        stream = pa.FixedSizeBufferWriter(buf)
        stream.set_memcopy_threads(4)
        pa.ipc.write_tensor(tensor, stream)  # Writes tensor's 552 bytes to Plasma stream

        # Seal the Plasma object
        self.plasma_client.seal(object_id)

        time.sleep(2)
        # Handover request to LIFL gateway core
        # self.core(object_id_str)
    """

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_DISTRIBUTE:
            self.dist_tag = tag
            self._distribute_weights(tag)

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        # self._update_weights()

        selected_ends = channel.ends()
        # datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        # send out global model parameters to trainers
        for end in selected_ends:
            logger.debug(f"sending weights to {end}")
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    # MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

    def _update_weights(self):
        self.weights = self.model.state_dict()

    def initialize(self):
        pass

    def load_data(self) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            task_write_weights_to_plasma = Tasklet("write_weights_to_plasma", self.write_weights_to_plasma)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> loop(
                task_put
                >> task_get
                # >> task_write_weights_to_plasma
                # task_write_weights_to_plasma
            )
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'LIFL Gateway')
    parser.add_argument('config', nargs='?', default="./config.json")
    parser.add_argument('--log-level', help = 'Log level', default = DEFAULT_LOG_LEVEL)

    args = parser.parse_args()

    # logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.getLevelName(args.log_level.upper()))
    config = Config(args.config)

    # Creating a LIFL gateway object
    gw = LIFLGateway(config)
    gw.compose()
    gw.run()

    # Print bpf trace logs
    """
    while True:
        try:
            # bpf.trace_print()
            time.sleep(1)
        except KeyboardInterrupt:
            server.stop()
            print("Gateway stopped.")
            sys.exit(0)
    """