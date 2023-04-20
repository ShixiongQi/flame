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
"""Channel manager."""

import asyncio
import atexit
import logging
import sys
from typing import Optional

from flame.backends import backend_provider
from flame.channel import Channel
from flame.common.constants import BackendEvent
from flame.common.util import background_thread_loop, run_async
from flame.config import Config
from flame.selectors import selector_provider

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def custom_excepthook(exc_type, exc_value, exc_traceback):
    """Implement a custom exception hook.

    NOTE: this custom version is implemented due to the following warning
    message printed at the end of execution:
    "Error in sys.excepthook:

    Original exception was:"
    This is caused by _inner() function in cleanup().
    A root-cause is not identified. As a workaround, this custom hook is
    implemented and set to sys.excepthook
    """
    logger.critical(
        "Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = custom_excepthook


class ChannelManager(object):
    """ChannelManager manages channels and creates a singleton instance."""

    _instance = None

    _config = None
    _job_id = None
    _role = None

    _channels = None

    _loop = None

    _backend = None  # default backend in case there is no per-channel backend
    _backends = dict()  # backend per channel

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            logger.info("creating a ChannelManager instance")
            cls._instance = super(ChannelManager, cls).__new__(cls)
        return cls._instance

    def __call__(self, config: Config):
        """Initialize instance variables."""
        self._config = config
        self._job_id = self._config.job.job_id
        self._role = self._config.role
        self._task_id = self._config.task_id

        self._channels = {}

        self._setup_backends()

        atexit.register(self.cleanup)

    def _setup_backends(self):
        async def inner(q: asyncio.Queue) -> None:
            # create a coroutine task
            coro = self._backend_eventq_task(q)
            _ = asyncio.create_task(coro)

        for ch_name, channel in self._config.channels.items():
            # rename backend in channel config as sort to avoid confusion
            sort = channel.backend
            if not sort:
                # channel doesn't have its own backend, notthing to do
                continue

            # get a backend instance
            backend = backend_provider.get(sort)
            broker_host = channel.broker_host or self._config.brokers.sort_to_host[sort]

            backend.configure(broker_host, self._job_id, self._task_id)
            _ = run_async(inner(backend.eventq()), backend.loop())

            self._backends[ch_name] = backend

        if len(self._backends) == len(self._config.channels):
            # every channel has its own backend
            # no need to have a default backend
            return

        # set up a default backend
        sort = self._config.backend
        self._backend = backend_provider.get(sort, role = self._role)
        broker_host = self._config.brokers.sort_to_host[sort]
        self._backend.configure(broker_host, self._job_id, self._task_id)
        _ = run_async(inner(self._backend.eventq()), self._backend.loop())

    async def _backend_eventq_task(self, eventq):
        while True:
            (event_type, info) = await eventq.get()

            if event_type == BackendEvent.DISCONNECT:
                for _, channel in self._channels.items():
                    await channel.remove(info)

    def join_all(self) -> None:
        """join_all ensures that a role joins all of its channels."""
        for ch_name in self._config.channels.keys():
            self.join(ch_name, None)

    def join_cp(self) -> None:
        """join_cp ensures that a role only joins control plane channels (between coordinator and others)."""
        logger.info(f"Channels: {self._config.channels.keys()}")
        for ch_name in self._config.channels.keys():
            channel_config = self._config.channels[ch_name]

            if "coordinator" == channel_config.pair[0] or "coordinator" == channel_config.pair[1]:
                self.join(ch_name, None, 0)
            else:
                logger.info(f"{ch_name} is a dataplane channel... Skip it for now.")
                continue

    def join_dp(self, mid_agg_urls: dict, num_ends: int) -> None:
        """join_dp ensures that a role only joins dataplane channels (top_aggregator -> mid_aggregator, trainer -> mid_aggregator)."""
        for ch_name in self._config.channels.keys():
            channel_config = self._config.channels[ch_name]

            if "coordinator" == channel_config.pair[0] or "coordinator" == channel_config.pair[1]:
                continue
            else:
                self.join(ch_name, mid_agg_urls, num_ends)

    def join(self, name: str, mid_agg_urls: dict, num_ends: int) -> bool:
        """Join a channel."""
        if self.is_joined(name):
            return True

        channel_config = self._config.channels[name]

        if self._role == channel_config.pair[0]:
            me = channel_config.pair[0]
            other = channel_config.pair[1]
        else:
            me = channel_config.pair[1]
            other = channel_config.pair[0]

        groupby = channel_config.group_by.groupable_value(
            self._config.group_association.get(name)
        )

        selector = selector_provider.get(
            self._config.selector.sort, **self._config.selector.kwargs
        )

        if name in self._backends:
            backend = self._backends[name]
        else:
            logger.info(f"no backend found for channel {name}; use default")
            backend = self._backend

        self._channels[name] = Channel(
            num_ends, mid_agg_urls, backend, selector, self._job_id, name, me, other, groupby
        )
        self._channels[name].join()

    def leave(self, name):
        """Leave a channel."""
        if not self.is_joined(name):
            return
        # TODO: reset_channel isn't implemented yet

    def get_by_tag(self, tag: str) -> Optional[Channel]:
        """Return a channel object that matches a given function tag."""
        if tag not in self._config.func_tag_map:
            return None

        channel_name = self._config.func_tag_map[tag]
        logger.debug(f"{tag} through {channel_name}")
        return self.get(channel_name)

    def get(self, name: str) -> Optional[Channel]:
        """Return a channel object in a given channel name."""
        if not self.is_joined(name):
            # didn't join the channel yet
            logger.warning(f"{name} hasn't been joined")
            return None

        return self._channels[name]

    def is_joined(self, name):
        """Check if node joined a channel or not."""
        return True if name in self._channels else False

    def cleanup(self):
        """Clean up pending asyncio tasks."""
        for _, ch in self._channels.items():
            ch.cleanup()

        async def _inner(backend):
            # TODO: need better mechanism to wait tx completion
            # as a temporary measure, sleep 5 seconds
            await asyncio.sleep(5)

            # clean up backend
            await backend.cleanup()

            for task in asyncio.all_tasks():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"successfully cancelled {task.get_name()}")

            logger.debug("done with cleaning up asyncio tasks")

        if self._backend:
            _ = run_async(_inner(self._backend), self._backend.loop())

        for k, v in self._backends.items():
            _ = run_async(_inner(v), v.loop())
