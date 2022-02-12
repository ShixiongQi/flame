"""Fledge plugin."""

import importlib
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Union

import yaml

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Plugin type enum class."""

    ANALYZER = 1


class Plugin(ABC):
    """Abstract class for plugin."""

    @abstractmethod
    def callback(self):
        """Abstract method to return plugin callback function."""


class PluginManager(object):
    """plugin manager class."""

    def __init__(self, plugins_path: str = '/etc/fledge/plugin') -> None:
        """Initialize instance."""
        self.filter: set[str] = set()
        self.plugins: dict[PluginType, list[Plugin]] = dict()

        for filename in os.listdir(plugins_path):
            filepath = os.path.join(plugins_path, filename)
            cls_name, package, ptype = self.parse_plugin(filepath)
            self.register_plugin(cls_name, package, ptype)

    def parse_plugin(self, filepath: str) -> Tuple[str, str, PluginType]:
        """Parse plugin configuration."""
        with open(filepath, 'r') as stream:
            data = yaml.safe_load(stream)

        for key in ['class', 'package', 'type']:
            if key not in data:
                raise KeyError(f"{key} not found")

        class_name = data['class']
        package = data['package']
        ptype_string = data['type'].upper()

        try:
            ptype = PluginType[ptype_string]
        except KeyError:
            raise KeyError(f"unknown plugin type: {ptype_string}")

        return class_name, package, ptype

    def register_plugin(self, cls_name: str, package: str,
                        ptype: PluginType) -> None:
        """Register a plugin."""
        plugin_key = package + "." + cls_name
        if plugin_key in self.filter:
            logger.debug(f"plugin {cls_name} from {package} already loaded")
            return

        self.filter.add(plugin_key)

        module = importlib.import_module(package)

        try:
            cls_obj = getattr(module, cls_name)
        except AttributeError:
            raise AttributeError(f'class {cls_name} not found')

        plugin_instance = cls_obj()
        if not isinstance(plugin_instance, Plugin):
            raise TypeError("not a plugin")

        # TODO: implement security check
        #       e.g., only allow  registration of whitelisted plugins

        if ptype not in self.plugins:
            self.plugins[ptype] = []

        self.plugins[ptype].append(plugin_instance)

    def get_plugins(self, ptype: PluginType) -> Union[None, list[Plugin]]:
        """Return a list of plugins of a given plugin type."""
        return self.plugins[ptype] if ptype in self.plugins else None