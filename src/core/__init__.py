"""
Core module for the Neuromorphic Biomimetic UCAV SDK.

This module provides the foundational components and utilities for the SDK.
"""

# Import key components to make them available at the module level
from .component import Component
from .registry import ComponentRegistry, registry
from .configuration import Configuration, global_config
from .exceptions import OberonError, ConfigurationError, ComponentError
from .logging import logger, configure_logging