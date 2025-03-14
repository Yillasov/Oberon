"""
Test configuration for the Neuromorphic Biomimetic UCAV SDK.

This file contains pytest fixtures and configuration for the test suite.
"""
import os
import sys
import pytest
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Configuration, logger, configure_logging


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Configure logging for tests."""
    configure_logging(console_level="debug")
    return logger


@pytest.fixture
def config():
    """Provide a clean configuration instance for tests."""
    return Configuration()


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"