"""
Unit tests for the logging module.
"""
import pytest
import logging
from src.core.logging import OberonLogger, configure_logging


class TestLogging:
    """Tests for the logging module."""
    
    def test_logger_creation(self):
        """Test creating a logger."""
        logger = OberonLogger("test_logger")
        assert logger.logger.name == "test_logger"
        
        # Test that the default handler is added
        assert len(logger.handlers) > 0
    
    def test_log_levels(self):
        """Test setting log levels."""
        logger = OberonLogger("test_levels")
        
        # Test setting level
        logger.set_level("debug")
        assert logger.logger.level == logging.DEBUG
        
        logger.set_level("info")
        assert logger.logger.level == logging.INFO
        
        logger.set_level("warning")
        assert logger.logger.level == logging.WARNING
        
        logger.set_level("error")
        assert logger.logger.level == logging.ERROR
        
        logger.set_level("critical")
        assert logger.logger.level == logging.CRITICAL
    
    def test_file_handler(self, tmp_path):
        """Test adding a file handler."""
        log_file = tmp_path / "test.log"
        logger = OberonLogger("test_file")
        
        # Add file handler
        logger.add_file_handler(log_file, level="debug")
        
        # Log a message
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check that the file exists and contains the messages
        assert log_file.exists()
        log_content = log_file.read_text()
        
        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content
    
    def test_configure_logging(self):
        """Test the configure_logging function."""
        # Test with default settings
        configure_logging()
        
        # Test with custom settings
        configure_logging(console_level="debug", file_level="info")