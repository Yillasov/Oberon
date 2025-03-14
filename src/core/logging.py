"""
Logging system for the Neuromorphic Biomimetic UCAV SDK.

Provides standardized logging capabilities throughout the SDK.
"""
import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Configure default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}


class OberonLogger:
    """
    Logger class for the Neuromorphic Biomimetic UCAV SDK.
    
    Provides standardized logging with configurable outputs.
    """
    
    def __init__(self, name: str = "oberon"):
        """
        Initialize a new logger.
        
        Args:
            name: Logger name.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.handlers = []
        
        # Add console handler by default
        self.add_console_handler()
    
    def add_console_handler(self, level: str = "info", 
                           format_str: str = DEFAULT_FORMAT) -> None:
        """
        Add a console handler to the logger.
        
        Args:
            level: Log level for the handler.
            format_str: Log format string.
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        formatter = logging.Formatter(format_str, DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.handlers.append(handler)
    
    def add_file_handler(self, file_path: Optional[Union[str, Path]] = None, 
                        level: str = "info", format_str: str = DEFAULT_FORMAT) -> None:
        """
        Add a file handler to the logger.
        
        Args:
            file_path: Path to the log file. If None, a default path is used.
            level: Log level for the handler.
            format_str: Log format string.
        """
        if file_path is None:
            # Create default log directory in user's home directory
            log_dir = Path.home() / ".oberon" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log file with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = log_dir / f"oberon_{timestamp}.log"
        else:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(file_path)
        handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        formatter = logging.Formatter(format_str, DEFAULT_DATE_FORMAT)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.handlers.append(handler)
    
    def set_level(self, level: str) -> None:
        """
        Set the logger's level.
        
        Args:
            level: Log level to set.
        """
        self.logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: Message to log.
            **kwargs: Additional context to include in the log.
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log.
            **kwargs: Additional context to include in the log.
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log.
            **kwargs: Additional context to include in the log.
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log.
            **kwargs: Additional context to include in the log.
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.error(message)
    
    def critical(self, message: str, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: Message to log.
            **kwargs: Additional context to include in the log.
        """
        if kwargs:
            message = f"{message} - Context: {kwargs}"
        self.logger.critical(message)
    
    def clear_handlers(self) -> None:
        """Remove all handlers from the logger."""
        for handler in self.handlers:
            self.logger.removeHandler(handler)
        self.handlers = []


# Create global logger instance
logger = OberonLogger()


def configure_logging(console_level: str = "info", 
                     file_level: Optional[str] = None,
                     log_file: Optional[Union[str, Path]] = None) -> None:
    """
    Configure global logging settings.
    
    Args:
        console_level: Log level for console output.
        file_level: Log level for file output. If None, no file logging is set up.
        log_file: Path to the log file. If None, a default path is used.
    """
    # Clear existing handlers
    logger.clear_handlers()
    
    # Add console handler
    logger.add_console_handler(level=console_level)
    
    # Add file handler if requested
    if file_level is not None:
        logger.add_file_handler(file_path=log_file, level=file_level)