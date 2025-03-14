"""
Exception handling for the Neuromorphic Biomimetic UCAV SDK.

Defines custom exceptions and error handling utilities.
"""
from typing import Optional, Dict, Any, Type


class OberonError(Exception):
    """Base exception class for all SDK exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a new OberonError.
        
        Args:
            message: Error message.
            details: Additional error details.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(OberonError):
    """Exception raised for configuration-related errors."""
    pass


class ComponentError(OberonError):
    """Exception raised for component-related errors."""
    pass


class HardwareError(OberonError):
    """Exception raised for hardware-related errors."""
    pass


class SimulationError(OberonError):
    """Exception raised for simulation-related errors."""
    pass


class EthicsViolationError(OberonError):
    """Exception raised for ethics violations."""
    pass


class ValidationError(OberonError):
    """Exception raised for validation errors."""
    pass


def handle_exception(exception: Exception, reraise: bool = True) -> None:
    """
    Handle an exception with appropriate logging.
    
    Args:
        exception: The exception to handle.
        reraise: Whether to reraise the exception after handling.
    
    Raises:
        The original exception if reraise is True.
    """
    from .logging import logger
    
    if isinstance(exception, OberonError):
        logger.error(f"{exception.__class__.__name__}: {exception}")
    else:
        logger.error(f"Unexpected error: {exception}")
    
    if reraise:
        raise exception


def safe_execute(func, *args, **kwargs):
    """
    Execute a function safely with exception handling.
    
    Args:
        func: Function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        The result of the function call or None if an exception occurred.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_exception(e, reraise=False)
        return None