"""
Configuration management for the Neuromorphic Biomimetic UCAV SDK.

Provides utilities for loading, saving, and managing configuration settings.
"""
import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from .exceptions import ConfigurationError


class Configuration:
    """
    Configuration management class.
    
    Handles loading, saving, and accessing configuration settings.
    """
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize a new configuration instance.
        
        Args:
            config_data: Initial configuration data (optional).
        """
        self._config: Dict[str, Any] = config_data or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested access).
            default: Default value to return if key is not found.
            
        Returns:
            The configuration value or default if not found.
        """
        parts = key.split('.')
        current = self._config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested access).
            value: Value to set.
        """
        parts = key.split('.')
        current = self._config
        
        # Navigate to the correct nested dictionary
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
                
        # Set the value
        current[parts[-1]] = value
    
    def update(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration with new data.
        
        Args:
            config_data: Configuration data to update with.
        """
        self._deep_update(self._config, config_data)
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            target: Target dictionary to update.
            source: Source dictionary with new values.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def load_file(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Supports JSON and YAML formats based on file extension.
        
        Args:
            file_path: Path to the configuration file.
            
        Raises:
            ConfigurationError: If the file cannot be loaded.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {file_path.suffix}")
                
                if config_data:
                    self.update(config_data)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def save_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Supports JSON and YAML formats based on file extension.
        
        Args:
            file_path: Path to save the configuration file.
            
        Raises:
            ConfigurationError: If the file cannot be saved.
        """
        file_path = Path(file_path)
        
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False)
                elif file_path.suffix.lower() == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return self._config.copy()
    
    def clear(self) -> None:
        """Clear all configuration data."""
        self._config.clear()


# Global configuration instance
global_config = Configuration()