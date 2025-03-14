"""
Unified configuration system for the Oberon project.
"""
import os
import json
import logging
from typing import Any, Dict, Optional, List, Union, Callable

logger = logging.getLogger("config_manager")

class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass

class ConfigValidator:
    """Validator for configuration parameters."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Union[type, List[type]]) -> bool:
        """Validate that a value is of the expected type."""
        if isinstance(expected_type, list):
            return any(isinstance(value, t) for t in expected_type)
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                      max_val: Optional[Union[int, float]] = None) -> bool:
        """Validate that a numeric value is within the specified range."""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    @staticmethod
    def validate_options(value: Any, options: List[Any]) -> bool:
        """Validate that a value is one of the specified options."""
        return value in options
    
    @staticmethod
    def validate_path(value: str, must_exist: bool = True) -> bool:
        """Validate that a path exists if required."""
        if must_exist:
            return os.path.exists(value)
        return True
    
    @staticmethod
    def validate_custom(value: Any, validator_func: Callable[[Any], bool]) -> bool:
        """Validate using a custom validation function."""
        return validator_func(value)


class ConfigManager:
    """
    Centralized configuration management for Oberon.
    
    This class provides a unified way to manage configurations across the system,
    supporting environment-specific settings, validation, and versioning.
    """
    
    # Schema defines expected types and validation rules for config parameters
    DEFAULT_SCHEMA = {
        "version": {
            "type": str,
            "required": True,
            "validator": lambda v: len(v.split('.')) >= 3  # Semantic versioning
        },
        "paths.base": {
            "type": str,
            "required": True
        },
        "paths.test_results": {
            "type": str,
            "required": True
        },
        "paths.regression_history": {
            "type": str,
            "required": True
        },
        "testing.stop_on_failure": {
            "type": bool,
            "required": True
        },
        "testing.log_level": {
            "type": str,
            "required": True,
            "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        },
        "system.update_rate": {
            "type": [int, float],
            "required": True,
            "min": 0.1,
            "max": 1000
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, env: str = "development", 
                schema: Optional[Dict] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration directory or file
            env: Environment name (development, testing, production)
            schema: Custom validation schema (defaults to DEFAULT_SCHEMA)
        """
        self.env = env
        self.base_path = os.path.expanduser("~/Oberon")
        self.schema = schema or self.DEFAULT_SCHEMA
        
        # Determine config path
        if config_path:
            self.config_path = config_path
        else:
            # Check for environment variable override
            env_config_path = os.environ.get('OBERON_CONFIG_PATH')
            if env_config_path:
                self.config_path = env_config_path
            else:
                self.config_path = os.path.join(self.base_path, "config", f"{env}.json")
        
        # Load base and environment-specific configurations
        self.config = self._load_config()
        self.version = self.config.get("version", "0.0.0")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration on initialization
        self.validation_errors = []
        if not self.validate(raise_exception=False):
            logger.warning(f"Configuration validation failed with {len(self.validation_errors)} errors")
            for error in self.validation_errors:
                logger.warning(f"  - {error}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with environment layering."""
        # Start with default configuration
        config = self._get_default_config()
        
        # Load base configuration if it exists
        base_config_path = os.path.join(os.path.dirname(self.config_path), "base.json")
        if os.path.exists(base_config_path):
            try:
                with open(base_config_path, 'r') as f:
                    base_config = json.load(f)
                    self._deep_merge(config, base_config)
                    logger.debug(f"Loaded base configuration from {base_config_path}")
            except Exception as e:
                logger.error(f"Error loading base configuration: {e}")
        
        # Load environment-specific configuration
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    env_config = json.load(f)
                    self._deep_merge(config, env_config)
                    logger.debug(f"Loaded environment configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading environment configuration: {e}")
        else:
            logger.warning(f"Environment configuration file not found: {self.config_path}")
            # Create default config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load local overrides if they exist (not version controlled)
        local_config_path = os.path.join(os.path.dirname(self.config_path), "local.json")
        if os.path.exists(local_config_path):
            try:
                with open(local_config_path, 'r') as f:
                    local_config = json.load(f)
                    self._deep_merge(config, local_config)
                    logger.debug(f"Loaded local configuration from {local_config_path}")
            except Exception as e:
                logger.error(f"Error loading local configuration: {e}")
        
        return config
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """
        Deep merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Look for environment variables with OBERON_CONFIG_ prefix
        prefix = "OBERON_CONFIG_"
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Convert environment variable name to config key path
                # e.g. OBERON_CONFIG_SYSTEM_UPDATE_RATE -> system.update_rate
                key_path = env_var[len(prefix):].lower().replace('_', '.')
                
                # Try to convert value to appropriate type
                try:
                    # Try as JSON first (for complex types)
                    typed_value = json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to string
                    typed_value = value
                
                # Set the configuration value
                self.set(key_path, typed_value)
                logger.debug(f"Applied environment override: {key_path} = {typed_value}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "version": "0.0.0",
            "paths": {
                "base": self.base_path,
                "test_results": os.path.join(self.base_path, "test_results"),
                "regression_history": os.path.join(self.base_path, "test_results", "regression_history")
            },
            "testing": {
                "stop_on_failure": False,
                "log_level": "INFO"
            },
            "system": {
                "update_rate": 10  # Hz
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration value (e.g., "paths.base")
            default: Default value if the key doesn't exist
            
        Returns:
            The configuration value or default
        """
        parts = key_path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration value (e.g., "paths.base")
            value: Value to set
        """
        parts = key_path.split('.')
        config = self.config
        
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def save(self, include_local: bool = False) -> bool:
        """
        Save the current configuration to file.
        
        Args:
            include_local: Whether to save local overrides separately
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine what to save to the environment config vs local config
            if include_local:
                # Load the current env config to compare
                env_config = {}
                if os.path.exists(self.config_path):
                    with open(self.config_path, 'r') as f:
                        env_config = json.load(f)
                
                # Create a local config with differences
                local_config = {}
                self._extract_differences(self.config, env_config, local_config)
                
                # Save local config if it has content
                if local_config:
                    local_path = os.path.join(os.path.dirname(self.config_path), "local.json")
                    with open(local_path, 'w') as f:
                        json.dump(local_config, f, indent=2)
                    logger.info(f"Saved local configuration to {local_path}")
            
            # Save main config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _extract_differences(self, current: Dict, base: Dict, result: Dict) -> None:
        """
        Extract differences between current and base configs.
        
        Args:
            current: Current configuration
            base: Base configuration to compare against
            result: Dictionary to store differences in
        """
        for key, value in current.items():
            if key not in base:
                # Key doesn't exist in base, include it
                result[key] = value
            elif isinstance(value, dict) and isinstance(base[key], dict):
                # Recurse into nested dictionaries
                result[key] = {}
                self._extract_differences(value, base[key], result[key])
                # Remove empty dictionaries
                if not result[key]:
                    del result[key]
            elif value != base[key]:
                # Value is different, include it
                result[key] = value
    
    def switch_environment(self, new_env: str) -> None:
        """
        Switch to a different environment configuration.
        
        Args:
            new_env: New environment name
        """
        if new_env == self.env:
            return
        
        # Save current config if needed
        # (could add logic here to prompt user)
        
        # Update environment and config path
        self.env = new_env
        self.config_path = os.path.join(os.path.dirname(self.config_path), f"{new_env}.json")
        
        # Reload configuration
        self.config = self._load_config()
        self.version = self.config.get("version", "0.0.0")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate new configuration
        self.validate(raise_exception=False)
        
        logger.info(f"Switched to environment: {new_env}")
    
    def validate(self, raise_exception: bool = True) -> bool:
        """
        Validate the configuration against the schema.
        
        Args:
            raise_exception: Whether to raise an exception on validation failure
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ConfigValidationError: If validation fails and raise_exception is True
        """
        self.validation_errors = []
        
        # Check all schema rules
        for key_path, rules in self.schema.items():
            value = self.get(key_path)
            
            # Check if required
            if rules.get("required", False) and value is None:
                self.validation_errors.append(f"Missing required configuration key: {key_path}")
                continue
                
            # Skip validation if value is None and not required
            if value is None:
                continue
                
            # Type validation
            expected_type = rules.get("type")
            if expected_type and not ConfigValidator.validate_type(value, expected_type):
                type_names = [t.__name__ for t in expected_type] if isinstance(expected_type, list) else [expected_type.__name__]
                self.validation_errors.append(
                    f"Invalid type for {key_path}: expected {' or '.join(type_names)}, got {type(value).__name__}"
                )
                
            # Range validation for numeric values
            if isinstance(value, (int, float)) and (
                "min" in rules or "max" in rules
            ):
                if not ConfigValidator.validate_range(
                    value, rules.get("min"), rules.get("max")
                ):
                    min_val = rules.get("min", "any")
                    max_val = rules.get("max", "any")
                    self.validation_errors.append(
                        f"Value for {key_path} out of range: {value} (expected between {min_val} and {max_val})"
                    )
                    
            # Options validation
            if "options" in rules:
                if not ConfigValidator.validate_options(value, rules["options"]):
                    self.validation_errors.append(
                        f"Invalid value for {key_path}: {value} (expected one of {rules['options']})"
                    )
                    
            # Path validation
            if "path_must_exist" in rules and isinstance(value, str):
                if not ConfigValidator.validate_path(value, rules["path_must_exist"]):
                    self.validation_errors.append(
                        f"Path does not exist: {value} (for {key_path})"
                    )
                    
            # Custom validator
            if "validator" in rules and callable(rules["validator"]):
                if not ConfigValidator.validate_custom(value, rules["validator"]):
                    self.validation_errors.append(
                        f"Custom validation failed for {key_path}: {value}"
                    )
        
        # Handle validation errors
        if self.validation_errors and raise_exception:
            error_msg = "\n".join(self.validation_errors)
            raise ConfigValidationError(f"Configuration validation failed:\n{error_msg}")
            
        return len(self.validation_errors) == 0


# Global configuration instance
_config_instance = None

def get_config(config_path: Optional[str] = None, env: str = "development", 
              schema: Optional[Dict] = None) -> ConfigManager:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file
        env: Environment name
        schema: Custom validation schema
        
    Returns:
        ConfigManager instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path, env, schema)
    
    return _config_instance