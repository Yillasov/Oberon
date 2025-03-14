"""
Configuration loader for neuromorphic system parameters.
"""
import os
import json
import yaml
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
import logging
from pathlib import Path
from .system_core import SystemCore
from .component_registry import ComponentRegistry, ComponentMetadata


@dataclass
class ConfigSection:
    """Configuration section with validation."""
    name: str
    parameters: Dict[str, Any]
    schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_params: Set[str] = field(default_factory=set)
    
    def validate(self) -> List[str]:
        """Validate configuration section against schema."""
        errors = []
        
        # Check required parameters
        for param in self.required_params:
            if param not in self.parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate against schema
        for param_name, param_value in self.parameters.items():
            if param_name in self.schema:
                param_schema = self.schema[param_name]
                
                # Type validation
                expected_type = param_schema.get("type")
                if expected_type:
                    if expected_type == "int" and not isinstance(param_value, int):
                        errors.append(f"Parameter {param_name} should be an integer")
                    elif expected_type == "float" and not isinstance(param_value, (int, float)):
                        errors.append(f"Parameter {param_name} should be a number")
                    elif expected_type == "str" and not isinstance(param_value, str):
                        errors.append(f"Parameter {param_name} should be a string")
                    elif expected_type == "bool" and not isinstance(param_value, bool):
                        errors.append(f"Parameter {param_name} should be a boolean")
                    elif expected_type == "list" and not isinstance(param_value, list):
                        errors.append(f"Parameter {param_name} should be a list")
                    elif expected_type == "dict" and not isinstance(param_value, dict):
                        errors.append(f"Parameter {param_name} should be a dictionary")
                
                # Range validation
                if isinstance(param_value, (int, float)):
                    min_val = param_schema.get("min")
                    max_val = param_schema.get("max")
                    
                    if min_val is not None and param_value < min_val:
                        errors.append(f"Parameter {param_name} should be >= {min_val}")
                    if max_val is not None and param_value > max_val:
                        errors.append(f"Parameter {param_name} should be <= {max_val}")
                
                # Enum validation
                allowed_values = param_schema.get("allowed_values")
                if allowed_values and param_value not in allowed_values:
                    errors.append(
                        f"Parameter {param_name} should be one of: {', '.join(map(str, allowed_values))}"
                    )
        
        return errors


class ConfigLoader:
    """Configuration loader for system parameters."""
    
    def __init__(self, system_core: SystemCore, registry: ComponentRegistry):
        self.core = system_core
        self.registry = registry
        self.logger = system_core.logger
        
        # Configuration storage
        self.config_sections: Dict[str, ConfigSection] = {}
        self.config_files: List[str] = []
        self.base_config_dir = "/Users/yessine/Oberon/config"
        
        # Register with system
        self.core.register_component("config_loader", self)
        self.registry.register(
            self,
            ComponentMetadata(
                name="config_loader",
                component_type="integration",
                version="1.0",
                provides={"config_service"},
                startup_priority=95  # Start very early
            )
        )
    
    async def start(self):
        """Start the configuration loader."""
        self.logger.info("Configuration loader starting")
        
        # Load default configuration
        default_config = os.path.join(self.base_config_dir, "default.yaml")
        if os.path.exists(default_config):
            self.load_config_file(default_config)
        
        # Load environment-specific configuration
        env = os.environ.get("NEUROMORPHIC_ENV", "development")
        env_config = os.path.join(self.base_config_dir, f"{env}.yaml")
        if os.path.exists(env_config):
            self.load_config_file(env_config)
    
    async def stop(self):
        """Stop the configuration loader."""
        self.logger.info("Configuration loader stopping")
    
    def load_config_file(self, file_path: str) -> bool:
        """Load configuration from file."""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Config file not found: {file_path}")
                return False
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
            else:
                self.logger.error(f"Unsupported config file format: {file_ext}")
                return False
            
            # Process configuration sections
            for section_name, section_data in config_data.items():
                if section_name in self.config_sections:
                    # Update existing section
                    self.config_sections[section_name].parameters.update(section_data)
                else:
                    # Create new section
                    self.config_sections[section_name] = ConfigSection(
                        name=section_name,
                        parameters=section_data
                    )
            
            self.config_files.append(file_path)
            self.logger.info(f"Loaded configuration from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading config file {file_path}: {str(e)}")
            return False
    
    def register_schema(self, 
                      section: str, 
                      schema: Dict[str, Dict[str, Any]],
                      required_params: Optional[List[str]] = None) -> bool:
        """Register schema for configuration section."""
        if section not in self.config_sections:
            # Create empty section
            self.config_sections[section] = ConfigSection(
                name=section,
                parameters={}
            )
        
        self.config_sections[section].schema = schema
        
        if required_params:
            self.config_sections[section].required_params = set(required_params)
        
        # Validate against existing configuration
        errors = self.config_sections[section].validate()
        if errors:
            for error in errors:
                self.logger.warning(f"Config validation error in {section}: {error}")
        
        return len(errors) == 0
    
    def get_param(self, 
                section: str, 
                param: str, 
                default: Any = None) -> Any:
        """Get configuration parameter."""
        if section not in self.config_sections:
            return default
            
        section_config = self.config_sections[section]
        return section_config.parameters.get(param, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        if section not in self.config_sections:
            return {}
            
        return self.config_sections[section].parameters.copy()
    
    def set_param(self, 
                section: str, 
                param: str, 
                value: Any) -> bool:
        """Set configuration parameter at runtime."""
        if section not in self.config_sections:
            self.config_sections[section] = ConfigSection(
                name=section,
                parameters={}
            )
        
        self.config_sections[section].parameters[param] = value
        
        # Validate if schema exists
        if self.config_sections[section].schema:
            errors = self.config_sections[section].validate()
            if errors:
                for error in errors:
                    self.logger.warning(f"Config validation error in {section}: {error}")
                return False
        
        return True
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        if not file_path:
            # Use default path
            file_path = os.path.join(self.base_config_dir, "runtime.yaml")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert to serializable format
            config_data = {
                section.name: section.parameters
                for section in self.config_sections.values()
            }
            
            # Save based on file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.yaml', '.yml']:
                with open(file_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            elif file_ext == '.json':
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
            else:
                self.logger.error(f"Unsupported config file format: {file_ext}")
                return False
            
            self.logger.info(f"Saved configuration to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving config to {file_path}: {str(e)}")
            return False