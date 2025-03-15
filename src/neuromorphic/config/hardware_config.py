"""
Configuration system for handling hardware-specific parameters.
"""
import json
import os
import yaml
from typing import Dict, Any, Optional, List
from enum import Enum

class HardwarePlatform(Enum):
    """Supported neuromorphic hardware platforms."""
    LOIHI = "loihi"
    SPINNAKER = "spinnaker"
    TRUENORTH = "truenorth"
    SIMULATION = "simulation"

class HardwareConfig:
    """Configuration manager for neuromorphic hardware."""
    
    def __init__(self, platform: HardwarePlatform = HardwarePlatform.SIMULATION):
        """
        Initialize hardware configuration.
        
        Args:
            platform: Target hardware platform
        """
        self.platform = platform
        self.config_data = {}
        self.default_configs = self._load_default_configs()
        
        # Load platform-specific defaults
        self._load_platform_defaults()
    
    def _load_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load default configurations for all platforms."""
        return {
            HardwarePlatform.LOIHI.value: {
                "neuron": {
                    "threshold": 100,
                    "decay": 0.9,
                    "refractory_period": 2,
                    "reset_mode": "zero"
                },
                "synapse": {
                    "weight_precision": 8,
                    "delay_precision": 4,
                    "max_delay": 15,
                    "learning_enabled": True
                },
                "chip": {
                    "cores_per_chip": 128,
                    "neurons_per_core": 1024,
                    "synapses_per_neuron": 4096
                },
                "runtime": {
                    "timestep_us": 1000,
                    "host_sync_interval": 10,
                    "spike_buffer_size": 1024
                }
            },
            HardwarePlatform.SPINNAKER.value: {
                "neuron": {
                    "threshold": 1.0,
                    "tau_m": 20.0,
                    "tau_refrac": 5.0,
                    "v_reset": 0.0,
                    "v_rest": 0.0
                },
                "synapse": {
                    "weight_precision": 16,
                    "delay_precision": 4,
                    "max_delay": 144,
                    "learning_enabled": True
                },
                "chip": {
                    "cores_per_chip": 18,
                    "neurons_per_core": 100,
                    "synapses_per_neuron": 1000
                },
                "runtime": {
                    "timestep_us": 1000,
                    "spike_buffer_size": 512,
                    "use_live_packet_gatherer": True
                }
            },
            HardwarePlatform.TRUENORTH.value: {
                "neuron": {
                    "threshold": 1.0,
                    "leak": 0,
                    "reset": 0
                },
                "synapse": {
                    "weight_precision": 1,
                    "delay_precision": 0,
                    "max_delay": 0,
                    "learning_enabled": False
                },
                "chip": {
                    "cores_per_chip": 4096,
                    "neurons_per_core": 256,
                    "synapses_per_neuron": 256
                },
                "runtime": {
                    "timestep_us": 1000,
                    "spike_buffer_size": 256
                }
            },
            HardwarePlatform.SIMULATION.value: {
                "neuron": {
                    "threshold": 1.0,
                    "decay": 0.95,
                    "refractory_period": 2,
                    "reset_mode": "subtract"
                },
                "synapse": {
                    "weight_precision": 32,
                    "delay_precision": 32,
                    "max_delay": 100,
                    "learning_enabled": True
                },
                "runtime": {
                    "timestep_us": 1000,
                    "spike_buffer_size": 4096
                }
            }
        }
    
    def _load_platform_defaults(self):
        """Load default configuration for the current platform."""
        if self.platform.value in self.default_configs:
            self.config_data = self.default_configs[self.platform.value].copy()
    
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration was loaded successfully
        """
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            return False
        
        try:
            _, ext = os.path.splitext(config_path)
            if ext.lower() == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            elif ext.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                print(f"Unsupported configuration format: {ext}")
                return False
            
            # Merge with existing configuration
            self._merge_config(loaded_config)
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """
        Merge new configuration with existing configuration.
        
        Args:
            new_config: New configuration to merge
        """
        for section, values in new_config.items():
            if section not in self.config_data:
                self.config_data[section] = {}
            
            if isinstance(values, dict):
                for key, value in values.items():
                    self.config_data[section][key] = value
            else:
                self.config_data[section] = values
    
    def save_config(self, config_path: str) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            True if configuration was saved successfully
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            _, ext = os.path.splitext(config_path)
            if ext.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config_data, f, indent=2)
            elif ext.lower() in ['.yml', '.yaml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False)
            else:
                print(f"Unsupported configuration format: {ext}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        if section in self.config_data and key in self.config_data[section]:
            return self.config_data[section][key]
        return default
    
    def set(self, section: str, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Configuration section
            
        Returns:
            Section configuration
        """
        return self.config_data.get(section, {})
    
    def get_platform_limits(self) -> Dict[str, Any]:
        """
        Get hardware platform limits.
        
        Returns:
            Dictionary of platform limits
        """
        if self.platform == HardwarePlatform.LOIHI:
            return {
                "max_neurons": 128 * 1024,  # 128 cores * 1024 neurons
                "weight_range": (-128, 127),  # 8-bit signed
                "delay_range": (0, 15),  # 4-bit unsigned
                "threshold_range": (0, 4095)  # 12-bit unsigned
            }
        elif self.platform == HardwarePlatform.SPINNAKER:
            return {
                "max_neurons": 18 * 100,  # 18 cores * 100 neurons
                "weight_range": (-32768, 32767),  # 16-bit signed
                "delay_range": (1, 144),  # 1-144 timesteps
                "threshold_range": (0, 1e6)  # Floating point
            }
        elif self.platform == HardwarePlatform.TRUENORTH:
            return {
                "max_neurons": 4096 * 256,  # 4096 cores * 256 neurons
                "weight_range": (0, 1),  # Binary weights
                "delay_range": (0, 0),  # No delays
                "threshold_range": (0, 511)  # 9-bit unsigned
            }
        else:  # Simulation
            return {
                "max_neurons": 1e6,  # Virtually unlimited
                "weight_range": (-1e6, 1e6),  # Floating point
                "delay_range": (0, 1000),  # Virtually unlimited
                "threshold_range": (0, 1e6)  # Floating point
            }
    
    def validate_network_parameters(self, network_params: Dict[str, Any]) -> List[str]:
        """
        Validate network parameters against hardware constraints.
        
        Args:
            network_params: Network parameters to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        limits = self.get_platform_limits()
        
        # Check neuron count
        if "neuron_count" in network_params:
            if network_params["neuron_count"] > limits["max_neurons"]:
                errors.append(f"Neuron count exceeds platform limit: {network_params['neuron_count']} > {limits['max_neurons']}")
        
        # Check weights
        if "weights" in network_params and isinstance(network_params["weights"], list):
            min_weight = min(network_params["weights"])
            max_weight = max(network_params["weights"])
            
            if min_weight < limits["weight_range"][0] or max_weight > limits["weight_range"][1]:
                errors.append(f"Weight range exceeds platform limits: [{min_weight}, {max_weight}] not in {limits['weight_range']}")
        
        # Check thresholds
        if "thresholds" in network_params and isinstance(network_params["thresholds"], list):
            min_threshold = min(network_params["thresholds"])
            max_threshold = max(network_params["thresholds"])
            
            if min_threshold < limits["threshold_range"][0] or max_threshold > limits["threshold_range"][1]:
                errors.append(f"Threshold range exceeds platform limits: [{min_threshold}, {max_threshold}] not in {limits['threshold_range']}")
        
        return errors