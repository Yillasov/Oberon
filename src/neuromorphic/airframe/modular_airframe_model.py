"""
Modular Airframe parametric model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import json


class ModularAirframe:
    """Modular airframe with reconfigurable components and neuromorphic adaptation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'core_length': 8.5,      # meters
            'core_diameter': 1.8,    # meters
            'max_modules': 6,        # maximum number of attachable modules
            'current_modules': 4,    # currently attached modules
            
            # Module specifications
            'modules': {
                'wing_modules': {
                    'count': 2,
                    'span_per_module': 3.2,  # meters
                    'chord': 1.4,            # meters
                    'attachment_points': 4   # attachment points per module
                },
                'control_modules': {
                    'count': 1,
                    'control_surfaces': 6,    # control surfaces per module
                    'actuators_per_surface': 2
                },
                'payload_modules': {
                    'count': 1,
                    'volume': 2.8,           # cubic meters
                    'max_mass': 450          # kg
                }
            },
            
            # Neuromorphic control system
            'neuromorphic_system': {
                'core_processors': 3,
                'module_processors': 1,      # processors per module
                'interface_bandwidth': 1.2,  # Gbps between modules
                'sensors_per_module': 12,
                'update_rate': 180.0,        # Hz
                'power_consumption': 42.0,   # Watts
                'reconfiguration_capability': True
            },
            
            # Neuromorphic constraints
            'neuromorphic_constraints': {
                'max_module_latency': 0.008,  # seconds
                'min_interface_reliability': 0.9995,  # reliability factor
                'max_reconfiguration_time': 0.5,  # seconds
                'min_redundancy_factor': 1.5,
                'power_per_module': 8.0  # Watts
            }
        }
        
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        self._calculate_derived_parameters()
        self._validate_constraints()
    
    def _calculate_derived_parameters(self) -> None:
        """Calculate derived parameters."""
        # Calculate total number of modules
        self.total_modules = sum(module['count'] for module in self.config['modules'].values())
        
        # Calculate total wing area
        wing_modules = self.config['modules']['wing_modules']
        self.wing_area = wing_modules['count'] * wing_modules['span_per_module'] * wing_modules['chord']
        
        # Calculate total processors
        self.total_processors = (self.config['neuromorphic_system']['core_processors'] + 
                                self.config['neuromorphic_system']['module_processors'] * self.total_modules)
        
        # Calculate total sensors
        self.total_sensors = self.config['neuromorphic_system']['sensors_per_module'] * self.total_modules
        
        # Calculate module latency
        base_latency = 1.0 / self.config['neuromorphic_system']['update_rate']
        interface_latency = 0.001 * self.total_modules  # 1ms per module for communication
        self.module_latency = base_latency + interface_latency
        
        # Calculate redundancy factor
        self.redundancy_factor = self.total_processors / max(1, self.total_modules)
        
        # Calculate total power consumption
        self.total_power = (self.config['neuromorphic_system']['power_consumption'] + 
                           self.config['neuromorphic_constraints']['power_per_module'] * self.total_modules)
    
    def _validate_constraints(self) -> None:
        """Validate neuromorphic modular constraints."""
        self.violations = []
        
        # Check module latency
        if self.module_latency > self.config['neuromorphic_constraints']['max_module_latency']:
            self.violations.append(
                f"Excessive module latency: {self.module_latency*1000:.2f}ms > "
                f"{self.config['neuromorphic_constraints']['max_module_latency']*1000:.2f}ms maximum"
            )
        
        # Check redundancy factor
        if self.redundancy_factor < self.config['neuromorphic_constraints']['min_redundancy_factor']:
            self.violations.append(
                f"Insufficient processing redundancy: {self.redundancy_factor:.2f} < "
                f"{self.config['neuromorphic_constraints']['min_redundancy_factor']:.2f} required"
            )
        
        # Check if total modules exceeds maximum
        if self.total_modules > self.config['max_modules']:
            self.violations.append(
                f"Too many modules: {self.total_modules} > {self.config['max_modules']} maximum"
            )
        
        # Check power requirements
        max_power = (self.config['neuromorphic_system']['power_consumption'] * 1.5)  # 50% margin
        if self.total_power > max_power:
            self.violations.append(
                f"Power requirement too high: {self.total_power:.1f}W > {max_power:.1f}W available"
            )
    
    def reconfigure(self, new_configuration: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate reconfiguration to a new module setup.
        
        Args:
            new_configuration: Dictionary with new module counts
        """
        # Store original configuration
        original_config = {
            'wing_modules': self.config['modules']['wing_modules']['count'],
            'control_modules': self.config['modules']['control_modules']['count'],
            'payload_modules': self.config['modules']['payload_modules']['count']
        }
        
        # Apply new configuration temporarily
        for module_type, count in new_configuration.items():
            if module_type in self.config['modules']:
                self.config['modules'][module_type]['count'] = count
        
        # Recalculate parameters
        self._calculate_derived_parameters()
        self._validate_constraints()
        
        # Calculate reconfiguration metrics
        modules_changed = sum(abs(new_configuration.get(k, 0) - original_config.get(k, 0)) 
                             for k in set(new_configuration) | set(original_config))
        
        reconfiguration_time = modules_changed * 0.1  # 0.1 seconds per module change
        
        # Check if reconfiguration is valid
        is_valid = len(self.violations) == 0
        
        # Calculate stability impact
        stability_impact = modules_changed * 0.15  # 15% stability reduction per module changed
        
        # Calculate adaptation time
        adaptation_time = reconfiguration_time + modules_changed * 0.2  # base + adaptation time
        
        # Calculate neuromorphic load during reconfiguration
        neuromorphic_load = 0.4 + modules_changed * 0.1
        
        # Restore original configuration
        for module_type, count in original_config.items():
            self.config['modules'][module_type]['count'] = count
        
        self._calculate_derived_parameters()
        
        return {
            'is_valid': is_valid,
            'violations': self.violations.copy() if not is_valid else [],
            'reconfiguration_time': reconfiguration_time,
            'adaptation_time': adaptation_time,
            'stability_impact': stability_impact,
            'neuromorphic_load': min(1.0, neuromorphic_load),
            'total_modules_after': sum(new_configuration.values())
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for current configuration."""
        # Calculate wing loading
        wing_loading = 2000.0 / max(0.1, self.wing_area)  # Assuming 2000kg base mass
        
        # Calculate control authority
        control_modules = self.config['modules']['control_modules']
        control_surfaces = control_modules['count'] * control_modules['control_surfaces']
        control_authority = 0.6 + 0.4 * (control_surfaces / 12)  # Normalized to 12 surfaces
        
        # Calculate system responsiveness
        responsiveness = min(1.0, 0.01 / max(0.001, self.module_latency))
        
        # Calculate adaptability score
        if self.config['neuromorphic_system']['reconfiguration_capability']:
            adaptability = 0.7 + 0.3 * (self.redundancy_factor / 
                                       self.config['neuromorphic_constraints']['min_redundancy_factor'])
        else:
            adaptability = 0.3
        
        # Calculate payload efficiency
        payload_modules = self.config['modules']['payload_modules']
        payload_capacity = payload_modules['count'] * payload_modules['max_mass']
        payload_efficiency = payload_capacity / (2000.0 + payload_capacity)  # payload fraction
        
        return {
            'wing_loading': wing_loading,
            'control_authority': control_authority,
            'system_responsiveness': responsiveness,
            'adaptability': adaptability,
            'payload_efficiency': payload_efficiency,
            'neuromorphic_utilization': self.total_processors / (self.total_modules * 2)
        }
    
    def estimate_flight_characteristics(self, speed: float) -> Dict[str, float]:
        """Estimate flight characteristics at given airspeed."""
        # Get performance metrics
        metrics = self.get_performance_metrics()
        
        # Calculate lift coefficient (simplified)
        cl = 0.3 + 0.7 / metrics['wing_loading']
        
        # Calculate drag coefficient (simplified)
        cd = 0.02 + cl**2 / (3.14 * 4.0)  # Assuming AR of 4.0
        
        # Calculate turn rate
        turn_rate = 9.81 * np.sqrt(metrics['control_authority']) / speed
        
        # Calculate roll rate
        wing_modules = self.config['modules']['wing_modules']
        wingspan = wing_modules['count'] * wing_modules['span_per_module']
        roll_rate = 180 * metrics['control_authority'] / wingspan
        
        # Calculate neuromorphic benefit
        neuromorphic_benefit = 0.2 * metrics['system_responsiveness'] * self.redundancy_factor
        
        return {
            'lift_coefficient': cl,
            'drag_coefficient': cd,
            'lift_to_drag': cl / cd,
            'turn_rate': turn_rate,  # rad/s
            'roll_rate': roll_rate,  # deg/s
            'neuromorphic_benefit': neuromorphic_benefit,
            'reconfiguration_penalty': 0.0  # No penalty in stable configuration
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_modular_airframe(core_length: float, modules: Dict[str, int]) -> ModularAirframe:
    """Create custom modular airframe model."""
    return ModularAirframe({
        'core_length': core_length,
        'core_diameter': core_length / 4.5,
        'max_modules': 8,
        'current_modules': sum(modules.values()),
        'modules': {
            'wing_modules': {
                'count': modules.get('wing_modules', 2),
                'span_per_module': core_length / 2.5,
                'chord': core_length / 6.0
            },
            'control_modules': {
                'count': modules.get('control_modules', 1),
                'control_surfaces': 6
            },
            'payload_modules': {
                'count': modules.get('payload_modules', 1),
                'volume': core_length * 0.3,
                'max_mass': core_length * 50
            }
        },
        'neuromorphic_system': {
            'core_processors': max(2, int(core_length / 3)),
            'module_processors': 1,
            'sensors_per_module': max(8, int(core_length * 1.5))
        }
    })