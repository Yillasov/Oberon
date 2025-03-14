"""
Micro-UAV parametric airframe model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any
import json


class MicroUAVAirframe:
    """Micro-UAV airframe with size-constrained neuromorphic integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'wingspan': 0.25,  # meters
            'length': 0.18,    # meters
            'mass': 0.08,      # kg
            'propeller_diameter': 0.06,  # meters
            
            # Neuromorphic system
            'neuromorphic_system': {
                'processor_mass': 0.005,  # kg
                'sensor_count': 4,
                'power_consumption': 0.35,  # Watts
                'update_rate': 200.0,  # Hz
            },
            
            # Size constraints
            'size_constraints': {
                'max_component_dimension': 0.015,  # meters
                'max_processor_volume': 0.000008,  # cubic meters
                'min_trace_width': 0.0001,  # meters (PCB constraint)
                'max_power_density': 500.0  # W/kg
            }
        }
        
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        self._validate_size_constraints()
    
    def _validate_size_constraints(self) -> None:
        """Validate neuromorphic size constraints."""
        self.size_violations = []
        
        # Check power density
        power_density = self.config['neuromorphic_system']['power_consumption'] / self.config['neuromorphic_system']['processor_mass']
        if power_density > self.config['size_constraints']['max_power_density']:
            self.size_violations.append("Power density too high for micro-UAV thermal constraints")
        
        # Check mass ratio
        processor_mass_ratio = self.config['neuromorphic_system']['processor_mass'] / self.config['mass']
        if processor_mass_ratio > 0.1:
            self.size_violations.append("Neuromorphic system too heavy relative to airframe")
    
    def get_flight_performance(self) -> Dict[str, float]:
        """Get flight performance metrics."""
        # Simple flight endurance calculation (very approximate)
        battery_capacity = 0.001  # kWh (1 Wh)
        total_power = self.config['neuromorphic_system']['power_consumption'] + 1.0  # Add 1W for propulsion
        
        endurance_hours = battery_capacity / total_power
        
        return {
            'flight_endurance': endurance_hours * 60,  # minutes
            'processor_mass_ratio': self.config['neuromorphic_system']['processor_mass'] / self.config['mass'],
            'power_density': self.config['neuromorphic_system']['power_consumption'] / self.config['neuromorphic_system']['processor_mass'],
            'max_range': endurance_hours * 20  # km, assuming 20 km/h speed
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_micro_uav(wingspan: float, mass: float) -> MicroUAVAirframe:
    """Create custom micro-UAV airframe model."""
    return MicroUAVAirframe({
        'wingspan': wingspan,
        'length': wingspan * 0.7,
        'mass': mass,
        'propeller_diameter': wingspan * 0.24
    })