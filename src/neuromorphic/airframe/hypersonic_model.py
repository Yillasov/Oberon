"""
Hypersonic parametric airframe model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any
import json


class HypersonicAirframe:
    """Hypersonic airframe with thermal-resistant neuromorphic integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'length': 5.8,     # meters
            'wingspan': 2.4,   # meters
            'body_diameter': 0.8,  # meters
            'mach_design': 8.0,  # design Mach number
            
            # Thermal characteristics
            'thermal_properties': {
                'max_skin_temp': 1800.0,  # Celsius
                'ablative_coating': True,
                'heat_shield_mass': 180.0,  # kg
                'thermal_conductivity': 0.5  # W/m-K
            },
            
            # Neuromorphic control system
            'neuromorphic_system': {
                'thermal_isolated_nodes': 5,
                'processing_capacity': 12.0,  # TOPS
                'sensor_count': 32,
                'update_rate': 2000.0,  # Hz
                'power_consumption': 45.0  # Watts
            },
            
            # Hypersonic constraints
            'neuromorphic_constraints': {
                'max_operating_temp': 125.0,  # Celsius
                'thermal_isolation_factor': 0.02,  # fraction of external temp
                'vibration_tolerance': 30.0,  # G
                'radiation_hardening': True
            }
        }
        
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        self._validate_hypersonic_constraints()
    
    def _validate_hypersonic_constraints(self) -> None:
        """Validate neuromorphic hypersonic constraints."""
        self.hypersonic_violations = []
        
        # Check thermal isolation
        max_external_temp = self.config['thermal_properties']['max_skin_temp']
        isolation_factor = self.config['neuromorphic_constraints']['thermal_isolation_factor']
        internal_temp = max_external_temp * isolation_factor
        
        if internal_temp > self.config['neuromorphic_constraints']['max_operating_temp']:
            self.hypersonic_violations.append(
                f"Thermal isolation insufficient: internal temp {internal_temp:.1f}°C exceeds max operating temp"
            )
        
        # Check update rate for hypersonic control
        min_update_rate = self.config['mach_design'] * 200  # Simplified requirement
        if self.config['neuromorphic_system']['update_rate'] < min_update_rate:
            self.hypersonic_violations.append(
                f"Update rate too low for Mach {self.config['mach_design']}: needs {min_update_rate} Hz"
            )
    
    def get_thermal_profile(self, mach: float, altitude: float) -> Dict[str, float]:
        """Calculate thermal profile at given flight conditions."""
        # Simplified thermal calculations for hypersonic flight
        # Stagnation temperature (simplified)
        ambient_temp = 288.15 - 0.0065 * altitude  # K
        stagnation_temp = ambient_temp * (1 + 0.2 * mach**2)
        
        # Skin temperature (simplified)
        skin_temp = stagnation_temp * 0.8  # Celsius
        
        # Internal temperature near neuromorphic systems
        internal_temp = skin_temp * self.config['neuromorphic_constraints']['thermal_isolation_factor']
        
        return {
            'skin_temperature': skin_temp - 273.15,  # Convert to Celsius
            'internal_temperature': internal_temp - 273.15,  # Convert to Celsius
            'temperature_margin': self.config['neuromorphic_constraints']['max_operating_temp'] - (internal_temp - 273.15),
            'cooling_required': max(0, (internal_temp - 273.15) - self.config['neuromorphic_constraints']['max_operating_temp'] + 10)
        }
    
    def get_control_requirements(self, mach: float) -> Dict[str, float]:
        """Get neuromorphic control requirements at given Mach number."""
        # Control authority decreases at hypersonic speeds
        control_authority = 1.0 / (0.5 + mach/10)
        
        # Required processing increases with Mach
        required_processing = self.config['neuromorphic_system']['processing_capacity'] * (mach / self.config['mach_design'])
        
        return {
            'control_authority': control_authority,
            'required_processing': required_processing,
            'processing_margin': self.config['neuromorphic_system']['processing_capacity'] / required_processing - 1.0,
            'required_update_rate': mach * 200,
            'update_rate_margin': self.config['neuromorphic_system']['update_rate'] / (mach * 200) - 1.0
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_hypersonic_airframe(length: float, mach_design: float) -> HypersonicAirframe:
    """Create custom hypersonic airframe model."""
    return HypersonicAirframe({
        'length': length,
        'wingspan': length * 0.4,
        'body_diameter': length * 0.14,
        'mach_design': mach_design,
        'thermal_properties': {
            'max_skin_temp': 1600 + mach_design * 50,
            'ablative_coating': mach_design > 7.0,
            'heat_shield_mass': length * 30,
            'thermal_conductivity': 0.5
        }
    })