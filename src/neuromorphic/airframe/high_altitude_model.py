"""
High-Altitude parametric airframe model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any
import json


class HighAltitudeAirframe:
    """High-altitude tailless swept-wing airframe with neuromorphic adaptation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'wingspan': 18.0,  # meters
            'wing_area': 32.0,  # square meters
            'aspect_ratio': 10.1,  # high for efficiency
            'sweep_angle': 15.0,  # degrees
            'service_ceiling': 20000,  # meters
            
            # Neuromorphic adaptation system
            'adaptation_system': {
                'air_density_sensors': 6,
                'temperature_sensors': 4,
                'adaptation_rate': 10.0,  # Hz
                'power_consumption': 3.5,  # Watts
            },
            
            # Neuromorphic constraints
            'neuromorphic_constraints': {
                'min_operating_temp': -70.0,  # Celsius
                'max_operating_temp': 50.0,  # Celsius
                'radiation_tolerance': 'high',  # cosmic radiation at altitude
                'power_scaling': 0.8  # power reduction at altitude
            }
        }
        
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        self._validate_altitude_constraints()
    
    def _validate_altitude_constraints(self) -> None:
        """Validate neuromorphic altitude constraints."""
        self.altitude_violations = []
        
        # Check temperature constraints
        if self.config['neuromorphic_constraints']['min_operating_temp'] > -60:
            self.altitude_violations.append("Minimum temperature rating insufficient for high altitude")
        
        # Check wing loading for high altitude
        wing_loading = 250.0 / self.config['wing_area']  # Assumed mass of 250kg
        if wing_loading > 15:
            self.altitude_violations.append("Wing loading too high for efficient high-altitude operation")
    
    def get_altitude_performance(self, altitude: float) -> Dict[str, float]:
        """Get performance metrics at specified altitude."""
        # Air density ratio (simplified)
        density_ratio = np.exp(-altitude / 7000)
        
        # Neuromorphic power adjustment with altitude
        power_factor = 1.0 - (1.0 - self.config['neuromorphic_constraints']['power_scaling']) * (
            altitude / self.config['service_ceiling'])
        
        return {
            'air_density_ratio': density_ratio,
            'neuromorphic_power': self.config['adaptation_system']['power_consumption'] * power_factor,
            'adaptation_rate': self.config['adaptation_system']['adaptation_rate'] * power_factor,
            'temperature': 15.0 - (altitude * 0.0065)  # Standard atmosphere approximation
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_high_altitude_airframe(wingspan: float, ceiling: float) -> HighAltitudeAirframe:
    """Create custom high-altitude airframe model."""
    return HighAltitudeAirframe({
        'wingspan': wingspan,
        'wing_area': wingspan**2 / 10.1,  # Maintain aspect ratio
        'service_ceiling': ceiling
    })