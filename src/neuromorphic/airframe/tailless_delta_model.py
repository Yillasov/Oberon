"""
Tailless Delta parametric airframe model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any
import json


class TaillessDeltaAirframe:
    """Tailless delta airframe with neuromorphic stability augmentation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'wingspan': 8.4,     # meters
            'root_chord': 6.2,   # meters
            'tip_chord': 0.8,    # meters
            'sweep_angle': 52.0, # degrees
            'wing_area': 28.0,   # square meters
            
            # Aerodynamic features
            'aero_features': {
                'elevons': True,
                'leading_edge_extensions': True,
                'vortex_generators': True,
                'winglets': False
            },
            
            # Neuromorphic stability system
            'stability_system': {
                'distributed_sensors': 18,
                'processing_nodes': 5,
                'update_rate': 800.0,  # Hz
                'power_consumption': 22.0,  # Watts
                'inherent_stability': -8.0  # percent (negative means unstable)
            },
            
            # Neuromorphic constraints
            'neuromorphic_constraints': {
                'max_latency': 0.004,  # seconds
                'min_sensor_density': 0.6,  # sensors per square meter
                'processing_redundancy': 2.5,  # redundancy factor
                'power_density_limit': 4.0  # W/kg
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
        # Calculate aspect ratio
        self.aspect_ratio = self.config['wingspan']**2 / self.config['wing_area']
        
        # Calculate taper ratio
        self.taper_ratio = self.config['tip_chord'] / self.config['root_chord']
        
        # Calculate mean aerodynamic chord
        self.mac = (2/3) * self.config['root_chord'] * (
            (1 + self.taper_ratio + self.taper_ratio**2) / (1 + self.taper_ratio))
        
        # Calculate sensor density
        self.sensor_density = self.config['stability_system']['distributed_sensors'] / self.config['wing_area']
        
        # Calculate system latency
        self.system_latency = 1.0 / self.config['stability_system']['update_rate'] + 0.001  # processing + sensor latency
    
    def _validate_constraints(self) -> None:
        """Validate neuromorphic constraints."""
        self.violations = []
        
        # Check sensor density
        if self.sensor_density < self.config['neuromorphic_constraints']['min_sensor_density']:
            self.violations.append(
                f"Insufficient sensor density: {self.sensor_density:.2f} sensors/m² < "
                f"{self.config['neuromorphic_constraints']['min_sensor_density']:.2f} required"
            )
        
        # Check system latency
        if self.system_latency > self.config['neuromorphic_constraints']['max_latency']:
            self.violations.append(
                f"Excessive system latency: {self.system_latency*1000:.2f}ms > "
                f"{self.config['neuromorphic_constraints']['max_latency']*1000:.2f}ms maximum"
            )
        
        # Check processing redundancy for unstable airframe
        required_redundancy = abs(self.config['stability_system']['inherent_stability']) / 5.0
        if self.config['neuromorphic_constraints']['processing_redundancy'] < required_redundancy:
            self.violations.append(
                f"Insufficient processing redundancy for unstable configuration: "
                f"{self.config['neuromorphic_constraints']['processing_redundancy']:.1f} < {required_redundancy:.1f} required"
            )
    
    def get_stability_metrics(self, speed: float) -> Dict[str, float]:
        """Calculate stability metrics at given airspeed."""
        # Dynamic pressure
        q = 0.5 * 1.225 * speed**2
        
        # Control effectiveness increases with dynamic pressure
        control_effectiveness = 1.0 - np.exp(-q / 5000)
        
        # Required update rate increases with speed for unstable configuration
        required_update_rate = 200 + speed * abs(self.config['stability_system']['inherent_stability']) / 10
        
        # Stability margin
        stability_margin = (self.config['stability_system']['update_rate'] / required_update_rate - 1.0) * 100
        
        return {
            'control_effectiveness': control_effectiveness,
            'required_update_rate': required_update_rate,
            'stability_margin': stability_margin,
            'sensor_coverage': self.sensor_density * self.config['wing_area'] / 
                              (self.config['wingspan'] + self.config['root_chord']),
            'neuromorphic_limited': required_update_rate > self.config['stability_system']['update_rate'] * 0.8
        }
    
    def get_maneuverability_metrics(self) -> Dict[str, float]:
        """Calculate maneuverability metrics."""
        # Simplified maneuverability calculations
        roll_rate = 120.0 * (1.0 - 0.1 * self.aspect_ratio)  # deg/s
        
        # Pitch rate depends on stability system
        pitch_rate = 40.0 * (1.0 + abs(self.config['stability_system']['inherent_stability']) / 10.0)
        
        # Turn rate depends on wing loading and aspect ratio
        wing_loading = 5000.0 / self.config['wing_area']  # Assumed mass of 5000kg
        turn_rate = 20.0 * (1.0 - wing_loading/300) * (1.0 - self.aspect_ratio/10)
        
        return {
            'max_roll_rate': roll_rate,  # deg/s
            'max_pitch_rate': pitch_rate,  # deg/s
            'max_turn_rate': turn_rate,  # deg/s
            'neuromorphic_augmentation': abs(self.config['stability_system']['inherent_stability']) / 10.0
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_tailless_delta(wingspan: float, sweep_angle: float, stability: float) -> TaillessDeltaAirframe:
    """Create custom tailless delta airframe model."""
    # Calculate root chord based on sweep angle and wingspan
    root_chord = wingspan / (2 * np.tan((90 - sweep_angle) * np.pi / 180))
    
    # Calculate approximate wing area
    wing_area = wingspan * (root_chord + wingspan/10) / 2
    
    return TaillessDeltaAirframe({
        'wingspan': wingspan,
        'root_chord': root_chord,
        'tip_chord': wingspan/10,
        'sweep_angle': sweep_angle,
        'wing_area': wing_area,
        'stability_system': {
            'inherent_stability': stability,
            'update_rate': 600 + abs(stability) * 50,
            'distributed_sensors': int(wing_area * 0.7)
        }
    })