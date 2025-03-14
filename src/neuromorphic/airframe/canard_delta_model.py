"""
Canard-Delta parametric airframe model for neuromorphic control systems.

This module provides a parametric model for highly maneuverable canard-delta UCAV airframes
with neuromorphic control for inherently unstable configurations.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json


class CanardDeltaAirframe:
    """Canard-Delta airframe model with neuromorphic stability augmentation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic airframe parameters
            'wingspan': 3.6,  # meters
            'length': 4.2,    # meters
            'wing_area': 4.8, # square meters
            'canard_area': 0.6, # square meters
            'canard_position': -3.2,  # meters from CG (negative is forward)
            'delta_sweep': 55.0,  # degrees
            'canard_sweep': 40.0,  # degrees
            
            # Mass properties
            'empty_mass': 180.0,  # kg
            'max_takeoff_mass': 320.0,  # kg
            'fuel_mass': 100.0,  # kg
            
            # Neuromorphic stability system
            'stability_system': {
                'response_time': 0.002,  # seconds
                'update_rate': 1000.0,  # Hz
                'sensor_latency': 0.001,  # seconds
                'actuator_latency': 0.005,  # seconds
                'stability_margin': -15.0,  # percent (negative means unstable)
                'redundancy_level': 3  # triple redundancy
            },
            
            # Neuromorphic processing nodes
            'processing_nodes': [
                {
                    'name': 'forward_stability',
                    'position': [-2.8, 0.0, 0.0],  # near canard
                    'processing_capacity': 4.0,  # TOPS
                    'power': 3.5,  # Watts
                    'control_surfaces': ['canard_left', 'canard_right']
                },
                {
                    'name': 'main_stability',
                    'position': [0.0, 0.0, 0.0],  # at CG
                    'processing_capacity': 8.0,  # TOPS
                    'power': 6.0,  # Watts
                    'control_surfaces': ['elevon_left_inner', 'elevon_right_inner']
                },
                {
                    'name': 'lateral_stability',
                    'position': [0.5, 0.0, 0.0],  # aft of CG
                    'processing_capacity': 4.0,  # TOPS
                    'power': 3.5,  # Watts
                    'control_surfaces': ['elevon_left_outer', 'elevon_right_outer']
                }
            ],
            
            # High-speed maneuverability constraints
            'maneuverability': {
                'max_g_load': 9.0,  # g
                'max_pitch_rate': 60.0,  # deg/s
                'max_roll_rate': 180.0,  # deg/s
                'max_yaw_rate': 40.0,  # deg/s
                'min_turn_radius': 150.0  # meters
            },
            
            # Control surfaces
            'control_surfaces': {
                'canard_left': {'area': 0.3, 'max_deflection': 30.0, 'rate_limit': 80.0},
                'canard_right': {'area': 0.3, 'max_deflection': 30.0, 'rate_limit': 80.0},
                'elevon_left_inner': {'area': 0.4, 'max_deflection': 25.0, 'rate_limit': 60.0},
                'elevon_right_inner': {'area': 0.4, 'max_deflection': 25.0, 'rate_limit': 60.0},
                'elevon_left_outer': {'area': 0.3, 'max_deflection': 35.0, 'rate_limit': 70.0},
                'elevon_right_outer': {'area': 0.3, 'max_deflection': 35.0, 'rate_limit': 70.0}
            }
        }
        
        if config:
            self._update_config(config)
        self._calculate_derived_parameters()
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration with provided values."""
        for key, value in config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
    
    def _calculate_derived_parameters(self) -> None:
        """Calculate derived parameters and validate constraints."""
        # Calculate wing loading
        self.wing_loading = self.config['max_takeoff_mass'] / self.config['wing_area']
        
        # Calculate aspect ratio
        self.aspect_ratio = self.config['wingspan']**2 / self.config['wing_area']
        
        # Calculate canard volume coefficient
        self.canard_volume = (self.config['canard_area'] * abs(self.config['canard_position'])) / (
            self.config['wing_area'] * self.config['length'])
        
        # Calculate total processing power
        self.processing_power = sum(node['power'] for node in self.config['processing_nodes'])
        
        # Calculate total processing capacity
        self.processing_capacity = sum(
            node['processing_capacity'] for node in self.config['processing_nodes'])
        
        # Validate neuromorphic stability constraints
        self._validate_stability_constraints()
    
    def _validate_stability_constraints(self) -> None:
        """Validate neuromorphic stability system constraints."""
        self.stability_violations = []
        
        # Check if processing nodes can handle the required update rate
        required_ops = self.config['stability_system']['update_rate'] * 0.5  # TOPS per 1000Hz
        if self.processing_capacity < required_ops:
            self.stability_violations.append(
                f"Insufficient processing capacity: {self.processing_capacity:.1f} TOPS < {required_ops:.1f} TOPS required"
            )
        
        # Check total system latency
        sensor_latency = self.config['stability_system']['sensor_latency']
        processing_latency = 1.0 / self.config['stability_system']['update_rate']
        actuator_latency = self.config['stability_system']['actuator_latency']
        total_latency = sensor_latency + processing_latency + actuator_latency
        
        # For unstable aircraft, total latency must be very low
        max_allowed_latency = 0.01  # 10ms for highly unstable aircraft
        if total_latency > max_allowed_latency:
            self.stability_violations.append(
                f"Excessive control latency: {total_latency*1000:.1f}ms > {max_allowed_latency*1000:.1f}ms maximum"
            )
        
        # Check if canard volume coefficient is sufficient for stability
        min_canard_volume = 0.10
        if self.canard_volume < min_canard_volume:
            self.stability_violations.append(
                f"Insufficient canard volume: {self.canard_volume:.3f} < {min_canard_volume:.3f} minimum"
            )
    
    def get_stability_margins(self, speed: float) -> Dict[str, float]:
        """
        Calculate stability margins at given airspeed.
        
        Args:
            speed: Airspeed in m/s
            
        Returns:
            Dictionary of stability margins
        """
        # Simplified stability calculation for canard-delta configuration
        # In reality, would use proper aerodynamic modeling
        
        # Static margin varies with speed for canard configuration
        # Typically becomes more unstable at higher speeds
        base_margin = self.config['stability_system']['stability_margin']
        speed_factor = 1.0 + (speed - 100) / 500  # Adjust based on speed
        static_margin = base_margin * speed_factor
        
        # Calculate control power available
        # Simplified - proportional to dynamic pressure and control surface area
        q = 0.5 * 1.225 * speed**2  # Dynamic pressure
        canard_control_power = q * self.config['canard_area'] * 0.8  # Simplified coefficient
        elevon_control_power = q * (
            self.config['control_surfaces']['elevon_left_inner']['area'] +
            self.config['control_surfaces']['elevon_right_inner']['area'] +
            self.config['control_surfaces']['elevon_left_outer']['area'] +
            self.config['control_surfaces']['elevon_right_outer']['area']
        ) * 0.6  # Simplified coefficient
        
        # Calculate required neuromorphic control rate
        # Higher speeds and more negative static margins require faster control
        required_update_rate = 200 + abs(static_margin) * 20 + speed / 2
        
        return {
            'static_margin': static_margin,
            'canard_control_power': canard_control_power,
            'elevon_control_power': elevon_control_power,
            'required_update_rate': required_update_rate,
            'current_update_rate': self.config['stability_system']['update_rate'],
            'control_margin': min(
                self.config['stability_system']['update_rate'] / required_update_rate - 1.0,
                canard_control_power / (abs(static_margin) * 100) - 1.0
            ) * 100  # percentage
        }
    
    def get_neuromorphic_requirements(self) -> Dict[str, Any]:
        """Get neuromorphic system requirements for stability augmentation."""
        return {
            'violations': self.stability_violations,
            'processing_metrics': {
                'total_capacity': self.processing_capacity,
                'total_power': self.processing_power,
                'update_rate': self.config['stability_system']['update_rate'],
                'response_time': self.config['stability_system']['response_time']
            },
            'stability_metrics': {
                'static_margin': self.config['stability_system']['stability_margin'],
                'canard_volume': self.canard_volume,
                'redundancy_level': self.config['stability_system']['redundancy_level']
            },
            'control_distribution': {
                node['name']: node['control_surfaces'] 
                for node in self.config['processing_nodes']
            }
        }
    
    def estimate_maneuverability(self, speed: float) -> Dict[str, float]:
        """
        Estimate maneuverability metrics at given airspeed.
        
        Args:
            speed: Airspeed in m/s
            
        Returns:
            Dictionary of maneuverability metrics
        """
        # Dynamic pressure
        q = 0.5 * 1.225 * speed**2
        
        # Turn rate and radius (simplified)
        max_g = min(self.config['maneuverability']['max_g_load'], 
                   q * self.config['wing_area'] * 1.2 / (self.config['max_takeoff_mass'] * 9.81))
        turn_rate = np.sqrt((max_g**2 - 1) * 9.81**2) / speed  # rad/s
        turn_radius = speed**2 / (9.81 * np.sqrt(max_g**2 - 1))
        
        # Maximum rates based on control power (simplified)
        max_pitch_rate = min(
            self.config['maneuverability']['max_pitch_rate'],
            q * self.config['canard_area'] * 1.5 * 180 / (np.pi * self.config['max_takeoff_mass'])
        )
        
        max_roll_rate = min(
            self.config['maneuverability']['max_roll_rate'],
            q * (self.config['control_surfaces']['elevon_left_outer']['area'] * 2) * 
            3.0 * 180 / (np.pi * self.config['max_takeoff_mass'])
        )
        
        return {
            'turn_rate': turn_rate * 180 / np.pi,  # deg/s
            'turn_radius': turn_radius,  # meters
            'max_g': max_g,
            'max_pitch_rate': max_pitch_rate,  # deg/s
            'max_roll_rate': max_roll_rate,  # deg/s
            'neuromorphic_limited': self.config['stability_system']['update_rate'] < 
                                   (max_pitch_rate * 15)  # True if neuromorphic system limits performance
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load_model(cls, filename: str) -> 'CanardDeltaAirframe':
        """Load model configuration from file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        return cls(config)


def create_default_canard_delta() -> CanardDeltaAirframe:
    """Create default canard-delta airframe model."""
    return CanardDeltaAirframe()


def create_custom_canard_delta(wingspan: float, length: float,
                              stability_margin: float,
                              neuromorphic_config: Dict[str, Any]) -> CanardDeltaAirframe:
    """
    Create custom canard-delta airframe model.
    
    Args:
        wingspan: Wingspan in meters
        length: Length in meters
        stability_margin: Static stability margin (negative for unstable)
        neuromorphic_config: Neuromorphic system configuration
        
    Returns:
        Customized canard-delta airframe model
    """
    config = {
        'wingspan': wingspan,
        'length': length,
        'wing_area': wingspan**2 / 2.7,  # Approximation for delta wing
        'canard_area': wingspan**2 / 20.0,  # Typical canard sizing
        'canard_position': -length * 0.75  # Canard position from CG
    }
    
    if 'stability_system' not in neuromorphic_config:
        neuromorphic_config['stability_system'] = {}
    
    neuromorphic_config['stability_system']['stability_margin'] = stability_margin
    
    for key, value in neuromorphic_config.items():
        config[key] = value
    
    return CanardDeltaAirframe(config)