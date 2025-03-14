"""
Parametric airframe model for neuromorphic control systems.

This module provides a simple parametric model for UCAV airframes
with specific constraints for neuromorphic hardware integration.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os


class ParametricAirframe:
    """Parametric airframe model with neuromorphic constraints."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parametric airframe model.
        
        Args:
            config: Configuration dictionary for airframe parameters
        """
        # Default configuration
        self.config = {
            # Basic airframe parameters
            'wingspan': 2.5,  # meters
            'length': 1.8,    # meters
            'wing_area': 1.2, # square meters
            'aspect_ratio': 5.2,
            'taper_ratio': 0.4,
            'sweep_angle': 30.0,  # degrees
            'dihedral_angle': 2.0, # degrees
            
            # Mass properties
            'empty_mass': 15.0,  # kg
            'max_takeoff_mass': 25.0,  # kg
            'fuel_mass': 5.0,  # kg
            
            # Neuromorphic hardware constraints
            'neuromorphic_nodes': [
                {'name': 'main_controller', 'position': [0.2, 0.0, 0.0], 'mass': 0.15, 'power': 5.0},
                {'name': 'wing_controller', 'position': [0.0, 0.5, 0.0], 'mass': 0.08, 'power': 2.5},
                {'name': 'tail_controller', 'position': [-0.8, 0.0, 0.0], 'mass': 0.08, 'power': 2.5}
            ],
            'thermal_constraints': {
                'max_temp': 85.0,  # Celsius
                'cooling_capacity': 20.0,  # Watts
                'thermal_zones': ['nose', 'wing_roots', 'tail']
            },
            'vibration_constraints': {
                'max_acceleration': 5.0,  # g
                'resonant_frequencies': [20.0, 45.0, 120.0]  # Hz
            },
            
            # Control surfaces
            'control_surfaces': {
                'elevons': {'count': 2, 'area_ratio': 0.15, 'max_deflection': 30.0},
                'rudder': {'count': 1, 'area_ratio': 0.10, 'max_deflection': 25.0}
            }
        }
        
        # Update with provided configuration
        if config:
            self._update_config(config)
        
        # Derived parameters
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
        """Calculate derived parameters from basic configuration."""
        # Wing geometry
        self.config['wing_span'] = self.config['wingspan']
        self.config['wing_chord_root'] = 2 * self.config['wing_area'] / (
            self.config['wingspan'] * (1 + self.config['taper_ratio']))
        self.config['wing_chord_tip'] = self.config['wing_chord_root'] * self.config['taper_ratio']
        
        # Inertia estimation (simplified)
        self.config['inertia'] = {
            'Ixx': self.config['empty_mass'] * (self.config['wingspan'] ** 2) / 12,
            'Iyy': self.config['empty_mass'] * (self.config['length'] ** 2) / 12,
            'Izz': self.config['empty_mass'] * (self.config['wingspan'] ** 2 + self.config['length'] ** 2) / 12
        }
        
        # Neuromorphic hardware placement validation
        self._validate_neuromorphic_placement()
    
    def _validate_neuromorphic_placement(self) -> None:
        """Validate neuromorphic hardware placement constraints."""
        # Check thermal zones
        self.thermal_violations = []
        
        # Simple thermal model - check if nodes are too close to each other
        nodes = self.config['neuromorphic_nodes']
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                pos1 = np.array(node1['position'])
                pos2 = np.array(node2['position'])
                distance = np.linalg.norm(pos1 - pos2)
                
                # If nodes are too close, potential thermal issue
                if distance < 0.2 and (node1['power'] + node2['power']) > 5.0:
                    self.thermal_violations.append(
                        f"Thermal constraint: {node1['name']} and {node2['name']} too close")
        
        # Check vibration constraints
        # In a real implementation, would use modal analysis
        self.vibration_violations = []
        wing_node = next((n for n in nodes if n['name'] == 'wing_controller'), None)
        if wing_node and abs(wing_node['position'][1]) > 0.7 * self.config['wingspan'] / 2:
            self.vibration_violations.append(
                "Vibration constraint: wing_controller too close to wingtip")
    
    def get_center_of_gravity(self) -> np.ndarray:
        """
        Calculate center of gravity based on component placement.
        
        Returns:
            Center of gravity [x, y, z] in meters from nose
        """
        total_mass = self.config['empty_mass']
        mass_moment = np.zeros(3)
        
        # Add neuromorphic hardware contribution
        for node in self.config['neuromorphic_nodes']:
            mass = node['mass']
            position = np.array(node['position'])
            mass_moment += mass * position
            total_mass += mass
        
        return mass_moment / total_mass
    
    def get_aerodynamic_properties(self, angle_of_attack: float, 
                                  sideslip: float) -> Dict[str, float]:
        """
        Get simplified aerodynamic properties at given flight condition.
        
        Args:
            angle_of_attack: Angle of attack in degrees
            sideslip: Sideslip angle in degrees
            
        Returns:
            Dictionary of aerodynamic coefficients
        """
        # Very simplified aerodynamic model
        # In a real implementation, would use proper aerodynamic modeling
        
        # Convert to radians
        aoa_rad = np.radians(angle_of_attack)
        beta_rad = np.radians(sideslip)
        
        # Simple linear model for lift coefficient
        cl = 0.1 + 0.1 * angle_of_attack  # per degree
        
        # Simple quadratic model for drag coefficient
        cd = 0.02 + 0.01 * angle_of_attack**2
        
        # Simple linear model for side force
        cy = -0.05 * sideslip  # per degree
        
        # Simple linear models for moments
        cm = -0.02 * angle_of_attack  # pitch moment
        cn = 0.05 * sideslip  # yaw moment
        cl_roll = -0.01 * sideslip  # roll moment
        
        return {
            'CL': cl,
            'CD': cd,
            'CY': cy,
            'Cm': cm,
            'Cn': cn,
            'Cl': cl_roll
        }
    
    def get_neuromorphic_constraints(self) -> Dict[str, Any]:
        """
        Get neuromorphic hardware constraints for the airframe.
        
        Returns:
            Dictionary of neuromorphic constraints
        """
        return {
            'thermal_violations': self.thermal_violations,
            'vibration_violations': self.vibration_violations,
            'power_available': 50.0,  # Watts
            'cooling_capacity': self.config['thermal_constraints']['cooling_capacity'],
            'max_temperature': self.config['thermal_constraints']['max_temp'],
            'node_positions': {node['name']: node['position'] 
                              for node in self.config['neuromorphic_nodes']}
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load_model(cls, filename: str) -> 'ParametricAirframe':
        """Load model configuration from file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        return cls(config)


def create_default_model() -> ParametricAirframe:
    """Create a default parametric airframe model."""
    return ParametricAirframe()


def create_custom_model(wingspan: float, length: float, 
                       neuromorphic_config: Dict[str, Any]) -> ParametricAirframe:
    """
    Create a custom parametric airframe model.
    
    Args:
        wingspan: Wingspan in meters
        length: Length in meters
        neuromorphic_config: Neuromorphic hardware configuration
        
    Returns:
        Customized parametric airframe model
    """
    config = {
        'wingspan': wingspan,
        'length': length,
        'wing_area': 0.5 * wingspan * wingspan / 5.0,  # Approximation
    }
    
    if neuromorphic_config:
        config['neuromorphic_nodes'] = neuromorphic_config.get('nodes', [])
        if 'thermal_constraints' in neuromorphic_config:
            config['thermal_constraints'] = neuromorphic_config['thermal_constraints']
    
    return ParametricAirframe(config)