"""
Flying Wing parametric airframe model for neuromorphic control systems.

This module provides a parametric model for flying wing UCAV airframes
with distributed neuromorphic processing constraints.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os


class FlyingWingAirframe:
    """Flying wing airframe model with distributed neuromorphic constraints."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize flying wing airframe model.
        
        Args:
            config: Configuration dictionary for airframe parameters
        """
        # Default configuration
        self.config = {
            # Basic airframe parameters
            'wingspan': 3.2,  # meters
            'root_chord': 1.2,  # meters
            'tip_chord': 0.4,  # meters
            'sweep_angle': 35.0,  # degrees
            'twist_angle': -2.0,  # degrees (washout)
            'thickness_ratio': 0.12,  # t/c ratio
            
            # Mass properties
            'empty_mass': 18.0,  # kg
            'max_takeoff_mass': 30.0,  # kg
            'fuel_mass': 7.0,  # kg
            
            # Distributed neuromorphic hardware
            'neuromorphic_nodes': [
                {'name': 'central_processor', 'position': [0.0, 0.0, 0.0], 'mass': 0.2, 'power': 8.0},
                {'name': 'left_wing_inner', 'position': [0.1, -0.8, 0.0], 'mass': 0.05, 'power': 1.5},
                {'name': 'right_wing_inner', 'position': [0.1, 0.8, 0.0], 'mass': 0.05, 'power': 1.5},
                {'name': 'left_wing_outer', 'position': [0.0, -1.4, 0.0], 'mass': 0.05, 'power': 1.5},
                {'name': 'right_wing_outer', 'position': [0.0, 1.4, 0.0], 'mass': 0.05, 'power': 1.5}
            ],
            
            # EMI constraints for neuromorphic hardware
            'emi_constraints': {
                'max_field_strength': 3.0,  # V/m
                'sensitive_frequencies': [2.4, 5.0, 1.5],  # GHz
                'shielding_zones': ['central', 'wing_roots']
            },
            
            # Power distribution constraints
            'power_constraints': {
                'max_power': 60.0,  # Watts
                'battery_capacity': 150.0,  # Watt-hours
                'distribution_efficiency': 0.92,
                'max_voltage': 12.0  # Volts
            },
            
            # Control surfaces
            'control_surfaces': {
                'elevons': {'count': 4, 'area_ratio': 0.18, 'max_deflection': 25.0},
                'split_rudders': {'count': 2, 'area_ratio': 0.08, 'max_deflection': 45.0}
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
        self.config['wing_area'] = 0.5 * self.config['wingspan'] * (
            self.config['root_chord'] + self.config['tip_chord'])
        self.config['aspect_ratio'] = self.config['wingspan']**2 / self.config['wing_area']
        self.config['taper_ratio'] = self.config['tip_chord'] / self.config['root_chord']
        
        # Mean aerodynamic chord
        self.config['mac'] = (2/3) * self.config['root_chord'] * (
            1 + self.config['taper_ratio'] + self.config['taper_ratio']**2) / (
            1 + self.config['taper_ratio'])
        
        # Inertia estimation (simplified)
        span_sq = self.config['wingspan']**2
        chord_sq = self.config['root_chord']**2
        mass = self.config['empty_mass']
        
        self.config['inertia'] = {
            'Ixx': mass * span_sq / 12,  # Roll inertia
            'Iyy': mass * (chord_sq + span_sq/4) / 12,  # Pitch inertia
            'Izz': mass * (chord_sq + span_sq) / 12  # Yaw inertia
        }
        
        # Neuromorphic hardware validation
        self._validate_neuromorphic_distribution()
    
    def _validate_neuromorphic_distribution(self) -> None:
        """Validate distributed neuromorphic hardware constraints."""
        # Check EMI constraints
        self.emi_violations = []
        
        # Check node spacing for EMI interference
        nodes = self.config['neuromorphic_nodes']
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                pos1 = np.array(node1['position'])
                pos2 = np.array(node2['position'])
                distance = np.linalg.norm(pos1 - pos2)
                
                # Simple EMI model - nodes too close may interfere
                if distance < 0.3 and (node1['power'] > 3.0 or node2['power'] > 3.0):
                    self.emi_violations.append(
                        f"EMI constraint: {node1['name']} and {node2['name']} may interfere")
        
        # Check power distribution
        self.power_violations = []
        total_power = sum(node['power'] for node in nodes)
        
        if total_power > self.config['power_constraints']['max_power']:
            self.power_violations.append(
                f"Power constraint: Total power {total_power}W exceeds maximum {self.config['power_constraints']['max_power']}W")
        
        # Check wing loading for distributed nodes
        wing_loading = self.config['max_takeoff_mass'] / self.config['wing_area']
        if wing_loading > 50:  # kg/m²
            self.power_violations.append(
                f"Wing loading too high: {wing_loading:.1f} kg/m² may affect distributed sensing")
    
    def get_center_of_gravity(self) -> np.ndarray:
        """
        Calculate center of gravity based on component placement.
        
        Returns:
            Center of gravity [x, y, z] in meters from reference point
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
        # Flying wing aerodynamic model (simplified)
        
        # Convert to radians
        aoa_rad = np.radians(angle_of_attack)
        beta_rad = np.radians(sideslip)
        
        # Flying wing lift model with reflex airfoil
        cl_alpha = 0.08  # per degree
        cl_0 = 0.15
        cl = cl_0 + cl_alpha * angle_of_attack
        
        # Flying wing drag model
        cd_0 = 0.015
        k = 0.04
        cd = cd_0 + k * cl**2
        
        # Side force
        cy = -0.04 * sideslip
        
        # Moment coefficients for flying wing
        cm_0 = 0.02  # Positive for reflex airfoil
        cm_alpha = -0.01  # per degree
        cm = cm_0 + cm_alpha * angle_of_attack
        
        cn = 0.04 * sideslip  # yaw moment
        cl_roll = -0.015 * sideslip  # roll moment
        
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
        Get distributed neuromorphic hardware constraints.
        
        Returns:
            Dictionary of neuromorphic constraints
        """
        return {
            'emi_violations': self.emi_violations,
            'power_violations': self.power_violations,
            'power_available': self.config['power_constraints']['max_power'],
            'battery_capacity': self.config['power_constraints']['battery_capacity'],
            'node_positions': {node['name']: node['position'] 
                              for node in self.config['neuromorphic_nodes']},
            'communication_latency': {
                'central_to_wing': 0.5,  # ms
                'wing_to_wing': 1.2,  # ms
                'total_processing': 5.0  # ms
            }
        }
    
    def estimate_flight_endurance(self) -> float:
        """
        Estimate flight endurance based on power consumption.
        
        Returns:
            Estimated flight endurance in minutes
        """
        total_power = sum(node['power'] for node in self.config['neuromorphic_nodes'])
        # Add estimated propulsion power (simplified)
        propulsion_power = 200.0  # Watts
        total_power += propulsion_power
        
        # Calculate endurance
        battery_capacity = self.config['power_constraints']['battery_capacity']
        efficiency = self.config['power_constraints']['distribution_efficiency']
        
        endurance_hours = (battery_capacity * efficiency) / total_power
        return endurance_hours * 60.0  # Convert to minutes
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load_model(cls, filename: str) -> 'FlyingWingAirframe':
        """Load model configuration from file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        return cls(config)


def create_default_flying_wing() -> FlyingWingAirframe:
    """Create a default flying wing airframe model."""
    return FlyingWingAirframe()


def create_custom_flying_wing(wingspan: float, root_chord: float,
                             neuromorphic_config: Dict[str, Any]) -> FlyingWingAirframe:
    """
    Create a custom flying wing airframe model.
    
    Args:
        wingspan: Wingspan in meters
        root_chord: Root chord in meters
        neuromorphic_config: Neuromorphic hardware configuration
        
    Returns:
        Customized flying wing airframe model
    """
    config = {
        'wingspan': wingspan,
        'root_chord': root_chord,
        'tip_chord': root_chord * 0.3,  # Default taper ratio
    }
    
    if neuromorphic_config:
        if 'nodes' in neuromorphic_config:
            config['neuromorphic_nodes'] = neuromorphic_config['nodes']
        if 'emi_constraints' in neuromorphic_config:
            config['emi_constraints'] = neuromorphic_config['emi_constraints']
        if 'power_constraints' in neuromorphic_config:
            config['power_constraints'] = neuromorphic_config['power_constraints']
    
    return FlyingWingAirframe(config)