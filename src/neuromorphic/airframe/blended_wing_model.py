"""
Blended Wing Body parametric airframe model for neuromorphic control systems.

This module provides a parametric model for BWB UCAV airframes
with high-density neuromorphic computing integration.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json


class BlendedWingAirframe:
    """Blended wing body airframe with high-density neuromorphic integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic airframe parameters
            'length': 4.5,  # meters
            'wingspan': 8.0,  # meters
            'center_section_width': 2.0,  # meters
            'max_thickness': 0.45,  # meters
            'sweep_angle': 40.0,  # degrees
            
            # Mass properties
            'empty_mass': 450.0,  # kg
            'max_takeoff_mass': 750.0,  # kg
            'fuel_mass': 200.0,  # kg
            
            # Neuromorphic computing bays
            'computing_bays': [
                {
                    'name': 'primary_bay',
                    'position': [0.2, 0.0, 0.0],
                    'volume': 0.08,  # cubic meters
                    'max_heat_load': 800.0,  # Watts
                    'processors': [
                        {'type': 'neural_engine', 'cores': 128, 'power': 150.0},
                        {'type': 'synapse_processor', 'cores': 64, 'power': 100.0}
                    ]
                },
                {
                    'name': 'secondary_bay',
                    'position': [-0.3, 0.0, 0.0],
                    'volume': 0.05,
                    'max_heat_load': 500.0,
                    'processors': [
                        {'type': 'neural_engine', 'cores': 64, 'power': 75.0},
                        {'type': 'synapse_processor', 'cores': 32, 'power': 50.0}
                    ]
                }
            ],
            
            # Thermal management system
            'thermal_system': {
                'coolant_flow_rate': 2.0,  # liters/second
                'max_coolant_temp': 65.0,  # Celsius
                'heat_exchanger_efficiency': 0.85,
                'cooling_zones': ['primary_bay', 'secondary_bay', 'wing_roots']
            },
            
            # Power system
            'power_system': {
                'max_power': 2000.0,  # Watts
                'nominal_voltage': 270.0,  # Volts DC
                'battery_capacity': 45.0,  # kWh
                'power_density': 2.5  # kW/kg
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
        """Calculate derived parameters and validate thermal constraints."""
        # Calculate total computing power
        self.total_compute_power = sum(
            sum(proc['power'] for proc in bay['processors'])
            for bay in self.config['computing_bays']
        )
        
        # Calculate thermal loads
        self.thermal_loads = {}
        for bay in self.config['computing_bays']:
            self.thermal_loads[bay['name']] = sum(
                proc['power'] * 0.9  # Assume 90% of power becomes heat
                for proc in bay['processors']
            )
        
        # Validate thermal constraints
        self._validate_thermal_constraints()
    
    def _validate_thermal_constraints(self) -> None:
        """Validate thermal management system constraints."""
        self.thermal_violations = []
        
        # Check each computing bay
        for bay in self.config['computing_bays']:
            heat_load = self.thermal_loads[bay['name']]
            if heat_load > bay['max_heat_load']:
                self.thermal_violations.append(
                    f"Thermal overload in {bay['name']}: {heat_load:.1f}W > {bay['max_heat_load']:.1f}W"
                )
        
        # Calculate required coolant temperature rise
        coolant_flow = self.config['thermal_system']['coolant_flow_rate']
        total_heat = sum(self.thermal_loads.values())
        temp_rise = total_heat / (coolant_flow * 4184)  # Specific heat of water
        
        if temp_rise > 15.0:  # Maximum allowed temperature rise
            self.thermal_violations.append(
                f"Excessive coolant temperature rise: {temp_rise:.1f}°C"
            )
    
    def get_compute_density(self) -> Dict[str, float]:
        """Calculate neuromorphic computing density metrics."""
        total_volume = sum(bay['volume'] for bay in self.config['computing_bays'])
        total_cores = sum(
            sum(proc['cores'] for proc in bay['processors'])
            for bay in self.config['computing_bays']
        )
        
        return {
            'compute_density': total_cores / total_volume,  # cores/m³
            'power_density': self.total_compute_power / total_volume,  # W/m³
            'thermal_density': sum(self.thermal_loads.values()) / total_volume  # W/m³
        }
    
    def estimate_cooling_performance(self, ambient_temp: float) -> Dict[str, float]:
        """
        Estimate cooling system performance.
        
        Args:
            ambient_temp: Ambient temperature in Celsius
            
        Returns:
            Dictionary of cooling performance metrics
        """
        coolant_flow = self.config['thermal_system']['coolant_flow_rate']
        efficiency = self.config['thermal_system']['heat_exchanger_efficiency']
        total_heat = sum(self.thermal_loads.values())
        
        # Simple heat exchanger model
        coolant_temp_rise = total_heat / (coolant_flow * 4184)
        max_coolant_temp = ambient_temp + coolant_temp_rise / efficiency
        
        return {
            'max_coolant_temp': max_coolant_temp,
            'heat_rejection_rate': total_heat * efficiency,
            'pumping_power': coolant_flow * 0.5  # Simplified pump power calculation
        }
    
    def get_neuromorphic_constraints(self) -> Dict[str, Any]:
        """Get neuromorphic hardware integration constraints."""
        return {
            'thermal_violations': self.thermal_violations,
            'compute_metrics': self.get_compute_density(),
            'power_available': self.config['power_system']['max_power'],
            'cooling_capacity': sum(
                bay['max_heat_load'] for bay in self.config['computing_bays']
            ),
            'total_compute_cores': sum(
                sum(proc['cores'] for proc in bay['processors'])
                for bay in self.config['computing_bays']
            )
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load_model(cls, filename: str) -> 'BlendedWingAirframe':
        """Load model configuration from file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        return cls(config)


def create_default_bwb() -> BlendedWingAirframe:
    """Create default blended wing body model."""
    return BlendedWingAirframe()


def create_custom_bwb(length: float, wingspan: float,
                      compute_config: Dict[str, Any]) -> BlendedWingAirframe:
    """Create custom blended wing body model."""
    config = {
        'length': length,
        'wingspan': wingspan,
        'center_section_width': wingspan * 0.25
    }
    
    if compute_config:
        if 'computing_bays' in compute_config:
            config['computing_bays'] = compute_config['computing_bays']
        if 'thermal_system' in compute_config:
            config['thermal_system'] = compute_config['thermal_system']
        if 'power_system' in compute_config:
            config['power_system'] = compute_config['power_system']
    
    return BlendedWingAirframe(config)