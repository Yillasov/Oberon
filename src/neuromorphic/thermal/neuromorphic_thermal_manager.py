"""
Thermal management system for neuromorphic hardware near propulsion systems.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ThermalZone(Enum):
    CORE = "core"
    INTERFACE = "interface"
    PERIPHERAL = "peripheral"
    PROPULSION_BOUNDARY = "propulsion_boundary"


@dataclass
class ThermalSpecs:
    """Thermal specifications for neuromorphic hardware."""
    max_core_temp: float = 85.0      # °C
    max_junction_temp: float = 95.0   # °C
    thermal_resistance: float = 0.5   # °C/W
    heat_capacity: float = 0.75       # J/g°C
    cooling_efficiency: float = 0.85  # 0-1
    thermal_conductivity: float = 150  # W/m·K (aluminum)


class NeuromorphicThermalManager:
    """Thermal management for neuromorphic hardware near propulsion."""
    
    def __init__(self, specs: ThermalSpecs = ThermalSpecs()):
        self.specs = specs
        self.thermal_state = {
            ThermalZone.CORE: 25.0,
            ThermalZone.INTERFACE: 25.0,
            ThermalZone.PERIPHERAL: 25.0,
            ThermalZone.PROPULSION_BOUNDARY: 25.0
        }
        
        self.cooling_power = 0.0
        self.heat_flux = {zone: 0.0 for zone in ThermalZone}
        self.thermal_gradient = {zone: 0.0 for zone in ThermalZone}
        
        # Thermal protection state
        self.protection_active = False
        self.throttling_level = 0.0
    
    def update(self, 
              propulsion_temp: float,
              neuromorphic_power: float,
              ambient_temp: float,
              dt: float) -> Dict[str, Any]:
        """Update thermal management system."""
        # Calculate heat transfer from propulsion
        propulsion_heat = self._calculate_propulsion_heat(propulsion_temp)
        
        # Calculate neuromorphic heat generation
        core_heat = neuromorphic_power * (1 - self.specs.cooling_efficiency)
        
        # Update thermal zones
        self._update_thermal_zones(propulsion_heat, core_heat, ambient_temp, dt)
        
        # Calculate required cooling
        self.cooling_power = self._calculate_cooling_requirement()
        
        # Check thermal protection needs
        self._check_thermal_protection()
        
        return self.get_status()
    
    def _calculate_propulsion_heat(self, propulsion_temp: float) -> float:
        """Calculate heat transfer from propulsion system."""
        boundary_temp = self.thermal_state[ThermalZone.PROPULSION_BOUNDARY]
        temp_difference = propulsion_temp - boundary_temp
        
        # Simple heat transfer model
        heat_transfer = (temp_difference * self.specs.thermal_conductivity) / 0.1  # 10cm distance
        return max(0, heat_transfer)
    
    def _update_thermal_zones(self, 
                            propulsion_heat: float,
                            core_heat: float,
                            ambient_temp: float,
                            dt: float):
        """Update temperatures in different thermal zones."""
        # Update propulsion boundary
        self.heat_flux[ThermalZone.PROPULSION_BOUNDARY] = propulsion_heat
        self.thermal_state[ThermalZone.PROPULSION_BOUNDARY] += \
            (propulsion_heat * dt) / (self.specs.heat_capacity * 100)  # Assumed mass 100g
        
        # Update core temperature
        core_delta = (core_heat - self.cooling_power) * dt / \
                    (self.specs.heat_capacity * 50)  # Assumed core mass 50g
        self.thermal_state[ThermalZone.CORE] += core_delta
        
        # Update interface temperature (between core and boundary)
        interface_temp = (self.thermal_state[ThermalZone.CORE] + 
                        self.thermal_state[ThermalZone.PROPULSION_BOUNDARY]) / 2
        self.thermal_state[ThermalZone.INTERFACE] = interface_temp
        
        # Update peripheral temperature
        peripheral_cooling = (self.thermal_state[ThermalZone.PERIPHERAL] - ambient_temp) * \
                           0.1 * dt  # Natural cooling
        self.thermal_state[ThermalZone.PERIPHERAL] -= peripheral_cooling
        
        # Calculate thermal gradients
        for zone in ThermalZone:
            self.thermal_gradient[zone] = (self.thermal_state[zone] - ambient_temp) / \
                                        self.specs.thermal_resistance
    
    def _calculate_cooling_requirement(self) -> float:
        """Calculate required cooling power."""
        core_temp = self.thermal_state[ThermalZone.CORE]
        target_temp = self.specs.max_core_temp * 0.9  # 10% margin
        
        if core_temp > target_temp:
            # Proportional cooling response
            cooling_power = (core_temp - target_temp) * 5  # 5W cooling per °C above target
            return min(cooling_power, 50.0)  # Max 50W cooling
        return 0.0
    
    def _check_thermal_protection(self):
        """Check and activate thermal protection if needed."""
        core_temp = self.thermal_state[ThermalZone.CORE]
        
        if core_temp > self.specs.max_core_temp:
            self.protection_active = True
            # Calculate throttling based on temperature excess
            self.throttling_level = min(1.0, (core_temp - self.specs.max_core_temp) / 10)
        elif core_temp < self.specs.max_core_temp - 5:  # 5°C hysteresis
            self.protection_active = False
            self.throttling_level = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current thermal management status."""
        return {
            'temperatures': {zone.value: temp 
                           for zone, temp in self.thermal_state.items()},
            'heat_flux': {zone.value: flux 
                         for zone, flux in self.heat_flux.items()},
            'thermal_gradient': {zone.value: grad 
                               for zone, grad in self.thermal_gradient.items()},
            'cooling_power': self.cooling_power,
            'protection_active': self.protection_active,
            'throttling_level': self.throttling_level
        }
    
    def get_cooling_recommendation(self) -> Dict[str, float]:
        """Get cooling system recommendations."""
        return {
            'fan_speed': min(1.0, self.cooling_power / 50.0),
            'liquid_cooling_power': max(0.0, self.cooling_power - 25.0),
            'thermal_interface_pressure': 0.8 if self.protection_active else 0.5
        }


def create_thermal_manager(propulsion_specs: Dict[str, float]) -> NeuromorphicThermalManager:
    """Create a thermal manager with propulsion-specific settings."""
    specs = ThermalSpecs(
        max_core_temp=85.0,
        max_junction_temp=95.0,
        thermal_resistance=0.5,
        heat_capacity=0.75,
        cooling_efficiency=0.85,
        thermal_conductivity=150.0
    )
    return NeuromorphicThermalManager(specs)