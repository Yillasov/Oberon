"""
Modular engine system with hot-swappable components and dynamic configuration.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class ComponentType(Enum):
    COMBUSTION = "combustion"
    COOLING = "cooling"
    FUEL = "fuel"
    POWER = "power"
    CONTROL = "control"
    SENSOR = "sensor"


@dataclass
class ComponentSpec:
    """Specification for engine components."""
    type: ComponentType
    power_rating: float
    efficiency: float
    response_time: float
    thermal_limit: float
    weight: float
    status: str = "ready"


class EngineComponent(ABC):
    """Base class for all engine components."""
    
    @abstractmethod
    def update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Update component state."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        pass


class CombustionModule(EngineComponent):
    """Combustion system module."""
    
    def __init__(self, spec: ComponentSpec):
        self.spec = spec
        self.state = {
            "power": 0.0,
            "temperature": 298.15,
            "efficiency": spec.efficiency,
            "fuel_flow": 0.0
        }
    
    def update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        target_power = inputs.get("power_demand", 0.0)
        fuel_quality = inputs.get("fuel_quality", 1.0)
        
        # Power response
        power_error = target_power - self.state["power"]
        self.state["power"] += power_error * dt / self.spec.response_time
        
        # Calculate fuel flow
        self.state["fuel_flow"] = self.state["power"] / (43e6 * self.state["efficiency"])
        
        # Temperature dynamics
        heat_generation = (1.0 - self.state["efficiency"]) * self.state["power"]
        self.state["temperature"] += heat_generation * dt / 1000
        
        # Efficiency adjustment
        self.state["efficiency"] = self.spec.efficiency * fuel_quality
        
        return self.state
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": ComponentType.COMBUSTION.value,
            "state": self.state,
            "health": "normal" if self.state["temperature"] < self.spec.thermal_limit else "warning"
        }


class CoolingModule(EngineComponent):
    """Cooling system module."""
    
    def __init__(self, spec: ComponentSpec):
        self.spec = spec
        self.state = {
            "cooling_power": 0.0,
            "flow_rate": 0.0,
            "efficiency": spec.efficiency
        }
    
    def update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        target_temp = inputs.get("target_temperature", 350)
        current_temp = inputs.get("current_temperature", 298.15)
        
        # Calculate required cooling
        temp_difference = current_temp - target_temp
        required_cooling = max(0, temp_difference * 100)
        
        # Update cooling power
        self.state["cooling_power"] = min(required_cooling, self.spec.power_rating)
        self.state["flow_rate"] = self.state["cooling_power"] / (4200 * temp_difference) \
                                 if temp_difference > 0 else 0
        
        return self.state
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": ComponentType.COOLING.value,
            "state": self.state,
            "health": "normal" if self.state["cooling_power"] < self.spec.power_rating else "warning"
        }


class PowerModule(EngineComponent):
    """Power management module."""
    
    def __init__(self, spec: ComponentSpec):
        self.spec = spec
        self.state = {
            "power_output": 0.0,
            "load_factor": 0.0,
            "efficiency": spec.efficiency
        }
    
    def update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        input_power = inputs.get("input_power", 0.0)
        
        # Power conversion
        self.state["power_output"] = input_power * self.state["efficiency"]
        self.state["load_factor"] = self.state["power_output"] / self.spec.power_rating
        
        # Efficiency degradation under high load
        if self.state["load_factor"] > 0.9:
            self.state["efficiency"] = self.spec.efficiency * \
                                     (1.0 - (self.state["load_factor"] - 0.9) * 0.2)
        
        return self.state
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": ComponentType.POWER.value,
            "state": self.state,
            "health": "normal" if self.state["load_factor"] < 0.95 else "warning"
        }


class ModularEngine:
    """Modular engine with hot-swappable components."""
    
    def __init__(self):
        self.components: Dict[str, EngineComponent] = {}
        self.state = {
            "total_power": 0.0,
            "system_efficiency": 0.0,
            "temperature": 298.15,
            "status": "ready"
        }
    
    def add_component(self, name: str, component: EngineComponent):
        """Add or replace a component."""
        self.components[name] = component
    
    def remove_component(self, name: str) -> Optional[EngineComponent]:
        """Remove a component."""
        return self.components.pop(name, None)
    
    def update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Update engine state."""
        component_states = {}
        total_power_in = 0.0
        total_power_out = 0.0
        
        # Update combustion components
        for name, component in self.components.items():
            if isinstance(component, CombustionModule):
                state = component.update(inputs, dt)
                component_states[name] = state
                total_power_in += state["power"]
        
        # Update cooling components
        cooling_inputs = {
            "current_temperature": self.state["temperature"],
            "target_temperature": inputs.get("target_temperature", 350)
        }
        total_cooling = 0.0
        for name, component in self.components.items():
            if isinstance(component, CoolingModule):
                state = component.update(cooling_inputs, dt)
                component_states[name] = state
                total_cooling += state["cooling_power"]
        
        # Update power components
        power_inputs = {"input_power": total_power_in}
        for name, component in self.components.items():
            if isinstance(component, PowerModule):
                state = component.update(power_inputs, dt)
                component_states[name] = state
                total_power_out += state["power_output"]
        
        # Update system state
        self.state["total_power"] = total_power_out
        self.state["system_efficiency"] = total_power_out / total_power_in \
                                        if total_power_in > 0 else 0.0
        
        # Temperature dynamics
        heat_generation = total_power_in - total_power_out
        cooling_effect = total_cooling * dt / 1000
        self.state["temperature"] += (heat_generation - cooling_effect) * dt / 1000
        
        # Update status
        self._update_system_status()
        
        return self.state
    
    def _update_system_status(self):
        """Update overall system status."""
        component_health = [c.get_status()["health"] for c in self.components.values()]
        if any(health == "critical" for health in component_health):
            self.state["status"] = "critical"
        elif any(health == "warning" for health in component_health):
            self.state["status"] = "warning"
        else:
            self.state["status"] = "normal"
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed system diagnostics."""
        return {
            "system_state": self.state,
            "components": {
                name: component.get_status()
                for name, component in self.components.items()
            }
        }


# Example usage
def create_standard_engine() -> ModularEngine:
    """Create a standard engine configuration."""
    engine = ModularEngine()
    
    # Add components
    engine.add_component("main_combustion", CombustionModule(
        ComponentSpec(
            type=ComponentType.COMBUSTION,
            power_rating=500.0,
            efficiency=0.35,
            response_time=0.5,
            thermal_limit=1200.0,
            weight=100.0
        )
    ))
    
    engine.add_component("cooling_system", CoolingModule(
        ComponentSpec(
            type=ComponentType.COOLING,
            power_rating=200.0,
            efficiency=0.90,
            response_time=0.2,
            thermal_limit=400.0,
            weight=50.0
        )
    ))
    
    engine.add_component("power_unit", PowerModule(
        ComponentSpec(
            type=ComponentType.POWER,
            power_rating=450.0,
            efficiency=0.95,
            response_time=0.1,
            thermal_limit=350.0,
            weight=75.0
        )
    ))
    
    return engine