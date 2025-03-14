"""
Flexible hybrid engine model with multi-source power management.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class PowerSource(Enum):
    ELECTRIC = "electric"
    COMBUSTION = "combustion"
    HYDROGEN = "hydrogen"
    CAPACITOR = "capacitor"


@dataclass
class SourceSpecs:
    """Power source specifications."""
    max_power: float          # kW
    efficiency: float         # Base efficiency
    response_time: float      # seconds
    startup_time: float       # seconds
    shutdown_time: float      # seconds
    thermal_limit: float      # K
    energy_capacity: float    # kWh


class FlexibleHybridEngine:
    """Multi-source hybrid engine with dynamic power management."""
    
    def __init__(self):
        self.sources = {
            PowerSource.ELECTRIC: SourceSpecs(200.0, 0.95, 0.1, 0.5, 0.3, 360.0, 100.0),
            PowerSource.COMBUSTION: SourceSpecs(300.0, 0.35, 2.0, 5.0, 3.0, 1200.0, 400.0),
            PowerSource.HYDROGEN: SourceSpecs(150.0, 0.60, 1.0, 2.0, 1.0, 360.0, 200.0),
            PowerSource.CAPACITOR: SourceSpecs(100.0, 0.98, 0.01, 0.1, 0.1, 330.0, 5.0)
        }
        
        self.state = {src: {
            "power_output": 0.0,
            "temperature": 298.15,
            "efficiency": spec.efficiency,
            "energy_level": 1.0,
            "status": "ready",
            "load_factor": 0.0
        } for src, spec in self.sources.items()}
        
        self.power_distribution = {}
        self.active_sources = set()
        self.transition_states = {}
        
        # System parameters
        self.params = {
            "power_threshold": 0.1,     # Minimum power fraction to activate source
            "transition_rate": 0.2,     # Rate of power transition
            "thermal_factor": 0.8,      # Thermal limit factor
            "efficiency_weight": 0.4,   # Weight for efficiency in source selection
            "response_weight": 0.3,     # Weight for response time in source selection
            "thermal_weight": 0.3       # Weight for thermal condition in source selection
        }
    
    def update(self, power_demand: float, dt: float, ambient_temp: float = 293.15) -> Dict[str, Any]:
        """Update hybrid engine state."""
        # Determine optimal power distribution
        self._distribute_power(power_demand)
        
        # Update each power source
        total_power = 0.0
        for source, target_power in self.power_distribution.items():
            source_state = self._update_source(source, target_power, dt, ambient_temp)
            total_power += source_state["power_output"]
        
        # Update system efficiency
        system_efficiency = self._calculate_system_efficiency()
        
        return {
            "total_power": total_power,
            "system_efficiency": system_efficiency,
            "source_states": self.state,
            "power_distribution": self.power_distribution
        }
    
    def _distribute_power(self, power_demand: float):
        """Distribute power demand among available sources."""
        self.power_distribution = {}
        remaining_power = power_demand
        
        # Calculate source scores
        source_scores = self._calculate_source_scores()
        sorted_sources = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Distribute power based on scores and constraints
        for source, score in sorted_sources:
            if remaining_power <= 0:
                break
                
            spec = self.sources[source]
            state = self.state[source]
            
            # Calculate available power from this source
            available = min(
                spec.max_power * state["energy_level"],
                remaining_power,
                spec.max_power * self.params["thermal_factor"] * 
                (1.0 - (state["temperature"] - 298.15) / (spec.thermal_limit - 298.15))
            )
            
            if available > spec.max_power * self.params["power_threshold"]:
                self.power_distribution[source] = available
                remaining_power -= available
                
                if source not in self.active_sources:
                    self.transition_states[source] = "starting"
                    self.active_sources.add(source)
        
        # Update source states for deactivation
        for source in list(self.active_sources):
            if source not in self.power_distribution:
                self.transition_states[source] = "stopping"
                self.active_sources.remove(source)
    
    def _calculate_source_scores(self) -> Dict[PowerSource, float]:
        """Calculate selection scores for each power source."""
        scores = {}
        for source, spec in self.sources.items():
            state = self.state[source]
            
            # Efficiency score
            efficiency_score = state["efficiency"] / max(s.efficiency 
                                                       for s in self.sources.values())
            
            # Response score
            response_score = min(1.0, 1.0 / spec.response_time) / max(
                min(1.0, 1.0 / s.response_time) for s in self.sources.values())
            
            # Thermal score
            temp_margin = (spec.thermal_limit - state["temperature"]) / \
                        (spec.thermal_limit - 298.15)
            thermal_score = max(0.0, min(1.0, temp_margin))
            
            # Combined score
            scores[source] = (
                efficiency_score * self.params["efficiency_weight"] +
                response_score * self.params["response_weight"] +
                thermal_score * self.params["thermal_weight"]
            )
        
        return scores
    
    def _update_source(self, source: PowerSource, target_power: float, 
                      dt: float, ambient_temp: float) -> Dict[str, Any]:
        """Update individual power source state."""
        spec = self.sources[source]
        state = self.state[source]
        
        # Handle transitions
        if source in self.transition_states:
            if self.transition_states[source] == "starting":
                target_power *= min(1.0, dt / spec.startup_time)
                if dt >= spec.startup_time:
                    self.transition_states.pop(source)
            elif self.transition_states[source] == "stopping":
                target_power *= max(0.0, 1.0 - dt / spec.shutdown_time)
                if dt >= spec.shutdown_time:
                    self.transition_states.pop(source)
        
        # Update power output with response time
        power_error = target_power - state["power_output"]
        state["power_output"] += power_error * dt / spec.response_time
        
        # Update load factor
        state["load_factor"] = state["power_output"] / spec.max_power
        
        # Update thermal state
        self._update_thermal_state(source, dt, ambient_temp)
        
        # Update energy level
        energy_consumed = (state["power_output"] / state["efficiency"]) * dt / 3600
        state["energy_level"] -= energy_consumed / spec.energy_capacity
        state["energy_level"] = max(0.0, min(1.0, state["energy_level"]))
        
        # Update efficiency based on load and temperature
        self._update_source_efficiency(source)
        
        return state
    
    def _update_thermal_state(self, source: PowerSource, dt: float, ambient_temp: float):
        """Update thermal state of power source."""
        spec = self.sources[source]
        state = self.state[source]
        
        # Heat generation
        heat_generation = (1.0 - state["efficiency"]) * state["power_output"]
        
        # Cooling
        temp_difference = state["temperature"] - ambient_temp
        cooling = 0.1 * temp_difference  # Simple cooling model
        
        # Net temperature change
        net_heat = heat_generation - cooling
        state["temperature"] += net_heat * dt / 10.0  # Thermal mass factor
        
        # Limit temperature
        state["temperature"] = min(state["temperature"], spec.thermal_limit)
    
    def _update_source_efficiency(self, source: PowerSource):
        """Update source efficiency based on operating conditions."""
        spec = self.sources[source]
        state = self.state[source]
        
        # Base efficiency
        base = spec.efficiency
        
        # Load factor influence
        load_factor = state["load_factor"]
        load_efficiency = 1.0 - 0.1 * abs(load_factor - 0.7)  # Optimal at 70% load
        
        # Temperature influence
        temp_factor = 1.0 - 0.2 * (state["temperature"] - 298.15) / \
                     (spec.thermal_limit - 298.15)
        
        # Combined efficiency
        state["efficiency"] = base * load_efficiency * temp_factor
    
    def _calculate_system_efficiency(self) -> float:
        """Calculate overall system efficiency."""
        total_power_out = sum(state["power_output"] for state in self.state.values())
        total_power_in = sum(state["power_output"] / state["efficiency"] 
                           for state in self.state.values())
        
        return total_power_out / total_power_in if total_power_in > 0 else 0.0
    
    def get_source_availability(self) -> Dict[PowerSource, Dict[str, float]]:
        """Get availability status of all power sources."""
        availability = {}
        for source, spec in self.sources.items():
            state = self.state[source]
            
            # Calculate various availability factors
            energy_factor = state["energy_level"]
            thermal_factor = max(0.0, (spec.thermal_limit - state["temperature"]) / 
                               (spec.thermal_limit - 298.15))
            
            availability[source] = {
                "power_available": spec.max_power * min(energy_factor, thermal_factor),
                "energy_remaining": state["energy_level"] * spec.energy_capacity,
                "thermal_headroom": thermal_factor,
                "current_efficiency": state["efficiency"]
            }
        
        return availability


class HybridController:
    """Controller for flexible hybrid engine system."""
    
    def __init__(self, engine: FlexibleHybridEngine, control_rate: float = 50.0):
        self.engine = engine
        self.control_rate = control_rate
        self.last_update = 0.0
        self.mode = "balanced"  # balanced, efficiency, performance, endurance
        
        self.setpoints = {
            "power": 0.0,
            "min_efficiency": 0.7,
            "thermal_margin": 0.2,
            "energy_reserve": 0.15
        }
    
    def process_state(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Process engine state and generate control outputs."""
        current_time = state.get("time", 0.0)
        
        if current_time - self.last_update < 1.0 / self.control_rate:
            return {"power_demand": self.setpoints["power"]}
        
        availability = self.engine.get_source_availability()
        
        # Adjust power demand based on mode and conditions
        adjusted_power = self._adjust_power_demand(availability)
        
        self.last_update = current_time
        return {"power_demand": adjusted_power}
    
    def _adjust_power_demand(self, availability: Dict[PowerSource, Dict[str, float]]) -> float:
        """Adjust power demand based on operating mode and conditions."""
        base_power = self.setpoints["power"]
        
        if self.mode == "efficiency":
            # Prioritize efficient sources
            efficient_power = sum(
                data["power_available"] 
                for source, data in availability.items()
                if data["current_efficiency"] > self.setpoints["min_efficiency"]
            )
            return min(base_power, efficient_power)
            
        elif self.mode == "performance":
            # Use all available power
            max_power = sum(data["power_available"] for data in availability.values())
            return min(base_power, max_power)
            
        elif self.mode == "endurance":
            # Maintain energy reserves
            conservative_power = sum(
                data["power_available"] 
                for source, data in availability.items()
                if data["energy_remaining"] > self.setpoints["energy_reserve"]
            )
            return min(base_power, conservative_power * 0.8)
            
        else:  # balanced mode
            # Balance between efficiency and availability
            balanced_power = sum(
                data["power_available"] * data["current_efficiency"]
                for data in availability.values()
            ) / len(availability)
            return min(base_power, balanced_power)
    
    def set_mode(self, mode: str, power: float):
        """Set operating mode and power demand."""
        self.mode = mode
        self.setpoints["power"] = max(0.0, power)