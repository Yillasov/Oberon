"""
Biomimetic engine model with adaptive control and bio-inspired energy conversion.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class EngineState(Enum):
    IDLE = "idle"
    ADAPTING = "adapting"
    OPTIMAL = "optimal"
    REGENERATING = "regenerating"
    STRESSED = "stressed"


@dataclass
class BiomimeticSpecs:
    """Biomimetic engine specifications."""
    max_power: float = 100.0        # kW
    base_efficiency: float = 0.75   # Base efficiency
    adaptation_rate: float = 0.2    # Rate of performance adaptation
    recovery_rate: float = 0.1      # Recovery rate after stress
    energy_density: float = 5.0     # kWh/kg
    storage_capacity: float = 40.0  # kWh
    thermal_tolerance: float = 30.0  # °C range


class BiomimeticEngine:
    """Bio-inspired engine with adaptive characteristics."""
    
    def __init__(self):
        self.specs = BiomimeticSpecs()
        self.state = {
            "power_output": 0.0,        # kW
            "efficiency": 0.75,         # Current efficiency
            "temperature": 298.15,      # K
            "stress_level": 0.0,        # 0-1 scale
            "adaptation_level": 0.0,    # 0-1 scale
            "energy_level": 1.0,        # Fraction of capacity
            "metabolic_rate": 0.0,      # kW
            "regeneration_rate": 0.0    # kW
        }
        
        self.operating_state = EngineState.IDLE
        self.adaptation_memory = []
        self.stress_history = []
        
        # Bio-inspired parameters
        self.params = {
            "optimal_temp": 308.15,     # K (35°C)
            "temp_range": 15.0,         # K
            "stress_threshold": 0.7,    # Stress level threshold
            "adaptation_threshold": 0.3, # Adaptation trigger threshold
            "memory_length": 100,       # Length of adaptation memory
            "regeneration_factor": 0.15  # Energy regeneration factor
        }
    
    def update(self, power_demand: float, dt: float, ambient_temp: float = 293.15) -> Dict[str, Any]:
        """Update engine state with bio-inspired dynamics."""
        # Update stress level based on power demand
        self._update_stress_level(power_demand, dt)
        
        # Adaptive response to stress
        self._adapt_to_conditions(dt)
        
        # Calculate available power
        max_available = self.specs.max_power * self.state["energy_level"] * \
                       (1.0 - self.state["stress_level"])
        actual_power = min(power_demand, max_available)
        
        # Update metabolic rate (power consumption)
        base_metabolic_rate = 0.05 * self.specs.max_power  # Base consumption
        stress_induced_rate = base_metabolic_rate * (1 + self.state["stress_level"])
        self.state["metabolic_rate"] = stress_induced_rate
        
        # Energy consumption and regeneration
        self._update_energy_levels(actual_power, dt)
        
        # Temperature response
        self._update_thermal_state(actual_power, ambient_temp, dt)
        
        # Update efficiency based on conditions
        self._update_efficiency()
        
        # Update power output
        self.state["power_output"] = actual_power
        
        return self.state
    
    def _update_stress_level(self, power_demand: float, dt: float):
        """Update engine stress level based on operating conditions."""
        power_ratio = power_demand / self.specs.max_power
        temp_stress = abs(self.state["temperature"] - self.params["optimal_temp"]) / \
                     self.params["temp_range"]
        
        new_stress = 0.6 * power_ratio + 0.4 * temp_stress
        
        # Gradual stress change with memory effect
        self.state["stress_level"] += (new_stress - self.state["stress_level"]) * \
                                    dt * self.specs.adaptation_rate
        self.state["stress_level"] = max(0.0, min(1.0, self.state["stress_level"]))
        
        self.stress_history.append(self.state["stress_level"])
        if len(self.stress_history) > self.params["memory_length"]:
            self.stress_history.pop(0)
    
    def _adapt_to_conditions(self, dt: float):
        """Implement bio-inspired adaptation mechanisms."""
        avg_stress = sum(self.stress_history) / len(self.stress_history) \
                    if self.stress_history else 0.0
        
        if avg_stress > self.params["stress_threshold"]:
            # Initiate stress response
            self.operating_state = EngineState.STRESSED
            adaptation_rate = self.specs.adaptation_rate * (1.0 - self.state["adaptation_level"])
            
        elif avg_stress < self.params["adaptation_threshold"]:
            # Recovery phase
            self.operating_state = EngineState.REGENERATING
            adaptation_rate = -self.specs.adaptation_rate * self.state["adaptation_level"]
            
        else:
            # Normal operation with maintained adaptation
            self.operating_state = EngineState.OPTIMAL
            adaptation_rate = 0.0
        
        # Update adaptation level
        self.state["adaptation_level"] += adaptation_rate * dt
        self.state["adaptation_level"] = max(0.0, min(1.0, self.state["adaptation_level"]))
        
        # Store adaptation memory
        self.adaptation_memory.append(self.state["adaptation_level"])
        if len(self.adaptation_memory) > self.params["memory_length"]:
            self.adaptation_memory.pop(0)
    
    def _update_energy_levels(self, power_output: float, dt: float):
        """Update energy levels with bio-inspired regeneration."""
        # Energy consumption
        total_consumption = (power_output + self.state["metabolic_rate"]) * dt / 3600
        
        # Bio-inspired regeneration during low stress periods
        if self.operating_state == EngineState.REGENERATING:
            regeneration = self.params["regeneration_factor"] * \
                         (1.0 - self.state["energy_level"]) * dt / 3600
            self.state["regeneration_rate"] = regeneration * 3600  # Convert to kW
        else:
            regeneration = 0.0
            self.state["regeneration_rate"] = 0.0
        
        # Update energy level
        energy_change = regeneration - total_consumption
        self.state["energy_level"] += energy_change / self.specs.storage_capacity
        self.state["energy_level"] = max(0.0, min(1.0, self.state["energy_level"]))
    
    def _update_thermal_state(self, power_output: float, ambient_temp: float, dt: float):
        """Update thermal state with bio-inspired regulation."""
        # Heat generation
        heat_generation = (1.0 - self.state["efficiency"]) * power_output + \
                         0.5 * self.state["metabolic_rate"]
        
        # Bio-inspired adaptive cooling
        cooling_factor = 0.5 + 0.5 * self.state["adaptation_level"]
        temp_difference = self.state["temperature"] - ambient_temp
        cooling_power = cooling_factor * temp_difference
        
        # Net temperature change
        net_heat = heat_generation - cooling_power
        temp_change = net_heat * dt / (10.0 + 5.0 * self.state["adaptation_level"])
        
        self.state["temperature"] += temp_change
    
    def _update_efficiency(self):
        """Update efficiency based on bio-inspired factors."""
        base = self.specs.base_efficiency
        
        # Adaptation benefit
        adaptation_bonus = 0.1 * self.state["adaptation_level"]
        
        # Stress penalty
        stress_penalty = 0.15 * self.state["stress_level"]
        
        # Temperature factor
        temp_optimal = 1.0 - 0.5 * abs(self.state["temperature"] - 
                                      self.params["optimal_temp"]) / \
                                      self.params["temp_range"]
        
        # Energy level factor
        energy_factor = 0.9 + 0.1 * self.state["energy_level"]
        
        self.state["efficiency"] = base * (1.0 + adaptation_bonus - stress_penalty) * \
                                 temp_optimal * energy_factor
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostic data."""
        return {
            "performance": {
                "power_output": self.state["power_output"],
                "efficiency": self.state["efficiency"],
                "operating_state": self.operating_state.value
            },
            "adaptation": {
                "stress_level": self.state["stress_level"],
                "adaptation_level": self.state["adaptation_level"],
                "recent_stress_avg": sum(self.stress_history[-10:]) / 
                                   min(len(self.stress_history), 10)
            },
            "energy": {
                "level": self.state["energy_level"],
                "metabolic_rate": self.state["metabolic_rate"],
                "regeneration_rate": self.state["regeneration_rate"]
            },
            "thermal": {
                "temperature": self.state["temperature"],
                "temp_stress": abs(self.state["temperature"] - 
                                 self.params["optimal_temp"]) / 
                                 self.params["temp_range"]
            }
        }


class BiomimeticController:
    """Bio-inspired adaptive controller."""
    
    def __init__(self, engine: BiomimeticEngine, control_rate: float = 50.0):
        self.engine = engine
        self.control_rate = control_rate
        self.last_update = 0.0
        self.setpoints = {
            "power": 0.0,
            "efficiency_target": 0.8,
            "stress_limit": 0.8
        }
        self.adaptation_state = "normal"
        self.control_memory = []
    
    def process_state(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Process engine state with bio-inspired control strategy."""
        current_time = state.get("time", 0.0)
        
        if current_time - self.last_update < 1.0 / self.control_rate:
            return {"power_demand": self._calculate_power_demand()}
        
        diagnostics = self.engine.get_diagnostics()
        
        # Adaptive control based on stress and adaptation levels
        if diagnostics["adaptation"]["stress_level"] > self.setpoints["stress_limit"]:
            self.adaptation_state = "protective"
            power_factor = 0.6
        elif diagnostics["adaptation"]["adaptation_level"] > 0.8:
            self.adaptation_state = "optimized"
            power_factor = 1.1
        else:
            self.adaptation_state = "normal"
            power_factor = 1.0
        
        # Store control decision
        self.control_memory.append({
            "time": current_time,
            "state": self.adaptation_state,
            "power_factor": power_factor
        })
        
        if len(self.control_memory) > 100:
            self.control_memory.pop(0)
        
        self.last_update = current_time
        return {"power_demand": self._calculate_power_demand() * power_factor}
    
    def _calculate_power_demand(self) -> float:
        """Calculate power demand with bio-inspired adaptation."""
        base_demand = self.setpoints["power"]
        
        if self.adaptation_state == "protective":
            return base_demand * 0.6
        elif self.adaptation_state == "optimized":
            return min(base_demand * 1.1, self.engine.specs.max_power)
        
        return base_demand
    
    def set_power_demand(self, power: float):
        """Set desired power output with adaptation memory."""
        self.setpoints["power"] = max(0.0, min(power, self.engine.specs.max_power))