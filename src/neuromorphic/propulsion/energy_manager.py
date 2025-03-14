"""
Energy management system for propulsion optimization.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class PowerMode(Enum):
    ECO = "eco"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    EMERGENCY = "emergency"


@dataclass
class EnergyConfig:
    """Energy management configuration."""
    max_power_rate: float = 100.0    # kW/s
    min_efficiency: float = 0.65     # Minimum acceptable efficiency
    reserve_factor: float = 0.15     # Energy reserve requirement
    thermal_threshold: float = 0.85   # Thermal limit threshold
    recovery_rate: float = 0.05      # Power recovery rate


class EnergyManager:
    """Propulsion energy management system."""
    
    def __init__(self, config: EnergyConfig = EnergyConfig()):
        self.config = config
        self.state = {
            "mode": PowerMode.BALANCED,
            "power_limit": float('inf'),
            "efficiency_target": 0.8,
            "energy_reserve": 1.0,
            "thermal_status": "normal",
            "optimization_score": 1.0
        }
        
        self.history = {
            "power": [],
            "efficiency": [],
            "temperature": []
        }
        
        self.constraints = {
            "thermal": 1.0,
            "efficiency": 1.0,
            "energy": 1.0
        }
    
    def optimize(self, engine_state: Dict[str, Any], 
                vector_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize energy usage based on engine and vectoring states."""
        # Update constraints based on current states
        self._update_constraints(engine_state, vector_state)
        
        # Calculate optimal power limit
        power_limit = self._calculate_power_limit(engine_state)
        
        # Determine efficiency target
        efficiency_target = self._determine_efficiency_target()
        
        # Update system state
        self.state["power_limit"] = power_limit
        self.state["efficiency_target"] = efficiency_target
        
        # Generate optimization commands
        commands = self._generate_commands(engine_state)
        
        # Update history
        self._update_history(engine_state)
        
        return commands
    
    def _update_constraints(self, engine_state: Dict[str, Any], 
                          vector_state: Optional[Dict[str, Any]]):
        """Update system constraints based on current states."""
        # Thermal constraint
        if "temperature" in engine_state:
            temp_ratio = (engine_state["temperature"] - 298.15) / 100
            self.constraints["thermal"] = max(0.0, 1.0 - temp_ratio)
        
        # Efficiency constraint
        if "efficiency" in engine_state:
            eff_ratio = engine_state["efficiency"] / self.config.min_efficiency
            self.constraints["efficiency"] = min(1.0, eff_ratio)
        
        # Energy reserve constraint
        if "energy_level" in engine_state:
            energy_margin = engine_state["energy_level"] - self.config.reserve_factor
            self.constraints["energy"] = max(0.0, energy_margin)
        
        # Consider vectoring impact if available
        if vector_state:
            vector_efficiency = self._calculate_vector_efficiency(vector_state)
            self.constraints["efficiency"] *= vector_efficiency
    
    def _calculate_power_limit(self, engine_state: Dict[str, Any]) -> float:
        """Calculate optimal power limit based on constraints."""
        base_limit = engine_state.get("max_power", float('inf'))
        
        # Apply constraints
        thermal_limit = base_limit * self.constraints["thermal"]
        efficiency_limit = base_limit * self.constraints["efficiency"]
        energy_limit = base_limit * self.constraints["energy"]
        
        # Mode-specific limits
        mode_factors = {
            PowerMode.ECO: 0.7,
            PowerMode.BALANCED: 0.9,
            PowerMode.PERFORMANCE: 1.0,
            PowerMode.EMERGENCY: 1.2
        }
        
        mode_limit = base_limit * mode_factors[self.state["mode"]]
        
        return min(thermal_limit, efficiency_limit, energy_limit, mode_limit)
    
    def _determine_efficiency_target(self) -> float:
        """Determine target efficiency based on mode and constraints."""
        base_target = 0.8
        
        mode_adjustments = {
            PowerMode.ECO: 0.1,
            PowerMode.BALANCED: 0.0,
            PowerMode.PERFORMANCE: -0.05,
            PowerMode.EMERGENCY: -0.15
        }
        
        adjusted_target = base_target + mode_adjustments[self.state["mode"]]
        return max(self.config.min_efficiency, 
                  min(0.95, adjusted_target * self.constraints["efficiency"]))
    
    def _calculate_vector_efficiency(self, vector_state: Dict[str, Any]) -> float:
        """Calculate efficiency impact of thrust vectoring."""
        if "vector_angles" not in vector_state:
            return 1.0
            
        angles = vector_state["vector_angles"]
        pitch = abs(angles.get("pitch", 0.0))
        yaw = abs(angles.get("yaw", 0.0))
        
        # Efficiency loss increases with deflection angle
        angle_factor = 1.0 - 0.002 * (pitch + yaw)
        return max(0.8, angle_factor)
    
    def _generate_commands(self, engine_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization commands."""
        current_power = engine_state.get("power_output", 0.0)
        
        # Calculate optimal power adjustment
        power_error = self.state["power_limit"] - current_power
        max_change = self.config.max_power_rate * 0.1  # 100ms adjustment period
        power_adjustment = np.clip(power_error, -max_change, max_change)
        
        return {
            "power_target": current_power + power_adjustment,
            "efficiency_target": self.state["efficiency_target"],
            "mode": self.state["mode"].value,
            "constraints": self.constraints.copy()
        }
    
    def _update_history(self, engine_state: Dict[str, Any]):
        """Update performance history."""
        self.history["power"].append(engine_state.get("power_output", 0.0))
        self.history["efficiency"].append(engine_state.get("efficiency", 0.0))
        self.history["temperature"].append(engine_state.get("temperature", 298.15))
        
        # Maintain history length
        max_length = 1000
        for key in self.history:
            if len(self.history[key]) > max_length:
                self.history[key] = self.history[key][-max_length:]
    
    def set_mode(self, mode: PowerMode):
        """Set power management mode."""
        self.state["mode"] = mode
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed energy management diagnostics."""
        return {
            "state": self.state.copy(),
            "constraints": self.constraints.copy(),
            "performance": {
                "average_power": np.mean(self.history["power"]) if self.history["power"] else 0.0,
                "average_efficiency": np.mean(self.history["efficiency"]) if self.history["efficiency"] else 0.0,
                "thermal_trend": np.mean(self.history["temperature"][-10:]) - 
                                np.mean(self.history["temperature"][-20:-10]) 
                                if len(self.history["temperature"]) >= 20 else 0.0
            }
        }