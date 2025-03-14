"""
Hydrogen fuel cell engine model with advanced control interface.
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FuelCellSpecs:
    """Hydrogen fuel cell specifications."""
    max_power: float = 150.0        # kW
    nominal_voltage: float = 650.0  # V
    stack_count: int = 400          # Number of cells in stack
    hydrogen_capacity: float = 5.0  # kg
    operating_pressure: float = 3.0 # bar
    min_cell_voltage: float = 0.6   # V
    max_current_density: float = 1.2 # A/cm²
    active_area: float = 400.0      # cm²


class HydrogenFuelCell:
    """Hydrogen fuel cell engine with integrated systems management."""
    
    def __init__(self):
        self.specs = FuelCellSpecs()
        self.state = {
            "power_output": 0.0,      # kW
            "stack_voltage": 650.0,    # V
            "stack_current": 0.0,      # A
            "temperature": 333.15,     # K (60°C nominal)
            "hydrogen_level": 1.0,     # Fraction remaining
            "air_flow": 0.0,          # kg/s
            "efficiency": 0.60,        # Current efficiency
            "membrane_humidity": 0.85,  # Relative humidity
            "pressure_drop": 0.0       # bar
        }
        
        # System parameters
        self.params = {
            "min_temp": 313.15,      # K (40°C)
            "max_temp": 358.15,      # K (85°C)
            "optimal_temp": 333.15,   # K (60°C)
            "purge_interval": 180.0,  # seconds
            "min_pressure": 1.5,      # bar
            "stoich_ratio": 1.5,      # Air stoichiometry
            "thermal_mass": 10.0,     # kJ/K
            "cooling_coefficient": 0.5 # kW/K
        }
        
        self.time_since_purge = 0.0
        self.compressor_power = 0.0
        
    def update(self, power_demand: float, dt: float, ambient_temp: float = 293.15) -> Dict[str, Any]:
        """Update fuel cell state."""
        # Limit power demand to available capacity
        max_available = self.specs.max_power * self.state["hydrogen_level"]
        target_power = min(power_demand, max_available)
        
        # Calculate stack current based on power demand
        target_current = (target_power * 1000) / self.state["stack_voltage"]
        current_density = target_current / self.specs.active_area
        
        # Voltage calculation with losses
        nernst_voltage = self.specs.nominal_voltage
        activation_loss = 0.1 * np.log(current_density + 1e-6)
        ohmic_loss = 0.07 * current_density
        concentration_loss = 0.05 * (current_density / self.specs.max_current_density)**2
        
        cell_voltage = nernst_voltage - activation_loss - ohmic_loss - concentration_loss
        self.state["stack_voltage"] = max(
            self.specs.min_cell_voltage * self.specs.stack_count,
            cell_voltage * self.specs.stack_count
        )
        
        # Update current and power
        self.state["stack_current"] = target_current
        self.state["power_output"] = (self.state["stack_voltage"] * 
                                    self.state["stack_current"]) / 1000  # Convert to kW
        
        # Air management
        required_air = self._calculate_air_flow(self.state["stack_current"])
        self.state["air_flow"] = required_air
        self.compressor_power = self._calculate_compressor_power(required_air)
        
        # Thermal management
        self._update_thermal(dt, ambient_temp)
        
        # Hydrogen consumption
        hydrogen_rate = self._calculate_hydrogen_consumption(
            self.state["stack_current"], self.state["efficiency"])
        self.state["hydrogen_level"] -= (hydrogen_rate * dt) / self.specs.hydrogen_capacity
        self.state["hydrogen_level"] = max(0.0, min(1.0, self.state["hydrogen_level"]))
        
        # Membrane management
        self._update_membrane_state(dt)
        
        # Periodic purge check
        self.time_since_purge += dt
        if self.time_since_purge >= self.params["purge_interval"]:
            self._perform_purge()
        
        # Update efficiency
        self._update_efficiency()
        
        return self.state
    
    def _calculate_air_flow(self, current: float) -> float:
        """Calculate required air flow rate."""
        faraday_constant = 96485  # C/mol
        o2_per_electron = 0.25    # mol O2 per mol e-
        air_o2_fraction = 0.21
        
        o2_flow = (current * o2_per_electron) / faraday_constant
        return (o2_flow / air_o2_fraction) * self.params["stoich_ratio"] * 32e-3  # kg/s
    
    def _calculate_compressor_power(self, air_flow: float) -> float:
        """Calculate compressor power consumption."""
        pressure_ratio = self.specs.operating_pressure
        efficiency = 0.7
        gamma = 1.4  # Air heat capacity ratio
        
        power = (air_flow * 287 * 293.15 / efficiency) * \
                ((pressure_ratio)**((gamma-1)/gamma) - 1)
        return power / 1000  # kW
    
    def _update_thermal(self, dt: float, ambient_temp: float):
        """Update thermal state of the system."""
        # Heat generation from inefficiencies
        heat_generation = (1 - self.state["efficiency"]) * self.state["power_output"]
        
        # Cooling power
        temp_difference = self.state["temperature"] - ambient_temp
        cooling_power = temp_difference * self.params["cooling_coefficient"]
        
        # Net heat flow
        net_heat = heat_generation - cooling_power
        
        # Temperature change
        self.state["temperature"] += (net_heat * dt) / self.params["thermal_mass"]
        self.state["temperature"] = min(max(self.params["min_temp"], 
                                          self.state["temperature"]),
                                      self.params["max_temp"])
    
    def _calculate_hydrogen_consumption(self, current: float, efficiency: float) -> float:
        """Calculate hydrogen consumption rate in kg/s."""
        faraday_constant = 96485  # C/mol
        h2_molar_mass = 2.016e-3  # kg/mol
        
        # Theoretical consumption
        h2_rate = (current / (2 * faraday_constant)) * h2_molar_mass
        
        # Add inefficiencies
        return h2_rate / efficiency
    
    def _update_membrane_state(self, dt: float):
        """Update membrane humidity state."""
        target_humidity = 0.85
        if self.state["temperature"] > self.params["optimal_temp"]:
            target_humidity -= 0.1 * (self.state["temperature"] - 
                                    self.params["optimal_temp"]) / 20
        
        humidity_change = (target_humidity - self.state["membrane_humidity"]) * dt
        self.state["membrane_humidity"] += humidity_change
    
    def _perform_purge(self):
        """Perform hydrogen purge cycle."""
        self.time_since_purge = 0.0
        self.state["efficiency"] *= 1.02  # Small efficiency recovery
        self.state["pressure_drop"] = 0.0
    
    def _update_efficiency(self):
        """Update system efficiency based on operating conditions."""
        base_efficiency = 0.65
        
        # Temperature factor
        temp_factor = 1.0 - 0.005 * abs(self.state["temperature"] - 
                                       self.params["optimal_temp"])
        
        # Load factor
        load_factor = 1.0 - 0.1 * abs(self.state["power_output"] / 
                                     self.specs.max_power - 0.7)
        
        # Membrane factor
        membrane_factor = self.state["membrane_humidity"] / 0.85
        
        self.state["efficiency"] = base_efficiency * temp_factor * \
                                 load_factor * membrane_factor
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostic data."""
        return {
            "performance": {
                "power_output": self.state["power_output"],
                "efficiency": self.state["efficiency"],
                "voltage": self.state["stack_voltage"],
                "current": self.state["stack_current"]
            },
            "thermal": {
                "temperature": self.state["temperature"],
                "cooling_power": (self.state["temperature"] - 293.15) * 
                                self.params["cooling_coefficient"]
            },
            "fuel": {
                "hydrogen_level": self.state["hydrogen_level"],
                "air_flow": self.state["air_flow"],
                "compressor_power": self.compressor_power
            },
            "membrane": {
                "humidity": self.state["membrane_humidity"],
                "pressure_drop": self.state["pressure_drop"]
            }
        }


class FuelCellController:
    """Controller for hydrogen fuel cell system."""
    
    def __init__(self, fuel_cell: HydrogenFuelCell, control_rate: float = 100.0):
        self.fuel_cell = fuel_cell
        self.control_rate = control_rate
        self.last_update = 0.0
        self.setpoints = {
            "power": 0.0,
            "temperature": fuel_cell.params["optimal_temp"],
            "min_hydrogen": 0.1
        }
        self.mode = "normal"  # normal, eco, performance
    
    def process_state(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Process fuel cell state and generate control outputs."""
        current_time = state.get("time", 0.0)
        
        if current_time - self.last_update < 1.0 / self.control_rate:
            return {"power_demand": self._calculate_power_demand()}
        
        diagnostics = self.fuel_cell.get_diagnostics()
        
        # Temperature management
        if diagnostics["thermal"]["temperature"] > self.fuel_cell.params["max_temp"] - 5:
            self.mode = "eco"
            self.setpoints["power"] *= 0.8
        
        # Hydrogen management
        if diagnostics["fuel"]["hydrogen_level"] < self.setpoints["min_hydrogen"]:
            self.mode = "eco"
            self.setpoints["power"] *= 0.5
        
        # Membrane protection
        if diagnostics["membrane"]["humidity"] < 0.7:
            self.mode = "eco"
            self.setpoints["power"] *= 0.7
        
        self.last_update = current_time
        return {"power_demand": self._calculate_power_demand()}
    
    def _calculate_power_demand(self) -> float:
        """Calculate power demand based on mode and conditions."""
        base_demand = self.setpoints["power"]
        
        if self.mode == "eco":
            return base_demand * 0.8
        elif self.mode == "performance":
            return min(base_demand * 1.2, self.fuel_cell.specs.max_power)
        
        return base_demand
    
    def set_power_demand(self, power: float, mode: str = "normal"):
        """Set desired power output and operation mode."""
        self.setpoints["power"] = max(0.0, min(power, self.fuel_cell.specs.max_power))
        self.mode = mode