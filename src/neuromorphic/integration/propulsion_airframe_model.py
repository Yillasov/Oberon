"""
Integrated performance model for propulsion-airframe interactions.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class InteractionMode(Enum):
    NOMINAL = "nominal"
    HIGH_ALPHA = "high_alpha"
    TRANSONIC = "transonic"
    SUPERSONIC = "supersonic"


@dataclass
class AirframeConfig:
    """Airframe configuration parameters."""
    reference_area: float = 25.0     # m²
    wingspan: float = 10.0           # m
    mean_chord: float = 2.5          # m
    engine_position: np.ndarray = np.array([0.0, 0.0, 0.0])  # [x, y, z] in m
    max_thrust_angle: float = 20.0   # degrees
    structural_limit: float = 3.5    # g


class PropulsionAirframeModel:
    """Integrated model for propulsion-airframe interactions."""
    
    def __init__(self, config: AirframeConfig = AirframeConfig()):
        self.config = config
        self.state = {
            'mode': InteractionMode.NOMINAL,
            'thrust_interference': 0.0,
            'inlet_efficiency': 1.0,
            'structural_load': 0.0,
            'thermal_interaction': 0.0
        }
        
        # Performance matrices
        self.interference_matrix = np.zeros((4, 4))  # Thrust interference effects
        self.thermal_matrix = np.zeros((4, 4))       # Thermal coupling effects
        
        self._initialize_matrices()
    
    def _initialize_matrices(self):
        """Initialize interaction matrices."""
        # Thrust interference matrix [alpha, beta, mach, thrust]
        self.interference_matrix = np.array([
            [1.0, 0.02, 0.05, 0.03],
            [0.02, 1.0, 0.04, 0.02],
            [0.05, 0.04, 1.0, 0.06],
            [0.03, 0.02, 0.06, 1.0]
        ])
        
        # Thermal coupling matrix [engine_temp, mach, altitude, load]
        self.thermal_matrix = np.array([
            [1.0, 0.08, 0.03, 0.02],
            [0.08, 1.0, 0.05, 0.03],
            [0.03, 0.05, 1.0, 0.04],
            [0.02, 0.03, 0.04, 1.0]
        ])
    
    def update(self, 
              flight_state: Dict[str, Any],
              engine_state: Dict[str, Any],
              thermal_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update propulsion-airframe interaction model."""
        # Extract relevant states
        alpha = flight_state.get('alpha', 0.0)
        beta = flight_state.get('beta', 0.0)
        mach = flight_state.get('mach', 0.0)
        thrust = engine_state.get('thrust', 0.0)
        
        # Update interaction mode
        self._update_mode(alpha, mach)
        
        # Calculate interactions
        self._calculate_thrust_interference(alpha, beta, mach, thrust)
        self._calculate_inlet_efficiency(alpha, beta, mach)
        self._calculate_structural_loads(thrust, flight_state)
        self._calculate_thermal_interaction(engine_state, thermal_state)
        
        return self.get_status()
    
    def _update_mode(self, alpha: float, mach: float):
        """Update interaction mode based on flight conditions."""
        if mach >= 1.0:
            self.state['mode'] = InteractionMode.SUPERSONIC
        elif mach > 0.8:
            self.state['mode'] = InteractionMode.TRANSONIC
        elif abs(alpha) > 15.0:
            self.state['mode'] = InteractionMode.HIGH_ALPHA
        else:
            self.state['mode'] = InteractionMode.NOMINAL
    
    def _calculate_thrust_interference(self, 
                                    alpha: float, 
                                    beta: float,
                                    mach: float,
                                    thrust: float):
        """Calculate thrust interference effects."""
        state_vector = np.array([alpha, beta, mach, thrust])
        interference = np.dot(self.interference_matrix, state_vector)
        
        # Apply mode-specific corrections
        mode_factors = {
            InteractionMode.NOMINAL: 1.0,
            InteractionMode.HIGH_ALPHA: 1.5,
            InteractionMode.TRANSONIC: 1.3,
            InteractionMode.SUPERSONIC: 0.8
        }
        
        self.state['thrust_interference'] = np.sum(interference) * \
                                          mode_factors[self.state['mode']]
    
    def _calculate_inlet_efficiency(self, alpha: float, beta: float, mach: float):
        """Calculate inlet efficiency based on flow conditions."""
        base_efficiency = 1.0 - 0.1 * abs(alpha/15.0) - 0.05 * abs(beta/10.0)
        
        # Mach effects
        if mach > 1.0:
            base_efficiency *= 0.95
        elif mach > 0.8:
            base_efficiency *= 0.98
        
        self.state['inlet_efficiency'] = max(0.7, base_efficiency)
    
    def _calculate_structural_loads(self, thrust: float, flight_state: Dict[str, Any]):
        """Calculate structural loads from propulsion effects."""
        # Basic thrust-induced load
        thrust_load = thrust / (self.config.reference_area * 1000)  # kN/m²
        
        # Add maneuver loads
        g_load = flight_state.get('g_load', 1.0)
        total_load = thrust_load * g_load
        
        # Add thrust vectoring effects if present
        if 'thrust_vector' in flight_state:
            vector_angle = np.rad2deg(np.arctan2(
                flight_state['thrust_vector'][1],
                flight_state['thrust_vector'][0]
            ))
            total_load *= (1 + abs(vector_angle) / self.config.max_thrust_angle)
        
        self.state['structural_load'] = min(total_load, self.config.structural_limit)
    
    def _calculate_thermal_interaction(self, 
                                    engine_state: Dict[str, Any],
                                    thermal_state: Dict[str, Any]):
        """Calculate thermal interactions between propulsion and airframe."""
        engine_temp = engine_state.get('temperature', 300.0)
        mach = engine_state.get('mach', 0.0)
        altitude = engine_state.get('altitude', 0.0)
        load = self.state['structural_load']
        
        state_vector = np.array([
            engine_temp/1000,  # Normalize temperature
            mach,
            altitude/10000,    # Normalize altitude
            load/self.config.structural_limit
        ])
        
        thermal_coupling = np.dot(self.thermal_matrix, state_vector)
        self.state['thermal_interaction'] = np.sum(thermal_coupling)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current interaction status."""
        return {
            'mode': self.state['mode'].value,
            'thrust_interference': self.state['thrust_interference'],
            'inlet_efficiency': self.state['inlet_efficiency'],
            'structural_load': self.state['structural_load'],
            'thermal_interaction': self.state['thermal_interaction'],
            'performance_limits': self._get_performance_limits()
        }
    
    def _get_performance_limits(self) -> Dict[str, float]:
        """Calculate current performance limits."""
        return {
            'max_thrust_rating': 1.0 - 0.2 * self.state['thermal_interaction'],
            'structural_margin': 1.0 - self.state['structural_load'] / 
                               self.config.structural_limit,
            'thermal_margin': max(0.0, 1.0 - self.state['thermal_interaction'])
        }