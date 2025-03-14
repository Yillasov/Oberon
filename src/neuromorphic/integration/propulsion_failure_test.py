"""
Basic test scenarios for propulsion failure modes.
"""
import numpy as np
from typing import Dict, Any
from enum import Enum


class FailureMode(Enum):
    NOMINAL = "nominal"
    THRUST_LOSS = "thrust_loss"
    THERMAL_RUNAWAY = "thermal_runaway"
    INLET_BLOCKAGE = "inlet_blockage"


class PropulsionFailureTest:
    """Basic propulsion failure mode testing."""
    
    def __init__(self):
        self.current_mode = FailureMode.NOMINAL
        self.failure_intensity = 0.0
        self.test_duration = 0.0
        
    def inject_failure(self, mode: FailureMode, intensity: float = 0.5):
        """Inject specific failure mode."""
        self.current_mode = mode
        self.failure_intensity = np.clip(intensity, 0.0, 1.0)
        self.test_duration = 0.0
    
    def update(self, dt: float, engine_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update failure simulation."""
        self.test_duration += dt
        
        if self.current_mode == FailureMode.NOMINAL:
            return engine_state.copy()
        
        modified_state = engine_state.copy()
        
        if self.current_mode == FailureMode.THRUST_LOSS:
            modified_state['thrust'] *= (1.0 - self.failure_intensity)
            modified_state['efficiency'] *= (1.0 - 0.5 * self.failure_intensity)
            
        elif self.current_mode == FailureMode.THERMAL_RUNAWAY:
            temp_increase = 100 * self.failure_intensity * self.test_duration
            modified_state['temperature'] += temp_increase
            modified_state['efficiency'] *= max(0.3, 1.0 - 0.3 * self.failure_intensity)
            
        elif self.current_mode == FailureMode.INLET_BLOCKAGE:
            modified_state['inlet_efficiency'] *= (1.0 - self.failure_intensity)
            modified_state['thrust'] *= (1.0 - 0.8 * self.failure_intensity)
        
        return modified_state
    
    def get_status(self) -> Dict[str, Any]:
        """Get current test status."""
        return {
            'mode': self.current_mode.value,
            'intensity': self.failure_intensity,
            'duration': self.test_duration
        }


def run_basic_test_scenario():
    """Run a basic failure test sequence."""
    failure_test = PropulsionFailureTest()
    
    # Sample engine state
    engine_state = {
        'thrust': 50000.0,
        'temperature': 800.0,
        'efficiency': 0.9,
        'inlet_efficiency': 1.0
    }
    
    # Test sequence
    test_sequence = [
        (FailureMode.THRUST_LOSS, 0.3),
        (FailureMode.THERMAL_RUNAWAY, 0.4),
        (FailureMode.INLET_BLOCKAGE, 0.5)
    ]
    
    results = []
    dt = 0.1
    
    for mode, intensity in test_sequence:
        # Inject failure
        failure_test.inject_failure(mode, intensity)
        
        # Run for 2 seconds
        for _ in range(20):
            modified_state = failure_test.update(dt, engine_state)
            results.append({
                'time': failure_test.test_duration,
                'mode': mode.value,
                'state': modified_state
            })
    
    return results