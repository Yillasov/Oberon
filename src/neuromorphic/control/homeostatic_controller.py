"""
Homeostatic regulation-inspired control algorithm for neuromorphic hardware.

This module implements a simplified homeostatic control system,
inspired by how biological systems maintain internal stability.
"""
import numpy as np
from typing import Dict, Callable


class HomeostaticRegulator:
    """Simple implementation of a homeostatic regulator."""
    
    def __init__(self, setpoint: float = 0.0, gain: float = 1.0):
        """
        Initialize homeostatic regulator.
        
        Args:
            setpoint: Target value to maintain
            gain: Response strength to deviations
        """
        self.setpoint = setpoint
        self.gain = gain
        self.adaptation_rate = 0.01
        self.response_history = []
        self.max_history = 10
    
    def update(self, current_value: float) -> float:
        """
        Update regulator and get corrective response.
        
        Args:
            current_value: Current measured value
            
        Returns:
            Corrective response
        """
        # Calculate error
        error = self.setpoint - current_value
        
        # Calculate response
        response = self.gain * error
        
        # Store response in history
        self.response_history.append(response)
        if len(self.response_history) > self.max_history:
            self.response_history.pop(0)
        
        # Adapt gain based on response history
        if len(self.response_history) >= 3:
            # If responses are oscillating, reduce gain
            if (self.response_history[-1] * self.response_history[-2] < 0 and
                self.response_history[-2] * self.response_history[-3] < 0):
                self.gain *= (1.0 - self.adaptation_rate)
            # If responses are consistently in same direction, increase gain
            elif (self.response_history[-1] * self.response_history[-2] > 0 and
                  self.response_history[-2] * self.response_history[-3] > 0):
                self.gain *= (1.0 + self.adaptation_rate)
        
        return response


class HomeostaticController:
    """
    Homeostatic regulation-based controller.
    
    This controller uses homeostatic regulators to maintain
    stable flight conditions.
    """
    
    def __init__(self):
        """Initialize the homeostatic controller."""
        # Create homeostatic regulators
        self.roll_regulator = HomeostaticRegulator(setpoint=0.0, gain=0.5)
        self.pitch_regulator = HomeostaticRegulator(setpoint=0.0, gain=0.4)
        self.yaw_regulator = HomeostaticRegulator(setpoint=0.0, gain=0.3)
        self.altitude_regulator = HomeostaticRegulator(setpoint=10.0, gain=0.2)
        
        # Energy conservation mechanism
        self.energy_level = 1.0
        self.energy_consumption_rate = 0.001
        self.energy_recovery_rate = 0.0005
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using homeostatic regulation.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Extract state variables
        position = state['position']
        orientation = state['orientation']
        
        # Update energy level
        self.energy_level -= self.energy_consumption_rate
        self.energy_level += self.energy_recovery_rate
        self.energy_level = np.clip(self.energy_level, 0.1, 1.0)
        
        # Scale regulator gains based on energy level
        self.roll_regulator.gain = 0.5 * self.energy_level
        self.pitch_regulator.gain = 0.4 * self.energy_level
        self.yaw_regulator.gain = 0.3 * self.energy_level
        
        # Update regulators
        aileron = self.roll_regulator.update(orientation[0])
        elevator_pitch = self.pitch_regulator.update(orientation[1])
        rudder = self.yaw_regulator.update(orientation[2])
        
        # Altitude regulation
        altitude_response = self.altitude_regulator.update(position[2])
        elevator_alt = 0.1 * altitude_response
        throttle = 0.5 + 0.2 * altitude_response
        
        # Combine pitch responses
        elevator = elevator_pitch + elevator_alt
        
        # Prepare control outputs
        controls = {
            'aileron': float(np.clip(aileron, -1.0, 1.0)),
            'elevator': float(np.clip(elevator, -1.0, 1.0)),
            'rudder': float(np.clip(rudder, -1.0, 1.0)),
            'throttle': float(np.clip(throttle, 0.0, 1.0))
        }
        
        return controls


def create_controller() -> Callable:
    """
    Create a homeostatic controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = HomeostaticController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function