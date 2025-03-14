"""
Reflex-inspired control algorithm for neuromorphic hardware.

This module implements a minimal reflex-based controller inspired by
simple biological reflex arcs.
"""
import numpy as np
from typing import Dict, Callable


class ReflexController:
    """
    Simple reflex-based controller.
    
    This controller uses direct stimulus-response mappings similar to
    biological reflex arcs, with minimal processing.
    """
    
    def __init__(self):
        """Initialize the reflex controller."""
        # Target values
        self.target_altitude = 10.0  # meters
        self.target_speed = 4.0      # m/s
        
        # Response thresholds
        self.roll_threshold = 0.2    # radians
        self.pitch_threshold = 0.15  # radians
        
        # Response strengths
        self.roll_response = 0.5
        self.pitch_response = 0.4
        self.altitude_response = 0.3
        self.speed_response = 0.2
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using reflex model.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Extract state variables
        position = state['position']
        velocity = state['velocity']
        orientation = state['orientation']
        
        # Simple reflex responses
        
        # Roll stabilization reflex
        roll = orientation[0]
        if abs(roll) > self.roll_threshold:
            aileron = -self.roll_response * np.sign(roll)
        else:
            aileron = 0.0
        
        # Pitch stabilization reflex
        pitch = orientation[1]
        if abs(pitch) > self.pitch_threshold:
            elevator_pitch = -self.pitch_response * np.sign(pitch)
        else:
            elevator_pitch = 0.0
        
        # Altitude maintenance reflex
        altitude_error = self.target_altitude - position[2]
        elevator_alt = self.altitude_response * np.sign(altitude_error)
        
        # Combine pitch responses
        elevator = elevator_pitch + elevator_alt
        
        # Speed maintenance reflex
        current_speed = np.linalg.norm(velocity[:2])
        speed_error = self.target_speed - current_speed
        throttle = 0.5 + self.speed_response * np.sign(speed_error)
        
        # Simple yaw damping
        rudder = -0.1 * orientation[2]
        
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
    Create a reflex controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = ReflexController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function