"""
Taxis-inspired control algorithm for neuromorphic hardware.

This module implements a minimal taxis-based controller inspired by
simple biological movement behaviors like phototaxis or chemotaxis.
"""
import numpy as np
from typing import Dict, Callable


class TaxisController:
    """
    Simple taxis-based controller.
    
    This controller mimics how simple organisms move toward or away
    from stimuli (like light, chemicals, or heat).
    """
    
    def __init__(self):
        """Initialize the taxis controller."""
        # Target beacon position
        self.beacon_position = np.array([50.0, 50.0, 10.0])
        
        # Response gains
        self.turn_gain = 0.3
        self.climb_gain = 0.2
        self.speed_gain = 0.1
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using taxis model.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Extract state variables
        position = state['position']
        orientation = state['orientation']
        
        # Calculate vector to beacon
        to_beacon = self.beacon_position - position
        
        # Calculate horizontal direction to beacon
        heading_to_beacon = np.arctan2(to_beacon[1], to_beacon[0])
        
        # Calculate current heading
        current_heading = orientation[2]  # yaw angle
        
        # Calculate heading error (simple version)
        heading_error = heading_to_beacon - current_heading
        
        # Normalize to [-pi, pi]
        if heading_error > np.pi:
            heading_error -= 2 * np.pi
        elif heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Simple taxis responses
        
        # Turn toward beacon
        aileron = self.turn_gain * heading_error
        rudder = self.turn_gain * heading_error
        
        # Climb/descend toward beacon altitude
        altitude_error = to_beacon[2]
        elevator = self.climb_gain * altitude_error
        
        # Speed based on distance
        distance = np.linalg.norm(to_beacon)
        throttle = 0.5 + self.speed_gain * min(distance / 10.0, 1.0)
        
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
    Create a taxis controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = TaxisController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function