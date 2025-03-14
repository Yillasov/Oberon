"""
Tropism-inspired control algorithm for neuromorphic hardware.

This module implements an extremely simple controller inspired by
plant tropisms like phototropism or gravitropism.
"""
import numpy as np
from typing import Dict, Callable


class TropismController:
    """
    Extremely simple tropism-based controller.
    
    This controller uses only gravity and sun position as reference,
    similar to how plants orient themselves.
    """
    
    def __init__(self):
        """Initialize the tropism controller."""
        # Sun direction (fixed in sky)
        self.sun_direction = np.array([0.5, 0.5, 1.0])
        self.sun_direction = self.sun_direction / np.linalg.norm(self.sun_direction)
        
        # Gravity direction
        self.gravity_direction = np.array([0.0, 0.0, -1.0])
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using tropism model.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Extract orientation
        orientation = state['orientation']
        
        # Calculate aircraft's up vector (opposite to gravity in aircraft frame)
        roll, pitch, yaw = orientation
        
        # Extremely simplified - just use orientation angles directly
        # Positive roll means right wing down
        # Positive pitch means nose up
        
        # Gravitropism - try to keep wings level
        aileron = -0.5 * roll
        
        # Phototropism - try to point toward sun
        # Simplest possible version - just maintain slight positive pitch
        elevator = 0.2 - 0.3 * pitch
        
        # Minimal yaw control - just try to dampen any yaw
        rudder = -0.2 * yaw
        
        # Fixed throttle for simplicity
        throttle = 0.6
        
        # Prepare control outputs
        controls = {
            'aileron': float(np.clip(aileron, -1.0, 1.0)),
            'elevator': float(np.clip(elevator, -1.0, 1.0)),
            'rudder': float(np.clip(rudder, -1.0, 1.0)),
            'throttle': float(throttle)
        }
        
        return controls


def create_controller() -> Callable:
    """
    Create a tropism controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = TropismController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function