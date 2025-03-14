"""
Vestibular-inspired control algorithm for neuromorphic hardware.

This module implements a simplified model of the vestibular system
for balance and orientation control in dynamic environments.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class VestibularSensor:
    """Simulates the vestibular system's sensing capabilities."""
    
    def __init__(self):
        """Initialize vestibular sensor model."""
        # Sensor gains
        self.linear_accel_gain = 1.0
        self.angular_vel_gain = 2.0
        
        # Adaptation parameters
        self.adaptation_rate = 0.05
        self.baseline = np.zeros(6)  # 3 linear + 3 angular
    
    def sense(self, velocity: np.ndarray, angular_velocity: np.ndarray, 
             dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vestibular sensing from motion.
        
        Args:
            velocity: 3D velocity vector
            angular_velocity: 3D angular velocity vector
            dt: Time step in seconds
            
        Returns:
            Tuple of (linear_acceleration, angular_velocity_sensed)
        """
        # Estimate linear acceleration from velocity changes
        # In a real system, we'd have actual accelerometer data
        linear_accel = np.zeros(3)
        if hasattr(self, 'prev_velocity'):
            linear_accel = (velocity - self.prev_velocity) / dt
        self.prev_velocity = velocity.copy()
        
        # Apply sensor gains
        linear_accel_sensed = linear_accel * self.linear_accel_gain
        angular_vel_sensed = angular_velocity * self.angular_vel_gain
        
        # Combine into single sensory vector and adapt baseline
        sensory_input = np.concatenate([linear_accel_sensed, angular_vel_sensed])
        self.baseline += self.adaptation_rate * (sensory_input - self.baseline) * dt
        
        # Return baseline-adapted sensory signals
        adapted_signal = sensory_input - self.baseline
        
        return adapted_signal[:3], adapted_signal[3:]


class VestibularController:
    """
    Vestibular-inspired controller for balance and orientation.
    
    This controller mimics how the vestibular system maintains
    balance and orientation in biological systems.
    """
    
    def __init__(self):
        """Initialize the vestibular controller."""
        self.sensor = VestibularSensor()
        
        # Control gains
        self.roll_gain = 0.8
        self.pitch_gain = 0.6
        self.yaw_gain = 0.3
        self.altitude_gain = 0.5
        
        # Desired orientation (level flight)
        self.target_orientation = np.zeros(3)
        self.target_altitude = 10.0
        
        # Damping factors
        self.damping = np.array([0.2, 0.2, 0.1])
        
        # Previous state for derivative estimation
        self.prev_orientation = None
        self.prev_time = 0.0
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using vestibular model.
        
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
        angular_velocity = state['angular_velocity']
        
        # Calculate time step
        dt = time - self.prev_time if self.prev_time else 0.01
        self.prev_time = time
        
        # Get vestibular sensing
        linear_accel, angular_vel_sensed = self.sensor.sense(velocity, angular_velocity, dt)
        
        # Calculate orientation error (how far from level flight)
        orientation_error = orientation - self.target_orientation
        
        # Calculate altitude error
        altitude_error = position[2] - self.target_altitude
        
        # Initialize previous orientation if needed
        if self.prev_orientation is None:
            self.prev_orientation = orientation.copy()
        
        # Calculate orientation change rate
        orientation_rate = (orientation - self.prev_orientation) / dt
        self.prev_orientation = orientation.copy()
        
        # Compute control outputs using a reflexive approach
        
        # Roll control (aileron) - try to level the wings
        aileron = -self.roll_gain * orientation_error[0] - self.damping[0] * angular_vel_sensed[0]
        
        # Pitch control (elevator) - maintain altitude and level pitch
        elevator = -self.pitch_gain * orientation_error[1] - self.damping[1] * angular_vel_sensed[1]
        elevator -= self.altitude_gain * altitude_error  # Adjust for altitude
        
        # Yaw control (rudder) - dampen yaw oscillations
        rudder = -self.yaw_gain * orientation_error[2] - self.damping[2] * angular_vel_sensed[2]
        
        # Throttle control - maintain altitude
        throttle = 0.5 - 0.1 * altitude_error
        
        # Add vestibular reflexes - compensate for sensed accelerations
        aileron += 0.1 * linear_accel[1]  # Side acceleration affects roll
        elevator -= 0.1 * linear_accel[0]  # Forward acceleration affects pitch
        
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
    Create a vestibular controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = VestibularController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function