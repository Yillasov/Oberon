"""
Neural oscillator-inspired control algorithm for neuromorphic hardware.

This module implements a simple neural oscillator network for generating
rhythmic control patterns similar to biological central pattern generators.
"""
import numpy as np
from typing import Dict, Callable


class NeuralOscillator:
    """Simple implementation of a neural oscillator."""
    
    def __init__(self, frequency: float = 1.0, amplitude: float = 1.0, phase: float = 0.0):
        """
        Initialize neural oscillator.
        
        Args:
            frequency: Oscillation frequency in Hz
            amplitude: Oscillation amplitude
            phase: Initial phase offset in radians
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        
        # Internal state variables
        self.x = 0.0
        self.y = 0.0
        
        # Coupling weights
        self.self_weight = 1.0
        self.cross_weight = -1.0
    
    def update(self, dt: float, external_input: float = 0.0) -> float:
        """
        Update oscillator state and return output.
        
        Args:
            dt: Time step in seconds
            external_input: External input to the oscillator
            
        Returns:
            Current oscillator output
        """
        # Simple harmonic oscillator implementation
        omega = 2.0 * np.pi * self.frequency
        
        # Update phase
        self.phase += omega * dt
        self.phase %= 2.0 * np.pi
        
        # Calculate output
        output = self.amplitude * np.sin(self.phase) + 0.1 * external_input
        
        return output


class OscillatorController:
    """
    Neural oscillator-based controller.
    
    This controller uses coupled neural oscillators to generate
    rhythmic control patterns.
    """
    
    def __init__(self):
        """Initialize the oscillator controller."""
        # Create oscillators for each control dimension
        self.roll_oscillator = NeuralOscillator(frequency=0.2, amplitude=0.2)
        self.pitch_oscillator = NeuralOscillator(frequency=0.1, amplitude=0.1, phase=np.pi/2)
        self.yaw_oscillator = NeuralOscillator(frequency=0.05, amplitude=0.1, phase=np.pi/4)
        
        # Target values
        self.target_altitude = 15.0  # meters
        
        # Control gains
        self.roll_gain = 0.3
        self.pitch_gain = 0.4
        self.yaw_gain = 0.2
        self.altitude_gain = 0.1
        
        # Last update time
        self.last_time = 0.0
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using oscillator network.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Calculate time step
        dt = time - self.last_time if self.last_time > 0 else 0.01
        self.last_time = time
        
        # Extract state variables
        position = state['position']
        orientation = state['orientation']
        
        # Calculate altitude error
        altitude_error = self.target_altitude - position[2]
        
        # Update oscillators with feedback from state
        roll_input = orientation[0]  # Current roll
        pitch_input = orientation[1]  # Current pitch
        yaw_input = orientation[2]  # Current yaw
        
        roll_output = self.roll_oscillator.update(dt, roll_input)
        pitch_output = self.pitch_oscillator.update(dt, pitch_input)
        yaw_output = self.yaw_oscillator.update(dt, yaw_input)
        
        # Compute control outputs
        
        # Aileron - oscillatory pattern with roll stabilization
        aileron = roll_output - self.roll_gain * orientation[0]
        
        # Elevator - oscillatory pattern with pitch stabilization and altitude control
        elevator = pitch_output - self.pitch_gain * orientation[1] + self.altitude_gain * altitude_error
        
        # Rudder - oscillatory pattern with yaw damping
        rudder = yaw_output - self.yaw_gain * orientation[2]
        
        # Throttle - maintain altitude
        throttle = 0.5 + 0.1 * altitude_error
        
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
    Create an oscillator controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = OscillatorController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function