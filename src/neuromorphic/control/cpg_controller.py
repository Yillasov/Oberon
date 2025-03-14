"""
Central Pattern Generator-inspired control algorithm for neuromorphic hardware.

This module implements a simplified central pattern generator for control,
inspired by neural circuits that produce rhythmic motor patterns in animals.
"""
import numpy as np
from typing import Dict, Callable


class Oscillator:
    """Simple implementation of a neural oscillator."""
    
    def __init__(self, frequency: float = 1.0, amplitude: float = 1.0, phase: float = 0.0):
        """
        Initialize oscillator.
        
        Args:
            frequency: Oscillation frequency in Hz
            amplitude: Oscillation amplitude
            phase: Initial phase offset in radians
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.state = 0.0
        self.adaptation = 0.0
        self.adaptation_rate = 0.05
    
    def update(self, dt: float, coupling: float = 0.0) -> float:
        """
        Update oscillator state.
        
        Args:
            dt: Time step
            coupling: Coupling input from other oscillators
            
        Returns:
            Current oscillator output
        """
        # Update phase
        self.phase += 2 * np.pi * self.frequency * dt
        
        # Calculate base oscillation
        self.state = self.amplitude * np.sin(self.phase)
        
        # Apply coupling and adaptation
        self.state += coupling - self.adaptation
        
        # Update adaptation (homeostatic mechanism)
        self.adaptation += self.adaptation_rate * (self.state - 0) * dt
        self.adaptation *= (1.0 - 0.01 * dt)  # Slow decay
        
        return self.state


class CPGNetwork:
    """Simple implementation of a central pattern generator network."""
    
    def __init__(self):
        """Initialize CPG network."""
        # Create oscillators for each control dimension
        self.roll_osc = Oscillator(frequency=0.2, amplitude=0.3, phase=0.0)
        self.pitch_osc = Oscillator(frequency=0.3, amplitude=0.2, phase=np.pi/2)
        self.yaw_osc = Oscillator(frequency=0.1, amplitude=0.2, phase=np.pi)
        self.throttle_osc = Oscillator(frequency=0.05, amplitude=0.1, phase=0.0)
        
        # Coupling weights between oscillators
        self.coupling_matrix = np.array([
            [0.0, 0.1, 0.05, 0.0],   # roll to others
            [0.1, 0.0, 0.1, 0.0],    # pitch to others
            [0.05, 0.1, 0.0, 0.0],   # yaw to others
            [0.0, 0.0, 0.0, 0.0]     # throttle to others
        ])
        
        # Last update time
        self.last_time = 0.0
    
    def update(self, dt: float, sensory_feedback: np.ndarray) -> np.ndarray:
        """
        Update CPG network.
        
        Args:
            dt: Time step
            sensory_feedback: Feedback from sensors (orientation errors)
            
        Returns:
            Control outputs
        """
        # Get current states
        states = np.array([
            self.roll_osc.state,
            self.pitch_osc.state,
            self.yaw_osc.state,
            self.throttle_osc.state
        ])
        
        # Calculate coupling inputs
        coupling = np.dot(self.coupling_matrix, states)
        
        # Update oscillators with coupling and sensory feedback
        roll = self.roll_osc.update(dt, coupling[0] + 0.2 * sensory_feedback[0])
        pitch = self.pitch_osc.update(dt, coupling[1] + 0.2 * sensory_feedback[1])
        yaw = self.yaw_osc.update(dt, coupling[2] + 0.2 * sensory_feedback[2])
        throttle = self.throttle_osc.update(dt, coupling[3] + 0.2 * sensory_feedback[3])
        
        return np.array([roll, pitch, yaw, throttle])


class CPGController:
    """
    Central Pattern Generator-based controller.
    
    This controller uses a CPG network to generate rhythmic
    control patterns with sensory feedback.
    """
    
    def __init__(self):
        """Initialize the CPG controller."""
        # Create CPG network
        self.cpg = CPGNetwork()
        
        # Target values
        self.target_altitude = 10.0  # meters
        
        # Last update time
        self.last_time = 0.0
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using CPG network.
        
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
        
        # Prepare sensory feedback
        sensory_feedback = np.array([
            -orientation[0],  # roll error
            -orientation[1],  # pitch error
            -orientation[2],  # yaw error
            altitude_error    # altitude error
        ])
        
        # Update CPG network
        cpg_outputs = self.cpg.update(dt, sensory_feedback)
        
        # Extract control signals and add direct error correction
        aileron = cpg_outputs[0] - 0.3 * orientation[0]
        elevator = cpg_outputs[1] - 0.3 * orientation[1] + 0.1 * altitude_error
        rudder = cpg_outputs[2] - 0.3 * orientation[2]
        throttle = 0.5 + cpg_outputs[3] + 0.1 * altitude_error
        
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
    Create a CPG controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = CPGController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function
