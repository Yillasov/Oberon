"""
Hebbian learning-inspired control algorithm for neuromorphic hardware.

This module implements a simplified Hebbian learning network for control,
inspired by the principle that "neurons that fire together, wire together".
"""
import numpy as np
from typing import Dict, Callable


class HebbianNeuron:
    """Simple implementation of a neuron with Hebbian learning."""
    
    def __init__(self, num_inputs: int = 3):
        """
        Initialize Hebbian neuron.
        
        Args:
            num_inputs: Number of input connections
        """
        self.weights = np.zeros(num_inputs)
        self.activation = 0.0
        self.learning_rate = 0.01
        self.decay_rate = 0.001
    
    def update(self, inputs: np.ndarray) -> float:
        """
        Update neuron activation and weights.
        
        Args:
            inputs: Input values
            
        Returns:
            Current activation
        """
        # Calculate activation
        self.activation = np.dot(self.weights, inputs)
        
        # Apply Hebbian learning: strengthen weights for correlated inputs
        for i in range(len(self.weights)):
            # Weight change proportional to input and output correlation
            self.weights[i] += self.learning_rate * self.activation * inputs[i]
            
            # Weight decay to prevent unbounded growth
            self.weights[i] -= self.decay_rate * self.weights[i]
        
        return self.activation


class ThrustOscillationNeuron(HebbianNeuron):
    """Specialized Hebbian neuron for thrust oscillation control."""
    
    def __init__(self, num_inputs: int = 4):
        super().__init__(num_inputs)
        self.frequency_memory = np.zeros(10)  # Remember recent frequencies
        self.memory_index = 0
        self.resonance_weight = 0.3
    
    def update_with_frequency(self, inputs: np.ndarray, frequency: float) -> float:
        # Store frequency in memory
        self.frequency_memory[self.memory_index] = frequency
        self.memory_index = (self.memory_index + 1) % len(self.frequency_memory)
        
        # Add frequency-based modulation
        resonance = np.sin(2 * np.pi * np.mean(self.frequency_memory) * self.resonance_weight)
        
        # Update with standard Hebbian learning
        activation = super().update(inputs)
        
        return activation * (1.0 + 0.2 * resonance)


class HebbianController:
    """
    Hebbian learning-based controller.
    
    This controller uses a simple Hebbian learning network to adapt
    control responses based on experience.
    """
    
    def __init__(self):
        """Initialize the Hebbian controller."""
        # Create control neurons
        self.aileron_neuron = HebbianNeuron(num_inputs=3)  # roll, pitch, yaw
        self.elevator_neuron = HebbianNeuron(num_inputs=3)
        self.rudder_neuron = HebbianNeuron(num_inputs=3)
        self.throttle_neuron = HebbianNeuron(num_inputs=3)
        
        # Target values
        self.target_altitude = 10.0  # meters
        
        # Initial weights to provide basic stability
        self.aileron_neuron.weights = np.array([-0.5, 0.0, 0.0])  # Respond to roll
        self.elevator_neuron.weights = np.array([0.0, -0.5, 0.0])  # Respond to pitch
        self.rudder_neuron.weights = np.array([0.0, 0.0, -0.3])    # Respond to yaw
        
        # Success signal
        self.prev_altitude_error = 0.0
        
        # Add thrust oscillation control
        self.thrust_neuron = ThrustOscillationNeuron(num_inputs=4)
        self.oscillation_history = []
        self.damping_factor = 0.2
        
        # Oscillation state
        self.oscillation_state = {
            'frequency': 0.0,
            'amplitude': 0.0,
            'phase': 0.0,
            'damping': self.damping_factor
        }
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using Hebbian learning.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Extract state variables
        position = state['position']
        orientation = state['orientation']
        
        # Calculate altitude error
        altitude_error = self.target_altitude - position[2]
        
        # Calculate success signal (reduction in altitude error)
        success = abs(self.prev_altitude_error) - abs(altitude_error)
        self.prev_altitude_error = altitude_error
        
        # Adjust learning rate based on success
        learning_rate_modifier = 1.0 + success
        self.aileron_neuron.learning_rate = 0.01 * max(0.1, learning_rate_modifier)
        self.elevator_neuron.learning_rate = 0.01 * max(0.1, learning_rate_modifier)
        self.rudder_neuron.learning_rate = 0.01 * max(0.1, learning_rate_modifier)
        self.throttle_neuron.learning_rate = 0.01 * max(0.1, learning_rate_modifier)
        
        # Update neurons with orientation inputs
        aileron = self.aileron_neuron.update(orientation)
        elevator = self.elevator_neuron.update(orientation)
        rudder = self.rudder_neuron.update(orientation)
        
        # Throttle depends on altitude error
        throttle_inputs = np.array([altitude_error, orientation[1], 0.0])
        throttle = 0.5 + 0.1 * self.throttle_neuron.update(throttle_inputs)
        
        # Prepare control outputs
        controls = {
            'aileron': float(np.clip(aileron, -1.0, 1.0)),
            'elevator': float(np.clip(elevator + 0.1 * altitude_error, -1.0, 1.0)),
            'rudder': float(np.clip(rudder, -1.0, 1.0)),
            'throttle': float(np.clip(throttle, 0.0, 1.0))
        }
        
        return controls


def create_controller() -> Callable:
    """
    Create a Hebbian controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = HebbianController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function
