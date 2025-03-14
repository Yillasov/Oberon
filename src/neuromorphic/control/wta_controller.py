"""
Winner-Take-All-inspired control algorithm for neuromorphic hardware.

This module implements a simplified Winner-Take-All network for control,
inspired by competitive neural circuits in biological systems.
"""
import numpy as np
from typing import Dict, Callable, List


class WTANeuron:
    """Simple implementation of a neuron in a Winner-Take-All network."""
    
    def __init__(self, bias: float = 0.0):
        """
        Initialize WTA neuron.
        
        Args:
            bias: Baseline activation bias
        """
        self.activation = 0.0
        self.bias = bias
        self.decay_rate = 0.2
    
    def update(self, input_value: float) -> float:
        """
        Update neuron activation.
        
        Args:
            input_value: Input stimulus to the neuron
            
        Returns:
            Current activation
        """
        # Update activation with input and bias
        self.activation = (1 - self.decay_rate) * self.activation + self.decay_rate * (input_value + self.bias)
        
        return self.activation


class WTANetwork:
    """Simple implementation of a Winner-Take-All neural network."""
    
    def __init__(self, num_neurons: int = 4):
        """
        Initialize WTA network.
        
        Args:
            num_neurons: Number of neurons in the network
        """
        self.neurons = [WTANeuron() for _ in range(num_neurons)]
        self.inhibition_strength = 0.3
    
    def update(self, inputs: List[float]) -> List[float]:
        """
        Update network with new inputs.
        
        Args:
            inputs: Input values for each neuron
            
        Returns:
            Activations of all neurons
        """
        # Update each neuron with its input
        activations = [neuron.update(input_val) for neuron, input_val in zip(self.neurons, inputs)]
        
        # Find the winner (neuron with highest activation)
        winner_idx = np.argmax(activations)
        
        # Apply lateral inhibition from winner to other neurons
        for i, neuron in enumerate(self.neurons):
            if i != winner_idx:
                neuron.activation -= self.inhibition_strength * activations[winner_idx]
                neuron.activation = max(0.0, neuron.activation)  # Ensure non-negative
        
        # Return updated activations
        return [neuron.activation for neuron in self.neurons]


class WTAController:
    """
    Winner-Take-All-based controller.
    
    This controller uses a simple WTA network to select between
    different control behaviors.
    """
    
    def __init__(self):
        """Initialize the WTA controller."""
        # Create WTA network for behavior selection
        self.behavior_network = WTANetwork(num_neurons=4)  # 4 behaviors
        
        # Target values
        self.target_altitude = 10.0  # meters
        
        # Behavior biases
        self.behavior_biases = [0.1, 0.0, 0.0, 0.0]  # Slight bias toward level flight
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using WTA network.
        
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
        
        # Calculate behavior inputs
        behavior_inputs = [
            1.0 - min(1.0, abs(orientation[0]) + abs(orientation[1])),  # Level flight (high when level)
            abs(altitude_error) / 5.0,                                  # Altitude correction
            abs(orientation[2]) / np.pi,                                # Yaw correction
            0.1                                                         # Exploration (constant low input)
        ]
        
        # Add biases to inputs
        behavior_inputs = [inp + bias for inp, bias in zip(behavior_inputs, self.behavior_biases)]
        
        # Update WTA network
        behavior_activations = self.behavior_network.update(behavior_inputs)
        
        # Compute control outputs based on winning behavior
        
        # Level flight behavior
        level_controls = {
            'aileron': -0.5 * orientation[0],
            'elevator': -0.5 * orientation[1],
            'rudder': -0.2 * orientation[2],
            'throttle': 0.5
        }
        
        # Altitude correction behavior
        altitude_controls = {
            'aileron': 0.0,
            'elevator': 0.3 * np.sign(altitude_error),
            'rudder': 0.0,
            'throttle': 0.5 + 0.2 * np.sign(altitude_error)
        }
        
        # Yaw correction behavior
        yaw_controls = {
            'aileron': 0.0,
            'elevator': 0.0,
            'rudder': -0.5 * orientation[2],
            'throttle': 0.5
        }
        
        # Exploration behavior (gentle random movements)
        exploration_controls = {
            'aileron': 0.1 * np.sin(time),
            'elevator': 0.1 * np.cos(time),
            'rudder': 0.1 * np.sin(2 * time),
            'throttle': 0.5
        }
        
        # Combine controls based on behavior activations
        controls = {
            'aileron': 0.0,
            'elevator': 0.0,
            'rudder': 0.0,
            'throttle': 0.0
        }
        
        # Weight each behavior by its activation
        behaviors = [level_controls, altitude_controls, yaw_controls, exploration_controls]
        total_activation = sum(behavior_activations)
        
        if total_activation > 0:
            for control_key in controls:
                controls[control_key] = sum(
                    behavior[control_key] * activation 
                    for behavior, activation in zip(behaviors, behavior_activations)
                ) / total_activation
        
        # Clip control outputs
        for key in controls:
            if key == 'throttle':
                controls[key] = float(np.clip(controls[key], 0.0, 1.0))
            else:
                controls[key] = float(np.clip(controls[key], -1.0, 1.0))
        
        return controls


def create_controller() -> Callable:
    """
    Create a WTA controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = WTAController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function