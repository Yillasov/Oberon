"""
Synfire chain-inspired control algorithm for neuromorphic hardware.

This module implements a simplified synfire chain for control,
inspired by how neural circuits propagate activity in a precise sequence.
"""
import numpy as np
from typing import Dict, Callable, List


class SynfireNode:
    """Simple implementation of a node in a synfire chain."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize synfire node.
        
        Args:
            threshold: Activation threshold
        """
        self.activation = 0.0
        self.threshold = threshold
        self.decay_rate = 0.2
    
    def update(self, input_value: float) -> float:
        """
        Update node activation.
        
        Args:
            input_value: Input stimulus to the node
            
        Returns:
            Current activation
        """
        # Update activation with input and decay
        self.activation = (1 - self.decay_rate) * self.activation + input_value
        
        # Threshold activation
        output = 0.0
        if self.activation > self.threshold:
            output = self.activation
            self.activation *= 0.5  # Reset after firing
        
        return output


class SynfireChain:
    """Simple implementation of a synfire chain."""
    
    def __init__(self, length: int = 5):
        """
        Initialize synfire chain.
        
        Args:
            length: Number of nodes in the chain
        """
        self.nodes = [SynfireNode(threshold=0.5 - 0.05 * i) for i in range(length)]
        self.outputs = np.zeros(length)
    
    def update(self, input_value: float) -> np.ndarray:
        """
        Update chain with new input.
        
        Args:
            input_value: Input value to the first node
            
        Returns:
            Outputs of all nodes
        """
        # Input to first node
        self.outputs[0] = self.nodes[0].update(input_value)
        
        # Propagate through chain
        for i in range(1, len(self.nodes)):
            self.outputs[i] = self.nodes[i].update(self.outputs[i-1])
        
        return self.outputs


class SynfireController:
    """
    Synfire chain-based controller.
    
    This controller uses synfire chains to generate sequential
    control patterns.
    """
    
    def __init__(self):
        """Initialize the synfire controller."""
        # Create synfire chains for each control dimension
        self.roll_chain = SynfireChain(length=5)
        self.pitch_chain = SynfireChain(length=5)
        self.yaw_chain = SynfireChain(length=5)
        
        # Target values
        self.target_altitude = 10.0  # meters
        
        # Control weights
        self.roll_weights = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        self.pitch_weights = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        self.yaw_weights = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        
        # Last update time
        self.last_time = 0.0
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using synfire chains.
        
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
        
        # Update synfire chains with orientation errors
        roll_outputs = self.roll_chain.update(abs(orientation[0]))
        pitch_outputs = self.pitch_chain.update(abs(orientation[1]))
        yaw_outputs = self.yaw_chain.update(abs(orientation[2]))
        
        # Compute control outputs using weighted sum of chain outputs
        aileron = -np.sign(orientation[0]) * np.dot(roll_outputs, self.roll_weights)
        elevator = -np.sign(orientation[1]) * np.dot(pitch_outputs, self.pitch_weights)
        rudder = -np.sign(orientation[2]) * np.dot(yaw_outputs, self.yaw_weights)
        
        # Add altitude correction to elevator
        elevator += 0.1 * altitude_error
        
        # Throttle based on altitude
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
    Create a synfire controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = SynfireController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function