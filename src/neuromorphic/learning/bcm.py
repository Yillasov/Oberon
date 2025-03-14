"""
BCM (Bienenstock-Cooper-Munro) learning rule for spiking neural networks.

This algorithm implements a rate-based learning rule that modulates
synaptic plasticity based on the history of postsynaptic activity.
"""
import numpy as np
from typing import List, Dict, Any, Optional


class BCMLearning:
    """
    BCM (Bienenstock-Cooper-Munro) learning rule.
    
    This algorithm implements a rate-based learning rule where synaptic changes
    depend on the current input activity and a nonlinear function of the 
    postsynaptic activity relative to a dynamic threshold.
    """
    
    def __init__(self, 
                 num_inputs: int,
                 num_outputs: int,
                 learning_rate: float = 0.01,
                 theta_decay: float = 0.01,
                 activity_decay: float = 0.1):
        """
        Initialize the BCM learning algorithm.
        
        Args:
            num_inputs: Number of input neurons
            num_outputs: Number of output neurons
            learning_rate: Learning rate for weight updates
            theta_decay: Decay rate for the sliding threshold
            activity_decay: Decay rate for activity traces
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.theta_decay = theta_decay
        self.activity_decay = activity_decay
        
        # Initialize weights
        self.weights = np.random.rand(num_outputs, num_inputs) * 0.1
        
        # Activity traces
        self.input_activity = np.zeros(num_inputs)
        self.output_activity = np.zeros(num_outputs)
        
        # Sliding modification threshold (θ)
        self.theta = np.ones(num_outputs) * 0.5
        
    def reset(self) -> None:
        """Reset the learning algorithm state."""
        self.input_activity = np.zeros(self.num_inputs)
        self.output_activity = np.zeros(self.num_outputs)
        self.theta = np.ones(self.num_outputs) * 0.5
    
    def update(self, input_spikes: np.ndarray, output_spikes: np.ndarray, dt: float) -> None:
        """
        Update weights based on input and output activity.
        
        Args:
            input_spikes: Input spike pattern
            output_spikes: Output spike pattern
            dt: Time step in milliseconds
        """
        # Update activity traces
        self.input_activity = (1 - self.activity_decay) * self.input_activity + self.activity_decay * input_spikes
        self.output_activity = (1 - self.activity_decay) * self.output_activity + self.activity_decay * output_spikes
        
        # Update sliding threshold (θ)
        self.theta = (1 - self.theta_decay) * self.theta + self.theta_decay * (self.output_activity ** 2)
        
        # Calculate BCM weight updates for each output neuron
        for j in range(self.num_outputs):
            # Phi function: output_activity * (output_activity - theta)
            phi = self.output_activity[j] * (self.output_activity[j] - self.theta[j])
            
            # Update weights based on BCM rule
            self.weights[j] += self.learning_rate * phi * self.input_activity
        
        # Clip weights to prevent runaway values
        self.weights = np.clip(self.weights, 0.0, 1.0)
    
    def get_weights(self) -> np.ndarray:
        """Get the current weight matrix."""
        return self.weights.copy()
    
    def get_thresholds(self) -> np.ndarray:
        """Get the current threshold values."""
        return self.theta.copy()