"""
Triplet-based STDP learning algorithm for spiking neural networks.
"""
import numpy as np
from typing import List, Dict, Any, Optional


class TripletSTDP:
    """
    Triplet-based STDP learning algorithm.
    
    This algorithm extends pair-based STDP by considering spike triplets,
    which better captures experimental observations of synaptic plasticity.
    """
    
    def __init__(self, 
                 num_inputs: int,
                 num_outputs: int,
                 lr_plus: float = 0.005,    # Learning rate for potentiation
                 lr_minus: float = 0.005,   # Learning rate for depression
                 tau_plus: float = 16.8,    # Time constant for pre-traces
                 tau_minus: float = 33.7,   # Time constant for post-traces
                 tau_x: float = 101.0,      # Time constant for triplet pre-trace
                 tau_y: float = 114.0):     # Time constant for triplet post-trace
        """
        Initialize the Triplet STDP learning algorithm.
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr_plus = lr_plus
        self.lr_minus = lr_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_x = tau_x
        self.tau_y = tau_y
        
        # Initialize weights
        self.weights = np.random.rand(num_outputs, num_inputs) * 0.1
        
        # Initialize traces
        self.pre_trace = np.zeros(num_inputs)       # Standard pre-synaptic trace
        self.post_trace = np.zeros(num_outputs)     # Standard post-synaptic trace
        self.pre_trace_slow = np.zeros(num_inputs)  # Slow pre-synaptic trace for triplets
        self.post_trace_slow = np.zeros(num_outputs) # Slow post-synaptic trace for triplets
        
    def reset(self) -> None:
        """Reset the learning algorithm state."""
        self.pre_trace = np.zeros(self.num_inputs)
        self.post_trace = np.zeros(self.num_outputs)
        self.pre_trace_slow = np.zeros(self.num_inputs)
        self.post_trace_slow = np.zeros(self.num_outputs)
    
    def update(self, input_spikes: np.ndarray, output_spikes: np.ndarray, dt: float) -> None:
        """
        Update weights based on input and output spikes.
        
        Args:
            input_spikes: Input spike pattern
            output_spikes: Output spike pattern
            dt: Time step in milliseconds
        """
        # Decay all traces
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        self.post_trace *= np.exp(-dt / self.tau_minus)
        self.pre_trace_slow *= np.exp(-dt / self.tau_x)
        self.post_trace_slow *= np.exp(-dt / self.tau_y)
        
        # Update traces based on new spikes
        for i in range(self.num_inputs):
            if input_spikes[i]:
                self.pre_trace[i] += 1.0
                self.pre_trace_slow[i] += 1.0
                
                # LTD: pre-spike after post-spike (standard STDP)
                for j in range(self.num_outputs):
                    # Weight update proportional to post-trace and slow post-trace
                    self.weights[j, i] -= self.lr_minus * self.post_trace[j] * (1.0 + self.post_trace_slow[j])
                    
        for j in range(self.num_outputs):
            if output_spikes[j]:
                self.post_trace[j] += 1.0
                self.post_trace_slow[j] += 1.0
                
                # LTP: post-spike after pre-spike (triplet rule)
                for i in range(self.num_inputs):
                    # Weight update proportional to pre-trace and slow pre-trace
                    self.weights[j, i] += self.lr_plus * self.pre_trace[i] * (1.0 + self.pre_trace_slow[i])
        
        # Clip weights to prevent runaway values
        self.weights = np.clip(self.weights, 0.0, 1.0)
    
    def get_weights(self) -> np.ndarray:
        """Get the current weight matrix."""
        return self.weights.copy()