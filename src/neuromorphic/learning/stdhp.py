"""
Spike-Timing-Dependent Homeostatic Plasticity (STDHP) learning algorithm.

This module implements a biologically plausible learning algorithm that combines
STDP with homeostatic mechanisms to maintain stable neural activity.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class STDHP:
    """
    Spike-Timing-Dependent Homeostatic Plasticity learning algorithm.
    
    This algorithm extends traditional STDP with homeostatic mechanisms that
    regulate neural activity to maintain stability and prevent runaway dynamics.
    """
    
    def __init__(self, 
                 num_inputs: int,
                 num_outputs: int,
                 learning_rate: float = 0.01,
                 stdp_window: float = 20.0,
                 target_rate: float = 0.1,
                 homeostatic_rate: float = 0.001,
                 activity_window: int = 1000):
        """
        Initialize the STDHP learning algorithm.
        
        Args:
            num_inputs: Number of input neurons
            num_outputs: Number of output neurons
            learning_rate: Base learning rate for STDP weight updates
            stdp_window: Time window for STDP in milliseconds
            target_rate: Target firing rate for homeostatic regulation
            homeostatic_rate: Learning rate for homeostatic adjustments
            activity_window: Window size for calculating average activity
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window
        self.target_rate = target_rate
        self.homeostatic_rate = homeostatic_rate
        self.activity_window = activity_window
        
        # Initialize weights
        self.weights = np.random.randn(num_outputs, num_inputs) * 0.1
        
        # Initialize thresholds for homeostatic regulation
        self.thresholds = np.ones(num_outputs)
        
        # Spike timing records
        self.input_spike_times = np.zeros(num_inputs) - 1000.0
        self.output_spike_times = np.zeros(num_outputs) - 1000.0
        
        # Activity history for homeostatic regulation
        self.activity_history = np.zeros((num_outputs, activity_window))
        self.current_step = 0
        
        # Performance metrics
        self.weight_changes = []
        self.threshold_changes = []
        self.average_activity = np.zeros(num_outputs)
        
    def reset(self) -> None:
        """Reset the learning algorithm state."""
        self.input_spike_times = np.zeros(self.num_inputs) - 1000.0
        self.output_spike_times = np.zeros(self.num_outputs) - 1000.0
        self.activity_history = np.zeros((self.num_outputs, self.activity_window))
        self.current_step = 0
        self.average_activity = np.zeros(self.num_outputs)
        
    def update_weights(self, t: float) -> None:
        """
        Update weights based on STDP rules.
        
        Args:
            t: Current simulation time
        """
        weight_changes = np.zeros_like(self.weights)
        
        # For each output neuron that just spiked
        for i in range(self.num_outputs):
            if t - self.output_spike_times[i] < 1.0:  # If output neuron just spiked
                for j in range(self.num_inputs):
                    # Time difference between input and output spikes
                    dt = self.output_spike_times[i] - self.input_spike_times[j]
                    
                    # If input spike happened before output spike (causal)
                    if 0 < dt < self.stdp_window:
                        # Strengthen connection (LTP)
                        weight_changes[i, j] = self.learning_rate * np.exp(-dt / self.stdp_window)
                    # If input spike happened after output spike (acausal)
                    elif -self.stdp_window < dt < 0:
                        # Weaken connection (LTD)
                        weight_changes[i, j] = -self.learning_rate * np.exp(dt / self.stdp_window)
        
        # Apply weight changes
        self.weights += weight_changes
        
        # Clip weights to prevent runaway values
        self.weights = np.clip(self.weights, 0.0, 1.0)
        
        # Record metrics
        self.weight_changes.append(np.mean(np.abs(weight_changes)))
    
    def update_homeostasis(self) -> None:
        """
        Update homeostatic thresholds based on recent activity.
        """
        # Calculate average activity over the window
        for i in range(self.num_outputs):
            self.average_activity[i] = np.mean(self.activity_history[i])
            
            # Adjust threshold based on difference from target rate
            activity_error = self.average_activity[i] - self.target_rate
            threshold_change = self.homeostatic_rate * activity_error
            
            # Apply threshold change
            self.thresholds[i] += threshold_change
            
            # Ensure thresholds stay positive
            self.thresholds[i] = max(0.1, self.thresholds[i])
        
        # Record metrics
        self.threshold_changes.append(np.mean(np.abs(self.thresholds - 1.0)))
    
    def record_spikes(self, input_spikes: np.ndarray, output_spikes: np.ndarray, t: float) -> None:
        """
        Record spike times and update activity history.
        
        Args:
            input_spikes: Boolean array indicating input spikes
            output_spikes: Boolean array indicating output spikes
            t: Current simulation time
        """
        # Record input spike times
        for i in range(self.num_inputs):
            if input_spikes[i]:
                self.input_spike_times[i] = t
        
        # Record output spike times and update activity history
        for i in range(self.num_outputs):
            if output_spikes[i]:
                self.output_spike_times[i] = t
                
            # Update activity history (1 if spiked, 0 otherwise)
            self.activity_history[i, self.current_step % self.activity_window] = float(output_spikes[i])
        
        # Increment step counter
        self.current_step += 1
        
        # Update homeostasis every activity_window steps
        if self.current_step % self.activity_window == 0:
            self.update_homeostasis()
    
    def get_weights(self) -> np.ndarray:
        """
        Get the current weight matrix.
        
        Returns:
            Current weight matrix
        """
        return self.weights.copy()
    
    def get_thresholds(self) -> np.ndarray:
        """
        Get the current threshold values.
        
        Returns:
            Current threshold values
        """
        return self.thresholds.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set the weight matrix.
        
        Args:
            weights: New weight matrix
        """
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight matrix shape mismatch: expected {self.weights.shape}, got {weights.shape}")
        self.weights = weights.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the learning algorithm.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            "weight_changes": self.weight_changes,
            "threshold_changes": self.threshold_changes,
            "average_activity": self.average_activity.copy(),
            "thresholds": self.thresholds.copy()
        }