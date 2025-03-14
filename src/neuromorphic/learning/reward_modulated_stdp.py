"""
Reward-modulated STDP learning algorithm for spiking neural networks.

This module implements a biologically plausible reinforcement learning algorithm
based on reward-modulated spike-timing-dependent plasticity.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time


class RewardModulatedSTDP:
    """
    Reward-modulated STDP learning algorithm.
    
    This algorithm extends traditional STDP with a reward signal that modulates
    the strength of weight updates, enabling reinforcement learning in SNNs.
    """
    
    def __init__(self, 
                 num_inputs: int,
                 num_outputs: int,
                 learning_rate: float = 0.01,
                 eligibility_trace_decay: float = 0.95,
                 reward_baseline_decay: float = 0.01,
                 stdp_window: float = 20.0):
        """
        Initialize the reward-modulated STDP learning algorithm.
        
        Args:
            num_inputs: Number of input neurons
            num_outputs: Number of output neurons
            learning_rate: Base learning rate for weight updates
            eligibility_trace_decay: Decay factor for eligibility traces
            reward_baseline_decay: Decay factor for reward baseline
            stdp_window: Time window for STDP in milliseconds
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.eligibility_trace_decay = eligibility_trace_decay
        self.reward_baseline_decay = reward_baseline_decay
        self.stdp_window = stdp_window
        
        # Initialize weights
        self.weights = np.random.randn(num_outputs, num_inputs) * 0.1
        
        # Initialize eligibility traces
        self.eligibility_traces = np.zeros((num_outputs, num_inputs))
        
        # Initialize reward baseline (for reward prediction)
        self.reward_baseline = 0.0
        
        # Spike timing records
        self.input_spike_times = np.zeros(num_inputs) - 1000.0
        self.output_spike_times = np.zeros(num_outputs) - 1000.0
        
        # Performance metrics
        self.total_reward = 0.0
        self.reward_history = []
        self.weight_changes = []
        
    def reset(self) -> None:
        """Reset the learning algorithm state."""
        self.eligibility_traces = np.zeros((self.num_outputs, self.num_inputs))
        self.input_spike_times = np.zeros(self.num_inputs) - 1000.0
        self.output_spike_times = np.zeros(self.num_outputs) - 1000.0
        
    def update_eligibility_traces(self, t: float) -> None:
        """
        Update eligibility traces based on recent spike timing.
        
        Args:
            t: Current simulation time
        """
        # Decay existing eligibility traces
        self.eligibility_traces *= self.eligibility_trace_decay
        
        # Update traces based on STDP rules
        for i in range(self.num_outputs):
            if t - self.output_spike_times[i] < 1.0:  # If output neuron just spiked
                for j in range(self.num_inputs):
                    # Time difference between input and output spikes
                    dt = self.output_spike_times[i] - self.input_spike_times[j]
                    
                    # If input spike happened before output spike (causal)
                    if 0 < dt < self.stdp_window:
                        # Positive eligibility (strengthen connection)
                        self.eligibility_traces[i, j] += np.exp(-dt / self.stdp_window)
                    # If input spike happened after output spike (acausal)
                    elif -self.stdp_window < dt < 0:
                        # Negative eligibility (weaken connection)
                        self.eligibility_traces[i, j] -= np.exp(dt / self.stdp_window)
    
    def apply_reward(self, reward: float) -> None:
        """
        Apply reward signal to update weights based on eligibility traces.
        
        Args:
            reward: Reward signal value
        """
        # Update reward baseline (moving average)
        reward_prediction_error = reward - self.reward_baseline
        self.reward_baseline += self.reward_baseline_decay * reward_prediction_error
        
        # Update weights based on reward prediction error and eligibility traces
        delta_w = self.learning_rate * reward_prediction_error * self.eligibility_traces
        self.weights += delta_w
        
        # Clip weights to prevent runaway values
        self.weights = np.clip(self.weights, -1.0, 1.0)
        
        # Record metrics
        self.total_reward += reward
        self.reward_history.append(reward)
        self.weight_changes.append(np.mean(np.abs(delta_w)))
    
    def record_spikes(self, input_spikes: np.ndarray, output_spikes: np.ndarray, t: float) -> None:
        """
        Record spike times for inputs and outputs.
        
        Args:
            input_spikes: Boolean array indicating input spikes
            output_spikes: Boolean array indicating output spikes
            t: Current simulation time
        """
        # Record input spike times
        for i in range(self.num_inputs):
            if input_spikes[i]:
                self.input_spike_times[i] = t
        
        # Record output spike times
        for i in range(self.num_outputs):
            if output_spikes[i]:
                self.output_spike_times[i] = t
    
    def get_weights(self) -> np.ndarray:
        """
        Get the current weight matrix.
        
        Returns:
            Current weight matrix
        """
        return self.weights.copy()
    
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
            "total_reward": self.total_reward,
            "reward_history": self.reward_history,
            "weight_changes": self.weight_changes,
            "reward_baseline": self.reward_baseline
        }