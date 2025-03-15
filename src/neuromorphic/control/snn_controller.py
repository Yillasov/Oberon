"""
Spiking Neural Network implementations optimized for control systems.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import time

class SNNController:
    """Base class for SNN-based controllers."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        """
        Initialize SNN controller.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            hidden_size: Number of hidden neurons
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # Initialize neuron states
        self.membrane_potentials = np.zeros(hidden_size + output_size)
        self.refractory_counters = np.zeros(hidden_size + output_size)
        
        # Initialize weights with random values
        self.weights_ih = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.weights_ho = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
        
        # Neuron parameters
        self.thresholds = np.ones(hidden_size + output_size) * 1.0
        self.decay_factors = np.ones(hidden_size + output_size) * 0.9
        self.refractory_period = 2  # timesteps
        
        # Spike history for analysis
        self.spike_history = []
        self.membrane_history = []
        
    def reset(self):
        """Reset the network state."""
        self.membrane_potentials = np.zeros(self.hidden_size + self.output_size)
        self.refractory_counters = np.zeros(self.hidden_size + self.output_size)
        self.spike_history = []
        self.membrane_history = []
        
    def step(self, input_spikes: np.ndarray) -> np.ndarray:
        """
        Run one timestep of the SNN.
        
        Args:
            input_spikes: Binary input spike vector
            
        Returns:
            Binary output spike vector
        """
        # Ensure input is binary
        input_spikes = (input_spikes > 0).astype(np.float32)
        
        # Forward pass to hidden layer
        hidden_inputs = np.dot(self.weights_ih, input_spikes)
        
        # Update hidden neuron states
        self._update_neurons(hidden_inputs, 0, self.hidden_size)
        
        # Get hidden layer spikes
        hidden_spikes = (self.membrane_potentials[:self.hidden_size] >= 
                         self.thresholds[:self.hidden_size]).astype(np.float32)
        
        # Reset membrane potential for neurons that spiked
        self.membrane_potentials[:self.hidden_size] *= (1 - hidden_spikes)
        
        # Set refractory period for neurons that spiked
        self.refractory_counters[:self.hidden_size] += hidden_spikes * self.refractory_period
        
        # Forward pass to output layer
        output_inputs = np.dot(self.weights_ho, hidden_spikes)
        
        # Update output neuron states
        self._update_neurons(output_inputs, self.hidden_size, self.hidden_size + self.output_size)
        
        # Get output layer spikes
        output_spikes = (self.membrane_potentials[self.hidden_size:] >= 
                         self.thresholds[self.hidden_size:]).astype(np.float32)
        
        # Reset membrane potential for neurons that spiked
        self.membrane_potentials[self.hidden_size:] *= (1 - output_spikes)
        
        # Set refractory period for neurons that spiked
        self.refractory_counters[self.hidden_size:] += output_spikes * self.refractory_period
        
        # Store spike and membrane history
        self.spike_history.append(np.concatenate([hidden_spikes, output_spikes]))
        self.membrane_history.append(self.membrane_potentials.copy())
        
        return output_spikes
        
    def _update_neurons(self, inputs: np.ndarray, start_idx: int, end_idx: int):
        """Update neuron states for a layer."""
        # Decrease refractory counters
        self.refractory_counters[start_idx:end_idx] = np.maximum(
            0, self.refractory_counters[start_idx:end_idx] - 1)
        
        # Only update neurons not in refractory period
        active_mask = (self.refractory_counters[start_idx:end_idx] == 0)
        
        # Apply decay to membrane potentials
        self.membrane_potentials[start_idx:end_idx] *= self.decay_factors[start_idx:end_idx]
        
        # Add inputs to membrane potentials for active neurons
        self.membrane_potentials[start_idx:end_idx] += inputs * active_mask
        
    def get_control_output(self, output_spikes: np.ndarray, 
                          integration_window: int = 10) -> np.ndarray:
        """
        Convert output spikes to control signals.
        
        Args:
            output_spikes: Binary output spike vector
            integration_window: Number of timesteps to integrate
            
        Returns:
            Control signal values
        """
        # Use spike history for temporal integration
        if len(self.spike_history) < integration_window:
            window = self.spike_history
        else:
            window = self.spike_history[-integration_window:]
        
        # Extract output neuron spikes from history
        output_history = np.array([spikes[-self.output_size:] for spikes in window])
        
        # Compute spike rates over the window
        spike_rates = np.sum(output_history, axis=0) / len(window)
        
        return spike_rates


class PIDSNNController(SNNController):
    """SNN implementation of a PID controller."""
    
    def __init__(self, input_size: int, output_size: int, 
                kp: float = 1.0, ki: float = 0.5, kd: float = 0.2):
        """
        Initialize PID SNN controller.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        # Create separate populations for P, I, and D components
        hidden_size = 3 * input_size
        super().__init__(input_size, output_size, hidden_size)
        
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Error history for integral and derivative
        self.error_history = []
        
        # Specialized connection weights for PID
        self._setup_pid_weights()
        
    def _setup_pid_weights(self):
        """Set up specialized weights for PID control."""
        # Divide hidden layer into P, I, and D populations
        p_size = i_size = d_size = self.input_size
        
        # P neurons receive direct connections
        self.weights_ih[:p_size] = np.eye(self.input_size) * self.kp
        
        # I neurons have stronger recurrent connections
        self.weights_ih[p_size:p_size+i_size] = np.eye(self.input_size) * self.ki
        
        # D neurons have inhibitory connections to detect changes
        self.weights_ih[p_size+i_size:] = np.eye(self.input_size) * self.kd
        
        # Adjust decay factors for different populations
        # P neurons: standard decay
        # I neurons: slower decay (integration)
        # D neurons: faster decay (differentiation)
        self.decay_factors[:p_size] = 0.9
        self.decay_factors[p_size:p_size+i_size] = 0.99  # Slower decay for integration
        self.decay_factors[p_size+i_size:self.hidden_size] = 0.7  # Faster decay for differentiation
        
    def step(self, input_spikes: np.ndarray) -> np.ndarray:
        """
        Run one timestep of the PID SNN.
        
        Args:
            input_spikes: Binary input spike vector (error signal)
            
        Returns:
            Binary output spike vector
        """
        # Store error for history
        self.error_history.append(input_spikes.copy())
        
        # Compute derivative component (if we have history)
        if len(self.error_history) > 1:
            derivative = input_spikes - self.error_history[-2]
        else:
            derivative = np.zeros_like(input_spikes)
        
        # Prepare combined input: [error, error, derivative]
        p_size = i_size = d_size = self.input_size
        combined_input = np.zeros(3 * self.input_size)
        combined_input[:p_size] = input_spikes  # P component
        combined_input[p_size:p_size+i_size] = input_spikes  # I component
        combined_input[p_size+i_size:] = derivative  # D component
        
        # Process through the network
        return super().step(combined_input)


class AdaptiveSNNController(SNNController):
    """SNN controller with adaptive thresholds for improved stability."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64,
                adaptation_rate: float = 0.05):
        """
        Initialize adaptive SNN controller.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            hidden_size: Number of hidden neurons
            adaptation_rate: Rate of threshold adaptation
        """
        super().__init__(input_size, output_size, hidden_size)
        self.adaptation_rate = adaptation_rate
        self.threshold_baseline = self.thresholds.copy()
        
    def step(self, input_spikes: np.ndarray) -> np.ndarray:
        """
        Run one timestep of the adaptive SNN.
        
        Args:
            input_spikes: Binary input spike vector
            
        Returns:
            Binary output spike vector
        """
        # Run standard step
        output_spikes = super().step(input_spikes)
        
        # Adapt thresholds based on recent activity
        if len(self.spike_history) > 1:
            recent_activity = self.spike_history[-1]
            
            # Increase thresholds for neurons that spiked (homeostasis)
            self.thresholds += recent_activity * self.adaptation_rate
            
            # Gradually return to baseline
            self.thresholds = (0.99 * self.thresholds + 
                              0.01 * self.threshold_baseline)
        
        return output_spikes