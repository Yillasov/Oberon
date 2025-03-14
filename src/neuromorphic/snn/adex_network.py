"""
Spiking Neural Network implementation using Adaptive Exponential Integrate-and-Fire neurons.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .adex_neuron import AdExNeuron


class AdExNetwork:
    """
    Spiking Neural Network using Adaptive Exponential Integrate-and-Fire neurons.
    
    This class implements an SNN with AdEx neurons, which provide a good balance
    between biological realism and computational efficiency.
    """
    
    def __init__(self, num_inputs: int, num_neurons: int, neuron_params: Optional[Dict[str, float]] = None):
        """
        Initialize an AdEx Neural Network.
        
        Args:
            num_inputs: Number of input channels
            num_neurons: Number of neurons in the network
            neuron_params: Optional parameters for the AdEx neurons
        """
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        
        # Create neurons
        self.neurons = []
        for _ in range(num_neurons):
            if neuron_params:
                self.neurons.append(AdExNeuron(**neuron_params))
            else:
                self.neurons.append(AdExNeuron())
        
        # Initialize weights randomly
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.1
        
        # Spike history
        self.spike_history: List[List[bool]] = []
        
        # Synaptic plasticity parameters (STDP)
        self.enable_plasticity = False
        self.stdp_lr = 0.01
        self.stdp_window = 20.0  # ms
        self.last_input_spike_times = np.zeros(num_inputs) - 1000.0
        
    def reset(self) -> None:
        """Reset the network state."""
        for neuron in self.neurons:
            neuron.reset()
        self.spike_history = []
        self.last_input_spike_times = np.zeros(self.num_inputs) - 1000.0
        
    def _apply_stdp(self, neuron_idx: int, t: float) -> None:
        """
        Apply spike-timing-dependent plasticity to update weights.
        
        Args:
            neuron_idx: Index of the neuron that just spiked
            t: Current simulation time
        """
        if not self.enable_plasticity:
            return
            
        # For each input, update the weight based on timing difference
        for i in range(self.num_inputs):
            # Time difference between input and output spikes
            dt = t - self.last_input_spike_times[i]
            
            # If input spike happened before output spike (causal)
            if 0 < dt < self.stdp_window:
                # Strengthen connection (long-term potentiation)
                self.weights[neuron_idx, i] += self.stdp_lr * np.exp(-dt / self.stdp_window)
            # If input spike happened after output spike (acausal)
            elif -self.stdp_window < dt < 0:
                # Weaken connection (long-term depression)
                self.weights[neuron_idx, i] -= self.stdp_lr * np.exp(dt / self.stdp_window)
    
    def step(self, inputs: np.ndarray, dt: float, t: float) -> np.ndarray:
        """
        Simulate one time step of the network.
        
        Args:
            inputs: Input signals (shape: num_inputs)
            dt: Time step in milliseconds
            t: Current simulation time
            
        Returns:
            Array of output spikes (shape: num_neurons)
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Record input spike times for STDP
        if self.enable_plasticity:
            for i in range(self.num_inputs):
                if inputs[i] > 0:
                    self.last_input_spike_times[i] = t
        
        # Calculate input currents for each neuron
        currents = np.dot(self.weights, inputs)
        
        # Update each neuron and collect spikes
        spikes = np.zeros(self.num_neurons, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.update(currents[i], dt, t)
            
            # Apply STDP if the neuron spiked
            if spikes[i] and self.enable_plasticity:
                self._apply_stdp(i, t)
        
        # Record spike history
        self.spike_history.append(spikes.tolist())
        
        return spikes
    
    def simulate(self, input_signals: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate the network for multiple time steps.
        
        Args:
            input_signals: Input signals (shape: [time_steps, num_inputs])
            dt: Time step in milliseconds
            
        Returns:
            Array of output spikes (shape: [time_steps, num_neurons])
        """
        time_steps = input_signals.shape[0]
        output_spikes = np.zeros((time_steps, self.num_neurons), dtype=bool)
        
        # Reset network state
        self.reset()
        
        # Simulate each time step
        for t in range(time_steps):
            output_spikes[t] = self.step(input_signals[t], dt, t * dt)
        
        return output_spikes
    
    def enable_stdp(self, learning_rate: float = 0.01, window: float = 20.0) -> None:
        """
        Enable spike-timing-dependent plasticity for learning.
        
        Args:
            learning_rate: Learning rate for STDP
            window: Time window for STDP in milliseconds
        """
        self.enable_plasticity = True
        self.stdp_lr = learning_rate
        self.stdp_window = window
    
    def disable_stdp(self) -> None:
        """Disable spike-timing-dependent plasticity."""
        self.enable_plasticity = False