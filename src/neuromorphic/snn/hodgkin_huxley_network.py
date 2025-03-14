"""
Spiking Neural Network implementation using Hodgkin-Huxley neurons.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .hodgkin_huxley import HodgkinHuxleyNeuron


class HodgkinHuxleyNetwork:
    """
    Spiking Neural Network using Hodgkin-Huxley neurons.
    
    This class implements an SNN with biophysically realistic Hodgkin-Huxley neurons.
    """
    
    def __init__(self, num_inputs: int, num_neurons: int):
        """
        Initialize a Hodgkin-Huxley Neural Network.
        
        Args:
            num_inputs: Number of input channels
            num_neurons: Number of neurons in the network
        """
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        
        # Create neurons
        self.neurons = [HodgkinHuxleyNeuron() for _ in range(num_neurons)]
        
        # Initialize weights randomly
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.1
        
        # Spike history
        self.spike_history: List[List[bool]] = []
        
    def reset(self) -> None:
        """Reset the network state."""
        for neuron in self.neurons:
            neuron.reset()
        self.spike_history = []
        
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
        
        # Calculate input currents for each neuron
        currents = np.dot(self.weights, inputs)
        
        # Update each neuron and collect spikes
        spikes = np.zeros(self.num_neurons, dtype=bool)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.update(currents[i], dt, t)
        
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