"""
Spiking Neural Network implementation using Izhikevich neurons.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .izhikevich_neuron import IzhikevichNeuron


class IzhikevichNetwork:
    """
    Spiking Neural Network using Izhikevich neurons.
    
    This class implements an SNN with Izhikevich neurons, which can model
    various biological neuron behaviors.
    """
    
    def __init__(self, num_inputs: int, num_neurons: int, neuron_types: Optional[List[str]] = None):
        """
        Initialize an Izhikevich Neural Network.
        
        Args:
            num_inputs: Number of input channels
            num_neurons: Number of neurons in the network
            neuron_types: List of neuron types to create (options: 'regular', 'chattering', 
                          'fast_spiking', 'low_threshold', 'resonator')
        """
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        
        # Create neurons based on specified types
        self.neurons = []
        if neuron_types is None:
            # Default to regular spiking neurons
            self.neurons = [IzhikevichNeuron() for _ in range(num_neurons)]
        else:
            for i in range(num_neurons):
                neuron_type = neuron_types[i % len(neuron_types)]
                self.neurons.append(self._create_neuron(neuron_type))
        
        # Initialize weights randomly
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.1
        
        # Spike history
        self.spike_history: List[List[bool]] = []
        
    def _create_neuron(self, neuron_type: str) -> IzhikevichNeuron:
        """
        Create a neuron with specific parameters based on type.
        
        Args:
            neuron_type: Type of neuron to create
            
        Returns:
            Configured IzhikevichNeuron instance
        """
        if neuron_type == 'regular':
            return IzhikevichNeuron(a=0.02, b=0.2, c=-65, d=8)
        elif neuron_type == 'chattering':
            return IzhikevichNeuron(a=0.02, b=0.2, c=-50, d=2)
        elif neuron_type == 'fast_spiking':
            return IzhikevichNeuron(a=0.1, b=0.2, c=-65, d=2)
        elif neuron_type == 'low_threshold':
            return IzhikevichNeuron(a=0.02, b=0.25, c=-65, d=2)
        elif neuron_type == 'resonator':
            return IzhikevichNeuron(a=0.1, b=0.26, c=-65, d=2)
        else:
            # Default to regular spiking
            return IzhikevichNeuron()
    
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