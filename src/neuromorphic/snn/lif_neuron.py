"""
Basic Leaky Integrate-and-Fire (LIF) neuron model for spiking neural networks.
"""
import numpy as np
from typing import List, Optional, Dict, Any


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model.
    
    This implements a basic LIF neuron with configurable parameters.
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 rest_potential: float = 0.0,
                 reset_potential: float = 0.0,
                 tau_m: float = 10.0,  # Membrane time constant (ms)
                 refractory_period: float = 2.0):  # Refractory period (ms)
        """
        Initialize a LIF neuron.
        
        Args:
            threshold: Firing threshold potential
            rest_potential: Resting membrane potential
            reset_potential: Post-spike reset potential
            tau_m: Membrane time constant in milliseconds
            refractory_period: Refractory period in milliseconds
        """
        self.threshold = threshold
        self.rest_potential = rest_potential
        self.reset_potential = reset_potential
        self.tau_m = tau_m
        self.refractory_period = refractory_period
        
        # State variables
        self.membrane_potential = rest_potential
        self.last_spike_time = -float('inf')
        self.is_refractory = False
        
    def reset(self) -> None:
        """Reset the neuron state."""
        self.membrane_potential = self.rest_potential
        self.last_spike_time = -float('inf')
        self.is_refractory = False
        
    def update(self, current: float, dt: float, t: float) -> bool:
        """
        Update neuron state for a single time step.
        
        Args:
            current: Input current
            dt: Time step in milliseconds
            t: Current simulation time
            
        Returns:
            True if the neuron fired, False otherwise
        """
        # Check if neuron is in refractory period
        if t - self.last_spike_time < self.refractory_period:
            self.is_refractory = True
            return False
        else:
            self.is_refractory = False
        
        # Update membrane potential using Euler method
        dv = (-(self.membrane_potential - self.rest_potential) + current) / self.tau_m
        self.membrane_potential += dv * dt
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.reset_potential
            self.last_spike_time = t
            return True
        
        return False