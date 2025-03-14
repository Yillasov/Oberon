"""
Izhikevich neuron model for spiking neural networks.
"""
import numpy as np
from typing import List, Optional, Dict, Any


class IzhikevichNeuron:
    """
    Izhikevich neuron model.
    
    This implements the Izhikevich neuron model which can reproduce various
    firing patterns observed in biological neurons.
    """
    
    def __init__(self, 
                 a: float = 0.02,  # Recovery parameter
                 b: float = 0.2,   # Sensitivity of recovery variable
                 c: float = -65.0, # After-spike reset value for v
                 d: float = 8.0):  # After-spike reset increment for u
        """
        Initialize an Izhikevich neuron.
        
        Args:
            a: Time scale of recovery variable u
            b: Sensitivity of recovery variable u to membrane potential v
            c: After-spike reset value for membrane potential v
            d: After-spike reset increment for recovery variable u
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # State variables
        self.v = c  # Membrane potential
        self.u = b * c  # Recovery variable
        
    def reset(self) -> None:
        """Reset the neuron state."""
        self.v = self.c
        self.u = self.b * self.c
        
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
        # Update membrane potential and recovery variable
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + current) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt
        
        self.v += dv
        self.u += du
        
        # Check for spike
        if self.v >= 30.0:
            self.v = self.c
            self.u += self.d
            return True
        
        return False