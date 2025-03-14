"""
Adaptive Exponential Integrate-and-Fire (AdEx) neuron model for spiking neural networks.
"""
import numpy as np
from typing import List, Optional, Dict, Any


class AdExNeuron:
    """
    Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
    
    This implements the AdEx model which extends the leaky integrate-and-fire model
    with an exponential term for spike initiation and an adaptation variable.
    """
    
    def __init__(self, 
                 C: float = 281.0,       # Membrane capacitance (pF)
                 g_L: float = 30.0,      # Leak conductance (nS)
                 E_L: float = -70.6,     # Leak reversal potential (mV)
                 V_T: float = -50.4,     # Spike threshold (mV)
                 delta_T: float = 2.0,   # Slope factor (mV)
                 tau_w: float = 144.0,   # Adaptation time constant (ms)
                 a: float = 4.0,         # Subthreshold adaptation (nS)
                 b: float = 80.5,        # Spike-triggered adaptation (pA)
                 V_reset: float = -70.6, # Reset potential (mV)
                 V_peak: float = 20.0):  # Spike detection threshold (mV)
        """
        Initialize an AdEx neuron.
        
        Args:
            C: Membrane capacitance (pF)
            g_L: Leak conductance (nS)
            E_L: Leak reversal potential (mV)
            V_T: Spike threshold (mV)
            delta_T: Slope factor (mV)
            tau_w: Adaptation time constant (ms)
            a: Subthreshold adaptation (nS)
            b: Spike-triggered adaptation (pA)
            V_reset: Reset potential (mV)
            V_peak: Spike detection threshold (mV)
        """
        self.C = C
        self.g_L = g_L
        self.E_L = E_L
        self.V_T = V_T
        self.delta_T = delta_T
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.V_reset = V_reset
        self.V_peak = V_peak
        
        # State variables
        self.V = E_L  # Membrane potential (mV)
        self.w = 0.0  # Adaptation variable (pA)
        
        # Spike tracking
        self.last_spike_time = -1000.0
        
    def reset(self) -> None:
        """Reset the neuron state."""
        self.V = self.E_L
        self.w = 0.0
        self.last_spike_time = -1000.0
        
    def update(self, current: float, dt: float, t: float) -> bool:
        """
        Update neuron state for a single time step.
        
        Args:
            current: Input current (pA)
            dt: Time step in milliseconds
            t: Current simulation time
            
        Returns:
            True if the neuron fired, False otherwise
        """
        # Calculate exponential term
        if (self.V - self.V_T) / self.delta_T > 20:
            # Avoid numerical instability
            exp_term = np.exp(20)
        else:
            exp_term = np.exp((self.V - self.V_T) / self.delta_T)
        
        # Calculate membrane potential derivative
        dV = ((-self.g_L * (self.V - self.E_L) + 
               self.g_L * self.delta_T * exp_term - 
               self.w + current) / self.C)
        
        # Calculate adaptation variable derivative
        dw = ((self.a * (self.V - self.E_L) - self.w) / self.tau_w)
        
        # Update state variables using Euler method
        self.V += dt * dV
        self.w += dt * dw
        
        # Check for spike
        if self.V >= self.V_peak:
            self.V = self.V_reset
            self.w += self.b
            self.last_spike_time = t
            return True
        
        return False