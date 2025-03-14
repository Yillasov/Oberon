"""
Hodgkin-Huxley neuron model for spiking neural networks.
"""
import numpy as np
from typing import List, Optional, Dict, Any


class HodgkinHuxleyNeuron:
    """
    Hodgkin-Huxley neuron model.
    
    This implements the classic Hodgkin-Huxley model which provides a detailed
    biophysical simulation of action potential generation in neurons.
    """
    
    def __init__(self, 
                 C_m: float = 1.0,      # Membrane capacitance (μF/cm²)
                 g_Na: float = 120.0,   # Maximum sodium conductance (mS/cm²)
                 g_K: float = 36.0,     # Maximum potassium conductance (mS/cm²)
                 g_L: float = 0.3,      # Leak conductance (mS/cm²)
                 E_Na: float = 50.0,    # Sodium reversal potential (mV)
                 E_K: float = -77.0,    # Potassium reversal potential (mV)
                 E_L: float = -54.387): # Leak reversal potential (mV)
        """
        Initialize a Hodgkin-Huxley neuron.
        
        Args:
            C_m: Membrane capacitance (μF/cm²)
            g_Na: Maximum sodium conductance (mS/cm²)
            g_K: Maximum potassium conductance (mS/cm²)
            g_L: Leak conductance (mS/cm²)
            E_Na: Sodium reversal potential (mV)
            E_K: Potassium reversal potential (mV)
            E_L: Leak reversal potential (mV)
        """
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        
        # State variables
        self.V = -65.0  # Membrane potential (mV)
        self.m = 0.05   # Sodium activation gating variable
        self.h = 0.6    # Sodium inactivation gating variable
        self.n = 0.32   # Potassium activation gating variable
        
        # Spike detection
        self.spike_threshold = 0.0  # Threshold for spike detection (mV)
        self.last_spike_time = -1000.0
        
    def reset(self) -> None:
        """Reset the neuron state."""
        self.V = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.last_spike_time = -1000.0
    
    def _alpha_m(self, V: float) -> float:
        """Sodium activation rate."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def _beta_m(self, V: float) -> float:
        """Sodium deactivation rate."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def _alpha_h(self, V: float) -> float:
        """Sodium inactivation rate."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def _beta_h(self, V: float) -> float:
        """Sodium deinactivation rate."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def _alpha_n(self, V: float) -> float:
        """Potassium activation rate."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def _beta_n(self, V: float) -> float:
        """Potassium deactivation rate."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def update(self, current: float, dt: float, t: float) -> bool:
        """
        Update neuron state for a single time step.
        
        Args:
            current: Input current (μA/cm²)
            dt: Time step in milliseconds
            t: Current simulation time
            
        Returns:
            True if the neuron fired, False otherwise
        """
        # Calculate channel dynamics
        alpha_m = self._alpha_m(self.V)
        beta_m = self._beta_m(self.V)
        alpha_h = self._alpha_h(self.V)
        beta_h = self._beta_h(self.V)
        alpha_n = self._alpha_n(self.V)
        beta_n = self._beta_n(self.V)
        
        # Update gating variables
        self.m += dt * (alpha_m * (1 - self.m) - beta_m * self.m)
        self.h += dt * (alpha_h * (1 - self.h) - beta_h * self.h)
        self.n += dt * (alpha_n * (1 - self.n) - beta_n * self.n)
        
        # Calculate ionic currents
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        
        # Update membrane potential
        dV = (current - I_Na - I_K - I_L) / self.C_m
        self.V += dt * dV
        
        # Detect spike
        if self.V > self.spike_threshold and (t - self.last_spike_time) > 2.0:
            self.last_spike_time = t
            return True
        
        return False