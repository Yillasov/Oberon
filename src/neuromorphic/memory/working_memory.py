"""
Working Memory model for neuromorphic computing.

This module implements a basic working memory model that can temporarily
store and maintain information using recurrent spiking neural networks.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class WorkingMemory:
    """
    Working Memory model based on persistent neural activity.
    
    This model implements a form of short-term memory that maintains
    information through sustained neural activity patterns in a recurrent network.
    """
    
    def __init__(self, 
                 size: int,
                 tau_m: float = 20.0,      # Membrane time constant (ms)
                 tau_s: float = 5.0,       # Synaptic time constant (ms)
                 threshold: float = 0.8,    # Firing threshold
                 refractory: float = 2.0,   # Refractory period (ms)
                 noise_level: float = 0.05): # Background noise level
        """
        Initialize the working memory model.
        
        Args:
            size: Number of neurons in the memory network
            tau_m: Membrane time constant in milliseconds
            tau_s: Synaptic time constant in milliseconds
            threshold: Firing threshold for neurons
            refractory: Refractory period in milliseconds
            noise_level: Level of background noise
        """
        self.size = size
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.threshold = threshold
        self.refractory = refractory
        self.noise_level = noise_level
        
        # Initialize recurrent connection weights
        # Create clusters of neurons that sustain activity
        self.weights = np.zeros((size, size))
        num_clusters = max(1, size // 10)
        cluster_size = size // num_clusters
        
        for c in range(num_clusters):
            start_idx = c * cluster_size
            end_idx = (c + 1) * cluster_size if c < num_clusters - 1 else size
            
            # Create strong recurrent connections within clusters
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:  # No self-connections
                        self.weights[i, j] = 0.2 + 0.1 * np.random.rand()
        
        # Add some weak random connections between clusters
        self.weights += 0.05 * np.random.rand(size, size)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        
        # Neuron state variables
        self.membrane_potential = np.zeros(size)
        self.synaptic_current = np.zeros(size)
        self.last_spike_time = np.ones(size) * -1000.0
        self.spike_history = []
        
        # Memory state tracking
        self.active_clusters = np.zeros(num_clusters, dtype=bool)
        self.num_clusters = num_clusters
        self.cluster_size = cluster_size
        
    def reset(self) -> None:
        """Reset the memory state."""
        self.membrane_potential = np.zeros(self.size)
        self.synaptic_current = np.zeros(self.size)
        self.last_spike_time = np.ones(self.size) * -1000.0
        self.spike_history = []
        self.active_clusters = np.zeros(self.num_clusters, dtype=bool)
    
    def update(self, input_spikes: Optional[np.ndarray], dt: float, t: float) -> np.ndarray:
        """
        Update the memory state for one time step.
        
        Args:
            input_spikes: Optional input spike pattern (None for no input)
            dt: Time step in milliseconds
            t: Current simulation time
            
        Returns:
            Output spike pattern
        """
        # Apply input if provided
        if input_spikes is not None:
            if len(input_spikes) != self.size:
                raise ValueError(f"Input size must be {self.size}")
            self.synaptic_current += input_spikes
        
        # Add random background noise
        noise = np.random.rand(self.size) < (self.noise_level * dt / 1000.0)
        self.synaptic_current += noise
        
        # Decay synaptic currents
        self.synaptic_current *= np.exp(-dt / self.tau_s)
        
        # Decay membrane potentials
        self.membrane_potential *= np.exp(-dt / self.tau_m)
        
        # Integrate synaptic currents into membrane potentials
        self.membrane_potential += self.synaptic_current
        
        # Check for spikes
        output_spikes = np.zeros(self.size, dtype=bool)
        for i in range(self.size):
            # Check if neuron is not in refractory period and above threshold
            if (t - self.last_spike_time[i] > self.refractory and 
                self.membrane_potential[i] > self.threshold):
                output_spikes[i] = True
                self.last_spike_time[i] = t
                self.membrane_potential[i] = 0.0  # Reset potential
        
        # Process recurrent connections
        if np.any(output_spikes):
            # Calculate recurrent input
            recurrent_input = np.dot(self.weights, output_spikes.astype(float))
            self.synaptic_current += recurrent_input
        
        # Track active clusters
        for c in range(self.num_clusters):
            start_idx = c * self.cluster_size
            end_idx = (c + 1) * self.cluster_size if c < self.num_clusters - 1 else self.size
            
            # Cluster is active if at least 20% of its neurons fired recently
            recent_activity = np.mean(t - self.last_spike_time[start_idx:end_idx] < 50.0)
            self.active_clusters[c] = recent_activity > 0.2
        
        # Store spike history
        self.spike_history.append(output_spikes.copy())
        if len(self.spike_history) > 1000:  # Limit history size
            self.spike_history.pop(0)
        
        return output_spikes
    
    def store(self, pattern: np.ndarray, duration: float = 50.0, dt: float = 1.0) -> None:
        """
        Store a pattern in working memory by presenting it as input.
        
        Args:
            pattern: Binary pattern to store
            duration: Duration to present the pattern (ms)
            dt: Time step for simulation (ms)
        """
        if len(pattern) != self.size:
            raise ValueError(f"Pattern size must be {self.size}")
        
        # Present the pattern as input for the specified duration
        t = 0.0
        while t < duration:
            self.update(pattern, dt, t)
            t += dt
    
    def recall(self, duration: float = 100.0, dt: float = 1.0) -> np.ndarray:
        """
        Recall the current memory state.
        
        Args:
            duration: Duration to run recall (ms)
            dt: Time step for simulation (ms)
            
        Returns:
            Average activity pattern over the recall period
        """
        # Run the network without input to see sustained activity
        activity = np.zeros(self.size)
        t = 0.0
        steps = 0
        
        while t < duration:
            spikes = self.update(None, dt, t)
            activity += spikes
            t += dt
            steps += 1
        
        # Return average activity
        return activity / steps if steps > 0 else activity
    
    def get_memory_state(self) -> Dict[str, Any]:
        """
        Get the current state of the working memory.
        
        Returns:
            Dictionary with memory state information
        """
        # Calculate recent activity (last 50ms)
        recent_spikes = np.array(self.spike_history[-50:]) if len(self.spike_history) > 0 else np.array([])
        recent_activity = np.mean(recent_spikes, axis=0) if len(recent_spikes) > 0 else np.zeros(self.size)
        
        return {
            "active_clusters": self.active_clusters.copy(),
            "recent_activity": recent_activity,
            "active_neurons": np.sum(recent_activity > 0),
            "memory_load": np.sum(self.active_clusters) / self.num_clusters
        }