"""
Example demonstrating the use of Spike-Timing-Dependent Homeostatic Plasticity.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from src.neuromorphic.snn.lif_neuron import LIFNeuron
from src.neuromorphic.learning.stdhp import STDHP


class HomeostasisSNN:
    """
    Spiking neural network with homeostatic plasticity for demonstration.
    """
    
    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initialize a homeostatic SNN.
        
        Args:
            num_inputs: Number of input neurons
            num_outputs: Number of output neurons
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # Create learning algorithm
        self.learning_algorithm = STDHP(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            learning_rate=0.01,
            target_rate=0.05,  # Target firing rate (5%)
            homeostatic_rate=0.001,
            activity_window=1000
        )
        
        # Get initial weights and thresholds
        self.weights = self.learning_algorithm.get_weights()
        self.thresholds = self.learning_algorithm.get_thresholds()
        
        # Create output neurons with adjustable thresholds
        self.output_neurons = [
            LIFNeuron(threshold=self.thresholds[i], tau_m=20.0) 
            for i in range(num_outputs)
        ]
        
        # Activity tracking
        self.input_activity = np.zeros(num_inputs)
        self.output_activity = np.zeros(num_outputs)
        self.time_step = 0
    
    def step(self, input_spikes: np.ndarray, dt: float, t: float) -> np.ndarray:
        """
        Simulate one time step of the network.
        
        Args:
            input_spikes: Input spike pattern
            dt: Time step in milliseconds
            t: Current simulation time
            
        Returns:
            Output spike pattern
        """
        # Update input activity
        self.input_activity = 0.99 * self.input_activity + 0.01 * input_spikes
        
        # Calculate input currents for each output neuron
        currents = np.dot(self.weights, input_spikes)
        
        # Update each output neuron with its current threshold
        output_spikes = np.zeros(self.num_outputs, dtype=bool)
        for i, neuron in enumerate(self.output_neurons):
            # Update neuron threshold from learning algorithm
            neuron.threshold = self.thresholds[i]
            
            # Update neuron
            output_spikes[i] = neuron.update(currents[i], dt, t)
        
        # Update output activity
        self.output_activity = 0.99 * self.output_activity + 0.01 * output_spikes
        
        # Record spikes for learning
        self.learning_algorithm.record_spikes(input_spikes, output_spikes, t)
        
        # Update weights based on STDP
        self.learning_algorithm.update_weights(t)
        
        # Get updated weights and thresholds
        self.weights = self.learning_algorithm.get_weights()
        self.thresholds = self.learning_algorithm.get_thresholds()
        
        self.time_step += 1
        
        return output_spikes
    
    def reset(self) -> None:
        """Reset the network state."""
        for neuron in self.output_neurons:
            neuron.reset()
        self.learning_algorithm.reset()
        self.input_activity = np.zeros(self.num_inputs)
        self.output_activity = np.zeros(self.num_outputs)
        self.time_step = 0


def run_homeostasis_experiment(num_steps: int = 10000) -> Tuple[List[float], List[float], List[float]]:
    """
    Run an experiment to demonstrate homeostatic regulation.
    
    Args:
        num_steps: Number of simulation steps
        
    Returns:
        Tuple of (activity_levels, thresholds, weight_changes)
    """
    # Create a network
    num_inputs = 100
    num_outputs = 10
    snn = HomeostasisSNN(num_inputs, num_outputs)
    
    # Metrics to track
    activity_levels = []
    thresholds = []
    weight_changes = []
    
    # Run simulation
    for step in range(num_steps):
        # Generate random input spikes with changing statistics
        # First half: low activity, second half: high activity
        if step < num_steps // 2:
            input_rate = 0.02  # 2% firing rate
        else:
            input_rate = 0.2   # 20% firing rate
            
        input_spikes = np.random.rand(num_inputs) < input_rate
        
        # Simulate one step
        snn.step(input_spikes, 1.0, float(step))
        
        # Record metrics
        activity_levels.append(np.mean(snn.output_activity))
        thresholds.append(np.mean(snn.thresholds))
        
        if len(snn.learning_algorithm.weight_changes) > 0:
            weight_changes.append(snn.learning_algorithm.weight_changes[-1])
        else:
            weight_changes.append(0.0)
        
        # Print progress
        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}/{num_steps}, Activity: {activity_levels[-1]:.4f}, Threshold: {thresholds[-1]:.4f}")
    
    return activity_levels, thresholds, weight_changes


def plot_homeostasis_results(activity_levels: List[float], thresholds: List[float], weight_changes: List[float]) -> None:
    """
    Plot the results of the homeostasis experiment.
    
    Args:
        activity_levels: List of average output activity levels
        thresholds: List of average threshold values
        weight_changes: List of average weight changes
    """
    plt.figure(figsize=(12, 10))
    
    # Plot activity levels
    plt.subplot(3, 1, 1)
    plt.plot(activity_levels)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Target Rate')
    plt.title('Average Output Activity')
    plt.xlabel('Time Step')
    plt.ylabel('Activity Level')
    plt.legend()
    
    # Plot thresholds
    plt.subplot(3, 1, 2)
    plt.plot(thresholds)
    plt.title('Average Neuron Thresholds')
    plt.xlabel('Time Step')
    plt.ylabel('Threshold')
    
    # Plot weight changes
    plt.subplot(3, 1, 3)
    plt.plot(weight_changes)
    plt.title('Average Weight Changes')
    plt.xlabel('Time Step')
    plt.ylabel('Weight Change')
    
    plt.tight_layout()
    plt.savefig('homeostatic_plasticity_results.png')
    plt.show()


if __name__ == "__main__":
    # Run the experiment
    activity_levels, thresholds, weight_changes = run_homeostasis_experiment(num_steps=20000)
    
    # Plot results
    plot_homeostasis_results(activity_levels, thresholds, weight_changes)