"""
Example demonstrating the use of reward-modulated STDP for reinforcement learning.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from src.neuromorphic.snn.lif_neuron import LIFNeuron
from src.neuromorphic.learning.reward_modulated_stdp import RewardModulatedSTDP


class SimpleSNN:
    """
    Simple spiking neural network for demonstrating reward-modulated STDP.
    """
    
    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initialize a simple SNN.
        
        Args:
            num_inputs: Number of input neurons
            num_outputs: Number of output neurons
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # Create output neurons
        self.output_neurons = [LIFNeuron(threshold=1.0, tau_m=10.0) for _ in range(num_outputs)]
        
        # Create learning algorithm
        self.learning_algorithm = RewardModulatedSTDP(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            learning_rate=0.01,
            eligibility_trace_decay=0.95
        )
        
        # Get initial weights from learning algorithm
        self.weights = self.learning_algorithm.get_weights()
    
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
        # Calculate input currents for each output neuron
        currents = np.dot(self.weights, input_spikes)
        
        # Update each output neuron
        output_spikes = np.zeros(self.num_outputs, dtype=bool)
        for i, neuron in enumerate(self.output_neurons):
            output_spikes[i] = neuron.update(currents[i], dt, t)
        
        # Record spikes for learning
        self.learning_algorithm.record_spikes(input_spikes, output_spikes, t)
        
        # Update eligibility traces
        self.learning_algorithm.update_eligibility_traces(t)
        
        return output_spikes
    
    def apply_reward(self, reward: float) -> None:
        """
        Apply reward signal to update weights.
        
        Args:
            reward: Reward signal
        """
        self.learning_algorithm.apply_reward(reward)
        self.weights = self.learning_algorithm.get_weights()
    
    def reset(self) -> None:
        """Reset the network state."""
        for neuron in self.output_neurons:
            neuron.reset()
        self.learning_algorithm.reset()


def run_pattern_learning_experiment(num_trials: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Run an experiment to learn a specific input-output pattern.
    
    Args:
        num_trials: Number of learning trials
        
    Returns:
        Tuple of (rewards, weight_changes)
    """
    # Create a simple SNN
    num_inputs = 10
    num_outputs = 2
    snn = SimpleSNN(num_inputs, num_outputs)
    
    # Define target pattern (we want output neuron 0 to fire for pattern A and output neuron 1 for pattern B)
    pattern_a = np.zeros(num_inputs, dtype=bool)
    pattern_a[0:5] = True  # First half of inputs active
    
    pattern_b = np.zeros(num_inputs, dtype=bool)
    pattern_b[5:10] = True  # Second half of inputs active
    
    # Learning loop
    rewards = []
    weight_changes = []
    
    for trial in range(num_trials):
        snn.reset()
        
        # Choose pattern randomly
        if np.random.rand() < 0.5:
            pattern = pattern_a
            target_output = 0
        else:
            pattern = pattern_b
            target_output = 1
        
        # Run for 100ms
        output_spikes = np.zeros(num_outputs, dtype=bool)
        for t in range(100):  # 100ms with 1ms time steps
            output_spikes = snn.step(pattern, 1.0, float(t))
            
            # If we get any output spikes, break
            if np.any(output_spikes):
                break
        
        # Calculate reward (1 if correct output neuron fired, -1 if wrong neuron fired, 0 if no firing)
        reward = 0.0
        if output_spikes[target_output]:
            reward = 1.0
        elif np.any(output_spikes):
            reward = -1.0
        
        # Apply reward
        snn.apply_reward(reward)
        
        # Record metrics
        rewards.append(reward)
        weight_changes.append(np.mean(np.abs(snn.learning_algorithm.eligibility_traces)))
        
        # Print progress
        if (trial + 1) % 100 == 0:
            success_rate = np.mean([r == 1.0 for r in rewards[-100:]])
            print(f"Trial {trial + 1}/{num_trials}, Success rate: {success_rate:.2f}")
    
    return rewards, weight_changes


def plot_results(rewards: List[float], weight_changes: List[float]) -> None:
    """
    Plot the results of the learning experiment.
    
    Args:
        rewards: List of rewards received
        weight_changes: List of weight changes
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title('Reward Signal')
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    
    # Plot smoothed success rate
    window_size = 50
    success_rate = [np.mean([r == 1.0 for r in rewards[max(0, i-window_size):i+1]]) 
                   for i in range(len(rewards))]
    
    plt.subplot(2, 1, 2)
    plt.plot(success_rate)
    plt.title('Success Rate (Moving Average)')
    plt.xlabel('Trial')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('reward_modulated_stdp_learning.png')
    plt.show()


if __name__ == "__main__":
    # Run the experiment
    rewards, weight_changes = run_pattern_learning_experiment(num_trials=1000)
    
    # Plot results
    plot_results(rewards, weight_changes)