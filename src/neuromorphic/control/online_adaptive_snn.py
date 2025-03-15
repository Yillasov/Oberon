"""
Adaptive SNN algorithm that can continue learning during operation.
"""
import numpy as np
from typing import Dict, Any, List, Optional
import time
from .snn_controller import SNNController

class OnlineAdaptiveSNN(SNNController):
    """SNN controller that can adapt during operation."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64,
                online_learning_rate: float = 0.001,
                reward_window: int = 50):
        """
        Initialize online adaptive SNN controller.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            hidden_size: Number of hidden neurons
            online_learning_rate: Learning rate for online adaptation
            reward_window: Window size for reward history
        """
        super().__init__(input_size, output_size, hidden_size)
        
        # Online learning parameters
        self.online_learning_rate = online_learning_rate
        self.reward_window = reward_window
        self.reward_history = []
        
        # Store recent activity for online learning
        self.recent_inputs = []
        self.recent_hidden = []
        self.recent_outputs = []
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_count = 0
    
    def step(self, input_spikes: np.ndarray) -> np.ndarray:
        """
        Run one timestep of the adaptive SNN with memory of recent activity.
        
        Args:
            input_spikes: Binary input spike vector
            
        Returns:
            Binary output spike vector
        """
        # Store input for learning
        self.recent_inputs.append(input_spikes.copy())
        
        # Run standard step
        output_spikes = super().step(input_spikes)
        
        # Store hidden and output activity for learning
        if len(self.spike_history) > 0:
            self.recent_hidden.append(self.spike_history[-1][:self.hidden_size].copy())
            self.recent_outputs.append(output_spikes.copy())
        
        # Limit history length
        if len(self.recent_inputs) > self.reward_window:
            self.recent_inputs.pop(0)
            self.recent_hidden.pop(0)
            self.recent_outputs.pop(0)
        
        return output_spikes
    
    def provide_reward(self, reward: float):
        """
        Provide reward signal for recent actions.
        
        Args:
            reward: Reward value (-1 to 1, where 1 is good)
        """
        # Store reward
        self.reward_history.append(reward)
        
        # Limit history length
        if len(self.reward_history) > self.reward_window:
            self.reward_history.pop(0)
        
        # Adapt weights if we have enough history
        if len(self.reward_history) >= 3:
            self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt weights based on reward history."""
        # Only adapt if we have enough activity history
        if len(self.recent_hidden) < 2 or len(self.recent_outputs) < 2:
            return
        
        # Calculate reward trend (positive means improving)
        recent_rewards = self.reward_history[-3:]
        reward_trend = recent_rewards[-1] - recent_rewards[0]
        
        # Only adapt if performance is not improving
        if reward_trend <= 0:
            # Get recent activity
            recent_hidden = self.recent_hidden[-1]
            recent_input = self.recent_inputs[-1]
            
            # Simple exploratory adaptation - small random changes
            # to weights connected to active neurons
            
            # Adapt hidden-to-output weights
            for i in range(self.output_size):
                # Find active hidden neurons
                active_hidden = np.where(recent_hidden > 0)[0]
                if len(active_hidden) > 0:
                    # Randomly select one active connection to modify
                    h_idx = np.random.choice(active_hidden)
                    # Apply small random change
                    self.weights_ho[i, h_idx] += np.random.normal(0, 0.01) * self.online_learning_rate
            
            # Adapt input-to-hidden weights
            for i in range(self.hidden_size):
                # Find active inputs
                active_inputs = np.where(recent_input > 0)[0]
                if len(active_inputs) > 0:
                    # Randomly select one active connection to modify
                    in_idx = np.random.choice(active_inputs)
                    # Apply small random change
                    self.weights_ih[i, in_idx] += np.random.normal(0, 0.01) * self.online_learning_rate
            
            self.adaptation_count += 1
            
            # Track performance after adaptation
            self.performance_history.append({
                'reward': self.reward_history[-1],
                'adaptation_count': self.adaptation_count,
                'time': time.time()
            })
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the adaptation process.
        
        Returns:
            Dictionary with adaptation statistics
        """
        if not self.reward_history:
            return {'adaptations': 0, 'avg_reward': 0}
            
        return {
            'adaptations': self.adaptation_count,
            'avg_reward': np.mean(self.reward_history),
            'recent_reward': self.reward_history[-1] if self.reward_history else 0,
            'reward_trend': np.diff(self.reward_history[-10:]).mean() if len(self.reward_history) >= 10 else 0
        }