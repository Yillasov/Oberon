"""
Neuromorphic reinforcement learning for mission adaptation.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from .strategic_controller import StrategicDecision
from .mission_hierarchy import MissionHierarchy


@dataclass
class RewardSignal:
    """Reward signal for reinforcement learning."""
    timestamp: float
    value: float
    source: str
    context: Dict[str, Any]


class NeuromorphicRL:
    """Neuromorphic reinforcement learning for mission adaptation."""
    
    def __init__(self, mission: MissionHierarchy):
        self.mission = mission
        
        # Learning parameters
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.exploration_rate = 0.2
        
        # Neural network parameters
        self.state_size = 32
        self.action_size = 16
        self.hidden_size = 24
        
        # Initialize neural weights
        self.weights_ih = np.random.normal(0, 0.1, (self.hidden_size, self.state_size))
        self.weights_ho = np.random.normal(0, 0.1, (self.action_size, self.hidden_size))
        
        # Experience memory
        self.memory_size = 1000
        self.experiences: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self.reward_history: List[RewardSignal] = []
    
    async def process_state(self, 
                          state: np.ndarray, 
                          decision: StrategicDecision) -> Dict[str, Any]:
        """Process current state and generate adaptation signal."""
        # Generate action
        action = self._select_action(state)
        
        # Calculate reward
        reward = self._calculate_reward(state, action, decision)
        
        # Store experience
        next_state = self._predict_next_state(state, action)
        self._store_experience(state, action, reward, next_state)
        
        # Learn from experience
        if len(self.experiences) >= 32:  # Batch size
            await self._learn()
        
        return {
            'action': action.tolist(),
            'reward': float(reward),
            'exploration_rate': float(self.exploration_rate)
        }
    
    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.exploration_rate:
            return np.random.normal(0, 1, self.action_size)
        
        # Forward pass through network
        hidden = np.tanh(np.dot(self.weights_ih, state))
        action = np.tanh(np.dot(self.weights_ho, hidden))
        
        return action
    
    def _calculate_reward(self,
                         state: np.ndarray,
                         action: np.ndarray,
                         decision: StrategicDecision) -> float:
        """Calculate reward based on state transition and decision impact."""
        # Base reward from decision confidence
        base_reward = decision.confidence * decision.impact_estimate
        
        # Additional reward based on mission progress
        progress_reward = sum(self.mission.completion_status.values()) / \
                         len(self.mission.tasks)
        
        # Penalty for resource overconsumption
        resource_penalty = 0.0
        if self.mission.resource_utilization:
            avg_utilization = sum(self.mission.resource_utilization.values()) / \
                            len(self.mission.resource_utilization)
            resource_penalty = max(0, avg_utilization - 0.8)  # Penalty above 80% usage
        
        reward = base_reward + progress_reward - resource_penalty
        
        # Store reward signal
        self.reward_history.append(RewardSignal(
            timestamp=datetime.now().timestamp(),
            value=float(reward),
            source=decision.action_type,
            context={'state_norm': float(np.linalg.norm(state)),
                    'action_norm': float(np.linalg.norm(action))}
        ))
        
        return reward
    
    def _predict_next_state(self,
                           state: np.ndarray,
                           action: np.ndarray) -> np.ndarray:
        """Predict next state based on current state and action."""
        # Simple state transition model
        delta = np.tanh(action) * 0.1
        next_state = state + delta
        return np.clip(next_state, -1, 1)
    
    def _store_experience(self,
                         state: np.ndarray,
                         action: np.ndarray,
                         reward: float,
                         next_state: np.ndarray):
        """Store experience in memory."""
        self.experiences.append((state, action, reward, next_state))
        if len(self.experiences) > self.memory_size:
            self.experiences.pop(0)
    
    async def _learn(self):
        """Learn from stored experiences."""
        # Sample batch
        batch_indices = np.random.choice(
            len(self.experiences), 32, replace=False)
        batch = [self.experiences[i] for i in batch_indices]
        
        # Unpack batch
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        
        # Calculate target Q-values
        next_hidden = np.tanh(np.dot(self.weights_ih, next_states.T))
        next_q_values = np.tanh(np.dot(self.weights_ho, next_hidden))
        targets = rewards + self.discount_factor * np.max(next_q_values, axis=0)
        
        # Update weights
        hidden = np.tanh(np.dot(self.weights_ih, states.T))
        outputs = np.tanh(np.dot(self.weights_ho, hidden))
        
        # Backpropagation
        output_delta = (targets - np.max(outputs, axis=0)) * (1 - outputs ** 2)
        hidden_delta = np.dot(self.weights_ho.T, output_delta) * (1 - hidden ** 2)
        
        # Update weights
        self.weights_ho += self.learning_rate * np.outer(output_delta, hidden)
        self.weights_ih += self.learning_rate * np.outer(hidden_delta, states)
        
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.reward_history:
            return {'status': 'no_data'}
        
        recent_rewards = [r.value for r in self.reward_history[-100:]]
        return {
            'average_reward': float(np.mean(recent_rewards)),
            'reward_variance': float(np.var(recent_rewards)),
            'exploration_rate': float(self.exploration_rate),
            'memory_usage': len(self.experiences) / self.memory_size,
            'learning_rate': float(self.learning_rate),
            'recent_sources': [r.source for r in self.reward_history[-10:]]
        }