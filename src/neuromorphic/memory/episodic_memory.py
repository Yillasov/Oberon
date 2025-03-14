"""
Simple Episodic Memory model for neuromorphic computing.

This module implements a basic episodic memory model that can store and
retrieve sequences of events or patterns.
"""
import numpy as np
from typing import List, Dict, Any, Optional


class EpisodicMemory:
    """
    Simple Episodic Memory model.
    
    This model stores sequences of patterns as episodes and can retrieve
    them given a partial cue.
    """
    
    def __init__(self, pattern_size: int, max_episodes: int = 10):
        """
        Initialize the episodic memory model.
        
        Args:
            pattern_size: Size of each pattern in the episodes
            max_episodes: Maximum number of episodes to store
        """
        self.pattern_size = pattern_size
        self.max_episodes = max_episodes
        
        # Storage for episodes (list of sequences)
        self.episodes = []
        
        # Current episode being stored
        self.current_episode = []
        
        # Similarity threshold for pattern matching
        self.similarity_threshold = 0.7
    
    def store_pattern(self, pattern: np.ndarray, new_episode: bool = False) -> None:
        """
        Store a pattern in the current episode.
        
        Args:
            pattern: Pattern to store
            new_episode: Whether to start a new episode
        """
        if len(pattern) != self.pattern_size:
            raise ValueError(f"Pattern size must be {self.pattern_size}")
        
        # Start a new episode if requested or if current episode is complete
        if new_episode or not self.current_episode:
            # Store the previous episode if it exists
            if self.current_episode:
                self.episodes.append(self.current_episode.copy())
                
                # Remove oldest episode if we exceed max_episodes
                if len(self.episodes) > self.max_episodes:
                    self.episodes.pop(0)
            
            # Start a new episode
            self.current_episode = [pattern.copy()]
        else:
            # Add to current episode
            self.current_episode.append(pattern.copy())
    
    def end_episode(self) -> None:
        """End the current episode and store it."""
        if self.current_episode:
            self.episodes.append(self.current_episode.copy())
            self.current_episode = []
            
            # Remove oldest episode if we exceed max_episodes
            if len(self.episodes) > self.max_episodes:
                self.episodes.pop(0)
    
    def retrieve(self, cue: np.ndarray, max_length: int = 5) -> List[np.ndarray]:
        """
        Retrieve an episode given a cue pattern.
        
        Args:
            cue: Pattern to use as retrieval cue
            max_length: Maximum length of sequence to retrieve
            
        Returns:
            Retrieved sequence of patterns
        """
        if len(cue) != self.pattern_size:
            raise ValueError(f"Cue size must be {self.pattern_size}")
        
        best_match = None
        best_similarity = -1
        best_position = 0
        
        # Find the best matching pattern across all episodes
        for episode_idx, episode in enumerate(self.episodes):
            for pos, pattern in enumerate(episode):
                similarity = self._calculate_similarity(cue, pattern)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = episode
                    best_position = pos
        
        # If no match found, return empty list
        if best_match is None:
            return []
        
        # Return the sequence starting from the matched position
        result = best_match[best_position:]
        
        # Limit to max_length
        return result[:max_length]
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (1.0 = identical, 0.0 = completely different)
        """
        # Simple dot product similarity normalized by pattern size
        return np.sum(pattern1 * pattern2) / self.pattern_size
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the episodic memory.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "num_episodes": len(self.episodes),
            "current_episode_length": len(self.current_episode),
            "avg_episode_length": np.mean([len(ep) for ep in self.episodes]) if self.episodes else 0,
            "max_episode_length": max([len(ep) for ep in self.episodes]) if self.episodes else 0
        }
    
    def clear(self) -> None:
        """Clear all episodes from memory."""
        self.episodes = []
        self.current_episode = []