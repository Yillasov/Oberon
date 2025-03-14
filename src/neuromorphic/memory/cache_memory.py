"""
Simple Cache Memory model for neuromorphic computing.

This module implements a minimal cache memory that stores key-value pairs
with a basic forgetting mechanism based on access frequency.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class CacheMemory:
    """
    Simple Cache Memory model.
    
    This model stores key-value pairs with access counters and implements
    a basic forgetting mechanism based on usage frequency.
    """
    
    def __init__(self, capacity: int = 10):
        """
        Initialize the cache memory model.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.cache = {}  # Dictionary to store key-value pairs
        self.access_count = {}  # Track access frequency
        self.insertion_order = []  # Track insertion order
    
    def store(self, key: Any, value: np.ndarray) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Key to associate with the value
            value: Value to store
        """
        # If cache is full, remove least accessed item
        if len(self.cache) >= self.capacity and key not in self.cache:
            self._forget_item()
        
        # Store the value
        self.cache[key] = value.copy()
        
        # Initialize or reset access count
        self.access_count[key] = 1
        
        # Update insertion order
        if key in self.insertion_order:
            self.insertion_order.remove(key)
        self.insertion_order.append(key)
    
    def retrieve(self, key: Any) -> Optional[np.ndarray]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Retrieved value or None if not found
        """
        if key in self.cache:
            # Increment access count
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def _forget_item(self) -> None:
        """
        Remove the least accessed item from the cache.
        If multiple items have the same access count, remove the oldest.
        """
        if not self.cache:
            return
        
        # Find the least accessed item
        min_access = min(self.access_count.values())
        least_accessed = [k for k, v in self.access_count.items() if v == min_access]
        
        # If multiple items have the same access count, remove the oldest
        for key in self.insertion_order:
            if key in least_accessed:
                # Remove the item
                del self.cache[key]
                del self.access_count[key]
                self.insertion_order.remove(key)
                break
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.access_count = {}
        self.insertion_order = []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache memory.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "utilization": len(self.cache) / self.capacity if self.capacity > 0 else 0,
            "avg_access_count": np.mean(list(self.access_count.values())) if self.access_count else 0
        }
    
    def get_most_accessed(self, n: int = 3) -> List[Tuple[Any, int]]:
        """
        Get the most frequently accessed items.
        
        Args:
            n: Number of items to return
            
        Returns:
            List of (key, access_count) tuples
        """
        sorted_items = sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]