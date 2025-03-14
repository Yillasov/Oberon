"""
Base component system for the Neuromorphic Biomimetic UCAV SDK.

This module provides the foundational Component class that all UCAV
components will inherit from, establishing a consistent interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set
import uuid


class Component(ABC):
    """
    Abstract base class for all UCAV components.
    
    Provides core functionality for component identification, 
    configuration, and lifecycle management.
    """
    
    def __init__(self, component_id: Optional[str] = None, **kwargs):
        """
        Initialize a new component.
        
        Args:
            component_id: Unique identifier for this component. If None, a UUID will be generated.
            **kwargs: Additional configuration parameters for the component.
        """
        self.id = component_id if component_id else str(uuid.uuid4())
        self.config = kwargs
        self.parent = None
        self.children: List[Component] = []
        self._is_initialized = False
        
    def add_child(self, component: 'Component') -> None:
        """
        Add a child component to this component.
        
        Args:
            component: The component to add as a child.
        """
        if component not in self.children:
            self.children.append(component)
            component.parent = self
    
    def remove_child(self, component: 'Component') -> None:
        """
        Remove a child component from this component.
        
        Args:
            component: The component to remove.
        """
        if component in self.children:
            self.children.remove(component)
            component.parent = None
    
    def initialize(self) -> None:
        """Initialize the component and its children."""
        if not self._is_initialized:
            self._initialize()
            self._is_initialized = True
            
        for child in self.children:
            if not child._is_initialized:
                child.initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Component-specific initialization logic.
        
        Must be implemented by subclasses.
        """
        pass
    
    def shutdown(self) -> None:
        """Shutdown the component and its children."""
        # Shutdown children first
        for child in self.children:
            child.shutdown()
            
        # Then shutdown self
        if self._is_initialized:
            self._shutdown()
            self._is_initialized = False
    
    def _shutdown(self) -> None:
        """
        Component-specific shutdown logic.
        
        Can be overridden by subclasses.
        """
        pass
    
    def update(self, delta_time: float) -> None:
        """
        Update component state.
        
        Args:
            delta_time: Time elapsed since last update in seconds.
        """
        self._update(delta_time)
        
        # Update children
        for child in self.children:
            child.update(delta_time)
    
    def _update(self, delta_time: float) -> None:
        """
        Component-specific update logic.
        
        Can be overridden by subclasses.
        
        Args:
            delta_time: Time elapsed since last update in seconds.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the component's configuration.
        
        Returns:
            Dict containing the component's configuration.
        """
        return self.config.copy()
    
    def set_config(self, **kwargs) -> None:
        """
        Update the component's configuration.
        
        Args:
            **kwargs: Configuration parameters to update.
        """
        self.config.update(kwargs)
        
    def __repr__(self) -> str:
        """String representation of the component."""
        return f"{self.__class__.__name__}(id={self.id})"