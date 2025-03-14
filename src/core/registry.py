"""
Component registry for the Neuromorphic Biomimetic UCAV SDK.

Provides a central registry for tracking and retrieving components.
"""
from typing import Dict, Type, Optional, List
from .component import Component


class ComponentRegistry:
    """
    Registry for managing component instances.
    
    Provides functionality to register, retrieve, and manage components.
    """
    
    def __init__(self):
        """Initialize a new component registry."""
        self._components: Dict[str, Component] = {}
        self._component_types: Dict[str, Type[Component]] = {}
    
    def register_component(self, component: Component) -> None:
        """
        Register a component instance.
        
        Args:
            component: The component to register.
        """
        self._components[component.id] = component
    
    def unregister_component(self, component_id: str) -> None:
        """
        Unregister a component instance.
        
        Args:
            component_id: ID of the component to unregister.
        """
        if component_id in self._components:
            del self._components[component_id]
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """
        Get a component by ID.
        
        Args:
            component_id: ID of the component to retrieve.
            
        Returns:
            The component if found, None otherwise.
        """
        return self._components.get(component_id)
    
    def register_component_type(self, name: str, component_class: Type[Component]) -> None:
        """
        Register a component type.
        
        Args:
            name: Name to register the component type under.
            component_class: The component class to register.
        """
        self._component_types[name] = component_class
    
    def get_component_type(self, name: str) -> Optional[Type[Component]]:
        """
        Get a component type by name.
        
        Args:
            name: Name of the component type.
            
        Returns:
            The component class if found, None otherwise.
        """
        return self._component_types.get(name)
    
    def get_all_components(self) -> List[Component]:
        """
        Get all registered components.
        
        Returns:
            List of all registered components.
        """
        return list(self._components.values())
    
    def clear(self) -> None:
        """Clear all registered components and component types."""
        self._components.clear()
        self._component_types.clear()


# Global component registry instance
registry = ComponentRegistry()