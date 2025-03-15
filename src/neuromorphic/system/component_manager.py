"""
System to manage dependencies and components for neuromorphic applications.
"""
import importlib
import logging
import os
import json
from typing import Dict, Any, List, Optional, Callable, Type, Set
from enum import Enum

class ComponentType(Enum):
    """Types of components in the neuromorphic system."""
    ENCODER = "encoder"
    DECODER = "decoder"
    NETWORK = "network"
    MONITOR = "monitor"
    HARDWARE = "hardware"
    SIMULATOR = "simulator"
    UTILITY = "utility"

class ComponentStatus(Enum):
    """Status of components in the system."""
    AVAILABLE = "available"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

class Component:
    """Base class for all system components."""
    
    def __init__(self, name: str, component_type: ComponentType):
        """
        Initialize component.
        
        Args:
            name: Unique name of the component
            component_type: Type of the component
        """
        self.name = name
        self.type = component_type
        self.status = ComponentStatus.AVAILABLE
        self.dependencies = set()
        self.config = {}
        self.logger = logging.getLogger(f"Component-{name}")
    
    def add_dependency(self, component_name: str):
        """
        Add a dependency to this component.
        
        Args:
            component_name: Name of the dependency
        """
        self.dependencies.add(component_name)
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the component.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def activate(self) -> bool:
        """
        Activate the component.
        
        Returns:
            True if activation was successful
        """
        try:
            self.status = ComponentStatus.ACTIVE
            self.logger.info(f"Component {self.name} activated")
            return True
        except Exception as e:
            self.status = ComponentStatus.ERROR
            self.logger.error(f"Error activating component {self.name}: {e}")
            return False
    
    def deactivate(self) -> bool:
        """
        Deactivate the component.
        
        Returns:
            True if deactivation was successful
        """
        try:
            self.status = ComponentStatus.AVAILABLE
            self.logger.info(f"Component {self.name} deactivated")
            return True
        except Exception as e:
            self.status = ComponentStatus.ERROR
            self.logger.error(f"Error deactivating component {self.name}: {e}")
            return False
    
    def get_status(self) -> ComponentStatus:
        """
        Get component status.
        
        Returns:
            Current status of the component
        """
        return self.status
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get component information.
        
        Returns:
            Dictionary with component information
        """
        return {
            "name": self.name,
            "type": self.type.value,
            "status": self.status.value,
            "dependencies": list(self.dependencies)
        }


class ComponentRegistry:
    """Registry for managing system components."""
    
    def __init__(self):
        """Initialize component registry."""
        self.components = {}
        self.component_classes = {}
        self.logger = logging.getLogger("ComponentRegistry")
    
    def register_component_class(self, component_type: ComponentType, 
                               class_path: str, class_name: str):
        """
        Register a component class.
        
        Args:
            component_type: Type of component
            class_path: Import path to the class
            class_name: Name of the class
        """
        key = f"{component_type.value}.{class_name}"
        self.component_classes[key] = {
            "type": component_type,
            "path": class_path,
            "name": class_name
        }
        self.logger.info(f"Registered component class: {key}")
    
    def create_component(self, component_type: ComponentType, class_name: str, 
                        instance_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Component]:
        """
        Create a component instance.
        
        Args:
            component_type: Type of component
            class_name: Name of the component class
            instance_name: Name for the new instance
            config: Optional configuration
            
        Returns:
            Created component or None if creation failed
        """
        key = f"{component_type.value}.{class_name}"
        if key not in self.component_classes:
            self.logger.error(f"Component class not found: {key}")
            return None
        
        if instance_name in self.components:
            self.logger.error(f"Component with name {instance_name} already exists")
            return None
        
        try:
            # Import the module and create an instance
            class_info = self.component_classes[key]
            module = importlib.import_module(class_info["path"])
            component_class = getattr(module, class_info["name"])
            
            # Create the component
            component = component_class(instance_name, component_type)
            
            # Configure if needed
            if config:
                component.configure(config)
            
            # Register the component
            self.components[instance_name] = component
            self.logger.info(f"Created component: {instance_name} ({key})")
            
            return component
            
        except Exception as e:
            self.logger.error(f"Error creating component {instance_name}: {e}")
            return None
    
    def get_component(self, name: str) -> Optional[Component]:
        """
        Get a component by name.
        
        Args:
            name: Name of the component
            
        Returns:
            Component or None if not found
        """
        return self.components.get(name)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[Component]:
        """
        Get all components of a specific type.
        
        Args:
            component_type: Type of components to get
            
        Returns:
            List of components
        """
        return [
            component for component in self.components.values()
            if component.type == component_type
        ]
    
    def get_active_components(self) -> List[Component]:
        """
        Get all active components.
        
        Returns:
            List of active components
        """
        return [
            component for component in self.components.values()
            if component.status == ComponentStatus.ACTIVE
        ]
    
    def remove_component(self, name: str) -> bool:
        """
        Remove a component.
        
        Args:
            name: Name of the component
            
        Returns:
            True if component was removed
        """
        if name not in self.components:
            return False
        
        # Deactivate first
        component = self.components[name]
        if component.status == ComponentStatus.ACTIVE:
            component.deactivate()
        
        # Remove
        del self.components[name]
        self.logger.info(f"Removed component: {name}")
        return True
    
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Get the dependency graph of all components.
        
        Returns:
            Dictionary mapping component names to sets of dependency names
        """
        return {
            name: component.dependencies
            for name, component in self.components.items()
        }
    
    def resolve_dependencies(self, component_name: str) -> List[str]:
        """
        Resolve dependencies for a component in order of activation.
        
        Args:
            component_name: Name of the component
            
        Returns:
            List of component names in order of activation
        """
        if component_name not in self.components:
            return []
        
        # Build dependency graph
        graph = self.get_dependency_graph()
        
        # Track visited components and activation order
        visited = set()
        activation_order = []
        
        def visit(name):
            if name in visited:
                return
            
            if name not in self.components:
                self.logger.warning(f"Missing dependency: {name}")
                return
            
            visited.add(name)
            
            # Visit dependencies first
            for dep in graph.get(name, set()):
                visit(dep)
            
            activation_order.append(name)
        
        # Start DFS from the requested component
        visit(component_name)
        
        return activation_order
    
    def activate_component(self, name: str, resolve_deps: bool = True) -> bool:
        """
        Activate a component and optionally its dependencies.
        
        Args:
            name: Name of the component
            resolve_deps: Whether to resolve and activate dependencies
            
        Returns:
            True if activation was successful
        """
        if name not in self.components:
            self.logger.error(f"Component not found: {name}")
            return False
        
        component = self.components[name]
        
        # If already active, nothing to do
        if component.status == ComponentStatus.ACTIVE:
            return True
        
        # Resolve dependencies if requested
        if resolve_deps:
            activation_order = self.resolve_dependencies(name)
            
            # Activate dependencies in order
            for dep_name in activation_order:
                if dep_name == name:
                    continue  # Skip the component itself
                
                dep = self.components[dep_name]
                if dep.status != ComponentStatus.ACTIVE:
                    if not dep.activate():
                        self.logger.error(f"Failed to activate dependency: {dep_name}")
                        return False
        
        # Activate the component
        return component.activate()
    
    def deactivate_component(self, name: str, cascade: bool = False) -> bool:
        """
        Deactivate a component and optionally components that depend on it.
        
        Args:
            name: Name of the component
            cascade: Whether to deactivate dependent components
            
        Returns:
            True if deactivation was successful
        """
        if name not in self.components:
            self.logger.error(f"Component not found: {name}")
            return False
        
        component = self.components[name]
        
        # If already inactive, nothing to do
        if component.status != ComponentStatus.ACTIVE:
            return True
        
        # If cascade is requested, find and deactivate dependent components
        if cascade:
            # Find components that depend on this one
            dependents = [
                c_name for c_name, c in self.components.items()
                if name in c.dependencies and c.status == ComponentStatus.ACTIVE
            ]
            
            # Deactivate dependents
            for dep_name in dependents:
                if not self.deactivate_component(dep_name, cascade=True):
                    self.logger.error(f"Failed to deactivate dependent: {dep_name}")
                    return False
        
        # Deactivate the component
        return component.deactivate()
    
    def save_registry(self, filepath: str) -> bool:
        """
        Save registry state to file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if save was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare data to save
            data = {
                "components": {
                    name: {
                        "type": component.type.value,
                        "status": component.status.value,
                        "dependencies": list(component.dependencies),
                        "config": component.config
                    }
                    for name, component in self.components.items()
                },
                "component_classes": self.component_classes
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Registry saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
            return False
    
    def load_registry(self, filepath: str) -> bool:
        """
        Load registry state from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            True if load was successful
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Registry file not found: {filepath}")
            return False
        
        try:
            # Load from file
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load component classes
            self.component_classes = data.get("component_classes", {})
            
            # Load components
            for name, info in data.get("components", {}).items():
                component_type = ComponentType(info["type"])
                
                # Create a basic component
                component = Component(name, component_type)
                component.status = ComponentStatus(info["status"])
                component.dependencies = set(info["dependencies"])
                component.config = info["config"]
                
                # Register the component
                self.components[name] = component
            
            self.logger.info(f"Registry loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}")
            return False


# Simple usage example
def component_manager_example():
    """Example of using the component manager."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create registry
    registry = ComponentRegistry()
    
    # Register component classes
    registry.register_component_class(
        ComponentType.ENCODER, 
        "neuromorphic.encoding.spike_encoder", 
        "SpikeEncoder"
    )
    
    registry.register_component_class(
        ComponentType.DECODER, 
        "neuromorphic.encoding.spike_decoder", 
        "SpikeDecoder"
    )
    
    registry.register_component_class(
        ComponentType.NETWORK, 
        "neuromorphic.network.snn", 
        "SpikingNeuralNetwork"
    )
    
    # Create components
    encoder = registry.create_component(
        ComponentType.ENCODER,
        "SpikeEncoder",
        "input_encoder",
        {"input_size": 10, "encoding_method": "rate"}
    )
    
    decoder = registry.create_component(
        ComponentType.DECODER,
        "SpikeDecoder",
        "output_decoder",
        {"output_size": 5, "decoding_method": "rate"}
    )
    
    network = registry.create_component(
        ComponentType.NETWORK,
        "SpikingNeuralNetwork",
        "main_network",
        {"neuron_count": 100, "learning_enabled": True}
    )
    
    # Set up dependencies
    if network:
        network.add_dependency("input_encoder")
        network.add_dependency("output_decoder")
    
    # Activate with dependency resolution
    if network:
        success = registry.activate_component("main_network", resolve_deps=True)
        print(f"Network activation {'successful' if success else 'failed'}")
    
    # Get active components
    active = registry.get_active_components()
    print(f"Active components: {[c.name for c in active]}")
    
    # Save registry state
    registry.save_registry("/Users/yessine/Oberon/models/registry.json")


if __name__ == "__main__":
    component_manager_example()