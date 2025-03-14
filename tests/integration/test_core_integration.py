"""
Integration tests for the core module.
"""
import pytest
from src.core import Component, Configuration, registry


class SimpleComponent(Component):
    """Simple component implementation for testing."""
    
    def __init__(self, component_id=None, **kwargs):
        super().__init__(component_id, **kwargs)
        self.initialized = False
    
    def _initialize(self):
        self.initialized = True


class TestCoreIntegration:
    """Integration tests for core module components."""
    
    def test_component_registry_integration(self):
        """Test integration between Component and ComponentRegistry."""
        # Create components
        comp1 = SimpleComponent(component_id="comp1")
        comp2 = SimpleComponent(component_id="comp2")
        
        # Register components
        registry.register_component(comp1)
        registry.register_component(comp2)
        
        # Retrieve components
        retrieved_comp1 = registry.get_component("comp1")
        retrieved_comp2 = registry.get_component("comp2")
        
        assert retrieved_comp1 is comp1
        assert retrieved_comp2 is comp2
        
        # Initialize components
        comp1.initialize()
        assert comp1.initialized
        
        # Unregister component
        registry.unregister_component("comp1")
        assert registry.get_component("comp1") is None
        assert registry.get_component("comp2") is comp2
        
        # Clear registry
        registry.clear()
        assert registry.get_component("comp2") is None
    
    def test_component_configuration_integration(self):
        """Test integration between Component and Configuration."""
        # Create configuration
        config = Configuration({
            "components": {
                "test_component": {
                    "param1": "value1",
                    "param2": 42
                }
            }
        })
        
        # Create component with configuration
        component_config = config.get("components.test_component")
        component = SimpleComponent(**component_config)
        
        assert component.config["param1"] == "value1"
        assert component.config["param2"] == 42
        
        # Update configuration
        config.set("components.test_component.param1", "updated_value")
        component.set_config(**config.get("components.test_component"))
        
        assert component.config["param1"] == "updated_value"