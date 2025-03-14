"""
Unit tests for the Component class.
"""
import pytest
from src.core import Component


class TestComponent(Component):
    """Test implementation of the Component class."""
    
    def __init__(self, component_id=None, **kwargs):
        super().__init__(component_id, **kwargs)
        self.initialized = False
        self.updated = False
        self.update_count = 0
    
    def _initialize(self):
        self.initialized = True
    
    def _update(self, delta_time):
        self.updated = True
        self.update_count += 1


class TestComponentClass:
    """Tests for the Component class."""
    
    def test_component_initialization(self):
        """Test that a component can be initialized."""
        component = TestComponent()
        assert component.id is not None
        assert not component.initialized
        
        component.initialize()
        assert component.initialized
    
    def test_component_update(self):
        """Test that a component can be updated."""
        component = TestComponent()
        component.initialize()
        
        component.update(0.1)
        assert component.updated
        assert component.update_count == 1
        
        component.update(0.1)
        assert component.update_count == 2
    
    def test_component_hierarchy(self):
        """Test component parent-child relationships."""
        parent = TestComponent(component_id="parent")
        child1 = TestComponent(component_id="child1")
        child2 = TestComponent(component_id="child2")
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        assert len(parent.children) == 2
        assert child1.parent == parent
        assert child2.parent == parent
        
        # Test initialization propagation
        parent.initialize()
        assert parent.initialized
        assert child1.initialized
        assert child2.initialized
        
        # Test update propagation
        parent.update(0.1)
        assert parent.updated
        assert child1.updated
        assert child2.updated
        
        # Test removal
        parent.remove_child(child1)
        assert len(parent.children) == 1
        assert child1.parent is None
    
    def test_component_config(self):
        """Test component configuration."""
        component = TestComponent(test_param="value")
        assert component.config["test_param"] == "value"
        
        component.set_config(new_param="new_value")
        assert component.config["new_param"] == "new_value"
        
        config = component.get_config()
        assert config["test_param"] == "value"
        assert config["new_param"] == "new_value"