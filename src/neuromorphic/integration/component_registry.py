"""
Component registration system for neuromorphic architecture.
"""
from typing import Dict, Any, List, Set, Optional, Callable, Type
from dataclasses import dataclass, field
import asyncio
import time
from .system_core import SystemCore, ComponentStatus


@dataclass
class ComponentMetadata:
    """Metadata for registered components."""
    name: str
    component_type: str
    version: str = "1.0"
    dependencies: Set[str] = field(default_factory=set)
    provides: Set[str] = field(default_factory=set)
    startup_priority: int = 50  # 0-100, higher starts earlier
    shutdown_priority: int = 50  # 0-100, higher stops later
    config: Dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """Registry for system components with dependency management."""
    
    def __init__(self, system_core: SystemCore):
        self.core = system_core
        self.metadata: Dict[str, ComponentMetadata] = {}
        self.lifecycle_hooks: Dict[str, Dict[str, List[Callable]]] = {}
        
        # Register self with system core
        self.core.register_component("component_registry", self)
    
    def register(self, 
                component: Any, 
                metadata: ComponentMetadata) -> bool:
        """Register component with metadata."""
        if metadata.name in self.metadata:
            self.core.logger.warning(f"Component {metadata.name} already registered")
            return False
        
        # Store metadata
        self.metadata[metadata.name] = metadata
        
        # Initialize lifecycle hooks
        self.lifecycle_hooks[metadata.name] = {
            "pre_start": [],
            "post_start": [],
            "pre_stop": [],
            "post_stop": []
        }
        
        # Register with system core
        success = self.core.register_component(metadata.name, component)
        
        if success:
            self.core.logger.info(
                f"Registered {metadata.component_type} component: {metadata.name} v{metadata.version}"
            )
        
        return success
    
    def add_lifecycle_hook(self, 
                         component_name: str, 
                         hook_type: str, 
                         hook: Callable) -> bool:
        """Add lifecycle hook for component."""
        if component_name not in self.lifecycle_hooks:
            self.core.logger.error(f"Component {component_name} not registered")
            return False
            
        if hook_type not in self.lifecycle_hooks[component_name]:
            self.core.logger.error(f"Invalid hook type: {hook_type}")
            return False
            
        self.lifecycle_hooks[component_name][hook_type].append(hook)
        return True
    
    async def start_components(self) -> bool:
        """Start components in dependency order."""
        # Sort components by dependencies and priority
        start_order = self._calculate_start_order()
        
        for component_name in start_order:
            # Run pre-start hooks
            await self._run_hooks(component_name, "pre_start")
            
            # Start component
            component_data = self.core.components.get(component_name)
            if not component_data:
                self.core.logger.error(f"Component {component_name} not found in core")
                continue
                
            component = component_data["instance"]
            try:
                if hasattr(component, "start") and callable(component.start):
                    if asyncio.iscoroutinefunction(component.start):
                        await component.start()
                    else:
                        component.start()
                        
                component_data["status"] = ComponentStatus.ONLINE
                component_data["last_update"] = time.time()
                self.core.logger.info(f"Started component {component_name}")
                
                # Run post-start hooks
                await self._run_hooks(component_name, "post_start")
                
            except Exception as e:
                component_data["status"] = ComponentStatus.ERROR
                self.core.logger.error(f"Failed to start {component_name}: {str(e)}")
        
        return True
    
    async def stop_components(self) -> bool:
        """Stop components in reverse dependency order."""
        # Calculate stop order (reverse of start order with priority adjustment)
        stop_order = list(reversed(self._calculate_start_order()))
        
        for component_name in stop_order:
            # Run pre-stop hooks
            await self._run_hooks(component_name, "pre_stop")
            
            # Stop component
            component_data = self.core.components.get(component_name)
            if not component_data:
                continue
                
            component = component_data["instance"]
            try:
                if hasattr(component, "stop") and callable(component.stop):
                    if asyncio.iscoroutinefunction(component.stop):
                        await component.stop()
                    else:
                        component.stop()
                        
                component_data["status"] = ComponentStatus.OFFLINE
                self.core.logger.info(f"Stopped component {component_name}")
                
                # Run post-stop hooks
                await self._run_hooks(component_name, "post_stop")
                
            except Exception as e:
                self.core.logger.error(f"Failed to stop {component_name}: {str(e)}")
        
        return True
    
    def _calculate_start_order(self) -> List[str]:
        """Calculate component start order based on dependencies."""
        # Create dependency graph
        graph = {name: meta.dependencies for name, meta in self.metadata.items()}
        
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected: {node}")
            if node in visited:
                return
                
            temp_visited.add(node)
            
            # Visit dependencies
            for dep in graph.get(node, set()):
                if dep in graph:
                    visit(dep)
            
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        # Visit all nodes
        for node in graph:
            if node not in visited:
                visit(node)
        
        # Sort by priority within dependency constraints
        priority_dict = {name: meta.startup_priority for name, meta in self.metadata.items()}
        return sorted(order, key=lambda x: priority_dict.get(x, 50), reverse=True)
    
    async def _run_hooks(self, component_name: str, hook_type: str):
        """Run lifecycle hooks for component."""
        if component_name not in self.lifecycle_hooks:
            return
            
        for hook in self.lifecycle_hooks[component_name].get(hook_type, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                self.core.logger.error(
                    f"Error in {hook_type} hook for {component_name}: {str(e)}"
                )
    
    def get_component_info(self, component_name: str) -> Dict[str, Any]:
        """Get detailed component information."""
        if component_name not in self.metadata:
            return {}
            
        meta = self.metadata[component_name]
        component_data = self.core.components.get(component_name, {})
        
        return {
            "name": meta.name,
            "type": meta.component_type,
            "version": meta.version,
            "dependencies": list(meta.dependencies),
            "provides": list(meta.provides),
            "status": component_data.get("status", ComponentStatus.OFFLINE).name,
            "last_update": component_data.get("last_update", 0)
        }