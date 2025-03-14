"""
Startup/shutdown sequence handler for neuromorphic system.
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from .system_core import SystemCore, ComponentStatus
from .component_registry import ComponentRegistry, ComponentMetadata
from .event_bus import EventBus
from .config_loader import ConfigLoader


class SystemState(Enum):
    """System lifecycle states."""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4


class SystemLifecycle:
    """Startup/shutdown sequence handler for neuromorphic system."""
    
    def __init__(
        self, 
        system_core: SystemCore, 
        registry: ComponentRegistry, 
        event_bus: EventBus,
        config_loader: ConfigLoader
    ):
        self.core = system_core
        self.registry = registry
        self.event_bus = event_bus
        self.config = config_loader
        self.logger = system_core.logger
        
        # System state
        self.state = SystemState.STOPPED
        self.startup_time = 0
        self.shutdown_hooks: List[Callable] = []
        
        # Register with system
        self.core.register_component("system_lifecycle", self)
        self.registry.register(
            self,
            ComponentMetadata(
                name="system_lifecycle",
                component_type="integration",
                version="1.0",
                provides={"lifecycle_management"},
                dependencies={"event_bus", "config_service"},
                startup_priority=99  # Highest priority
            )
        )
    
    async def start_system(self):
        """Start the entire system in the correct sequence."""
        if self.state != SystemState.STOPPED:
            self.logger.warning(f"Cannot start system from state {self.state.name}")
            return False
        
        try:
            self.state = SystemState.STARTING
            self.startup_time = time.time()
            self.logger.info("System starting...")
            
            # Publish system starting event
            await self.event_bus.publish(
                "system_lifecycle",
                "system_starting",
                {"timestamp": self.startup_time}
            )
            
            # Start components using registry
            await self.registry.start_components()
            
            # Update system state
            self.state = SystemState.RUNNING
            
            # Publish system started event
            await self.event_bus.publish(
                "system_lifecycle",
                "system_started",
                {
                    "timestamp": time.time(),
                    "startup_time": time.time() - self.startup_time
                }
            )
            
            self.logger.info(f"System started in {time.time() - self.startup_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"System startup failed: {str(e)}")
            
            # Publish error event
            await self.event_bus.publish(
                "system_lifecycle",
                "system_error",
                {
                    "timestamp": time.time(),
                    "error": str(e)
                }
            )
            
            return False
    
    async def stop_system(self):
        """Stop the entire system in the correct sequence."""
        if self.state not in [SystemState.RUNNING, SystemState.ERROR]:
            self.logger.warning(f"Cannot stop system from state {self.state.name}")
            return False
        
        try:
            self.state = SystemState.STOPPING
            stop_time = time.time()
            self.logger.info("System stopping...")
            
            # Publish system stopping event
            await self.event_bus.publish(
                "system_lifecycle",
                "system_stopping",
                {"timestamp": stop_time}
            )
            
            # Run shutdown hooks
            for hook in self.shutdown_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    self.logger.error(f"Shutdown hook error: {str(e)}")
            
            # Stop components using registry
            await self.registry.stop_components()
            
            # Update system state
            self.state = SystemState.STOPPED
            
            # Publish system stopped event
            await self.event_bus.publish(
                "system_lifecycle",
                "system_stopped",
                {
                    "timestamp": time.time(),
                    "shutdown_time": time.time() - stop_time
                }
            )
            
            self.logger.info(f"System stopped in {time.time() - stop_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"System shutdown failed: {str(e)}")
            
            # Publish error event
            await self.event_bus.publish(
                "system_lifecycle",
                "system_error",
                {
                    "timestamp": time.time(),
                    "error": str(e)
                }
            )
            
            return False
    
    def add_shutdown_hook(self, hook: Callable):
        """Add a function to be called during system shutdown."""
        self.shutdown_hooks.append(hook)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state information."""
        uptime = 0
        if self.state == SystemState.RUNNING and self.startup_time > 0:
            uptime = time.time() - self.startup_time
            
        return {
            "state": self.state.name,
            "uptime": uptime,
            "startup_time": self.startup_time,
            "component_count": len(self.core.components)
        }