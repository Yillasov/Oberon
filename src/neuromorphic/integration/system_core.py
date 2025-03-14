"""
Central integration module for neuromorphic system components.
"""
import asyncio
from typing import Dict, Any, List, Callable, Optional
from enum import Enum
import logging
import json
import time


class ComponentStatus(Enum):
    OFFLINE = 0
    INITIALIZING = 1
    ONLINE = 2
    DEGRADED = 3
    ERROR = 4


class SystemCore:
    """Central integration system for neuromorphic components."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.components: Dict[str, Any] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.status = ComponentStatus.OFFLINE
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration."""
        if not config_path:
            return {"name": "Neuromorphic System", "version": "0.1"}
            
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Config load error: {str(e)}")
            return {"name": "Neuromorphic System", "version": "0.1"}
    
    def _setup_logger(self) -> logging.Logger:
        """Set up system logger."""
        logger = logging.getLogger("neuromorphic.system")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def register_component(self, name: str, component: Any) -> bool:
        """Register a system component."""
        if name in self.components:
            self.logger.warning(f"Component {name} already registered")
            return False
            
        self.components[name] = {
            "instance": component,
            "status": ComponentStatus.OFFLINE,
            "last_update": time.time()
        }
        self.logger.info(f"Component {name} registered")
        return True
    
    def subscribe_event(self, event_name: str, handler: Callable) -> bool:
        """Subscribe to system events."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
            
        self.event_handlers[event_name].append(handler)
        return True
    
    async def publish_event(self, event_name: str, data: Any) -> None:
        """Publish event to subscribers."""
        if event_name not in self.event_handlers:
            return
            
        for handler in self.event_handlers[event_name]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Event handler error: {str(e)}")
    
    async def start_system(self) -> bool:
        """Start all system components."""
        self.status = ComponentStatus.INITIALIZING
        self.logger.info("System starting...")
        
        # Start components in dependency order
        for name, component_data in self.components.items():
            try:
                component = component_data["instance"]
                if hasattr(component, "start") and callable(component.start):
                    if asyncio.iscoroutinefunction(component.start):
                        await component.start()
                    else:
                        component.start()
                        
                component_data["status"] = ComponentStatus.ONLINE
                component_data["last_update"] = time.time()
                self.logger.info(f"Component {name} started")
                
            except Exception as e:
                component_data["status"] = ComponentStatus.ERROR
                self.logger.error(f"Failed to start component {name}: {str(e)}")
        
        self.status = ComponentStatus.ONLINE
        await self.publish_event("system_started", {"timestamp": time.time()})
        return True
    
    async def stop_system(self) -> bool:
        """Stop all system components."""
        self.logger.info("System stopping...")
        
        # Stop components in reverse order
        for name, component_data in reversed(list(self.components.items())):
            try:
                component = component_data["instance"]
                if hasattr(component, "stop") and callable(component.stop):
                    if asyncio.iscoroutinefunction(component.stop):
                        await component.stop()
                    else:
                        component.stop()
                        
                component_data["status"] = ComponentStatus.OFFLINE
                self.logger.info(f"Component {name} stopped")
                
            except Exception as e:
                self.logger.error(f"Failed to stop component {name}: {str(e)}")
        
        self.status = ComponentStatus.OFFLINE
        await self.publish_event("system_stopped", {"timestamp": time.time()})
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        component_statuses = {
            name: {
                "status": data["status"].name,
                "last_update": data["last_update"]
            }
            for name, data in self.components.items()
        }
        
        return {
            "system_status": self.status.name,
            "component_count": len(self.components),
            "components": component_statuses,
            "timestamp": time.time()
        }