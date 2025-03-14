"""
Simple event bus for inter-component communication in neuromorphic systems.
"""
from typing import Dict, Any, List, Callable, Optional, Set
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from .system_core import SystemCore, ComponentStatus
from .component_registry import ComponentRegistry, ComponentMetadata


@dataclass
class EventMessage:
    """Message structure for event bus communication."""
    event_id: str
    event_type: str
    source: str
    timestamp: float
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Simple event bus for inter-component communication."""
    
    def __init__(self, system_core: SystemCore, registry: ComponentRegistry):
        self.core = system_core
        self.registry = registry
        
        # Event subscriptions: {event_type: {component_name: [handlers]}}
        self.subscriptions: Dict[str, Dict[str, List[Callable]]] = {}
        
        # Event history for debugging
        self.event_history: List[EventMessage] = []
        self.max_history = 100
        
        # Register with system
        self.core.register_component("event_bus", self)
        self.registry.register(
            self,
            ComponentMetadata(
                name="event_bus",
                component_type="integration",
                version="1.0",
                provides={"event_bus"},
                startup_priority=90  # Start early
            )
        )
    
    async def start(self):
        """Start the event bus."""
        self.core.logger.info("Event bus starting")
        # Subscribe to component status changes
        self.core.subscribe_event("component_status_change", self._handle_status_change)
    
    async def stop(self):
        """Stop the event bus."""
        self.core.logger.info("Event bus stopping")
        self.subscriptions.clear()
    
    def subscribe(self, 
                 component_name: str, 
                 event_type: str, 
                 handler: Callable[[EventMessage], None]) -> bool:
        """Subscribe component to event type."""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = {}
            
        if component_name not in self.subscriptions[event_type]:
            self.subscriptions[event_type][component_name] = []
            
        self.subscriptions[event_type][component_name].append(handler)
        self.core.logger.debug(f"Component {component_name} subscribed to {event_type}")
        return True
    
    def unsubscribe(self, component_name: str, event_type: Optional[str] = None) -> bool:
        """Unsubscribe component from events."""
        if event_type is None:
            # Unsubscribe from all events
            for evt_type in self.subscriptions:
                if component_name in self.subscriptions[evt_type]:
                    del self.subscriptions[evt_type][component_name]
            return True
            
        if event_type in self.subscriptions:
            if component_name in self.subscriptions[event_type]:
                del self.subscriptions[event_type][component_name]
                return True
                
        return False
    
    async def publish(self, 
                    source: str, 
                    event_type: str, 
                    data: Any, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Publish event to subscribers."""
        event_id = str(uuid.uuid4())
        timestamp = time.time()
        
        event = EventMessage(
            event_id=event_id,
            event_type=event_type,
            source=source,
            timestamp=timestamp,
            data=data,
            metadata=metadata or {}
        )
        
        # Store in history
        self._add_to_history(event)
        
        # Notify subscribers
        await self._notify_subscribers(event)
        
        return event_id
    
    async def _notify_subscribers(self, event: EventMessage):
        """Notify all subscribers of an event."""
        if event.event_type not in self.subscriptions:
            return
            
        for component_name, handlers in self.subscriptions[event.event_type].items():
            # Check if component is active
            component_data = self.core.components.get(component_name)
            if not component_data or component_data["status"] != ComponentStatus.ONLINE:
                continue
                
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.core.logger.error(
                        f"Error in event handler for {component_name}: {str(e)}"
                    )
    
    def _add_to_history(self, event: EventMessage):
        """Add event to history with size limit."""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
    
    async def _handle_status_change(self, data: Dict[str, Any]):
        """Handle component status change events."""
        component_name = data.get("component")
        new_status = data.get("status")
        
        if component_name and new_status == ComponentStatus.OFFLINE.name:
            # Component went offline, remove subscriptions
            self.unsubscribe(component_name)
    
    def get_event_types(self) -> Set[str]:
        """Get all registered event types."""
        return set(self.subscriptions.keys())
    
    def get_subscribers(self, event_type: str) -> List[str]:
        """Get all subscribers for an event type."""
        if event_type not in self.subscriptions:
            return []
            
        return list(self.subscriptions[event_type].keys())
    
    def get_recent_events(self, 
                        event_type: Optional[str] = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events for debugging."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]
        
        return [
            {
                "id": e.event_id,
                "type": e.event_type,
                "source": e.source,
                "timestamp": e.timestamp,
                "data_summary": str(e.data)[:100] + "..." if len(str(e.data)) > 100 else str(e.data)
            }
            for e in events
        ]