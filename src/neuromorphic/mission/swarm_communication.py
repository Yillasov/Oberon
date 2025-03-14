"""
Resilient communication protocols for neuromorphic swarm operations.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import hashlib
from datetime import datetime
from .swarm_behavior import SwarmAgent, SwarmRole
from .distributed_planner import PlanningTask


class MessagePriority(Enum):
    EMERGENCY = 3
    HIGH = 2
    NORMAL = 1
    LOW = 0


@dataclass
class SwarmMessage:
    """Message structure for swarm communication."""
    message_id: str
    sender_id: str
    recipients: Set[str]
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: float
    ttl: int = 5  # Time-to-live hops
    signature: Optional[str] = None


class CommunicationChannel:
    """Resilient communication channel for swarm agents."""
    
    def __init__(self, reliability_threshold: float = 0.7):
        self.message_queue: List[SwarmMessage] = []
        self.delivered_messages: Dict[str, Set[str]] = {}
        self.channel_status: Dict[str, float] = {}
        self.reliability_threshold = reliability_threshold
        self.message_handlers: Dict[str, Callable] = {}
        
        # Communication metrics
        self.latency_history: List[float] = []
        self.reliability_metrics: Dict[str, float] = {}
        self.bandwidth_usage: Dict[str, int] = {}
        
        # Fault tolerance
        self.backup_routes: Dict[str, List[str]] = {}
        self.error_counters: Dict[str, int] = {}
        self.max_retries = 3
    
    async def broadcast_message(self, 
                              message: SwarmMessage,
                              source_agent: SwarmAgent) -> bool:
        """Broadcast message to swarm network."""
        try:
            # Sign message
            message.signature = self._sign_message(message)
            
            # Update channel status
            await self._update_channel_status(source_agent)
            
            # Prioritize message
            self._prioritize_message(message)
            
            # Attempt delivery
            success = await self._deliver_message(message)
            
            if success:
                self._update_metrics(message, 0.0)  # Instant delivery
                return True
            
            # Fallback to backup routes
            if message.priority in [MessagePriority.EMERGENCY, MessagePriority.HIGH]:
                return await self._attempt_backup_delivery(message)
            
            return False
            
        except Exception as e:
            self._handle_error(str(e), source_agent.agent_id)
            return False
    
    def register_handler(self, 
                        message_type: str, 
                        handler: Callable[[SwarmMessage], None]):
        """Register message handler for specific message type."""
        self.message_handlers[message_type] = handler
    
    async def _deliver_message(self, message: SwarmMessage) -> bool:
        """Attempt to deliver message to recipients."""
        start_time = datetime.now().timestamp()
        
        # Check channel reliability
        reliable_channels = {
            recipient: self.channel_status.get(recipient, 0.0)
            for recipient in message.recipients
            if self.channel_status.get(recipient, 0.0) > self.reliability_threshold
        }
        
        if not reliable_channels:
            return False
        
        # Deliver to reliable channels
        delivered_to = set()
        for recipient, reliability in reliable_channels.items():
            if await self._send_to_recipient(message, recipient):
                delivered_to.add(recipient)
        
        # Record delivery status
        self.delivered_messages[message.message_id] = delivered_to
        
        # Update metrics
        delivery_time = datetime.now().timestamp() - start_time
        self._update_metrics(message, delivery_time)
        
        return len(delivered_to) > 0
    
    async def _send_to_recipient(self, 
                               message: SwarmMessage, 
                               recipient: str) -> bool:
        """Send message to specific recipient."""
        try:
            if message.ttl <= 0:
                return False
            
            # Verify message integrity
            if not self._verify_message(message):
                return False
            
            # Process message
            if recipient in self.message_handlers:
                await asyncio.create_task(
                    self.message_handlers[recipient](message)
                )
            
            # Update bandwidth usage
            self._update_bandwidth(message, recipient)
            
            return True
            
        except Exception as e:
            self._handle_error(str(e), recipient)
            return False
    
    async def _attempt_backup_delivery(self, message: SwarmMessage) -> bool:
        """Attempt delivery through backup routes."""
        if not self.backup_routes:
            return False
        
        message.ttl -= 1
        if message.ttl <= 0:
            return False
        
        # Try backup routes
        for recipient in message.recipients:
            if recipient in self.backup_routes:
                for backup_route in self.backup_routes[recipient]:
                    if await self._send_to_recipient(message, backup_route):
                        return True
        
        return False
    
    def _sign_message(self, message: SwarmMessage) -> str:
        """Create message signature for integrity verification."""
        content = f"{message.message_id}{message.sender_id}" + \
                 f"{sorted(message.recipients)}{message.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _verify_message(self, message: SwarmMessage) -> bool:
        """Verify message integrity."""
        if not message.signature:
            return False
        return message.signature == self._sign_message(message)
    
    def _prioritize_message(self, message: SwarmMessage):
        """Prioritize message in queue."""
        insert_idx = 0
        for idx, queued in enumerate(self.message_queue):
            if queued.priority.value < message.priority.value:
                insert_idx = idx
                break
        self.message_queue.insert(insert_idx, message)
    
    async def _update_channel_status(self, agent: SwarmAgent):
        """Update communication channel status."""
        for neighbor in self._get_connected_agents(agent):
            reliability = self._calculate_channel_reliability(agent, neighbor)
            self.channel_status[neighbor] = reliability
            
            if reliability < self.reliability_threshold:
                await self._establish_backup_route(agent, neighbor)
    
    def _get_connected_agents(self, agent: SwarmAgent) -> List[str]:
        """Get list of connected agent IDs."""
        # Implementation depends on swarm topology
        return [a for a in self.channel_status.keys()
                if a != agent.agent_id]
    
    def _calculate_channel_reliability(self,
                                    source: SwarmAgent,
                                    target: str) -> float:
        """Calculate channel reliability metric."""
        if target not in self.error_counters:
            return 1.0
        
        errors = self.error_counters[target]
        return max(0.0, 1.0 - (errors / self.max_retries))
    
    async def _establish_backup_route(self,
                                    agent: SwarmAgent,
                                    target: str):
        """Establish backup communication route."""
        connected = self._get_connected_agents(agent)
        reliable = [
            a for a in connected
            if self.channel_status.get(a, 0.0) > self.reliability_threshold
        ]
        
        if reliable:
            self.backup_routes[target] = reliable
    
    def _update_metrics(self,
                       message: SwarmMessage,
                       delivery_time: float):
        """Update communication metrics."""
        self.latency_history.append(delivery_time)
        if len(self.latency_history) > 1000:
            self.latency_history.pop(0)
        
        self.reliability_metrics[message.sender_id] = len(
            self.delivered_messages.get(message.message_id, set())
        ) / len(message.recipients)
    
    def _update_bandwidth(self,
                         message: SwarmMessage,
                         recipient: str):
        """Update bandwidth usage metrics."""
        message_size = len(str(message.payload))
        if recipient in self.bandwidth_usage:
            self.bandwidth_usage[recipient] += message_size
        else:
            self.bandwidth_usage[recipient] = message_size
    
    def _handle_error(self, error: str, agent_id: str):
        """Handle communication errors."""
        if agent_id in self.error_counters:
            self.error_counters[agent_id] += 1
        else:
            self.error_counters[agent_id] = 1
        
        if self.error_counters[agent_id] >= self.max_retries:
            self.channel_status[agent_id] = 0.0