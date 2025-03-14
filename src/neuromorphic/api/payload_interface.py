"""
Standardized API for payload-to-neuromorphic controller communication.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from datetime import datetime


class PayloadType(Enum):
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    PROCESSOR = "processor"
    HYBRID = "hybrid"


class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class PayloadConfig:
    """Payload configuration parameters."""
    id: str
    type: PayloadType
    update_rate: float  # Hz
    priority: MessagePriority
    data_format: Dict[str, str]


class PayloadMessage:
    """Standard message format for payload communication."""
    
    def __init__(self, 
                 payload_id: str,
                 message_type: str,
                 data: Dict[str, Any],
                 priority: MessagePriority = MessagePriority.NORMAL):
        self.payload_id = payload_id
        self.message_type = message_type
        self.data = data
        self.priority = priority
        self.timestamp = datetime.now().timestamp()
        self.sequence_number = 0  # Set by communication handler
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            'payload_id': self.payload_id,
            'message_type': self.message_type,
            'data': self.data,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'sequence': self.sequence_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PayloadMessage':
        """Create message from dictionary."""
        msg = cls(
            payload_id=data['payload_id'],
            message_type=data['message_type'],
            data=data['data'],
            priority=MessagePriority(data['priority'])
        )
        msg.timestamp = data['timestamp']
        msg.sequence_number = data['sequence']
        return msg


class PayloadInterface:
    """Interface for payload-to-neuromorphic controller communication."""
    
    def __init__(self):
        self.payloads: Dict[str, PayloadConfig] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.sequence_counter = 0
        self.message_queue = asyncio.Queue()
        
        # Message validation schemas
        self.data_schemas: Dict[str, Dict] = {}
    
    def register_payload(self, config: PayloadConfig) -> bool:
        """Register a new payload with the interface."""
        if config.id in self.payloads:
            return False
        
        self.payloads[config.id] = config
        self.message_handlers[config.id] = []
        self._create_data_schema(config)
        return True
    
    def _create_data_schema(self, config: PayloadConfig):
        """Create data validation schema for payload."""
        self.data_schemas[config.id] = {
            'type': 'object',
            'properties': config.data_format,
            'required': list(config.data_format.keys())
        }
    
    async def send_message(self, message: PayloadMessage) -> bool:
        """Send message to neuromorphic controller."""
        if message.payload_id not in self.payloads:
            return False
        
        # Validate message format
        if not self._validate_message(message):
            return False
        
        # Add sequence number
        message.sequence_number = self._get_next_sequence()
        
        # Queue message based on priority
        await self.message_queue.put((message.priority.value, message))
        return True
    
    def _validate_message(self, message: PayloadMessage) -> bool:
        """Validate message format against schema."""
        if message.payload_id not in self.data_schemas:
            return False
            
        schema = self.data_schemas[message.payload_id]
        try:
            # Basic type validation
            for key, type_str in schema['properties'].items():
                if key not in message.data:
                    return False
                if not isinstance(message.data[key], eval(type_str)):
                    return False
            return True
        except:
            return False
    
    def _get_next_sequence(self) -> int:
        """Get next message sequence number."""
        self.sequence_counter += 1
        return self.sequence_counter
    
    def register_handler(self, 
                        payload_id: str, 
                        handler: Callable[[PayloadMessage], None]):
        """Register message handler for payload."""
        if payload_id in self.message_handlers:
            self.message_handlers[payload_id].append(handler)
    
    async def process_messages(self):
        """Process queued messages."""
        while True:
            priority, message = await self.message_queue.get()
            
            # Process message based on priority
            if message.payload_id in self.message_handlers:
                for handler in self.message_handlers[message.payload_id]:
                    try:
                        handler(message)
                    except Exception as e:
                        print(f"Error processing message: {e}")
            
            self.message_queue.task_done()
    
    def get_payload_status(self, payload_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of registered payload."""
        if payload_id not in self.payloads:
            return None
            
        config = self.payloads[payload_id]
        return {
            'id': payload_id,
            'type': config.type.value,
            'update_rate': config.update_rate,
            'priority': config.priority.value,
            'message_count': self.sequence_counter
        }


# Example usage
async def example_usage():
    # Create interface
    interface = PayloadInterface()
    
    # Register payload
    sensor_config = PayloadConfig(
        id="sensor1",
        type=PayloadType.SENSOR,
        update_rate=10.0,
        priority=MessagePriority.NORMAL,
        data_format={
            'temperature': 'float',
            'pressure': 'float',
            'status': 'str'
        }
    )
    interface.register_payload(sensor_config)
    
    # Create message handler
    def handle_sensor_data(message: PayloadMessage):
        print(f"Received sensor data: {message.data}")
    
    # Register handler
    interface.register_handler("sensor1", handle_sensor_data)
    
    # Send test message
    message = PayloadMessage(
        payload_id="sensor1",
        message_type="data",
        data={
            'temperature': 25.5,
            'pressure': 101.3,
            'status': 'nominal'
        }
    )
    
    await interface.send_message(message)
    
    # Start message processing
    await interface.process_messages()