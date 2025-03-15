"""
Messaging system for communication between neuromorphic and conventional components.
"""
import numpy as np
import time
import queue
from typing import Dict, Any, List, Callable, Optional, Union
from enum import Enum
import threading

class MessageType(Enum):
    """Types of messages that can be exchanged."""
    SENSOR_DATA = 1
    CONTROL_COMMAND = 2
    STATE_UPDATE = 3
    CONFIGURATION = 4
    STATUS = 5
    ALERT = 6

class Message:
    """Message container for inter-component communication."""
    
    def __init__(self, msg_type: MessageType, source: str, target: str, 
                data: Dict[str, Any], timestamp: Optional[float] = None):
        """
        Initialize message.
        
        Args:
            msg_type: Type of message
            source: Source component ID
            target: Target component ID
            data: Message payload
            timestamp: Message timestamp (defaults to current time)
        """
        self.msg_type = msg_type
        self.source = source
        self.target = target
        self.data = data
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.id = f"{source}_{target}_{self.timestamp}"
    
    def __str__(self) -> str:
        """String representation of message."""
        return f"Message({self.msg_type.name}, {self.source}->{self.target}, {len(self.data)} items)"

class NeuralBridge:
    """Bridge for communication between neuromorphic and conventional components."""
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize neural bridge.
        
        Args:
            buffer_size: Maximum number of messages in queue
        """
        self.buffer_size = buffer_size
        self.message_queues = {}
        self.subscribers = {}
        self.converters = {}
        self.lock = threading.RLock()
        
    def register_component(self, component_id: str):
        """
        Register a component with the messaging system.
        
        Args:
            component_id: Unique component identifier
        """
        with self.lock:
            if component_id not in self.message_queues:
                self.message_queues[component_id] = queue.Queue(self.buffer_size)
                self.subscribers[component_id] = []
    
    def register_converter(self, source_type: str, target_type: str, 
                          converter_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Register a data converter between different component types.
        
        Args:
            source_type: Source data type
            target_type: Target data type
            converter_func: Function to convert data from source to target format
        """
        key = f"{source_type}_{target_type}"
        self.converters[key] = converter_func
    
    def subscribe(self, subscriber_id: str, source_id: str, 
                 callback: Callable[[Message], None]):
        """
        Subscribe to messages from a specific source.
        
        Args:
            subscriber_id: ID of subscribing component
            source_id: ID of source component
            callback: Function to call when message is received
        """
        with self.lock:
            if source_id not in self.subscribers:
                self.register_component(source_id)
            
            self.subscribers[source_id].append((subscriber_id, callback))
    
    def send_message(self, message: Message):
        """
        Send a message to its target.
        
        Args:
            message: Message to send
        """
        with self.lock:
            # Ensure target exists
            if message.target not in self.message_queues:
                self.register_component(message.target)
            
            # Add to target's queue
            try:
                self.message_queues[message.target].put_nowait(message)
            except queue.Full:
                # If queue is full, remove oldest message and try again
                try:
                    self.message_queues[message.target].get_nowait()
                    self.message_queues[message.target].put_nowait(message)
                except (queue.Empty, queue.Full):
                    pass
            
            # Notify subscribers
            self._notify_subscribers(message)
    
    def _notify_subscribers(self, message: Message):
        """
        Notify subscribers about a new message.
        
        Args:
            message: Message that was sent
        """
        # Notify subscribers to the target
        for subscriber_id, callback in self.subscribers.get(message.target, []):
            try:
                callback(message)
            except Exception as e:
                print(f"Error in subscriber callback {subscriber_id}: {e}")
    
    def receive_message(self, component_id: str, block: bool = False, 
                       timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive a message for a component.
        
        Args:
            component_id: ID of receiving component
            block: Whether to block until a message is available
            timeout: Maximum time to wait for a message
            
        Returns:
            Message or None if no message is available
        """
        with self.lock:
            if component_id not in self.message_queues:
                return None
        
        try:
            return self.message_queues[component_id].get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def convert_data(self, data: Dict[str, Any], source_type: str, 
                    target_type: str) -> Dict[str, Any]:
        """
        Convert data between different formats.
        
        Args:
            data: Data to convert
            source_type: Source data type
            target_type: Target data type
            
        Returns:
            Converted data
        """
        key = f"{source_type}_{target_type}"
        if key in self.converters:
            return self.converters[key](data)
        return data  # Return unchanged if no converter is registered


class SpikeEncoder:
    """Encodes conventional data to spike trains for neuromorphic processing."""
    
    def __init__(self, input_size: int, encoding_method: str = "rate"):
        """
        Initialize spike encoder.
        
        Args:
            input_size: Number of input values to encode
            encoding_method: Encoding method ('rate', 'temporal', or 'population')
        """
        self.input_size = input_size
        self.encoding_method = encoding_method
        self.scale_factors = np.ones(input_size)
        self.thresholds = np.zeros(input_size)
    
    def configure(self, scale_factors: np.ndarray, thresholds: np.ndarray):
        """
        Configure encoder parameters.
        
        Args:
            scale_factors: Scaling factors for input values
            thresholds: Threshold values for spike generation
        """
        if len(scale_factors) != self.input_size or len(thresholds) != self.input_size:
            raise ValueError("Parameter dimensions must match input size")
        
        self.scale_factors = scale_factors
        self.thresholds = thresholds
    
    def encode(self, values: np.ndarray, duration_ms: int = 100, 
              dt_ms: int = 1) -> np.ndarray:
        """
        Encode values as spike trains.
        
        Args:
            values: Input values to encode
            duration_ms: Duration of spike train in milliseconds
            dt_ms: Time step in milliseconds
            
        Returns:
            Spike train array (input_size x timesteps)
        """
        if len(values) != self.input_size:
            raise ValueError(f"Expected {self.input_size} values, got {len(values)}")
        
        timesteps = duration_ms // dt_ms
        spikes = np.zeros((self.input_size, timesteps), dtype=np.uint8)
        
        if self.encoding_method == "rate":
            # Rate coding: spike frequency proportional to value
            for i in range(self.input_size):
                rate = (values[i] - self.thresholds[i]) * self.scale_factors[i]
                rate = max(0, min(1, rate))  # Clamp to [0, 1]
                
                # Generate spikes with probability proportional to rate
                for t in range(timesteps):
                    if np.random.random() < rate:
                        spikes[i, t] = 1
                        
        elif self.encoding_method == "temporal":
            # Temporal coding: spike timing encodes value
            for i in range(self.input_size):
                value = (values[i] - self.thresholds[i]) * self.scale_factors[i]
                value = max(0, min(1, value))  # Clamp to [0, 1]
                
                # Spike time proportional to value (earlier = higher)
                spike_time = int((1 - value) * timesteps)
                if 0 <= spike_time < timesteps:
                    spikes[i, spike_time] = 1
                    
        elif self.encoding_method == "population":
            # Population coding: multiple neurons represent one value
            neurons_per_input = 10
            expanded_spikes = np.zeros((self.input_size * neurons_per_input, timesteps), dtype=np.uint8)
            
            for i in range(self.input_size):
                value = (values[i] - self.thresholds[i]) * self.scale_factors[i]
                value = max(0, min(1, value))  # Clamp to [0, 1]
                
                # Each neuron has a preferred value
                for n in range(neurons_per_input):
                    preferred = n / (neurons_per_input - 1)
                    # Gaussian tuning curve
                    response = np.exp(-((value - preferred) ** 2) / 0.05)
                    
                    # Generate spikes based on response
                    for t in range(timesteps):
                        if np.random.random() < response:
                            expanded_spikes[i * neurons_per_input + n, t] = 1
            
            spikes = expanded_spikes
        
        return spikes


class SpikeDecoder:
    """Decodes spike trains from neuromorphic components to conventional data."""
    
    def __init__(self, output_size: int, decoding_method: str = "rate"):
        """
        Initialize spike decoder.
        
        Args:
            output_size: Number of output values to decode
            decoding_method: Decoding method ('rate', 'temporal', or 'first_spike')
        """
        self.output_size = output_size
        self.decoding_method = decoding_method
        self.scale_factors = np.ones(output_size)
        self.offsets = np.zeros(output_size)
    
    def configure(self, scale_factors: np.ndarray, offsets: np.ndarray):
        """
        Configure decoder parameters.
        
        Args:
            scale_factors: Scaling factors for output values
            offsets: Offset values for output generation
        """
        if len(scale_factors) != self.output_size or len(offsets) != self.output_size:
            raise ValueError("Parameter dimensions must match output size")
        
        self.scale_factors = scale_factors
        self.offsets = offsets
    
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """
        Decode spike trains to values.
        
        Args:
            spikes: Spike train array (neurons x timesteps)
            
        Returns:
            Decoded values
        """
        if spikes.shape[0] < self.output_size:
            raise ValueError(f"Expected at least {self.output_size} neurons, got {spikes.shape[0]}")
        
        values = np.zeros(self.output_size)
        
        if self.decoding_method == "rate":
            # Rate coding: value proportional to spike frequency
            for i in range(self.output_size):
                rate = np.mean(spikes[i])
                values[i] = rate * self.scale_factors[i] + self.offsets[i]
                
        elif self.decoding_method == "temporal":
            # Temporal coding: value encoded in spike timing
            for i in range(self.output_size):
                spike_indices = np.where(spikes[i] > 0)[0]
                if len(spike_indices) > 0:
                    # First spike time determines value
                    first_spike = spike_indices[0]
                    # Earlier spikes = higher values
                    value = 1.0 - (first_spike / spikes.shape[1])
                    values[i] = value * self.scale_factors[i] + self.offsets[i]
                    
        elif self.decoding_method == "first_spike":
            # First spike coding: only first neuron to spike per group is considered
            timesteps = spikes.shape[1]
            for i in range(self.output_size):
                # Find first spike for each neuron
                first_spikes = np.ones(self.output_size) * timesteps
                for n in range(self.output_size):
                    spike_indices = np.where(spikes[n] > 0)[0]
                    if len(spike_indices) > 0:
                        first_spikes[n] = spike_indices[0]
                
                # Neuron with earliest spike determines value
                if np.min(first_spikes) < timesteps:
                    winner = np.argmin(first_spikes)
                    values[i] = winner / (self.output_size - 1)
                    values[i] = values[i] * self.scale_factors[i] + self.offsets[i]
        
        return values