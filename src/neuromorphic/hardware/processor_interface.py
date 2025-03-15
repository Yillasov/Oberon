"""
Interface definitions for neuromorphic processors.

This module provides abstract base classes and interfaces for interacting
with different neuromorphic hardware implementations.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union


class NeuromorphicProcessor(ABC):
    """
    Abstract base class for neuromorphic processors.
    
    This interface defines the common methods that all neuromorphic
    processor implementations should provide.
    """
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the processor with the given parameters.
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            True if configuration was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_weights(self, weights: np.ndarray) -> bool:
        """
        Load synaptic weights into the processor.
        
        Args:
            weights: Weight matrix to load
            
        Returns:
            True if weights were loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def run(self, input_spikes: np.ndarray, duration: float) -> np.ndarray:
        """
        Run the processor for the specified duration with the given input.
        
        Args:
            input_spikes: Input spike pattern
            duration: Duration to run in milliseconds
            
        Returns:
            Output spike pattern
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the processor state."""
        pass
    
    @abstractmethod
    def get_power_usage(self) -> float:
        """
        Get the estimated power usage of the processor.
        
        Returns:
            Power usage in milliwatts
        """
        pass
    
    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get information about the hardware platform.
        
        Returns:
            Dictionary containing hardware specifications and capabilities
        """
        pass
    
    @abstractmethod
    def set_neuron_parameters(self, neuron_ids: List[int], parameters: Dict[str, Any]) -> bool:
        """
        Set parameters for specific neurons.
        
        Args:
            neuron_ids: List of neuron IDs to configure
            parameters: Dictionary of parameter names and values
            
        Returns:
            True if parameters were set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def set_synapse_parameters(self, pre_ids: List[int], post_ids: List[int], 
                              parameters: Dict[str, Any]) -> bool:
        """
        Set parameters for specific synapses.
        
        Args:
            pre_ids: List of presynaptic neuron IDs
            post_ids: List of postsynaptic neuron IDs
            parameters: Dictionary of parameter names and values
            
        Returns:
            True if parameters were set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_neuron_states(self, neuron_ids: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Get the current state of neurons.
        
        Args:
            neuron_ids: Optional list of neuron IDs to query (all neurons if None)
            
        Returns:
            Dictionary mapping state variables to their values
        """
        pass
    
    @abstractmethod
    def load_network(self, network_definition: Dict[str, Any]) -> bool:
        """
        Load a complete network definition.
        
        Args:
            network_definition: Dictionary containing the network structure and parameters
            
        Returns:
            True if network was loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def save_network(self) -> Dict[str, Any]:
        """
        Save the current network configuration.
        
        Returns:
            Dictionary containing the network structure and parameters
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the hardware.
        
        Returns:
            True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the hardware."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the processor is connected to hardware.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def run_batch(self, input_batch: List[np.ndarray], durations: List[float]) -> List[np.ndarray]:
        """
        Run multiple inputs in batch mode.
        
        Args:
            input_batch: List of input spike patterns
            durations: List of durations to run each input
            
        Returns:
            List of output spike patterns
        """
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from the processor.
        
        Returns:
            Dictionary containing performance metrics
        """
        pass
    
    @abstractmethod
    def supports_feature(self, feature_name: str) -> bool:
        """
        Check if the processor supports a specific feature.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if the feature is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_neuron_models(self) -> List[str]:
        """
        Get the list of supported neuron models.
        
        Returns:
            List of supported neuron model names
        """
        pass
    
    @abstractmethod
    def get_supported_learning_rules(self) -> List[str]:
        """
        Get the list of supported learning rules.
        
        Returns:
            List of supported learning rule names
        """
        pass
    
    @abstractmethod
    def apply_learning_rule(self, rule_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Apply a learning rule to update weights.
        
        Args:
            rule_name: Name of the learning rule to apply
            parameters: Parameters for the learning rule
            
        Returns:
            True if learning rule was applied successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def set_spike_recording(self, neuron_ids: List[int], enable: bool = True) -> bool:
        """
        Enable or disable spike recording for specific neurons.
        
        Args:
            neuron_ids: List of neuron IDs to configure
            enable: True to enable recording, False to disable
            
        Returns:
            True if recording configuration was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_recorded_spikes(self, neuron_ids: Optional[List[int]] = None) -> Dict[int, List[float]]:
        """
        Get recorded spikes from specified neurons.
        
        Args:
            neuron_ids: Optional list of neuron IDs to query (all recorded neurons if None)
            
        Returns:
            Dictionary mapping neuron IDs to lists of spike times
        """
        pass


class SpikeEncoder(ABC):
    """
    Interface for encoding data into spike patterns.
    """
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data into spike patterns.
        
        Args:
            data: Input data to encode
            
        Returns:
            Encoded spike pattern
        """
        pass


class SpikeDecoder(ABC):
    """
    Interface for decoding spike patterns into data.
    """
    
    @abstractmethod
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """
        Decode spike patterns into data.
        
        Args:
            spikes: Spike pattern to decode
            
        Returns:
            Decoded data
        """
        pass


class RateEncoder(SpikeEncoder):
    """
    Simple rate-based spike encoder.
    """
    
    def __init__(self, threshold: float = 0.5, max_rate: float = 100.0, duration: float = 100.0):
        """
        Initialize the rate encoder.
        
        Args:
            threshold: Threshold for generating spikes
            max_rate: Maximum firing rate in Hz
            duration: Duration of the encoding window in milliseconds
        """
        self.threshold = threshold
        self.max_rate = max_rate
        self.duration = duration
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data as firing rates.
        
        Args:
            data: Input data to encode (values should be between 0 and 1)
            
        Returns:
            Binary spike pattern
        """
        # Normalize data to [0, 1]
        normalized = np.clip(data, 0, 1)
        
        # Calculate firing probabilities
        firing_probs = normalized * self.max_rate * (self.duration / 1000.0)
        
        # Generate spikes
        spikes = np.random.rand(*data.shape) < firing_probs
        
        return spikes


class RateDecoder(SpikeDecoder):
    """
    Simple rate-based spike decoder.
    """
    
    def __init__(self, max_rate: float = 100.0, duration: float = 100.0):
        """
        Initialize the rate decoder.
        
        Args:
            max_rate: Maximum firing rate in Hz
            duration: Duration of the decoding window in milliseconds
        """
        self.max_rate = max_rate
        self.duration = duration
    
    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """
        Decode spike patterns into firing rates.
        
        Args:
            spikes: Binary spike pattern
            
        Returns:
            Decoded data (values between 0 and 1)
        """
        # Count spikes
        spike_count = np.sum(spikes, axis=0) if len(spikes.shape) > 1 else spikes
        
        # Convert to rates
        rates = spike_count / (self.max_rate * (self.duration / 1000.0))
        
        # Clip to [0, 1]
        return np.clip(rates, 0, 1)


class ProcessorManager:
    """
    Manager class for handling multiple neuromorphic processors.
    """
    
    def __init__(self):
        """Initialize the processor manager."""
        self.processors = {}
    
    def register_processor(self, name: str, processor: NeuromorphicProcessor) -> None:
        """
        Register a processor with the manager.
        
        Args:
            name: Name to associate with the processor
            processor: Processor instance
        """
        self.processors[name] = processor
    
    def get_processor(self, name: str) -> Optional[NeuromorphicProcessor]:
        """
        Get a processor by name.
        
        Args:
            name: Name of the processor
            
        Returns:
            Processor instance or None if not found
        """
        return self.processors.get(name)
    
    def list_processors(self) -> List[str]:
        """
        Get a list of registered processors.
        
        Returns:
            List of processor names
        """
        return list(self.processors.keys())