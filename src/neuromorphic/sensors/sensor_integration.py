"""
Basic sensor integration models for neuromorphic hardware.

This module implements simplified sensor integration and fusion techniques,
inspired by how biological systems process and combine sensory information.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class SensorPreprocessor:
    """Simple implementation of sensor preprocessing."""
    
    def __init__(self, smoothing_factor: float = 0.8):
        """
        Initialize sensor preprocessor.
        
        Args:
            smoothing_factor: Weight for exponential smoothing (0-1)
        """
        self.smoothing_factor = smoothing_factor
        self.last_value = None
        self.spike_threshold = 3.0  # Threshold for spike detection
    
    def process(self, raw_value: float) -> Tuple[float, bool]:
        """
        Process raw sensor value.
        
        Args:
            raw_value: Raw sensor reading
            
        Returns:
            Tuple of (processed_value, is_spike)
        """
        # Initialize last value if needed
        if self.last_value is None:
            self.last_value = raw_value
            return raw_value, False
        
        # Check for spikes
        is_spike = abs(raw_value - self.last_value) > self.spike_threshold
        
        # Apply exponential smoothing
        smoothed = self.smoothing_factor * self.last_value + (1 - self.smoothing_factor) * raw_value
        
        # Update last value
        self.last_value = smoothed
        
        return smoothed, is_spike


class SensorFusion:
    """Simple implementation of sensor fusion."""
    
    def __init__(self, sensor_weights: Optional[Dict[str, float]] = None):
        """
        Initialize sensor fusion.
        
        Args:
            sensor_weights: Dictionary mapping sensor names to weights
        """
        self.sensor_weights = sensor_weights or {}
        self.preprocessors = {}
    
    def add_sensor(self, sensor_name: str, weight: float = 1.0):
        """
        Add a sensor to the fusion system.
        
        Args:
            sensor_name: Name of the sensor
            weight: Weight for this sensor in fusion
        """
        self.sensor_weights[sensor_name] = weight
        self.preprocessors[sensor_name] = SensorPreprocessor()
    
    def fuse(self, sensor_readings: Dict[str, float]) -> Dict[str, float]:
        """
        Fuse multiple sensor readings.
        
        Args:
            sensor_readings: Dictionary mapping sensor names to values
            
        Returns:
            Dictionary of fused values
        """
        # Process each sensor reading
        processed_readings = {}
        spike_detected = False
        
        for sensor_name, value in sensor_readings.items():
            # Create preprocessor if needed
            if sensor_name not in self.preprocessors:
                self.add_sensor(sensor_name)
            
            # Process reading
            processed, is_spike = self.preprocessors[sensor_name].process(value)
            processed_readings[sensor_name] = processed
            
            if is_spike:
                spike_detected = True
        
        # Fuse orientation sensors if available
        fused_values = {}
        
        # Handle orientation fusion
        orientation_sensors = ['roll', 'pitch', 'yaw']
        if all(sensor in processed_readings for sensor in orientation_sensors):
            orientation = np.array([
                processed_readings['roll'],
                processed_readings['pitch'],
                processed_readings['yaw']
            ])
            fused_values['orientation'] = orientation
        
        # Handle position fusion
        position_sensors = ['x', 'y', 'z']
        if all(sensor in processed_readings for sensor in position_sensors):
            position = np.array([
                processed_readings['x'],
                processed_readings['y'],
                processed_readings['z']
            ])
            fused_values['position'] = position
        
        # Add spike detection flag
        fused_values['spike_detected'] = spike_detected
        
        return fused_values


class SensorIntegration:
    """
    Sensor integration system.
    
    This system preprocesses and fuses sensor data for use by controllers.
    """
    
    def __init__(self):
        """Initialize the sensor integration system."""
        # Create sensor fusion module
        self.fusion = SensorFusion()
        
        # Add default sensors
        self.fusion.add_sensor('roll', weight=1.0)
        self.fusion.add_sensor('pitch', weight=1.0)
        self.fusion.add_sensor('yaw', weight=1.0)
        self.fusion.add_sensor('x', weight=1.0)
        self.fusion.add_sensor('y', weight=1.0)
        self.fusion.add_sensor('z', weight=1.0)
        
        # Sensor health status
        self.sensor_health = {}
    
    def process_sensors(self, raw_readings: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Process and integrate sensor readings.
        
        Args:
            raw_readings: Dictionary of raw sensor readings
            
        Returns:
            Dictionary of processed state variables
        """
        # Update sensor health
        for sensor_name in raw_readings:
            if sensor_name not in self.sensor_health:
                self.sensor_health[sensor_name] = 1.0
        
        # Fuse sensor readings
        fused_values = self.fusion.fuse(raw_readings)
        
        # Create state dictionary for controllers
        state = {}
        
        if 'orientation' in fused_values:
            state['orientation'] = fused_values['orientation']
        
        if 'position' in fused_values:
            state['position'] = fused_values['position']
        
        # If we detected a spike, reduce confidence in affected sensors
        if fused_values.get('spike_detected', False):
            for sensor_name in self.sensor_health:
                if sensor_name in raw_readings:
                    self.sensor_health[sensor_name] *= 0.9
        
        # Add sensor health to state
        state['sensor_health'] = self.sensor_health
        
        return state


def create_sensor_integration() -> SensorIntegration:
    """
    Create a sensor integration system.
    
    Returns:
        Configured sensor integration system
    """
    return SensorIntegration()