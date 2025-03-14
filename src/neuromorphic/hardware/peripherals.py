"""
Simple sensor and actuator abstractions for neuromorphic hardware.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum


class SensorType(Enum):
    """Types of sensors supported by the system."""
    VISION = 0
    AUDIO = 1
    TACTILE = 2
    TEMPERATURE = 3
    ACCELERATION = 4
    GYROSCOPE = 5
    CUSTOM = 6


class ActuatorType(Enum):
    """Types of actuators supported by the system."""
    MOTOR = 0
    SERVO = 1
    LED = 2
    SPEAKER = 3
    VIBRATION = 4
    CUSTOM = 5


class Sensor(ABC):
    """Abstract base class for neuromorphic sensors."""
    
    def __init__(self, sensor_id: str, sensor_type: SensorType):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self._enabled = False
    
    @abstractmethod
    def read(self) -> np.ndarray:
        """Read data from the sensor."""
        pass
    
    def enable(self) -> bool:
        """Enable the sensor."""
        self._enabled = True
        return True
    
    def disable(self) -> bool:
        """Disable the sensor."""
        self._enabled = False
        return True
    
    def is_enabled(self) -> bool:
        """Check if the sensor is enabled."""
        return self._enabled


class Actuator(ABC):
    """Abstract base class for neuromorphic actuators."""
    
    def __init__(self, actuator_id: str, actuator_type: ActuatorType):
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self._enabled = False
    
    @abstractmethod
    def write(self, data: np.ndarray) -> bool:
        """Write data to the actuator."""
        pass
    
    def enable(self) -> bool:
        """Enable the actuator."""
        self._enabled = True
        return True
    
    def disable(self) -> bool:
        """Disable the actuator."""
        self._enabled = False
        return True
    
    def is_enabled(self) -> bool:
        """Check if the actuator is enabled."""
        return self._enabled


class DVSSensor(Sensor):
    """Dynamic Vision Sensor (event-based camera)."""
    
    def __init__(self, sensor_id: str, width: int = 128, height: int = 128):
        super().__init__(sensor_id, SensorType.VISION)
        self.width = width
        self.height = height
    
    def read(self) -> np.ndarray:
        """Read events from the DVS sensor."""
        if not self._enabled:
            return np.array([])
        
        # Simulate sparse events (x, y, polarity, timestamp)
        num_events = np.random.randint(10, 100)
        events = np.zeros((num_events, 4))
        events[:, 0] = np.random.randint(0, self.width, num_events)  # x
        events[:, 1] = np.random.randint(0, self.height, num_events)  # y
        events[:, 2] = np.random.choice([0, 1], num_events)  # polarity
        events[:, 3] = np.random.rand(num_events) * 0.1  # timestamp (in seconds)
        
        return events


class MotorActuator(Actuator):
    """Simple motor actuator."""
    
    def __init__(self, actuator_id: str, channels: int = 1):
        super().__init__(actuator_id, ActuatorType.MOTOR)
        self.channels = channels
        self.current_values = np.zeros(channels)
    
    def write(self, data: np.ndarray) -> bool:
        """Set motor values (-1.0 to 1.0 range)."""
        if not self._enabled:
            return False
        
        if len(data) != self.channels:
            return False
        
        # Clip values to valid range
        self.current_values = np.clip(data, -1.0, 1.0)
        return True
    
    def get_state(self) -> np.ndarray:
        """Get current motor values."""
        return self.current_values.copy()


class PeripheralManager:
    """Manager for sensors and actuators."""
    
    def __init__(self):
        self.sensors = {}
        self.actuators = {}
    
    def add_sensor(self, sensor: Sensor) -> None:
        """Register a sensor with the manager."""
        self.sensors[sensor.sensor_id] = sensor
    
    def add_actuator(self, actuator: Actuator) -> None:
        """Register an actuator with the manager."""
        self.actuators[actuator.actuator_id] = actuator
    
    def get_sensor(self, sensor_id: str) -> Optional[Sensor]:
        """Get a sensor by ID."""
        return self.sensors.get(sensor_id)
    
    def get_actuator(self, actuator_id: str) -> Optional[Actuator]:
        """Get an actuator by ID."""
        return self.actuators.get(actuator_id)
    
    def enable_all(self) -> None:
        """Enable all peripherals."""
        for sensor in self.sensors.values():
            sensor.enable()
        for actuator in self.actuators.values():
            actuator.enable()
    
    def disable_all(self) -> None:
        """Disable all peripherals."""
        for sensor in self.sensors.values():
            sensor.disable()
        for actuator in self.actuators.values():
            actuator.disable()