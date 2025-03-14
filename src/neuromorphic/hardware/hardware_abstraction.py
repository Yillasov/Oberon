"""
Hardware Abstraction Layer for neuromorphic control systems.

This module provides a simple abstraction layer to interface between
neuromorphic controllers and various hardware platforms.
"""
import numpy as np
from typing import Dict, List, Callable, Any, Optional
import time


class SensorInterface:
    """Interface for hardware sensors."""
    
    def __init__(self, sensor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize sensor interface.
        
        Args:
            sensor_config: Configuration dictionary for sensors
        """
        self.config = sensor_config or {}
        self.sensor_scaling = {
            'accelerometer': 1.0,
            'gyroscope': 1.0,
            'magnetometer': 1.0,
            'barometer': 1.0,
            'gps': 1.0
        }
        
        # Apply configuration
        if 'scaling' in self.config:
            self.sensor_scaling.update(self.config['scaling'])
        
        # Simulated sensor noise levels
        self.noise_levels = {
            'accelerometer': 0.05,
            'gyroscope': 0.02,
            'magnetometer': 0.1,
            'barometer': 0.2,
            'gps': 0.5
        }
    
    def read_sensors(self) -> Dict[str, np.ndarray]:
        """
        Read all available sensors.
        
        Returns:
            Dictionary of sensor readings
        """
        # In a real implementation, this would interface with hardware
        # For now, return simulated values
        readings = {
            'accelerometer': self._simulate_accelerometer(),
            'gyroscope': self._simulate_gyroscope(),
            'barometer': self._simulate_barometer(),
            'gps': self._simulate_gps()
        }
        
        return readings
    
    def _add_noise(self, base_value: float, sensor_type: str) -> float:
        """Add realistic noise to sensor reading."""
        noise_level = self.noise_levels.get(sensor_type, 0.1)
        return base_value + np.random.normal(0, noise_level)
    
    def _simulate_accelerometer(self) -> np.ndarray:
        """Simulate accelerometer readings."""
        # Simulate gravity + small random accelerations
        ax = self._add_noise(0.0, 'accelerometer')
        ay = self._add_noise(0.0, 'accelerometer')
        az = self._add_noise(-9.81, 'accelerometer')  # Gravity
        
        return np.array([ax, ay, az]) * self.sensor_scaling['accelerometer']
    
    def _simulate_gyroscope(self) -> np.ndarray:
        """Simulate gyroscope readings."""
        # Small random angular velocities
        wx = self._add_noise(0.0, 'gyroscope')
        wy = self._add_noise(0.0, 'gyroscope')
        wz = self._add_noise(0.0, 'gyroscope')
        
        return np.array([wx, wy, wz]) * self.sensor_scaling['gyroscope']
    
    def _simulate_barometer(self) -> np.ndarray:
        """Simulate barometer readings."""
        # Simulate pressure at some altitude
        pressure = self._add_noise(101325.0, 'barometer')  # Standard pressure (Pa)
        temperature = self._add_noise(25.0, 'barometer')  # Temperature (C)
        
        return np.array([pressure, temperature]) * self.sensor_scaling['barometer']
    
    def _simulate_gps(self) -> np.ndarray:
        """Simulate GPS readings."""
        # Simulate position
        latitude = self._add_noise(37.7749, 'gps')  # Example: San Francisco
        longitude = self._add_noise(-122.4194, 'gps')
        altitude = self._add_noise(10.0, 'gps')
        
        return np.array([latitude, longitude, altitude]) * self.sensor_scaling['gps']


class ActuatorInterface:
    """Interface for hardware actuators."""
    
    def __init__(self, actuator_config: Optional[Dict[str, Any]] = None):
        """
        Initialize actuator interface.
        
        Args:
            actuator_config: Configuration dictionary for actuators
        """
        self.config = actuator_config or {}
        self.actuator_scaling = {
            'motors': 1.0,
            'servos': 1.0
        }
        
        # Apply configuration
        if 'scaling' in self.config:
            self.actuator_scaling.update(self.config['scaling'])
        
        # Last command sent to actuators
        self.last_command = {}
        
        # Simulated actuator response delay (ms)
        self.response_delay = 20
    
    def send_commands(self, commands: Dict[str, float]) -> bool:
        """
        Send commands to actuators.
        
        Args:
            commands: Dictionary of actuator commands
            
        Returns:
            Success status
        """
        # In a real implementation, this would interface with hardware
        # For now, just simulate a delay and store the commands
        time.sleep(self.response_delay / 1000.0)
        
        # Apply scaling
        scaled_commands = {}
        for name, value in commands.items():
            if 'motor' in name:
                scaled_commands[name] = value * self.actuator_scaling['motors']
            elif 'servo' in name:
                scaled_commands[name] = value * self.actuator_scaling['servos']
            else:
                scaled_commands[name] = value
        
        # Store commands
        self.last_command = scaled_commands
        
        return True
    
    def get_actuator_status(self) -> Dict[str, Any]:
        """
        Get status of actuators.
        
        Returns:
            Dictionary of actuator status
        """
        # In a real implementation, this would read from hardware
        # For now, just return the last command with simulated feedback
        status = {
            'commands': self.last_command,
            'temperatures': {
                'motor1': 35.0 + np.random.normal(0, 2.0),
                'motor2': 36.0 + np.random.normal(0, 2.0),
                'motor3': 34.0 + np.random.normal(0, 2.0),
                'motor4': 35.5 + np.random.normal(0, 2.0)
            },
            'voltages': {
                'main': 11.8 + np.random.normal(0, 0.1),
                'logic': 5.0 + np.random.normal(0, 0.05)
            }
        }
        
        return status


class HardwareAbstractionLayer:
    """
    Hardware Abstraction Layer for neuromorphic control systems.
    
    This layer provides a unified interface to sensors and actuators
    across different hardware platforms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HAL.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Create interfaces
        self.sensors = SensorInterface(self.config.get('sensors'))
        self.actuators = ActuatorInterface(self.config.get('actuators'))
        
        # State conversion functions
        self.sensor_to_state = self._default_sensor_to_state
        self.control_to_actuator = self._default_control_to_actuator
        
        # Override with custom functions if provided
        if 'sensor_to_state_func' in self.config:
            self.sensor_to_state = self.config['sensor_to_state_func']
        if 'control_to_actuator_func' in self.config:
            self.control_to_actuator = self.config['control_to_actuator_func']
    
    def _default_sensor_to_state(self, sensor_readings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Default conversion from sensor readings to controller state.
        
        Args:
            sensor_readings: Raw sensor readings
            
        Returns:
            State dictionary for controllers
        """
        state = {}
        
        # Extract orientation from gyroscope
        if 'gyroscope' in sensor_readings:
            state['orientation'] = sensor_readings['gyroscope']
        
        # Extract position from GPS
        if 'gps' in sensor_readings:
            state['position'] = sensor_readings['gps']
        
        # Extract acceleration
        if 'accelerometer' in sensor_readings:
            state['acceleration'] = sensor_readings['accelerometer']
        
        return state
    
    def _default_control_to_actuator(self, control: Dict[str, float]) -> Dict[str, float]:
        """
        Default conversion from control outputs to actuator commands.
        
        Args:
            control: Control outputs from controller
            
        Returns:
            Actuator commands
        """
        commands = {}
        
        # Map control outputs to motor commands for a quadcopter
        if 'aileron' in control:
            # Roll control affects motor differential
            commands['motor1'] = 0.5 + control.get('throttle', 0.5) - control['aileron']
            commands['motor2'] = 0.5 + control.get('throttle', 0.5) + control['aileron']
            commands['motor3'] = 0.5 + control.get('throttle', 0.5) - control['aileron']
            commands['motor4'] = 0.5 + control.get('throttle', 0.5) + control['aileron']
        
        if 'elevator' in control:
            # Pitch control affects front/back differential
            commands['motor1'] = commands.get('motor1', 0.5) - control['elevator']
            commands['motor2'] = commands.get('motor2', 0.5) - control['elevator']
            commands['motor3'] = commands.get('motor3', 0.5) + control['elevator']
            commands['motor4'] = commands.get('motor4', 0.5) + control['elevator']
        
        if 'rudder' in control:
            # Yaw control affects diagonal differential
            commands['motor1'] = commands.get('motor1', 0.5) - control['rudder']
            commands['motor2'] = commands.get('motor2', 0.5) + control['rudder']
            commands['motor3'] = commands.get('motor3', 0.5) + control['rudder']
            commands['motor4'] = commands.get('motor4', 0.5) - control['rudder']
        
        # Ensure all commands are within bounds
        for key in commands:
            commands[key] = max(0.0, min(1.0, commands[key]))
        
        return commands
    
    def read_state(self) -> Dict[str, np.ndarray]:
        """
        Read sensors and convert to controller state.
        
        Returns:
            State dictionary for controllers
        """
        # Read raw sensor data
        sensor_readings = self.sensors.read_sensors()
        
        # Convert to controller state
        state = self.sensor_to_state(sensor_readings)
        
        return state
    
    def write_control(self, control: Dict[str, float]) -> bool:
        """
        Convert control outputs to actuator commands and send.
        
        Args:
            control: Control outputs from controller
            
        Returns:
            Success status
        """
        # Convert to actuator commands
        commands = self.control_to_actuator(control)
        
        # Send to actuators
        success = self.actuators.send_commands(commands)
        
        return success


def create_hal(config: Optional[Dict[str, Any]] = None) -> HardwareAbstractionLayer:
    """
    Create a hardware abstraction layer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured HAL instance
    """
    return HardwareAbstractionLayer(config)