"""
UCAV component model for representing all subsystems.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

class ComponentType(Enum):
    """Types of UCAV components."""
    SENSOR = 1
    ACTUATOR = 2
    PROPULSION = 3
    POWER = 4
    NAVIGATION = 5
    COMMUNICATION = 6
    PAYLOAD = 7
    STRUCTURE = 8
    CONTROL = 9

class Component:
    """Base class for all UCAV components."""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        """
        Initialize component.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component
        """
        self.component_id = component_id
        self.component_type = component_type
        self.state = {}
        self.inputs = {}
        self.outputs = {}
        self.connected_components = []
        self.health = 1.0  # Health factor (0-1)
        self.power_consumption = 0.0  # Watts
        self.weight = 0.0  # kg
        self.enabled = True
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update component state.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Updated outputs
        """
        # Base implementation does nothing
        return self.outputs
    
    def set_inputs(self, inputs: Dict[str, Any]):
        """
        Set component inputs.
        
        Args:
            inputs: Dictionary of input values
        """
        self.inputs.update(inputs)
    
    def connect(self, component: 'Component'):
        """
        Connect to another component.
        
        Args:
            component: Component to connect to
        """
        if component not in self.connected_components:
            self.connected_components.append(component)
    
    def set_health(self, health: float):
        """
        Set component health.
        
        Args:
            health: Health factor (0-1)
        """
        self.health = max(0.0, min(1.0, health))
    
    def disable(self):
        """Disable the component."""
        self.enabled = False
    
    def enable(self):
        """Enable the component."""
        self.enabled = True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.
        
        Returns:
            Dictionary with component status
        """
        return {
            "id": self.component_id,
            "type": self.component_type.name,
            "health": self.health,
            "enabled": self.enabled,
            "power": self.power_consumption,
            "state": self.state
        }


class Sensor(Component):
    """Base class for sensor components."""
    
    def __init__(self, component_id: str, update_rate: float = 10.0):
        """
        Initialize sensor.
        
        Args:
            component_id: Unique identifier for the component
            update_rate: Sensor update rate in Hz
        """
        super().__init__(component_id, ComponentType.SENSOR)
        self.update_rate = update_rate
        self.last_update_time = 0.0
        self.noise_factor = 0.01
        self.outputs = {"value": 0.0, "timestamp": 0.0}
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update sensor reading.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Sensor outputs
        """
        self.last_update_time += dt
        
        # Only update at specified rate
        if self.last_update_time >= 1.0 / self.update_rate:
            self.last_update_time = 0.0
            
            # Add noise based on health
            noise = np.random.normal(0, self.noise_factor * (2.0 - self.health))
            
            # Update sensor value (to be implemented by subclasses)
            self._read_sensor_value(noise)
            
            # Update timestamp
            self.outputs["timestamp"] += dt
        
        return self.outputs
    
    def _read_sensor_value(self, noise: float):
        """
        Read sensor value (to be implemented by subclasses).
        
        Args:
            noise: Noise value to add
        """
        pass


class Actuator(Component):
    """Base class for actuator components."""
    
    def __init__(self, component_id: str, response_time: float = 0.1):
        """
        Initialize actuator.
        
        Args:
            component_id: Unique identifier for the component
            response_time: Actuator response time in seconds
        """
        super().__init__(component_id, ComponentType.ACTUATOR)
        self.response_time = response_time
        self.target_position = 0.0
        self.current_position = 0.0
        self.inputs = {"command": 0.0}
        self.outputs = {"position": 0.0, "velocity": 0.0}
        self.max_velocity = 1.0  # Units per second
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update actuator position.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Actuator outputs
        """
        if not self.enabled:
            return self.outputs
            
        # Get command input
        self.target_position = self.inputs.get("command", self.target_position)
        
        # Calculate position error
        error = self.target_position - self.current_position
        
        # Calculate velocity based on response time and health
        velocity = error / self.response_time * self.health
        
        # Limit velocity
        velocity = max(-self.max_velocity, min(velocity, self.max_velocity))
        
        # Update position
        self.current_position += velocity * dt
        
        # Update outputs
        self.outputs["position"] = self.current_position
        self.outputs["velocity"] = velocity
        
        return self.outputs


class PropulsionSystem(Component):
    """Propulsion system component."""
    
    def __init__(self, component_id: str, max_thrust: float = 1000.0):
        """
        Initialize propulsion system.
        
        Args:
            component_id: Unique identifier for the component
            max_thrust: Maximum thrust in Newtons
        """
        super().__init__(component_id, ComponentType.PROPULSION)
        self.max_thrust = max_thrust
        self.current_thrust = 0.0
        self.throttle = 0.0
        self.fuel_consumption_rate = 0.1  # kg/s at max throttle
        self.fuel_remaining = 100.0  # kg
        self.inputs = {"throttle": 0.0}
        self.outputs = {
            "thrust": 0.0,
            "fuel_remaining": self.fuel_remaining,
            "temperature": 20.0
        }
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update propulsion system.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Propulsion system outputs
        """
        if not self.enabled or self.fuel_remaining <= 0:
            self.current_thrust = 0.0
            self.outputs["thrust"] = 0.0
            return self.outputs
            
        # Get throttle input
        self.throttle = self.inputs.get("throttle", self.throttle)
        self.throttle = max(0.0, min(1.0, self.throttle))
        
        # Calculate thrust based on throttle and health
        self.current_thrust = self.throttle * self.max_thrust * self.health
        
        # Calculate fuel consumption
        fuel_consumption = self.throttle * self.fuel_consumption_rate * dt
        self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_consumption)
        
        # Calculate temperature (simplified model)
        temperature = 20.0 + 80.0 * self.throttle
        
        # Update outputs
        self.outputs["thrust"] = self.current_thrust
        self.outputs["fuel_remaining"] = self.fuel_remaining
        self.outputs["temperature"] = temperature
        
        # Update power consumption
        self.power_consumption = 10.0 + 90.0 * self.throttle
        
        return self.outputs


class NavigationSystem(Component):
    """Navigation system component."""
    
    def __init__(self, component_id: str):
        """
        Initialize navigation system.
        
        Args:
            component_id: Unique identifier for the component
        """
        super().__init__(component_id, ComponentType.NAVIGATION)
        self.position = np.zeros(3)  # x, y, z
        self.velocity = np.zeros(3)  # vx, vy, vz
        self.attitude = np.zeros(3)  # roll, pitch, yaw
        self.inputs = {
            "gps": np.zeros(3),
            "imu": np.zeros(6)  # 3 accel, 3 gyro
        }
        self.outputs = {
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "attitude": np.zeros(3),
            "waypoint_distance": 0.0
        }
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.power_consumption = 15.0
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update navigation system.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Navigation system outputs
        """
        if not self.enabled:
            return self.outputs
            
        # Get sensor inputs
        gps = self.inputs.get("gps", np.zeros(3))
        imu = self.inputs.get("imu", np.zeros(6))
        
        # Simple sensor fusion (in reality, would use Kalman filter)
        # Update position and velocity based on GPS
        self.position = gps
        
        # Update attitude based on IMU
        accel = imu[:3]
        gyro = imu[3:]
        self.attitude += gyro * dt
        
        # Calculate distance to current waypoint
        waypoint_distance = 0.0
        if self.waypoints and self.current_waypoint_idx < len(self.waypoints):
            current_waypoint = self.waypoints[self.current_waypoint_idx]
            waypoint_distance = np.linalg.norm(self.position - current_waypoint)
            
            # Check if waypoint reached
            if waypoint_distance < 10.0:  # 10m threshold
                self.current_waypoint_idx = min(self.current_waypoint_idx + 1, len(self.waypoints) - 1)
        
        # Update outputs
        self.outputs["position"] = self.position.copy()
        self.outputs["velocity"] = self.velocity.copy()
        self.outputs["attitude"] = self.attitude.copy()
        self.outputs["waypoint_distance"] = waypoint_distance
        
        return self.outputs
    
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """
        Set navigation waypoints.
        
        Args:
            waypoints: List of waypoint positions
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0


class PowerSystem(Component):
    """Power system component."""
    
    def __init__(self, component_id: str, max_power: float = 1000.0):
        """
        Initialize power system.
        
        Args:
            component_id: Unique identifier for the component
            max_power: Maximum power output in Watts
        """
        super().__init__(component_id, ComponentType.POWER)
        self.max_power = max_power
        self.current_power = 0.0
        self.battery_capacity = 1000.0  # Watt-hours
        self.battery_remaining = self.battery_capacity
        self.outputs = {
            "power_available": max_power,
            "battery_remaining": self.battery_capacity,
            "battery_percentage": 100.0
        }
    
    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update power system.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Power system outputs
        """
        if not self.enabled:
            self.outputs["power_available"] = 0.0
            return self.outputs
            
        # Calculate total power consumption from connected components
        total_consumption = sum(comp.power_consumption for comp in self.connected_components)
        
        # Limit power to available
        self.current_power = min(total_consumption, self.max_power * self.health)
        
        # Update battery
        energy_consumed = self.current_power * dt / 3600.0  # Convert to watt-hours
        self.battery_remaining = max(0.0, self.battery_remaining - energy_consumed)
        battery_percentage = 100.0 * self.battery_remaining / self.battery_capacity
        
        # Update outputs
        self.outputs["power_available"] = self.max_power * self.health
        self.outputs["battery_remaining"] = self.battery_remaining
        self.outputs["battery_percentage"] = battery_percentage
        
        return self.outputs


class UCAVSystem:
    """Complete UCAV system model."""
    
    def __init__(self):
        """Initialize UCAV system."""
        self.components = {}
        self.time = 0.0
    
    def add_component(self, component: Component):
        """
        Add component to the system.
        
        Args:
            component: Component to add
        """
        self.components[component.component_id] = component
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """
        Get component by ID.
        
        Args:
            component_id: Component ID
            
        Returns:
            Component or None if not found
        """
        return self.components.get(component_id)
    
    def update(self, dt: float):
        """
        Update all components.
        
        Args:
            dt: Time step in seconds
        """
        # Update time
        self.time += dt
        
        # Update components
        for component in self.components.values():
            outputs = component.update(dt)
            
            # Propagate outputs to connected components
            for connected in component.connected_components:
                connected.set_inputs({f"{component.component_id}_{k}": v for k, v in outputs.items()})
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            Dictionary with system status
        """
        return {
            "time": self.time,
            "components": {comp_id: comp.get_status() for comp_id, comp in self.components.items()}
        }