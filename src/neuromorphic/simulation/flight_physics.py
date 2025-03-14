"""
Simple flight physics simulation for neuromorphic hardware testing.

This module provides a lightweight aircraft simulation model that can be
used to benchmark neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional


class AircraftModel:
    """Simple 3D aircraft physics model."""
    
    def __init__(self):
        # State variables
        self.position = np.zeros(3)  # x, y, z in meters
        self.velocity = np.zeros(3)  # vx, vy, vz in m/s
        self.orientation = np.zeros(3)  # roll, pitch, yaw in radians
        self.angular_velocity = np.zeros(3)  # roll, pitch, yaw rates in rad/s
        
        # Aircraft parameters
        self.mass = 1.0  # kg
        self.inertia = np.array([0.1, 0.2, 0.3])  # kg*m^2
        self.wing_area = 0.5  # m^2
        self.wing_span = 1.0  # m
        self.air_density = 1.225  # kg/m^3
        
        # Control inputs (normalized -1 to 1)
        self.controls = {
            'aileron': 0.0,
            'elevator': 0.0,
            'rudder': 0.0,
            'throttle': 0.0
        }
        
        # Aerodynamic coefficients
        self.drag_coef = 0.05
        self.lift_coef = 0.4
        self.moment_coef = np.array([0.1, 0.2, 0.05])  # roll, pitch, yaw
    
    def set_controls(self, aileron: float, elevator: float, 
                    rudder: float, throttle: float) -> None:
        """Set control surface positions (-1 to 1 range)."""
        self.controls['aileron'] = np.clip(aileron, -1.0, 1.0)
        self.controls['elevator'] = np.clip(elevator, -1.0, 1.0)
        self.controls['rudder'] = np.clip(rudder, -1.0, 1.0)
        self.controls['throttle'] = np.clip(throttle, 0.0, 1.0)
    
    def update(self, dt: float) -> None:
        """Update aircraft state for one time step."""
        # Calculate airspeed
        airspeed = np.linalg.norm(self.velocity)
        if airspeed < 0.1:
            airspeed = 0.1  # Prevent division by zero
        
        # Calculate angle of attack and sideslip
        if self.velocity[0] != 0:
            alpha = np.arctan2(self.velocity[2], self.velocity[0])  # Angle of attack
            beta = np.arcsin(self.velocity[1] / airspeed)  # Sideslip angle
        else:
            alpha, beta = 0.0, 0.0
        
        # Calculate aerodynamic forces
        q = 0.5 * self.air_density * airspeed**2 * self.wing_area
        
        # Lift and drag in body frame
        lift = q * self.lift_coef * (1.0 + 5.0 * alpha)
        drag = q * self.drag_coef * (1.0 + 10.0 * alpha**2)
        
        # Thrust
        thrust = 10.0 * self.controls['throttle']
        
        # Transform forces to global frame
        cos_pitch = np.cos(self.orientation[1])
        sin_pitch = np.sin(self.orientation[1])
        cos_yaw = np.cos(self.orientation[2])
        sin_yaw = np.sin(self.orientation[2])
        
        # Simplified force calculation
        forces = np.zeros(3)
        forces[0] = thrust - drag * cos_pitch * cos_yaw
        forces[1] = -drag * cos_pitch * sin_yaw
        forces[2] = -lift - drag * sin_pitch
        
        # Add gravity
        forces[2] += -9.81 * self.mass
        
        # Calculate moments
        moments = np.zeros(3)
        moments[0] = q * self.moment_coef[0] * self.controls['aileron']  # Roll
        moments[1] = q * self.moment_coef[1] * self.controls['elevator']  # Pitch
        moments[2] = q * self.moment_coef[2] * self.controls['rudder']    # Yaw
        
        # Update linear velocity
        self.velocity += forces / self.mass * dt
        
        # Update angular velocity
        self.angular_velocity += moments / self.inertia * dt
        
        # Update position
        self.position += self.velocity * dt
        
        # Update orientation
        self.orientation += self.angular_velocity * dt
        
        # Normalize angles to [-pi, pi]
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get the current aircraft state."""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy()
        }


class FlightSimulation:
    """Simple flight simulation environment."""
    
    def __init__(self):
        self.aircraft = AircraftModel()
        self.time = 0.0
        self.history = []
    
    def reset(self, position: Optional[np.ndarray] = None, 
             velocity: Optional[np.ndarray] = None) -> None:
        """Reset the simulation."""
        self.aircraft = AircraftModel()
        if position is not None:
            self.aircraft.position = position.copy()
        if velocity is not None:
            self.aircraft.velocity = velocity.copy()
        self.time = 0.0
        self.history = []
    
    def step(self, controls: Dict[str, float], dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Advance simulation by one time step.
        
        Args:
            controls: Dictionary with control inputs
            dt: Time step in seconds
            
        Returns:
            Current aircraft state
        """
        self.aircraft.set_controls(
            controls.get('aileron', 0.0),
            controls.get('elevator', 0.0),
            controls.get('rudder', 0.0),
            controls.get('throttle', 0.0)
        )
        
        self.aircraft.update(dt)
        self.time += dt
        
        state = self.aircraft.get_state()
        self.history.append((self.time, state))
        
        return state
    
    def run(self, control_func, duration: float, dt: float = 0.01) -> List[Tuple[float, Dict]]:
        """
        Run simulation for specified duration.
        
        Args:
            control_func: Function that returns control inputs given state
            duration: Duration in seconds
            dt: Time step in seconds
            
        Returns:
            List of (time, state) tuples
        """
        steps = int(duration / dt)
        for _ in range(steps):
            state = self.aircraft.get_state()
            controls = control_func(state, self.time)
            self.step(controls, dt)
        
        return self.history


"""
Physics-based modeling for control surface interactions in neuromorphic simulations.
"""
import numpy as np
from typing import Dict, Any, List, Optional


class FlightPhysics:
    """Simple physics model for control surface interactions."""
    
    def __init__(self, airframe_model: Any):
        """
        Initialize flight physics model.
        
        Args:
            airframe_model: Airframe model instance
        """
        self.airframe = airframe_model
        
        # Default aerodynamic coefficients
        self.aero_coeffs = {
            # Control effectiveness
            "aileron_effectiveness": 0.05,  # roll rate per degree
            "elevator_effectiveness": 0.03,  # pitch rate per degree
            "rudder_effectiveness": 0.02,   # yaw rate per degree
            
            # Damping coefficients
            "roll_damping": -0.5,
            "pitch_damping": -0.8,
            "yaw_damping": -0.4,
            
            # Stability coefficients
            "roll_stability": -0.1,
            "pitch_stability": -0.2,
            "yaw_stability": -0.15,
            
            # Thrust parameters
            "thrust_max": 500.0,  # Newtons
            "mass": 100.0,        # kg
            "inertia": np.array([20.0, 30.0, 25.0])  # kg·m²
        }
        
        # Update coefficients based on airframe if available
        if hasattr(airframe_model, 'get_performance_metrics'):
            metrics = airframe_model.get_performance_metrics()
            if 'control_authority' in metrics:
                control_factor = metrics['control_authority']
                self.aero_coeffs["aileron_effectiveness"] *= control_factor
                self.aero_coeffs["elevator_effectiveness"] *= control_factor
                self.aero_coeffs["rudder_effectiveness"] *= control_factor
    
    def update_physics(self, state: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Update physics based on current state and control inputs.
        
        Args:
            state: Current simulation state
            dt: Time step in seconds
        
        Returns:
            Updated simulation state
        """
        # Extract current state
        velocity = state["velocity"]
        attitude = state["attitude"]
        angular_velocity = state["angular_velocity"]
        control_inputs = state["control_inputs"]
        
        # Extract control surface deflections
        aileron = control_inputs.get("aileron", 0.0) * 20.0  # Scale to degrees
        elevator = control_inputs.get("elevator", 0.0) * 20.0
        rudder = control_inputs.get("rudder", 0.0) * 20.0
        throttle = control_inputs.get("throttle", 0.0)
        
        # Calculate airspeed (simplified)
        airspeed = np.linalg.norm(velocity)
        if airspeed < 0.1:
            airspeed = 0.1  # Prevent division by zero
        
        # Calculate dynamic pressure factor (q = 0.5 * rho * V^2)
        q_factor = airspeed**2 * 0.6
        
        # Calculate control moments
        roll_moment = self.aero_coeffs["aileron_effectiveness"] * aileron * q_factor
        pitch_moment = self.aero_coeffs["elevator_effectiveness"] * elevator * q_factor
        yaw_moment = self.aero_coeffs["rudder_effectiveness"] * rudder * q_factor
        
        # Add stability moments
        roll_moment += self.aero_coeffs["roll_stability"] * attitude[0] * q_factor
        pitch_moment += self.aero_coeffs["pitch_stability"] * attitude[1] * q_factor
        yaw_moment += self.aero_coeffs["yaw_stability"] * attitude[2] * q_factor
        
        # Add damping moments
        roll_moment += self.aero_coeffs["roll_damping"] * angular_velocity[0] * q_factor
        pitch_moment += self.aero_coeffs["pitch_damping"] * angular_velocity[1] * q_factor
        yaw_moment += self.aero_coeffs["yaw_damping"] * angular_velocity[2] * q_factor
        
        # Calculate angular acceleration
        angular_accel = np.array([
            roll_moment / self.aero_coeffs["inertia"][0],
            pitch_moment / self.aero_coeffs["inertia"][1],
            yaw_moment / self.aero_coeffs["inertia"][2]
        ])
        
        # Update angular velocity
        new_angular_velocity = angular_velocity + angular_accel * dt
        
        # Calculate thrust
        thrust = throttle * self.aero_coeffs["thrust_max"]
        
        # Calculate acceleration (simplified)
        # Convert attitude to direction cosine matrix (simplified)
        pitch = attitude[1]
        roll = attitude[0]
        yaw = attitude[2]
        
        # Simplified direction vector
        direction = np.array([
            np.cos(pitch) * np.sin(yaw),
            np.cos(pitch) * np.cos(yaw),
            np.sin(pitch)
        ])
        
        # Apply thrust in body direction
        accel = direction * thrust / self.aero_coeffs["mass"]
        
        # Add gravity (simplified)
        accel[2] -= 9.81
        
        # Add drag (simplified)
        drag_factor = 0.01 * airspeed
        if np.linalg.norm(velocity) > 0:
            accel -= velocity * drag_factor / np.linalg.norm(velocity)
        
        # Update velocity
        new_velocity = velocity + accel * dt
        
        # Return updated state
        return {
            "velocity": new_velocity,
            "angular_velocity": new_angular_velocity,
            "acceleration": accel,
            "angular_acceleration": angular_accel
        }


def integrate_flight_physics(digital_twin):
    """Integrate flight physics into an existing digital twin."""
    physics = FlightPhysics(digital_twin.airframe_model)
    
    # Store the original update method
    original_update = digital_twin._update_simulation
    
    # Create enhanced update method
    def enhanced_update():
        # Call original update first
        original_update()
        
        # Apply physics update
        physics_update = physics.update_physics(
            digital_twin.current_state, 
            digital_twin.time_step
        )
        
        # Update state with physics results
        digital_twin.current_state["velocity"] = physics_update["velocity"]
        digital_twin.current_state["angular_velocity"] = physics_update["angular_velocity"]
        digital_twin.current_state["acceleration"] = physics_update["acceleration"]
        digital_twin.current_state["angular_acceleration"] = physics_update["angular_acceleration"]
    
    # Replace the update method
    digital_twin._update_simulation = enhanced_update
    
    return digital_twin