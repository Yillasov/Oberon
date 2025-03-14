"""
Basic safety protocols for neuromorphic control systems.

This module implements simplified safety checks and emergency responses
to ensure safe operation of neuromorphic-controlled systems.
"""
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional


class SafetyBound:
    """Simple implementation of a safety boundary check."""
    
    def __init__(self, min_value: float, max_value: float, name: str):
        """
        Initialize safety bound.
        
        Args:
            min_value: Minimum safe value
            max_value: Maximum safe value
            name: Name of the parameter being checked
        """
        self.min_value = min_value
        self.max_value = max_value
        self.name = name
        self.violation_count = 0
        self.max_violations = 3  # Number of violations before triggering emergency
    
    def check(self, value: float) -> bool:
        """
        Check if value is within safe bounds.
        
        Args:
            value: Value to check
            
        Returns:
            True if safe, False if violation
        """
        is_safe = self.min_value <= value <= self.max_value
        
        if not is_safe:
            self.violation_count += 1
        else:
            # Reset violation count if value returns to safe range
            self.violation_count = max(0, self.violation_count - 1)
        
        return is_safe
    
    def is_emergency(self) -> bool:
        """
        Check if this bound has triggered an emergency.
        
        Returns:
            True if emergency should be triggered
        """
        return self.violation_count >= self.max_violations


class SafetyMonitor:
    """Simple implementation of a safety monitoring system."""
    
    def __init__(self):
        """Initialize safety monitor."""
        # Define safety bounds
        self.bounds = [
            # Orientation bounds (in radians)
            SafetyBound(-0.5, 0.5, "roll"),
            SafetyBound(-0.5, 0.5, "pitch"),
            SafetyBound(-np.pi, np.pi, "yaw"),
            
            # Altitude bounds (in meters)
            SafetyBound(2.0, 50.0, "altitude"),
            
            # Velocity bounds (in m/s)
            SafetyBound(-10.0, 10.0, "velocity_x"),
            SafetyBound(-10.0, 10.0, "velocity_y"),
            SafetyBound(-5.0, 5.0, "velocity_z"),
        ]
        
        # Safety status
        self.emergency_mode = False
        self.warning_mode = False
        
        # Last check time
        self.last_check_time = 0.0
    
    def check_safety(self, state: Dict[str, np.ndarray], time: float) -> Tuple[bool, List[str]]:
        """
        Check if current state is safe.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        # Extract relevant state variables
        violations = []
        
        # Check orientation bounds
        if 'orientation' in state:
            orientation = state['orientation']
            if len(orientation) >= 3:
                # Check roll
                if not self.bounds[0].check(orientation[0]):
                    violations.append(f"Roll out of bounds: {orientation[0]:.2f}")
                
                # Check pitch
                if not self.bounds[1].check(orientation[1]):
                    violations.append(f"Pitch out of bounds: {orientation[1]:.2f}")
                
                # Check yaw
                if not self.bounds[2].check(orientation[2]):
                    violations.append(f"Yaw out of bounds: {orientation[2]:.2f}")
        
        # Check altitude bounds
        if 'position' in state:
            position = state['position']
            if len(position) >= 3:
                altitude = position[2]
                if not self.bounds[3].check(altitude):
                    violations.append(f"Altitude out of bounds: {altitude:.2f}")
        
        # Check velocity bounds if available
        if 'velocity' in state:
            velocity = state['velocity']
            if len(velocity) >= 3:
                # Check velocity_x
                if not self.bounds[4].check(velocity[0]):
                    violations.append(f"X velocity out of bounds: {velocity[0]:.2f}")
                
                # Check velocity_y
                if not self.bounds[5].check(velocity[1]):
                    violations.append(f"Y velocity out of bounds: {velocity[1]:.2f}")
                
                # Check velocity_z
                if not self.bounds[6].check(velocity[2]):
                    violations.append(f"Z velocity out of bounds: {velocity[2]:.2f}")
        
        # Check for emergency conditions
        for bound in self.bounds:
            if bound.is_emergency():
                self.emergency_mode = True
                violations.append(f"EMERGENCY: Persistent {bound.name} violations")
        
        # Set warning mode if any violations
        self.warning_mode = len(violations) > 0
        
        # Update last check time
        self.last_check_time = time
        
        return len(violations) == 0, violations


class SafetyController:
    """
    Safety controller that overrides normal control in emergency situations.
    
    This controller implements basic emergency responses to unsafe conditions.
    """
    
    def __init__(self):
        """Initialize safety controller."""
        # Create safety monitor
        self.monitor = SafetyMonitor()
        
        # Emergency response parameters
        self.recovery_altitude = 10.0  # meters
        self.recovery_time = 0.0
        self.recovery_duration = 5.0  # seconds
    
    def compute_emergency_control(self, state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute emergency control outputs.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Emergency control dictionary
        """
        # Extract state variables
        position = state.get('position', np.zeros(3))
        orientation = state.get('orientation', np.zeros(3))
        
        # Basic recovery controls - level off and maintain safe altitude
        controls = {
            'aileron': -0.5 * orientation[0],  # Counter roll
            'elevator': -0.5 * orientation[1],  # Counter pitch
            'rudder': -0.2 * orientation[2],   # Counter yaw
            'throttle': 0.5                    # Moderate throttle
        }
        
        # Adjust throttle based on altitude
        altitude = position[2]
        if altitude < self.recovery_altitude:
            # Increase throttle if too low
            controls['throttle'] = 0.8
            controls['elevator'] = 0.2  # Pitch up slightly
        elif altitude > self.recovery_altitude + 5.0:
            # Decrease throttle if too high
            controls['throttle'] = 0.3
            controls['elevator'] = -0.1  # Pitch down slightly
        
        return controls
    
    def apply_safety_protocols(self, state: Dict[str, np.ndarray], 
                              normal_controls: Dict[str, float],
                              time: float) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Apply safety protocols to control outputs.
        
        Args:
            state: Current state dictionary
            normal_controls: Control outputs from normal controller
            time: Current simulation time
            
        Returns:
            Tuple of (safe_controls, safety_status)
        """
        # Check safety
        is_safe, violations = self.monitor.check_safety(state, time)
        
        # Prepare safety status
        safety_status = {
            'is_safe': is_safe,
            'warning_mode': self.monitor.warning_mode,
            'emergency_mode': self.monitor.emergency_mode,
            'violations': violations
        }
        
        # If in emergency mode, use emergency controls
        if self.monitor.emergency_mode:
            # Check if we should exit emergency mode
            if time - self.recovery_time > self.recovery_duration and is_safe:
                self.monitor.emergency_mode = False
                self.recovery_time = 0.0
                return normal_controls, safety_status
            
            # Otherwise, use emergency controls
            if self.recovery_time == 0.0:
                self.recovery_time = time
            
            emergency_controls = self.compute_emergency_control(state)
            return emergency_controls, safety_status
        
        # If safe, use normal controls
        return normal_controls, safety_status


def create_safety_controller() -> SafetyController:
    """
    Create a safety controller.
    
    Returns:
        Configured safety controller
    """
    return SafetyController()