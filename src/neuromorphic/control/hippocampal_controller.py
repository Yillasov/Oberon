"""
Hippocampal-inspired control algorithm for neuromorphic hardware.

This module implements a simplified model of the hippocampal place cell system
for spatial navigation and path planning.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque


class PlaceCell:
    """Simulates a hippocampal place cell with spatial receptive field."""
    
    def __init__(self, center: np.ndarray, width: float = 5.0):
        """
        Initialize place cell with a specific receptive field.
        
        Args:
            center: Center position of the place field [x, y, z]
            width: Width of the receptive field (sigma of Gaussian)
        """
        self.center = center
        self.width = width
        self.activity = 0.0
    
    def compute_activity(self, position: np.ndarray) -> float:
        """
        Compute place cell activity based on current position.
        
        Args:
            position: Current position [x, y, z]
            
        Returns:
            Activity level (0-1)
        """
        # Use 2D distance (x,y plane) for place field
        distance = np.linalg.norm(position[:2] - self.center[:2])
        self.activity = np.exp(-(distance**2) / (2 * self.width**2))
        return self.activity


class HeadDirectionCell:
    """Simulates a head direction cell with directional tuning."""
    
    def __init__(self, preferred_direction: float):
        """
        Initialize head direction cell.
        
        Args:
            preferred_direction: Preferred heading in radians
        """
        self.preferred_direction = preferred_direction
        self.activity = 0.0
        self.tuning_width = 0.5  # Width of tuning curve (radians)
    
    def compute_activity(self, heading: float) -> float:
        """
        Compute head direction cell activity based on current heading.
        
        Args:
            heading: Current heading in radians
            
        Returns:
            Activity level (0-1)
        """
        # Calculate angular distance (circular)
        angle_diff = np.abs(heading - self.preferred_direction)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        # Compute activity using von Mises distribution (circular Gaussian)
        self.activity = np.exp(np.cos(angle_diff) / self.tuning_width - 1)
        return self.activity


class HippocampalController:
    """
    Hippocampal-inspired controller for spatial navigation.
    
    This controller mimics how the hippocampus and associated structures
    enable spatial navigation in mammals.
    """
    
    def __init__(self, environment_size: float = 100.0):
        """
        Initialize the hippocampal controller.
        
        Args:
            environment_size: Size of the environment in meters
        """
        # Create place cell map
        self.place_cells = []
        grid_size = 5  # Number of cells in each dimension
        spacing = environment_size / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * spacing + spacing/2
                y = j * spacing + spacing/2
                self.place_cells.append(PlaceCell(np.array([x, y, 0])))
        
        # Create head direction cells
        self.hd_cells = []
        num_hd_cells = 8
        for i in range(num_hd_cells):
            direction = i * 2 * np.pi / num_hd_cells
            self.hd_cells.append(HeadDirectionCell(direction))
        
        # Navigation targets
        self.waypoints = deque([
            np.array([20.0, 20.0, 10.0]),
            np.array([80.0, 20.0, 10.0]),
            np.array([80.0, 80.0, 10.0]),
            np.array([20.0, 80.0, 10.0])
        ])
        self.current_target = self.waypoints[0]
        self.target_threshold = 10.0  # Distance to consider waypoint reached
        
        # Control parameters
        self.max_roll = 0.3  # Maximum roll angle
        self.cruise_speed = 5.0  # Target speed
        self.prev_position = None
        self.memory_trace = []  # Simplified cognitive map
    
    def update_cognitive_map(self, position: np.ndarray) -> None:
        """
        Update internal cognitive map with current position.
        
        Args:
            position: Current position
        """
        if self.prev_position is not None:
            # Only store if moved significantly
            if np.linalg.norm(position - self.prev_position) > 2.0:
                self.memory_trace.append(position.copy())
                # Keep memory trace limited
                if len(self.memory_trace) > 20:
                    self.memory_trace.pop(0)
        
        self.prev_position = position.copy()
    
    def check_waypoint(self, position: np.ndarray) -> bool:
        """
        Check if current waypoint is reached and update if needed.
        
        Args:
            position: Current position
            
        Returns:
            True if waypoint was updated
        """
        if np.linalg.norm(position[:2] - self.current_target[:2]) < self.target_threshold:
            # Move to next waypoint
            self.waypoints.append(self.waypoints.popleft())
            self.current_target = self.waypoints[0]
            return True
        return False
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using hippocampal navigation model.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Extract state variables
        position = state['position']
        velocity = state['velocity']
        orientation = state['orientation']
        
        # Update cognitive map
        self.update_cognitive_map(position)
        
        # Check if waypoint reached
        self.check_waypoint(position)
        
        # Compute place cell activities
        for cell in self.place_cells:
            cell.compute_activity(position)
        
        # Compute head direction cell activities
        heading = orientation[2]  # Yaw angle
        for cell in self.hd_cells:
            cell.compute_activity(heading)
        
        # Calculate vector to target
        to_target = self.current_target - position
        target_distance = np.linalg.norm(to_target[:2])
        
        # Calculate desired heading
        desired_heading = np.arctan2(to_target[1], to_target[0])
        
        # Calculate heading error (circular)
        heading_error = desired_heading - heading
        if heading_error > np.pi:
            heading_error -= 2 * np.pi
        elif heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Calculate altitude error
        altitude_error = self.current_target[2] - position[2]
        
        # Compute control outputs
        
        # Rudder control based on heading error
        rudder = 0.5 * heading_error
        
        # Roll control - bank into turns
        roll_target = self.max_roll * np.sign(heading_error) * min(1.0, abs(heading_error))
        roll_error = roll_target - orientation[0]
        aileron = 0.5 * roll_error
        
        # Pitch and throttle for altitude control
        elevator = 0.2 * altitude_error - 0.1 * orientation[1]
        throttle = 0.5 + 0.1 * altitude_error
        
        # Adjust throttle based on speed
        current_speed = np.linalg.norm(velocity[:2])
        speed_error = self.cruise_speed - current_speed
        throttle += 0.1 * speed_error
        
        # Prepare control outputs
        controls = {
            'aileron': float(np.clip(aileron, -1.0, 1.0)),
            'elevator': float(np.clip(elevator, -1.0, 1.0)),
            'rudder': float(np.clip(rudder, -1.0, 1.0)),
            'throttle': float(np.clip(throttle, 0.0, 1.0))
        }
        
        return controls


def create_controller() -> Callable:
    """
    Create a hippocampal controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = HippocampalController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function