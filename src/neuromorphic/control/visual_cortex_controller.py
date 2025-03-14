"""
Visual cortex-inspired control algorithm for neuromorphic hardware.

This module implements a simplified model of the visual cortex's processing
of optical flow for navigation and obstacle avoidance.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class MotionDetector:
    """Simulates motion detection neurons in the visual cortex."""
    
    def __init__(self, num_directions: int = 8, num_elevations: int = 3):
        """
        Initialize motion detector array.
        
        Args:
            num_directions: Number of horizontal directions
            num_elevations: Number of vertical elevations
        """
        self.num_directions = num_directions
        self.num_elevations = num_elevations
        
        # Create directional preferences
        self.directions = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
        self.elevations = np.linspace(-np.pi/3, np.pi/3, num_elevations)
        
        # Activity levels for each detector
        self.activities = np.zeros((num_elevations, num_directions))
        
        # Temporal filter for motion detection
        self.prev_image = None
        self.sensitivity = 1.0
    
    def compute_flow(self, velocity: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Compute optical flow based on motion.
        
        Args:
            velocity: 3D velocity vector
            orientation: 3D orientation vector
            
        Returns:
            2D array of motion detector activities
        """
        # Extract orientation angles
        roll, pitch, yaw = orientation
        
        # Create simplified rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        
        # Transform velocity to body frame
        vx_body = cos_yaw * velocity[0] + sin_yaw * velocity[1]
        vy_body = -sin_yaw * velocity[0] + cos_yaw * velocity[1]
        vz_body = velocity[2]
        
        # Compute flow for each detector
        for i, elevation in enumerate(self.elevations):
            for j, azimuth in enumerate(self.directions):
                # Direction vector in body frame
                dx = cos_pitch * np.cos(azimuth)
                dy = np.sin(azimuth)
                dz = sin_pitch * np.cos(azimuth)
                
                # Dot product gives flow magnitude
                flow = -(vx_body * dx + vy_body * dy + vz_body * dz)
                
                # Apply sensitivity and store
                self.activities[i, j] = flow * self.sensitivity
        
        return self.activities


class VisualCortexController:
    """
    Visual cortex-inspired controller for navigation.
    
    This controller mimics how the visual cortex processes optical flow
    for navigation and obstacle avoidance.
    """
    
    def __init__(self):
        """Initialize the visual cortex controller."""
        # Motion detectors
        self.motion_detector = MotionDetector(num_directions=8, num_elevations=3)
        
        # Control parameters
        self.target_altitude = 10.0  # meters
        self.cruise_speed = 5.0      # m/s
        self.balance_weight = 0.5    # Weight for flow balancing
        
        # Flow balance parameters
        self.left_right_balance = 0.0
        self.up_down_balance = 0.0
        self.flow_memory = 0.8       # Memory coefficient for flow smoothing
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using visual cortex model.
        
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
        
        # Compute optical flow
        flow = self.motion_detector.compute_flow(velocity, orientation)
        
        # Calculate flow balances
        
        # Left-right balance (for roll control)
        left_flow = np.sum(flow[:, :4])
        right_flow = np.sum(flow[:, 4:])
        current_lr_balance = left_flow - right_flow
        
        # Smooth the balance with memory
        self.left_right_balance = (self.flow_memory * self.left_right_balance + 
                                  (1 - self.flow_memory) * current_lr_balance)
        
        # Up-down balance (for pitch control)
        up_flow = np.sum(flow[0, :])
        down_flow = np.sum(flow[2, :])
        current_ud_balance = up_flow - down_flow
        
        # Smooth the balance with memory
        self.up_down_balance = (self.flow_memory * self.up_down_balance + 
                               (1 - self.flow_memory) * current_ud_balance)
        
        # Calculate altitude error
        altitude_error = self.target_altitude - position[2]
        
        # Calculate speed error
        current_speed = np.linalg.norm(velocity[:2])
        speed_error = self.cruise_speed - current_speed
        
        # Compute control outputs
        
        # Roll control (aileron) - balance left/right flow
        aileron = -0.3 * self.left_right_balance
        
        # Pitch control (elevator) - balance up/down flow and maintain altitude
        elevator = -0.2 * self.up_down_balance + 0.1 * altitude_error
        
        # Yaw control (rudder) - dampen yaw motion
        # Use flow difference between diagonal directions
        diag1 = flow[:, 1] + flow[:, 5]  # NE + SW
        diag2 = flow[:, 3] + flow[:, 7]  # NW + SE
        yaw_balance = np.sum(diag1 - diag2)
        rudder = -0.2 * yaw_balance
        
        # Throttle control - maintain speed and altitude
        throttle = 0.5 + 0.1 * speed_error + 0.05 * altitude_error
        
        # Prepare control outputs
        controls = {
            'aileron': float(np.clip(aileron, -1.0, 1.0)),
            'elevator': float(np.clip(elevator, -1.0, 1.0)),
            'rudder': float(np.clip(rudder, -1.0, 1.0)),
            'throttle': float(np.clip(throttle, 0.0, 1.0))
        }
        
        return controls


class FocusOfExpansion:
    """Detects the focus of expansion in optical flow."""
    
    def __init__(self, resolution: int = 10):
        """
        Initialize focus of expansion detector.
        
        Args:
            resolution: Grid resolution for FOE detection
        """
        self.resolution = resolution
        self.grid_x = np.linspace(-1, 1, resolution)
        self.grid_y = np.linspace(-1, 1, resolution)
        self.foe = np.zeros(2)  # x, y coordinates of FOE
    
    def detect(self, flow: np.ndarray) -> np.ndarray:
        """
        Detect focus of expansion from flow field.
        
        Args:
            flow: Optical flow field
            
        Returns:
            Coordinates of focus of expansion
        """
        # Simplified FOE detection
        # In a real system, this would use the full flow field
        
        # Use flow balance to estimate FOE
        horizontal_balance = np.sum(flow[:, :4]) - np.sum(flow[:, 4:])
        vertical_balance = np.sum(flow[0, :]) - np.sum(flow[2, :])
        
        # Normalize to [-1, 1]
        total_flow = np.sum(np.abs(flow))
        if total_flow > 0.1:
            self.foe[0] = horizontal_balance / total_flow
            self.foe[1] = vertical_balance / total_flow
        
        return self.foe


def create_controller() -> Callable:
    """
    Create a visual cortex controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = VisualCortexController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function