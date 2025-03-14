"""
Integrated navigation and control system for neuromorphic operations.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
from .nav_system import NavigationSystem, NavState, NavigationMode


class ControlMode(Enum):
    AGGRESSIVE = "aggressive"     # Fast response, high power
    NORMAL = "normal"            # Balanced control
    CONSERVATIVE = "conservative" # Energy-saving, smooth control


@dataclass
class ControlCommand:
    """Control command structure."""
    thrust: np.ndarray    # [x, y, z] thrust commands
    torque: np.ndarray    # [roll, pitch, yaw] commands
    timestamp: float
    mode: ControlMode


class IntegratedController:
    """Integrated navigation and control system."""
    
    def __init__(self, nav_system: NavigationSystem):
        self.nav = nav_system
        self.mode = ControlMode.NORMAL
        
        # Control parameters
        self.max_thrust = 10.0   # N
        self.max_torque = 2.0    # Nm
        self.control_rate = 20   # Hz
        
        # Performance tracking
        self.position_error = np.zeros(3)
        self.orientation_error = np.zeros(3)
        
        # Target states
        self.target_position = np.zeros(3)
        self.target_orientation = np.zeros(3)
    
    async def update_control(self) -> ControlCommand:
        """Update control commands based on navigation state."""
        try:
            # Get current navigation state
            nav_state = await self.nav.update_navigation()
            
            # Calculate errors
            self.position_error = self.target_position - nav_state.position
            self.orientation_error = self.target_orientation - nav_state.orientation
            
            # Generate control commands
            command = self._generate_commands(nav_state)
            
            # Adjust control mode based on nav mode
            self._align_control_mode(nav_state.mode)
            
            return command
            
        except Exception as e:
            print(f"Control error: {str(e)}")
            return ControlCommand(
                thrust=np.zeros(3),
                torque=np.zeros(3),
                timestamp=datetime.now().timestamp(),
                mode=self.mode
            )
    
    def _generate_commands(self, nav_state: NavState) -> ControlCommand:
        """Generate control commands based on current state."""
        # Simple PD control
        kp_pos = self._get_position_gains()
        kp_ori = self._get_orientation_gains()
        
        # Calculate commands
        thrust = np.clip(
            kp_pos * self.position_error - 0.5 * nav_state.velocity,
            -self.max_thrust,
            self.max_thrust
        )
        
        torque = np.clip(
            kp_ori * self.orientation_error,
            -self.max_torque,
            self.max_torque
        )
        
        return ControlCommand(
            thrust=thrust,
            torque=torque,
            timestamp=datetime.now().timestamp(),
            mode=self.mode
        )
    
    def _get_position_gains(self) -> float:
        """Get position control gains based on mode."""
        if self.mode == ControlMode.AGGRESSIVE:
            return 2.0
        elif self.mode == ControlMode.NORMAL:
            return 1.0
        else:
            return 0.5
    
    def _get_orientation_gains(self) -> float:
        """Get orientation control gains based on mode."""
        if self.mode == ControlMode.AGGRESSIVE:
            return 1.0
        elif self.mode == ControlMode.NORMAL:
            return 0.5
        else:
            return 0.25
    
    def _align_control_mode(self, nav_mode: NavigationMode):
        """Align control mode with navigation mode."""
        if nav_mode == NavigationMode.PRECISION:
            self.mode = ControlMode.AGGRESSIVE
        elif nav_mode == NavigationMode.BALANCED:
            self.mode = ControlMode.NORMAL
        else:
            self.mode = ControlMode.CONSERVATIVE
    
    def set_target(self, position: np.ndarray, orientation: np.ndarray):
        """Set target position and orientation."""
        self.target_position = position
        self.target_orientation = orientation
    
    def get_errors(self) -> Dict[str, np.ndarray]:
        """Get current control errors."""
        return {
            'position_error': self.position_error,
            'orientation_error': self.orientation_error
        }