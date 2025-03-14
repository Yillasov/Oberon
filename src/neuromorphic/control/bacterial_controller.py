"""
Bacterial chemotaxis-inspired control algorithm for neuromorphic hardware.

This module implements a simplified bacterial navigation model for control,
inspired by how bacteria sense and move toward favorable conditions.
"""
import numpy as np
from typing import Dict, Callable


class ChemotaxisCell:
    """Simple implementation of a bacterial chemotaxis cell."""
    
    def __init__(self, adaptation_rate: float = 0.1):
        """
        Initialize chemotaxis cell.
        
        Args:
            adaptation_rate: Rate of adaptation to stimulus
        """
        self.current_stimulus = 0.0
        self.stimulus_memory = 0.0
        self.adaptation_rate = adaptation_rate
        self.tumble_probability = 0.5
        self.run_direction = np.random.uniform(-1.0, 1.0)
    
    def update(self, stimulus: float, dt: float) -> float:
        """
        Update cell state and get movement response.
        
        Args:
            stimulus: Current stimulus value
            dt: Time step
            
        Returns:
            Movement response
        """
        # Calculate temporal gradient (change in stimulus)
        stimulus_change = stimulus - self.stimulus_memory
        
        # Update memory with adaptation
        self.stimulus_memory += self.adaptation_rate * (stimulus - self.stimulus_memory) * dt
        self.current_stimulus = stimulus
        
        # Determine if we should tumble (change direction) or run
        if stimulus_change > 0:
            # Improving conditions - keep going in same direction
            self.tumble_probability = max(0.1, self.tumble_probability - 0.1 * dt)
        else:
            # Worsening conditions - more likely to change direction
            self.tumble_probability = min(0.9, self.tumble_probability + 0.2 * dt)
        
        # Tumble (change direction) with probability
        if np.random.random() < self.tumble_probability * dt:
            self.run_direction = np.random.uniform(-1.0, 1.0)
        
        return self.run_direction


class BacterialController:
    """
    Bacterial chemotaxis-based controller.
    
    This controller uses bacterial navigation principles to generate
    control outputs that seek favorable conditions.
    """
    
    def __init__(self):
        """Initialize the bacterial controller."""
        # Create chemotaxis cells for each control dimension
        self.roll_cell = ChemotaxisCell(adaptation_rate=0.2)
        self.pitch_cell = ChemotaxisCell(adaptation_rate=0.2)
        self.yaw_cell = ChemotaxisCell(adaptation_rate=0.2)
        self.throttle_cell = ChemotaxisCell(adaptation_rate=0.1)
        
        # Target values
        self.target_altitude = 10.0  # meters
        
        # Last update time
        self.last_time = 0.0
        
        # Performance memory
        self.performance_memory = 0.0
    
    def compute_control(self, state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        """
        Compute control outputs using bacterial chemotaxis.
        
        Args:
            state: Current state dictionary
            time: Current simulation time
            
        Returns:
            Control dictionary
        """
        # Calculate time step
        dt = time - self.last_time if self.last_time > 0 else 0.01
        self.last_time = time
        
        # Extract state variables
        position = state['position']
        orientation = state['orientation']
        
        # Calculate performance metric (negative of errors)
        altitude_error = abs(self.target_altitude - position[2])
        orientation_error = np.sum(np.abs(orientation))
        current_performance = -1.0 * (altitude_error + orientation_error)
        
        # Update cells with performance as stimulus
        roll_response = self.roll_cell.update(current_performance, dt)
        pitch_response = self.pitch_cell.update(current_performance, dt)
        yaw_response = self.yaw_cell.update(current_performance, dt)
        throttle_response = self.throttle_cell.update(current_performance, dt)
        
        # Combine random exploration with direct error correction
        aileron = roll_response * 0.2 - 0.3 * orientation[0]
        elevator = pitch_response * 0.2 - 0.3 * orientation[1]
        rudder = yaw_response * 0.2 - 0.3 * orientation[2]
        
        # Throttle depends on altitude error
        altitude_correction = 0.1 * (self.target_altitude - position[2])
        throttle = 0.5 + throttle_response * 0.1 + altitude_correction
        
        # Prepare control outputs
        controls = {
            'aileron': float(np.clip(aileron, -1.0, 1.0)),
            'elevator': float(np.clip(elevator, -1.0, 1.0)),
            'rudder': float(np.clip(rudder, -1.0, 1.0)),
            'throttle': float(np.clip(throttle, 0.0, 1.0))
        }
        
        # Update performance memory
        self.performance_memory = current_performance
        
        return controls


def create_controller() -> Callable:
    """
    Create a bacterial controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = BacterialController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, time)
    
    return control_function