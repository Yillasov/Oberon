"""
Cerebellar-inspired control algorithm for neuromorphic hardware.

This module implements a simplified model of cerebellar motor control
for adaptive learning in control tasks.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class PurkinjeCellModel:
    """Simplified model of cerebellar Purkinje cells."""
    
    def __init__(self, num_inputs: int = 10):
        """
        Initialize Purkinje cell model.
        
        Args:
            num_inputs: Number of input connections
        """
        self.weights = np.random.normal(0, 0.1, num_inputs)
        self.learning_rate = 0.01
        self.activity = 0.0
    
    def compute(self, inputs: np.ndarray) -> float:
        """
        Compute Purkinje cell activity.
        
        Args:
            inputs: Input signals
            
        Returns:
            Cell activity
        """
        self.activity = np.dot(self.weights, inputs)
        return self.activity
    
    def learn(self, inputs: np.ndarray, error: float) -> None:
        """
        Update weights based on error signal.
        
        Args:
            inputs: Input signals
            error: Error signal
        """
        # Simplified cerebellar learning rule
        self.weights -= self.learning_rate * error * inputs


class CerebellarController:
    """
    Cerebellar-inspired adaptive controller.
    
    This controller mimics the cerebellum's role in motor learning
    and predictive control.
    """
    
    def __init__(self, num_features: int = 12):
        """
        Initialize the cerebellar controller.
        
        Args:
            num_features: Number of input features
        """
        # Create Purkinje cells for each control output
        self.pc_aileron = PurkinjeCellModel(num_features)
        self.pc_elevator = PurkinjeCellModel(num_features)
        self.pc_rudder = PurkinjeCellModel(num_features)
        self.pc_throttle = PurkinjeCellModel(num_features)
        
        # Target values
        self.target_altitude = 5.0
        self.target_speed = 3.0
        
        # Previous errors for learning
        self.prev_errors = np.zeros(4)
        
        # Feature extraction parameters
        self.delay_line = [np.zeros(3) for _ in range(4)]  # Simple delay line for temporal features
    
    def extract_features(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract features from state for cerebellar processing.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Feature vector
        """
        # Extract basic state information
        position = state['position']
        velocity = state['velocity']
        orientation = state['orientation']
        
        # Update delay line (simple temporal memory)
        self.delay_line.pop(0)
        self.delay_line.append(np.array([
            position[2],  # altitude
            np.linalg.norm(velocity[:2]),  # horizontal speed
            orientation[1]  # pitch
        ]))
        
        # Compute features
        features = []
        
        # Current state features
        features.append(position[2])  # altitude
        features.append(np.linalg.norm(velocity[:2]))  # speed
        features.append(orientation[0])  # roll
        features.append(orientation[1])  # pitch
        features.append(orientation[2])  # yaw
        
        # Temporal difference features
        for i in range(3):
            features.append(self.delay_line[-1][i] - self.delay_line[-2][i])
        
        # Nonlinear combinations
        features.append(np.sin(orientation[0]))  # sin(roll)
        features.append(np.cos(orientation[1]))  # cos(pitch)
        features.append(position[2] * np.linalg.norm(velocity[:2]))  # altitude * speed
        features.append(1.0)  # bias term
        
        return np.array(features)
    
    def compute_control(self, state: Dict[str, np.ndarray], dt: float) -> Dict[str, float]:
        """
        Compute control outputs using cerebellar model.
        
        Args:
            state: Current state dictionary
            dt: Time step in seconds
            
        Returns:
            Control dictionary
        """
        # Extract features
        features = self.extract_features(state)
        
        # Compute Purkinje cell outputs
        aileron = self.pc_aileron.compute(features)
        elevator = self.pc_elevator.compute(features)
        rudder = self.pc_rudder.compute(features)
        throttle = self.pc_throttle.compute(features)
        
        # Calculate errors for learning
        altitude_error = state['position'][2] - self.target_altitude
        speed_error = np.linalg.norm(state['velocity'][:2]) - self.target_speed
        roll_error = state['orientation'][0]  # Target roll is 0
        yaw_rate_error = state['angular_velocity'][2]  # Target yaw rate is 0
        
        errors = np.array([roll_error, altitude_error, yaw_rate_error, speed_error])
        
        # Learn from errors
        self.pc_aileron.learn(features, errors[0])
        self.pc_elevator.learn(features, errors[1])
        self.pc_rudder.learn(features, errors[2])
        self.pc_throttle.learn(features, errors[3])
        
        # Store errors for next iteration
        self.prev_errors = errors
        
        # Prepare control outputs
        controls = {
            'aileron': float(np.clip(aileron, -1.0, 1.0)),
            'elevator': float(np.clip(elevator, -1.0, 1.0)),
            'rudder': float(np.clip(rudder, -1.0, 1.0)),
            'throttle': float(np.clip(0.5 + throttle, 0.0, 1.0))
        }
        
        return controls


def create_controller() -> Callable:
    """
    Create a cerebellar controller function for use with the simulation.
    
    Returns:
        Control function that takes state and time as input
    """
    controller = CerebellarController()
    
    def control_function(state: Dict[str, np.ndarray], time: float) -> Dict[str, float]:
        return controller.compute_control(state, 0.01)
    
    return control_function