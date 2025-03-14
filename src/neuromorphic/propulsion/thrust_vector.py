"""
Thrust vectoring control module for propulsion systems.
"""
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class VectorMode(Enum):
    FIXED = "fixed"
    PITCH = "pitch"
    YAW = "yaw"
    FULL = "full"


@dataclass
class VectoringSpecs:
    """Thrust vectoring specifications."""
    max_angle: float = 20.0         # degrees
    slew_rate: float = 60.0         # degrees/second
    response_time: float = 0.05     # seconds
    mechanical_limit: float = 25.0  # degrees
    precision: float = 0.1          # degrees


class ThrustVectorControl:
    """Thrust vectoring control system."""
    
    def __init__(self, specs: VectoringSpecs = VectoringSpecs()):
        self.specs = specs
        self.state = {
            "pitch_angle": 0.0,     # degrees
            "yaw_angle": 0.0,       # degrees
            "pitch_rate": 0.0,      # degrees/s
            "yaw_rate": 0.0,        # degrees/s
            "mode": VectorMode.FIXED,
            "thrust_magnitude": 0.0, # N
            "vectored_thrust": np.array([0.0, 0.0, 0.0])  # [x, y, z] components
        }
        
        self.target = {
            "pitch": 0.0,
            "yaw": 0.0
        }
    
    def update(self, thrust: float, pitch_cmd: float, yaw_cmd: float, 
               dt: float) -> Dict[str, Any]:
        """Update thrust vector state."""
        # Limit commands to mechanical limits
        pitch_cmd = np.clip(pitch_cmd, -self.specs.mechanical_limit, 
                          self.specs.mechanical_limit)
        yaw_cmd = np.clip(yaw_cmd, -self.specs.mechanical_limit, 
                         self.specs.mechanical_limit)
        
        # Calculate angle changes with slew rate limiting
        pitch_error = pitch_cmd - self.state["pitch_angle"]
        yaw_error = yaw_cmd - self.state["yaw_angle"]
        
        max_change = self.specs.slew_rate * dt
        
        pitch_change = np.clip(pitch_error, -max_change, max_change)
        yaw_change = np.clip(yaw_error, -max_change, max_change)
        
        # Update angles
        self.state["pitch_angle"] += pitch_change
        self.state["yaw_angle"] += yaw_change
        
        # Update rates
        self.state["pitch_rate"] = pitch_change / dt
        self.state["yaw_rate"] = yaw_change / dt
        
        # Store thrust magnitude
        self.state["thrust_magnitude"] = thrust
        
        # Calculate vectored thrust components
        self._calculate_thrust_vector(thrust)
        
        return self.state
    
    def _calculate_thrust_vector(self, thrust: float):
        """Calculate thrust vector components."""
        # Convert angles to radians
        pitch_rad = np.radians(self.state["pitch_angle"])
        yaw_rad = np.radians(self.state["yaw_angle"])
        
        # Calculate direction cosines
        x = np.cos(pitch_rad) * np.cos(yaw_rad)
        y = np.sin(yaw_rad)
        z = -np.sin(pitch_rad) * np.cos(yaw_rad)
        
        # Calculate thrust components
        self.state["vectored_thrust"] = thrust * np.array([x, y, z])
    
    def get_thrust_components(self) -> Tuple[float, float, float]:
        """Get thrust components in x, y, z axes."""
        return tuple(self.state["vectored_thrust"])
    
    def set_mode(self, mode: VectorMode):
        """Set vectoring mode."""
        self.state["mode"] = mode
        if mode == VectorMode.FIXED:
            self.target["pitch"] = 0.0
            self.target["yaw"] = 0.0
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed vectoring diagnostics."""
        return {
            "angles": {
                "pitch": self.state["pitch_angle"],
                "yaw": self.state["yaw_angle"]
            },
            "rates": {
                "pitch": self.state["pitch_rate"],
                "yaw": self.state["yaw_rate"]
            },
            "thrust": {
                "magnitude": self.state["thrust_magnitude"],
                "vector": self.state["vectored_thrust"].tolist()
            },
            "mode": self.state["mode"].value
        }


class VectoredThrustAdapter:
    """Adapter to integrate thrust vectoring with existing engines."""
    
    def __init__(self, vector_control: ThrustVectorControl):
        self.vector_control = vector_control
        self.last_thrust = 0.0
    
    def process_engine_output(self, engine_state: Dict[str, Any], 
                            vector_commands: Dict[str, float], 
                            dt: float) -> Dict[str, Any]:
        """Process engine output and apply thrust vectoring."""
        # Extract thrust from engine state (adapt this based on engine type)
        thrust = engine_state.get("power_output", 0.0) * 1000  # Convert kW to N
        
        # Get vectoring commands
        pitch_cmd = vector_commands.get("pitch", 0.0)
        yaw_cmd = vector_commands.get("yaw", 0.0)
        
        # Update vector control
        vector_state = self.vector_control.update(thrust, pitch_cmd, yaw_cmd, dt)
        
        # Combine engine and vector states
        combined_state = {
            **engine_state,
            "thrust_vector": vector_state["vectored_thrust"],
            "vector_angles": {
                "pitch": vector_state["pitch_angle"],
                "yaw": vector_state["yaw_angle"]
            }
        }
        
        self.last_thrust = thrust
        return combined_state


# Example usage with existing engines
def add_thrust_vectoring(engine: Any) -> VectoredThrustAdapter:
    """Add thrust vectoring to an existing engine."""
    vector_control = ThrustVectorControl(
        VectoringSpecs(
            max_angle=20.0,
            slew_rate=60.0,
            response_time=0.05,
            mechanical_limit=25.0,
            precision=0.1
        )
    )
    return VectoredThrustAdapter(vector_control)