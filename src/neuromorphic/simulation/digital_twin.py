"""
Digital Twin simulation environment for neuromorphic airframe models.
"""
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable
import threading
import json


class DigitalTwin:
    """Real-time digital twin simulation for neuromorphic airframe models."""
    
    def __init__(self, airframe_model: Any, update_rate: float = 50.0):
        """
        Initialize digital twin simulation.
        
        Args:
            airframe_model: Airframe model instance
            update_rate: Simulation update rate in Hz
        """
        self.airframe_model = airframe_model
        self.update_rate = update_rate
        self.time_step = 1.0 / update_rate
        
        # Simulation state
        self.current_state = {
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "attitude": np.zeros(3),
            "angular_velocity": np.zeros(3),
            "time": 0.0,
            "control_inputs": {},
            "sensor_outputs": {}
        }
        
        # Simulation flags
        self.running = False
        self.sim_thread = None
        self.callbacks = []
        self.physics_enabled = False
        
        # Performance metrics
        self.execution_times = []
        self.real_time_factor = 1.0
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback function to receive simulation updates."""
        self.callbacks.append(callback)
    
    def set_control_inputs(self, control_inputs: Dict[str, float]) -> None:
        """Set control inputs for the simulation."""
        self.current_state["control_inputs"] = control_inputs
    
    def _update_simulation(self) -> None:
        """Update simulation state."""
        # Simple physics update (placeholder for more complex dynamics)
        dt = self.time_step
        
        # Get control inputs
        control_inputs = self.current_state["control_inputs"]
        
        # Update position based on velocity
        self.current_state["position"] += self.current_state["velocity"] * dt
        
        # Update attitude based on angular velocity
        self.current_state["attitude"] += self.current_state["angular_velocity"] * dt
        
        # Apply control surface effects if no external physics is enabled
        if not self.physics_enabled and control_inputs:
            # Simple control surface effects
            aileron = control_inputs.get("aileron", 0.0)
            elevator = control_inputs.get("elevator", 0.0)
            rudder = control_inputs.get("rudder", 0.0)
            throttle = control_inputs.get("throttle", 0.0)
            
            # Apply control effects
            self.current_state["angular_velocity"][0] += aileron * 0.5  # Roll
            self.current_state["angular_velocity"][1] += elevator * 0.3  # Pitch
            self.current_state["angular_velocity"][2] += rudder * 0.2    # Yaw
            
            # Apply throttle
            speed = np.linalg.norm(self.current_state["velocity"])
            if speed < 30.0 * throttle:
                # Simplified acceleration
                direction = np.array([0.0, 1.0, 0.0])  # Forward
                self.current_state["velocity"] += direction * throttle * 0.5 * dt
        
        # Update time
        self.current_state["time"] += dt
        
        # Generate sensor outputs (simplified)
        self.current_state["sensor_outputs"] = {
            "accelerometer": np.random.normal(0, 0.1, 3),
            "gyroscope": self.current_state["angular_velocity"] + np.random.normal(0, 0.01, 3),
            "pressure": 101325 - 12 * self.current_state["position"][2],
            "temperature": 25.0 + np.random.normal(0, 0.1)
        }
    
    def _simulation_loop(self) -> None:
        """Main simulation loop."""
        last_time = time.time()
        
        while self.running:
            start_time = time.time()
            
            # Update simulation
            self._update_simulation()
            
            # Call registered callbacks
            for callback in self.callbacks:
                callback(self.current_state)
            
            # Calculate execution time
            exec_time = time.time() - start_time
            self.execution_times.append(exec_time)
            
            # Limit to real-time if possible
            elapsed = time.time() - last_time
            sleep_time = max(0, self.time_step - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Update real-time factor
            actual_step = time.time() - last_time
            self.real_time_factor = self.time_step / max(0.001, actual_step)
            last_time = time.time()
            
            # Trim execution time history
            if len(self.execution_times) > 100:
                self.execution_times = self.execution_times[-100:]
    
    def start(self) -> None:
        """Start the simulation."""
        if not self.running:
            self.running = True
            self.sim_thread = threading.Thread(target=self._simulation_loop)
            self.sim_thread.daemon = True
            self.sim_thread.start()
    
    def stop(self) -> None:
        """Stop the simulation."""
        self.running = False
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get simulation performance metrics."""
        if not self.execution_times:
            return {"avg_exec_time": 0, "real_time_factor": 0, "cpu_load": 0}
        
        avg_exec_time = sum(self.execution_times) / len(self.execution_times)
        cpu_load = avg_exec_time / self.time_step
        
        return {
            "avg_exec_time": avg_exec_time,
            "real_time_factor": self.real_time_factor,
            "cpu_load": cpu_load,
            "update_rate": self.update_rate
        }


class NeuromorphicController:
    """Simple neuromorphic controller for digital twin simulation."""
    
    def __init__(self, control_rate: float = 100.0):
        """Initialize neuromorphic controller."""
        self.control_rate = control_rate
        self.last_update = 0.0
        self.control_outputs = {}
    
    def process_state(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Process simulation state and generate control outputs."""
        current_time = state["time"]
        
        # Only update at specified control rate
        if current_time - self.last_update < 1.0 / self.control_rate:
            return self.control_outputs
        
        # Simple PID-like control (placeholder for neuromorphic algorithms)
        attitude_error = -state["attitude"]  # Try to maintain level flight
        
        self.control_outputs = {
            "aileron": attitude_error[0] * 0.5,
            "elevator": attitude_error[1] * 0.5,
            "rudder": attitude_error[2] * 0.5,
            "throttle": 0.5
        }
        
        self.last_update = current_time
        return self.control_outputs


def run_simple_simulation(airframe_model: Any, duration: float = 10.0) -> Dict[str, Any]:
    """Run a simple digital twin simulation."""
    # Create digital twin and controller
    twin = DigitalTwin(airframe_model)
    controller = NeuromorphicController()
    
    # Setup control feedback loop
    def control_callback(state):
        control_inputs = controller.process_state(state)
        twin.set_control_inputs(control_inputs)
    
    # Register callback
    twin.register_callback(control_callback)
    
    # Run simulation
    twin.start()
    time.sleep(duration)
    twin.stop()
    
    # Return results
    return {
        "final_state": twin.current_state,
        "performance": twin.get_performance_metrics()
    }