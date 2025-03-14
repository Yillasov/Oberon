"""
Failure mode analysis tools for neuromorphic airframe-controller interactions.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import matplotlib.pyplot as plt
from pathlib import Path


class FailureModeAnalysis:
    """Basic failure mode analysis for neuromorphic control systems."""
    
    def __init__(self, digital_twin, controller):
        """
        Initialize failure mode analysis.
        
        Args:
            digital_twin: Digital twin simulation instance
            controller: Neuromorphic controller to test
        """
        self.digital_twin = digital_twin
        self.controller = controller
        self.failure_modes = self._default_failure_modes()
        self.analysis_results = {}
        
    def _default_failure_modes(self) -> Dict[str, Dict[str, Any]]:
        """Define default failure modes to analyze."""
        return {
            "sensor_drift": {
                "description": "Gradual sensor drift in gyroscope",
                "apply": lambda state, t: self._apply_sensor_drift(state, t, 0.01),
                "duration": 10.0,
                "critical_threshold": 0.5  # rad/s drift
            },
            "actuator_saturation": {
                "description": "Control surface actuator saturation",
                "apply": lambda state, t: self._apply_actuator_saturation(state),
                "duration": 5.0,
                "critical_threshold": 0.8  # 80% of time saturated
            },
            "comm_delay": {
                "description": "Communication delay between sensors and controller",
                "apply": lambda state, t: self._apply_comm_delay(state, 0.1),
                "duration": 8.0,
                "critical_threshold": 0.2  # 200ms delay
            },
            "control_surface_jam": {
                "description": "Aileron control surface jam",
                "apply": lambda state, t: self._apply_control_surface_jam(state, "aileron", 0.2),
                "duration": 7.0,
                "critical_threshold": 0.3  # rad attitude error
            }
        }
    
    def _apply_sensor_drift(self, state: Dict[str, Any], time: float, drift_rate: float) -> Dict[str, Any]:
        """Apply gradual sensor drift to gyroscope readings."""
        if "sensor_outputs" in state and "gyroscope" in state["sensor_outputs"]:
            drift = np.array([drift_rate * time, 0, 0])
            if isinstance(state["sensor_outputs"]["gyroscope"], np.ndarray):
                state["sensor_outputs"]["gyroscope"] += drift
            else:
                state["sensor_outputs"]["gyroscope"] = [
                    state["sensor_outputs"]["gyroscope"][0] + drift[0],
                    state["sensor_outputs"]["gyroscope"][1] + drift[1],
                    state["sensor_outputs"]["gyroscope"][2] + drift[2]
                ]
        return state
    
    def _apply_actuator_saturation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply actuator saturation to control inputs."""
        if "control_inputs" in state:
            for control in ["aileron", "elevator", "rudder"]:
                if control in state["control_inputs"]:
                    # Apply saturation at ±0.3
                    state["control_inputs"][control] = max(-0.3, min(0.3, state["control_inputs"][control]))
        return state
    
    def _apply_comm_delay(self, state: Dict[str, Any], delay: float) -> Dict[str, Any]:
        """Simulate communication delay by using slightly outdated sensor data."""
        # This is a simplified simulation of delay - in a real implementation,
        # you would buffer past states and return an older one
        if not hasattr(self, '_sensor_buffer'):
            self._sensor_buffer = state["sensor_outputs"].copy() if "sensor_outputs" in state else {}
            self._buffer_time = state["time"] if "time" in state else 0
            
        current_time = state["time"] if "time" in state else 0
        
        # Only update buffer if enough time has passed (simulating delay)
        if current_time - self._buffer_time >= delay:
            self._sensor_buffer = state["sensor_outputs"].copy() if "sensor_outputs" in state else {}
            self._buffer_time = current_time
            
        # Use delayed sensor data
        if "sensor_outputs" in state:
            state["sensor_outputs"] = self._sensor_buffer
            
        return state
    
    def _apply_control_surface_jam(self, state: Dict[str, Any], surface: str, position: float) -> Dict[str, Any]:
        """Simulate a jammed control surface."""
        if "control_inputs" in state and surface in state["control_inputs"]:
            state["control_inputs"][surface] = position
        return state
    
    def analyze_failure_mode(self, mode_name: str) -> Dict[str, Any]:
        """
        Analyze a specific failure mode.
        
        Args:
            mode_name: Name of the failure mode to analyze
        """
        if mode_name not in self.failure_modes:
            return {"error": f"Failure mode '{mode_name}' not found"}
            
        failure_mode = self.failure_modes[mode_name]
        print(f"Analyzing failure mode: {mode_name} - {failure_mode['description']}")
        
        # Store original callback to restore later
        original_callbacks = self.digital_twin.callbacks.copy()
        
        # Create data collection arrays
        times = []
        attitudes = []
        control_inputs = []
        
        # Create failure injection callback
        def failure_callback(state):
            # Apply the failure mode
            modified_state = failure_mode["apply"](state.copy(), state["time"])
            
            # For analysis purposes, we don't modify the original state
            # but pass the modified state to the controller
            if hasattr(self.controller, 'process_state'):
                control_outputs = self.controller.process_state(modified_state)
                self.digital_twin.set_control_inputs(control_outputs)
            
            # Collect data for analysis
            times.append(state["time"])
            attitudes.append(state["attitude"].tolist() if isinstance(state["attitude"], np.ndarray) 
                            else state["attitude"])
            control_inputs.append(state["control_inputs"].copy() if "control_inputs" in state else {})
        
        # Reset simulation state
        self.digital_twin.current_state["position"] = np.zeros(3)
        self.digital_twin.current_state["velocity"] = np.array([0, 30, 0])  # Forward velocity
        self.digital_twin.current_state["attitude"] = np.zeros(3)
        self.digital_twin.current_state["angular_velocity"] = np.zeros(3)
        self.digital_twin.current_state["time"] = 0.0
        
        # Clear callbacks and add our failure callback
        self.digital_twin.callbacks = [failure_callback]
        
        # Start simulation if not running
        was_running = self.digital_twin.running
        if not was_running:
            self.digital_twin.start()
        
        # Run for specified duration
        import time as time_module
        time_module.sleep(failure_mode["duration"])
        
        # Restore original state
        if not was_running:
            self.digital_twin.stop()
        self.digital_twin.callbacks = original_callbacks
        
        # Analyze results
        attitudes_array = np.array(attitudes)
        max_attitude_error = np.max(np.abs(attitudes_array))
        
        # Check for control saturation
        saturation_count = 0
        total_samples = len(control_inputs)
        
        for inputs in control_inputs:
            for control, value in inputs.items():
                if abs(value) > 0.9:  # 90% of max as saturation threshold
                    saturation_count += 1
                    break
        
        saturation_percentage = saturation_count / max(1, total_samples)
        
        # Determine if failure is critical
        is_critical = False
        critical_reason = ""
        
        if "critical_threshold" in failure_mode:
            if mode_name == "sensor_drift" and max_attitude_error > failure_mode["critical_threshold"]:
                is_critical = True
                critical_reason = f"Max attitude error {max_attitude_error:.2f} exceeds threshold {failure_mode['critical_threshold']:.2f}"
            elif mode_name == "actuator_saturation" and saturation_percentage > failure_mode["critical_threshold"]:
                is_critical = True
                critical_reason = f"Control saturation {saturation_percentage:.2f} exceeds threshold {failure_mode['critical_threshold']:.2f}"
            elif mode_name == "control_surface_jam" and max_attitude_error > failure_mode["critical_threshold"]:
                is_critical = True
                critical_reason = f"Max attitude error {max_attitude_error:.2f} exceeds threshold {failure_mode['critical_threshold']:.2f}"
        
        # Compile results
        results = {
            "mode": mode_name,
            "description": failure_mode["description"],
            "duration": failure_mode["duration"],
            "max_attitude_error": float(max_attitude_error),
            "control_saturation_percentage": float(saturation_percentage),
            "is_critical": is_critical,
            "critical_reason": critical_reason,
            "data": {
                "times": times,
                "attitudes": attitudes,
                "control_inputs": control_inputs
            }
        }
        
        self.analysis_results[mode_name] = results
        return results
    
    def analyze_all_failure_modes(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all defined failure modes."""
        results = {}
        
        for mode_name in self.failure_modes:
            results[mode_name] = self.analyze_failure_mode(mode_name)
            
        return results
    
    def generate_report(self, output_dir: str) -> str:
        """Generate a failure mode analysis report."""
        if not self.analysis_results:
            return "No analysis results available. Run analyze_failure_mode first."
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary report
        report_file = output_path / "failure_analysis_report.json"
        
        # Create simplified report without full data arrays
        report_data = {}
        for mode, results in self.analysis_results.items():
            report_data[mode] = {k: v for k, v in results.items() if k != 'data'}
            
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Generate plots if matplotlib is available
        try:
            for mode, results in self.analysis_results.items():
                if 'data' in results:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot attitude response
                    attitudes = np.array(results['data']['attitudes'])
                    times = results['data']['times']
                    
                    plt.subplot(2, 1, 1)
                    plt.plot(times, attitudes[:, 0], 'r-', label='Roll')
                    plt.plot(times, attitudes[:, 1], 'g-', label='Pitch')
                    plt.plot(times, attitudes[:, 2], 'b-', label='Yaw')
                    plt.title(f"Failure Mode: {mode} - {results['description']}")
                    plt.ylabel('Attitude (rad)')
                    plt.legend()
                    
                    # Plot control inputs if available
                    plt.subplot(2, 1, 2)
                    control_data = {k: [] for k in ['aileron', 'elevator', 'rudder', 'throttle']}
                    
                    for inputs in results['data']['control_inputs']:
                        for control in control_data:
                            control_data[control].append(inputs.get(control, 0))
                    
                    for control, values in control_data.items():
                        if any(values):  # Only plot if there are non-zero values
                            plt.plot(times[:len(values)], values, label=control)
                    
                    plt.xlabel('Time (s)')
                    plt.ylabel('Control Input')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(output_path / f"{mode}_analysis.png")
                    plt.close()
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
            
        return str(report_file)


def quick_failure_analysis(digital_twin, controller, mode=None):
    """Run a quick failure mode analysis."""
    analyzer = FailureModeAnalysis(digital_twin, controller)
    
    if mode:
        results = analyzer.analyze_failure_mode(mode)
        
        # Print summary
        print(f"\nFailure Mode Analysis: {mode}")
        print(f"Description: {results['description']}")
        print(f"Max Attitude Error: {results['max_attitude_error']:.4f} rad")
        print(f"Control Saturation: {results['control_saturation_percentage']*100:.1f}%")
        print(f"Critical: {'YES - ' + results['critical_reason'] if results['is_critical'] else 'NO'}")
    else:
        results = analyzer.analyze_all_failure_modes()
        
        # Print summary
        print("\nFailure Mode Analysis Summary:")
        for mode, result in results.items():
            status = "CRITICAL" if result.get("is_critical", False) else "ACCEPTABLE"
            print(f"  {mode}: {status}")
            print(f"    Max Attitude Error: {result['max_attitude_error']:.4f} rad")
            print(f"    Control Saturation: {result['control_saturation_percentage']*100:.1f}%")
            if result.get("is_critical", False):
                print(f"    Reason: {result['critical_reason']}")
    
    return results