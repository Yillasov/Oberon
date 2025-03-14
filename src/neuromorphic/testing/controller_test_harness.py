"""
Test harness for neuromorphic controllers with simulated airframes.
"""
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable
import json
import os
from pathlib import Path


class ControllerTestHarness:
    """Basic test harness for neuromorphic controllers."""
    
    def __init__(self, digital_twin, controller, test_scenarios=None):
        """
        Initialize controller test harness.
        
        Args:
            digital_twin: Digital twin simulation instance
            controller: Neuromorphic controller to test
            test_scenarios: List of test scenarios to run
        """
        self.digital_twin = digital_twin
        self.controller = controller
        self.test_scenarios = test_scenarios or self._default_scenarios()
        self.results = {}
        self.current_scenario = None
        self.test_data = []
        
    def _default_scenarios(self) -> List[Dict[str, Any]]:
        """Create default test scenarios."""
        return [
            {
                "name": "straight_level",
                "duration": 10.0,
                "initial_conditions": {
                    "position": [0, 0, 100],
                    "velocity": [0, 30, 0],
                    "attitude": [0, 0, 0],
                    "angular_velocity": [0, 0, 0]
                },
                "success_criteria": {
                    "max_attitude_deviation": 0.1,
                    "altitude_maintained": True
                }
            },
            {
                "name": "roll_response",
                "duration": 5.0,
                "initial_conditions": {
                    "position": [0, 0, 100],
                    "velocity": [0, 30, 0],
                    "attitude": [0, 0, 0],
                    "angular_velocity": [0, 0, 0]
                },
                "commands": [
                    {"time": 1.0, "command": {"aileron": 0.5, "elevator": 0, "rudder": 0, "throttle": 0.5}},
                    {"time": 3.0, "command": {"aileron": 0, "elevator": 0, "rudder": 0, "throttle": 0.5}}
                ],
                "success_criteria": {
                    "roll_response_time": 1.0,
                    "roll_overshoot": 0.2
                }
            }
        ]
    
    def _data_collection_callback(self, state: Dict[str, Any]) -> None:
        """Collect data during test execution."""
        if self.current_scenario:
            # Store relevant state data
            data_point = {
                "time": state["time"],
                "position": state["position"].tolist() if isinstance(state["position"], np.ndarray) else state["position"],
                "velocity": state["velocity"].tolist() if isinstance(state["velocity"], np.ndarray) else state["velocity"],
                "attitude": state["attitude"].tolist() if isinstance(state["attitude"], np.ndarray) else state["attitude"],
                "angular_velocity": state["angular_velocity"].tolist() if isinstance(state["angular_velocity"], np.ndarray) else state["angular_velocity"],
                "control_inputs": state["control_inputs"]
            }
            self.test_data.append(data_point)
            
            # Apply scenario commands if needed
            if "commands" in self.current_scenario:
                for cmd in self.current_scenario["commands"]:
                    if abs(state["time"] - cmd["time"]) < self.digital_twin.time_step:
                        self.digital_twin.set_control_inputs(cmd["command"])
    
    def run_test(self, scenario_name=None) -> Dict[str, Any]:
        """
        Run a specific test scenario or all scenarios.
        
        Args:
            scenario_name: Name of scenario to run, or None to run all
        """
        if scenario_name:
            scenarios = [s for s in self.test_scenarios if s["name"] == scenario_name]
            if not scenarios:
                return {"error": f"Scenario '{scenario_name}' not found"}
        else:
            scenarios = self.test_scenarios
        
        results = {}
        
        for scenario in scenarios:
            print(f"Running test scenario: {scenario['name']}")
            
            # Reset test data
            self.test_data = []
            self.current_scenario = scenario
            
            # Register data collection callback
            self.digital_twin.register_callback(self._data_collection_callback)
            
            # Set initial conditions
            initial = scenario["initial_conditions"]
            self.digital_twin.current_state["position"] = np.array(initial["position"])
            self.digital_twin.current_state["velocity"] = np.array(initial["velocity"])
            self.digital_twin.current_state["attitude"] = np.array(initial["attitude"])
            self.digital_twin.current_state["angular_velocity"] = np.array(initial["angular_velocity"])
            self.digital_twin.current_state["time"] = 0.0
            
            # Start simulation if not running
            was_running = self.digital_twin.running
            if not was_running:
                self.digital_twin.start()
            
            # Connect controller to digital twin
            if hasattr(self.controller, 'process_state'):
                def controller_callback(state):
                    control_inputs = self.controller.process_state(state)
                    # Only set control inputs if not overridden by scenario commands
                    if not any(abs(state["time"] - cmd["time"]) < self.digital_twin.time_step 
                              for cmd in scenario.get("commands", [])):
                        self.digital_twin.set_control_inputs(control_inputs)
                
                self.digital_twin.register_callback(controller_callback)
            
            # Run for scenario duration
            time.sleep(scenario["duration"])
            
            # Evaluate results
            scenario_results = self._evaluate_scenario(scenario)
            results[scenario["name"]] = scenario_results
            
            # Stop simulation if it wasn't running before
            if not was_running:
                self.digital_twin.stop()
            
            self.current_scenario = None
            
        self.results = results
        return results
    
    def _evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate test results against success criteria."""
        if not self.test_data:
            return {"success": False, "reason": "No test data collected"}
        
        criteria = scenario.get("success_criteria", {})
        results = {"success": True, "metrics": {}}
        
        # Extract time series data
        times = [d["time"] for d in self.test_data]
        attitudes = np.array([d["attitude"] for d in self.test_data])
        positions = np.array([d["position"] for d in self.test_data])
        
        # Calculate metrics
        if "max_attitude_deviation" in criteria:
            max_roll = np.max(np.abs(attitudes[:, 0]))
            max_pitch = np.max(np.abs(attitudes[:, 1]))
            max_yaw = np.max(np.abs(attitudes[:, 2]))
            max_deviation = max(max_roll, max_pitch, max_yaw)
            
            results["metrics"]["max_attitude_deviation"] = max_deviation
            if max_deviation > criteria["max_attitude_deviation"]:
                results["success"] = False
                results["reason"] = f"Attitude deviation {max_deviation} exceeds limit {criteria['max_attitude_deviation']}"
        
        if "altitude_maintained" in criteria and criteria["altitude_maintained"]:
            altitude_change = np.max(positions[:, 2]) - np.min(positions[:, 2])
            results["metrics"]["altitude_change"] = altitude_change
            
            if altitude_change > 10.0:  # 10 meter threshold
                results["success"] = False
                results["reason"] = f"Altitude not maintained, changed by {altitude_change}m"
        
        if "roll_response_time" in criteria:
            # Find when roll command was issued
            cmd_times = [cmd["time"] for cmd in scenario.get("commands", []) 
                        if "aileron" in cmd["command"] and abs(cmd["command"]["aileron"]) > 0.1]
            
            if cmd_times:
                cmd_time = cmd_times[0]
                # Find when roll rate reached 50% of max
                roll_rates = np.array([d["angular_velocity"][0] for d in self.test_data])
                max_roll_rate = np.max(np.abs(roll_rates))
                
                response_indices = np.where(np.abs(roll_rates) > 0.5 * max_roll_rate)[0]
                if len(response_indices) > 0:
                    response_idx = response_indices[0]
                    response_time = times[response_idx] - cmd_time
                    
                    results["metrics"]["roll_response_time"] = response_time
                    if response_time > criteria["roll_response_time"]:
                        results["success"] = False
                        results["reason"] = f"Roll response time {response_time}s exceeds limit {criteria['roll_response_time']}s"
        
        return results
    
    def save_results(self, output_dir: str) -> str:
        """Save test results and data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary results
        summary_file = output_path / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save detailed test data
        data_file = output_path / "test_data.json"
        with open(data_file, 'w') as f:
            json.dump(self.test_data, f, indent=2)
        
        return str(summary_file)


def quick_controller_test(digital_twin, controller, scenario_name=None):
    """Run a quick controller test with default settings."""
    harness = ControllerTestHarness(digital_twin, controller)
    results = harness.run_test(scenario_name)
    
    # Print summary
    print("\nTest Results Summary:")
    for name, result in results.items():
        status = "PASSED" if result.get("success", False) else "FAILED"
        reason = result.get("reason", "")
        print(f"  {name}: {status} {reason}")
        
        if "metrics" in result:
            print("  Metrics:")
            for metric, value in result["metrics"].items():
                print(f"    {metric}: {value}")
    
    return results