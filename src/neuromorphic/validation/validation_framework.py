"""
Simple validation framework for neuromorphic control systems.

This module provides tools to evaluate controller performance
across different scenarios and metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any
import os
import json

# Import test scenarios
from neuromorphic.tests.scenario_tests import (
    TestNormalFlight,
    TestDisturbanceResponse,
    TestEmergencyScenarios
)


class PerformanceMetrics:
    """Calculate performance metrics for controller evaluation."""
    
    @staticmethod
    def stability_score(states: List[Dict[str, np.ndarray]]) -> float:
        """
        Calculate stability score based on orientation variance.
        
        Args:
            states: List of state dictionaries
            
        Returns:
            Stability score (higher is better)
        """
        # Extract orientation data
        orientations = []
        for state in states:
            if 'orientation' in state:
                orientations.append(state['orientation'])
        
        if not orientations:
            return 0.0
        
        # Calculate variance of orientation
        orientations = np.array(orientations)
        variance = np.var(orientations, axis=0).sum()
        
        # Convert to score (lower variance = higher score)
        return 100.0 / (1.0 + 10.0 * variance)
    
    @staticmethod
    def energy_efficiency(controls: List[Dict[str, float]]) -> float:
        """
        Calculate energy efficiency based on control effort.
        
        Args:
            controls: List of control dictionaries
            
        Returns:
            Efficiency score (higher is better)
        """
        if not controls:
            return 0.0
        
        # Calculate average throttle use
        throttle_values = [c.get('throttle', 0.0) for c in controls]
        avg_throttle = sum(throttle_values) / len(throttle_values)
        
        # Calculate control activity (sum of absolute control changes)
        control_activity = 0.0
        for i in range(1, len(controls)):
            for key in controls[i]:
                if key in controls[i-1]:
                    control_activity += abs(controls[i][key] - controls[i-1][key])
        
        # Normalize by number of transitions and controls
        if len(controls) > 1:
            control_activity /= (len(controls) - 1) * len(controls[0])
        
        # Compute efficiency score
        return 100.0 * (1.0 - 0.5 * avg_throttle - 0.5 * control_activity)
    
    @staticmethod
    def safety_score(safety_status: List[Dict[str, Any]]) -> float:
        """
        Calculate safety score based on violations and emergencies.
        
        Args:
            safety_status: List of safety status dictionaries
            
        Returns:
            Safety score (higher is better)
        """
        if not safety_status:
            return 0.0
        
        # Count violations and emergencies
        violation_count = 0
        emergency_count = 0
        
        for status in safety_status:
            if not status.get('is_safe', True):
                violation_count += 1
            if status.get('emergency_mode', False):
                emergency_count += 1
        
        # Calculate safety percentage
        safety_percentage = 100.0 * (1.0 - violation_count / len(safety_status))
        
        # Penalize for emergencies
        emergency_penalty = 20.0 * emergency_count / len(safety_status)
        
        return max(0.0, safety_percentage - emergency_penalty)


class ValidationFramework:
    """Framework for validating neuromorphic controllers."""
    
    def __init__(self, output_dir: str = "/Users/yessine/Oberon/results"):
        """
        Initialize validation framework.
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize test scenarios
        self.normal_flight = TestNormalFlight()
        self.disturbance_response = TestDisturbanceResponse()
        self.emergency_scenarios = TestEmergencyScenarios()
        
        # Set up scenarios
        self.scenarios = {
            'normal_flight': self.normal_flight.normal_flight_scenario,
            'wind_gust': self.disturbance_response.wind_gust_scenario,
            'altitude_loss': self.emergency_scenarios.altitude_loss_scenario
        }
    
    def validate_controller(self, controller_name: str) -> Dict[str, Dict[str, float]]:
        """
        Validate a controller across all scenarios.
        
        Args:
            controller_name: Name of controller to validate
            
        Returns:
            Dictionary of performance metrics by scenario
        """
        results = {}
        
        # Run each scenario
        for scenario_name, scenario_func in self.scenarios.items():
            # Run scenario
            scenario_results = self.normal_flight.run_scenario(scenario_func, controller_name)
            
            # Calculate metrics
            stability = PerformanceMetrics.stability_score(scenario_results['states'])
            efficiency = PerformanceMetrics.energy_efficiency(scenario_results['controls'])
            safety = PerformanceMetrics.safety_score(scenario_results['safety_status'])
            
            # Overall score (weighted average)
            overall = 0.4 * stability + 0.3 * efficiency + 0.3 * safety
            
            # Store results
            results[scenario_name] = {
                'stability': stability,
                'efficiency': efficiency,
                'safety': safety,
                'overall': overall
            }
        
        return results
    
    def validate_all_controllers(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Validate all available controllers.
        
        Returns:
            Dictionary of results by controller and scenario
        """
        controllers = [
            'homeostatic',
            'synfire',
            'reservoir',
            'cpg',
            'bacterial'
        ]
        
        all_results = {}
        for controller in controllers:
            all_results[controller] = self.validate_controller(controller)
        
        return all_results
    
    def save_results(self, results: Dict[str, Dict[str, Dict[str, float]]], filename: str = "validation_results.json"):
        """
        Save validation results to file.
        
        Args:
            results: Validation results
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy values to Python native types
        serializable_results = {}
        for controller, scenarios in results.items():
            serializable_results[controller] = {}
            for scenario, metrics in scenarios.items():
                serializable_results[controller][scenario] = {
                    k: float(v) for k, v in metrics.items()
                }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def plot_results(self, results: Dict[str, Dict[str, Dict[str, float]]], filename: str = "validation_results.png"):
        """
        Plot validation results.
        
        Args:
            results: Validation results
            filename: Output filename
        """
        # Extract overall scores
        controllers = list(results.keys())
        scenarios = list(results[controllers[0]].keys())
        
        scores = np.zeros((len(controllers), len(scenarios)))
        for i, controller in enumerate(controllers):
            for j, scenario in enumerate(scenarios):
                scores[i, j] = results[controller][scenario]['overall']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bar_width = 0.15
        index = np.arange(len(scenarios))
        
        for i, controller in enumerate(controllers):
            plt.bar(index + i * bar_width, scores[i, :], bar_width,
                   label=controller.capitalize())
        
        plt.xlabel('Scenario')
        plt.ylabel('Overall Score')
        plt.title('Controller Performance by Scenario')
        plt.xticks(index + bar_width * (len(controllers) - 1) / 2, [s.replace('_', ' ').title() for s in scenarios])
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()


def run_validation():
    """Run validation on all controllers and generate reports."""
    framework = ValidationFramework()
    results = framework.validate_all_controllers()
    framework.save_results(results)
    framework.plot_results(results)
    return results


if __name__ == "__main__":
    run_validation()