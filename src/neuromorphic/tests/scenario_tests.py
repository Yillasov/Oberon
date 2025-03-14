"""
Scenario-based tests for neuromorphic control systems.

This module provides realistic test scenarios to validate
the behavior of neuromorphic controllers in common flight situations.
"""
import numpy as np
import unittest
from typing import Dict, List, Tuple, Optional

# Import modules to test
from neuromorphic.control.homeostatic_controller import HomeostaticController
from neuromorphic.control.synfire_controller import SynfireController
from neuromorphic.control.reservoir_controller import ReservoirController
from neuromorphic.control.cpg_controller import CPGController
from neuromorphic.control.bacterial_controller import BacterialController
from neuromorphic.sensors.sensor_integration import SensorIntegration
from neuromorphic.safety.safety_protocols import SafetyController


class ScenarioTestBase(unittest.TestCase):
    """Base class for scenario tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create components
        self.sensor_integration = SensorIntegration()
        self.controllers = {
            'homeostatic': HomeostaticController(),
            'synfire': SynfireController(),
            'reservoir': ReservoirController(),
            'cpg': CPGController(),
            'bacterial': BacterialController()
        }
        self.safety_controller = SafetyController()
        
        # Simulation parameters
        self.dt = 0.1  # Time step
        self.duration = 10.0  # Simulation duration
        
        # Results storage
        self.results = {}
    
    def run_scenario(self, scenario_func, controller_name='homeostatic'):
        """
        Run a specific scenario with the specified controller.
        
        Args:
            scenario_func: Function that generates scenario states
            controller_name: Name of controller to use
            
        Returns:
            Dictionary of simulation results
        """
        controller = self.controllers[controller_name]
        
        # Initialize results
        times = []
        states = []
        controls = []
        safety_status = []
        
        # Run simulation
        time = 0.0
        while time <= self.duration:
            # Get state from scenario
            raw_readings = scenario_func(time)
            
            # Process sensor readings
            state = self.sensor_integration.process_sensors(raw_readings)
            
            # Compute normal controls
            normal_control = controller.compute_control(state, time)
            
            # Apply safety protocols
            safe_control, status = self.safety_controller.apply_safety_protocols(
                state, normal_control, time
            )
            
            # Store results
            times.append(time)
            states.append(state)
            controls.append(safe_control)
            safety_status.append(status)
            
            # Advance time
            time += self.dt
        
        # Compile results
        results = {
            'times': times,
            'states': states,
            'controls': controls,
            'safety_status': safety_status
        }
        
        return results


class TestNormalFlight(ScenarioTestBase):
    """Test normal flight scenarios."""
    
    def normal_flight_scenario(self, time):
        """
        Generate a normal flight scenario.
        
        Args:
            time: Current simulation time
            
        Returns:
            Dictionary of sensor readings
        """
        # Stable flight with small oscillations
        roll = 0.1 * np.sin(0.5 * time)
        pitch = 0.1 * np.cos(0.3 * time)
        yaw = 0.05 * np.sin(0.2 * time)
        
        # Constant position with small drift
        x = 10.0 + 0.1 * time
        y = 20.0 + 0.05 * np.sin(0.1 * time)
        z = 10.0 + 0.1 * np.cos(0.2 * time)
        
        # Create sensor readings
        readings = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'x': x,
            'y': y,
            'z': z
        }
        
        return readings
    
    def test_normal_flight_all_controllers(self):
        """Test all controllers in normal flight scenario."""
        for controller_name in self.controllers:
            with self.subTest(controller=controller_name):
                results = self.run_scenario(self.normal_flight_scenario, controller_name)
                
                # Check that flight remains stable
                for i, status in enumerate(results['safety_status']):
                    self.assertTrue(status['is_safe'], 
                                   f"Controller {controller_name} failed safety at t={results['times'][i]}")
                
                # Check that controls remain within bounds
                for controls in results['controls']:
                    for control_name, value in controls.items():
                        self.assertGreaterEqual(value, -1.0)
                        self.assertLessEqual(value, 1.0)


class TestDisturbanceResponse(ScenarioTestBase):
    """Test response to disturbances."""
    
    def wind_gust_scenario(self, time):
        """
        Generate a wind gust scenario.
        
        Args:
            time: Current simulation time
            
        Returns:
            Dictionary of sensor readings
        """
        # Normal flight
        roll = 0.1 * np.sin(0.5 * time)
        pitch = 0.1 * np.cos(0.3 * time)
        yaw = 0.05 * np.sin(0.2 * time)
        
        # Position with normal drift
        x = 10.0 + 0.1 * time
        y = 20.0 + 0.05 * np.sin(0.1 * time)
        z = 10.0 + 0.1 * np.cos(0.2 * time)
        
        # Add wind gust at t=5.0
        if 5.0 <= time <= 6.0:
            # Strong roll disturbance
            roll += 0.4 * (time - 5.0) * (6.0 - time)
            # Altitude drop
            z -= 1.0 * (time - 5.0) * (6.0 - time)
        
        # Create sensor readings
        readings = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'x': x,
            'y': y,
            'z': z
        }
        
        return readings
    
    def test_wind_gust_response(self):
        """Test controller response to wind gust."""
        for controller_name in self.controllers:
            with self.subTest(controller=controller_name):
                results = self.run_scenario(self.wind_gust_scenario, controller_name)
                
                # Find the gust period in results
                gust_indices = [i for i, t in enumerate(results['times']) if 5.0 <= t <= 6.0]
                
                # Check that controllers respond to the disturbance
                for i in gust_indices:
                    controls = results['controls'][i]
                    # Should counter the roll disturbance
                    self.assertLess(controls['aileron'], 0, 
                                   f"Controller {controller_name} failed to counter roll at t={results['times'][i]}")
                    
                # Check recovery after gust
                recovery_indices = [i for i, t in enumerate(results['times']) if t > 7.0]
                if recovery_indices:
                    # Get the last few states to check stability
                    final_states = [results['states'][i] for i in recovery_indices[-3:]]
                    
                    # Check that orientation returned to stable
                    for state in final_states:
                        if 'orientation' in state:
                            for angle in state['orientation']:
                                self.assertLess(abs(angle), 0.3, 
                                              f"Controller {controller_name} failed to recover stability")


class TestEmergencyScenarios(ScenarioTestBase):
    """Test emergency scenarios."""
    
    def altitude_loss_scenario(self, time):
        """
        Generate an altitude loss scenario.
        
        Args:
            time: Current simulation time
            
        Returns:
            Dictionary of sensor readings
        """
        # Normal flight initially
        roll = 0.1 * np.sin(0.5 * time)
        pitch = 0.1 * np.cos(0.3 * time)
        yaw = 0.05 * np.sin(0.2 * time)
        
        # Position with normal drift
        x = 10.0 + 0.1 * time
        y = 20.0 + 0.05 * np.sin(0.1 * time)
        
        # Altitude loss starting at t=3.0
        if time < 3.0:
            z = 10.0
        else:
            z = max(1.0, 10.0 - 2.0 * (time - 3.0))
        
        # Create sensor readings
        readings = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'x': x,
            'y': y,
            'z': z
        }
        
        return readings
    
    def test_altitude_emergency(self):
        """Test emergency response to altitude loss."""
        for controller_name in self.controllers:
            with self.subTest(controller=controller_name):
                results = self.run_scenario(self.altitude_loss_scenario, controller_name)
                
                # Find when altitude gets critically low
                emergency_indices = []
                for i, state in enumerate(results['states']):
                    if 'position' in state and state['position'][2] < 3.0:
                        emergency_indices.append(i)
                
                # Check that emergency mode is activated
                if emergency_indices:
                    # At least one of the later states should trigger emergency mode
                    emergency_triggered = False
                    for i in emergency_indices[-3:]:
                        if results['safety_status'][i]['emergency_mode']:
                            emergency_triggered = True
                            break
                    
                    self.assertTrue(emergency_triggered, 
                                   f"Controller {controller_name} failed to trigger emergency mode")
                    
                    # Check emergency response
                    for i in emergency_indices:
                        if results['safety_status'][i]['emergency_mode']:
                            controls = results['controls'][i]
                            # Should increase throttle to regain altitude
                            self.assertGreater(controls['throttle'], 0.7, 
                                             f"Emergency response failed to increase throttle at t={results['times'][i]}")


def run_scenario_tests():
    """Run all scenario tests."""
    unittest.main()


if __name__ == "__main__":
    run_scenario_tests()