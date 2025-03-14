"""
Test suite for neuromorphic control systems.

This module provides a comprehensive set of tests for validating
the functionality of neuromorphic controllers, sensor integration,
and safety protocols.
"""
import numpy as np
import unittest
from typing import Dict, List, Callable

# Import modules to test
from neuromorphic.control.homeostatic_controller import HomeostaticController, HomeostaticRegulator
from neuromorphic.control.synfire_controller import SynfireController, SynfireChain
from neuromorphic.control.reservoir_controller import ReservoirController, SimpleReservoir
from neuromorphic.control.cpg_controller import CPGController, CPGNetwork
from neuromorphic.control.bacterial_controller import BacterialController, ChemotaxisCell
from neuromorphic.sensors.sensor_integration import SensorIntegration, SensorPreprocessor
from neuromorphic.safety.safety_protocols import SafetyController, SafetyMonitor


class TestHomeostaticRegulator(unittest.TestCase):
    """Test cases for the homeostatic regulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.regulator = HomeostaticRegulator(setpoint=10.0, gain=0.5)
    
    def test_initialization(self):
        """Test regulator initialization."""
        self.assertEqual(self.regulator.setpoint, 10.0)
        self.assertEqual(self.regulator.gain, 0.5)
    
    def test_update(self):
        """Test regulator update function."""
        # Test response to error
        response = self.regulator.update(8.0)
        self.assertEqual(response, 1.0)  # (10.0 - 8.0) * 0.5 = 1.0
        
        # Test response to no error
        response = self.regulator.update(10.0)
        self.assertEqual(response, 0.0)  # (10.0 - 10.0) * 0.5 = 0.0
        
        # Test response to negative error
        response = self.regulator.update(12.0)
        self.assertEqual(response, -1.0)  # (10.0 - 12.0) * 0.5 = -1.0
    
    def test_adaptation(self):
        """Test regulator adaptation."""
        # Create oscillating responses
        self.regulator.response_history = [1.0, -1.0, 1.0]
        
        # Update with a value that causes oscillation
        initial_gain = self.regulator.gain
        self.regulator.update(8.0)
        
        # Gain should decrease due to oscillation
        self.assertLess(self.regulator.gain, initial_gain)
        
        # Create consistent responses
        self.regulator.response_history = [1.0, 1.0, 1.0]
        
        # Update with a value that causes consistent response
        initial_gain = self.regulator.gain
        self.regulator.update(8.0)
        
        # Gain should increase due to consistent response
        self.assertGreater(self.regulator.gain, initial_gain)


class TestHomeostaticController(unittest.TestCase):
    """Test cases for the homeostatic controller."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = HomeostaticController()
        
        # Create a simple state
        self.state = {
            'position': np.array([0.0, 0.0, 5.0]),
            'orientation': np.array([0.1, -0.2, 0.3])
        }
    
    def test_compute_control(self):
        """Test control computation."""
        controls = self.controller.compute_control(self.state, 0.0)
        
        # Check that all expected controls are present
        self.assertIn('aileron', controls)
        self.assertIn('elevator', controls)
        self.assertIn('rudder', controls)
        self.assertIn('throttle', controls)
        
        # Check that controls are within bounds
        for control_name, value in controls.items():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)
    
    def test_energy_conservation(self):
        """Test energy conservation mechanism."""
        # Initial energy level
        initial_energy = self.controller.energy_level
        
        # Compute control multiple times to simulate time passing
        for i in range(10):
            self.controller.compute_control(self.state, float(i))
        
        # Energy should change
        self.assertNotEqual(self.controller.energy_level, initial_energy)
        
        # Energy should stay within bounds
        self.assertGreaterEqual(self.controller.energy_level, 0.1)
        self.assertLessEqual(self.controller.energy_level, 1.0)


class TestSensorPreprocessor(unittest.TestCase):
    """Test cases for the sensor preprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = SensorPreprocessor(smoothing_factor=0.8)
    
    def test_smoothing(self):
        """Test sensor smoothing."""
        # First value should pass through unchanged
        value, is_spike = self.preprocessor.process(10.0)
        self.assertEqual(value, 10.0)
        self.assertFalse(is_spike)
        
        # Second value should be smoothed
        value, is_spike = self.preprocessor.process(15.0)
        expected = 0.8 * 10.0 + 0.2 * 15.0
        self.assertAlmostEqual(value, expected)
        
        # Large jump should be detected as spike
        value, is_spike = self.preprocessor.process(30.0)
        self.assertTrue(is_spike)


class TestSafetyMonitor(unittest.TestCase):
    """Test cases for the safety monitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = SafetyMonitor()
        
        # Create a safe state
        self.safe_state = {
            'position': np.array([0.0, 0.0, 10.0]),
            'orientation': np.array([0.1, 0.1, 0.1]),
            'velocity': np.array([1.0, 1.0, 1.0])
        }
        
        # Create an unsafe state
        self.unsafe_state = {
            'position': np.array([0.0, 0.0, 1.0]),  # Too low
            'orientation': np.array([0.6, 0.1, 0.1]),  # Roll too high
            'velocity': np.array([1.0, 1.0, 1.0])
        }
    
    def test_safe_state(self):
        """Test safety check with safe state."""
        is_safe, violations = self.monitor.check_safety(self.safe_state, 0.0)
        self.assertTrue(is_safe)
        self.assertEqual(len(violations), 0)
        self.assertFalse(self.monitor.warning_mode)
        self.assertFalse(self.monitor.emergency_mode)
    
    def test_unsafe_state(self):
        """Test safety check with unsafe state."""
        is_safe, violations = self.monitor.check_safety(self.unsafe_state, 0.0)
        self.assertFalse(is_safe)
        self.assertGreater(len(violations), 0)
        self.assertTrue(self.monitor.warning_mode)
        
        # Emergency mode should not be triggered yet
        self.assertFalse(self.monitor.emergency_mode)
        
        # Check multiple times to trigger emergency
        for i in range(5):
            self.monitor.check_safety(self.unsafe_state, float(i))
        
        # Now emergency mode should be triggered
        self.assertTrue(self.monitor.emergency_mode)


class TestSafetyController(unittest.TestCase):
    """Test cases for the safety controller."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = SafetyController()
        
        # Create a safe state
        self.safe_state = {
            'position': np.array([0.0, 0.0, 10.0]),
            'orientation': np.array([0.1, 0.1, 0.1])
        }
        
        # Create an unsafe state
        self.unsafe_state = {
            'position': np.array([0.0, 0.0, 1.0]),  # Too low
            'orientation': np.array([0.6, 0.1, 0.1])  # Roll too high
        }
        
        # Create normal controls
        self.normal_controls = {
            'aileron': 0.2,
            'elevator': -0.3,
            'rudder': 0.1,
            'throttle': 0.5
        }
    
    def test_safe_operation(self):
        """Test safety controller with safe state."""
        controls, status = self.controller.apply_safety_protocols(
            self.safe_state, self.normal_controls, 0.0
        )
        
        # Controls should be unchanged
        self.assertEqual(controls, self.normal_controls)
        
        # Status should indicate safe operation
        self.assertTrue(status['is_safe'])
        self.assertFalse(status['warning_mode'])
        self.assertFalse(status['emergency_mode'])
    
    def test_emergency_response(self):
        """Test safety controller emergency response."""
        # Trigger emergency mode
        for i in range(5):
            self.controller.apply_safety_protocols(
                self.unsafe_state, self.normal_controls, float(i)
            )
        
        # Now get emergency controls
        controls, status = self.controller.apply_safety_protocols(
            self.unsafe_state, self.normal_controls, 5.0
        )
        
        # Controls should be different from normal
        self.assertNotEqual(controls, self.normal_controls)
        
        # Status should indicate emergency
        self.assertFalse(status['is_safe'])
        self.assertTrue(status['warning_mode'])
        self.assertTrue(status['emergency_mode'])
        
        # Emergency controls should try to correct the problem
        self.assertLess(controls['aileron'], 0)  # Should counter positive roll
        self.assertGreater(controls['throttle'], 0.5)  # Should increase altitude


class TestIntegration(unittest.TestCase):
    """Integration tests for the neuromorphic control system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create components
        self.sensor_integration = SensorIntegration()
        self.homeostatic_controller = HomeostaticController()
        self.safety_controller = SafetyController()
        
        # Create raw sensor readings
        self.raw_readings = {
            'roll': 0.1,
            'pitch': -0.2,
            'yaw': 0.3,
            'x': 10.0,
            'y': 20.0,
            'z': 5.0
        }
    
    def test_control_pipeline(self):
        """Test the full control pipeline."""
        # Process sensor readings
        state = self.sensor_integration.process_sensors(self.raw_readings)
        
        # Compute normal controls
        normal_controls = self.homeostatic_controller.compute_control(state, 0.0)
        
        # Apply safety protocols
        safe_controls, status = self.safety_controller.apply_safety_protocols(
            state, normal_controls, 0.0
        )
        
        # Check that pipeline produces valid controls
        self.assertIsInstance(safe_controls, dict)
        for control_name, value in safe_controls.items():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()