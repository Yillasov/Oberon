"""
Hardware integration examples for neuromorphic control systems.

This module demonstrates how to integrate the neuromorphic control
system with specific hardware platforms.
"""
import numpy as np
import time
from typing import Dict, Any, Optional

from neuromorphic.hardware.hardware_abstraction import create_hal, HardwareAbstractionLayer
from neuromorphic.control.homeostatic_controller import HomeostaticController
from neuromorphic.safety.safety_protocols import SafetyController


def custom_sensor_mapping(sensor_readings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Custom mapping from sensor readings to controller state.
    
    Args:
        sensor_readings: Raw sensor readings
        
    Returns:
        State dictionary for controllers
    """
    state = {}
    
    # Custom orientation calculation from accelerometer and gyroscope
    if 'accelerometer' in sensor_readings and 'gyroscope' in sensor_readings:
        # Simple sensor fusion (in practice, would use a proper filter)
        accel = sensor_readings['accelerometer']
        gyro = sensor_readings['gyroscope']
        
        # Estimate orientation (very simplified)
        roll = np.arctan2(accel[1], accel[2])
        pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        yaw = gyro[2]  # Just use gyro for yaw
        
        state['orientation'] = np.array([roll, pitch, yaw])
    
    # Extract altitude from barometer
    if 'barometer' in sensor_readings:
        pressure = sensor_readings['barometer'][0]
        # Simple pressure to altitude conversion
        altitude = 44330.0 * (1.0 - (pressure / 101325.0)**(1.0/5.255))
        
        # Use GPS for horizontal position if available
        if 'gps' in sensor_readings:
            state['position'] = np.array([
                sensor_readings['gps'][0],
                sensor_readings['gps'][1],
                altitude
            ])
        else:
            state['position'] = np.array([0.0, 0.0, altitude])
    
    return state


def custom_hardware_example():
    """Example of using custom hardware configuration."""
    # Create custom HAL configuration
    config = {
        'sensors': {
            'scaling': {
                'accelerometer': 0.8,  # Reduce accelerometer sensitivity
                'gyroscope': 1.2       # Increase gyroscope sensitivity
            }
        },
        'actuators': {
            'scaling': {
                'motors': 0.9  # Limit motor output to 90%
            }
        },
        'sensor_to_state_func': custom_sensor_mapping
    }
    
    # Create HAL with custom config
    hal = create_hal(config)
    controller = HomeostaticController()
    safety = SafetyController()
    
    print("Starting custom hardware example...")
    
    # Run for 5 iterations
    for i in range(5):
        # Read state with custom mapping
        state = hal.read_state()
        
        # Compute control
        control = controller.compute_control(state, time.time())
        safe_control, _ = safety.apply_safety_protocols(state, control, time.time())
        
        # Send to hardware with custom scaling
        hal.write_control(safe_control)
        
        # Get actuator status
        status = hal.actuators.get_actuator_status()
        
        # Print status
        print(f"Iteration {i+1}:")
        print(f"  Battery: {status['voltages']['main']:.2f}V")
        print(f"  Motor Temps: " + 
              ", ".join([f"{k}={v:.1f}°C" for k, v in status['temperatures'].items()]))
        
        time.sleep(0.1)
    
    print("Custom hardware example completed.")


def run_hardware_examples():
    """Run all hardware examples."""
    custom_hardware_example()


if __name__ == "__main__":
    run_hardware_examples()