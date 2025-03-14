"""
VTOL parametric airframe model for neuromorphic control systems.

This module provides a parametric model for VTOL UCAV airframes
with neuromorphic sensing and transition flight control.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json


class VTOLAirframe:
    """VTOL airframe model with neuromorphic sensing and transition control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic airframe parameters
            'wingspan': 2.4,  # meters
            'length': 1.8,    # meters
            'wing_area': 1.0, # square meters
            'vertical_thrust_count': 4,  # number of vertical lift motors
            'transition_mechanism': 'tilt_wing',  # tilt_wing, tilt_rotor, or separate_lift
            
            # Mass properties
            'empty_mass': 12.0,  # kg
            'max_takeoff_mass': 18.0,  # kg
            'battery_mass': 4.0,  # kg
            
            # Neuromorphic sensing array
            'sensing_array': {
                'pressure_sensors': {
                    'count': 32,
                    'positions': 'distributed_wing',  # wing surface distribution pattern
                    'sampling_rate': 200,  # Hz
                    'power_per_sensor': 0.02  # Watts
                },
                'flow_sensors': {
                    'count': 16,
                    'positions': 'leading_edge',  # leading edge distribution
                    'sampling_rate': 100,  # Hz
                    'power_per_sensor': 0.05  # Watts
                },
                'vibration_sensors': {
                    'count': 8,
                    'positions': 'structural_nodes',  # key structural points
                    'sampling_rate': 500,  # Hz
                    'power_per_sensor': 0.03  # Watts
                }
            },
            
            # Neuromorphic processing
            'processing_nodes': [
                {
                    'name': 'main_flight_controller',
                    'position': [0.0, 0.0, 0.0],
                    'processing_capacity': 5.0,  # TOPS (Tera Operations Per Second)
                    'power': 4.0,  # Watts
                    'sensor_groups': ['imu', 'gps', 'primary_control']
                },
                {
                    'name': 'transition_controller',
                    'position': [0.1, 0.0, 0.0],
                    'processing_capacity': 2.0,  # TOPS
                    'power': 2.5,  # Watts
                    'sensor_groups': ['motor_control', 'transition_mechanism']
                },
                {
                    'name': 'flow_sensing_processor',
                    'position': [-0.1, 0.0, 0.0],
                    'processing_capacity': 3.0,  # TOPS
                    'power': 3.0,  # Watts
                    'sensor_groups': ['pressure_sensors', 'flow_sensors']
                }
            ],
            
            # Transition flight constraints
            'transition_envelope': {
                'min_transition_speed': 12.0,  # m/s
                'max_transition_speed': 18.0,  # m/s
                'max_pitch_rate': 20.0,  # deg/s
                'max_roll_rate': 30.0,  # deg/s
                'transition_duration': [3.0, 8.0]  # min and max seconds
            },
            
            # Power system
            'power_system': {
                'battery_capacity': 250.0,  # Wh
                'max_discharge_rate': 20.0,  # C
                'nominal_voltage': 22.2,  # V
                'power_distribution_efficiency': 0.92
            }
        }
        
        if config:
            self._update_config(config)
        self._calculate_derived_parameters()
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration with provided values."""
        for key, value in config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
    
    def _calculate_derived_parameters(self) -> None:
        """Calculate derived parameters and validate constraints."""
        # Calculate total sensor power
        self.sensor_power = 0.0
        for sensor_type, specs in self.config['sensing_array'].items():
            self.sensor_power += specs['count'] * specs['power_per_sensor']
        
        # Calculate total processing power
        self.processing_power = sum(node['power'] for node in self.config['processing_nodes'])
        
        # Calculate total neuromorphic system power
        self.total_neuromorphic_power = self.sensor_power + self.processing_power
        
        # Calculate sensor density
        total_sensors = sum(specs['count'] for specs in self.config['sensing_array'].values())
        self.sensor_density = total_sensors / self.config['wing_area']
        
        # Validate neuromorphic constraints
        self._validate_neuromorphic_constraints()
        
        # Calculate transition flight parameters
        self._calculate_transition_parameters()
    
    def _validate_neuromorphic_constraints(self) -> None:
        """Validate neuromorphic system constraints."""
        self.neuromorphic_violations = []
        
        # Check power budget
        max_neuromorphic_power = 15.0  # Maximum allowed for neuromorphic systems
        if self.total_neuromorphic_power > max_neuromorphic_power:
            self.neuromorphic_violations.append(
                f"Power budget exceeded: {self.total_neuromorphic_power:.1f}W > {max_neuromorphic_power:.1f}W"
            )
        
        # Check sensor density constraints
        max_sensor_density = 100.0  # sensors per square meter
        if self.sensor_density > max_sensor_density:
            self.neuromorphic_violations.append(
                f"Sensor density too high: {self.sensor_density:.1f} > {max_sensor_density:.1f} sensors/m²"
            )
        
        # Check processing capacity for sensor data
        total_sensor_data_rate = 0.0
        for sensor_type, specs in self.config['sensing_array'].items():
            # Assume 2 bytes per sample
            data_rate = specs['count'] * specs['sampling_rate'] * 2 / 1000000  # MB/s
            total_sensor_data_rate += data_rate
        
        # Check if processing nodes can handle the data rate
        # Assume 1 TOPS can process 10 MB/s of sensor data (simplified)
        total_processing_capacity = sum(node['processing_capacity'] 
                                       for node in self.config['processing_nodes'])
        max_data_rate = total_processing_capacity * 10  # MB/s
        
        if total_sensor_data_rate > max_data_rate:
            self.neuromorphic_violations.append(
                f"Sensor data rate too high: {total_sensor_data_rate:.2f} MB/s > {max_data_rate:.2f} MB/s"
            )
    
    def _calculate_transition_parameters(self) -> None:
        """Calculate parameters related to transition flight."""
        # Wing loading
        self.wing_loading = self.config['max_takeoff_mass'] / self.config['wing_area']
        
        # Estimate stall speed (simplified)
        # Assume CL_max of 1.5
        rho = 1.225  # air density at sea level
        cl_max = 1.5
        self.stall_speed = np.sqrt((2 * self.wing_loading * 9.81) / (rho * cl_max))
        
        # Check transition envelope
        if self.stall_speed > self.config['transition_envelope']['min_transition_speed']:
            self.neuromorphic_violations.append(
                f"Stall speed ({self.stall_speed:.1f} m/s) exceeds minimum transition speed "
                f"({self.config['transition_envelope']['min_transition_speed']:.1f} m/s)"
            )
    
    def get_transition_control_requirements(self) -> Dict[str, Any]:
        """Get control requirements for transition flight."""
        # Calculate required control rates based on transition envelope
        min_speed = self.config['transition_envelope']['min_transition_speed']
        max_speed = self.config['transition_envelope']['max_transition_speed']
        max_duration = self.config['transition_envelope']['transition_duration'][1]
        
        # Calculate acceleration during transition
        accel = (max_speed - min_speed) / max_duration
        
        # Calculate required control update rate based on dynamics
        # Rule of thumb: control rate should be at least 10x the fastest dynamic
        max_rate = max(self.config['transition_envelope']['max_pitch_rate'],
                      self.config['transition_envelope']['max_roll_rate'])
        min_control_rate = max_rate * 10  # Hz
        
        return {
            'min_control_rate': min_control_rate,
            'acceleration': accel,
            'min_processing_capacity': min_control_rate * 0.05,  # TOPS (simplified)
            'sensor_requirements': {
                'attitude_accuracy': 0.5,  # degrees
                'velocity_accuracy': 0.2,  # m/s
                'position_accuracy': 1.0   # meters
            }
        }
    
    def get_neuromorphic_constraints(self) -> Dict[str, Any]:
        """Get neuromorphic system constraints."""
        return {
            'violations': self.neuromorphic_violations,
            'power_consumption': {
                'sensors': self.sensor_power,
                'processing': self.processing_power,
                'total': self.total_neuromorphic_power
            },
            'sensor_metrics': {
                'total_sensors': sum(specs['count'] for specs in self.config['sensing_array'].values()),
                'sensor_density': self.sensor_density,
                'distributed_sensing_coverage': 0.85  # Percentage of surface covered
            },
            'processing_metrics': {
                'total_capacity': sum(node['processing_capacity'] 
                                     for node in self.config['processing_nodes']),
                'transition_specific_capacity': next(
                    (node['processing_capacity'] for node in self.config['processing_nodes'] 
                     if node['name'] == 'transition_controller'),
                    0.0
                )
            }
        }
    
    def estimate_hover_endurance(self) -> float:
        """
        Estimate hover endurance in minutes.
        
        Returns:
            Estimated hover endurance in minutes
        """
        # Simplified power calculation for hover
        # Assume 150W/kg for VTOL in hover (simplified)
        hover_power = self.config['max_takeoff_mass'] * 150.0  # Watts
        
        # Add neuromorphic system power
        total_power = hover_power + self.total_neuromorphic_power
        
        # Calculate endurance
        battery_capacity = self.config['power_system']['battery_capacity']  # Wh
        efficiency = self.config['power_system']['power_distribution_efficiency']
        
        endurance_hours = (battery_capacity * efficiency) / total_power
        return endurance_hours * 60.0  # Convert to minutes
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load_model(cls, filename: str) -> 'VTOLAirframe':
        """Load model configuration from file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        return cls(config)


def create_default_vtol() -> VTOLAirframe:
    """Create default VTOL airframe model."""
    return VTOLAirframe()


def create_custom_vtol(wingspan: float, length: float, 
                      transition_type: str,
                      sensor_config: Dict[str, Any]) -> VTOLAirframe:
    """
    Create custom VTOL airframe model.
    
    Args:
        wingspan: Wingspan in meters
        length: Length in meters
        transition_type: Type of transition mechanism
        sensor_config: Neuromorphic sensor configuration
        
    Returns:
        Customized VTOL airframe model
    """
    config = {
        'wingspan': wingspan,
        'length': length,
        'wing_area': 0.5 * wingspan * wingspan / 4.0,  # Approximation
        'transition_mechanism': transition_type
    }
    
    if sensor_config:
        if 'sensing_array' in sensor_config:
            config['sensing_array'] = sensor_config['sensing_array']
        if 'processing_nodes' in sensor_config:
            config['processing_nodes'] = sensor_config['processing_nodes']
    
    return VTOLAirframe(config)