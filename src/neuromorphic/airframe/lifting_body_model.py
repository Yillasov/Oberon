"""
Lifting Body parametric airframe model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import json


class LiftingBodyAirframe:
    """Lifting body airframe with neuromorphic flow control system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'length': 14.5,        # meters
            'width': 8.2,          # meters
            'height': 2.8,         # meters
            'planform_area': 65.0, # square meters
            'volume': 85.0,        # cubic meters
            
            # Aerodynamic features
            'aero_features': {
                'body_camber': 0.08,       # body camber ratio
                'nose_bluntness': 0.35,    # 0-1 scale (0=sharp, 1=blunt)
                'side_strakes': True,      # side strakes for vortex control
                'control_surfaces': {
                    'elevons': True,
                    'body_flaps': True,
                    'rudders': True,
                    'active_flow_ports': 24  # number of active flow control ports
                }
            },
            
            # Neuromorphic flow control system
            'neuromorphic_system': {
                'pressure_sensors': 86,
                'temperature_sensors': 42,
                'flow_actuators': 36,
                'processing_nodes': 8,
                'update_rate': 250.0,  # Hz
                'power_consumption': 48.0,  # Watts
                'thermal_management': True
            },
            
            # Neuromorphic constraints
            'neuromorphic_constraints': {
                'min_sensor_density': 1.2,  # sensors per square meter
                'max_response_time': 0.012,  # seconds
                'max_operating_temp': 95.0,  # Celsius
                'min_actuator_coverage': 0.6,  # fraction of critical areas
                'max_power_density': 0.8  # W/cm²
            }
        }
        
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        self._calculate_derived_parameters()
        self._validate_constraints()
    
    def _calculate_derived_parameters(self) -> None:
        """Calculate derived parameters."""
        # Calculate wetted area (approximate)
        self.wetted_area = self.config['planform_area'] * 2.1  # Factor for body curvature
        
        # Calculate fineness ratio
        self.fineness_ratio = self.config['length'] / np.sqrt(4 * self.config['planform_area'] / np.pi)
        
        # Calculate volumetric efficiency
        self.volumetric_efficiency = self.config['volume'] / (self.config['length'] * 
                                                             self.config['width'] * 
                                                             self.config['height'])
        
        # Calculate sensor density
        total_sensors = (self.config['neuromorphic_system']['pressure_sensors'] + 
                         self.config['neuromorphic_system']['temperature_sensors'])
        self.sensor_density = total_sensors / self.wetted_area
        
        # Calculate actuator coverage
        # Assume critical areas are 40% of wetted area
        critical_area = self.wetted_area * 0.4
        self.actuator_coverage = self.config['neuromorphic_system']['flow_actuators'] / (critical_area * 0.15)
        
        # Calculate system response time
        self.response_time = 1.0 / self.config['neuromorphic_system']['update_rate'] + 0.004  # processing delay
        
        # Calculate power density (W/cm²)
        electronics_area = 500  # cm² (approximate area of electronics)
        self.power_density = self.config['neuromorphic_system']['power_consumption'] / electronics_area
    
    def _validate_constraints(self) -> None:
        """Validate neuromorphic lifting body constraints."""
        self.violations = []
        
        # Check sensor density
        if self.sensor_density < self.config['neuromorphic_constraints']['min_sensor_density']:
            self.violations.append(
                f"Insufficient sensor density: {self.sensor_density:.2f} sensors/m² < "
                f"{self.config['neuromorphic_constraints']['min_sensor_density']:.2f} required"
            )
        
        # Check response time
        if self.response_time > self.config['neuromorphic_constraints']['max_response_time']:
            self.violations.append(
                f"Response time too slow: {self.response_time*1000:.2f}ms > "
                f"{self.config['neuromorphic_constraints']['max_response_time']*1000:.2f}ms maximum"
            )
        
        # Check actuator coverage
        if self.actuator_coverage < self.config['neuromorphic_constraints']['min_actuator_coverage']:
            self.violations.append(
                f"Insufficient actuator coverage: {self.actuator_coverage:.2f} < "
                f"{self.config['neuromorphic_constraints']['min_actuator_coverage']:.2f} required"
            )
        
        # Check power density
        if self.power_density > self.config['neuromorphic_constraints']['max_power_density']:
            self.violations.append(
                f"Excessive power density: {self.power_density:.2f} W/cm² > "
                f"{self.config['neuromorphic_constraints']['max_power_density']:.2f} W/cm² maximum"
            )
    
    def get_aerodynamic_characteristics(self, mach: float, aoa: float) -> Dict[str, float]:
        """
        Calculate aerodynamic characteristics at given flight condition.
        
        Args:
            mach: Mach number
            aoa: Angle of attack in degrees
        """
        # Base lift coefficient (simplified model)
        cl_base = 0.04 + 0.03 * aoa + self.config['aero_features']['body_camber'] * 0.8
        
        # Mach effects on lift
        if mach < 0.8:
            mach_lift_factor = 1.0
        elif mach < 1.2:
            mach_lift_factor = 1.0 + (mach - 0.8) * 0.5  # Transonic lift increase
        else:
            mach_lift_factor = 1.2 - (mach - 1.2) * 0.1  # Supersonic lift decrease
        
        # Neuromorphic flow control effects
        if self.config['aero_features']['control_surfaces']['active_flow_ports'] > 0:
            # Flow control more effective at high AoA and transonic speeds
            flow_control_factor = 1.0
            if aoa > 10:
                flow_control_factor += 0.15 * min(1.0, (aoa - 10) / 10)
            if 0.8 < mach < 1.2:
                flow_control_factor += 0.1 * (1.0 - abs(mach - 1.0) / 0.2)
        else:
            flow_control_factor = 1.0
        
        # Final lift coefficient
        cl = cl_base * mach_lift_factor * flow_control_factor
        
        # Drag coefficient (simplified)
        cd_base = 0.02 + 0.001 * aoa**2
        cd_wave = 0.0
        
        # Wave drag in transonic and supersonic
        if mach > 0.8:
            cd_wave = 0.1 * (mach - 0.8)**2 * (1.0 - self.config['aero_features']['nose_bluntness'] * 0.3)
        
        # Flow control drag reduction
        if self.config['aero_features']['control_surfaces']['active_flow_ports'] > 0:
            flow_drag_reduction = 0.1 * (self.actuator_coverage / self.config['neuromorphic_constraints']['min_actuator_coverage'])
        else:
            flow_drag_reduction = 0.0
        
        # Final drag coefficient
        cd = (cd_base + cd_wave) * (1.0 - flow_drag_reduction)
        
        # Thermal characteristics
        if mach > 1.0:
            stagnation_temp_rise = (mach**2) * 40  # Approximate temperature rise in Celsius
        else:
            stagnation_temp_rise = mach * 10
        
        # Neuromorphic system thermal load
        thermal_load = 0.3 + 0.7 * min(1.0, stagnation_temp_rise / 100)
        
        return {
            'lift_coefficient': cl,
            'drag_coefficient': cd,
            'lift_to_drag_ratio': cl / max(0.01, cd),
            'stagnation_temp_rise': stagnation_temp_rise,
            'neuromorphic_thermal_load': thermal_load,
            'flow_control_effectiveness': flow_control_factor - 1.0
        }
    
    def get_stability_characteristics(self, mach: float, aoa: float) -> Dict[str, float]:
        """Calculate stability characteristics at given flight condition."""
        # Static margin (simplified)
        static_margin_base = 0.05 - self.config['aero_features']['body_camber'] * 0.2
        
        # Mach effects on stability
        if mach < 0.8:
            mach_stability_factor = 1.0
        elif mach < 1.2:
            mach_stability_factor = 1.0 - (mach - 0.8) * 0.5  # Reduced stability in transonic
        else:
            mach_stability_factor = 0.8 + (mach - 1.2) * 0.1  # Increasing stability in supersonic
        
        # Neuromorphic augmentation
        if self.config['neuromorphic_system']['update_rate'] > 200:
            augmentation_factor = 1.0 + 0.2 * (self.config['neuromorphic_system']['update_rate'] - 200) / 300
        else:
            augmentation_factor = 1.0
        
        # Final static margin
        static_margin = static_margin_base * mach_stability_factor * augmentation_factor
        
        # Dynamic stability (simplified)
        pitch_damping = -0.4 - 0.1 * (aoa / 10)
        if self.config['aero_features']['control_surfaces']['active_flow_ports'] > 0:
            pitch_damping *= 1.2  # Better damping with active flow control
        
        return {
            'static_margin': static_margin,
            'pitch_damping': pitch_damping,
            'neuromorphic_augmentation': augmentation_factor - 1.0,
            'control_authority': 1.0 - 0.3 * (mach / 3) + 0.2 * self.actuator_coverage,
            'stability_limited_aoa': 25.0 + 5.0 * self.actuator_coverage
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_lifting_body(length: float, width: float, camber: float) -> LiftingBodyAirframe:
    """Create custom lifting body airframe model."""
    # Calculate approximate height
    height = width * 0.35
    
    # Calculate approximate planform area
    planform_area = length * width * 0.75
    
    # Calculate approximate volume
    volume = length * width * height * 0.7
    
    return LiftingBodyAirframe({
        'length': length,
        'width': width,
        'height': height,
        'planform_area': planform_area,
        'volume': volume,
        'aero_features': {
            'body_camber': camber,
            'nose_bluntness': 0.3 + camber * 0.5,
            'side_strakes': True,
            'control_surfaces': {
                'active_flow_ports': int(length * 2)
            }
        },
        'neuromorphic_system': {
            'pressure_sensors': int(planform_area * 1.3),
            'temperature_sensors': int(planform_area * 0.7),
            'flow_actuators': int(planform_area * 0.6),
            'processing_nodes': max(6, int(planform_area / 10))
        }
    })