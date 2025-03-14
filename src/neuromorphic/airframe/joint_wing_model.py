"""
Joint Wing parametric airframe model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import json


class JointWingAirframe:
    """Joint wing (box wing) airframe with neuromorphic load distribution control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'front_wingspan': 12.0,  # meters
            'rear_wingspan': 10.0,   # meters
            'front_chord': 1.8,      # meters
            'rear_chord': 1.5,       # meters
            'longitudinal_separation': 6.0,  # meters
            'vertical_separation': 2.2,      # meters
            
            # Structural features
            'structural_features': {
                'joint_type': 'winglet',  # winglet, strut, or blended
                'front_sweep': 15.0,      # degrees
                'rear_sweep': -10.0,      # degrees (negative for forward sweep)
                'joint_sweep': 40.0,      # degrees
                'structural_nodes': 8     # main structural connection points
            },
            
            # Neuromorphic load sensing system
            'neuromorphic_system': {
                'strain_sensors': 64,
                'pressure_sensors': 48,
                'accelerometers': 24,
                'processing_nodes': 8,
                'update_rate': 200.0,  # Hz
                'power_consumption': 28.0,  # Watts
                'load_redistribution': True
            },
            
            # Neuromorphic constraints
            'neuromorphic_constraints': {
                'min_sensor_coverage': 0.75,  # fraction of critical areas
                'max_response_time': 0.015,   # seconds
                'min_processing_redundancy': 2.0,  # redundancy factor
                'max_weight_penalty': 0.05,   # fraction of structural weight
                'communication_latency': 0.002  # seconds between nodes
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
        # Calculate total wing area
        self.front_wing_area = self.config['front_wingspan'] * self.config['front_chord']
        self.rear_wing_area = self.config['rear_wingspan'] * self.config['rear_chord']
        self.total_wing_area = self.front_wing_area + self.rear_wing_area
        
        # Calculate joint area (approximate)
        joint_length = np.sqrt((self.config['front_wingspan'] - self.config['rear_wingspan'])**2 / 4 + 
                              self.config['longitudinal_separation']**2 + 
                              self.config['vertical_separation']**2)
        joint_chord = (self.config['front_chord'] + self.config['rear_chord']) / 2
        self.joint_area = joint_length * joint_chord * 0.8  # 0.8 factor for joint taper
        
        # Calculate total wetted area
        self.total_wetted_area = self.total_wing_area + self.joint_area
        
        # Calculate sensor coverage
        total_sensors = (self.config['neuromorphic_system']['strain_sensors'] + 
                         self.config['neuromorphic_system']['pressure_sensors'] +
                         self.config['neuromorphic_system']['accelerometers'])
        
        # Estimate critical areas (joints and high-stress regions)
        critical_area_fraction = 0.4  # 40% of total area is considered critical
        critical_area = self.total_wetted_area * critical_area_fraction
        
        # Calculate sensor density in critical areas
        # Assume 70% of sensors are in critical areas
        critical_area_sensors = total_sensors * 0.7
        self.critical_sensor_coverage = critical_area_sensors / (critical_area * 5)  # 5 sensors per unit area for full coverage
        
        # Calculate system response time
        self.response_time = (1.0 / self.config['neuromorphic_system']['update_rate'] + 
                             self.config['neuromorphic_constraints']['communication_latency'])
    
    def _validate_constraints(self) -> None:
        """Validate neuromorphic joint wing constraints."""
        self.violations = []
        
        # Check sensor coverage
        if self.critical_sensor_coverage < self.config['neuromorphic_constraints']['min_sensor_coverage']:
            self.violations.append(
                f"Insufficient sensor coverage: {self.critical_sensor_coverage:.2f} < "
                f"{self.config['neuromorphic_constraints']['min_sensor_coverage']:.2f} required"
            )
        
        # Check response time
        if self.response_time > self.config['neuromorphic_constraints']['max_response_time']:
            self.violations.append(
                f"Response time too slow: {self.response_time*1000:.2f}ms > "
                f"{self.config['neuromorphic_constraints']['max_response_time']*1000:.2f}ms maximum"
            )
        
        # Check processing redundancy
        nodes_required = max(4, self.config['structural_features']['structural_nodes'] / 2)
        actual_redundancy = self.config['neuromorphic_system']['processing_nodes'] / nodes_required
        
        if actual_redundancy < self.config['neuromorphic_constraints']['min_processing_redundancy']:
            self.violations.append(
                f"Insufficient processing redundancy: {actual_redundancy:.2f} < "
                f"{self.config['neuromorphic_constraints']['min_processing_redundancy']:.2f} required"
            )
        
        # Check weight penalty
        neuromorphic_weight = 0.2 * self.config['neuromorphic_system']['processing_nodes'] + 0.05 * (
            self.config['neuromorphic_system']['strain_sensors'] + 
            self.config['neuromorphic_system']['pressure_sensors'] + 
            self.config['neuromorphic_system']['accelerometers'])
        
        structural_weight = 5.0 * self.total_wetted_area  # Simplified estimate
        weight_penalty = neuromorphic_weight / structural_weight
        
        if weight_penalty > self.config['neuromorphic_constraints']['max_weight_penalty']:
            self.violations.append(
                f"Excessive weight penalty: {weight_penalty:.3f} > "
                f"{self.config['neuromorphic_constraints']['max_weight_penalty']:.3f} maximum"
            )
    
    def get_load_distribution(self, flight_condition: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate load distribution for given flight condition.
        
        Args:
            flight_condition: Dictionary with:
                - 'speed': airspeed in m/s
                - 'aoa': angle of attack in degrees
                - 'roll_rate': roll rate in deg/s
                - 'pitch_rate': pitch rate in deg/s
                - 'load_factor': g-load
        """
        # Extract flight parameters
        speed = flight_condition.get('speed', 100.0)
        aoa = flight_condition.get('aoa', 2.0)
        roll_rate = flight_condition.get('roll_rate', 0.0)
        pitch_rate = flight_condition.get('pitch_rate', 0.0)
        load_factor = flight_condition.get('load_factor', 1.0)
        
        # Calculate dynamic pressure
        q = 0.5 * 1.225 * speed**2
        
        # Calculate baseline lift distribution (simplified)
        front_lift_fraction = 0.6 - 0.1 * (aoa - 2) / 10  # Front wing carries more lift at low AoA
        rear_lift_fraction = 1.0 - front_lift_fraction
        
        # Adjust for pitch rate
        if pitch_rate > 0:  # Pitch up
            front_lift_fraction += 0.05 * pitch_rate / 10
            rear_lift_fraction -= 0.05 * pitch_rate / 10
        else:  # Pitch down
            front_lift_fraction -= 0.05 * abs(pitch_rate) / 10
            rear_lift_fraction += 0.05 * abs(pitch_rate) / 10
        
        # Calculate structural loads
        front_wing_load = q * self.front_wing_area * front_lift_fraction * load_factor
        rear_wing_load = q * self.rear_wing_area * rear_lift_fraction * load_factor
        
        # Calculate joint loads (simplified)
        joint_bending = abs(front_wing_load - rear_wing_load) * 0.5
        joint_torsion = abs(roll_rate) * (self.config['front_wingspan'] + self.config['rear_wingspan']) / 4
        
        # Calculate neuromorphic system load
        sensor_activity = 0.3 + 0.7 * (load_factor / 3.0)  # More active at higher loads
        processing_load = 0.4 + 0.3 * sensor_activity + 0.3 * (abs(roll_rate) + abs(pitch_rate)) / 20
        
        return {
            'front_wing_load': front_wing_load,
            'rear_wing_load': rear_wing_load,
            'joint_bending_moment': joint_bending,
            'joint_torsion_moment': joint_torsion,
            'load_asymmetry': abs(front_lift_fraction - rear_lift_fraction),
            'neuromorphic_load': min(1.0, processing_load),
            'sensor_activity': sensor_activity
        }
    
    def get_efficiency_metrics(self, flight_condition: Dict[str, float]) -> Dict[str, float]:
        """Calculate efficiency metrics for given flight condition."""
        # Get load distribution
        loads = self.get_load_distribution(flight_condition)
        
        # Calculate induced drag factor (simplified)
        # Joint wings can have lower induced drag due to the joint effect
        span_efficiency = 1.2  # Higher than conventional wings
        
        # Adjust for load asymmetry - more asymmetric loading reduces efficiency
        span_efficiency -= loads['load_asymmetry'] * 0.3
        
        # Calculate effective aspect ratio
        effective_ar = (self.config['front_wingspan'] + self.config['rear_wingspan'])**2 / (2 * self.total_wing_area)
        
        # Calculate L/D (simplified)
        speed = flight_condition.get('speed', 100.0)
        aoa = flight_condition.get('aoa', 2.0)
        
        cl = 0.1 + 0.1 * aoa  # Simplified lift coefficient
        cdi = cl**2 / (np.pi * effective_ar * span_efficiency)  # Induced drag
        cd0 = 0.02  # Parasite drag
        
        ld_ratio = cl / (cd0 + cdi)
        
        # Calculate structural efficiency with neuromorphic augmentation
        if self.config['neuromorphic_system']['load_redistribution']:
            structural_efficiency = 1.0 + 0.15 * loads['neuromorphic_load']
        else:
            structural_efficiency = 1.0
        
        return {
            'span_efficiency': span_efficiency,
            'effective_aspect_ratio': effective_ar,
            'lift_to_drag': ld_ratio,
            'structural_efficiency': structural_efficiency,
            'neuromorphic_benefit': structural_efficiency - 1.0
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_joint_wing(front_span: float, rear_span: float, separation: float) -> JointWingAirframe:
    """Create custom joint wing airframe model."""
    return JointWingAirframe({
        'front_wingspan': front_span,
        'rear_wingspan': rear_span,
        'front_chord': front_span / 6.5,
        'rear_chord': rear_span / 6.5,
        'longitudinal_separation': separation,
        'vertical_separation': separation * 0.35,
        'structural_features': {
            'structural_nodes': max(6, int(front_span + rear_span) // 3)
        },
        'neuromorphic_system': {
            'strain_sensors': int((front_span + rear_span) * 3),
            'pressure_sensors': int((front_span + rear_span) * 2.5),
            'accelerometers': int((front_span + rear_span) * 1.2),
            'processing_nodes': max(6, int((front_span + rear_span) // 3))
        }
    })