"""
Morphing Wing parametric airframe model for neuromorphic control systems.
"""
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import json


class MorphingWingAirframe:
    """Morphing wing airframe with neuromorphic shape control."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            # Basic parameters
            'wingspan_min': 6.0,     # meters (retracted)
            'wingspan_max': 9.5,     # meters (extended)
            'chord_root': 2.4,       # meters
            'chord_tip_min': 0.8,    # meters (minimum)
            'chord_tip_max': 1.5,    # meters (maximum)
            'sweep_range': [15.0, 35.0],  # degrees [min, max]
            
            # Morphing capabilities
            'morphing_features': {
                'span_morphing': True,
                'chord_morphing': True,
                'camber_morphing': True,
                'sweep_morphing': True,
                'twist_morphing': True,
                'morphing_segments': 5,  # number of independent segments
                'max_deformation': 0.25  # fraction of local dimension
            },
            
            # Neuromorphic control system
            'neuromorphic_system': {
                'strain_sensors': 48,
                'pressure_sensors': 36,
                'shape_memory_actuators': 24,
                'processing_nodes': 6,
                'update_rate': 120.0,  # Hz
                'power_consumption': 35.0,  # Watts
                'distributed_control': True
            },
            
            # Neuromorphic constraints
            'neuromorphic_constraints': {
                'max_morphing_rate': 0.15,  # fraction of full range per second
                'sensor_density_min': 1.2,  # sensors per square meter
                'actuator_density_min': 0.8,  # actuators per square meter
                'shape_precision': 0.005,  # meters
                'power_per_actuator': 1.2  # Watts
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
        # Calculate average wing area
        avg_wingspan = (self.config['wingspan_min'] + self.config['wingspan_max']) / 2
        avg_chord_tip = (self.config['chord_tip_min'] + self.config['chord_tip_max']) / 2
        self.avg_wing_area = avg_wingspan * (self.config['chord_root'] + avg_chord_tip) / 2
        
        # Calculate sensor and actuator densities
        total_sensors = (self.config['neuromorphic_system']['strain_sensors'] + 
                         self.config['neuromorphic_system']['pressure_sensors'])
        self.sensor_density = total_sensors / self.avg_wing_area
        
        self.actuator_density = self.config['neuromorphic_system']['shape_memory_actuators'] / self.avg_wing_area
        
        # Calculate morphing power requirements
        self.morphing_power = (self.config['neuromorphic_system']['shape_memory_actuators'] * 
                              self.config['neuromorphic_constraints']['power_per_actuator'])
        
        # Calculate maximum shape change rate
        segments = self.config['morphing_features']['morphing_segments']
        self.max_shape_rate = self.config['neuromorphic_constraints']['max_morphing_rate'] * segments
    
    def _validate_constraints(self) -> None:
        """Validate neuromorphic morphing constraints."""
        self.violations = []
        
        # Check sensor density
        if self.sensor_density < self.config['neuromorphic_constraints']['sensor_density_min']:
            self.violations.append(
                f"Insufficient sensor density: {self.sensor_density:.2f} sensors/m² < "
                f"{self.config['neuromorphic_constraints']['sensor_density_min']:.2f} required"
            )
        
        # Check actuator density
        if self.actuator_density < self.config['neuromorphic_constraints']['actuator_density_min']:
            self.violations.append(
                f"Insufficient actuator density: {self.actuator_density:.2f} actuators/m² < "
                f"{self.config['neuromorphic_constraints']['actuator_density_min']:.2f} required"
            )
        
        # Check power requirements
        if self.morphing_power > self.config['neuromorphic_system']['power_consumption']:
            self.violations.append(
                f"Insufficient power: {self.config['neuromorphic_system']['power_consumption']:.1f}W < "
                f"{self.morphing_power:.1f}W required for full morphing"
            )
        
        # Check update rate for morphing control
        min_update_rate = self.max_shape_rate * 20  # Need at least 20 updates per full morphing
        if self.config['neuromorphic_system']['update_rate'] < min_update_rate:
            self.violations.append(
                f"Insufficient update rate: {self.config['neuromorphic_system']['update_rate']:.1f}Hz < "
                f"{min_update_rate:.1f}Hz required for smooth morphing"
            )
    
    def get_configuration(self, morphing_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate wing configuration for given morphing parameters.
        
        Args:
            morphing_params: Dictionary with values from 0.0 to 1.0 for:
                - 'span': 0.0 = min wingspan, 1.0 = max wingspan
                - 'chord': 0.0 = min chord, 1.0 = max chord
                - 'sweep': 0.0 = min sweep, 1.0 = max sweep
                - 'camber': 0.0 = min camber, 1.0 = max camber
                - 'twist': 0.0 = min twist, 1.0 = max twist
        """
        # Get morphing parameters with defaults
        span_factor = morphing_params.get('span', 0.5)
        chord_factor = morphing_params.get('chord', 0.5)
        sweep_factor = morphing_params.get('sweep', 0.5)
        camber_factor = morphing_params.get('camber', 0.5)
        twist_factor = morphing_params.get('twist', 0.5)
        
        # Calculate current configuration
        wingspan = self.config['wingspan_min'] + span_factor * (
            self.config['wingspan_max'] - self.config['wingspan_min'])
        
        chord_tip = self.config['chord_tip_min'] + chord_factor * (
            self.config['chord_tip_max'] - self.config['chord_tip_min'])
        
        sweep = self.config['sweep_range'][0] + sweep_factor * (
            self.config['sweep_range'][1] - self.config['sweep_range'][0])
        
        # Calculate wing area
        wing_area = wingspan * (self.config['chord_root'] + chord_tip) / 2
        
        # Calculate aspect ratio
        aspect_ratio = wingspan**2 / wing_area
        
        return {
            'wingspan': wingspan,
            'chord_tip': chord_tip,
            'sweep_angle': sweep,
            'camber_factor': camber_factor,
            'twist_factor': twist_factor,
            'wing_area': wing_area,
            'aspect_ratio': aspect_ratio,
            'neuromorphic_load': self._calculate_neuromorphic_load(morphing_params)
        }
    
    def _calculate_neuromorphic_load(self, morphing_params: Dict[str, float]) -> float:
        """Calculate neuromorphic processing load for given morphing configuration."""
        # Base load
        base_load = 0.3
        
        # Additional load based on active morphing features
        active_features = sum(1 for k, v in morphing_params.items() if v > 0.1 and v < 0.9)
        feature_load = active_features * 0.1
        
        # Rate of change load (would be calculated from time derivatives in real system)
        rate_load = 0.2  # Placeholder
        
        return min(1.0, base_load + feature_load + rate_load)
    
    def get_performance_envelope(self, morphing_params: Dict[str, float]) -> Dict[str, float]:
        """Get performance envelope for current morphing configuration."""
        config = self.get_configuration(morphing_params)
        
        # Calculate lift coefficient range (simplified)
        cl_max = 1.2 + 0.4 * morphing_params.get('camber', 0.5)
        
        # Calculate drag characteristics (simplified)
        cd_min = 0.02 + 0.01 * (1.0 - morphing_params.get('span', 0.5))
        
        # Calculate L/D ratio
        ld_max = cl_max / cd_min
        
        # Calculate stall speed (simplified, assuming 5000kg aircraft)
        wing_loading = 5000 * 9.81 / config['wing_area']
        stall_speed = np.sqrt(2 * wing_loading / (1.225 * cl_max))
        
        return {
            'cl_max': cl_max,
            'cd_min': cd_min,
            'ld_max': ld_max,
            'stall_speed': stall_speed,
            'neuromorphic_load': config['neuromorphic_load'],
            'power_required': self.config['neuromorphic_system']['power_consumption'] * config['neuromorphic_load']
        }
    
    def save_model(self, filename: str) -> None:
        """Save model configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)


def create_morphing_wing(wingspan_range: Tuple[float, float], segments: int) -> MorphingWingAirframe:
    """Create custom morphing wing airframe model."""
    return MorphingWingAirframe({
        'wingspan_min': wingspan_range[0],
        'wingspan_max': wingspan_range[1],
        'chord_root': wingspan_range[0] / 3.0,
        'chord_tip_min': wingspan_range[0] / 7.5,
        'chord_tip_max': wingspan_range[0] / 5.0,
        'morphing_features': {
            'morphing_segments': segments
        },
        'neuromorphic_system': {
            'strain_sensors': int(segments * 8),
            'pressure_sensors': int(segments * 6),
            'shape_memory_actuators': int(segments * 4),
            'processing_nodes': max(3, segments // 2)
        }
    })