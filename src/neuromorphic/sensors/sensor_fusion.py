"""
Data fusion algorithms for multi-sensor payload integration.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from .sensor_interface import SensorType, SensorSpecs


class FusionMethod(Enum):
    KALMAN = "kalman"
    BAYESIAN = "bayesian"
    WEIGHTED_AVERAGE = "weighted_average"
    DEMPSTER_SHAFER = "dempster_shafer"


@dataclass
class FusionConfig:
    """Configuration for sensor fusion."""
    method: FusionMethod
    temporal_window: float  # seconds
    spatial_resolution: Tuple[float, ...]
    confidence_threshold: float
    max_delay: float  # seconds


class SensorFusion:
    """Multi-sensor data fusion processor."""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.sensor_data: Dict[str, List[Dict[str, Any]]] = {}
        self.fusion_weights: Dict[str, float] = {}
        self.covariance_matrix: Optional[np.ndarray] = None
        self.last_fusion_result: Optional[Dict[str, Any]] = None
        
        # Kalman filter state
        self.state_estimate = None
        self.error_covariance = None
    
    def add_sensor_data(self, 
                       sensor_id: str, 
                       data: Dict[str, Any], 
                       specs: SensorSpecs) -> bool:
        """Add new sensor data for fusion."""
        if sensor_id not in self.sensor_data:
            self.sensor_data[sensor_id] = []
            self._initialize_fusion_weights(sensor_id, specs)
        
        self.sensor_data[sensor_id].append({
            'timestamp': data['timestamp'],
            'data': data['data'],
            'confidence': self._calculate_confidence(data, specs)
        })
        
        # Maintain temporal window
        self._prune_old_data(sensor_id)
        return True
    
    def _initialize_fusion_weights(self, sensor_id: str, specs: SensorSpecs):
        """Initialize fusion weights for new sensor."""
        # Weight based on sensor precision and bandwidth
        base_weight = specs.precision * specs.bandwidth
        self.fusion_weights[sensor_id] = base_weight
        
        # Normalize weights
        total_weight = sum(self.fusion_weights.values())
        for sid in self.fusion_weights:
            self.fusion_weights[sid] /= total_weight
    
    def _calculate_confidence(self, 
                            data: Dict[str, Any], 
                            specs: SensorSpecs) -> float:
        """Calculate confidence score for sensor data."""
        # Basic confidence calculation based on data statistics
        if 'metadata' in data:
            meta = data['metadata']
            snr = (meta['mean'] - meta['min']) / (meta['std'] + 1e-6)
            range_ratio = (meta['max'] - meta['min']) / (
                specs.range_limits[1] - specs.range_limits[0])
            
            confidence = min(1.0, 0.5 * snr + 0.5 * range_ratio)
        else:
            confidence = 0.8  # Default confidence
        
        return confidence
    
    def _prune_old_data(self, sensor_id: str):
        """Remove data outside temporal window."""
        if not self.sensor_data[sensor_id]:
            return
            
        current_time = max(d['timestamp'] 
                         for d in self.sensor_data[sensor_id])
        self.sensor_data[sensor_id] = [
            d for d in self.sensor_data[sensor_id]
            if current_time - d['timestamp'] <= self.config.temporal_window
        ]
    
    async def fuse_data(self) -> Optional[Dict[str, Any]]:
        """Perform sensor data fusion."""
        if not self.sensor_data:
            return None
        
        if self.config.method == FusionMethod.KALMAN:
            result = await self._kalman_fusion()
        elif self.config.method == FusionMethod.BAYESIAN:
            result = await self._bayesian_fusion()
        elif self.config.method == FusionMethod.WEIGHTED_AVERAGE:
            result = await self._weighted_average_fusion()
        else:  # DEMPSTER_SHAFER
            result = await self._dempster_shafer_fusion()
        
        self.last_fusion_result = result
        return result
    
    async def _kalman_fusion(self) -> Dict[str, Any]:
        """Kalman filter-based fusion."""
        # Initialize state if needed
        if self.state_estimate is None:
            self._initialize_kalman_state()
        
        for sensor_id, data_list in self.sensor_data.items():
            if not data_list:
                continue
            
            latest_data = data_list[-1]
            measurement = latest_data['data']
            confidence = latest_data['confidence']
            
            # Kalman update
            kalman_gain = self.error_covariance / (
                self.error_covariance + 1/confidence)
            self.state_estimate += kalman_gain * (
                measurement - self.state_estimate)
            self.error_covariance = (
                1 - kalman_gain) * self.error_covariance
        
        return {
            'timestamp': max(data_list[-1]['timestamp'] 
                           for data_list in self.sensor_data.values() if data_list),
            'fused_data': self.state_estimate,
            'uncertainty': self.error_covariance,
            'method': FusionMethod.KALMAN.value
        }
    
    def _initialize_kalman_state(self):
        """Initialize Kalman filter state."""
        # Use first available sensor data for initialization
        for data_list in self.sensor_data.values():
            if data_list:
                self.state_estimate = data_list[0]['data']
                self.error_covariance = np.eye(
                    np.array(self.state_estimate).shape[0]) * 1.0
                break
    
    async def _bayesian_fusion(self) -> Dict[str, Any]:
        """Bayesian fusion using sensor confidences."""
        fused_data = None
        total_confidence = 0.0
        latest_timestamp = 0.0
        
        for sensor_id, data_list in self.sensor_data.items():
            if not data_list:
                continue
                
            latest_data = data_list[-1]
            confidence = latest_data['confidence']
            
            if fused_data is None:
                fused_data = latest_data['data'] * confidence
            else:
                fused_data += latest_data['data'] * confidence
            
            total_confidence += confidence
            latest_timestamp = max(latest_timestamp, 
                                 latest_data['timestamp'])
        
        if fused_data is not None:
            fused_data /= total_confidence
        
        return {
            'timestamp': latest_timestamp,
            'fused_data': fused_data,
            'confidence': total_confidence,
            'method': FusionMethod.BAYESIAN.value
        }
    
    async def _weighted_average_fusion(self) -> Dict[str, Any]:
        """Weighted average fusion using predefined weights."""
        fused_data = None
        latest_timestamp = 0.0
        
        for sensor_id, data_list in self.sensor_data.items():
            if not data_list or sensor_id not in self.fusion_weights:
                continue
                
            latest_data = data_list[-1]
            weight = self.fusion_weights[sensor_id]
            
            if fused_data is None:
                fused_data = latest_data['data'] * weight
            else:
                fused_data += latest_data['data'] * weight
            
            latest_timestamp = max(latest_timestamp, 
                                 latest_data['timestamp'])
        
        return {
            'timestamp': latest_timestamp,
            'fused_data': fused_data,
            'weights': self.fusion_weights.copy(),
            'method': FusionMethod.WEIGHTED_AVERAGE.value
        }
    
    async def _dempster_shafer_fusion(self) -> Dict[str, Any]:
        """Dempster-Shafer evidence theory fusion."""
        belief_functions = {}
        latest_timestamp = 0.0
        
        for sensor_id, data_list in self.sensor_data.items():
            if not data_list:
                continue
                
            latest_data = data_list[-1]
            confidence = latest_data['confidence']
            
            # Create basic belief assignment
            belief_functions[sensor_id] = {
                'belief': latest_data['data'] * confidence,
                'plausibility': latest_data['data'] * (2 - confidence)
            }
            
            latest_timestamp = max(latest_timestamp, 
                                 latest_data['timestamp'])
        
        # Combine evidence using Dempster's rule
        combined_belief = None
        combined_plausibility = None
        
        for belief_func in belief_functions.values():
            if combined_belief is None:
                combined_belief = belief_func['belief']
                combined_plausibility = belief_func['plausibility']
            else:
                # Dempster's combination rule
                combined_belief = (combined_belief * belief_func['belief']) / (
                    1 - np.minimum(combined_plausibility, 
                                 belief_func['plausibility']))
                combined_plausibility = (
                    combined_plausibility * belief_func['plausibility'])
        
        return {
            'timestamp': latest_timestamp,
            'fused_data': combined_belief,
            'plausibility': combined_plausibility,
            'method': FusionMethod.DEMPSTER_SHAFER.value
        }
    
    def get_fusion_status(self) -> Dict[str, Any]:
        """Get current fusion status."""
        return {
            'active_sensors': list(self.sensor_data.keys()),
            'fusion_method': self.config.method.value,
            'temporal_window': self.config.temporal_window,
            'weights': self.fusion_weights,
            'last_result': self.last_fusion_result
        }