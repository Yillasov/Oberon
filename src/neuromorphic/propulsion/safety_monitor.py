"""
Safety monitoring system for propulsion systems.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import time


class SafetyLevel(Enum):
    NOMINAL = "nominal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyThresholds:
    """Safety thresholds for propulsion monitoring."""
    max_temperature: float = 450.0    # K
    max_pressure: float = 40.0        # Bar
    max_vibration: float = 2.0        # g
    min_efficiency: float = 0.60      # 0-1
    max_thrust_oscillation: float = 0.15  # Normalized
    response_timeout: float = 0.5     # seconds


class PropulsionSafetyMonitor:
    """Safety monitoring system for propulsion systems."""
    
    def __init__(self, thresholds: SafetyThresholds = SafetyThresholds()):
        self.thresholds = thresholds
        self.state = {
            'safety_level': SafetyLevel.NOMINAL,
            'warnings': [],
            'last_update': time.time(),
            'fault_counter': 0,
            'emergency_shutdown_active': False
        }
        
        self.history = {
            'temperature': [],
            'pressure': [],
            'vibration': [],
            'efficiency': [],
            'thrust_oscillation': []
        }
        
        self.safety_actions = {
            SafetyLevel.WARNING: self._handle_warning,
            SafetyLevel.CRITICAL: self._handle_critical,
            SafetyLevel.EMERGENCY: self._handle_emergency
        }
    
    def monitor(self, engine_state: Dict[str, Any], 
                thrust_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Monitor engine parameters and assess safety status."""
        current_time = time.time()
        dt = current_time - self.state['last_update']
        self.state['last_update'] = current_time
        
        # Clear previous warnings
        self.state['warnings'] = []
        
        # Check response time
        if dt > self.thresholds.response_timeout:
            self._add_warning("Response timeout exceeded")
        
        # Temperature check
        if 'temperature' in engine_state:
            temp = engine_state['temperature']
            self._check_parameter('temperature', temp, self.thresholds.max_temperature)
            self.history['temperature'].append(temp)
        
        # Pressure check
        if 'pressure' in engine_state:
            pressure = engine_state['pressure']
            self._check_parameter('pressure', pressure, self.thresholds.max_pressure)
            self.history['pressure'].append(pressure)
        
        # Vibration check
        if 'vibration' in engine_state:
            vibration = engine_state['vibration']
            self._check_parameter('vibration', vibration, self.thresholds.max_vibration)
            self.history['vibration'].append(vibration)
        
        # Efficiency check
        if 'efficiency' in engine_state:
            efficiency = engine_state['efficiency']
            if efficiency < self.thresholds.min_efficiency:
                self._add_warning(f"Low efficiency: {efficiency:.2f}")
            self.history['efficiency'].append(efficiency)
        
        # Thrust oscillation check
        if thrust_state and 'amplitude' in thrust_state:
            oscillation = thrust_state['amplitude']
            if oscillation > self.thresholds.max_thrust_oscillation:
                self._add_warning(f"High thrust oscillation: {oscillation:.2f}")
            self.history['thrust_oscillation'].append(oscillation)
        
        # Update safety level
        self._update_safety_level()
        
        # Execute safety actions
        if self.state['safety_level'] != SafetyLevel.NOMINAL:
            self.safety_actions[self.state['safety_level']]()
        
        return self.get_status()
    
    def _check_parameter(self, name: str, value: float, threshold: float):
        """Check parameter against threshold."""
        if value > threshold:
            self._add_warning(f"{name.capitalize()} exceeded: {value:.2f}")
            if value > threshold * 1.2:  # 20% over threshold
                self.state['fault_counter'] += 1
    
    def _add_warning(self, message: str):
        """Add warning message and increment fault counter."""
        self.state['warnings'].append(message)
        self.state['fault_counter'] += 1
    
    def _update_safety_level(self):
        """Update safety level based on warnings and faults."""
        if self.state['emergency_shutdown_active']:
            self.state['safety_level'] = SafetyLevel.EMERGENCY
        elif self.state['fault_counter'] > 10:
            self.state['safety_level'] = SafetyLevel.CRITICAL
        elif self.state['warnings']:
            self.state['safety_level'] = SafetyLevel.WARNING
        else:
            self.state['safety_level'] = SafetyLevel.NOMINAL
            self.state['fault_counter'] = max(0, self.state['fault_counter'] - 1)
    
    def _handle_warning(self):
        """Handle warning level safety issues."""
        # Implement warning level actions
        pass
    
    def _handle_critical(self):
        """Handle critical level safety issues."""
        if self.state['fault_counter'] > 20:
            self.state['emergency_shutdown_active'] = True
    
    def _handle_emergency(self):
        """Handle emergency level safety issues."""
        self.state['emergency_shutdown_active'] = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            'safety_level': self.state['safety_level'].value,
            'warnings': self.state['warnings'].copy(),
            'fault_counter': self.state['fault_counter'],
            'emergency_shutdown': self.state['emergency_shutdown_active'],
            'trends': self._calculate_trends()
        }
    
    def _calculate_trends(self) -> Dict[str, float]:
        """Calculate parameter trends."""
        trends = {}
        for param, history in self.history.items():
            if len(history) >= 10:
                recent_avg = np.mean(history[-5:])
                previous_avg = np.mean(history[-10:-5])
                trends[param] = (recent_avg - previous_avg) / previous_avg
            else:
                trends[param] = 0.0
        return trends
    
    def reset_emergency(self):
        """Reset emergency shutdown state."""
        self.state['emergency_shutdown_active'] = False
        self.state['fault_counter'] = 0