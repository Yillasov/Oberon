"""
Adaptive engine model with AI-driven optimization and control.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import torch
import torch.nn as nn


@dataclass
class EngineParameters:
    """Engine parameters with adaptive ranges."""
    max_power: float = 250.0        # kW
    base_efficiency: float = 0.82   # Base efficiency
    thermal_capacity: float = 15.0  # kJ/K
    response_time: float = 0.2      # seconds
    learning_rate: float = 0.01     # AI model learning rate
    memory_length: int = 1000       # Historical data points
    prediction_horizon: int = 100   # Future prediction steps


class PerformancePredictor(nn.Module):
    """Neural network for engine performance prediction."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # Predict efficiency, temperature, wear, power
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdaptiveEngine:
    """Adaptive engine with AI-driven optimization."""
    
    def __init__(self):
        self.params = EngineParameters()
        self.state = {
            "power_output": 0.0,
            "efficiency": self.params.base_efficiency,
            "temperature": 298.15,
            "wear_level": 0.0,
            "optimization_score": 1.0,
            "predicted_maintenance": float('inf'),
            "operating_mode": "normal"
        }
        
        # Performance history
        self.history = {
            "power": deque(maxlen=self.params.memory_length),
            "efficiency": deque(maxlen=self.params.memory_length),
            "temperature": deque(maxlen=self.params.memory_length),
            "wear": deque(maxlen=self.params.memory_length)
        }
        
        # AI components
        self.predictor = PerformancePredictor()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), 
                                        lr=self.params.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Adaptive parameters
        self.adaptive_params = {
            "combustion_timing": 0.0,    # -1 to 1
            "fuel_mixture": 0.5,         # 0 to 1
            "cooling_flow": 0.5,         # 0 to 1
            "power_curve": 0.5           # 0 to 1
        }
        
        self.optimization_state = {
            "last_update": 0.0,
            "update_interval": 0.1,
            "learning_enabled": True
        }
    
    def update(self, power_demand: float, dt: float, ambient_temp: float = 293.15) -> Dict[str, Any]:
        """Update engine state with AI optimization."""
        # Prepare input features for AI prediction
        features = self._prepare_features(power_demand, ambient_temp)
        
        # Get AI predictions
        predictions = self._get_predictions(features)
        
        # Optimize control parameters
        self._optimize_parameters(predictions, power_demand)
        
        # Update engine state
        self._update_state(power_demand, dt, ambient_temp)
        
        # Update historical data
        self._update_history()
        
        # Train AI model if enabled
        if self.optimization_state["learning_enabled"]:
            self._train_model()
        
        return self.state
    
    def _prepare_features(self, power_demand: float, ambient_temp: float) -> torch.Tensor:
        """Prepare input features for AI model."""
        features = [
            power_demand / self.params.max_power,
            self.state["temperature"] / 500.0,  # Normalize temperature
            self.state["wear_level"],
            self.adaptive_params["combustion_timing"],
            self.adaptive_params["fuel_mixture"],
            self.adaptive_params["cooling_flow"],
            self.adaptive_params["power_curve"],
            (ambient_temp - 273.15) / 50.0  # Normalize ambient temperature
        ]
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_predictions(self, features: torch.Tensor) -> torch.Tensor:
        """Get predictions from AI model."""
        self.predictor.eval()
        with torch.no_grad():
            predictions = self.predictor(features)
        return predictions
    
    def _optimize_parameters(self, predictions: torch.Tensor, power_demand: float):
        """Optimize engine parameters based on AI predictions."""
        pred_efficiency, pred_temp, pred_wear, pred_power = predictions.numpy()
        
        # Adjust combustion timing
        if pred_efficiency < self.state["efficiency"]:
            self.adaptive_params["combustion_timing"] += 0.1 * \
                (self.state["efficiency"] - pred_efficiency)
        
        # Adjust fuel mixture
        power_error = power_demand - pred_power
        self.adaptive_params["fuel_mixture"] += 0.05 * power_error
        
        # Adjust cooling flow based on temperature prediction
        if pred_temp > 350:  # Temperature threshold
            self.adaptive_params["cooling_flow"] += 0.1
        elif pred_temp < 320:
            self.adaptive_params["cooling_flow"] -= 0.05
        
        # Adjust power curve based on wear prediction
        if pred_wear > self.state["wear_level"]:
            self.adaptive_params["power_curve"] -= 0.05
        
        # Clamp all parameters to valid ranges
        for param in self.adaptive_params:
            self.adaptive_params[param] = max(0.0, min(1.0, self.adaptive_params[param]))
    
    def _update_state(self, power_demand: float, dt: float, ambient_temp: float):
        """Update engine state based on optimized parameters."""
        # Power response
        target_power = min(power_demand, self.params.max_power)
        power_error = target_power - self.state["power_output"]
        self.state["power_output"] += power_error * dt / self.params.response_time
        
        # Efficiency calculation
        base_efficiency = self.params.base_efficiency
        timing_factor = 1.0 - 0.2 * abs(self.adaptive_params["combustion_timing"])
        mixture_factor = 1.0 - 0.15 * abs(self.adaptive_params["fuel_mixture"] - 0.5)
        self.state["efficiency"] = base_efficiency * timing_factor * mixture_factor
        
        # Temperature dynamics
        heat_generation = (1.0 - self.state["efficiency"]) * self.state["power_output"]
        cooling_power = self.adaptive_params["cooling_flow"] * \
                       (self.state["temperature"] - ambient_temp)
        temp_change = (heat_generation - cooling_power) * dt / self.params.thermal_capacity
        self.state["temperature"] += temp_change
        
        # Wear calculation
        stress_factor = (self.state["power_output"] / self.params.max_power) * \
                       (self.state["temperature"] - 298.15) / 100
        self.state["wear_level"] += stress_factor * dt * 1e-5
        
        # Update optimization score
        self.state["optimization_score"] = self._calculate_optimization_score()
    
    def _update_history(self):
        """Update historical data."""
        self.history["power"].append(self.state["power_output"])
        self.history["efficiency"].append(self.state["efficiency"])
        self.history["temperature"].append(self.state["temperature"])
        self.history["wear"].append(self.state["wear_level"])
    
    def _train_model(self):
        """Train AI model on historical data."""
        if len(self.history["power"]) < 100:  # Minimum data required
            return
            
        # Prepare training data
        X = []
        y = []
        for i in range(len(self.history["power"]) - 1):
            features = [
                self.history["power"][i] / self.params.max_power,
                self.history["temperature"][i] / 500.0,
                self.history["wear"][i],
                self.adaptive_params["combustion_timing"],
                self.adaptive_params["fuel_mixture"],
                self.adaptive_params["cooling_flow"],
                self.adaptive_params["power_curve"],
                293.15 / 500.0  # Normalized ambient temperature
            ]
            targets = [
                self.history["efficiency"][i+1],
                self.history["temperature"][i+1],
                self.history["wear"][i+1],
                self.history["power"][i+1]
            ]
            X.append(features)
            y.append(targets)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Train for one epoch
        self.predictor.train()
        self.optimizer.zero_grad()
        predictions = self.predictor(X)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        self.optimizer.step()
    
    def _calculate_optimization_score(self) -> float:
        """Calculate overall optimization score."""
        efficiency_score = self.state["efficiency"] / self.params.base_efficiency
        temp_score = max(0, 1.0 - (self.state["temperature"] - 320) / 50)
        wear_score = 1.0 - self.state["wear_level"]
        
        return (efficiency_score + temp_score + wear_score) / 3
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostic data."""
        return {
            "performance": {
                "power_output": self.state["power_output"],
                "efficiency": self.state["efficiency"],
                "optimization_score": self.state["optimization_score"]
            },
            "health": {
                "temperature": self.state["temperature"],
                "wear_level": self.state["wear_level"],
                "predicted_maintenance": self.state["predicted_maintenance"]
            },
            "optimization": {
                "combustion_timing": self.adaptive_params["combustion_timing"],
                "fuel_mixture": self.adaptive_params["fuel_mixture"],
                "cooling_flow": self.adaptive_params["cooling_flow"],
                "power_curve": self.adaptive_params["power_curve"]
            }
        }
    
    def set_learning_mode(self, enabled: bool):
        """Enable or disable AI learning."""
        self.optimization_state["learning_enabled"] = enabled
    
    def save_model(self, path: str):
        """Save AI model to file."""
        torch.save({
            'model_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load AI model from file."""
        checkpoint = torch.load(path)
        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])