"""
Strategic decision-making controller for mission execution.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime
from .mission_hierarchy import MissionHierarchy, TaskPriority
from .mission_simulator import SimulationConfig


class DecisionState(Enum):
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    ADAPTING = "adapting"
    EVALUATING = "evaluating"


@dataclass
class StrategicDecision:
    """Representation of a strategic decision."""
    decision_id: str
    timestamp: float
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    impact_estimate: float


class StrategicController:
    """High-level decision-making controller."""
    
    def __init__(self, mission: MissionHierarchy):
        self.mission = mission
        self.current_state = DecisionState.ANALYZING
        self.decisions: List[StrategicDecision] = []
        
        # Decision-making parameters
        self.confidence_threshold = 0.7
        self.impact_threshold = 0.5
        self.memory_length = 50
        
        # Neural processing
        self.decision_weights = np.random.normal(0.5, 0.1, (16, 16))
        self.state_memory = np.zeros((self.memory_length, 16))
        self.memory_index = 0
    
    async def process_mission_state(self) -> Dict[str, Any]:
        """Process current mission state and make strategic decisions."""
        # Encode current state
        state_vector = self._encode_mission_state()
        
        # Update state memory
        self.state_memory[self.memory_index] = state_vector
        self.memory_index = (self.memory_index + 1) % self.memory_length
        
        # Generate decision
        decision = await self._generate_decision(state_vector)
        
        if decision:
            self.decisions.append(decision)
            await self._execute_decision(decision)
        
        return {
            'current_state': self.current_state.value,
            'last_decision': decision.__dict__ if decision else None,
            'confidence': float(self._calculate_confidence()),
            'decision_count': len(self.decisions)
        }
    
    def _encode_mission_state(self) -> np.ndarray:
        """Encode mission state into neural vector."""
        state_vector = np.zeros(16)
        
        # Encode completion status
        completion_rate = sum(self.mission.completion_status.values()) / \
                         len(self.mission.tasks)
        state_vector[0:4] = completion_rate
        
        # Encode resource utilization
        if self.mission.resource_utilization:
            avg_utilization = sum(self.mission.resource_utilization.values()) / \
                            len(self.mission.resource_utilization)
            state_vector[4:8] = avg_utilization
        
        # Encode task priorities
        priority_counts = [0] * 4
        for task in self.mission.tasks.values():
            priority_counts[task.priority.value] += 1
        state_vector[8:12] = np.array(priority_counts) / len(self.mission.tasks)
        
        # Encode adaptation history
        if self.mission.adaptation_history:
            recent_adaptations = min(len(self.mission.adaptation_history), 4)
            state_vector[12:16] = recent_adaptations / 4
        
        return state_vector
    
    async def _generate_decision(self, 
                               state_vector: np.ndarray) -> Optional[StrategicDecision]:
        """Generate strategic decision based on state."""
        # Process state vector
        processed = np.dot(state_vector, self.decision_weights)
        activation = 1 / (1 + np.exp(-processed))
        
        # Calculate decision metrics
        confidence = float(np.mean(activation))
        impact = float(np.max(activation))
        
        if confidence < self.confidence_threshold or impact < self.impact_threshold:
            return None
        
        # Determine action type
        action_type = self._determine_action_type(activation)
        
        # Generate decision parameters
        parameters = self._generate_parameters(action_type, state_vector)
        
        return StrategicDecision(
            decision_id=f"D_{datetime.now().timestamp()}",
            timestamp=datetime.now().timestamp(),
            action_type=action_type,
            parameters=parameters,
            confidence=confidence,
            impact_estimate=impact
        )
    
    def _determine_action_type(self, activation: np.ndarray) -> str:
        """Determine type of strategic action."""
        max_index = np.argmax(activation)
        
        if max_index < 4:
            return "resource_reallocation"
        elif max_index < 8:
            return "priority_adjustment"
        elif max_index < 12:
            return "execution_optimization"
        else:
            return "adaptation_trigger"
    
    def _generate_parameters(self, 
                           action_type: str, 
                           state_vector: np.ndarray) -> Dict[str, Any]:
        """Generate parameters for strategic action."""
        if action_type == "resource_reallocation":
            return {
                'reallocation_factor': float(np.mean(state_vector[4:8])),
                'target_resources': list(self.mission.resource_utilization.keys())
            }
        elif action_type == "priority_adjustment":
            return {
                'adjustment_scale': float(np.mean(state_vector[8:12])),
                'target_priority': TaskPriority(
                    int(np.argmax(state_vector[8:12]))).name
            }
        elif action_type == "execution_optimization":
            return {
                'optimization_factor': float(np.mean(state_vector)),
                'target_phase': self.mission.current_phase.value
            }
        else:  # adaptation_trigger
            return {
                'adaptation_strength': float(np.mean(state_vector[12:16])),
                'target_parameters': ['learning_rate', 'activation_threshold']
            }
    
    async def _execute_decision(self, decision: StrategicDecision):
        """Execute strategic decision."""
        if decision.action_type == "resource_reallocation":
            await self._reallocate_resources(decision.parameters)
        elif decision.action_type == "priority_adjustment":
            await self._adjust_priorities(decision.parameters)
        elif decision.action_type == "execution_optimization":
            await self._optimize_execution(decision.parameters)
        elif decision.action_type == "adaptation_trigger":
            await self._trigger_adaptation(decision.parameters)
    
    async def _reallocate_resources(self, parameters: Dict[str, Any]):
        """Reallocate mission resources."""
        factor = parameters['reallocation_factor']
        for resource in parameters['target_resources']:
            if resource in self.mission.resource_utilization:
                self.mission.resource_utilization[resource] *= (1 + factor)
    
    async def _adjust_priorities(self, parameters: Dict[str, Any]):
        """Adjust task priorities."""
        scale = parameters['adjustment_scale']
        target = TaskPriority[parameters['target_priority']]
        
        for task in self.mission.tasks.values():
            if task.priority.value < target.value:
                new_priority = min(TaskPriority.CRITICAL.value,
                                 task.priority.value + int(scale * 2))
                task.priority = TaskPriority(new_priority)
    
    async def _optimize_execution(self, parameters: Dict[str, Any]):
        """Optimize mission execution."""
        factor = parameters['optimization_factor']
        self.mission.learning_rate *= (1 + factor * 0.1)
        self.mission.activation_threshold = max(0.3,
            self.mission.activation_threshold * (1 - factor * 0.1))
    
    async def _trigger_adaptation(self, parameters: Dict[str, Any]):
        """Trigger mission adaptation."""
        strength = parameters['adaptation_strength']
        if 'learning_rate' in parameters['target_parameters']:
            self.mission.learning_rate = max(0.001,
                self.mission.learning_rate * (1 + strength * 0.2))
        if 'activation_threshold' in parameters['target_parameters']:
            self.mission.activation_threshold = max(0.2,
                self.mission.activation_threshold * (1 - strength * 0.1))
    
    def _calculate_confidence(self) -> float:
        """Calculate overall decision-making confidence."""
        if not self.decisions:
            return 0.0
        
        recent_decisions = self.decisions[-min(len(self.decisions), 10):]
        return sum(d.confidence for d in recent_decisions) / len(recent_decisions)