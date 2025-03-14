"""
Explainable AI module for mission decision transparency.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from .adaptive_learning import NeuromorphicRL, RewardSignal
from .strategic_controller import StrategicDecision
from .mission_hierarchy import MissionHierarchy


@dataclass
class DecisionExplanation:
    """Detailed explanation of a strategic decision."""
    timestamp: float
    decision_id: str
    factors: Dict[str, float]
    context: Dict[str, Any]
    confidence_breakdown: Dict[str, float]
    alternatives: List[Dict[str, Any]]


class ExplainableAI:
    """Explainable AI system for mission decisions."""
    
    def __init__(self, 
                 mission: MissionHierarchy,
                 learner: NeuromorphicRL):
        self.mission = mission
        self.learner = learner
        self.explanations: List[DecisionExplanation] = []
        
        # Feature importance tracking
        self.feature_weights = np.ones(32) / 32
        self.context_history: List[Dict[str, Any]] = []
        self.importance_threshold = 0.1
    
    async def explain_decision(self, 
                             decision: StrategicDecision,
                             state: np.ndarray) -> DecisionExplanation:
        """Generate detailed explanation for a strategic decision."""
        # Analyze decision factors
        factors = self._analyze_decision_factors(decision, state)
        
        # Generate context
        context = self._generate_decision_context(decision)
        
        # Calculate confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(decision)
        
        # Generate alternative decisions
        alternatives = await self._generate_alternatives(decision, state)
        
        explanation = DecisionExplanation(
            timestamp=datetime.now().timestamp(),
            decision_id=decision.decision_id,
            factors=factors,
            context=context,
            confidence_breakdown=confidence_breakdown,
            alternatives=alternatives
        )
        
        self.explanations.append(explanation)
        return explanation
    
    def _analyze_decision_factors(self,
                                decision: StrategicDecision,
                                state: np.ndarray) -> Dict[str, float]:
        """Analyze factors contributing to the decision."""
        # Calculate feature importance
        importance = state * self.feature_weights
        significant_features = {}
        
        # Mission progress impact
        if np.sum(importance[0:8]) > self.importance_threshold:
            significant_features['mission_progress'] = float(np.mean(importance[0:8]))
        
        # Resource utilization impact
        if np.sum(importance[8:16]) > self.importance_threshold:
            significant_features['resource_impact'] = float(np.mean(importance[8:16]))
        
        # Priority consideration
        if np.sum(importance[16:24]) > self.importance_threshold:
            significant_features['priority_impact'] = float(np.mean(importance[16:24]))
        
        # Learning influence
        if np.sum(importance[24:32]) > self.importance_threshold:
            significant_features['learning_impact'] = float(np.mean(importance[24:32]))
        
        return significant_features
    
    def _generate_decision_context(self,
                                 decision: StrategicDecision) -> Dict[str, Any]:
        """Generate contextual information for the decision."""
        context = {
            'mission_state': self.mission.current_phase.value,
            'completion_rate': sum(self.mission.completion_status.values()) / 
                             len(self.mission.tasks),
            'active_tasks': len([t for t in self.mission.tasks.values() 
                               if self.mission.completion_status[t.task_id] < 1.0]),
            'resource_status': dict(self.mission.resource_utilization),
            'recent_adaptations': len(self.learner.reward_history[-10:])
        }
        
        self.context_history.append(context)
        return context
    
    def _calculate_confidence_breakdown(self,
                                     decision: StrategicDecision) -> Dict[str, float]:
        """Calculate detailed confidence breakdown."""
        breakdown = {
            'base_confidence': decision.confidence,
            'impact_confidence': decision.impact_estimate
        }
        
        # Add learning-based confidence
        if self.learner.reward_history:
            recent_rewards = [r.value for r in self.learner.reward_history[-5:]]
            breakdown['learning_confidence'] = float(np.mean(recent_rewards))
        
        # Add resource-based confidence
        if self.mission.resource_utilization:
            resource_efficiency = 1.0 - sum(self.mission.resource_utilization.values()) / \
                                len(self.mission.resource_utilization)
            breakdown['resource_confidence'] = float(resource_efficiency)
        
        return breakdown
    
    async def _generate_alternatives(self,
                                   decision: StrategicDecision,
                                   state: np.ndarray) -> List[Dict[str, Any]]:
        """Generate alternative decisions that were considered."""
        alternatives = []
        
        # Generate alternative actions using the learner
        for _ in range(3):  # Generate top 3 alternatives
            alt_action = self.learner._select_action(state)
            alt_reward = self.learner._calculate_reward(state, alt_action, decision)
            
            alternatives.append({
                'action_type': f"alternative_{len(alternatives)+1}",
                'estimated_reward': float(alt_reward),
                'relative_confidence': float(alt_reward / decision.confidence)
            })
        
        return sorted(alternatives, 
                     key=lambda x: x['estimated_reward'],
                     reverse=True)
    
    def get_explanation_summary(self, 
                              decision_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of decision explanations."""
        if not self.explanations:
            return {'status': 'no_explanations_available'}
        
        if decision_id:
            relevant_explanations = [e for e in self.explanations 
                                   if e.decision_id == decision_id]
            if not relevant_explanations:
                return {'status': 'decision_not_found'}
            
            explanation = relevant_explanations[-1]
            return {
                'decision_id': explanation.decision_id,
                'timestamp': explanation.timestamp,
                'key_factors': sorted(
                    explanation.factors.items(),
                    key=lambda x: x[1],
                    reverse=True
                ),
                'confidence': explanation.confidence_breakdown,
                'context': explanation.context,
                'alternatives_considered': len(explanation.alternatives)
            }
        
        # Summary of recent explanations
        recent = self.explanations[-5:]
        return {
            'total_decisions_explained': len(self.explanations),
            'recent_explanations': [
                {
                    'decision_id': e.decision_id,
                    'timestamp': e.timestamp,
                    'top_factor': max(e.factors.items(), key=lambda x: x[1])[0]
                }
                for e in recent
            ],
            'average_confidence': np.mean([
                sum(e.confidence_breakdown.values())
                for e in recent
            ])
        }