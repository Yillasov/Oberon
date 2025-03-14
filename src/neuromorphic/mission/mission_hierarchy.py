"""
Hierarchical mission representation system with neuromorphic processing capabilities.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime


class MissionPhase(Enum):
    PLANNING = "planning"
    INITIALIZATION = "initialization"
    EXECUTION = "execution"
    ADAPTATION = "adaptation"
    COMPLETION = "completion"
    ABORT = "abort"


class TaskPriority(Enum):
    CRITICAL = 3
    HIGH = 2
    MEDIUM = 1
    LOW = 0


@dataclass
class TaskNode:
    """Representation of a mission task node."""
    task_id: str
    description: str
    priority: TaskPriority
    dependencies: Set[str]
    estimated_duration: float
    resource_requirements: Dict[str, float]
    completion_criteria: Dict[str, Any]
    neural_signature: np.ndarray


class MissionHierarchy:
    """Hierarchical mission representation with neuromorphic processing."""
    
    def __init__(self, mission_id: str):
        self.mission_id = mission_id
        self.current_phase = MissionPhase.PLANNING
        self.tasks: Dict[str, TaskNode] = {}
        self.task_graph: Dict[str, Set[str]] = {}
        self.execution_order: List[str] = []
        
        # Neuromorphic processing parameters
        self.activation_patterns: Dict[str, np.ndarray] = {}
        self.synaptic_weights = np.random.normal(0.5, 0.1, (32, 32))
        self.learning_rate = 0.01
        self.activation_threshold = 0.6
        
        # Mission metrics
        self.completion_status: Dict[str, float] = {}
        self.resource_utilization: Dict[str, float] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Add optimization parameters
        self.population_size = 20
        self.max_iterations = 50
        self.optimization_history: List[Dict[str, float]] = []
    
    def add_task(self, task: TaskNode) -> bool:
        """Add a task to the mission hierarchy."""
        if task.task_id in self.tasks:
            return False
        
        self.tasks[task.task_id] = task
        self.task_graph[task.task_id] = set()
        self.completion_status[task.task_id] = 0.0
        
        # Initialize neural representation
        self.activation_patterns[task.task_id] = self._generate_neural_pattern(task)
        return True
    
    def _generate_neural_pattern(self, task: TaskNode) -> np.ndarray:
        """Generate neural activation pattern for task."""
        pattern = np.zeros(32)
        
        # Encode task properties into neural pattern
        pattern[0:8] = self._encode_priority(task.priority)
        pattern[8:16] = self._encode_duration(task.estimated_duration)
        pattern[16:24] = self._encode_resources(task.resource_requirements)
        pattern[24:32] = self._encode_dependencies(task.dependencies)
        
        return pattern
    
    def _encode_priority(self, priority: TaskPriority) -> np.ndarray:
        """Encode task priority into neural pattern."""
        encoding = np.zeros(8)
        encoding[priority.value * 2:(priority.value + 1) * 2] = 1
        return encoding
    
    def _encode_duration(self, duration: float) -> np.ndarray:
        """Encode estimated duration into neural pattern."""
        normalized = min(duration / 3600, 1.0)  # Normalize to 1-hour scale
        return np.array([normalized] * 8)
    
    def _encode_resources(self, 
                        requirements: Dict[str, float]) -> np.ndarray:
        """Encode resource requirements into neural pattern."""
        total_requirements = sum(requirements.values())
        normalized = min(total_requirements / 100, 1.0)
        return np.array([normalized] * 8)
    
    def _encode_dependencies(self, dependencies: Set[str]) -> np.ndarray:
        """Encode task dependencies into neural pattern."""
        encoding = np.zeros(8)
        if dependencies:
            encoding[:len(dependencies)] = 1
        return encoding
    
    async def process_mission_structure(self) -> Dict[str, Any]:
        """Process mission structure using neuromorphic approach."""
        # Generate activation matrix
        activation_matrix = np.vstack([
            self.activation_patterns[task_id]
            for task_id in self.tasks
        ])
        
        # Apply synaptic weights
        processed = np.dot(activation_matrix, self.synaptic_weights)
        
        # Apply activation function
        activated = 1 / (1 + np.exp(-processed))
        
        # Update task relationships based on activation patterns
        await self._update_task_relationships(activated)
        
        # Generate execution order
        self.execution_order = self._generate_execution_order()
        
        return {
            'task_count': len(self.tasks),
            'execution_order': self.execution_order,
            'activation_levels': activated.mean(axis=1).tolist()
        }
    
    async def _update_task_relationships(self, 
                                       activation_matrix: np.ndarray):
        """Update task relationships based on activation patterns."""
        task_ids = list(self.tasks.keys())
        
        for i, task_id in enumerate(task_ids):
            # Find strongly correlated tasks
            correlations = activation_matrix[i] @ activation_matrix.T
            related_indices = np.where(correlations > self.activation_threshold)[0]
            
            # Update task graph
            self.task_graph[task_id] = {
                task_ids[j] for j in related_indices
                if j != i and correlations[j] > self.activation_threshold
            }
    
    def _generate_execution_order(self) -> List[str]:
        """Generate optimal execution order using topological sort."""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(task_id: str):
            if task_id in temp_visited:
                raise ValueError("Cyclic dependency detected")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            # Visit dependencies first
            for dep in self.tasks[task_id].dependencies:
                visit(dep)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            order.append(task_id)
        
        # Sort tasks by priority and visit each
        sorted_tasks = sorted(
            self.tasks.keys(),
            key=lambda x: self.tasks[x].priority.value,
            reverse=True
        )
        
        for task_id in sorted_tasks:
            if task_id not in visited:
                visit(task_id)
        
        return order
    
    async def update_task_status(self,
                               task_id: str,
                               completion: float,
                               resources_used: Dict[str, float]) -> bool:
        """Update task status and adapt mission structure."""
        if task_id not in self.tasks:
            return False
        
        self.completion_status[task_id] = completion
        self.resource_utilization.update(resources_used)
        
        # Adapt neural weights based on performance
        self._adapt_neural_weights(task_id, completion, resources_used)
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now().timestamp(),
            'task_id': task_id,
            'completion': completion,
            'resources': resources_used
        })
        
        return True
    
    def _adapt_neural_weights(self,
                            task_id: str,
                            completion: float,
                            resources_used: Dict[str, float]):
        """Adapt neural weights based on task performance."""
        task_pattern = self.activation_patterns[task_id]
        
        # Calculate performance error
        error = 1.0 - completion
        
        # Update synaptic weights
        weight_update = self.learning_rate * error * \
            np.outer(task_pattern, task_pattern)
        self.synaptic_weights += weight_update
        
        # Normalize weights
        self.synaptic_weights = np.clip(self.synaptic_weights, 0, 1)
    
    def get_mission_status(self) -> Dict[str, Any]:
        """Get current mission status."""
        return {
            'mission_id': self.mission_id,
            'phase': self.current_phase.value,
            'tasks_total': len(self.tasks),
            'tasks_completed': sum(1 for c in self.completion_status.values() 
                                 if c >= 1.0),
            'average_completion': np.mean(list(self.completion_status.values())),
            'resource_utilization': self.resource_utilization,
            'execution_order': self.execution_order,
            'adaptations': len(self.adaptation_history)
        }
    
    async def optimize_mission_parameters(self) -> Dict[str, Any]:
        """Optimize mission parameters using particle swarm optimization."""
        # Initialize particles (possible solutions)
        particles = np.random.uniform(0, 1, (self.population_size, 3))
        velocities = np.zeros_like(particles)
        best_positions = particles.copy()
        best_scores = np.array([self._evaluate_solution(p) for p in particles])
        global_best = best_positions[np.argmax(best_scores)]
        
        for _ in range(self.max_iterations):
            # Update particle velocities and positions
            w = 0.7  # inertia
            c1 = 1.5  # cognitive parameter
            c2 = 1.5  # social parameter
            
            for i in range(self.population_size):
                # Update velocity
                velocities[i] = (w * velocities[i] +
                               c1 * np.random.random() * (best_positions[i] - particles[i]) +
                               c2 * np.random.random() * (global_best - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 1)
                
                # Update best positions
                score = self._evaluate_solution(particles[i])
                if score > best_scores[i]:
                    best_scores[i] = score
                    best_positions[i] = particles[i]
                    
                    if score > self._evaluate_solution(global_best):
                        global_best = particles[i].copy()
            
            # Record optimization progress
            self.optimization_history.append({
                'iteration': len(self.optimization_history),
                'best_score': float(max(best_scores))
            })
        
        # Apply optimized parameters
        self._apply_optimized_parameters(global_best)
        
        return {
            'optimized_parameters': global_best.tolist(),
            'final_score': float(max(best_scores)),
            'iterations': len(self.optimization_history)
        }
    
    def _evaluate_solution(self, parameters: np.ndarray) -> float:
        """Evaluate a potential solution."""
        # Simple evaluation based on resource usage and estimated completion time
        resource_weight = parameters[0]
        time_weight = parameters[1]
        priority_weight = parameters[2]
        
        score = 0.0
        for task in self.tasks.values():
            resource_score = 1.0 - sum(task.resource_requirements.values()) / 100
            time_score = 1.0 - task.estimated_duration / 3600
            priority_score = task.priority.value / 3
            
            score += (resource_weight * resource_score +
                     time_weight * time_score +
                     priority_weight * priority_score)
        
        return score / len(self.tasks) if self.tasks else 0.0
    
    def _apply_optimized_parameters(self, parameters: np.ndarray):
        """Apply optimized parameters to mission structure."""
        # Update learning rate and activation threshold
        self.learning_rate = parameters[0] * 0.1  # Scale to reasonable range
        self.activation_threshold = 0.4 + parameters[1] * 0.4  # Range: 0.4-0.8
        
        # Update synaptic weights scaling
        scale = 0.5 + parameters[2] * 0.5  # Range: 0.5-1.0
        self.synaptic_weights *= scale
