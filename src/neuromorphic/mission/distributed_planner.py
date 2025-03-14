"""
Distributed mission planning system using neuromorphic swarm intelligence.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
from .swarm_behavior import NeuromorphicSwarm, SwarmAgent, SwarmRole
from .mission_hierarchy import MissionHierarchy, TaskNode, TaskPriority


@dataclass
class PlanningTask:
    """Distributed planning task representation."""
    task_id: str
    priority: float
    dependencies: Set[str]
    assigned_agents: List[str]
    status: str
    local_consensus: float
    global_consensus: float


class DistributedPlanner:
    """Distributed mission planning system."""
    
    def __init__(self, mission: MissionHierarchy, swarm: NeuromorphicSwarm):
        self.mission = mission
        self.swarm = swarm
        self.planning_tasks: Dict[str, PlanningTask] = {}
        
        # Planning parameters
        self.consensus_threshold = 0.75
        self.local_window_size = 5
        self.max_iterations = 100
        
        # Consensus matrices
        self.local_consensus = np.zeros((len(swarm.agents), len(mission.tasks)))
        self.global_consensus = np.zeros(len(mission.tasks))
        
        self._initialize_planning_tasks()
    
    def _initialize_planning_tasks(self):
        """Initialize planning tasks from mission hierarchy."""
        for task_id, task in self.mission.tasks.items():
            self.planning_tasks[task_id] = PlanningTask(
                task_id=task_id,
                priority=float(task.priority.value),
                dependencies=task.dependencies.copy(),
                assigned_agents=[],
                status="pending",
                local_consensus=0.0,
                global_consensus=0.0
            )
    
    async def plan_mission(self) -> Dict[str, Any]:
        """Execute distributed mission planning."""
        iteration = 0
        while iteration < self.max_iterations:
            # Distribute tasks to agents
            await self._distribute_tasks()
            
            # Update local consensus
            await self._update_local_consensus()
            
            # Aggregate global consensus
            self._aggregate_global_consensus()
            
            # Check convergence
            if self._check_convergence():
                break
            
            iteration += 1
        
        return self._generate_plan_summary()
    
    async def _distribute_tasks(self):
        """Distribute planning tasks among swarm agents."""
        coordinator_agents = [
            agent for agent in self.swarm.agents
            if agent.role == SwarmRole.COORDINATOR
        ]
        
        for task_id, task in self.planning_tasks.items():
            if task.status == "pending":
                # Find suitable agents based on neural state similarity
                suitable_agents = []
                task_encoding = self._encode_task(self.mission.tasks[task_id])
                
                for agent in coordinator_agents:
                    similarity = np.dot(agent.state_vector, task_encoding) / \
                               (np.linalg.norm(agent.state_vector) * 
                                np.linalg.norm(task_encoding))
                    
                    if similarity > 0.6 and agent.energy > 0.3:
                        suitable_agents.append(agent.agent_id)
                
                if suitable_agents:
                    task.assigned_agents = suitable_agents
                    task.status = "assigned"
    
    async def _update_local_consensus(self):
        """Update local consensus for assigned tasks."""
        for i, agent in enumerate(self.swarm.agents):
            if agent.role != SwarmRole.COORDINATOR:
                continue
            
            for j, (task_id, task) in enumerate(self.planning_tasks.items()):
                if agent.agent_id in task.assigned_agents:
                    # Calculate local consensus based on neural state and dependencies
                    neighbors = self.swarm._get_neighbors(agent)
                    if not neighbors:
                        continue
                    
                    neighbor_states = np.array([
                        n.state_vector for n in neighbors
                    ])
                    
                    local_field = np.mean(neighbor_states, axis=0)
                    task_encoding = self._encode_task(self.mission.tasks[task_id])
                    
                    consensus = np.dot(local_field, task_encoding) / \
                              (np.linalg.norm(local_field) * 
                               np.linalg.norm(task_encoding))
                    
                    self.local_consensus[i, j] = consensus
    
    def _aggregate_global_consensus(self):
        """Aggregate local consensus into global consensus."""
        for j, (task_id, task) in enumerate(self.planning_tasks.items()):
            if task.status == "assigned":
                # Weight local consensus by agent experience
                weighted_consensus = np.sum([
                    self.local_consensus[i, j] * agent.experience
                    for i, agent in enumerate(self.swarm.agents)
                    if agent.agent_id in task.assigned_agents
                ])
                
                task.local_consensus = float(weighted_consensus)
                
                # Update global consensus
                self.global_consensus[j] = np.mean([
                    c for c in self.local_consensus[:, j] if c > 0
                ])
                
                task.global_consensus = float(self.global_consensus[j])
    
    def _check_convergence(self) -> bool:
        """Check if planning has converged."""
        return all(
            task.global_consensus > self.consensus_threshold
            for task in self.planning_tasks.values()
            if task.status == "assigned"
        )
    
    def _encode_task(self, task: TaskNode) -> np.ndarray:
        """Encode task properties into neural vector."""
        encoding = np.zeros(32)
        
        # Encode priority
        encoding[0:8] = task.priority.value / 3
        
        # Encode dependencies
        encoding[8:16] = len(task.dependencies) / 10
        
        # Encode resource requirements
        if task.resource_requirements:
            encoding[16:24] = sum(task.resource_requirements.values()) / 100
        
        # Encode estimated duration
        encoding[24:32] = min(task.estimated_duration / 3600, 1.0)
        
        return encoding
    
    def _generate_plan_summary(self) -> Dict[str, Any]:
        """Generate summary of the distributed planning process."""
        return {
            'planned_tasks': len([
                t for t in self.planning_tasks.values()
                if t.status == "assigned"
            ]),
            'total_tasks': len(self.planning_tasks),
            'average_consensus': float(np.mean([
                t.global_consensus
                for t in self.planning_tasks.values()
                if t.status == "assigned"
            ])),
            'task_assignments': {
                task_id: {
                    'assigned_agents': task.assigned_agents,
                    'consensus': task.global_consensus,
                    'priority': task.priority
                }
                for task_id, task in self.planning_tasks.items()
                if task.status == "assigned"
            }
        }