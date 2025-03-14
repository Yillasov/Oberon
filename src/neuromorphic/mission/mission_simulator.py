"""
Mission simulation environment for validation and testing.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
from .mission_hierarchy import MissionHierarchy, TaskPriority, TaskNode


@dataclass
class SimulationConfig:
    """Configuration for mission simulation."""
    time_scale: float = 1.0  # Simulation time scaling
    noise_level: float = 0.1  # Environmental noise
    failure_probability: float = 0.05
    resource_variability: float = 0.2
    max_duration: float = 3600  # seconds


class MissionSimulator:
    """Simulation environment for mission validation."""
    
    def __init__(self, mission: MissionHierarchy, config: SimulationConfig):
        self.mission = mission
        self.config = config
        self.simulation_time = 0.0
        self.active_tasks: Dict[str, float] = {}
        self.completed_tasks: List[str] = []
        self.resource_states: Dict[str, float] = {}
        self.simulation_history: List[Dict[str, Any]] = []
        
        # Initialize resource states
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize resource states from mission tasks."""
        all_resources = set()
        for task in self.mission.tasks.values():
            all_resources.update(task.resource_requirements.keys())
        
        for resource in all_resources:
            self.resource_states[resource] = 100.0  # Full capacity
    
    async def run_simulation(self) -> Dict[str, Any]:
        """Run mission simulation."""
        start_time = datetime.now()
        
        while (self.simulation_time < self.config.max_duration and
               len(self.completed_tasks) < len(self.mission.tasks)):
            
            # Update simulation time
            self.simulation_time += self.config.time_scale
            
            # Process active tasks
            await self._process_active_tasks()
            
            # Start new tasks
            await self._start_available_tasks()
            
            # Record simulation state
            self._record_simulation_state()
            
            # Add environmental effects
            self._apply_environmental_effects()
            
            await asyncio.sleep(0.01)  # Prevent blocking
        
        return self._generate_simulation_report(start_time)
    
    async def _process_active_tasks(self):
        """Process currently active tasks."""
        completed = []
        
        for task_id, progress in self.active_tasks.items():
            task = self.mission.tasks[task_id]
            
            # Calculate progress increment
            base_increment = self.config.time_scale / task.estimated_duration
            noise = np.random.normal(0, self.config.noise_level)
            progress_increment = max(0, base_increment + noise)
            
            # Update progress
            new_progress = progress + progress_increment
            
            if new_progress >= 1.0 or self._check_task_completion(task_id):
                completed.append(task_id)
                await self._complete_task(task_id)
            else:
                self.active_tasks[task_id] = new_progress
                # Update resource usage
                self._update_resources(task)
    
    def _check_task_completion(self, task_id: str) -> bool:
        """Check if task meets completion criteria."""
        task = self.mission.tasks[task_id]
        progress = self.active_tasks[task_id]
        
        # Basic completion check with random failure probability
        if np.random.random() < self.config.failure_probability:
            return False
        
        return progress >= 1.0
    
    async def _complete_task(self, task_id: str):
        """Handle task completion."""
        final_progress = self.active_tasks[task_id]
        del self.active_tasks[task_id]
        self.completed_tasks.append(task_id)
        
        # Update mission status
        resources_used = {
            resource: amount * final_progress
            for resource, amount in 
            self.mission.tasks[task_id].resource_requirements.items()
        }
        
        await self.mission.update_task_status(
            task_id, final_progress, resources_used)
    
    async def _start_available_tasks(self):
        """Start new tasks that are available for execution."""
        execution_order = self.mission.execution_order
        
        for task_id in execution_order:
            if (task_id not in self.active_tasks and
                task_id not in self.completed_tasks):
                
                task = self.mission.tasks[task_id]
                
                # Check dependencies
                if all(dep in self.completed_tasks 
                      for dep in task.dependencies):
                    # Check resource availability
                    if self._check_resource_availability(task):
                        self.active_tasks[task_id] = 0.0
    
    def _check_resource_availability(self, task: TaskNode) -> bool:
        """Check if required resources are available."""
        for resource, amount in task.resource_requirements.items():
            if self.resource_states[resource] < amount:
                return False
        return True
    
    def _update_resources(self, task: TaskNode):
        """Update resource states based on task usage."""
        for resource, amount in task.resource_requirements.items():
            usage = amount * self.config.time_scale / task.estimated_duration
            variation = np.random.uniform(
                -self.config.resource_variability,
                self.config.resource_variability
            )
            actual_usage = usage * (1 + variation)
            self.resource_states[resource] = max(
                0, self.resource_states[resource] - actual_usage)
    
    def _apply_environmental_effects(self):
        """Apply environmental effects to simulation."""
        for resource in self.resource_states:
            noise = np.random.normal(0, self.config.noise_level)
            self.resource_states[resource] = np.clip(
                self.resource_states[resource] + noise, 0, 100)
    
    def _record_simulation_state(self):
        """Record current simulation state."""
        self.simulation_history.append({
            'time': self.simulation_time,
            'active_tasks': self.active_tasks.copy(),
            'completed_tasks': self.completed_tasks.copy(),
            'resource_states': self.resource_states.copy()
        })
    
    def _generate_simulation_report(self, 
                                  start_time: datetime) -> Dict[str, Any]:
        """Generate final simulation report."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            'simulation_time': self.simulation_time,
            'real_time_duration': duration,
            'completed_tasks': len(self.completed_tasks),
            'total_tasks': len(self.mission.tasks),
            'completion_rate': len(self.completed_tasks) / len(self.mission.tasks),
            'resource_efficiency': self._calculate_resource_efficiency(),
            'timeline': [
                {
                    'time': state['time'],
                    'active_count': len(state['active_tasks']),
                    'completed_count': len(state['completed_tasks'])
                }
                for state in self.simulation_history
            ]
        }
    
    def _calculate_resource_efficiency(self) -> float:
        """Calculate overall resource usage efficiency."""
        if not self.completed_tasks:
            return 0.0
        
        total_efficiency = 0.0
        for resource, final_state in self.resource_states.items():
            initial_state = 100.0
            efficiency = 1.0 - (final_state / initial_state)
            total_efficiency += efficiency
        
        return total_efficiency / len(self.resource_states)