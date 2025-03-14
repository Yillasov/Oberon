"""
Basic health monitoring for neuromorphic system components.
"""
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from .system_core import SystemCore, ComponentStatus
from .component_registry import ComponentRegistry, ComponentMetadata


@dataclass
class HealthMetric:
    """Health metric with thresholds."""
    name: str
    value: float
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    def get_status(self) -> ComponentStatus:
        """Determine status based on thresholds."""
        if self.critical_threshold is not None:
            if (self.critical_threshold > 0 and self.value >= self.critical_threshold) or \
               (self.critical_threshold < 0 and self.value <= self.critical_threshold):
                return ComponentStatus.ERROR
                
        if self.warning_threshold is not None:
            if (self.warning_threshold > 0 and self.value >= self.warning_threshold) or \
               (self.warning_threshold < 0 and self.value <= self.warning_threshold):
                return ComponentStatus.DEGRADED
                
        return ComponentStatus.ONLINE


@dataclass
class ComponentHealth:
    """Health information for a component."""
    name: str
    status: ComponentStatus
    last_check: float
    metrics: Dict[str, HealthMetric]
    message: str = ""


class HealthMonitor:
    """Basic health monitoring for system components."""
    
    def __init__(self, system_core: SystemCore, registry: ComponentRegistry):
        self.core = system_core
        self.registry = registry
        self.logger = system_core.logger
        
        # Health status storage
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # Monitoring settings
        self.check_interval = 10  # seconds
        self.running = False
        self.monitor_task = None
        
        # Alert callbacks
        self.alert_handlers: List[Callable[[str, ComponentStatus, str], None]] = []
        
        # Register with system
        self.core.register_component("health_monitor", self)
        self.registry.register(
            self,
            ComponentMetadata(
                name="health_monitor",
                component_type="integration",
                version="1.0",
                provides={"health_monitoring"},
                startup_priority=80
            )
        )
    
    async def start(self):
        """Start health monitoring."""
        self.logger.info("Health monitor starting")
        self.running = True
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop health monitoring."""
        self.logger.info("Health monitor stopping")
        self.running = False
        
        if self.monitor_task:
            try:
                self.monitor_task.cancel()
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    def add_alert_handler(self, handler: Callable[[str, ComponentStatus, str], None]):
        """Add handler for health alerts."""
        self.alert_handlers.append(handler)
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            while self.running:
                await self._check_all_components()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            self.logger.info("Health monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in health monitoring loop: {str(e)}")
    
    async def _check_all_components(self):
        """Check health of all registered components."""
        for name, component_data in self.core.components.items():
            try:
                await self._check_component(name, component_data)
            except Exception as e:
                self.logger.error(f"Error checking component {name}: {str(e)}")
    
    async def _check_component(self, name: str, component_data: Dict[str, Any]):
        """Check health of a specific component."""
        component = component_data.get("instance")
        if not component:
            return
            
        # Get component status
        current_status = component_data.get("status", ComponentStatus.OFFLINE)
        
        # Initialize metrics
        metrics = {}
        
        # Add basic uptime metric
        uptime = time.time() - component_data.get("last_update", time.time())
        metrics["uptime"] = HealthMetric(
            name="uptime",
            value=uptime,
            warning_threshold=None,
            critical_threshold=None
        )
        
        # Check if component has health check method
        message = ""
        if hasattr(component, "health_check") and callable(component.health_check):
            try:
                if asyncio.iscoroutinefunction(component.health_check):
                    health_result = await component.health_check()
                else:
                    health_result = component.health_check()
                    
                if isinstance(health_result, dict):
                    # Process component metrics
                    for metric_name, metric_data in health_result.get("metrics", {}).items():
                        if isinstance(metric_data, dict):
                            metrics[metric_name] = HealthMetric(
                                name=metric_name,
                                value=metric_data.get("value", 0),
                                warning_threshold=metric_data.get("warning_threshold"),
                                critical_threshold=metric_data.get("critical_threshold")
                            )
                        else:
                            # Simple value
                            metrics[metric_name] = HealthMetric(
                                name=metric_name,
                                value=float(metric_data),
                                warning_threshold=None,
                                critical_threshold=None
                            )
                    
                    message = health_result.get("message", "")
                    
                    # Update status if provided
                    if "status" in health_result:
                        current_status = health_result["status"]
                        
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {str(e)}")
                current_status = ComponentStatus.ERROR
                message = f"Health check error: {str(e)}"
        
        # Determine worst status from metrics
        metric_status = ComponentStatus.ONLINE
        for metric in metrics.values():
            status = metric.get_status()
            if status.value > metric_status.value:
                metric_status = status
        
        # Use the worse of component status and metric status
        if metric_status.value > current_status.value:
            status = metric_status
        else:
            status = current_status
        
        # Check for status change
        previous_health = self.component_health.get(name)
        status_changed = previous_health is None or previous_health.status != status
        
        # Update health status
        self.component_health[name] = ComponentHealth(
            name=name,
            status=status,
            last_check=time.time(),
            metrics=metrics,
            message=message
        )
        
        # Update component status in core
        component_data["status"] = status
        
        # Send alerts if status changed to ERROR or DEGRADED
        if status_changed and status in [ComponentStatus.ERROR, ComponentStatus.DEGRADED]:
            self._send_alert(name, status, message)
    
    def _send_alert(self, component_name: str, status: ComponentStatus, message: str):
        """Send health alert to registered handlers."""
        self.logger.warning(f"Health alert for {component_name}: {status.name} - {message}")
        
        for handler in self.alert_handlers:
            try:
                handler(component_name, status, message)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {str(e)}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        # Count components by status
        status_counts = {status.name: 0 for status in ComponentStatus}
        for health in self.component_health.values():
            status_counts[health.status.name] += 1
        
        # Determine overall system status
        if status_counts[ComponentStatus.ERROR.name] > 0:
            overall_status = ComponentStatus.DEGRADED
        elif status_counts[ComponentStatus.OFFLINE.name] == len(self.component_health):
            overall_status = ComponentStatus.OFFLINE
        elif status_counts[ComponentStatus.DEGRADED.name] > 0:
            overall_status = ComponentStatus.DEGRADED
        else:
            overall_status = ComponentStatus.ONLINE
        
        return {
            "status": overall_status.name,
            "component_count": len(self.component_health),
            "status_counts": status_counts,
            "timestamp": time.time()
        }
    
    def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get health status for a specific component."""
        if component_name not in self.component_health:
            return None
            
        health = self.component_health[component_name]
        return {
            "status": health.status.name,
            "last_check": health.last_check,
            "metrics": {name: metric.value for name, metric in health.metrics.items()},
            "message": health.message
        }
    
    def get_detailed_metrics(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a component."""
        if component_name not in self.component_health:
            return None
            
        health = self.component_health[component_name]
        return {
            name: {
                "value": metric.value,
                "warning_threshold": metric.warning_threshold,
                "critical_threshold": metric.critical_threshold,
                "status": metric.get_status().name
            }
            for name, metric in health.metrics.items()
        }