"""
Validation framework for neuromorphic system components.
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from .system_core import SystemCore, ComponentStatus
from .component_registry import ComponentRegistry, ComponentMetadata


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass
class ValidationResult:
    """Result of a validation check."""
    component: str
    check_name: str
    level: ValidationLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_error(self) -> bool:
        return self.level == ValidationLevel.ERROR
    
    @property
    def is_warning(self) -> bool:
        return self.level == ValidationLevel.WARNING


class ValidationFramework:
    """Framework for validating system components."""
    
    def __init__(self, system_core: SystemCore, registry: ComponentRegistry):
        self.core = system_core
        self.registry = registry
        self.logger = system_core.logger
        
        # Validation storage
        self.validators: Dict[str, List[Callable]] = {}
        self.results: Dict[str, List[ValidationResult]] = {}
        
        # Register with system
        self.core.register_component("validation_framework", self)
        self.registry.register(
            self,
            ComponentMetadata(
                name="validation_framework",
                component_type="integration",
                version="1.0",
                provides={"validation_service"},
                dependencies={"health_monitoring"},
                startup_priority=75
            )
        )
    
    async def start(self):
        """Start validation framework."""
        self.logger.info("Validation framework starting")
    
    async def stop(self):
        """Stop validation framework."""
        self.logger.info("Validation framework stopping")
    
    def register_validator(self, component_name: str, validator: Callable) -> bool:
        """Register a validation function for a component."""
        if component_name not in self.validators:
            self.validators[component_name] = []
            
        self.validators[component_name].append(validator)
        return True
    
    async def validate_component(self, component_name: str) -> List[ValidationResult]:
        """Run all validators for a component."""
        if component_name not in self.validators:
            return []
            
        component_data = self.core.components.get(component_name)
        if not component_data:
            return [ValidationResult(
                component=component_name,
                check_name="component_exists",
                level=ValidationLevel.ERROR,
                message=f"Component {component_name} not found in system"
            )]
            
        component = component_data.get("instance")
        results = []
        
        for validator in self.validators[component_name]:
            try:
                if asyncio.iscoroutinefunction(validator):
                    result = await validator(component)
                else:
                    result = validator(component)
                    
                if isinstance(result, list):
                    results.extend(result)
                elif isinstance(result, ValidationResult):
                    results.append(result)
                    
            except Exception as e:
                results.append(ValidationResult(
                    component=component_name,
                    check_name=validator.__name__,
                    level=ValidationLevel.ERROR,
                    message=f"Validator failed: {str(e)}",
                    details={"exception": str(e)}
                ))
        
        # Store results
        self.results[component_name] = results
        
        # Log errors and warnings
        for result in results:
            if result.is_error:
                self.logger.error(f"Validation error in {component_name}: {result.message}")
            elif result.is_warning:
                self.logger.warning(f"Validation warning in {component_name}: {result.message}")
        
        return results
    
    async def validate_system(self) -> Dict[str, List[ValidationResult]]:
        """Validate all components in the system."""
        all_results = {}
        
        # Validate registered components
        for component_name in self.validators.keys():
            results = await self.validate_component(component_name)
            all_results[component_name] = results
        
        # Check for components without validators
        for component_name in self.core.components:
            if component_name not in self.validators:
                self.logger.info(f"Component {component_name} has no validators")
        
        return all_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        error_count = 0
        warning_count = 0
        info_count = 0
        
        components_with_errors = set()
        
        for component, results in self.results.items():
            for result in results:
                if result.level == ValidationLevel.ERROR:
                    error_count += 1
                    components_with_errors.add(component)
                elif result.level == ValidationLevel.WARNING:
                    warning_count += 1
                elif result.level == ValidationLevel.INFO:
                    info_count += 1
        
        return {
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "components_with_errors": list(components_with_errors),
            "validated_components": list(self.results.keys()),
            "timestamp": time.time()
        }
    
    def create_standard_validators(self, component_name: str) -> None:
        """Create standard validators for common component interfaces."""
        component_data = self.core.components.get(component_name)
        if not component_data:
            return
            
        component = component_data.get("instance")
        
        # Register standard validators
        self.register_validator(component_name, self._validate_lifecycle_methods)
        
        # Register health check validator if component has health_check method
        if hasattr(component, "health_check") and callable(component.health_check):
            self.register_validator(component_name, self._validate_health_check)
    
    def _validate_lifecycle_methods(self, component: Any) -> List[ValidationResult]:
        """Validate that component has proper lifecycle methods."""
        results = []
        component_name = None
        
        # Find component name
        for name, data in self.core.components.items():
            if data.get("instance") == component:
                component_name = name
                break
                
        if not component_name:
            return [ValidationResult(
                component="unknown",
                check_name="find_component",
                level=ValidationLevel.ERROR,
                message="Could not determine component name"
            )]
        
        # Check for start method
        if not hasattr(component, "start") or not callable(component.start):
            results.append(ValidationResult(
                component=component_name,
                check_name="has_start_method",
                level=ValidationLevel.WARNING,
                message=f"Component {component_name} missing start method"
            ))
        
        # Check for stop method
        if not hasattr(component, "stop") or not callable(component.stop):
            results.append(ValidationResult(
                component=component_name,
                check_name="has_stop_method",
                level=ValidationLevel.WARNING,
                message=f"Component {component_name} missing stop method"
            ))
        
        return results
    
    async def _validate_health_check(self, component: Any) -> List[ValidationResult]:
        """Validate component's health check method."""
        results = []
        component_name = None
        
        # Find component name
        for name, data in self.core.components.items():
            if data.get("instance") == component:
                component_name = name
                break
                
        if not component_name:
            return [ValidationResult(
                component="unknown",
                check_name="find_component",
                level=ValidationLevel.ERROR,
                message="Could not determine component name"
            )]
        
        # Call health check
        try:
            if asyncio.iscoroutinefunction(component.health_check):
                health_result = await component.health_check()
            else:
                health_result = component.health_check()
                
            if not isinstance(health_result, dict):
                results.append(ValidationResult(
                    component=component_name,
                    check_name="health_check_format",
                    level=ValidationLevel.ERROR,
                    message=f"Health check should return a dictionary, got {type(health_result)}"
                ))
            else:
                # Check for required fields
                if "metrics" not in health_result:
                    results.append(ValidationResult(
                        component=component_name,
                        check_name="health_check_metrics",
                        level=ValidationLevel.WARNING,
                        message="Health check missing 'metrics' field"
                    ))
                
        except Exception as e:
            results.append(ValidationResult(
                component=component_name,
                check_name="health_check_execution",
                level=ValidationLevel.ERROR,
                message=f"Health check failed: {str(e)}",
                details={"exception": str(e)}
            ))
        
        return results