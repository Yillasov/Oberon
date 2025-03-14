"""
Simplified incremental testing methodology for safe field validation.
"""
import os
import json
import time
import logging
from enum import Enum
from typing import Dict, List, Callable, Any, Optional


class TestPhase(Enum):
    """Test phases in order of increasing complexity."""
    SMOKE = 0      # Basic functionality tests
    SANITY = 1     # Core feature tests
    REGRESSION = 2 # Previous issue tests
    PERFORMANCE = 3 # Performance validation


class FieldValidator:
    """Simple incremental field validation framework."""
    
    def __init__(self, system=None):
        self.system = system
        self.logger = logging.getLogger("field_validator")
        self.test_registry = {}
        self.results = {}
        
        # Create results directory
        self.results_dir = "/Users/yessine/Oberon/test_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def register_test(self, name: str, phase: TestPhase, test_func: Callable):
        """Register a test with a specific phase."""
        if phase not in self.test_registry:
            self.test_registry[phase] = []
            
        self.test_registry[phase].append({
            "name": name,
            "func": test_func
        })
        
    def run_test(self, test_info: Dict) -> Dict:
        """Run a single test and return result."""
        name = test_info["name"]
        func = test_info["func"]
        
        self.logger.info(f"Running test: {name}")
        start_time = time.time()
        
        try:
            result = func(self.system)
            success = bool(result)
            error = None
        except Exception as e:
            success = False
            error = str(e)
            self.logger.error(f"Test failed: {name} - {error}")
            
        duration = time.time() - start_time
        
        return {
            "name": name,
            "success": success,
            "duration": duration,
            "error": error,
            "timestamp": time.time()
        }
    
    def run_phase(self, phase: TestPhase) -> bool:
        """Run all tests for a specific phase."""
        if phase not in self.test_registry:
            self.logger.warning(f"No tests registered for phase: {phase.name}")
            return True
            
        self.logger.info(f"Starting test phase: {phase.name}")
        phase_results = []
        
        for test_info in self.test_registry[phase]:
            result = self.run_test(test_info)
            phase_results.append(result)
            
            # Store in overall results
            self.results[test_info["name"]] = result
            
        # Save phase results
        self._save_phase_results(phase, phase_results)
        
        # Check if all tests passed
        return all(r["success"] for r in phase_results)
    
    def _save_phase_results(self, phase: TestPhase, results: List[Dict]):
        """Save phase results to file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{phase.name.lower()}_tests_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        summary = {
            "phase": phase.name,
            "timestamp": time.time(),
            "total": len(results),
            "passed": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Phase {phase.name}: {summary['passed']}/{summary['total']} tests passed")
    
    def validate_incrementally(self, stop_on_failure: bool = True) -> bool:
        """Run all test phases in order of increasing complexity."""
        self.logger.info("Starting incremental validation")
        self.results = {}
        
        for phase in sorted(TestPhase, key=lambda p: p.value):
            phase_passed = self.run_phase(phase)
            
            if not phase_passed and stop_on_failure:
                self.logger.error(f"Validation failed at phase: {phase.name}")
                return False
                
        return all(result["success"] for result in self.results.values())
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        if not self.results:
            return {"status": "No tests run"}
            
        return {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results.values() if r["success"]),
            "failed": sum(1 for r in self.results.values() if not r["success"]),
            "phases": {
                phase.name: {
                    "passed": sum(1 for name, r in self.results.items() 
                              if r["success"] and any(t["name"] == name for t in self.test_registry.get(phase, []))),
                    "total": len(self.test_registry.get(phase, []))
                }
                for phase in TestPhase
            }
        }


# Example test functions
def test_system_online(system):
    """Verify system is online and responsive."""
    return system is not None

def test_components_registered(system):
    """Verify core components are registered."""
    if not hasattr(system, "components"):
        return False
    required = ["system_core", "event_bus", "health_monitor"]
    return all(comp in system.components for comp in required)

def test_event_publishing(system):
    """Test basic event publishing."""
    if not hasattr(system, "event_bus"):
        return False
    try:
        system.event_bus.publish("test/event", {"test": True})
        return True
    except:
        return False


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create validator with mock system
    mock_system = type('MockSystem', (), {
        'components': {'system_core': {}, 'event_bus': {}, 'health_monitor': {}},
        'event_bus': type('MockEventBus', (), {'publish': lambda topic, data: None})
    })()
    
    validator = FieldValidator(mock_system)
    
    # Register tests
    validator.register_test("System Online", TestPhase.SMOKE, test_system_online)
    validator.register_test("Components Registered", TestPhase.SMOKE, test_components_registered)
    validator.register_test("Event Publishing", TestPhase.SANITY, test_event_publishing)
    
    # Run validation
    success = validator.validate_incrementally()
    print(f"Validation {'succeeded' if success else 'failed'}")
    print(json.dumps(validator.get_validation_summary(), indent=2))