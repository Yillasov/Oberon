"""
Simple automated regression testing for field updates.
"""
import os
import json
import time
import logging
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Set
from .incremental_validator import FieldValidator, TestPhase
from ..config.config_manager import get_config


class RegressionTester:
    """Automated regression testing for field updates."""
    
    def __init__(self, system=None, base_path=None):
        self.system = system
        self.config = get_config()
        
        # Use configuration or fallback to provided base_path
        self.base_path = base_path or self.config.get("paths.base")
        self.logger = logging.getLogger("regression_tester")
        
        # Initialize validator
        self.validator = FieldValidator(system)
        
        # Regression history
        self.history_dir = self.config.get(
            "paths.regression_history", 
            os.path.join(self.base_path, "test_results", "regression_history")
        )
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Known good test results
        self.baseline_path = os.path.join(self.history_dir, "baseline.json")
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> Dict:
        """Load baseline test results."""
        if os.path.exists(self.baseline_path):
            try:
                with open(self.baseline_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading baseline: {e}")
        
        return {"tests": {}, "version": "0.0.0"}
    
    def save_baseline(self, version: str) -> bool:
        """Save current test results as baseline."""
        # Run all tests
        self.validator.validate_incrementally(stop_on_failure=False)
        
        # Save results as baseline
        baseline = {
            "version": version,
            "timestamp": time.time(),
            "tests": self.validator.results
        }
        
        try:
            with open(self.baseline_path, 'w') as f:
                json.dump(baseline, f, indent=2)
            self.baseline = baseline
            self.logger.info(f"Saved baseline for version {version}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving baseline: {e}")
            return False
    
    def test_update(self, version: str) -> Dict[str, Any]:
        """Test an update against baseline."""
        self.logger.info(f"Testing update to version {version}")
        
        # Run regression tests
        self.validator.validate_incrementally(stop_on_failure=False)
        
        # Compare with baseline
        regression_results = self._compare_with_baseline()
        
        # Save results
        self._save_regression_results(version, regression_results)
        
        return regression_results
    
    def _compare_with_baseline(self) -> Dict[str, Any]:
        """Compare current test results with baseline."""
        if not self.baseline or "tests" not in self.baseline:
            return {"status": "error", "message": "No baseline available"}
        
        baseline_tests = self.baseline.get("tests", {})
        current_tests = self.validator.results
        
        # Find tests that were passing but now fail
        regressions = []
        for name, baseline_result in baseline_tests.items():
            if baseline_result.get("success", False):
                current_result = current_tests.get(name)
                if current_result and not current_result.get("success", False):
                    regressions.append({
                        "test": name,
                        "error": current_result.get("error", "Unknown error")
                    })
        
        # Find new tests that fail
        new_failures = []
        for name, result in current_tests.items():
            if name not in baseline_tests and not result.get("success", False):
                new_failures.append({
                    "test": name,
                    "error": result.get("error", "Unknown error")
                })
        
        return {
            "baseline_version": self.baseline.get("version", "unknown"),
            "total_tests": len(current_tests),
            "passed": sum(1 for r in current_tests.values() if r.get("success", False)),
            "failed": sum(1 for r in current_tests.values() if not r.get("success", False)),
            "regressions": regressions,
            "new_failures": new_failures,
            "has_regression": len(regressions) > 0
        }
    
    def _save_regression_results(self, version: str, results: Dict[str, Any]) -> None:
        """Save regression test results."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"regression_{version}_{timestamp}.json"
        filepath = os.path.join(self.history_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "version": version,
                    "timestamp": time.time(),
                    "results": results,
                    "tests": self.validator.results
                }, f, indent=2)
            self.logger.info(f"Saved regression results for version {version}")
        except Exception as e:
            self.logger.error(f"Error saving regression results: {e}")
    
    def register_regression_test(self, name: str, test_func) -> None:
        """Register a regression test."""
        self.validator.register_test(name, TestPhase.REGRESSION, test_func)
    
    def is_update_safe(self, version: str) -> bool:
        """Test if an update is safe to apply."""
        results = self.test_update(version)
        return not results.get("has_regression", False)


# Example regression tests
def test_memory_usage(system):
    """Test memory usage is within acceptable limits."""
    # Simple mock implementation
    return True

def test_response_time(system):
    """Test system response time is acceptable."""
    # Simple mock implementation
    start = time.time()
    # Simulate system operation
    time.sleep(0.01)
    return (time.time() - start) < 0.1


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create mock system
    mock_system = type('MockSystem', (), {
        'components': {'system_core': {}, 'event_bus': {}, 'health_monitor': {}},
        'event_bus': type('MockEventBus', (), {'publish': lambda topic, data: None})
    })()
    
    # Create regression tester
    tester = RegressionTester(mock_system)
    
    # Register regression tests
    tester.register_regression_test("Memory Usage", test_memory_usage)
    tester.register_regression_test("Response Time", test_response_time)
    
    # Save baseline for current version
    tester.save_baseline("1.0.0")
    
    # Test update
    results = tester.test_update("1.1.0")
    print(json.dumps(results, indent=2))
    
    # Check if update is safe
    is_safe = tester.is_update_safe("1.1.0")
    print(f"Update to 1.1.0 is {'safe' if is_safe else 'NOT safe'} to apply")