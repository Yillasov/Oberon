#!/usr/bin/env python3
"""
Script to run regression tests for CI/CD pipeline.
"""
import os
import sys
import logging
from ..config.config_manager import get_config
from .regression_tester import RegressionTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("regression_test_runner")

def main():
    """Run regression tests and report results."""
    logger.info("Starting regression tests")
    
    # Get configuration
    config = get_config()
    
    # Initialize regression tester
    tester = RegressionTester()
    
    # Get current version from environment or use default
    version = os.environ.get("OBERON_VERSION", "dev-build")
    
    # Run tests
    results = tester.test_update(version)
    
    # Log results summary
    logger.info(f"Tests completed: {results['total_tests']} total, "
                f"{results['passed']} passed, {results['failed']} failed")
    
    # Check for regressions
    if results.get('regressions'):
        logger.error(f"Found {len(results['regressions'])} regressions:")
        for reg in results['regressions']:
            logger.error(f"  - {reg['test']}: {reg['error']}")
        sys.exit(1)
    
    # Check for new failures
    if results.get('new_failures'):
        logger.warning(f"Found {len(results['new_failures'])} new failures:")
        for fail in results['new_failures']:
            logger.warning(f"  - {fail['test']}: {fail['error']}")
    
    logger.info("Regression tests completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main()