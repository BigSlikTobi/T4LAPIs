"""Test runner for LLM functionality tests."""

import os
import sys
import unittest
import logging

# Set up test environment
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during testing
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_llm_tests():
    """Run all LLM-related tests."""
    # Discover and run all LLM tests
    loader = unittest.TestLoader()
    
    # Load specific test modules
    test_modules = [
        'tests.test_llm_init',
        'tests.test_llm_entity_linker'
    ]
    
    suite = unittest.TestSuite()
    total_tests = 0
    
    for module_name in test_modules:
        try:
            module_suite = loader.loadTestsFromName(module_name)
            test_count = module_suite.countTestCases()
            suite.addTests(module_suite)
            total_tests += test_count
            print(f"✅ Loaded {test_count} tests from {module_name}")
        except Exception as e:
            print(f"❌ Failed to load tests from {module_name}: {e}")
    
    print(f"\n📊 Total tests loaded: {total_tests}")
    
    # Run the tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"LLM TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception: ')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == '__main__':
    success = run_llm_tests()
    sys.exit(0 if success else 1)
