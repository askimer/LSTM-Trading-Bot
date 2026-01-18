#!/usr/bin/env python3
"""
Test runner for RL Trading Bot
Executes all tests and provides a summary of results
"""

import unittest
import sys
import os
import argparse
from datetime import datetime


def run_all_tests():
    """Run all tests in the tests directory"""
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    # Create a test runner with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        buffer=True
    )
    
    print(f"Starting test execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    result = runner.run(suite)
    
    print("=" * 70)
    print(f"Test execution completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%" if result.testsRun > 0 else "0.00%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}:")
            print(f"    {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}:")
            print(f"    {traceback}")
    
    return result.wasSuccessful()


def run_specific_test(test_name):
    """Run a specific test module"""
    loader = unittest.TestLoader()
    
    try:
        # Try to load the specific test
        suite = loader.loadTestsFromName(f"tests.{test_name}")
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except AttributeError:
        print(f"Test module '{test_name}' not found in tests directory")
        return False


def list_available_tests():
    """List all available test modules"""
    test_dir = 'tests'
    if not os.path.exists(test_dir):
        print(f"Tests directory '{test_dir}' does not exist")
        return
    
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    if not test_files:
        print("No test files found in tests directory")
        return
    
    print("Available test modules:")
    for test_file in sorted(test_files):
        module_name = test_file[:-3]  # Remove .py extension
        print(f"  - {module_name}")


def main():
    parser = argparse.ArgumentParser(description='Test runner for RL Trading Bot')
    parser.add_argument('--list', action='store_true', help='List all available tests')
    parser.add_argument('--test', type=str, help='Run a specific test module (e.g., test_risk_management)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_tests()
        return
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
