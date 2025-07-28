#!/usr/bin/env python3
"""
Test runner for auto-update scripts.
Provides a convenient way to run tests locally with options.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n🧪 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code {e.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for auto-update scripts")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--auto-only", action="store_true", help="Run only auto-update tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    print(f"📁 Working directory: {project_root}")
    
    success_count = 0
    total_count = 0
    
    if args.auto_only:
        # Run only auto-update script tests
        tests = [
            (["python", "-m", "pytest", "tests/test_games_auto_update.py"] + 
             (["-v"] if args.verbose else []), "Games Auto-Update Tests"),
            (["python", "-m", "pytest", "tests/test_player_weekly_stats_auto_update.py"] + 
             (["-v"] if args.verbose else []), "Player Stats Auto-Update Tests"),
        ]
    else:
        # Run all tests
        if args.coverage:
            test_cmd = ["python", "-m", "pytest", "tests/", "--cov=src", "--cov=scripts", "--cov-report=term-missing"]
        else:
            test_cmd = ["python", "-m", "pytest", "tests/"]
        
        if args.verbose:
            test_cmd.append("-v")
            
        tests = [(test_cmd, "Full Test Suite")]
    
    # Add import tests
    tests.extend([
        (["python", "-c", "import sys; sys.path.append('scripts'); import games_auto_update; print('games_auto_update imported successfully')"], 
         "Games Auto-Update Import Test"),
        (["python", "-c", "import sys; sys.path.append('scripts'); import player_weekly_stats_auto_update; print('player_weekly_stats_auto_update imported successfully')"], 
         "Player Stats Auto-Update Import Test"),
    ])
    
    for cmd, description in tests:
        total_count += 1
        if run_command(cmd, description):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"📊 Test Results: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total_count - success_count} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
