#!/usr/bin/env python3
"""
Developer test runner for pytest-based suites and import checks.
Relocated from repo root to tools/validation/ to keep the root clean.
"""

import os
import sys
import subprocess
import argparse
import importlib.util
from pathlib import Path


PYTHON = sys.executable or "python3"


def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n🧪 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for auto-update scripts and more")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--auto-only", action="store_true", help="Run only auto-update tests")
    parser.add_argument("--storage", action="store_true", help="Run only storage manager tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Change to project root (tools/validation/ -> tools -> repo root)
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    print(f"📁 Working directory: {project_root}")

    success_count = 0
    total_count = 0

    # Detect pytest-cov availability when --coverage is requested
    has_pytest_cov = importlib.util.find_spec("pytest_cov") is not None

    if args.storage:
        tests = [
            ([PYTHON, "-m", "pytest", "tests/test_storage_manager.py"] + (["-v"] if args.verbose else []),
             "Storage Manager Tests"),
        ]
    elif args.auto_only:
        tests = [
            ([PYTHON, "-m", "pytest", "tests/test_games_auto_update.py"] + (["-v"] if args.verbose else []),
             "Games Auto-Update Tests"),
            ([PYTHON, "-m", "pytest", "tests/test_player_weekly_stats_auto_update.py"] + (["-v"] if args.verbose else []),
             "Player Stats Auto-Update Tests"),
        ]
    else:
        if args.coverage and has_pytest_cov:
            test_cmd = [
                PYTHON, "-m", "pytest", "tests/",
                "--cov=src", "--cov=scripts", "--cov-report=term-missing",
            ]
        else:
            if args.coverage and not has_pytest_cov:
                print("⚠️  pytest-cov not installed; running without coverage. Install 'pytest-cov' to enable.")
            test_cmd = [PYTHON, "-m", "pytest", "tests/"]
        if args.verbose:
            test_cmd.append("-v")
        tests = [(test_cmd, "Full Test Suite")]

    # Add import smoke tests
    tests.extend([
        ([PYTHON, "-c", "import sys; sys.path.append('scripts'); import games_auto_update; print('games_auto_update imported successfully')"],
         "Games Auto-Update Import Test"),
        ([PYTHON, "-c", "import sys; sys.path.append('scripts'); import player_weekly_stats_auto_update; print('player_weekly_stats_auto_update imported successfully')"],
         "Player Stats Auto-Update Import Test"),
    ])

    for cmd, description in tests:
        total_count += 1
        if run_command(cmd, description):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{total_count} tests passed")
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
