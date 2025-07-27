"""CLI utility functions and argument parsing patterns."""

import argparse
import sys
from typing import Callable, Any, Dict, Optional
from .logging import setup_logging


def setup_cli_parser(description: str, 
                    add_common_args: bool = True) -> argparse.ArgumentParser:
    """Create a standardized CLI argument parser.
    
    Args:
        description: Description of the script/command
        add_common_args: Whether to add common arguments (dry-run, verbose, etc.)
        
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description=description)
    
    if add_common_args:
        parser.add_argument(
            "--dry-run", 
            action="store_true", 
            help="Show what would be done without actually doing it"
        )
        parser.add_argument(
            "--clear", 
            action="store_true", 
            help="Clear existing data before loading"
        )
        parser.add_argument(
            "--verbose", "-v", 
            action="store_true", 
            help="Enable verbose logging"
        )
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set logging level"
        )
    
    return parser


def handle_cli_errors(func: Callable) -> Callable:
    """Decorator to handle common CLI errors and exit codes.
    
    Args:
        func: The main CLI function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs) -> int:
        try:
            result = func(*args, **kwargs)
            return 0 if result else 1
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 1
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return 1
    
    return wrapper


def setup_cli_logging(args: argparse.Namespace) -> None:
    """Set up logging based on CLI arguments.
    
    Args:
        args: Parsed command line arguments
    """
    log_level = "DEBUG" if getattr(args, 'verbose', False) else getattr(args, 'log_level', 'INFO')
    setup_logging(level=log_level)


def print_results(result: Dict[str, Any], 
                 operation: str = "operation",
                 dry_run: bool = False) -> None:
    """Print standardized results from data operations.
    
    Args:
        result: Result dictionary from data operation
        operation: Name of the operation for display
        dry_run: Whether this was a dry run
    """
    if result.get("success"):
        if dry_run:
            print(f"DRY RUN - Would perform {operation}")
            if "would_upsert" in result:
                print(f"Would upsert {result['would_upsert']} records")
            if "would_clear" in result:
                print(f"Would clear table: {result['would_clear']}")
            if "sample_record" in result and result["sample_record"]:
                print(f"Sample record: {result['sample_record']}")
        else:
            print(f"✅ Successfully completed {operation}")
            if "total_fetched" in result:
                print(f"Fetched: {result['total_fetched']} records")
            if "total_validated" in result:
                print(f"Validated: {result['total_validated']} records")
            if "upsert_result" in result and "affected_rows" in result["upsert_result"]:
                print(f"Upserted: {result['upsert_result']['affected_rows']} records")
    else:
        error_msg = result.get("error", result.get("message", "Unknown error"))
        print(f"❌ {operation.capitalize()} failed: {error_msg}")


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask user for confirmation.
    
    Args:
        message: Confirmation message to display
        default: Default value if user just presses enter
        
    Returns:
        True if user confirms, False otherwise
    """
    suffix = " [Y/n]" if default else " [y/N]"
    response = input(f"{message}{suffix}: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes', 'true', '1']
