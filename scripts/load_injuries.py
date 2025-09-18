#!/usr/bin/env python3
"""
CLI script for loading NFL injury data using the core data loader.
Replaces the standalone injury_updates/main.py script.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.injuries import InjuriesDataLoader
from src.core.utils.cli import setup_cli_parser, handle_cli_errors, setup_cli_logging, print_results


@handle_cli_errors
def main():
    """Main CLI function for loading injury data."""
    parser = setup_cli_parser("Load NFL injury data into the database")
    
    # Add injury-specific arguments
    parser.add_argument(
        "--years", 
        type=int, 
        nargs="+", 
        default=[2024],
        help="Years to load injury data for (default: 2024)"
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Version number for the data (auto-calculated if not provided)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to process per batch (default: 1000)"
    )
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    print(f"Loading injury data for years: {args.years}")
    
    try:
        # Create loader
        loader = InjuriesDataLoader()
        
        # Get version if not provided
        if args.version is None:
            version = loader.get_next_version()
            print(f"Auto-calculated version: {version}")
        else:
            version = args.version
            print(f"Using specified version: {version}")
        
        # Load data
        result = loader.load_injuries(
            years=args.years,
            dry_run=args.dry_run,
            clear_table=args.clear,
            version=version,
            batch_size=args.batch_size
        )
        
        # Print results
        if result["success"]:
            print_results(result, "injury data loading", args.dry_run)
            return True
        else:
            print(f"Error loading injury data: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    exit(main())