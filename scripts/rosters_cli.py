#!/usr/bin/env python3
"""
CLI script for loading NFL rosters data.
Supports loading by season with versioning and batch processing.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.rosters import RostersDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the rosters data loader."""
    parser = setup_cli_parser("Load NFL rosters data into the database")
    parser.add_argument("years", type=int, nargs="+", help="NFL season years (e.g., 2024 or 2023 2024)")
    parser.add_argument("--version", type=int, help="Version number to assign (auto-generated if not specified)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of records to process in each batch")
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("🏈 NFL Rosters Data Loader")
        print(f"Loading roster data for years {args.years}")
        
        # Create loader and run
        loader = RostersDataLoader()
        
        # Get current max version for display
        current_max_version = loader.get_current_max_version()
        version_to_use = args.version if args.version else current_max_version + 1
        
        print(f"Current max version in database: {current_max_version}")
        print(f"Version for this load: {version_to_use}")
        
        result = loader.load_rosters(
            years=args.years,
            dry_run=args.dry_run,
            version=args.version,
            batch_size=args.batch_size
        )
        
        # Print results using utility function
        operation = f"rosters data load for years {args.years}"
        print_results(result, operation, args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current roster record count
            total_rosters = loader.get_record_count()
            print(f"Total roster records in database: {total_rosters}")
            
            if "version" in result:
                print(f"Data loaded with version: {result['version']}")
        
        return result["success"]
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)