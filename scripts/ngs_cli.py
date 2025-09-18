#!/usr/bin/env python3
"""
CLI script for loading NFL Next Gen Stats data.
Supports loading different stat types (passing, rushing, receiving) by season.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.ngs import NextGenStatsDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the Next Gen Stats data loader."""
    parser = setup_cli_parser("Load NFL Next Gen Stats data into the database")
    parser.add_argument("stat_type", choices=['passing', 'rushing', 'receiving'], 
                       help="Type of NGS data to load")
    parser.add_argument("years", nargs='+', type=int, help="NFL season years (e.g., 2023 2024)")
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("🏈 NFL Next Gen Stats Data Loader")
        print(f"Loading {args.stat_type} NGS data for seasons: {', '.join(map(str, args.years))}")
        
        # Create loader and run
        loader = NextGenStatsDataLoader()
        result = loader.load_data(
            stat_type=args.stat_type,
            years=args.years,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        # Print results using utility function
        operation = f"Next Gen Stats {args.stat_type} data load for seasons {', '.join(map(str, args.years))}"
        print_results(result, operation, args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current record count
            total_records = loader.get_record_count()
            print(f"Total NGS records in database: {total_records:,}")
        
        return result["success"]
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)