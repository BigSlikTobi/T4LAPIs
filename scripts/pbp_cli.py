#!/usr/bin/env python3
"""
CLI script for loading NFL Play-by-Play data.
Supports loading by season with various options including downsampling for performance.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.pbp import PlayByPlayDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the play-by-play data loader."""
    parser = setup_cli_parser("Load NFL Play-by-Play data into the database")
    parser.add_argument("years", nargs='+', type=int, help="NFL season years (e.g., 2023 2024)")
    parser.add_argument(
        "--no-downsampling", 
        action="store_true", 
        help="Disable downsampling for full dataset (warning: very large!)"
    )
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("🏈 NFL Play-by-Play Data Loader")
        print(f"Loading play-by-play data for seasons: {', '.join(map(str, args.years))}")
        
        # Warn about large datasets
        downsampling = not args.no_downsampling
        if not downsampling:
            print("⚠️  WARNING: Full play-by-play dataset is very large (390+ columns, 50k+ plays per season)")
            print("   Consider using --dry-run first to see data size")
        
        # Create loader and run
        loader = PlayByPlayDataLoader()
        result = loader.load_data(
            years=args.years,
            downsampling=downsampling,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        # Print results using utility function
        operation = f"play-by-play data load for seasons {', '.join(map(str, args.years))}"
        print_results(result, operation, args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current record count
            total_plays = loader.get_record_count()
            print(f"Total plays in database: {total_plays:,}")
        
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