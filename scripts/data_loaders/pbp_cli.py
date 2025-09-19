#!/usr/bin/env python3
"""
CLI script for loading NFL Play-by-Play data.
Supports loading by season with various options including downsampling for performance.
NOTE: PBP is only available until season 2024 in nfl_data_py. For 2025+, we need to use import_weekly_data.
"""

import sys
import os
from pathlib import Path

# Add project root to path (robust to nesting)
def _repo_root() -> str:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / 'src').exists() and (p / 'README.md').exists():
            return str(p)
    return str(start.parents[0])

project_root = _repo_root()
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
    parser.add_argument(
        "--week",
        type=int,
        help="Filter to a specific week. If omitted, defaults to latest week available per season."
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
            week=args.week,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        # Print results using utility function
        week_suffix = f" (week {args.week})" if args.week else " (latest week)"
        operation = f"play-by-play data load for seasons {', '.join(map(str, args.years))}{week_suffix}"
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
    exit_code_or_bool = main()
    # handle_cli_errors returns 0/1 for tests; pass through directly if int
    if isinstance(exit_code_or_bool, int):
        sys.exit(exit_code_or_bool)
    else:
        sys.exit(0 if exit_code_or_bool else 1)
