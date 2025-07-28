#!/usr/bin/env python3
"""
CLI script for loading NFL player weekly stats data.
Supports loading by years and weeks with various options.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.player_weekly_stats import PlayerWeeklyStatsDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the player weekly stats data loader."""
    parser = setup_cli_parser("Load NFL player weekly stats data into the database")
    parser.add_argument("years", nargs="+", type=int, help="NFL season years (e.g., 2024 2023)")
    parser.add_argument("--weeks", nargs="*", type=int, help="Specific week numbers (optional)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing (default: 100)")
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("📊 NFL Player Weekly Stats Data Loader")
        if args.weeks:
            print(f"Loading stats for years {args.years}, weeks {args.weeks}")
        else:
            print(f"Loading stats for entire years {args.years}")
        
        # Create loader and run
        loader = PlayerWeeklyStatsDataLoader()
        result = loader.load_weekly_stats(
            years=args.years,
            weeks=args.weeks,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        # Print results using utility function
        years_str = ", ".join(map(str, args.years))
        weeks_str = f", weeks {args.weeks}" if args.weeks else ""
        operation = f"player weekly stats load for years {years_str}{weeks_str}"
        print_results(result, operation, args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current stats count
            total_stats = loader.get_stats_count()
            print(f"Total weekly stats records in database: {total_stats}")
        
        return result["success"]
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    main()
