#!/usr/bin/env python3
"""
CLI script for advanced NFL games data loading.
Supports loading by season, week, and other options.
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

from src.core.data.loaders.games import GamesDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the games data loader with advanced options."""
    parser = setup_cli_parser("Load NFL game data into the database")
    parser.add_argument("season", type=int, help="NFL season year (e.g., 2024)")
    parser.add_argument("--week", type=int, help="Specific week number (optional)")
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("🏈 NFL Games Data Loader")
        if args.week:
            print(f"Loading games for season {args.season}, week {args.week}")
        else:
            print(f"Loading games for entire season {args.season}")
        
        # Create loader and run
        loader = GamesDataLoader()
        result = loader.load_data(
            season=args.season,
            week=args.week,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        # Print results using utility function
        operation = f"games data load for season {args.season}" + (f", week {args.week}" if args.week else "")
        print_results(result, operation, args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current game count
            total_games = loader.get_game_count()
            print(f"Total games in database: {total_games}")
        
        return result["success"]
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    main()
