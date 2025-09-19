#!/usr/bin/env python3
"""
CLI script for loading NFL players data.
Supports loading by season with various options.
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

from src.core.data.loaders.players import PlayersDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the players data loader."""
    parser = setup_cli_parser("Load NFL players data into the database")
    parser.add_argument("season", type=int, help="NFL season year (e.g., 2024)")
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("🏈 NFL Players Data Loader")
        print(f"Loading player data for season {args.season}")
        
        # Create loader and run
        loader = PlayersDataLoader()
        result = loader.load_data(
            season=args.season,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        # Print results using utility function
        operation = f"players data load for season {args.season}"
        print_results(result, operation, args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current player count
            total_players = loader.get_player_count()
            print(f"Total players in database: {total_players}")
        
        return result["success"]
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    main()
