#!/usr/bin/env python3
"""
CLI script for loading NFL teams data.
Teams data is relatively static and typically loaded once.
"""

import sys
import os

# Expected number of NFL teams
EXPECTED_NFL_TEAMS_COUNT = 32

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.teams import TeamsDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the teams data loader."""
    parser = setup_cli_parser("Load NFL teams data into the database")
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("🏈 NFL Teams Data Loader")
        print("Loading current NFL team information")
        
        # Create loader and run
        loader = TeamsDataLoader()
        
        # Determine whether to clear first to avoid duplicates with insert semantics
        clear_mode = args.clear
        if not clear_mode:
            existing_count = loader.get_existing_teams_count()
            if existing_count >= EXPECTED_NFL_TEAMS_COUNT:
                print(f"Found {existing_count} existing team records (looks complete)")
                print("Teams already exist. Use --clear to replace existing data.")
                return True
            elif existing_count > 0:
                print(f"Found {existing_count} existing team records (incomplete). Will clear and reload to ensure consistency.")
                clear_mode = True

        result = loader.load_data(
            dry_run=args.dry_run,
            clear_table=clear_mode
        )
        
        # Print results using utility function
        print_results(result, "teams data load", args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current team count
            total_teams = loader.get_record_count()
            print(f"Total teams in database: {total_teams}")
        
        return result["success"]
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    main()
