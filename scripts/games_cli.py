#!/usr/bin/env python3
"""
CLI script for advanced NFL games data loading.
Supports loading by season, week, and other options.
"""

import sys
import os
import logging
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.games import GamesDataLoader


def main():
    """CLI interface for the games data loader with advanced options."""
    parser = argparse.ArgumentParser(description="Load NFL game data into the database")
    parser.add_argument("season", type=int, help="NFL season year (e.g., 2024)")
    parser.add_argument("--week", type=int, help="Specific week number (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    parser.add_argument("--clear", action="store_true", help="Clear existing game data before loading")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print(f"🏈 NFL Games Data Loader")
        if args.week:
            print(f"Loading games for season {args.season}, week {args.week}")
        else:
            print(f"Loading games for entire season {args.season}")
        
        # Create loader and run
        loader = GamesDataLoader()
        result = loader.load_games(
            season=args.season,
            week=args.week,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        if result["success"]:
            if args.dry_run:
                if args.week:
                    print(f"DRY RUN - Would upsert {result['would_upsert']} game records for season {args.season}, week {args.week}")
                else:
                    print(f"DRY RUN - Would upsert {result['would_upsert']} game records for entire season {args.season}")
                if result.get("sample_record"):
                    print(f"Sample record: {result['sample_record']}")
            else:
                if args.week:
                    print(f"✅ Successfully loaded game data for season {args.season}, week {args.week}")
                else:
                    print(f"✅ Successfully loaded game data for entire season {args.season}")
                print(f"Fetched: {result['total_fetched']} records")
                print(f"Validated: {result['total_validated']} records")
                print(f"Upserted: {result['upsert_result']['affected_rows']} records")
                
                # Show current game count
                total_games = loader.get_game_count()
                print(f"Total games in database: {total_games}")
        else:
            print(f"❌ Error: {result.get('error', result.get('message', 'Unknown error'))}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
