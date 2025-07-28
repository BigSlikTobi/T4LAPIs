#!/usr/bin/env python3
"""
Automated games data update script for GitHub workflows.
This script automatically detects the latest week in the database and upserts it,
then loads the next week if available.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.games import GamesDataLoader
from src.core.utils.cli import setup_cli_logging, print_results, handle_cli_errors
from src.core.db.database_init import get_supabase_client
import logging
from datetime import datetime


def get_latest_week_in_db(loader: GamesDataLoader, season: int) -> int:
    """Get the latest week for a given season in the database.
    
    Args:
        loader: GamesDataLoader instance
        season: NFL season year
        
    Returns:
        Latest week number (0 if no data found)
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logging.error("Could not connect to database")
            return 0
            
        # Query for the maximum week in the given season
        response = supabase.table("games").select("week").eq("season", season).order("week", desc=True).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["week"]
        else:
            logging.info(f"No games found for season {season}")
            return 0
            
    except Exception as e:
        logging.error(f"Error getting latest week: {e}")
        return 0


def get_current_nfl_season() -> int:
    """Get the current NFL season based on the date.
    
    Returns:
        Current NFL season year
    """
    now = datetime.now()
    # NFL season starts in September and goes into the next year
    if now.month >= 9:  # September onwards is the current season
        return now.year
    else:  # January to August is previous year's season
        return now.year - 1


@handle_cli_errors
def main():
    """Main function for automated games data updates."""
    # Set up logging
    class Args:
        def __init__(self):
            self.verbose = True
            self.quiet = False
            
    args = Args()
    setup_cli_logging(args)
    
    logger = logging.getLogger(__name__)
    
    try:
        print("🏈 Automated NFL Games Data Update")
        
        # Get current season
        current_season = get_current_nfl_season()
        print(f"Current NFL season: {current_season}")
        
        # Create loader
        loader = GamesDataLoader()
        
        # Get latest week in database
        latest_week = get_latest_week_in_db(loader, current_season)
        print(f"Latest week in database for season {current_season}: {latest_week}")
        
        weeks_to_process = []
        
        if latest_week == 0:
            # No data yet, start with week 1
            weeks_to_process = [1]
            print("No existing data found. Starting with week 1.")
        else:
            # Upsert the latest week (in case of updates) and add the next week
            weeks_to_process = [latest_week, latest_week + 1]
            print(f"Will upsert week {latest_week} and insert week {latest_week + 1}")
        
        results = []
        
        for week in weeks_to_process:
            # NFL regular season is typically weeks 1-18, playoffs are weeks 19-22
            if week > 22:
                print(f"Week {week} is beyond NFL season, skipping")
                continue
                
            print(f"\n📅 Processing season {current_season}, week {week}")
            
            # Load/upsert the week
            result = loader.load_data(
                season=current_season,
                week=week,
                dry_run=False,
                clear_table=False
            )
            
            results.append(result)
            
            # Print results
            operation = f"games data for season {current_season}, week {week}"
            print_results(result, operation, False)
            
            if result["success"]:
                print(f"✅ Successfully processed week {week}")
            else:
                print(f"❌ Failed to process week {week}")
                # Continue with next week even if one fails
        
        # Show final statistics
        total_games = loader.get_game_count()
        print(f"\n📊 Total games in database: {total_games}")
        
        # Check if all operations were successful
        all_successful = all(result["success"] for result in results)
        
        if all_successful:
            print("✅ All operations completed successfully!")
            return True
        else:
            print("⚠️ Some operations failed, check logs for details")
            return False
        
    except Exception as e:
        logger.error(f"Unexpected error in automated update: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
