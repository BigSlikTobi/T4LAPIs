#!/usr/bin/env python3
"""
Automated player weekly stats data update script for GitHub workflows.
This script automatically detects the latest week in the database and updates stats
for the most recent weeks that likely have new data.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.player_weekly_stats import PlayerWeeklyStatsDataLoader
from src.core.utils.cli import setup_cli_logging, print_results, handle_cli_errors
from src.core.db.database_init import get_supabase_client
import logging
from datetime import datetime


def get_latest_week_in_games_table(season: int) -> int:
    """Get the latest week for a given season in the games table.
    
    This tells us what weeks actually have games scheduled/completed.
    
    Args:
        season: NFL season year
        
    Returns:
        Latest week number (0 if no data found)
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logging.error("Could not connect to database")
            return 0
            
        # Query for the maximum week in the given season from games table
        response = supabase.table("games").select("week").eq("season", season).order("week", desc=True).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["week"]
        else:
            logging.info(f"No games found for season {season}")
            return 0
            
    except Exception as e:
        logging.error(f"Error getting latest week from games table: {e}")
        return 0


def get_latest_week_in_stats_table(season: int) -> int:
    """Get the latest week for a given season in the player_weekly_stats table.
    
    Args:
        season: NFL season year
        
    Returns:
        Latest week number (0 if no data found)
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logging.error("Could not connect to database")
            return 0
            
        # Query for the maximum week in the given season from player_weekly_stats table
        response = supabase.table("player_weekly_stats").select("week").eq("season", season).order("week", desc=True).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["week"]
        else:
            logging.info(f"No player stats found for season {season}")
            return 0
            
    except Exception as e:
        logging.error(f"Error getting latest week from stats table: {e}")
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
    """Main function for automated player weekly stats updates."""
    # Set up logging
    class Args:
        def __init__(self):
            self.verbose = True
            self.quiet = False
            
    args = Args()
    setup_cli_logging(args)
    
    logger = logging.getLogger(__name__)
    
    try:
        print("📊 Automated NFL Player Weekly Stats Update")
        
        # Get current season
        current_season = get_current_nfl_season()
        print(f"Current NFL season: {current_season}")
        
        # Create loader
        loader = PlayerWeeklyStatsDataLoader()
        
        # Get latest weeks from both tables
        latest_games_week = get_latest_week_in_games_table(current_season)
        latest_stats_week = get_latest_week_in_stats_table(current_season)
        
        print(f"Latest week in games table for season {current_season}: {latest_games_week}")
        print(f"Latest week in stats table for season {current_season}: {latest_stats_week}")
        
        if latest_games_week == 0:
            print("No games data found. Player stats update requires games data first.")
            return False
        
        weeks_to_process = []
        
        if latest_stats_week == 0:
            # No stats yet, start from week 1 up to latest games week
            weeks_to_process = list(range(1, latest_games_week + 1))
            print(f"No existing stats data found. Processing weeks 1 through {latest_games_week}.")
        else:
            # Update recent weeks (stats might change as games complete)
            # Process last 2 weeks to catch late updates and add any new weeks
            start_week = max(1, latest_stats_week - 1)  # Go back 1 week for updates
            end_week = latest_games_week
            weeks_to_process = list(range(start_week, end_week + 1))
            print(f"Will update weeks {start_week} through {end_week} (recent + new data)")
        
        if not weeks_to_process:
            print("No weeks to process. Stats are up to date.")
            return True
        
        results = []
        
        for week in weeks_to_process:
            # NFL regular season is typically weeks 1-18, playoffs are weeks 19-22
            if week > 22:
                print(f"Week {week} is beyond NFL season, skipping")
                continue
                
            print(f"\n📅 Processing season {current_season}, week {week} stats")
            
            # Load/upsert the week's stats
            result = loader.load_data(
                years=[current_season],
                weeks=[week],
                dry_run=False,
                clear_table=False
            )
            
            results.append(result)
            
            # Print results
            operation = f"player weekly stats for season {current_season}, week {week}"
            print_results(result, operation, False)
            
            if result["success"]:
                print(f"✅ Successfully processed week {week}")
            else:
                print(f"❌ Failed to process week {week}")
                # Continue with next week even if one fails
        
        # Show final statistics
        total_records = loader.get_record_count()
        print(f"\n📊 Total player weekly stat records in database: {total_records}")
        
        # Check if all operations were successful
        all_successful = all(result["success"] for result in results)
        
        if all_successful:
            print("✅ All operations completed successfully!")
            return True
        else:
            print("⚠️ Some operations failed, check logs for details")
            return False
        
    except Exception as e:
        logger.error(f"Unexpected error in automated stats update: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
