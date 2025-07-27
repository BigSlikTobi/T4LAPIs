#!/usr/bin/env python3
"""
Simple script to load NFL games data into Supabase database.
This demonstrates how to use the GamesDataLoader class.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.games import GamesDataLoader
from src.core.utils.logging import setup_logging


def main():
    """Load games data into the database."""
    # Setup logging
    setup_logging()
    
    try:
        print("🏈 Starting NFL Games Data Load")
        
        # Create loader instance
        loader = GamesDataLoader()
        
        # Check current game count
        existing_count = loader.get_game_count()
        print(f"Found {existing_count} existing game records")
        
        # Load games data for current season (2024)
        season = 2024
        print(f"Loading game data for entire season {season}")
        
        result = loader.load_data(season=season)
        
        if result["success"]:
            print(f"✅ Successfully loaded game data for season {season}")
            print(f"Fetched: {result['total_fetched']} records")
            print(f"Validated: {result['total_validated']} records")
            print(f"Upserted: {result['upsert_result']['affected_rows']} records")
            
            # Show final count
            final_count = loader.get_game_count()
            print(f"Total games in database: {final_count}")
        else:
            print(f"❌ Failed to load games: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
