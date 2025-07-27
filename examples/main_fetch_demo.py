#!/usr/bin/env python3
"""
Demonstration script for the four main NFL data fetching functions.
This script shows how to use the specific functions requested:
- fetch_team_data()
- fetch_player_data(years)
- fetch_game_schedule_data(years)
- fetch_weekly_stats_data(years)
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.data.fetch import (
    fetch_team_data,
    fetch_player_data,
    fetch_game_schedule_data,
    fetch_weekly_stats_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate the four main data fetching functions."""
    
    # Years to fetch data for
    years = [2023, 2024]
    
    logger.info("=" * 60)
    logger.info("NFL Data Fetching Demonstration")
    logger.info("=" * 60)
    
    try:
        # 1. Fetch team data
        logger.info("\n1. Fetching team data...")
        team_df = fetch_team_data()
        print(f"✅ Team data fetched: {len(team_df)} teams")
        print(f"Columns: {list(team_df.columns)}")
        print(f"Sample data:\n{team_df.head(3)}\n")
        
        # 2. Fetch player data
        logger.info("2. Fetching player roster data...")
        player_df = fetch_player_data(years)
        print(f"✅ Player data fetched: {len(player_df)} players")
        print(f"Columns: {list(player_df.columns)}")
        print(f"Sample data:\n{player_df.head(3)}\n")
        
        # 3. Fetch game schedule data
        logger.info("3. Fetching game schedule data...")
        schedule_df = fetch_game_schedule_data(years)
        print(f"✅ Schedule data fetched: {len(schedule_df)} games")
        print(f"Columns: {list(schedule_df.columns)}")
        print(f"Sample data:\n{schedule_df.head(3)}\n")
        
        # 4. Fetch weekly stats data
        logger.info("4. Fetching weekly stats data...")
        weekly_stats_df = fetch_weekly_stats_data(years)
        print(f"✅ Weekly stats data fetched: {len(weekly_stats_df)} records")
        print(f"Columns: {list(weekly_stats_df.columns)}")
        print(f"Sample data:\n{weekly_stats_df.head(3)}\n")
        
        logger.info("=" * 60)
        logger.info("✅ All data fetching operations completed successfully!")
        logger.info("=" * 60)
        
        # Summary
        print("\nSUMMARY:")
        print(f"- Teams: {len(team_df)} records")
        print(f"- Players: {len(player_df)} records")
        print(f"- Games: {len(schedule_df)} records")
        print(f"- Weekly Stats: {len(weekly_stats_df)} records")
        
    except Exception as e:
        logger.error(f"❌ Error during data fetching: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
