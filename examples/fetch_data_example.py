#!/usr/bin/env python3
"""
Example script demonstrating how to use the NFL data fetching functions.
This script shows how to fetch data for different NFL datasets.
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
    """Main function to demonstrate data fetching."""
    try:
        # Example years to fetch data for
        years = [2023, 2024]
        
        logger.info("Starting NFL data fetching demonstration...")
        
        # 1. Fetch team data
        logger.info("=" * 50)
        logger.info("Fetching team data...")
        team_df = fetch_team_data()
        logger.info(f"Team data shape: {team_df.shape}")
        logger.info(f"Team data columns: {list(team_df.columns)}")
        logger.info(f"Sample team data:\n{team_df.head()}")
        
        # 2. Fetch player data
        logger.info("=" * 50)
        logger.info("Fetching player roster data...")
        player_df = fetch_player_data(years)
        logger.info(f"Player data shape: {player_df.shape}")
        logger.info(f"Player data columns: {list(player_df.columns)}")
        logger.info(f"Sample player data:\n{player_df.head()}")
        
        # 3. Fetch game schedule data
        logger.info("=" * 50)
        logger.info("Fetching game schedule data...")
        schedule_df = fetch_game_schedule_data(years)
        logger.info(f"Schedule data shape: {schedule_df.shape}")
        logger.info(f"Schedule data columns: {list(schedule_df.columns)}")
        logger.info(f"Sample schedule data:\n{schedule_df.head()}")
        
        # 4. Fetch weekly stats data
        logger.info("=" * 50)
        logger.info("Fetching weekly stats data...")
        weekly_stats_df = fetch_weekly_stats_data(years)
        logger.info(f"Weekly stats data shape: {weekly_stats_df.shape}")
        logger.info(f"Weekly stats data columns: {list(weekly_stats_df.columns)}")
        logger.info(f"Sample weekly stats data:\n{weekly_stats_df.head()}")
        
        logger.info("=" * 50)
        logger.info("Data fetching demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data fetching demonstration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
