#!/usr/bin/env python3
"""
Simple script to load NFL players data into Supabase database.
This demonstrates how to use the PlayersDataLoader class.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.players import PlayersDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Load players data into the database."""
    try:
        logger.info("🏈 Starting NFL Players Data Load")
        
        # Create loader instance
        loader = PlayersDataLoader()
        
        # Check current player count
        existing_count = loader.get_player_count()
        logger.info(f"Found {existing_count} existing player records")
        
        # Load players data for current season (2025)
        season = 2025
        logger.info(f"Loading player data for season {season}")
        
        result = loader.load_players(season=season)
        
        if result["success"]:
            logger.info(f"✅ Successfully loaded player data for season {season}")
            logger.info(f"Fetched: {result['total_fetched']} records")
            logger.info(f"Validated: {result['total_validated']} records")
            logger.info(f"Upserted: {result['upsert_result']['affected_rows']} records")
            
            # Show final count
            final_count = loader.get_player_count()
            logger.info(f"Total players in database: {final_count}")
        else:
            logger.error(f"❌ Failed to load players data: {result.get('error', result.get('message', 'Unknown error'))}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
