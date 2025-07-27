#!/usr/bin/env python3
"""
Simple script to load NFL teams data into Supabase database.
This demonstrates how to use the TeamsDataLoader class.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.teams import TeamsDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Load teams data into the database."""
    try:
        logger.info("🏈 Starting NFL Teams Data Load")
        
        # Create loader instance
        loader = TeamsDataLoader()
        
        # Check if teams already exist
        existing_count = loader.get_existing_teams_count()
        logger.info(f"Found {existing_count} existing team records")
        
        if existing_count > 0:
            logger.info("Teams already exist in database")
            logger.info("Use --clear flag with the main script to replace existing data")
            return
        
        # Load teams data
        success = loader.load_teams()
        
        if success:
            final_count = loader.get_existing_teams_count()
            logger.info(f"✅ Successfully loaded {final_count} NFL teams into database")
        else:
            logger.error("❌ Failed to load teams data")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
