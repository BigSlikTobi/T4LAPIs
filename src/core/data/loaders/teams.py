"""
Teams data loader for populating the Supabase teams table.
This module provides functionality to fetch NFL team data and load it into the database.
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.fetch import fetch_team_data
from src.core.data.transform import TeamDataTransformer
from src.core.db.database_init import get_supabase_client

logger = logging.getLogger(__name__)


class TeamsDataLoader:
    """Class to handle loading team data into the database."""
    
    def __init__(self, table_name: str = "teams"):
        """
        Initialize the teams data loader.
        
        Args:
            table_name: Name of the database table (default: "teams")
        """
        self.table_name = table_name
        self.supabase = get_supabase_client()
        
        if not self.supabase:
            raise RuntimeError("Could not initialize Supabase client")
    
    def fetch_and_transform_teams(self) -> List[Dict[str, Any]]:
        """
        Fetch raw team data and transform it for database insertion.
        
        Returns:
            List of transformed team records
            
        Raises:
            Exception: If fetch or transform fails
        """
        logger.info("Fetching and transforming team data")
        
        # Fetch raw data
        raw_team_df = fetch_team_data()
        
        # Transform to database schema using the new class
        transformer = TeamDataTransformer()
        sanitized_teams = transformer.transform(raw_team_df)
        
        return sanitized_teams
    
    def clear_teams_table(self) -> bool:
        """
        Clear all records from the teams table.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Clearing all records from {self.table_name} table")
            
            response = self.supabase.table(self.table_name).delete().neq('team_abbr', '').execute()
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error clearing table: {response.error}")
                return False
                
            logger.info("Successfully cleared teams table")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear teams table: {e}")
            return False
    
    def insert_teams(self, team_records: List[Dict[str, Any]]) -> bool:
        """
        Insert team records into the database.
        
        Args:
            team_records: List of team record dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Inserting {len(team_records)} team records into {self.table_name}")
            
            response = self.supabase.table(self.table_name).insert(team_records).execute()
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error inserting teams: {response.error}")
                return False
            
            inserted_count = len(response.data) if hasattr(response, 'data') and response.data else 0
            logger.info(f"Successfully inserted {inserted_count} team records")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert teams: {e}")
            return False
    
    def get_existing_teams_count(self) -> int:
        """
        Get the count of existing teams in the database.
        
        Returns:
            Number of existing team records
        """
        try:
            response = self.supabase.table(self.table_name).select("team_abbr", count="exact").execute()
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error getting teams count: {response.error}")
                return 0
                
            return response.count if hasattr(response, 'count') else 0
            
        except Exception as e:
            logger.error(f"Failed to get teams count: {e}")
            return 0
    
    def load_teams(self, clear_existing: bool = False) -> bool:
        """
        Complete workflow to load team data into the database.
        
        Args:
            clear_existing: Whether to clear existing data first
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting teams data loading process")
            
            # Check existing data
            existing_count = self.get_existing_teams_count()
            logger.info(f"Found {existing_count} existing team records")
            
            if existing_count > 0 and not clear_existing:
                logger.warning("Teams already exist. Use clear_existing=True to replace them")
                return False
            
            if clear_existing and existing_count > 0:
                if not self.clear_teams_table():
                    return False
            
            # Fetch and transform data
            team_records = self.fetch_and_transform_teams()
            
            if not team_records:
                logger.error("No team records to insert")
                return False
            
            # Insert data
            success = self.insert_teams(team_records)
            
            if success:
                logger.info("Teams data loading completed successfully")
            else:
                logger.error("Teams data loading failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Teams data loading process failed: {e}")
            return False


def main():
    """Main function for command-line execution."""
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load NFL team data into Supabase')
    parser.add_argument('--clear', action='store_true', 
                       help='Clear existing team data before loading')
    parser.add_argument('--table', default='teams',
                       help='Database table name (default: teams)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Fetch and transform data but do not insert into database')
    
    args = parser.parse_args()
    
    try:
        loader = TeamsDataLoader(table_name=args.table)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No data will be inserted")
            team_records = loader.fetch_and_transform_teams()
            logger.info(f"Would insert {len(team_records)} team records:")
            for i, team in enumerate(team_records[:5], 1):  # Show first 5
                logger.info(f"  {i}. {team['team_abbr']} - {team['team_name']}")
            if len(team_records) > 5:
                logger.info(f"  ... and {len(team_records) - 5} more")
            return
        
        success = loader.load_teams(clear_existing=args.clear)
        
        if success:
            logger.info("✅ Teams data loading completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Teams data loading failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
