"""
Rosters data loader for populating the Supabase rosters table.
This module provides functionality to fetch NFL roster data and load it into the database.
"""

import pandas as pd
from typing import Type, Dict, Any, List, Optional

from .base import BaseDataLoader
from ..fetch import fetch_roster_data
from ..transform import RosterDataTransformer, BaseDataTransformer


class RostersDataLoader(BaseDataLoader):
    """Loads NFL roster data into the database with versioning support.
    
    Roster data is versioned to track changes over time and support
    incremental updates across seasons.
    """
    
    def __init__(self):
        """Initialize the rosters data loader."""
        super().__init__("rosters")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the RosterDataTransformer class."""
        return RosterDataTransformer
    
    def fetch_raw_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch raw roster data from nfl_data_py for specific years.
        
        Args:
            years: List of NFL season years
            
        Returns:
            Raw roster data DataFrame
        """
        self.logger.info(f"Fetching roster data for years {years}")
        return fetch_roster_data(years)
    
    def load_rosters(
        self, 
        years: List[int], 
        dry_run: bool = False, 
        clear_table: bool = False,
        version: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Load roster data for specified years.
        
        Args:
            years: List of years to load data for
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            version: Optional version number for the data (auto-calculated if None)
            batch_size: Number of records to process per batch
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Fetch raw data
            raw_df = self.fetch_raw_data(years)
            if raw_df.empty:
                self.logger.warning(f"No roster data found for years {years}")
                return {"success": False, "error": "No data found"}
            
            # Transform data
            transformer = self.transformer_class()
            records = transformer.transform(raw_df)
            
            # Add version information if provided
            if version is not None:
                for record in records:
                    record['version'] = version
            
            # Load to database
            if dry_run:
                self.logger.info(f"DRY RUN: Would insert {len(records)} roster records")
                return {
                    "success": True,
                    "records_processed": len(records),
                    "dry_run": True
                }
            
            result = self.db_manager.upsert_data(
                records=records,
                batch_size=batch_size,
                clear_table=clear_table,
                on_conflict="player_id,season,team"  # Unique constraint
            )
            
            self.logger.info(f"Successfully loaded {len(records)} roster records")
            return {
                "success": True,
                "inserted": result.inserted_count,
                "updated": result.updated_count,
                "errors": result.errors_count,
                "records_processed": len(records)
            }
            
        except Exception as e:
            self.logger.error(f"Error loading roster data: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def get_next_version(self) -> int:
        """Get the next version number for roster data.
        
        Returns:
            Next version number (current max + 1)
        """
        try:
            # Query for max version
            result = self.db_manager.client.table(self.table_name).select("version").order("version", desc=True).limit(1).execute()
            
            if result.data:
                max_version = result.data[0].get('version', 0)
                return (max_version or 0) + 1
            else:
                return 1
                
        except Exception as e:
            self.logger.error(f"Error getting next version: {e}")
            return 1