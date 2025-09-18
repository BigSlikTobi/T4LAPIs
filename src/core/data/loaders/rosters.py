"""
Rosters data loader for populating the Supabase rosters table.
This module provides functionality to fetch NFL roster data and load it into the database.
"""

import pandas as pd
from typing import Type, List, Dict, Any, Optional
import logging

from .base import BaseDataLoader
from ..transform import RostersDataTransformer, BaseDataTransformer

logger = logging.getLogger(__name__)


class RostersDataLoader(BaseDataLoader):
    """Loads NFL roster data into the database with versioning support.
    
    Fetches roster data from nfl_data_py and loads it into the rosters table
    with versioning to track roster changes over time.
    """
    
    def __init__(self):
        """Initialize the rosters data loader."""
        super().__init__("rosters")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the RostersDataTransformer class."""
        return RostersDataTransformer
    
    def fetch_raw_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch raw roster data from nfl_data_py for specified years.
        
        Args:
            years: List of NFL season years to fetch
            
        Returns:
            Raw roster data DataFrame
        """
        self.logger.info(f"Fetching roster data for years {years}")
        
        try:
            import nfl_data_py as nfl
            
            # Fetch seasonal roster data
            raw_data = nfl.import_seasonal_rosters(years)
            
            if raw_data is None or raw_data.empty:
                self.logger.warning(f"No roster data found for years {years}")
                return pd.DataFrame()
            
            self.logger.info(f"Fetched {len(raw_data)} roster records")
            return raw_data
            
        except Exception as e:
            self.logger.error(f"Error fetching roster data: {e}")
            return pd.DataFrame()
    
    def fetch_rosters(self, years: List[int]) -> pd.DataFrame:
        """Fetch roster data - wrapper for fetch_raw_data for API consistency.
        
        Args:
            years: List of NFL season years to fetch
            
        Returns:
            Raw roster data DataFrame
        """
        return self.fetch_raw_data(years=years)
    
    def transform_rosters(self, df: pd.DataFrame, version: Optional[int] = None) -> List[Dict[str, Any]]:
        """Transform roster data using the appropriate transformer.
        
        Args:
            df: Raw roster data DataFrame
            version: Optional version number to assign to records
            
        Returns:
            List of transformed roster records ready for database insertion
        """
        # Add version to DataFrame if provided
        if version is not None:
            df = df.copy()
            df['version'] = version
        
        return self.transform_data(df)
    
    def upsert_rosters(self, records: List[Dict[str, Any]], 
                      version: Optional[int] = None, 
                      batch_size: int = 1000) -> Dict[str, Any]:
        """Upsert roster records to the database.
        
        Args:
            records: List of roster records to upsert
            version: Optional version number (added to records if not present)
            batch_size: Number of records to process in each batch
            
        Returns:
            Dictionary with upsert operation results
        """
        # Add version to records if provided and not already present
        if version is not None:
            for record in records:
                if 'version' not in record:
                    record['version'] = version
        
        # Process in batches if needed
        if len(records) <= batch_size:
            return self.db_manager.upsert_records(records, on_conflict="slug")
        
        self.logger.info(f"Processing {len(records)} records in batches of {batch_size}")
        total_affected = 0
        errors = []
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} records)")
            
            result = self.db_manager.upsert_records(batch, on_conflict="slug")
            
            if result["success"]:
                total_affected += result["affected_rows"]
            else:
                errors.append(result["error"])
        
        success = len(errors) == 0
        return {
            "success": success,
            "affected_rows": total_affected,
            "batch_count": (len(records) + batch_size - 1) // batch_size,
            "errors": errors if errors else None
        }
    
    def get_current_max_version(self) -> int:
        """Get the current maximum version number from the rosters table.
        
        Returns:
            Maximum version number, or 0 if no records exist
        """
        try:
            response = self.db_manager.supabase.table(self.table_name)\
                .select("version")\
                .order("version", desc=True)\
                .limit(1)\
                .execute()
            
            if hasattr(response, 'data') and response.data:
                max_version = response.data[0].get('version', 0)
                return max_version if max_version is not None else 0
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error getting max version: {e}")
            return 0
    
    def load_rosters(self, years: List[int], dry_run: bool = False, 
                    version: Optional[int] = None, batch_size: int = 1000) -> Dict[str, Any]:
        """Complete workflow to load roster data with versioning.
        
        Args:
            years: List of NFL season years to fetch
            dry_run: If True, don't actually insert/update data
            version: Optional version number (auto-generated if not provided)
            batch_size: Number of records to process in each batch
            
        Returns:
            Dictionary with operation results
        """
        # Auto-generate version if not provided
        if version is None:
            version = self.get_current_max_version() + 1
            self.logger.info(f"Auto-generated version: {version}")
        
        # Fetch and transform data
        result = self.load_data(years=years, dry_run=dry_run)
        
        if not result["success"] or dry_run:
            return result
        
        # Add versioning and batch processing info to result
        result["version"] = version
        result["batch_size"] = batch_size
        
        return result