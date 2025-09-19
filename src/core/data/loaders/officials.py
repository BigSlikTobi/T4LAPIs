"""
Officials data loader for NFL game officiating analytics.
This module provides functionality to fetch NFL officials data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class OfficialsDataLoader(BaseDataLoader):
    """Loads NFL Officials data into the database.
    
    Officials data provides information about game officials and their assignments.
    """
    
    def __init__(self):
        """Initialize the Officials data loader."""
        super().__init__("officials")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the OfficialsDataTransformer class."""
        return OfficialsDataTransformer
    
    def fetch_raw_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch raw Officials data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            
        Returns:
            Raw Officials data DataFrame
        """
        self.logger.info(f"Fetching Officials data for years {years}")
        try:
            officials_df = nfl.import_officials(years)
            self.logger.info(f"Successfully fetched {len(officials_df)} Officials records for {len(years)} years")
            return officials_df
        except Exception as e:
            self.logger.error(f"Failed to fetch Officials data for years {years}: {e}")
            raise
    
    def load_officials_data(self, years: List[int], dry_run: bool = False, 
                           clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            years: List of NFL season years
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(years=years, dry_run=dry_run, clear_table=clear_table)


class OfficialsDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of Officials data.
    
    Transforms raw Officials data from nfl_data_py into the format expected
    by our officials database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for Officials data."""
        return [
            'season', 'week', 'game_id'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single Officials record.
        
        Args:
            row: Single row from raw Officials DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
            'game_id': str(row.get('game_id', '')) if pd.notna(row.get('game_id')) else None,
        }
        
        # Game details
        record.update({
            'game_date': str(row.get('game_date', '')) if pd.notna(row.get('game_date')) else None,
            'home_team': str(row.get('home_team', '')) if pd.notna(row.get('home_team')) else None,
            'away_team': str(row.get('away_team', '')) if pd.notna(row.get('away_team')) else None,
        })
        
        # Official positions and names
        record.update({
            'referee': str(row.get('referee', '')) if pd.notna(row.get('referee')) else None,
            'umpire': str(row.get('umpire', '')) if pd.notna(row.get('umpire')) else None,
            'down_judge': str(row.get('down_judge', '')) if pd.notna(row.get('down_judge')) else None,
            'line_judge': str(row.get('line_judge', '')) if pd.notna(row.get('line_judge')) else None,
            'field_judge': str(row.get('field_judge', '')) if pd.notna(row.get('field_judge')) else None,
            'side_judge': str(row.get('side_judge', '')) if pd.notna(row.get('side_judge')) else None,
            'back_judge': str(row.get('back_judge', '')) if pd.notna(row.get('back_judge')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed Officials record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('game_id'):
            return False
        
        # Must have valid season
        season = record.get('season')
        if not season or season < 1970 or season > 2030:
            return False
        
        # Week should be valid if present
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        return True