"""
NFL Combine data loader for player evaluation analytics.
This module provides functionality to fetch NFL Combine data and load it into the database.
"""

import pandas as pd
from typing import Type, List, Optional

from .base import BaseDataLoader
from ..fetch import fetch_combine_data
from ..transform import BaseDataTransformer


class CombineDataLoader(BaseDataLoader):
    """Loads NFL Combine data into the database.
    
    Combine data provides physical and athletic measurements for drafted players.
    """
    
    def __init__(self):
        """Initialize the NFL Combine data loader."""
        super().__init__("combine_data")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the CombineDataTransformer class."""
        return CombineDataTransformer
    
    def fetch_raw_data(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch raw NFL Combine data from nfl_data_py.
        
        Args:
            years: Optional list of years to fetch (if None, fetches all available)
            
        Returns:
            Raw NFL Combine data DataFrame
        """
        self.logger.info(f"Fetching NFL Combine data for years {years if years else 'all available'}")
        return fetch_combine_data(years)
    
    def load_combine_data(self, years: Optional[List[int]] = None, dry_run: bool = False, 
                         clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            years: Optional list of years to fetch
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(years=years, dry_run=dry_run, clear_table=clear_table)


class CombineDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of NFL Combine data.
    
    Transforms raw combine data from nfl_data_py into the format expected
    by our combine_data database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for NFL Combine data."""
        return [
            'player_name', 'pos', 'school', 'year'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single NFL Combine record.
        
        Args:
            row: Single row from raw combine DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'player_name': str(row.get('player_name', '')) if pd.notna(row.get('player_name')) else None,
            'pos': str(row.get('pos', '')) if pd.notna(row.get('pos')) else None,
            'school': str(row.get('school', '')) if pd.notna(row.get('school')) else None,
            'year': int(row.get('year', 0)) if pd.notna(row.get('year')) else None,
        }
        
        # Physical measurements
        record.update({
            'ht': float(row.get('ht', 0)) if pd.notna(row.get('ht')) else None,
            'wt': float(row.get('wt', 0)) if pd.notna(row.get('wt')) else None,
            'hand_size': float(row.get('hand_size', 0)) if pd.notna(row.get('hand_size')) else None,
            'arm_length': float(row.get('arm_length', 0)) if pd.notna(row.get('arm_length')) else None,
            'wingspan': float(row.get('wingspan', 0)) if pd.notna(row.get('wingspan')) else None,
        })
        
        # Athletic performance metrics
        record.update({
            'forty': float(row.get('forty', 0)) if pd.notna(row.get('forty')) else None,
            'bench': int(row.get('bench', 0)) if pd.notna(row.get('bench')) else None,
            'vertical': float(row.get('vertical', 0)) if pd.notna(row.get('vertical')) else None,
            'broad_jump': float(row.get('broad_jump', 0)) if pd.notna(row.get('broad_jump')) else None,
            'cone': float(row.get('cone', 0)) if pd.notna(row.get('cone')) else None,
            'shuttle': float(row.get('shuttle', 0)) if pd.notna(row.get('shuttle')) else None,
        })
        
        # Additional metrics
        record.update({
            'wonderlic': int(row.get('wonderlic', 0)) if pd.notna(row.get('wonderlic')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed NFL Combine record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_name') or not record.get('pos'):
            return False
        
        # Must have valid year
        year = record.get('year')
        if not year or year < 1987 or year > 2030:  # Combine started in 1987
            return False
        
        # Height and weight should be reasonable if present
        height = record.get('ht')
        if height and (height < 60 or height > 90):  # 5'0" to 7'6"
            return False
        
        weight = record.get('wt')
        if weight and (weight < 150 or weight > 400):  # Reasonable NFL weight range
            return False
        
        # 40-yard dash should be reasonable if present
        forty = record.get('forty')
        if forty and (forty < 4.0 or forty > 7.0):
            return False
        
        return True