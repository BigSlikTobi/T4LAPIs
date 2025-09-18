"""
NFL Draft data loader for player evaluation analytics.
This module provides functionality to fetch NFL Draft data and load it into the database.
"""

import pandas as pd
from typing import Type, List, Optional

from .base import BaseDataLoader
from ..fetch import fetch_draft_data
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class DraftDataLoader(BaseDataLoader):
    """Loads NFL Draft data into the database.
    
    Draft data provides complete draft history with career statistics.
    """
    
    def __init__(self):
        """Initialize the NFL Draft data loader."""
        super().__init__("draft_data")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the DraftDataTransformer class."""
        return DraftDataTransformer
    
    def fetch_raw_data(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch raw NFL Draft data from nfl_data_py.
        
        Args:
            years: Optional list of years to fetch (if None, fetches all available)
            
        Returns:
            Raw NFL Draft data DataFrame
        """
        self.logger.info(f"Fetching NFL Draft data for years {years if years else 'all available'}")
        return fetch_draft_data(years)
    
    def load_draft_data(self, years: Optional[List[int]] = None, dry_run: bool = False, 
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


class DraftValuesDataLoader(BaseDataLoader):
    """Loads NFL Draft Values data into the database.
    
    Draft values data provides draft pick value models.
    """
    
    def __init__(self):
        """Initialize the NFL Draft Values data loader."""
        super().__init__("draft_values")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the DraftValuesDataTransformer class."""
        return DraftValuesDataTransformer
    
    def fetch_raw_data(self) -> pd.DataFrame:
        """Fetch raw NFL Draft Values data from nfl_data_py.
        
        Returns:
            Raw NFL Draft Values data DataFrame
        """
        self.logger.info("Fetching NFL Draft Values data")
        try:
            draft_values_df = nfl.import_draft_values()
            self.logger.info(f"Successfully fetched {len(draft_values_df)} draft values records")
            return draft_values_df
        except Exception as e:
            self.logger.error(f"Failed to fetch draft values data: {e}")
            raise
    
    def load_draft_values_data(self, dry_run: bool = False, clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(dry_run=dry_run, clear_table=clear_table)


class DraftDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of NFL Draft data.
    
    Transforms raw draft data from nfl_data_py into the format expected
    by our draft_data database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for NFL Draft data."""
        return [
            'season', 'round', 'pick', 'team', 'player_name', 'pos', 'college'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single NFL Draft record.
        
        Args:
            row: Single row from raw draft DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'round': int(row.get('round', 0)) if pd.notna(row.get('round')) else None,
            'pick': int(row.get('pick', 0)) if pd.notna(row.get('pick')) else None,
            'team': str(row.get('team', '')) if pd.notna(row.get('team')) else None,
            'player_name': str(row.get('player_name', '')) if pd.notna(row.get('player_name')) else None,
            'pos': str(row.get('pos', '')) if pd.notna(row.get('pos')) else None,
            'college': str(row.get('college', '')) if pd.notna(row.get('college')) else None,
        }
        
        # Additional draft info
        record.update({
            'age': float(row.get('age', 0)) if pd.notna(row.get('age')) else None,
            'to': int(row.get('to', 0)) if pd.notna(row.get('to')) else None,
            'ap1': int(row.get('ap1', 0)) if pd.notna(row.get('ap1')) else None,
            'pb': int(row.get('pb', 0)) if pd.notna(row.get('pb')) else None,
            'st': int(row.get('st', 0)) if pd.notna(row.get('st')) else None,
        })
        
        # Career statistics
        record.update({
            'car_av': float(row.get('car_av', 0)) if pd.notna(row.get('car_av')) else None,
            'drav': float(row.get('drav', 0)) if pd.notna(row.get('drav')) else None,
            'g': int(row.get('g', 0)) if pd.notna(row.get('g')) else None,
            'cmp': int(row.get('cmp', 0)) if pd.notna(row.get('cmp')) else None,
            'att': int(row.get('att', 0)) if pd.notna(row.get('att')) else None,
            'yds': int(row.get('yds', 0)) if pd.notna(row.get('yds')) else None,
            'td': int(row.get('td', 0)) if pd.notna(row.get('td')) else None,
            'int': int(row.get('int', 0)) if pd.notna(row.get('int')) else None,
            'rush_att': int(row.get('rush_att', 0)) if pd.notna(row.get('rush_att')) else None,
            'rush_yds': int(row.get('rush_yds', 0)) if pd.notna(row.get('rush_yds')) else None,
            'rush_td': int(row.get('rush_td', 0)) if pd.notna(row.get('rush_td')) else None,
            'rec': int(row.get('rec', 0)) if pd.notna(row.get('rec')) else None,
            'rec_yds': int(row.get('rec_yds', 0)) if pd.notna(row.get('rec_yds')) else None,
            'rec_td': int(row.get('rec_td', 0)) if pd.notna(row.get('rec_td')) else None,
        })
        
        # Additional stats
        record.update({
            'solo': int(row.get('solo', 0)) if pd.notna(row.get('solo')) else None,
            'int_def': int(row.get('int_def', 0)) if pd.notna(row.get('int_def')) else None,
            'sk': float(row.get('sk', 0)) if pd.notna(row.get('sk')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed NFL Draft record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_name') or not record.get('team'):
            return False
        
        # Must have valid season
        season = record.get('season')
        if not season or season < 1936 or season > 2030:  # NFL Draft started in 1936
            return False
        
        # Round should be valid
        round_num = record.get('round')
        if not round_num or round_num < 1 or round_num > 32:  # Max rounds in modern draft
            return False
        
        # Pick should be reasonable
        pick = record.get('pick')
        if not pick or pick < 1 or pick > 300:  # Reasonable pick range
            return False
        
        return True


class DraftValuesDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of NFL Draft Values data.
    
    Transforms raw draft values data from nfl_data_py into the format expected
    by our draft_values database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for NFL Draft Values data."""
        return [
            'pick', 'otc', 'trade_value'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single NFL Draft Values record.
        
        Args:
            row: Single row from raw draft values DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        record = {
            'pick': int(row.get('pick', 0)) if pd.notna(row.get('pick')) else None,
            'otc': float(row.get('otc', 0)) if pd.notna(row.get('otc')) else None,
            'trade_value': float(row.get('trade_value', 0)) if pd.notna(row.get('trade_value')) else None,
            'stuart': float(row.get('stuart', 0)) if pd.notna(row.get('stuart')) else None,
            'johnson': float(row.get('johnson', 0)) if pd.notna(row.get('johnson')) else None,
            'hill': float(row.get('hill', 0)) if pd.notna(row.get('hill')) else None,
        }
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed NFL Draft Values record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have valid pick number
        pick = record.get('pick')
        if not pick or pick < 1 or pick > 300:
            return False
        
        return True