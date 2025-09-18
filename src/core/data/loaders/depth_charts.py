"""
Depth Charts data loader for NFL roster analytics.
This module provides functionality to fetch NFL depth chart data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class DepthChartsDataLoader(BaseDataLoader):
    """Loads NFL depth charts data into the database.
    
    Depth charts data provides weekly team depth chart positions for all players.
    """
    
    def __init__(self):
        """Initialize the depth charts data loader."""
        super().__init__("depth_charts")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the DepthChartsDataTransformer class."""
        return DepthChartsDataTransformer
    
    def fetch_raw_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch raw depth charts data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            
        Returns:
            Raw depth charts data DataFrame
        """
        self.logger.info(f"Fetching depth charts data for years {years}")
        try:
            depth_charts_df = nfl.import_depth_charts(years)
            self.logger.info(f"Successfully fetched {len(depth_charts_df)} depth charts records for {len(years)} years")
            return depth_charts_df
        except Exception as e:
            self.logger.error(f"Failed to fetch depth charts data for years {years}: {e}")
            raise
    
    def load_depth_charts_data(self, years: List[int], dry_run: bool = False, 
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


class DepthChartsDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of depth charts data.
    
    Transforms raw depth charts data from nfl_data_py into the format expected
    by our depth_charts database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for depth charts data."""
        return [
            'season', 'week', 'team', 'player_id', 'full_name', 'position', 'depth_team'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single depth charts record.
        
        Args:
            row: Single row from raw depth charts DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
            'team': str(row.get('team', '')) if pd.notna(row.get('team')) else None,
            'player_id': str(row.get('player_id', '')) if pd.notna(row.get('player_id')) else None,
            'full_name': str(row.get('full_name', '')) if pd.notna(row.get('full_name')) else None,
        }
        
        # Position and depth information
        record.update({
            'position': str(row.get('position', '')) if pd.notna(row.get('position')) else None,
            'depth_team': str(row.get('depth_team', '')) if pd.notna(row.get('depth_team')) else None,
            'club_depth_chart_position': str(row.get('club_depth_chart_position', '')) if pd.notna(row.get('club_depth_chart_position')) else None,
        })
        
        # Additional player info
        record.update({
            'jersey_number': int(row.get('jersey_number', 0)) if pd.notna(row.get('jersey_number')) else None,
            'formation': str(row.get('formation', '')) if pd.notna(row.get('formation')) else None,
            'game_type': str(row.get('game_type', '')) if pd.notna(row.get('game_type')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed depth charts record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_id') or not record.get('team'):
            return False
        
        # Must have valid season
        if not record.get('season'):
            return False
        
        # Season should be reasonable
        season = record.get('season')
        if season and (season < 2000 or season > 2030):
            return False
        
        # Week should be valid if present
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        # Must have position information
        if not record.get('position'):
            return False
        
        return True