"""
Snap Counts data loader for NFL analytics.
This module provides functionality to fetch NFL snap count data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class SnapCountsDataLoader(BaseDataLoader):
    """Loads NFL snap counts data into the database.
    
    Snap counts data provides insight into player utilization and usage rates.
    """
    
    def __init__(self):
        """Initialize the snap counts data loader."""
        super().__init__("snap_counts")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the SnapCountsDataTransformer class."""
        return SnapCountsDataTransformer
    
    def fetch_raw_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch raw snap counts data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            
        Returns:
            Raw snap counts data DataFrame
        """
        self.logger.info(f"Fetching snap counts data for years {years}")
        try:
            snap_counts_df = nfl.import_snap_counts(years)
            self.logger.info(f"Successfully fetched {len(snap_counts_df)} snap counts records for {len(years)} years")
            return snap_counts_df
        except Exception as e:
            self.logger.error(f"Failed to fetch snap counts data for years {years}: {e}")
            raise
    
    def load_snap_counts_data(self, years: List[int], dry_run: bool = False, 
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


class SnapCountsDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of snap counts data.
    
    Transforms raw snap counts data from nfl_data_py into the format expected
    by our snap_counts database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for snap counts data."""
        return [
            'player_id', 'player', 'team', 'season', 'week', 'game_id'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single snap counts record.
        
        Args:
            row: Single row from raw snap counts DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'player_id': str(row.get('player_id', '')) if pd.notna(row.get('player_id')) else None,
            'player': str(row.get('player', '')) if pd.notna(row.get('player')) else None,
            'team': str(row.get('team', '')) if pd.notna(row.get('team')) else None,
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
            'game_id': str(row.get('game_id', '')) if pd.notna(row.get('game_id')) else None,
        }
        
        # Position and opponent info
        record.update({
            'position': str(row.get('position', '')) if pd.notna(row.get('position')) else None,
            'opponent': str(row.get('opponent', '')) if pd.notna(row.get('opponent')) else None,
        })
        
        # Snap count data
        record.update({
            'offense_snaps': int(row.get('offense_snaps', 0)) if pd.notna(row.get('offense_snaps')) else None,
            'offense_pct': float(row.get('offense_pct', 0)) if pd.notna(row.get('offense_pct')) else None,
            'defense_snaps': int(row.get('defense_snaps', 0)) if pd.notna(row.get('defense_snaps')) else None,
            'defense_pct': float(row.get('defense_pct', 0)) if pd.notna(row.get('defense_pct')) else None,
            'st_snaps': int(row.get('st_snaps', 0)) if pd.notna(row.get('st_snaps')) else None,
            'st_pct': float(row.get('st_pct', 0)) if pd.notna(row.get('st_pct')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed snap counts record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_id') or not record.get('game_id'):
            return False
        
        # Must have valid season and team
        if not record.get('season') or not record.get('team'):
            return False
        
        # Season should be reasonable
        season = record.get('season')
        if season and (season < 2000 or season > 2030):
            return False
        
        # Week should be valid if present
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        # At least one snap count should be present
        offense_snaps = record.get('offense_snaps', 0) or 0
        defense_snaps = record.get('defense_snaps', 0) or 0
        st_snaps = record.get('st_snaps', 0) or 0
        
        if offense_snaps + defense_snaps + st_snaps == 0:
            return False
        
        return True