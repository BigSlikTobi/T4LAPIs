"""
Pro Football Reference data loader for advanced NFL analytics.
This module provides functionality to fetch PFR data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class ProFootballReferenceDataLoader(BaseDataLoader):
    """Loads Pro Football Reference data into the database.
    
    PFR data provides seasonal and weekly advanced statistics.
    """
    
    def __init__(self):
        """Initialize the Pro Football Reference data loader."""
        super().__init__("pfr_data")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the ProFootballReferenceDataTransformer class."""
        return ProFootballReferenceDataTransformer
    
    def fetch_raw_data(self, stat_type: str, years: List[int], weekly: bool = False) -> pd.DataFrame:
        """Fetch raw Pro Football Reference data from nfl_data_py.
        
        Args:
            stat_type: Type of PFR data ('pass', 'rec', 'rush')
            years: List of NFL season years
            weekly: If True, fetch weekly data; if False, fetch seasonal data
            
        Returns:
            Raw PFR data DataFrame
        """
        self.logger.info(f"Fetching PFR {'weekly' if weekly else 'seasonal'} {stat_type} data for years {years}")
        try:
            if weekly:
                pfr_df = nfl.import_weekly_pfr(stat_type, years)
            else:
                pfr_df = nfl.import_seasonal_pfr(stat_type, years)
            self.logger.info(f"Successfully fetched {len(pfr_df)} PFR {stat_type} records for {len(years)} years")
            return pfr_df
        except Exception as e:
            self.logger.error(f"Failed to fetch PFR {stat_type} data for years {years}: {e}")
            raise
    
    def load_pfr_data(self, stat_type: str, years: List[int], weekly: bool = False,
                      dry_run: bool = False, clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            stat_type: Type of PFR data ('pass', 'rec', 'rush')
            years: List of NFL season years
            weekly: If True, fetch weekly data; if False, fetch seasonal data
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(stat_type=stat_type, years=years, weekly=weekly,
                             dry_run=dry_run, clear_table=clear_table)


class ProFootballReferenceDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of PFR data.
    
    Transforms raw PFR data from nfl_data_py into the format expected
    by our pfr_data database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for PFR data."""
        return [
            'player_id', 'player_name', 'season', 'team_abbr'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single PFR record.
        
        Args:
            row: Single row from raw PFR DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'player_id': str(row.get('player_id', '')) if pd.notna(row.get('player_id')) else None,
            'player_name': str(row.get('player_name', '')) if pd.notna(row.get('player_name')) else None,
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
            'team_abbr': str(row.get('team_abbr', '')) if pd.notna(row.get('team_abbr')) else None,
        }
        
        # Position and games
        record.update({
            'position': str(row.get('position', '')) if pd.notna(row.get('position')) else None,
            'games': int(row.get('games', 0)) if pd.notna(row.get('games')) else None,
            'games_started': int(row.get('games_started', 0)) if pd.notna(row.get('games_started')) else None,
        })
        
        # Passing stats
        record.update({
            'pass_completions': int(row.get('pass_completions', 0)) if pd.notna(row.get('pass_completions')) else None,
            'pass_attempts': int(row.get('pass_attempts', 0)) if pd.notna(row.get('pass_attempts')) else None,
            'pass_yards': int(row.get('pass_yards', 0)) if pd.notna(row.get('pass_yards')) else None,
            'pass_tds': int(row.get('pass_tds', 0)) if pd.notna(row.get('pass_tds')) else None,
            'interceptions': int(row.get('interceptions', 0)) if pd.notna(row.get('interceptions')) else None,
            'sacks': int(row.get('sacks', 0)) if pd.notna(row.get('sacks')) else None,
            'sack_yards': int(row.get('sack_yards', 0)) if pd.notna(row.get('sack_yards')) else None,
            'pass_long': int(row.get('pass_long', 0)) if pd.notna(row.get('pass_long')) else None,
            'pass_rating': float(row.get('pass_rating', 0)) if pd.notna(row.get('pass_rating')) else None,
            'qbr': float(row.get('qbr', 0)) if pd.notna(row.get('qbr')) else None,
        })
        
        # Rushing stats
        record.update({
            'rush_attempts': int(row.get('rush_attempts', 0)) if pd.notna(row.get('rush_attempts')) else None,
            'rush_yards': int(row.get('rush_yards', 0)) if pd.notna(row.get('rush_yards')) else None,
            'rush_tds': int(row.get('rush_tds', 0)) if pd.notna(row.get('rush_tds')) else None,
            'rush_long': int(row.get('rush_long', 0)) if pd.notna(row.get('rush_long')) else None,
            'rush_yards_per_attempt': float(row.get('rush_yards_per_attempt', 0)) if pd.notna(row.get('rush_yards_per_attempt')) else None,
        })
        
        # Receiving stats
        record.update({
            'targets': int(row.get('targets', 0)) if pd.notna(row.get('targets')) else None,
            'receptions': int(row.get('receptions', 0)) if pd.notna(row.get('receptions')) else None,
            'receiving_yards': int(row.get('receiving_yards', 0)) if pd.notna(row.get('receiving_yards')) else None,
            'receiving_tds': int(row.get('receiving_tds', 0)) if pd.notna(row.get('receiving_tds')) else None,
            'receiving_long': int(row.get('receiving_long', 0)) if pd.notna(row.get('receiving_long')) else None,
            'receiving_yards_per_reception': float(row.get('receiving_yards_per_reception', 0)) if pd.notna(row.get('receiving_yards_per_reception')) else None,
            'receiving_yards_per_target': float(row.get('receiving_yards_per_target', 0)) if pd.notna(row.get('receiving_yards_per_target')) else None,
        })
        
        # Fantasy stats
        record.update({
            'fantasy_points': float(row.get('fantasy_points', 0)) if pd.notna(row.get('fantasy_points')) else None,
            'fantasy_points_ppr': float(row.get('fantasy_points_ppr', 0)) if pd.notna(row.get('fantasy_points_ppr')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed PFR record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_id') or not record.get('player_name'):
            return False
        
        # Must have valid season and team
        if not record.get('season') or not record.get('team_abbr'):
            return False
        
        # Season should be reasonable
        season = record.get('season')
        if season and (season < 1970 or season > 2030):
            return False
        
        # Week should be valid if present
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        return True