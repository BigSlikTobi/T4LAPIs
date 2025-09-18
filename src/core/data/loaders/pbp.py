"""
Play-by-Play data loader for advanced NFL analytics.
This module provides functionality to fetch NFL play-by-play data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..fetch import fetch_pbp_data
from ..transform import BaseDataTransformer


class PlayByPlayDataLoader(BaseDataLoader):
    """Loads NFL play-by-play data into the database.
    
    Play-by-play data is the largest dataset with 390+ columns and requires
    careful handling for performance and batching.
    """
    
    def __init__(self):
        """Initialize the play-by-play data loader."""
        super().__init__("play_by_play")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the PlayByPlayDataTransformer class."""
        return PlayByPlayDataTransformer
    
    def fetch_raw_data(self, years: List[int], downsampling: bool = True) -> pd.DataFrame:
        """Fetch raw play-by-play data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            downsampling: Whether to enable downsampling for performance
            
        Returns:
            Raw play-by-play data DataFrame
        """
        self.logger.info(f"Fetching play-by-play data for years {years}")
        return fetch_pbp_data(years, downsampling=downsampling)
    
    def load_pbp_data(self, years: List[int], dry_run: bool = False, 
                      clear_table: bool = False, downsampling: bool = True):
        """Legacy method for backward compatibility.
        
        Args:
            years: List of NFL season years
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            downsampling: Whether to enable downsampling for performance
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(years=years, dry_run=dry_run, 
                             clear_table=clear_table, downsampling=downsampling)


class PlayByPlayDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of play-by-play data.
    
    Transforms raw PBP data from nfl_data_py into the format expected
    by our play_by_play database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for play-by-play data."""
        return [
            'play_id', 'game_id', 'season', 'week', 'posteam', 'defteam',
            'down', 'ydstogo', 'yardline_100', 'quarter_seconds_remaining',
            'play_type', 'yards_gained', 'epa', 'wpa', 'wp'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single play-by-play record.
        
        Args:
            row: Single row from raw play-by-play DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers and game context
        record = {
            'play_id': str(row.get('play_id', '')),
            'game_id': str(row.get('game_id', '')),
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
            'game_date': str(row.get('game_date', '')) if pd.notna(row.get('game_date')) else None,
        }
        
        # Team information
        record.update({
            'posteam': str(row.get('posteam', '')) if pd.notna(row.get('posteam')) else None,
            'defteam': str(row.get('defteam', '')) if pd.notna(row.get('defteam')) else None,
            'home_team': str(row.get('home_team', '')) if pd.notna(row.get('home_team')) else None,
            'away_team': str(row.get('away_team', '')) if pd.notna(row.get('away_team')) else None,
        })
        
        # Game state
        record.update({
            'down': int(row.get('down', 0)) if pd.notna(row.get('down')) else None,
            'ydstogo': int(row.get('ydstogo', 0)) if pd.notna(row.get('ydstogo')) else None,
            'yardline_100': float(row.get('yardline_100', 0)) if pd.notna(row.get('yardline_100')) else None,
            'quarter_seconds_remaining': int(row.get('quarter_seconds_remaining', 0)) if pd.notna(row.get('quarter_seconds_remaining')) else None,
            'half_seconds_remaining': int(row.get('half_seconds_remaining', 0)) if pd.notna(row.get('half_seconds_remaining')) else None,
            'game_seconds_remaining': int(row.get('game_seconds_remaining', 0)) if pd.notna(row.get('game_seconds_remaining')) else None,
        })
        
        # Play details
        record.update({
            'play_type': str(row.get('play_type', '')) if pd.notna(row.get('play_type')) else None,
            'yards_gained': float(row.get('yards_gained', 0)) if pd.notna(row.get('yards_gained')) else None,
            'shotgun': bool(row.get('shotgun', False)) if pd.notna(row.get('shotgun')) else None,
            'no_huddle': bool(row.get('no_huddle', False)) if pd.notna(row.get('no_huddle')) else None,
            'qb_dropback': bool(row.get('qb_dropback', False)) if pd.notna(row.get('qb_dropback')) else None,
            'qb_scramble': bool(row.get('qb_scramble', False)) if pd.notna(row.get('qb_scramble')) else None,
        })
        
        # Advanced metrics
        record.update({
            'epa': float(row.get('epa', 0)) if pd.notna(row.get('epa')) else None,
            'wpa': float(row.get('wpa', 0)) if pd.notna(row.get('wpa')) else None,
            'wp': float(row.get('wp', 0)) if pd.notna(row.get('wp')) else None,
            'def_wp': float(row.get('def_wp', 0)) if pd.notna(row.get('def_wp')) else None,
            'home_wp': float(row.get('home_wp', 0)) if pd.notna(row.get('home_wp')) else None,
            'away_wp': float(row.get('away_wp', 0)) if pd.notna(row.get('away_wp')) else None,
        })
        
        # Passing stats
        record.update({
            'pass_attempt': bool(row.get('pass_attempt', False)) if pd.notna(row.get('pass_attempt')) else None,
            'pass_location': str(row.get('pass_location', '')) if pd.notna(row.get('pass_location')) else None,
            'air_yards': float(row.get('air_yards', 0)) if pd.notna(row.get('air_yards')) else None,
            'yards_after_catch': float(row.get('yards_after_catch', 0)) if pd.notna(row.get('yards_after_catch')) else None,
            'complete_pass': bool(row.get('complete_pass', False)) if pd.notna(row.get('complete_pass')) else None,
            'incomplete_pass': bool(row.get('incomplete_pass', False)) if pd.notna(row.get('incomplete_pass')) else None,
            'interception': bool(row.get('interception', False)) if pd.notna(row.get('interception')) else None,
        })
        
        # Rushing stats
        record.update({
            'rush_attempt': bool(row.get('rush_attempt', False)) if pd.notna(row.get('rush_attempt')) else None,
            'rushing_yards': float(row.get('rushing_yards', 0)) if pd.notna(row.get('rushing_yards')) else None,
            'fumble_lost': bool(row.get('fumble_lost', False)) if pd.notna(row.get('fumble_lost')) else None,
        })
        
        # Player IDs (important for linking)
        record.update({
            'passer_player_id': str(row.get('passer_player_id', '')) if pd.notna(row.get('passer_player_id')) else None,
            'receiver_player_id': str(row.get('receiver_player_id', '')) if pd.notna(row.get('receiver_player_id')) else None,
            'rusher_player_id': str(row.get('rusher_player_id', '')) if pd.notna(row.get('rusher_player_id')) else None,
        })
        
        # Scores
        record.update({
            'touchdown': bool(row.get('touchdown', False)) if pd.notna(row.get('touchdown')) else None,
            'pass_touchdown': bool(row.get('pass_touchdown', False)) if pd.notna(row.get('pass_touchdown')) else None,
            'rush_touchdown': bool(row.get('rush_touchdown', False)) if pd.notna(row.get('rush_touchdown')) else None,
            'total_home_score': int(row.get('total_home_score', 0)) if pd.notna(row.get('total_home_score')) else None,
            'total_away_score': int(row.get('total_away_score', 0)) if pd.notna(row.get('total_away_score')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed play-by-play record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('play_id') or not record.get('game_id'):
            return False
        
        # Must have valid season and week
        if not record.get('season') or not record.get('week'):
            return False
        
        # Season should be reasonable (between 1999 and current year + 1)
        season = record.get('season')
        if season and (season < 1999 or season > 2030):
            return False
        
        # Week should be valid (1-22 for regular season + playoffs)
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        return True