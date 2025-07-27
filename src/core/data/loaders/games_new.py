"""
Games data loader for populating the Supabase games table.
This module provides functionality to fetch NFL game data and load it into the database.
"""

import pandas as pd
from typing import Type, Optional

from .base import BaseDataLoader
from ..fetch import fetch_game_schedule_data
from ..transform import GameDataTransformer, BaseDataTransformer


class GamesDataLoader(BaseDataLoader):
    """Loads NFL game data into the database with upsert functionality.
    
    Games need upsert capability since scores and other details might be updated
    as games progress or after final scores are confirmed.
    """
    
    def __init__(self):
        """Initialize the games data loader."""
        super().__init__("games")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the GameDataTransformer class."""
        return GameDataTransformer
    
    def fetch_raw_data(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch raw game data from nfl_data_py for a specific season.
        
        Args:
            season: NFL season year
            week: Specific week number (if None, loads entire season)
            
        Returns:
            Raw game data DataFrame
        """
        self.logger.info(f"Fetching game data for season {season}" + 
                        (f", week {week}" if week else " (entire season)"))
        
        # Fetch all games for the season
        raw_games = fetch_game_schedule_data([season])
        
        # Filter by week if specified
        if week is not None:
            raw_games = raw_games[raw_games['week'] == week]
            if raw_games.empty:
                self.logger.warning(f"No game data found for season {season}, week {week}")
        
        return raw_games
    
    def load_games(self, season: int, week: Optional[int] = None, dry_run: bool = False, clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            season: NFL season year
            week: Specific week number (if None, loads entire season)
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(season=season, week=week, dry_run=dry_run, clear_table=clear_table)
    
    def get_game_count(self, season: Optional[int] = None, week: Optional[int] = None) -> int:
        """Legacy method for backward compatibility.
        
        Args:
            season: Filter by season (optional)
            week: Filter by week (optional)
            
        Returns:
            Number of game records matching filters
        """
        conditions = {}
        if season:
            conditions['season'] = season
        if week:
            conditions['week'] = week
        
        return self.get_record_count(conditions)
