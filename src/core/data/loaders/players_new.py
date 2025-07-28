"""
Players data loader for populating the Supabase players table.
This module provides functionality to fetch NFL player data and load it into the database.
"""

import pandas as pd
from typing import Type

from .base import BaseDataLoader
from ..fetch import fetch_player_data
from ..transform import PlayerDataTransformer, BaseDataTransformer


class PlayersDataLoader(BaseDataLoader):
    """Loads NFL player data into the database with upsert functionality.
    
    Players need upsert capability since they might change teams or positions during the season.
    """
    
    def __init__(self):
        """Initialize the players data loader."""
        super().__init__("players")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the PlayerDataTransformer class."""
        return PlayerDataTransformer
    
    def fetch_raw_data(self, season: int) -> pd.DataFrame:
        """Fetch raw player data from nfl_data_py for a specific season.
        
        Args:
            season: NFL season year
            
        Returns:
            Raw player data DataFrame
        """
        self.logger.info(f"Fetching player data for season {season}")
        return fetch_player_data([season])  # Pass as list
    
    def load_players(self, season: int, dry_run: bool = False, clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            season: NFL season year
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(season=season, dry_run=dry_run, clear_table=clear_table)
    
    def get_player_count(self) -> int:
        """Legacy method for backward compatibility.
        
        Returns:
            Number of player records
        """
        return self.get_record_count()
