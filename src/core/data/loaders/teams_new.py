"""
Teams data loader for populating the Supabase teams table.
This module provides functionality to fetch NFL team data and load it into the database.
"""

import pandas as pd
from typing import Type

from .base import BaseDataLoader
from ..fetch import fetch_team_data
from ..transform import TeamDataTransformer, BaseDataTransformer


class TeamsDataLoader(BaseDataLoader):
    """Loads NFL team data into the database.
    
    Teams are relatively static data that don't change frequently.
    """
    
    def __init__(self):
        """Initialize the teams data loader."""
        super().__init__("teams")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the TeamDataTransformer class."""
        return TeamDataTransformer
    
    def fetch_raw_data(self) -> pd.DataFrame:
        """Fetch raw team data from nfl_data_py.
        
        Returns:
            Raw team data DataFrame
        """
        self.logger.info("Fetching team data using nfl.import_team_desc()")
        return fetch_team_data()
    
    def load_teams(self, clear_existing: bool = False) -> bool:
        """Legacy method for backward compatibility.
        
        Args:
            clear_existing: Whether to clear existing data first
            
        Returns:
            True if successful, False otherwise
        """
        result = self.load_data(clear_table=clear_existing)
        return result["success"]
    
    def get_existing_teams_count(self) -> int:
        """Legacy method for backward compatibility.
        
        Returns:
            Number of existing team records
        """
        return self.get_record_count()
