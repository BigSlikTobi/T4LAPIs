"""
Sports betting lines data loader for NFL analytics.
This module provides functionality to fetch betting lines data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class LinesDataLoader(BaseDataLoader):
    """Loads sports betting lines data into the database.
    
    Lines data provides sports betting lines and odds information.
    """
    
    def __init__(self):
        """Initialize the Lines data loader."""
        super().__init__("lines_data")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the LinesDataTransformer class."""
        return LinesDataTransformer
    
    def fetch_raw_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch raw Lines data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            
        Returns:
            Raw Lines data DataFrame
        """
        self.logger.info(f"Fetching Lines data for years {years}")
        try:
            lines_df = nfl.import_sc_lines(years)
            self.logger.info(f"Successfully fetched {len(lines_df)} Lines records for {len(years)} years")
            return lines_df
        except Exception as e:
            self.logger.error(f"Failed to fetch Lines data for years {years}: {e}")
            raise
    
    def load_lines_data(self, years: List[int], dry_run: bool = False, 
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


class LinesDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of Lines data.
    
    Transforms raw Lines data from nfl_data_py into the format expected
    by our lines_data database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for Lines data."""
        return [
            'game_id', 'season', 'week'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single Lines record.
        
        Args:
            row: Single row from raw Lines DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'game_id': str(row.get('game_id', '')) if pd.notna(row.get('game_id')) else None,
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
        }
        
        # Game details
        record.update({
            'game_date': str(row.get('game_date', '')) if pd.notna(row.get('game_date')) else None,
            'away_team': str(row.get('away_team', '')) if pd.notna(row.get('away_team')) else None,
            'home_team': str(row.get('home_team', '')) if pd.notna(row.get('home_team')) else None,
        })
        
        # Betting lines
        record.update({
            'spread_line': float(row.get('spread_line', 0)) if pd.notna(row.get('spread_line')) else None,
            'total_line': float(row.get('total_line', 0)) if pd.notna(row.get('total_line')) else None,
            'under_odds': float(row.get('under_odds', 0)) if pd.notna(row.get('under_odds')) else None,
            'over_odds': float(row.get('over_odds', 0)) if pd.notna(row.get('over_odds')) else None,
            'away_moneyline': float(row.get('away_moneyline', 0)) if pd.notna(row.get('away_moneyline')) else None,
            'home_moneyline': float(row.get('home_moneyline', 0)) if pd.notna(row.get('home_moneyline')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed Lines record.
        
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
        if not season or season < 2000 or season > 2030:
            return False
        
        # Week should be valid if present
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        return True


class WinTotalsDataLoader(BaseDataLoader):
    """Loads season win totals data into the database.
    
    Win totals data provides preseason over/under win totals for teams.
    """
    
    def __init__(self):
        """Initialize the Win Totals data loader."""
        super().__init__("win_totals")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the WinTotalsDataTransformer class."""
        return WinTotalsDataTransformer
    
    def fetch_raw_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch raw Win Totals data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            
        Returns:
            Raw Win Totals data DataFrame
        """
        self.logger.info(f"Fetching Win Totals data for years {years}")
        try:
            win_totals_df = nfl.import_win_totals(years)
            self.logger.info(f"Successfully fetched {len(win_totals_df)} Win Totals records for {len(years)} years")
            return win_totals_df
        except Exception as e:
            self.logger.error(f"Failed to fetch Win Totals data for years {years}: {e}")
            raise
    
    def load_win_totals_data(self, years: List[int], dry_run: bool = False, 
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


class WinTotalsDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of Win Totals data.
    
    Transforms raw Win Totals data from nfl_data_py into the format expected
    by our win_totals database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for Win Totals data."""
        return [
            'season', 'team'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single Win Totals record.
        
        Args:
            row: Single row from raw Win Totals DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'team': str(row.get('team', '')) if pd.notna(row.get('team')) else None,
        }
        
        # Win totals data
        record.update({
            'win_total': float(row.get('win_total', 0)) if pd.notna(row.get('win_total')) else None,
            'over_odds': float(row.get('over_odds', 0)) if pd.notna(row.get('over_odds')) else None,
            'under_odds': float(row.get('under_odds', 0)) if pd.notna(row.get('under_odds')) else None,
            'projected_wins': float(row.get('projected_wins', 0)) if pd.notna(row.get('projected_wins')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed Win Totals record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('team'):
            return False
        
        # Must have valid season
        season = record.get('season')
        if not season or season < 2000 or season > 2030:
            return False
        
        # Win total should be reasonable if present
        win_total = record.get('win_total')
        if win_total is not None and (win_total < 0 or win_total > 20):
            return False
        
        return True