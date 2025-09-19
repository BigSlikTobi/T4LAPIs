"""
Football Study Hall (FTN) data loader for advanced NFL analytics.
This module provides functionality to fetch FTN formation and personnel data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class FootballStudyHallDataLoader(BaseDataLoader):
    """Loads Football Study Hall (FTN) data into the database.
    
    FTN data provides formation data and personnel groupings.
    """
    
    def __init__(self):
        """Initialize the Football Study Hall data loader."""
        super().__init__("ftn_data")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the FootballStudyHallDataTransformer class."""
        return FootballStudyHallDataTransformer
    
    def fetch_raw_data(self, years: List[int], week: int | None = None, player_id=None) -> pd.DataFrame:
        """Fetch raw FTN data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            week: Optional week to filter; if None, defaults to latest week per season
            player_id: Optional player filter (uses PFR player_id if present in data)
            
        Returns:
            Raw FTN data DataFrame
        """
        self.logger.info(f"Fetching FTN data for years {years}")
        try:
            ftn_df = nfl.import_ftn_data(years)
            self.logger.info(f"Successfully fetched {len(ftn_df)} FTN records for {len(years)} years")

            # Optional player filtering: FTN sometimes includes player identifiers; attempt common columns
            if player_id is not None and not ftn_df.empty:
                ftn_df = self._filter_by_player_id(ftn_df, player_id)

            # Week filtering
            if not ftn_df.empty and 'season' in ftn_df.columns and 'week' in ftn_df.columns:
                if week is not None:
                    before = len(ftn_df)
                    ftn_df = ftn_df[ftn_df['week'] == int(week)].copy()
                    self.logger.info(f"Filtered FTN to week {week}: {len(ftn_df)}/{before} records")
                else:
                    before = len(ftn_df)
                    latest_mask = ftn_df['week'] == ftn_df.groupby('season')['week'].transform('max')
                    ftn_df = ftn_df[latest_mask].copy()
                    try:
                        latest_per_season = ftn_df.groupby('season')['week'].max().to_dict()
                        self.logger.info(f"Defaulted to latest week per season for FTN: {latest_per_season}; kept {len(ftn_df)}/{before} records")
                    except Exception:
                        self.logger.info(f"Defaulted to latest week per season for FTN; kept {len(ftn_df)}/{before} records")

            return ftn_df
        except Exception as e:
            self.logger.error(f"Failed to fetch FTN data for years {years}: {e}")
            raise

    def _filter_by_player_id(self, df: pd.DataFrame, player_id) -> pd.DataFrame:
        """Generic player filter for FTN if player columns exist."""
        if isinstance(player_id, (list, tuple, set)):
            ids = {str(p).strip() for p in player_id if str(p).strip()}
        else:
            ids = {str(player_id).strip()} if str(player_id).strip() else set()

        if not ids:
            return df

        candidate_cols = ['player_id', 'pfr_player_id', 'gsis_id', 'rusher_player_id', 'receiver_player_id', 'passer_player_id']
        filter_col = next((c for c in candidate_cols if c in df.columns), None)
        if not filter_col:
            self.logger.info("No player id column found in FTN dataset; skipping player filter")
            return df

        before = len(df)
        filtered = df[df[filter_col].astype(str).isin(ids)].copy()
        self.logger.info(f"Applied player filter on '{filter_col}': kept {len(filtered)}/{before} FTN records for players {sorted(ids)}")
        return filtered
    
    def load_ftn_data(self, years: List[int], dry_run: bool = False, 
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


class FootballStudyHallDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of FTN data.
    
    Transforms raw FTN data from nfl_data_py into the format expected
    by our ftn_data database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for FTN data."""
        # FTN dataset from nfl_data_py does not include posteam/defteam.
        # Keep the minimal identifiers needed for joining with other data.
        return [
            'nflverse_play_id', 'season', 'week'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single FTN record.
        
        Args:
            row: Single row from raw FTN DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers (teams not present in FTN dataset)
        record = {
            'nflverse_play_id': str(row.get('nflverse_play_id', '')) if pd.notna(row.get('nflverse_play_id')) else None,
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
        }
        
        # Play information
        record.update({
            # Map available identifiers from FTN
            'game_id': str(row.get('nflverse_game_id', '')) if pd.notna(row.get('nflverse_game_id')) else None,
            'play_id': int(row.get('ftn_play_id')) if pd.notna(row.get('ftn_play_id')) else None,
            # The following may not exist in FTN; keep if present
            'down': int(row.get('down', 0)) if pd.notna(row.get('down')) else None,
            'ydstogo': int(row.get('ydstogo', 0)) if pd.notna(row.get('ydstogo')) else None,
            'play_type': str(row.get('play_type', '')) if pd.notna(row.get('play_type')) else None,
        })
        
        # Formation data (offensive)
        record.update({
            'offense_formation': str(row.get('offense_formation', '')) if pd.notna(row.get('offense_formation')) else None,
            'offense_personnel': str(row.get('offense_personnel', '')) if pd.notna(row.get('offense_personnel')) else None,
            # FTN column aliases
            'defenders_in_box': int(row.get('n_defense_box', 0)) if pd.notna(row.get('n_defense_box')) else None,
            'number_of_pass_rushers': int(row.get('n_pass_rushers', 0)) if pd.notna(row.get('n_pass_rushers')) else None,
        })
        
        # Personnel groupings
        record.update({
            'is_play_action': bool(row.get('is_play_action', False)) if pd.notna(row.get('is_play_action')) else None,
            'is_rpo': bool(row.get('is_rpo', False)) if pd.notna(row.get('is_rpo')) else None,
            # FTN has is_motion; map to motion_indicator for consistency
            'motion_indicator': str(row.get('motion_indicator')) if pd.notna(row.get('motion_indicator')) else (
                str(row.get('is_motion')) if pd.notna(row.get('is_motion')) else None
            ),
        })
        
        # Additional FTN metrics
        record.update({
            'snap_to_throw': float(row.get('snap_to_throw', 0)) if pd.notna(row.get('snap_to_throw')) else None,
            'pocket_time': float(row.get('pocket_time', 0)) if pd.notna(row.get('pocket_time')) else None,
            'time_to_pressure': float(row.get('time_to_pressure', 0)) if pd.notna(row.get('time_to_pressure')) else None,
            'was_pressure': bool(row.get('was_pressure', False)) if pd.notna(row.get('was_pressure')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed FTN record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('nflverse_play_id'):
            return False
        
        # Must have valid season
        if not record.get('season'):
            return False
        
        # Season should be reasonable (FTN data is more recent)
        season = record.get('season')
        if season and (season < 2018 or season > 2030):
            return False
        
        # Week should be valid if present
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        return True