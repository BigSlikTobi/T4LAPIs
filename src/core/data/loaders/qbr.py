"""
ESPN QBR data loader for quarterback analytics.
This module provides functionality to fetch ESPN QBR data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class ESPNQBRDataLoader(BaseDataLoader):
    """Loads ESPN QBR data into the database.
    
    QBR data provides ESPN's Total Quarterback Rating metrics.
    """
    
    def __init__(self):
        """Initialize the ESPN QBR data loader."""
        super().__init__("qbr_data")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the ESPNQBRDataTransformer class."""
        return ESPNQBRDataTransformer
    
    def fetch_raw_data(self, years: List[int], week: int | None = None, player_id=None) -> pd.DataFrame:
        """Fetch raw ESPN QBR data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            week: Optional week to filter; if None, defaults to latest week per season
            player_id: Optional GSIS or PFR player identifier(s) to filter by
            
        Returns:
            Raw ESPN QBR data DataFrame
        """
        self.logger.info(f"Fetching ESPN QBR data for years {years}")
        try:
            qbr_df = nfl.import_qbr(years)
            self.logger.info(f"Successfully fetched {len(qbr_df)} QBR records for {len(years)} years")

            # Normalize expected columns across nfl_data_py versions
            if not qbr_df.empty:
                # Derive 'week' from 'game_week' if not present
                if 'week' not in qbr_df.columns and 'game_week' in qbr_df.columns:
                    def _parse_week(v):
                        try:
                            # Accept ints/strings of ints
                            if pd.isna(v):
                                return None
                            s = str(v).strip()
                            if s.isdigit():
                                return int(s)
                            if s.lower().startswith('season'):
                                # Treat season total as week 0
                                return 0
                        except Exception:
                            pass
                        return None
                    qbr_df = qbr_df.copy()
                    qbr_df['week'] = qbr_df['game_week'].apply(_parse_week)

                # Create a unified 'name' field if missing
                if 'name' not in qbr_df.columns:
                    if 'name_display' in qbr_df.columns:
                        qbr_df['name'] = qbr_df['name_display']
                    elif 'name_short' in qbr_df.columns:
                        qbr_df['name'] = qbr_df['name_short']
                    elif {'name_first','name_last'}.issubset(qbr_df.columns):
                        qbr_df['name'] = (qbr_df['name_first'].fillna('') + ' ' + qbr_df['name_last'].fillna('')).str.strip()

                # Prefer team abbreviation when only 'team_abb' exists
                if 'team' not in qbr_df.columns and 'team_abb' in qbr_df.columns:
                    qbr_df['team'] = qbr_df['team_abb']

            # Optional player filtering
            if player_id is not None and not qbr_df.empty:
                qbr_df = self._filter_by_player_id(qbr_df, player_id)

            # Week filtering (QBR can be weekly; otherwise season total uses week 0)
            if not qbr_df.empty and 'season' in qbr_df.columns and 'week' in qbr_df.columns:
                if week is not None:
                    before = len(qbr_df)
                    qbr_df = qbr_df[qbr_df['week'] == int(week)].copy()
                    self.logger.info(f"Filtered QBR to week {week}: {len(qbr_df)}/{before} records")
                else:
                    before = len(qbr_df)
                    latest_mask = qbr_df['week'] == qbr_df.groupby('season')['week'].transform('max')
                    qbr_df = qbr_df[latest_mask].copy()
                    try:
                        latest_per_season = qbr_df.groupby('season')['week'].max().to_dict()
                        self.logger.info(f"Defaulted to latest week per season for QBR: {latest_per_season}; kept {len(qbr_df)}/{before} records")
                    except Exception:
                        self.logger.info(f"Defaulted to latest week per season for QBR; kept {len(qbr_df)}/{before} records")

            return qbr_df
        except Exception as e:
            self.logger.error(f"Failed to fetch QBR data for years {years}: {e}")
            raise
    
    def load_qbr_data(self, years: List[int], dry_run: bool = False, 
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

    def _filter_by_player_id(self, df: pd.DataFrame, player_id) -> pd.DataFrame:
        """Filter QBR dataset by player identifier across possible columns.
        ESPN QBR includes 'player_id' (usually ESPN id), but we also check common id columns.
        """
        if isinstance(player_id, (list, tuple, set)):
            ids = {str(p).strip() for p in player_id if str(p).strip()}
        else:
            ids = {str(player_id).strip()} if str(player_id).strip() else set()

        if not ids:
            return df

        candidate_cols = ['player_id', 'gsis_id', 'pfr_player_id', 'pfr_id']
        filter_col = next((c for c in candidate_cols if c in df.columns), None)
        if not filter_col:
            self.logger.info("No recognized player id column in QBR dataset; skipping player filter")
            return df

        before = len(df)
        filtered = df[df[filter_col].astype(str).isin(ids)].copy()
        self.logger.info(f"Applied player filter on '{filter_col}': kept {len(filtered)}/{before} QBR records for players {sorted(ids)}")
        return filtered


class ESPNQBRDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of ESPN QBR data.
    
    Transforms raw QBR data from nfl_data_py into the format expected
    by our qbr_data database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for ESPN QBR data."""
        # Minimal required fields after normalization
        return ['season', 'player_id']
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single ESPN QBR record.
        
        Args:
            row: Single row from raw QBR DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
            'player_id': str(row.get('player_id', '')) if pd.notna(row.get('player_id')) else None,
            'name': str(row.get('name', row.get('name_display', ''))) if pd.notna(row.get('name', row.get('name_display', None))) else None,
            'team': str(row.get('team', row.get('team_abb', ''))) if pd.notna(row.get('team', row.get('team_abb', None))) else None,
        }
        
        # Game information
        record.update({
            'game_id': str(row.get('game_id', '')) if pd.notna(row.get('game_id')) else None,
            'opponent': str(row.get('opponent', '')) if pd.notna(row.get('opponent')) else None,
        })
        
        # QBR metrics
        record.update({
            'qbr_total': float(row.get('qbr_total', 0)) if pd.notna(row.get('qbr_total')) else None,
            'pts_added': float(row.get('pts_added', 0)) if pd.notna(row.get('pts_added')) else None,
            'qb_plays': int(row.get('qb_plays', 0)) if pd.notna(row.get('qb_plays')) else None,
        })
        
        # Passing QBR breakdown
        record.update({
            'pass_epa': float(row.get('pass_epa', 0)) if pd.notna(row.get('pass_epa')) else None,
            'pass_qbr': float(row.get('pass_qbr', 0)) if pd.notna(row.get('pass_qbr')) else None,
            'pass_plays': int(row.get('pass_plays', 0)) if pd.notna(row.get('pass_plays')) else None,
        })
        
        # Rushing QBR breakdown
        record.update({
            'rush_epa': float(row.get('rush_epa', 0)) if pd.notna(row.get('rush_epa')) else None,
            'rush_qbr': float(row.get('rush_qbr', 0)) if pd.notna(row.get('rush_qbr')) else None,
            'rush_plays': int(row.get('rush_plays', 0)) if pd.notna(row.get('rush_plays')) else None,
        })
        
        # Penalty impact
        record.update({
            'penalty_epa': float(row.get('penalty_epa', 0)) if pd.notna(row.get('penalty_epa')) else None,
            'penalty_qbr': float(row.get('penalty_qbr', 0)) if pd.notna(row.get('penalty_qbr')) else None,
            'penalty_plays': int(row.get('penalty_plays', 0)) if pd.notna(row.get('penalty_plays')) else None,
        })
        
        # QBR raw values
        record.update({
            'qbr_raw': float(row.get('qbr_raw', 0)) if pd.notna(row.get('qbr_raw')) else None,
            'sack_epa': float(row.get('sack_epa', 0)) if pd.notna(row.get('sack_epa')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed ESPN QBR record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_id'):
            return False
        
        # Must have valid season
        if not record.get('season'):
            return False
        
        # Season should be reasonable (QBR started in 2006)
        season = record.get('season')
        if season and (season < 2006 or season > 2030):
            return False
        
        # Week should be valid if present
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        # QBR should be between 0 and 100 if present
        qbr_total = record.get('qbr_total')
        if qbr_total is not None and (qbr_total < 0 or qbr_total > 100):
            return False
        
        return True