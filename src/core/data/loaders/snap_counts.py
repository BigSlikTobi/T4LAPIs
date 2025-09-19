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
    
    def fetch_raw_data(self, years: List[int], player_id=None, week: int | None = None) -> pd.DataFrame:
        """Fetch raw snap counts data from nfl_data_py, with optional player filtering.
        
        Args:
            years: List of NFL season years
            player_id: Optional GSIS player ID or list of IDs to filter by
            week: Optional week to filter; when None, defaults to latest week per season
            
        Returns:
            Raw snap counts data DataFrame (optionally filtered)
        """
        self.logger.info(f"Fetching snap counts data for years {years}")
        try:
            snap_counts_df = nfl.import_snap_counts(years)
            self.logger.info(f"Successfully fetched {len(snap_counts_df)} snap counts records for {len(years)} years")

            # Optional player filtering
            if player_id is not None and not snap_counts_df.empty:
                snap_counts_df = self._filter_by_player_id(snap_counts_df, player_id)

            # Week filtering
            if not snap_counts_df.empty and 'season' in snap_counts_df.columns and 'week' in snap_counts_df.columns:
                if week is not None:
                    before = len(snap_counts_df)
                    snap_counts_df = snap_counts_df[snap_counts_df['week'] == int(week)].copy()
                    self.logger.info(f"Filtered snap counts to week {week}: {len(snap_counts_df)}/{before} records")
                else:
                    # Default to latest week per season
                    before = len(snap_counts_df)
                    latest_mask = snap_counts_df['week'] == snap_counts_df.groupby('season')['week'].transform('max')
                    snap_counts_df = snap_counts_df[latest_mask].copy()
                    try:
                        latest_per_season = snap_counts_df.groupby('season')['week'].max().to_dict()
                        self.logger.info(f"Defaulted to latest week per season for snap counts: {latest_per_season}; kept {len(snap_counts_df)}/{before} records")
                    except Exception:
                        self.logger.info(f"Defaulted to latest week per season for snap counts; kept {len(snap_counts_df)}/{before} records")

            return snap_counts_df
        except Exception as e:
            self.logger.error(f"Failed to fetch snap counts data for years {years}: {e}")
            raise

    def _filter_by_player_id(self, df: pd.DataFrame, player_id) -> pd.DataFrame:
        """Filter a DataFrame by player identifier, supporting multiple column names.
        
        Supports either a single string ID or a list of IDs. Attempts to use one of
        ['player_id', 'player_gsis_id', 'gsis_id', 'pfr_player_id'] depending on what the dataset provides.
        """
        # Normalize input to a set of strings
        if isinstance(player_id, (list, tuple, set)):
            ids = {str(p).strip() for p in player_id if str(p).strip()}
        else:
            ids = {str(player_id).strip()} if str(player_id).strip() else set()

        if not ids:
            return df

        # Helper to detect GSIS-style IDs like '00-0038507'
        def _looks_like_gsis(s: str) -> bool:
            return '-' in s and s.split('-')[0].isdigit()

        available_cols = set(df.columns)

        # If DataFrame already has GSIS column, filter directly
        for col in ["player_id", "player_gsis_id", "gsis_id"]:
            if col in available_cols:
                before = len(df)
                filtered = df[df[col].astype(str).isin(ids)].copy()
                self.logger.info(
                    f"Applied player filter on '{col}': kept {len(filtered)}/{before} records for players {sorted(ids)}"
                )
                return filtered

        # If it has PFR ids, map any GSIS inputs to PFR and filter
        if "pfr_player_id" in available_cols:
            gsis_ids = {i for i in ids if _looks_like_gsis(i)}
            pfr_inputs = {i for i in ids if not _looks_like_gsis(i)}

            if gsis_ids:
                mapped = self._map_gsis_to_pfr(gsis_ids)
                if not mapped:
                    self.logger.warning(f"Could not map GSIS IDs to PFR: {sorted(gsis_ids)}")
                pfr_inputs |= mapped

            if not pfr_inputs:
                self.logger.warning("Player filter found 'pfr_player_id' column but no PFR IDs to filter after mapping")
                return df

            before = len(df)
            filtered = df[df["pfr_player_id"].astype(str).isin(pfr_inputs)].copy()
            self.logger.info(
                f"Applied player filter on 'pfr_player_id': kept {len(filtered)}/{before} records for players {sorted(pfr_inputs)}"
            )
            return filtered

        self.logger.warning(
            f"Player filter requested but no suitable player id column found in DataFrame. "
            f"Available columns: {list(df.columns)[:10]}..."
        )
        return df

    def _map_gsis_to_pfr(self, gsis_ids: set) -> set:
        """Map GSIS player IDs to PFR player IDs using nfl_data_py players reference."""
        try:
            players_df = nfl.import_players()
            # Determine PFR and GSIS column names
            pfr_col = 'pfr_player_id' if 'pfr_player_id' in players_df.columns else (
                'pfr_id' if 'pfr_id' in players_df.columns else None
            )
            if not pfr_col:
                self.logger.warning("Players reference lacks PFR id columns; cannot map GSIS->PFR")
                return set()

            gsis_col = None
            for c in ['player_id', 'gsis_id', 'gsis', 'gsis_it']:
                if c in players_df.columns:
                    gsis_col = c
                    break
            if not gsis_col:
                self.logger.warning("Players reference lacks a GSIS id column; cannot map GSIS->PFR")
                return set()

            ref = players_df[[gsis_col, pfr_col]].dropna()
            ref[gsis_col] = ref[gsis_col].astype(str)
            ref[pfr_col] = ref[pfr_col].astype(str)
            mapped = set(ref[ref[gsis_col].isin({str(i) for i in gsis_ids})][pfr_col].unique().tolist())
            self.logger.info(f"Mapped {len(mapped)}/{len(gsis_ids)} GSIS IDs to PFR IDs using {gsis_col}->{pfr_col}")
            return mapped
        except Exception as e:
            self.logger.warning(f"Failed to map GSIS to PFR ids: {e}")
            return set()
    
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

    def transform(self, raw_df: pd.DataFrame) -> List[dict]:
        """Alias source-specific columns before base validation and transform.
        
        nfl_data_py's snap counts data exposes 'pfr_player_id'. We standardize this
        to 'player_id' so our required columns check passes and downstream schema
        remains consistent across loaders.
        """
        df = raw_df.copy()
        if 'player_id' not in df.columns and 'pfr_player_id' in df.columns:
            df['player_id'] = df['pfr_player_id']
        return super().transform(df)

    def transform(self, raw_df: pd.DataFrame) -> List[dict]:
        """Alias source-specific columns before base validation and transform.
        
        nfl_data_py's snap counts data exposes 'pfr_player_id'. We standardize this
        to 'player_id' so our required columns check passes and downstream schema
        remains consistent across loaders.
        """
        df = raw_df.copy()
        if 'player_id' not in df.columns and 'pfr_player_id' in df.columns:
            df['player_id'] = df['pfr_player_id']
        return super().transform(df)
    
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