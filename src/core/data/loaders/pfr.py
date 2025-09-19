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
    
    def fetch_raw_data(self, stat_type: str, years: List[int], weekly: bool = False, week: int | None = None, player_id=None) -> pd.DataFrame:
        """Fetch raw Pro Football Reference data from nfl_data_py.
        
        Args:
            stat_type: Type of PFR data ('pass', 'rec', 'rush')
            years: List of NFL season years
            weekly: If True, fetch weekly data; if False, fetch seasonal data
            week: Optional week to filter (applies to weekly=True). If None, defaults to latest week per season
            player_id: Optional GSIS or PFR player ID(s) to filter by. GSIS will be mapped to PFR.
            
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

            # Optional player filtering (PFR uses 'player_id')
            if player_id is not None and not pfr_df.empty:
                pfr_df = self._filter_by_player_id(pfr_df, player_id)

            # Optional week filtering for weekly datasets; default to latest week per season if week is None
            if weekly and not pfr_df.empty and 'season' in pfr_df.columns and 'week' in pfr_df.columns:
                if week is not None:
                    before = len(pfr_df)
                    pfr_df = pfr_df[pfr_df['week'] == int(week)].copy()
                    self.logger.info(f"Filtered PFR weekly {stat_type} to week {week}: {len(pfr_df)}/{before} records")
                else:
                    before = len(pfr_df)
                    latest_mask = pfr_df['week'] == pfr_df.groupby('season')['week'].transform('max')
                    pfr_df = pfr_df[latest_mask].copy()
                    try:
                        latest_per_season = pfr_df.groupby('season')['week'].max().to_dict()
                        self.logger.info(f"Defaulted to latest week per season for PFR {stat_type}: {latest_per_season}; kept {len(pfr_df)}/{before} records")
                    except Exception:
                        self.logger.info(f"Defaulted to latest week per season for PFR {stat_type}; kept {len(pfr_df)}/{before} records")

            return pfr_df
        except Exception as e:
            self.logger.error(f"Failed to fetch PFR {stat_type} data for years {years}: {e}")
            raise

    def _filter_by_player_id(self, df: pd.DataFrame, player_id) -> pd.DataFrame:
        """Filter PFR dataset by player identifier (accepts GSIS or PFR ids).
        
        Strategy:
        - If GSIS column exists in the DataFrame, filter directly by GSIS ids.
        - Otherwise, map GSIS -> PFR using nfl.import_players() and filter by PFR ids.
        - Non-GSIS inputs are treated as PFR ids and applied directly when possible.
        """
        # Normalize input to a set of strings
        if isinstance(player_id, (list, tuple, set)):
            ids = {str(p).strip() for p in player_id if str(p).strip()}
        else:
            ids = {str(player_id).strip()} if str(player_id).strip() else set()

        if not ids:
            return df

        # GSIS heuristic
        def _looks_like_gsis(s: str) -> bool:
            return '-' in s and s.split('-')[0].isdigit()

        gsis_ids = {i for i in ids if _looks_like_gsis(i)}
        pfr_inputs = {i for i in ids if not _looks_like_gsis(i)}

        available = set(df.columns)

        # If DataFrame includes a GSIS column, prefer direct filter for GSIS ids
        for gsis_col in ['player_gsis_id', 'gsis_id']:
            if gsis_col in available and gsis_ids:
                before = len(df)
                direct = df[df[gsis_col].astype(str).isin(gsis_ids)].copy()
                self.logger.info(f"Applied GSIS filter on '{gsis_col}': kept {len(direct)}/{before} PFR rows for {sorted(gsis_ids)}")
                # If also PFR ids provided and a PFR column exists, union results
                if pfr_inputs:
                    pfr_col = 'player_id' if 'player_id' in available else ('pfr_player_id' if 'pfr_player_id' in available else None)
                    if pfr_col:
                        add = df[df[pfr_col].astype(str).isin(pfr_inputs)].copy()
                        self.logger.info(f"Also applied PFR filter on '{pfr_col}': matched {len(add)} rows for {sorted(pfr_inputs)}")
                        if not add.empty:
                            direct = pd.concat([direct, add], ignore_index=True).drop_duplicates().reset_index(drop=True)
                return direct

        # No GSIS column; map GSIS -> PFR via players reference
        if gsis_ids:
            mapped = self._map_gsis_to_pfr(gsis_ids)
            if not mapped:
                self.logger.warning(f"Could not map GSIS IDs to PFR for filtering: {sorted(gsis_ids)}")
            pfr_inputs |= mapped

        if not pfr_inputs:
            self.logger.warning("Player filter resulted in no PFR ids to filter after mapping")
            return df

        pfr_col = 'player_id' if 'player_id' in available else ('pfr_player_id' if 'pfr_player_id' in available else None)
        if not pfr_col:
            self.logger.warning("Player filter requested but DataFrame lacks PFR id columns ('player_id'/'pfr_player_id')")
            return df

        before = len(df)
        filtered = df[df[pfr_col].astype(str).isin(pfr_inputs)].copy()
        self.logger.info(f"Applied PFR filter on '{pfr_col}': kept {len(filtered)}/{before} PFR rows for {sorted(pfr_inputs)}")
        return filtered

    def _map_gsis_to_pfr(self, gsis_ids: set) -> set:
        """Map GSIS player IDs to PFR player IDs using nfl_data_py players reference.
        Reuses logic similar to SnapCountsDataLoader.
        """
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

    def transform(self, raw_df: pd.DataFrame) -> List[dict]:
        """Alias source-specific columns (pfr_player_id/name, team) before validation."""
        df = raw_df.copy()
        # Player id/name aliasing
        if 'player_id' not in df.columns and 'pfr_player_id' in df.columns:
            df['player_id'] = df['pfr_player_id']
        if 'player_name' not in df.columns and 'pfr_player_name' in df.columns:
            df['player_name'] = df['pfr_player_name']
        # Team abbreviation aliasing
        if 'team_abbr' not in df.columns and 'team' in df.columns:
            df['team_abbr'] = df['team']
        return super().transform(df)
    
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