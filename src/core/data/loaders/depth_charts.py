"""
Depth Charts data loader for NFL roster analytics.
This module provides functionality to fetch NFL depth chart data and load it into the database.
"""

import pandas as pd
from typing import Type, List, Optional

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
from ..team_abbr import normalize_team_abbr
import nfl_data_py as nfl


class DepthChartsDataLoader(BaseDataLoader):
    """Loads NFL depth charts data into the database.
    
    Depth charts data provides weekly team depth chart positions for all players.
    """
    
    def __init__(self):
        """Initialize the depth charts data loader."""
        super().__init__("depth_charts")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the DepthChartsDataTransformer class."""
        return DepthChartsDataTransformer
    
    def fetch_raw_data(self, years: List[int], teams: Optional[List[str]] = None,
                       latest_only: bool = False, as_of: Optional[str] = None) -> pd.DataFrame:
        """Fetch raw depth charts data from nfl_data_py.
        
        Args:
            years: List of NFL season years
            teams: Optional list of team abbreviations to filter by (canonical, e.g., KC, SF)
            
        Returns:
            Raw depth charts data DataFrame
        """
        self.logger.info(f"Fetching depth charts data for years {years}")
        try:
            depth_charts_df = nfl.import_depth_charts(years)
            self.logger.info(f"Successfully fetched {len(depth_charts_df)} depth charts records for {len(years)} years")

            # Optional team filtering across schema variants
            if teams:
                # Normalize provided team codes
                norm_teams = [normalize_team_abbr(t) for t in teams if t]
                norm_teams = [t for t in norm_teams if t]
                if not norm_teams:
                    self.logger.warning("No valid team codes provided after normalization; skipping filter")
                    return depth_charts_df

                cols = set(depth_charts_df.columns)
                if 'team' in cols:
                    before = len(depth_charts_df)
                    depth_charts_df = depth_charts_df[depth_charts_df['team'].isin(norm_teams)]
                    self.logger.info(f"Filtered by team (v2): {before} -> {len(depth_charts_df)} records for teams {norm_teams}")
                elif 'club_code' in cols:
                    before = len(depth_charts_df)
                    depth_charts_df = depth_charts_df[depth_charts_df['club_code'].isin(norm_teams)]
                    self.logger.info(f"Filtered by team (v1): {before} -> {len(depth_charts_df)} records for teams {norm_teams}")
                else:
                    self.logger.warning("Could not apply team filter: no 'team' or 'club_code' column present")

            # Optionally keep only the latest snapshot per team
            if latest_only:
                cols = set(depth_charts_df.columns)
                if 'dt' in cols:
                    df = depth_charts_df.copy()
                    # Parse timestamps
                    try:
                        df['__dt'] = pd.to_datetime(df['dt'], errors='coerce')
                    except Exception:
                        self.logger.warning("Failed to parse 'dt' to datetime; skipping latest-only filter")
                        return depth_charts_df

                    if as_of:
                        try:
                            cutoff = pd.to_datetime(as_of)
                            before = len(df)
                            df = df[df['__dt'] <= cutoff]
                            self.logger.info(f"Applied as_of={as_of} filter: {before} -> {len(df)} rows")
                        except Exception as e:
                            self.logger.warning(f"Invalid as_of '{as_of}': {e}; ignoring")

                    if df.empty:
                        self.logger.warning("No rows remain after as_of filter; skipping latest-only filter")
                        return df

                    latest = df.groupby('team', as_index=False)['__dt'].max().rename(columns={'__dt': '__latest_dt'})
                    merged = df.merge(latest, on='team', how='inner')
                    filtered = merged[merged['__dt'] == merged['__latest_dt']].drop(columns=['__dt', '__latest_dt'])
                    self.logger.info(f"Latest-only per team (v2): {len(depth_charts_df)} -> {len(filtered)} rows")
                    depth_charts_df = filtered
                elif {'season', 'week'}.issubset(cols):
                    # Approximate latest for v1 by max (season, week) per team
                    df = depth_charts_df.copy()
                    # Ensure numeric
                    def _to_int(x):
                        try:
                            return int(float(x))
                        except Exception:
                            return None
                    df['__season'] = df['season'].apply(_to_int)
                    df['__week'] = df['week'].apply(_to_int)
                    # Drop rows without season/week
                    df = df.dropna(subset=['__season', '__week'])
                    # Get latest season per team
                    latest_season = df.groupby('club_code', as_index=False)['__season'].max().rename(columns={'__season':'__latest_season'})
                    df2 = df.merge(latest_season, on='club_code', how='inner')
                    df2 = df2[df2['__season'] == df2['__latest_season']]
                    # Then latest week per team in that season
                    latest_week = df2.groupby('club_code', as_index=False)['__week'].max().rename(columns={'__week':'__latest_week'})
                    df3 = df2.merge(latest_week, on='club_code', how='inner')
                    filtered = df3[df3['__week'] == df3['__latest_week']].drop(columns=['__season','__week','__latest_season','__latest_week'])
                    self.logger.info(f"Latest-only per team (v1): {len(depth_charts_df)} -> {len(filtered)} rows")
                    depth_charts_df = filtered
                else:
                    self.logger.warning("latest_only requested but no dt/season+week available; skipping")

            return depth_charts_df
        except Exception as e:
            self.logger.error(f"Failed to fetch depth charts data for years {years}: {e}")
            raise
    
    def load_depth_charts_data(self, years: List[int], dry_run: bool = False, 
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


class DepthChartsDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of depth charts data.
    
    Transforms raw depth charts data from nfl_data_py into the format expected
    by our depth_charts database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return a default set of required columns.

        Note: Actual validation is handled in _validate_required_columns to
        support multiple upstream schemas (pre-2025 and 2025+).
        """
        return ['gsis_id']

    def _validate_required_columns(self, raw_df: pd.DataFrame) -> None:
        """Validate presence of columns for either supported schema variant.

        Supports:
        - v1 (<= 2024): ['season','week','club_code','gsis_id','full_name','position','depth_team']
        - v2 (>= 2025): ['dt','team','player_name','gsis_id','pos_abb' or 'pos_name']
        """
        cols = set(raw_df.columns)
        v1_required = {'season', 'week', 'club_code', 'gsis_id', 'full_name', 'position'}
        v2_required_base = {'dt', 'team', 'gsis_id'}

        v1_ok = v1_required.issubset(cols)
        v2_ok = v2_required_base.issubset(cols) and (('pos_abb' in cols) or ('pos_name' in cols)) and ('player_name' in cols)

        if not (v1_ok or v2_ok):
            missing_v1 = list(v1_required - cols)
            missing_v2 = list(v2_required_base - cols)
            # Also check the either-or fields for v2
            if not (('pos_abb' in cols) or ('pos_name' in cols)):
                missing_v2.append('pos_abb|pos_name')
            if 'player_name' not in cols:
                missing_v2.append('player_name')
            raise ValueError(f"Missing required columns for depth charts. v1 missing: {missing_v1}; v2 missing: {missing_v2}")
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single depth charts record.
        
        Args:
            row: Single row from raw depth charts DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        cols = row.index
        # Branch based on schema variant
        if 'club_code' in cols:  # v1 (<= 2024)
            team_abbr = normalize_team_abbr(row.get('club_code')) if pd.notna(row.get('club_code')) else None
            pos_name = str(row.get('depth_position', '')) if pd.notna(row.get('depth_position')) else None
            pos_abb = str(row.get('position', '')) if pd.notna(row.get('position')) else None
            pos_slot = int(row.get('depth_team')) if pd.notna(row.get('depth_team')) else None
            record = {
                'gsis_id': str(row.get('gsis_id', '')) if pd.notna(row.get('gsis_id')) else None,
                'player_name': str(row.get('full_name', '')) if pd.notna(row.get('full_name')) else None,
                'team': team_abbr,
                'pos_grp': str(row.get('formation', '')) if pd.notna(row.get('formation')) else None,
                'pos_name': pos_name,
                'pos_abb': pos_abb,
                'pos_slot': pos_slot,
                'pos_rank': pos_slot,  # no explicit rank in v1; approximate with slot
            }
        else:  # v2 (>= 2025)
            # Derive season from timestamp; week is unknown
            season = None
            if pd.notna(row.get('dt')):
                try:
                    season = pd.to_datetime(row.get('dt')).year
                except Exception:
                    season = None

            team_abbr = normalize_team_abbr(row.get('team')) if pd.notna(row.get('team')) else None
            position_abb = str(row.get('pos_abb')) if pd.notna(row.get('pos_abb')) else None
            position_name = str(row.get('pos_name')) if pd.notna(row.get('pos_name')) else None

            record = {
                'gsis_id': str(row.get('gsis_id', '')) if pd.notna(row.get('gsis_id')) else None,
                'player_name': str(row.get('player_name', '')) if pd.notna(row.get('player_name')) else None,
                'team': team_abbr,
                'pos_grp': str(row.get('pos_grp', '')) if pd.notna(row.get('pos_grp')) else None,
                'pos_name': position_name,
                'pos_abb': position_abb,
                'pos_slot': int(row.get('pos_slot')) if pd.notna(row.get('pos_slot')) else None,
                'pos_rank': int(row.get('pos_rank')) if pd.notna(row.get('pos_rank')) else None,
            }
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed depth charts record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Required minimal fields for DB insert
        if not record.get('gsis_id') or not record.get('team') or not record.get('player_name'):
            return False
        # Prefer to have a position abbreviation
        if not record.get('pos_abb') and not record.get('pos_name'):
            return False

        # pos_slot and pos_rank should be positive if present
        for k in ('pos_slot', 'pos_rank'):
            if record.get(k) is not None:
                try:
                    if int(record[k]) < 1:
                        return False
                except Exception:
                    return False

        return True