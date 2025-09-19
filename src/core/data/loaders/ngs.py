"""
Next Gen Stats data loader for advanced NFL analytics.
This module provides functionality to fetch NFL Next Gen Stats data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..fetch import fetch_ngs_data
from ..transform import BaseDataTransformer


class NextGenStatsDataLoader(BaseDataLoader):
    """Loads NFL Next Gen Stats data into the database.
    
    NGS data includes advanced tracking metrics for passing, rushing, and receiving.
    """
    
    def __init__(self):
        """Initialize the Next Gen Stats data loader."""
        super().__init__("next_gen_stats")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the NextGenStatsDataTransformer class."""
        return NextGenStatsDataTransformer
    
    def fetch_raw_data(self, stat_type: str, years: List[int], player_id=None) -> pd.DataFrame:
        """Fetch raw Next Gen Stats data from nfl_data_py.
        
        Args:
            stat_type: Type of NGS data ('passing', 'rushing', 'receiving')
            years: List of NFL season years
            player_id: Optional GSIS player ID or list of IDs to filter results
            
        Returns:
            Raw Next Gen Stats data DataFrame
        """
        # Support auto mode by aggregating across all NGS types
        if stat_type == 'auto':
            combined: List[pd.DataFrame] = []
            for st in ['passing', 'rushing', 'receiving']:
                self.logger.info(f"Fetching Next Gen Stats {st} data for years {years}")
                sdf = fetch_ngs_data(st, years)
                # Apply player filter per-slice if requested
                if player_id:
                    sdf = self._filter_by_player_id(sdf, player_id, st)
                if not sdf.empty:
                    combined.append(sdf)
            if not combined:
                return pd.DataFrame()
            return pd.concat(combined, ignore_index=True, sort=False)

        self.logger.info(f"Fetching Next Gen Stats {stat_type} data for years {years}")
        df = fetch_ngs_data(stat_type, years)

        # Optional filtering by player_id (accept str or list)
        if player_id:
            df = self._filter_by_player_id(df, player_id, stat_type)

        return df

    def _filter_by_player_id(self, df: pd.DataFrame, player_id, stat_type: str) -> pd.DataFrame:
        """Filter an NGS dataframe by player_id(s).
        Accepts string (possibly comma-separated) or an iterable of IDs.
        """
        # Normalize incoming parameter into a set of string IDs
        if isinstance(player_id, (list, tuple, set)):
            player_ids = {str(x).strip() for x in player_id if x is not None}
        else:
            s = str(player_id).strip()
            # allow comma-separated list
            player_ids = {p.strip() for p in s.split(',') if p.strip()}

        if not player_ids or df.empty:
            return df

        initial_count = len(df)
        # Prefer filtering by player_gsis_id if present; fallback to player_id
        if 'player_gsis_id' in df.columns:
            out = df[df['player_gsis_id'].astype(str).isin(player_ids)]
        elif 'player_id' in df.columns:
            out = df[df['player_id'].astype(str).isin(player_ids)]
        else:
            out = df
        self.logger.info(
            f"Filtered NGS {stat_type} data by player_id(s) {sorted(player_ids)}: {len(out)} of {initial_count} records"
        )
        return out
    
    def load_ngs_data(self, stat_type: str, years: List[int], dry_run: bool = False, 
                      clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            stat_type: Type of NGS data ('passing', 'rushing', 'receiving')
            years: List of NFL season years
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(stat_type=stat_type, years=years, dry_run=dry_run, 
                             clear_table=clear_table)


class NextGenStatsDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of Next Gen Stats data.
    
    Transforms raw NGS data from nfl_data_py into the format expected
    by our next_gen_stats database table.
    """
    
    def transform(self, raw_df: pd.DataFrame) -> List[dict]:
        """Override to normalize expected column names before validation.
        
        nfl_data_py uses `player_gsis_id` and `player_position` in NGS outputs.
        Our pipeline expects `player_id` and `position`. Create aliases when needed
        so the base pipeline's required column checks pass.
        """
        df = raw_df.copy()
        # Alias player_gsis_id -> player_id when missing
        if 'player_id' not in df.columns and 'player_gsis_id' in df.columns:
            df['player_id'] = df['player_gsis_id']
        # Alias player_position -> position when missing
        if 'position' not in df.columns and 'player_position' in df.columns:
            df['position'] = df['player_position']
        return super().transform(df)
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for Next Gen Stats data."""
        return [
            'player_id', 'player_display_name', 'season', 'week', 'team_abbr'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single Next Gen Stats record.
        
        Args:
            row: Single row from raw NGS DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'player_id': str(row.get('player_id', '')) if pd.notna(row.get('player_id')) else None,
            'player_display_name': str(row.get('player_display_name', '')) if pd.notna(row.get('player_display_name')) else None,
            'season': int(row.get('season', 0)) if pd.notna(row.get('season')) else None,
            'week': int(row.get('week', 0)) if pd.notna(row.get('week')) else None,
            'team_abbr': str(row.get('team_abbr', '')) if pd.notna(row.get('team_abbr')) else None,
        }
        
        # Position and basic info
        record.update({
            'position': str(row.get('position', '')) if pd.notna(row.get('position')) else None,
            'player_short_name': str(row.get('player_short_name', '')) if pd.notna(row.get('player_short_name')) else None,
        })
        
        # Common NGS metrics (available in all stat types)
        record.update({
            'avg_time_to_throw': float(row.get('avg_time_to_throw', 0)) if pd.notna(row.get('avg_time_to_throw')) else None,
            'avg_completed_air_yards': float(row.get('avg_completed_air_yards', 0)) if pd.notna(row.get('avg_completed_air_yards')) else None,
            'avg_intended_air_yards': float(row.get('avg_intended_air_yards', 0)) if pd.notna(row.get('avg_intended_air_yards')) else None,
            'avg_air_yards_differential': float(row.get('avg_air_yards_differential', 0)) if pd.notna(row.get('avg_air_yards_differential')) else None,
        })
        
        # Passing-specific metrics
        record.update({
            'attempts': int(row.get('attempts', 0)) if pd.notna(row.get('attempts')) else None,
            'pass_yards': float(row.get('pass_yards', 0)) if pd.notna(row.get('pass_yards')) else None,
            'pass_touchdowns': int(row.get('pass_touchdowns', 0)) if pd.notna(row.get('pass_touchdowns')) else None,
            'interceptions': int(row.get('interceptions', 0)) if pd.notna(row.get('interceptions')) else None,
            'passer_rating': float(row.get('passer_rating', 0)) if pd.notna(row.get('passer_rating')) else None,
            'completions': int(row.get('completions', 0)) if pd.notna(row.get('completions')) else None,
            'completion_percentage': float(row.get('completion_percentage', 0)) if pd.notna(row.get('completion_percentage')) else None,
            'expected_completion_percentage': float(row.get('expected_completion_percentage', 0)) if pd.notna(row.get('expected_completion_percentage')) else None,
            'completion_percentage_above_expectation': float(row.get('completion_percentage_above_expectation', 0)) if pd.notna(row.get('completion_percentage_above_expectation')) else None,
            'avg_air_yards_to_sticks': float(row.get('avg_air_yards_to_sticks', 0)) if pd.notna(row.get('avg_air_yards_to_sticks')) else None,
        })
        
        # Rushing-specific metrics
        record.update({
            'carries': int(row.get('carries', 0)) if pd.notna(row.get('carries')) else None,
            'rush_yards': float(row.get('rush_yards', 0)) if pd.notna(row.get('rush_yards')) else None,
            'rush_touchdowns': int(row.get('rush_touchdowns', 0)) if pd.notna(row.get('rush_touchdowns')) else None,
            'efficiency': float(row.get('efficiency', 0)) if pd.notna(row.get('efficiency')) else None,
            'percent_attempts_gte_eight_defenders': float(row.get('percent_attempts_gte_eight_defenders', 0)) if pd.notna(row.get('percent_attempts_gte_eight_defenders')) else None,
            'avg_time_to_los': float(row.get('avg_time_to_los', 0)) if pd.notna(row.get('avg_time_to_los')) else None,
            'expected_rush_yards': float(row.get('expected_rush_yards', 0)) if pd.notna(row.get('expected_rush_yards')) else None,
            'rush_yards_over_expected': float(row.get('rush_yards_over_expected', 0)) if pd.notna(row.get('rush_yards_over_expected')) else None,
            'avg_rush_yards': float(row.get('avg_rush_yards', 0)) if pd.notna(row.get('avg_rush_yards')) else None,
            'rush_yards_over_expected_per_att': float(row.get('rush_yards_over_expected_per_att', 0)) if pd.notna(row.get('rush_yards_over_expected_per_att')) else None,
            'rush_pct_over_expected': float(row.get('rush_pct_over_expected', 0)) if pd.notna(row.get('rush_pct_over_expected')) else None,
        })
        
        # Receiving-specific metrics
        record.update({
            'targets': int(row.get('targets', 0)) if pd.notna(row.get('targets')) else None,
            'receptions': int(row.get('receptions', 0)) if pd.notna(row.get('receptions')) else None,
            'receiving_yards': float(row.get('receiving_yards', 0)) if pd.notna(row.get('receiving_yards')) else None,
            'receiving_touchdowns': int(row.get('receiving_touchdowns', 0)) if pd.notna(row.get('receiving_touchdowns')) else None,
            'target_share': float(row.get('target_share', 0)) if pd.notna(row.get('target_share')) else None,
            'avg_cushion': float(row.get('avg_cushion', 0)) if pd.notna(row.get('avg_cushion')) else None,
            'avg_separation': float(row.get('avg_separation', 0)) if pd.notna(row.get('avg_separation')) else None,
            'avg_target_separation': float(row.get('avg_target_separation', 0)) if pd.notna(row.get('avg_target_separation')) else None,
            'catch_percentage': float(row.get('catch_percentage', 0)) if pd.notna(row.get('catch_percentage')) else None,
            'share_of_intended_air_yards': float(row.get('share_of_intended_air_yards', 0)) if pd.notna(row.get('share_of_intended_air_yards')) else None,
            'avg_yac': float(row.get('avg_yac', 0)) if pd.notna(row.get('avg_yac')) else None,
            'avg_expected_yac': float(row.get('avg_expected_yac', 0)) if pd.notna(row.get('avg_expected_yac')) else None,
            'avg_yac_above_expectation': float(row.get('avg_yac_above_expectation', 0)) if pd.notna(row.get('avg_yac_above_expectation')) else None,
        })
        
        # Advanced tracking metrics
        record.update({
            'max_speed': float(row.get('max_speed', 0)) if pd.notna(row.get('max_speed')) else None,
            'avg_speed': float(row.get('avg_speed', 0)) if pd.notna(row.get('avg_speed')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed Next Gen Stats record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_id') or not record.get('player_display_name'):
            return False
        
        # Must have valid season and team
        if not record.get('season') or not record.get('team_abbr'):
            return False
        
        # Season should be reasonable (NGS data started around 2016)
        season = record.get('season')
        if season and (season < 2016 or season > 2030):
            return False
        
        # Week should be valid if present (1-22 for regular season + playoffs)
        week = record.get('week')
        if week and (week < 1 or week > 22):
            return False
        
        return True