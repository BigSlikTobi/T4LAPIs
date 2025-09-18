"""Tests for the Next Gen Stats data loader."""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
from src.core.data.loaders.ngs import NextGenStatsDataLoader, NextGenStatsDataTransformer


class TestNextGenStatsDataLoader(unittest.TestCase):
    """Test cases for NextGenStatsDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_raw_ngs_data = pd.DataFrame([
            {
                'player_id': 'P001',
                'player_display_name': 'Patrick Mahomes',
                'season': 2023,
                'week': 1,
                'team_abbr': 'KC',
                'position': 'QB',
                'attempts': 35,
                'pass_yards': 312,
                'pass_touchdowns': 2,
                'interceptions': 0,
                'passer_rating': 118.5,
                'completions': 24,
                'completion_percentage': 68.6,
                'expected_completion_percentage': 65.2,
                'completion_percentage_above_expectation': 3.4,
                'avg_time_to_throw': 2.85,
                'avg_completed_air_yards': 7.2,
                'avg_intended_air_yards': 8.1,
                'avg_air_yards_differential': -0.9,
                'avg_air_yards_to_sticks': 1.2
            },
            {
                'player_id': 'P002',
                'player_display_name': 'Travis Kelce',
                'season': 2023,
                'week': 1,
                'team_abbr': 'KC',
                'position': 'TE',
                'targets': 8,
                'receptions': 6,
                'receiving_yards': 81,
                'receiving_touchdowns': 1,
                'target_share': 22.9,
                'avg_cushion': 3.2,
                'avg_separation': 2.8,
                'avg_target_separation': 2.6,
                'catch_percentage': 75.0,
                'share_of_intended_air_yards': 18.5,
                'avg_yac': 4.5,
                'avg_expected_yac': 3.8,
                'avg_yac_above_expectation': 0.7
            }
        ])

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_init_success(self, mock_db_manager_class):
        """Test successful initialization of NextGenStatsDataLoader."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = NextGenStatsDataLoader()
        
        assert loader.table_name == "next_gen_stats"
        assert loader.db_manager == mock_db_manager
        mock_db_manager_class.assert_called_once_with("next_gen_stats")

    @patch('src.core.data.loaders.ngs.fetch_ngs_data')
    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_fetch_raw_data(self, mock_db_manager_class, mock_fetch):
        """Test fetching raw Next Gen Stats data."""
        mock_fetch.return_value = self.sample_raw_ngs_data
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = NextGenStatsDataLoader()
        result = loader.fetch_raw_data('passing', [2023])
        
        mock_fetch.assert_called_once_with('passing', [2023])
        assert len(result) == 2
        assert 'player_id' in result.columns

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_dry_run(self, mock_db_manager_class):
        """Test dry run mode for NGS data loading."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = NextGenStatsDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_ngs_data):
            result = loader.load_data(stat_type='passing', years=[2023], dry_run=True)
        
        assert result['success'] is True
        assert result['dry_run'] is True
        assert result['would_upsert'] == 2
        mock_db_manager.upsert_records.assert_not_called()


class TestNextGenStatsDataTransformer(unittest.TestCase):
    """Test cases for NextGenStatsDataTransformer."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = NextGenStatsDataTransformer()
        
        self.sample_passing_row = pd.Series({
            'player_id': 'P001',
            'player_display_name': 'Patrick Mahomes',
            'season': 2023,
            'week': 1,
            'team_abbr': 'KC',
            'position': 'QB',
            'attempts': 35,
            'pass_yards': 312,
            'pass_touchdowns': 2,
            'interceptions': 0,
            'passer_rating': 118.5,
            'completions': 24,
            'completion_percentage': 68.6,
            'expected_completion_percentage': 65.2,
            'completion_percentage_above_expectation': 3.4,
            'avg_time_to_throw': 2.85,
            'avg_completed_air_yards': 7.2,
            'avg_intended_air_yards': 8.1,
            'avg_air_yards_differential': -0.9,
            'avg_air_yards_to_sticks': 1.2
        })
        
        self.sample_receiving_row = pd.Series({
            'player_id': 'P002',
            'player_display_name': 'Travis Kelce',
            'season': 2023,
            'week': 1,
            'team_abbr': 'KC',
            'position': 'TE',
            'targets': 8,
            'receptions': 6,
            'receiving_yards': 81,
            'receiving_touchdowns': 1,
            'target_share': 22.9,
            'avg_cushion': 3.2,
            'avg_separation': 2.8,
            'avg_target_separation': 2.6,
            'catch_percentage': 75.0,
            'share_of_intended_air_yards': 18.5,
            'avg_yac': 4.5,
            'avg_expected_yac': 3.8,
            'avg_yac_above_expectation': 0.7
        })

    def test_get_required_columns(self):
        """Test required columns list."""
        required = self.transformer._get_required_columns()
        expected = [
            'player_id', 'player_display_name', 'season', 'week', 'team_abbr'
        ]
        assert required == expected

    def test_transform_passing_record_success(self):
        """Test successful transformation of a passing record."""
        result = self.transformer._transform_single_record(self.sample_passing_row)
        
        assert result is not None
        assert result['player_id'] == 'P001'
        assert result['player_display_name'] == 'Patrick Mahomes'
        assert result['season'] == 2023
        assert result['week'] == 1
        assert result['team_abbr'] == 'KC'
        assert result['position'] == 'QB'
        assert result['attempts'] == 35
        assert result['pass_yards'] == 312.0
        assert result['pass_touchdowns'] == 2
        assert result['interceptions'] == 0
        assert result['passer_rating'] == 118.5
        assert result['completion_percentage'] == 68.6
        assert result['avg_time_to_throw'] == 2.85

    def test_transform_receiving_record_success(self):
        """Test successful transformation of a receiving record."""
        result = self.transformer._transform_single_record(self.sample_receiving_row)
        
        assert result is not None
        assert result['player_id'] == 'P002'
        assert result['player_display_name'] == 'Travis Kelce'
        assert result['targets'] == 8
        assert result['receptions'] == 6
        assert result['receiving_yards'] == 81.0
        assert result['receiving_touchdowns'] == 1
        assert result['target_share'] == 22.9
        assert result['avg_separation'] == 2.8
        assert result['catch_percentage'] == 75.0

    def test_transform_record_with_nulls(self):
        """Test transformation with null/NaN values."""
        row_with_nulls = self.sample_passing_row.copy()
        row_with_nulls['avg_time_to_throw'] = pd.NA
        row_with_nulls['passer_rating'] = None
        
        result = self.transformer._transform_single_record(row_with_nulls)
        
        assert result is not None
        assert result['avg_time_to_throw'] is None
        assert result['passer_rating'] is None

    def test_validate_record_success(self):
        """Test successful record validation."""
        valid_record = {
            'player_id': 'P001',
            'player_display_name': 'Patrick Mahomes',
            'season': 2023,
            'team_abbr': 'KC'
        }
        
        assert self.transformer._validate_record(valid_record) is True

    def test_validate_record_missing_required_fields(self):
        """Test validation with missing required fields."""
        invalid_record = {
            'player_id': 'P001',
            # Missing player_display_name, season, team_abbr
        }
        
        assert self.transformer._validate_record(invalid_record) is False

    def test_validate_record_invalid_season(self):
        """Test validation with invalid season (too early for NGS data)."""
        invalid_record = {
            'player_id': 'P001',
            'player_display_name': 'Patrick Mahomes',
            'season': 2010,  # Before NGS era
            'team_abbr': 'KC'
        }
        
        assert self.transformer._validate_record(invalid_record) is False

    def test_validate_record_invalid_week(self):
        """Test validation with invalid week."""
        invalid_record = {
            'player_id': 'P001',
            'player_display_name': 'Patrick Mahomes',
            'season': 2023,
            'week': 25,  # Too high
            'team_abbr': 'KC'
        }
        
        assert self.transformer._validate_record(invalid_record) is False


if __name__ == '__main__':
    unittest.main()