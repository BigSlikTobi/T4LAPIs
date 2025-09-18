"""Tests for the play-by-play data loader."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.core.data.loaders.pbp import PlayByPlayDataLoader, PlayByPlayDataTransformer


class TestPlayByPlayDataLoader(unittest.TestCase):
    """Test cases for PlayByPlayDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_raw_pbp_data = pd.DataFrame([
            {
                'play_id': 'P001',
                'game_id': 'G001',
                'season': 2023,
                'week': 1,
                'game_date': '2023-09-10',
                'posteam': 'KC',
                'defteam': 'DET',
                'home_team': 'DET',
                'away_team': 'KC',
                'down': 1,
                'ydstogo': 10,
                'yardline_100': 75,
                'quarter_seconds_remaining': 450,
                'play_type': 'pass',
                'yards_gained': 8,
                'epa': 0.25,
                'wpa': 0.02,
                'wp': 0.55,
                'pass_attempt': True,
                'complete_pass': True,
                'air_yards': 6,
                'yards_after_catch': 2,
                'passer_player_id': 'QB001',
                'receiver_player_id': 'WR001',
                'touchdown': False,
                'total_home_score': 7,
                'total_away_score': 3
            },
            {
                'play_id': 'P002',
                'game_id': 'G001',
                'season': 2023,
                'week': 1,
                'game_date': '2023-09-10',
                'posteam': 'DET',
                'defteam': 'KC',
                'home_team': 'DET',
                'away_team': 'KC',
                'down': 2,
                'ydstogo': 8,
                'yardline_100': 65,
                'quarter_seconds_remaining': 420,
                'play_type': 'run',
                'yards_gained': 12,
                'epa': 0.45,
                'wpa': 0.03,
                'wp': 0.58,
                'rush_attempt': True,
                'rushing_yards': 12,
                'rusher_player_id': 'RB001',
                'touchdown': False,
                'total_home_score': 7,
                'total_away_score': 3
            }
        ])

        self.sample_transformed_pbp_data = [
            {
                'play_id': 'P001',
                'game_id': 'G001',
                'season': 2023,
                'week': 1,
                'game_date': '2023-09-10',
                'posteam': 'KC',
                'defteam': 'DET',
                'home_team': 'DET',
                'away_team': 'KC',
                'down': 1,
                'ydstogo': 10,
                'yardline_100': 75.0,
                'quarter_seconds_remaining': 450,
                'play_type': 'pass',
                'yards_gained': 8.0,
                'epa': 0.25,
                'wpa': 0.02,
                'wp': 0.55,
                'pass_attempt': True,
                'complete_pass': True,
                'air_yards': 6.0,
                'yards_after_catch': 2.0,
                'passer_player_id': 'QB001',
                'receiver_player_id': 'WR001',
                'touchdown': False,
                'total_home_score': 7,
                'total_away_score': 3
            }
        ]

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_init_success(self, mock_db_manager_class):
        """Test successful initialization of PlayByPlayDataLoader."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayByPlayDataLoader()
        
        assert loader.table_name == "play_by_play"
        assert loader.db_manager == mock_db_manager
        mock_db_manager_class.assert_called_once_with("play_by_play")

    @patch('src.core.data.loaders.pbp.fetch_pbp_data')
    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_fetch_raw_data(self, mock_db_manager_class, mock_fetch):
        """Test fetching raw play-by-play data."""
        mock_fetch.return_value = self.sample_raw_pbp_data
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayByPlayDataLoader()
        result = loader.fetch_raw_data([2023], downsampling=True)
        
        mock_fetch.assert_called_once_with([2023], downsampling=True)
        assert len(result) == 2
        assert 'play_id' in result.columns

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_dry_run(self, mock_db_manager_class):
        """Test dry run mode for play-by-play data loading."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayByPlayDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_pbp_data):
            result = loader.load_data(years=[2023], dry_run=True)
        
        assert result['success'] is True
        assert result['dry_run'] is True
        assert result['would_upsert'] == 2
        mock_db_manager.upsert_records.assert_not_called()


class TestPlayByPlayDataTransformer(unittest.TestCase):
    """Test cases for PlayByPlayDataTransformer."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = PlayByPlayDataTransformer()
        
        self.sample_row = pd.Series({
            'play_id': 'P001',
            'game_id': 'G001',
            'season': 2023,
            'week': 1,
            'game_date': '2023-09-10',
            'posteam': 'KC',
            'defteam': 'DET',
            'home_team': 'DET',
            'away_team': 'KC',
            'down': 1,
            'ydstogo': 10,
            'yardline_100': 75,
            'quarter_seconds_remaining': 450,
            'play_type': 'pass',
            'yards_gained': 8,
            'epa': 0.25,
            'wpa': 0.02,
            'wp': 0.55,
            'pass_attempt': True,
            'complete_pass': True,
            'air_yards': 6,
            'yards_after_catch': 2,
            'passer_player_id': 'QB001',
            'receiver_player_id': 'WR001',
            'touchdown': False,
            'total_home_score': 7,
            'total_away_score': 3
        })

    def test_get_required_columns(self):
        """Test required columns list."""
        required = self.transformer._get_required_columns()
        expected = [
            'play_id', 'game_id', 'season', 'week', 'posteam', 'defteam',
            'down', 'ydstogo', 'yardline_100', 'quarter_seconds_remaining',
            'play_type', 'yards_gained', 'epa', 'wpa', 'wp'
        ]
        assert required == expected

    def test_transform_single_record_success(self):
        """Test successful transformation of a single record."""
        result = self.transformer._transform_single_record(self.sample_row)
        
        assert result is not None
        assert result['play_id'] == 'P001'
        assert result['game_id'] == 'G001'
        assert result['season'] == 2023
        assert result['week'] == 1
        assert result['posteam'] == 'KC'
        assert result['defteam'] == 'DET'
        assert result['down'] == 1
        assert result['yards_gained'] == 8.0
        assert result['epa'] == 0.25
        assert result['pass_attempt'] is True
        assert result['complete_pass'] is True

    def test_transform_single_record_with_nulls(self):
        """Test transformation with null/NaN values."""
        row_with_nulls = self.sample_row.copy()
        row_with_nulls['air_yards'] = pd.NA
        row_with_nulls['yards_after_catch'] = None
        row_with_nulls['passer_player_id'] = ''
        
        result = self.transformer._transform_single_record(row_with_nulls)
        
        assert result is not None
        assert result['air_yards'] is None
        assert result['yards_after_catch'] is None
        assert result['passer_player_id'] == ''

    def test_validate_record_success(self):
        """Test successful record validation."""
        valid_record = {
            'play_id': 'P001',
            'game_id': 'G001',
            'season': 2023,
            'week': 1
        }
        
        assert self.transformer._validate_record(valid_record) is True

    def test_validate_record_missing_required_fields(self):
        """Test validation with missing required fields."""
        invalid_record = {
            'play_id': 'P001',
            # Missing game_id, season, week
        }
        
        assert self.transformer._validate_record(invalid_record) is False

    def test_validate_record_invalid_season(self):
        """Test validation with invalid season."""
        invalid_record = {
            'play_id': 'P001',
            'game_id': 'G001',
            'season': 1900,  # Too old
            'week': 1
        }
        
        assert self.transformer._validate_record(invalid_record) is False

    def test_validate_record_invalid_week(self):
        """Test validation with invalid week."""
        invalid_record = {
            'play_id': 'P001',
            'game_id': 'G001',
            'season': 2023,
            'week': 25  # Too high
        }
        
        assert self.transformer._validate_record(invalid_record) is False


if __name__ == '__main__':
    unittest.main()