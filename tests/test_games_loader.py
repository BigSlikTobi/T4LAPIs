"""Tests for the games data loader."""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
from src.core.data.loaders.games import GamesDataLoader


class TestGamesDataLoader(unittest.TestCase):
    """Test cases for GamesDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_raw_game_data = pd.DataFrame([
            {
                'game_id': 'G001',
                'season': 2024,
                'week': 1,
                'home_team': 'KC',
                'away_team': 'DEN',
                'gameday': '2024-09-05',
                'gametime': '20:20',
                'home_score': 21,
                'away_score': 17
            },
            {
                'game_id': 'G002',
                'season': 2024,
                'week': 1,
                'home_team': 'BUF',
                'away_team': 'NYJ',
                'gameday': '2024-09-09',
                'gametime': '20:15',
                'home_score': 24,
                'away_score': 20
            }
        ])

        self.sample_transformed_game_data = [
            {
                'game_id': 'G001',
                'season': 2024,
                'week': 1,
                'home_team': 'KC',
                'away_team': 'DEN',
                'game_date': '2024-09-05',
                'game_time': '20:20',
                'home_score': 21,
                'away_score': 17,
                'total_score': 38
            },
            {
                'game_id': 'G002',
                'season': 2024,
                'week': 1,
                'home_team': 'BUF',
                'away_team': 'NYJ',
                'game_date': '2024-09-09',
                'game_time': '20:15',
                'home_score': 24,
                'away_score': 20,
                'total_score': 44
            }
        ]

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_init(self, mock_db_manager_class):
        """Test loader initialization."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = GamesDataLoader()
        
        self.assertEqual(loader.table_name, "games")
        mock_db_manager_class.assert_called_once_with("games")

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('src.core.data.loaders.games.fetch_game_schedule_data')
    def test_fetch_raw_data_success(self, mock_fetch, mock_db_manager_class):
        """Test successful fetch of game data."""
        mock_db_manager_class.return_value = Mock()
        mock_fetch.return_value = self.sample_raw_game_data
        
        loader = GamesDataLoader()
        result = loader.fetch_raw_data(season=2024, week=1)
        
        mock_fetch.assert_called_once_with([2024])  # Called with list
        # Result should be filtered by week, but still same DataFrame for this test
        self.assertEqual(len(result), 2)  # Both games in our test data are week 1

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_transformer_class_property(self, mock_db_manager_class):
        """Test that transformer_class property returns correct class."""
        mock_db_manager_class.return_value = Mock()
        
        loader = GamesDataLoader()
        
        from src.core.data.transform import GameDataTransformer
        self.assertEqual(loader.transformer_class, GameDataTransformer)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_success(self, mock_db_manager_class):
        """Test successful data loading."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = GamesDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_game_data), \
             patch.object(loader.db_manager, 'upsert_records') as mock_insert:
            
            mock_insert.return_value = {
                'success': True,
                'affected_rows': 2
            }
            
            # Mock transformer
            from src.core.data.transform import GameDataTransformer
            with patch.object(GameDataTransformer, 'transform', return_value=self.sample_transformed_game_data):
                result = loader.load_data(season=2024, week=1)
                
                self.assertTrue(result['success'])
                self.assertEqual(result['total_fetched'], 2)  # Number of rows in DataFrame
                self.assertEqual(result['total_validated'], 2)  # 2 records
                self.assertEqual(result['upsert_result']['affected_rows'], 2)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_dry_run(self, mock_db_manager_class):
        """Test dry run mode."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = GamesDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_game_data), \
             patch.object(loader.db_manager, 'upsert_records') as mock_insert:
            
            # Mock transformer
            from src.core.data.transform import GameDataTransformer
            with patch.object(GameDataTransformer, 'transform', return_value=self.sample_transformed_game_data):
                result = loader.load_data(season=2024, week=1, dry_run=True)
                
                self.assertTrue(result['success'])
                self.assertTrue(result['dry_run'])
                mock_insert.assert_not_called()

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_game_count(self, mock_db_manager_class):
        """Test getting game count."""
        mock_db_manager = Mock()
        mock_db_manager.get_record_count.return_value = 285
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = GamesDataLoader()
        count = loader.get_game_count()
        
        self.assertEqual(count, 285)
        mock_db_manager.get_record_count.assert_called_once_with({})


if __name__ == '__main__':
    unittest.main()
