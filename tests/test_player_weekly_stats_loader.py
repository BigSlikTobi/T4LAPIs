"""Tests for the player weekly stats data loader."""

import unittest
import pandas as pd
from unittest.mock import Mock, patch
from src.core.data.loaders.player_weekly_stats import PlayerWeeklyStatsDataLoader


class TestPlayerWeeklyStatsDataLoader(unittest.TestCase):
    """Test cases for PlayerWeeklyStatsDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        import pandas as pd
        
        self.sample_raw_stats_data = pd.DataFrame([
            {
                'player_id': 'P001',
                'game_id': 'G001',
                'week': 1,
                'season': 2024,
                'passing_yards': 350,
                'passing_tds': 3,
                'rushing_yards': 25,
                'rushing_tds': 0
            },
            {
                'player_id': 'P002',
                'game_id': 'G001',
                'week': 1,
                'season': 2024,
                'receiving_yards': 120,
                'receiving_tds': 2,
                'receptions': 8,
                'targets': 10
            }
        ])

        self.sample_transformed_stats_data = [
            {
                'stat_id': 'S001',
                'player_id': 'P001',
                'game_id': 'G001',
                'week': 1,
                'season': 2024,
                'passing_yards': 350,
                'passing_tds': 3,
                'rushing_yards': 25,
                'rushing_tds': 0,
                'fantasy_points': 28.5
            },
            {
                'stat_id': 'S002',
                'player_id': 'P002',
                'game_id': 'G001',
                'week': 1,
                'season': 2024,
                'receiving_yards': 120,
                'receiving_tds': 2,
                'receptions': 8,
                'targets': 10,
                'fantasy_points': 24.0
            }
        ]

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_init(self, mock_db_manager_class):
        """Test loader initialization."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayerWeeklyStatsDataLoader()
        
        self.assertEqual(loader.table_name, "player_weekly_stats")
        mock_db_manager_class.assert_called_once_with("player_weekly_stats")

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('src.core.data.loaders.player_weekly_stats.fetch_weekly_stats_data')
    def test_fetch_raw_data_success(self, mock_fetch, mock_db_manager_class):
        """Test successful fetch of weekly stats data."""
        mock_db_manager_class.return_value = Mock()
        mock_fetch.return_value = self.sample_raw_stats_data
        
        loader = PlayerWeeklyStatsDataLoader()
        result = loader.fetch_raw_data(years=[2024], weeks=[1])
        
        mock_fetch.assert_called_once_with(years=[2024], weeks=[1])
        pd.testing.assert_frame_equal(result, self.sample_raw_stats_data)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_fetch_raw_data_default_years(self, mock_db_manager_class):
        """Test fetch with default years parameter."""
        mock_db_manager_class.return_value = Mock()
        
        loader = PlayerWeeklyStatsDataLoader()
        
        with patch('src.core.data.loaders.player_weekly_stats.fetch_weekly_stats_data') as mock_fetch:
            loader.fetch_raw_data()
            mock_fetch.assert_called_once_with(years=[2024], weeks=None)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_transformer_class_property(self, mock_db_manager_class):
        """Test that transformer_class property returns correct class."""
        mock_db_manager_class.return_value = Mock()
        
        loader = PlayerWeeklyStatsDataLoader()
        
        from src.core.data.transform import PlayerWeeklyStatsDataTransformer
        self.assertEqual(loader.transformer_class, PlayerWeeklyStatsDataTransformer)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_weekly_stats_success(self, mock_db_manager_class):
        """Test successful weekly stats loading using legacy method."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayerWeeklyStatsDataLoader()
        
        with patch.object(loader, 'load_data') as mock_load_data:
            mock_load_data.return_value = {
                'success': True,
                'total_fetched': 1,
                'total_validated': 2,
                'upsert_result': {'affected_rows': 2}
            }
            
            result = loader.load_weekly_stats(years=[2024], weeks=[1], batch_size=50)
            
            mock_load_data.assert_called_once_with(
                years=[2024],
                weeks=[1],
                batch_size=50
            )
            self.assertTrue(result['success'])

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_stats_count(self, mock_db_manager_class):
        """Test getting stats count."""
        mock_db_manager = Mock()
        mock_db_manager.get_record_count.return_value = 1500
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayerWeeklyStatsDataLoader()
        count = loader.get_stats_count()
        
        self.assertEqual(count, 1500)
        mock_db_manager.get_record_count.assert_called_once_with('player_weekly_stats')

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_success(self, mock_db_manager_class):
        """Test successful data loading."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayerWeeklyStatsDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_stats_data), \
             patch.object(loader.db_manager, 'upsert_records') as mock_insert:
            
            mock_insert.return_value = {
                'success': True,
                'affected_rows': 2
            }
            
            # Mock transformer class itself to avoid initialization issues
            from src.core.data.transform import PlayerWeeklyStatsDataTransformer
            with patch.object(PlayerWeeklyStatsDataTransformer, '__init__', return_value=None), \
                 patch.object(PlayerWeeklyStatsDataTransformer, 'transform', return_value=self.sample_transformed_stats_data):
                result = loader.load_data(years=[2024], weeks=[1])
                
                self.assertTrue(result['success'])
                self.assertEqual(result['total_fetched'], 2)  # 2 records in DataFrame
                self.assertEqual(result['total_validated'], 2)  # 2 records
                self.assertEqual(result['upsert_result']['affected_rows'], 2)


if __name__ == '__main__':
    unittest.main()
