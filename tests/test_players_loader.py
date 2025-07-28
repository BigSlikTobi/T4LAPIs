"""Tests for the players data loader."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.core.data.loaders.players import PlayersDataLoader


class TestPlayersDataLoader(unittest.TestCase):
    """Test cases for PlayersDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_raw_player_data = pd.DataFrame([
            {
                'player_id': 'P001',
                'player_name': 'Patrick Mahomes',
                'team': 'KC',
                'position': 'QB',
                'birth_date': '1995-09-17',
                'draft_year': '2017',
                'draft_round': '1',
                'draft_pick': '10',
                'weight': '230',
                'height': '74',
                'college': 'Texas Tech',
                'years_of_experience': '7'
            },
            {
                'player_id': 'P002',
                'player_name': 'Travis Kelce',
                'team': 'KC',
                'position': 'TE',
                'birth_date': '1989-10-05',
                'draft_year': '2013',
                'draft_round': '3',
                'draft_pick': '63',
                'weight': '250',
                'height': '77',
                'college': 'Cincinnati',
                'years_of_experience': '11'
            }
        ])

        self.sample_transformed_player_data = [
            {
                'player_id': 'P001',
                'player_name': 'Patrick Mahomes',
                'latest_team': 'KC',
                'position': 'QB',
                'position_group': 'QB',
                'birth_date': '1995-09-17',
                'draft_team': 'KC',
                'draft_year': 2017,
                'draft_round': 1,
                'draft_pick': 10,
                'weight': 230,
                'height_inches': 74,
                'college': 'Texas Tech',
                'years_of_experience': 7
            },
            {
                'player_id': 'P002',
                'player_name': 'Travis Kelce',
                'latest_team': 'KC',
                'position': 'TE',
                'position_group': 'Receiving',
                'birth_date': '1989-10-05',
                'draft_team': 'KC',
                'draft_year': 2013,
                'draft_round': 3,
                'draft_pick': 63,
                'weight': 250,
                'height_inches': 77,
                'college': 'Cincinnati',
                'years_of_experience': 11
            }
        ]

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_init(self, mock_db_manager_class):
        """Test loader initialization."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayersDataLoader()
        
        self.assertEqual(loader.table_name, "players")
        mock_db_manager_class.assert_called_once_with("players")

    @patch('src.core.data.loaders.players.fetch_player_data')
    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_fetch_raw_data_success(self, mock_db_manager_class, mock_fetch):
        """Test successful fetch of player data."""
        mock_db_manager_class.return_value = Mock()
        mock_fetch.return_value = self.sample_raw_player_data
        
        loader = PlayersDataLoader()
        result = loader.fetch_raw_data(season=2024)
        
        mock_fetch.assert_called_once_with([2024])  # Called with list
        pd.testing.assert_frame_equal(result, self.sample_raw_player_data)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_transformer_class_property(self, mock_db_manager_class):
        """Test that transformer_class property returns correct class."""
        mock_db_manager_class.return_value = Mock()
        
        loader = PlayersDataLoader()
        
        from src.core.data.transform import PlayerDataTransformer
        self.assertEqual(loader.transformer_class, PlayerDataTransformer)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_success(self, mock_db_manager_class):
        """Test successful data loading."""
        mock_db_manager = Mock()
        mock_db_manager.upsert_records.return_value = {
            'success': True,
            'affected_rows': 2
        }
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayersDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_player_data):
            # Mock transformer
            from src.core.data.transform import PlayerDataTransformer
            with patch.object(PlayerDataTransformer, 'transform', return_value=self.sample_transformed_player_data):
                result = loader.load_data(season=2024)
                
                self.assertTrue(result['success'])
                self.assertEqual(result['total_fetched'], 2)  # 2 records in dataframe
                self.assertEqual(result['total_validated'], 2)  # 2 records
                self.assertEqual(result['upsert_result']['affected_rows'], 2)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_dry_run(self, mock_db_manager_class):
        """Test dry run mode."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayersDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_player_data):
            # Mock transformer
            from src.core.data.transform import PlayerDataTransformer
            with patch.object(PlayerDataTransformer, 'transform', return_value=self.sample_transformed_player_data):
                result = loader.load_data(season=2024, dry_run=True)
                
                self.assertTrue(result['success'])
                self.assertTrue(result['dry_run'])
                mock_db_manager.upsert_records.assert_not_called()

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_player_count(self, mock_db_manager_class):
        """Test getting player count."""
        mock_db_manager = Mock()
        mock_db_manager.get_record_count.return_value = 100
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = PlayersDataLoader()
        count = loader.get_player_count()
        
        self.assertEqual(count, 100)
        mock_db_manager.get_record_count.assert_called_once_with(None)


if __name__ == '__main__':
    unittest.main()
