"""Tests for the rosters data loader."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
from src.core.data.loaders.rosters import RostersDataLoader


class TestRostersDataLoader(unittest.TestCase):
    """Test cases for RostersDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_raw_roster_data = pd.DataFrame([
            {
                'season': 2024,
                'team': 'KC',
                'position': 'QB',
                'depth_chart_position': 'QB',
                'jersey_number': 15.0,
                'status': 'ACT',
                'player_name': 'Patrick Mahomes',
                'first_name': 'Patrick',
                'last_name': 'Mahomes',
                'birth_date': pd.Timestamp('1995-09-17'),
                'height': 74.0,
                'weight': 230,
                'college': 'Texas Tech',
                'player_id': '00-0033873',
                'years_exp': 7,
                'headshot_url': 'https://example.com/mahomes.jpg',
                'ngs_position': 'QB',
                'week': 1,
                'game_type': 'REG',
                'status_description_abbr': 'ACT',
                'football_name': 'Patrick',
                'entry_year': 2017,
                'rookie_year': 2017.0,
                'draft_club': 'KC',
                'draft_number': 10.0,
                'age': 28.0
            },
            {
                'season': 2024,
                'team': 'KC',
                'position': 'TE',
                'depth_chart_position': 'TE',
                'jersey_number': 87.0,
                'status': 'ACT',
                'player_name': 'Travis Kelce',
                'first_name': 'Travis',
                'last_name': 'Kelce',
                'birth_date': pd.Timestamp('1989-10-05'),
                'height': 77.0,
                'weight': 250,
                'college': 'Cincinnati',
                'player_id': '00-0036355',
                'years_exp': 11,
                'headshot_url': 'https://example.com/kelce.jpg',
                'ngs_position': 'TE',
                'week': 1,
                'game_type': 'REG',
                'status_description_abbr': 'ACT',
                'football_name': 'Travis',
                'entry_year': 2013,
                'rookie_year': 2013.0,
                'draft_club': 'KC',
                'draft_number': 63.0,
                'age': 34.0
            }
        ])

        self.sample_transformed_roster_data = [
            {
                'slug': 'qb-patrick-mahomes-kc-15',
                'season': 2024,
                'team': 'KC',
                'position': 'QB',
                'depth_chart_position': 'QB',
                'jersey_number': 15,
                'status': 'ACT',
                'player_name': 'Patrick Mahomes',
                'first_name': 'Patrick',
                'last_name': 'Mahomes',
                'birth_date': '1995-09-17',
                'height': 74,
                'weight': 230,
                'college': 'Texas Tech',
                'player_id': '00-0033873',
                'years_exp': 7,
                'headshot_url': 'https://example.com/mahomes.jpg',
                'ngs_position': 'QB',
                'week': 1,
                'game_type': 'REG',
                'status_description_abbr': 'ACT',
                'football_name': 'Patrick',
                'entry_year': 2017,
                'rookie_year': 2017,
                'draft_club': 'KC',
                'draft_number': 10,
                'age': 28,
                'version': None
            },
            {
                'slug': 'te-travis-kelce-kc-87',
                'season': 2024,
                'team': 'KC',
                'position': 'TE',
                'depth_chart_position': 'TE',
                'jersey_number': 87,
                'status': 'ACT',
                'player_name': 'Travis Kelce',
                'first_name': 'Travis',
                'last_name': 'Kelce',
                'birth_date': '1989-10-05',
                'height': 77,
                'weight': 250,
                'college': 'Cincinnati',
                'player_id': '00-0036355',
                'years_exp': 11,
                'headshot_url': 'https://example.com/kelce.jpg',
                'ngs_position': 'TE',
                'week': 1,
                'game_type': 'REG',
                'status_description_abbr': 'ACT',
                'football_name': 'Travis',
                'entry_year': 2013,
                'rookie_year': 2013,
                'draft_club': 'KC',
                'draft_number': 63,
                'age': 34,
                'version': None
            }
        ]

    @patch('src.core.data.loaders.rosters.RostersDataLoader.__init__')
    def test_init_success(self, mock_init):
        """Test successful initialization of RostersDataLoader."""
        mock_init.return_value = None
        
        # Mock the initialization
        loader = RostersDataLoader()
        mock_init.assert_called_once()

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_seasonal_rosters')
    def test_fetch_raw_data_success(self, mock_import_rosters, mock_db_manager):
        """Test successful fetching of raw roster data."""
        mock_import_rosters.return_value = self.sample_raw_roster_data
        
        loader = RostersDataLoader()
        result = loader.fetch_raw_data([2024])
        
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 2)
        mock_import_rosters.assert_called_once_with([2024])

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_seasonal_rosters')
    def test_fetch_raw_data_empty(self, mock_import_rosters, mock_db_manager):
        """Test fetching when no roster data is available."""
        mock_import_rosters.return_value = pd.DataFrame()
        
        loader = RostersDataLoader()
        result = loader.fetch_raw_data([2024])
        
        self.assertTrue(result.empty)

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_seasonal_rosters')
    def test_fetch_raw_data_exception(self, mock_import_rosters, mock_db_manager):
        """Test handling of exceptions during data fetching."""
        mock_import_rosters.side_effect = Exception("API error")
        
        loader = RostersDataLoader()
        result = loader.fetch_raw_data([2024])
        
        self.assertTrue(result.empty)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_transform_rosters(self, mock_db_manager):
        """Test transformation of roster data."""
        loader = RostersDataLoader()
        
        # Mock the transform_data method to return expected data
        with patch.object(loader, 'transform_data') as mock_transform:
            mock_transform.return_value = self.sample_transformed_roster_data
            
            result = loader.transform_rosters(self.sample_raw_roster_data, version=1)
            
            self.assertEqual(len(result), 2)
            mock_transform.assert_called_once()

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_current_max_version(self, mock_db_manager):
        """Test getting current max version from database."""
        # Mock the database response
        mock_response = Mock()
        mock_response.data = [{'version': 3}]
        
        mock_db_manager.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        loader = RostersDataLoader()
        result = loader.get_current_max_version()
        
        self.assertEqual(result, 3)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_current_max_version_no_data(self, mock_db_manager):
        """Test getting max version when no data exists."""
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        mock_db_manager.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        loader = RostersDataLoader()
        result = loader.get_current_max_version()
        
        self.assertEqual(result, 0)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_upsert_rosters(self, mock_db_manager):
        """Test upserting roster records."""
        # Mock successful upsert
        mock_db_manager.return_value.upsert_records.return_value = {
            "success": True,
            "affected_rows": 2
        }
        
        loader = RostersDataLoader()
        result = loader.upsert_rosters(self.sample_transformed_roster_data, version=1)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["affected_rows"], 2)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_upsert_rosters_batch_processing(self, mock_db_manager):
        """Test batch processing during upsert."""
        # Mock successful upsert for batches
        mock_db_manager.return_value.upsert_records.return_value = {
            "success": True,
            "affected_rows": 1
        }
        
        loader = RostersDataLoader()
        
        # Test with batch size of 1 to force batching
        result = loader.upsert_rosters(self.sample_transformed_roster_data, batch_size=1)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["affected_rows"], 2)  # 2 batches of 1 each
        self.assertEqual(result["batch_count"], 2)

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_seasonal_rosters')
    def test_load_rosters_success(self, mock_import_rosters, mock_db_manager):
        """Test complete load_rosters workflow."""
        mock_import_rosters.return_value = self.sample_raw_roster_data
        
        # Mock successful load_data
        loader = RostersDataLoader()
        with patch.object(loader, 'load_data') as mock_load_data:
            mock_load_data.return_value = {
                "success": True,
                "total_fetched": 2,
                "total_validated": 2
            }
            
            with patch.object(loader, 'get_current_max_version', return_value=0):
                result = loader.load_rosters([2024])
                
                self.assertTrue(result["success"])
                self.assertEqual(result["version"], 1)  # Auto-generated

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_seasonal_rosters')
    def test_load_rosters_dry_run(self, mock_import_rosters, mock_db_manager):
        """Test load_rosters with dry run."""
        mock_import_rosters.return_value = self.sample_raw_roster_data
        
        loader = RostersDataLoader()
        with patch.object(loader, 'load_data') as mock_load_data:
            mock_load_data.return_value = {
                "success": True,
                "dry_run": True,
                "would_upsert": 2
            }
            
            result = loader.load_rosters([2024], dry_run=True)
            
            self.assertTrue(result["success"])
            self.assertTrue(result["dry_run"])

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_upsert_rosters_with_version_injection(self, mock_db_manager):
        """Test that version is properly injected into records."""
        # Mock successful upsert
        mock_db_manager.return_value.upsert_records.return_value = {
            "success": True,
            "affected_rows": 2
        }
        
        loader = RostersDataLoader()
        
        # Create records without version
        records_without_version = [
            {'slug': 'test-player-1', 'player_name': 'Test Player 1'},
            {'slug': 'test-player-2', 'player_name': 'Test Player 2'}
        ]
        
        result = loader.upsert_rosters(records_without_version, version=5)
        
        self.assertTrue(result["success"])
        
        # Verify that the version was added to the records
        for record in records_without_version:
            self.assertEqual(record['version'], 5)


if __name__ == '__main__':
    unittest.main()