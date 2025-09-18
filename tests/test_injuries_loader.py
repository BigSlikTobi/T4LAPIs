"""Tests for the injuries data loader."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
from src.core.data.loaders.injuries import InjuriesDataLoader


class TestInjuriesDataLoader(unittest.TestCase):
    """Test cases for InjuriesDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_raw_injury_data = pd.DataFrame([
            {
                'season': 2024,
                'game_type': 'REG',
                'team': 'KC',
                'week': 1,
                'gsis_id': '00-0033873',
                'position': 'QB',
                'full_name': 'Patrick Mahomes',
                'first_name': 'Patrick',
                'last_name': 'Mahomes',
                'report_primary_injury': 'Ankle',
                'report_secondary_injury': None,
                'report_status': 'Questionable',
                'practice_primary_injury': 'Ankle',
                'practice_secondary_injury': None,
                'practice_status': 'Limited',
                'date_modified': pd.Timestamp('2024-09-06 19:05:30')
            },
            {
                'season': 2024,
                'game_type': 'REG',
                'team': 'KC',
                'week': 2,
                'gsis_id': '00-0036355',
                'position': 'TE',
                'full_name': 'Travis Kelce',
                'first_name': 'Travis',
                'last_name': 'Kelce',
                'report_primary_injury': 'Knee',
                'report_secondary_injury': None,
                'report_status': 'Out',
                'practice_primary_injury': 'Knee',
                'practice_secondary_injury': None,
                'practice_status': 'Did Not Participate',
                'date_modified': pd.Timestamp('2024-09-13 18:30:00')
            }
        ])

        self.sample_transformed_injury_data = [
            {
                'player_id': '00-0033873',
                'season': 2024,
                'week': 1,
                'team': 'KC',
                'position': 'QB',
                'player_name': 'Patrick Mahomes',
                'report_primary_injury': 'Ankle',
                'report_secondary_injury': None,
                'report_status': 'Questionable',
                'practice_primary_injury': 'Ankle',
                'practice_secondary_injury': None,
                'practice_status': 'Limited',
                'game_type': 'REG',
                'date_modified': '2024-09-06T19:05:30',
                'version': None
            },
            {
                'player_id': '00-0036355',
                'season': 2024,
                'week': 2,
                'team': 'KC',
                'position': 'TE',
                'player_name': 'Travis Kelce',
                'report_primary_injury': 'Knee',
                'report_secondary_injury': None,
                'report_status': 'Out',
                'practice_primary_injury': 'Knee',
                'practice_secondary_injury': None,
                'practice_status': 'Did Not Participate',
                'game_type': 'REG',
                'date_modified': '2024-09-13T18:30:00',
                'version': None
            }
        ]

    @patch('src.core.data.loaders.injuries.InjuriesDataLoader.__init__')
    def test_init_success(self, mock_init):
        """Test successful initialization of InjuriesDataLoader."""
        mock_init.return_value = None
        
        # Mock the initialization
        loader = InjuriesDataLoader()
        mock_init.assert_called_once()

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_injuries')
    def test_fetch_raw_data_success(self, mock_import_injuries, mock_db_manager):
        """Test successful fetching of raw injury data."""
        mock_import_injuries.return_value = self.sample_raw_injury_data
        
        loader = InjuriesDataLoader()
        result = loader.fetch_raw_data([2024])
        
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 2)
        mock_import_injuries.assert_called_once_with([2024])

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_injuries')
    def test_fetch_raw_data_empty(self, mock_import_injuries, mock_db_manager):
        """Test fetching when no injury data is available."""
        mock_import_injuries.return_value = pd.DataFrame()
        
        loader = InjuriesDataLoader()
        result = loader.fetch_raw_data([2024])
        
        self.assertTrue(result.empty)

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_injuries')
    def test_fetch_raw_data_exception(self, mock_import_injuries, mock_db_manager):
        """Test handling of exceptions during data fetching."""
        mock_import_injuries.side_effect = Exception("API error")
        
        loader = InjuriesDataLoader()
        result = loader.fetch_raw_data([2024])
        
        self.assertTrue(result.empty)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_transform_injuries(self, mock_db_manager):
        """Test transformation of injury data."""
        loader = InjuriesDataLoader()
        
        # Mock the transform_data method to return expected data
        with patch.object(loader, 'transform_data') as mock_transform:
            mock_transform.return_value = self.sample_transformed_injury_data
            
            result = loader.transform_injuries(self.sample_raw_injury_data, version=1)
            
            self.assertEqual(len(result), 2)
            mock_transform.assert_called_once()

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_current_max_version(self, mock_db_manager):
        """Test getting current max version from database."""
        # Mock the database response
        mock_response = Mock()
        mock_response.data = [{'version': 5}]
        
        mock_db_manager.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        loader = InjuriesDataLoader()
        result = loader.get_current_max_version()
        
        self.assertEqual(result, 5)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_current_max_version_no_data(self, mock_db_manager):
        """Test getting max version when no data exists."""
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        mock_db_manager.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        loader = InjuriesDataLoader()
        result = loader.get_current_max_version()
        
        self.assertEqual(result, 0)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_upsert_injuries(self, mock_db_manager):
        """Test upserting injury records."""
        # Mock successful upsert
        mock_db_manager.return_value.upsert_records.return_value = {
            "success": True,
            "affected_rows": 2
        }
        
        loader = InjuriesDataLoader()
        result = loader.upsert_injuries(self.sample_transformed_injury_data, version=1)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["affected_rows"], 2)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_upsert_injuries_batch_processing(self, mock_db_manager):
        """Test batch processing during upsert."""
        # Mock successful upsert for batches
        mock_db_manager.return_value.upsert_records.return_value = {
            "success": True,
            "affected_rows": 1
        }
        
        loader = InjuriesDataLoader()
        
        # Test with batch size of 1 to force batching
        result = loader.upsert_injuries(self.sample_transformed_injury_data, batch_size=1)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["affected_rows"], 2)  # 2 batches of 1 each
        self.assertEqual(result["batch_count"], 2)

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_injuries')
    def test_load_injuries_success(self, mock_import_injuries, mock_db_manager):
        """Test complete load_injuries workflow."""
        mock_import_injuries.return_value = self.sample_raw_injury_data
        
        # Mock successful load_data
        loader = InjuriesDataLoader()
        with patch.object(loader, 'load_data') as mock_load_data:
            mock_load_data.return_value = {
                "success": True,
                "total_fetched": 2,
                "total_validated": 2
            }
            
            with patch.object(loader, 'get_current_max_version', return_value=0):
                result = loader.load_injuries([2024])
                
                self.assertTrue(result["success"])
                self.assertEqual(result["version"], 1)  # Auto-generated

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('nfl_data_py.import_injuries')
    def test_load_injuries_dry_run(self, mock_import_injuries, mock_db_manager):
        """Test load_injuries with dry run."""
        mock_import_injuries.return_value = self.sample_raw_injury_data
        
        loader = InjuriesDataLoader()
        with patch.object(loader, 'load_data') as mock_load_data:
            mock_load_data.return_value = {
                "success": True,
                "dry_run": True,
                "would_upsert": 2
            }
            
            result = loader.load_injuries([2024], dry_run=True)
            
            self.assertTrue(result["success"])
            self.assertTrue(result["dry_run"])


if __name__ == '__main__':
    unittest.main()