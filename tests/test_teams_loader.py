"""
Tests for the teams data loader module.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.core.data.loaders.teams import TeamsDataLoader


class TestTeamsDataLoader(unittest.TestCase):
    """Test cases for the TeamsDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Raw data as it would come from nfl_data_py
        self.sample_raw_team_data = pd.DataFrame([
            {
                'team_abbr': 'ARI',
                'team_name': 'Arizona Cardinals',
                'team_conference': 'NFC',
                'team_division': 'NFC West',
                'team_id': 3800,
                'team_nick': 'Cardinals',
                'team_color': '#97233F',
                'team_color2': '#000000',
                'team_color3': '#ffb612',
                'team_color4': '#a5acaf'
            },
            {
                'team_abbr': 'KC',
                'team_name': 'Kansas City Chiefs',
                'team_conference': 'AFC',
                'team_division': 'AFC West',
                'team_id': 2520,
                'team_nick': 'Chiefs',
                'team_color': '#E31837',
                'team_color2': '#FFB81C',
                'team_color3': '#000000',
                'team_color4': '#ffffff'
            }
        ])
        
        # Transformed data as it would be after transformation
        self.sample_team_records = [
            {
                'team_abbr': 'ARI',
                'team_name': 'Arizona Cardinals',
                'team_conference': 'NFC',
                'team_division': 'NFC West',
                'team_nfl_data_py_id': 3800,
                'team_nick': 'Cardinals',
                'team_color': '#97233F',
                'team_color2': '#000000',
                'team_color3': '#ffb612',
                'team_color4': '#a5acaf'
            },
            {
                'team_abbr': 'KC',
                'team_name': 'Kansas City Chiefs',
                'team_conference': 'AFC',
                'team_division': 'AFC West',
                'team_nfl_data_py_id': 2520,
                'team_nick': 'Chiefs',
                'team_color': '#E31837',
                'team_color2': '#FFB81C',
                'team_color3': '#000000',
                'team_color4': '#ffffff'
            }
        ]

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_init_success(self, mock_db_manager_class):
        """Test successful initialization of TeamsDataLoader."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TeamsDataLoader()
        
        self.assertEqual(loader.table_name, "teams")
        mock_db_manager_class.assert_called_once_with("teams")

    @patch('src.core.data.loaders.base.DatabaseManager')
    @patch('src.core.data.loaders.teams.fetch_team_data')
    def test_fetch_raw_data_success(self, mock_fetch, mock_db_manager_class):
        """Test successful fetch of team data."""
        mock_db_manager_class.return_value = Mock()
        mock_fetch.return_value = Mock()  # Mock DataFrame
        
        loader = TeamsDataLoader()
        result = loader.fetch_raw_data()
        
        mock_fetch.assert_called_once()
        self.assertEqual(result, mock_fetch.return_value)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_transformer_class_property(self, mock_db_manager_class):
        """Test that transformer_class property returns correct class."""
        mock_db_manager_class.return_value = Mock()
        
        loader = TeamsDataLoader()
        
        # Import the transformer class for comparison
        from src.core.data.transform import TeamDataTransformer
        self.assertEqual(loader.transformer_class, TeamDataTransformer)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_existing_teams_count_success(self, mock_db_manager_class):
        """Test successful retrieval of existing teams count."""
        mock_db_manager = Mock()
        mock_db_manager.get_record_count.return_value = 32
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TeamsDataLoader()
        count = loader.get_existing_teams_count()
        
        self.assertEqual(count, 32)
        mock_db_manager.get_record_count.assert_called_once_with(None)
    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_success(self, mock_db_manager_class):
        """Test successful load_data using new interface."""
        mock_db_manager = Mock()
        mock_db_manager.insert_records.return_value = {
            'success': True,
            'affected_rows': 2
        }
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TeamsDataLoader()
        
        # Mock the complete load_data flow with proper DataFrame
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_team_data):
            # Mock transformer
            from src.core.data.transform import TeamDataTransformer
            with patch.object(TeamDataTransformer, 'transform', return_value=self.sample_team_records):
                result = loader.load_data()
                
                self.assertTrue(result['success'])
                self.assertEqual(result['total_fetched'], 2)  # 2 records in DataFrame
                self.assertEqual(result['total_validated'], 2)  # 2 records
                self.assertEqual(result['upsert_result']['affected_rows'], 2)

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_dry_run(self, mock_db_manager_class):
        """Test load_data in dry run mode."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TeamsDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_team_data), \
             patch.object(loader.db_manager, 'upsert_records') as mock_insert:
            
            # Mock transformer
            from src.core.data.transform import TeamDataTransformer
            with patch.object(TeamDataTransformer, 'transform', return_value=self.sample_team_records):
                result = loader.load_data(dry_run=True)
                
                self.assertTrue(result['success'])
                self.assertTrue(result['dry_run'])
                mock_insert.assert_not_called()

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_with_clear(self, mock_db_manager_class):
        """Test load_data with clear_table=True."""
        mock_db_manager = Mock()
        mock_db_manager.clear_table.return_value = {'success': True}
        mock_db_manager.insert_records.return_value = {
            'success': True,
            'affected_rows': 2
        }
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TeamsDataLoader()
        
        with patch.object(loader, 'fetch_raw_data', return_value=self.sample_raw_team_data):
            
            # Mock transformer class to avoid initialization issues
            from src.core.data.transform import TeamDataTransformer
            with patch.object(TeamDataTransformer, '__init__', return_value=None), \
                 patch.object(TeamDataTransformer, 'transform', return_value=self.sample_team_records):
                result = loader.load_data(clear_table=True)
                
                self.assertTrue(result['success'])
                mock_db_manager.clear_table.assert_called_once_with("teams")


if __name__ == '__main__':
    unittest.main()
