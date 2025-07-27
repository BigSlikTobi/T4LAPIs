"""
Tests for the teams data loader module.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.core.data.loaders.teams import TeamsDataLoader


class TestTeamsDataLoader(unittest.TestCase):
    """Test cases for the TeamsDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
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

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_init_success(self, mock_get_client):
        """Test successful initialization of TeamsDataLoader."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        
        self.assertEqual(loader.table_name, "teams")
        self.assertEqual(loader.supabase, mock_client)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_init_no_client(self, mock_get_client):
        """Test initialization fails when no Supabase client available."""
        mock_get_client.return_value = None
        
        with self.assertRaises(RuntimeError) as context:
            TeamsDataLoader()
        
        self.assertIn("Could not initialize Supabase client", str(context.exception))

    @patch('src.core.data.loaders.teams.get_supabase_client')
    @patch('src.core.data.loaders.teams.fetch_team_data')
    @patch('src.core.data.loaders.teams.TeamDataTransformer')
    def test_fetch_and_transform_teams_success(self, mock_transformer_class, 
                                              mock_fetch, mock_get_client):
        """Test successful fetch and transform of team data."""
        mock_get_client.return_value = Mock()
        mock_fetch.return_value = Mock()  # Mock DataFrame
        
        # Mock the transformer instance and its transform method
        mock_transformer = Mock()
        mock_transformer.transform.return_value = self.sample_team_records
        mock_transformer_class.return_value = mock_transformer
        
        loader = TeamsDataLoader()
        result = loader.fetch_and_transform_teams()
        
        self.assertEqual(result, self.sample_team_records)
        mock_fetch.assert_called_once()
        mock_transformer_class.assert_called_once()
        mock_transformer.transform.assert_called_once()

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_get_existing_teams_count_success(self, mock_get_client):
        """Test successful retrieval of existing teams count."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = None
        mock_response.count = 32
        mock_client.table.return_value.select.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        count = loader.get_existing_teams_count()
        
        self.assertEqual(count, 32)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_get_existing_teams_count_error(self, mock_get_client):
        """Test handling of error when getting teams count."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = "Database error"
        mock_client.table.return_value.select.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        count = loader.get_existing_teams_count()
        
        self.assertEqual(count, 0)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_clear_teams_table_success(self, mock_get_client):
        """Test successful clearing of teams table."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = None
        mock_client.table.return_value.delete.return_value.neq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        result = loader.clear_teams_table()
        
        self.assertTrue(result)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_clear_teams_table_error(self, mock_get_client):
        """Test handling of error when clearing teams table."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = "Delete failed"
        mock_client.table.return_value.delete.return_value.neq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        result = loader.clear_teams_table()
        
        self.assertFalse(result)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_insert_teams_success(self, mock_get_client):
        """Test successful insertion of team records."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = None
        mock_response.data = self.sample_team_records
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        result = loader.insert_teams(self.sample_team_records)
        
        self.assertTrue(result)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_insert_teams_error(self, mock_get_client):
        """Test handling of error when inserting teams."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = "Insert failed"
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        result = loader.insert_teams(self.sample_team_records)
        
        self.assertFalse(result)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_load_teams_with_existing_data_no_clear(self, mock_get_client):
        """Test load_teams returns False when existing data and clear_existing=False."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        
        # Mock existing teams count > 0
        with patch.object(loader, 'get_existing_teams_count', return_value=32):
            result = loader.load_teams(clear_existing=False)
        
        self.assertFalse(result)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_load_teams_success_with_clear(self, mock_get_client):
        """Test successful load_teams with clear_existing=True."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        
        # Mock all methods to return success
        with patch.object(loader, 'get_existing_teams_count', return_value=32), \
             patch.object(loader, 'clear_teams_table', return_value=True), \
             patch.object(loader, 'fetch_and_transform_teams', return_value=self.sample_team_records), \
             patch.object(loader, 'insert_teams', return_value=True):
            
            result = loader.load_teams(clear_existing=True)
        
        self.assertTrue(result)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_load_teams_success_no_existing(self, mock_get_client):
        """Test successful load_teams with no existing data."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        
        # Mock all methods for fresh load
        with patch.object(loader, 'get_existing_teams_count', return_value=0), \
             patch.object(loader, 'fetch_and_transform_teams', return_value=self.sample_team_records), \
             patch.object(loader, 'insert_teams', return_value=True):
            
            result = loader.load_teams(clear_existing=False)
        
        self.assertTrue(result)

    @patch('src.core.data.loaders.teams.get_supabase_client')
    def test_load_teams_no_records_to_insert(self, mock_get_client):
        """Test load_teams fails when no records to insert."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        loader = TeamsDataLoader()
        
        # Mock fetch returning empty list
        with patch.object(loader, 'get_existing_teams_count', return_value=0), \
             patch.object(loader, 'fetch_and_transform_teams', return_value=[]):
            
            result = loader.load_teams()
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
