"""Tests for the games auto-update script."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add scripts directory to path for testing
scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import the module we want to test
try:
    import games_auto_update
except ImportError:
    # Fallback for different test environments
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
    import games_auto_update


class TestGamesAutoUpdate(unittest.TestCase):
    """Test cases for games auto-update script."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = Mock()
        self.mock_supabase = Mock()

    @patch('games_auto_update.get_supabase_client')
    def test_get_latest_week_in_db_success(self, mock_get_client):
        """Test getting latest week from database successfully."""
        # Mock successful database response
        mock_response = Mock()
        mock_response.data = [{'week': 5}]
        
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        self.mock_supabase.table.return_value = mock_table
        mock_get_client.return_value = self.mock_supabase
        
        result = games_auto_update.get_latest_week_in_db(self.mock_loader, 2024)
        
        self.assertEqual(result, 5)
        mock_get_client.assert_called_once()
        self.mock_supabase.table.assert_called_once_with("games")

    @patch('games_auto_update.get_supabase_client')
    def test_get_latest_week_in_db_no_data(self, mock_get_client):
        """Test getting latest week when no data exists."""
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        self.mock_supabase.table.return_value = mock_table
        mock_get_client.return_value = self.mock_supabase
        
        result = games_auto_update.get_latest_week_in_db(self.mock_loader, 2024)
        
        self.assertEqual(result, 0)

    @patch('games_auto_update.get_supabase_client')
    def test_get_latest_week_in_db_connection_failure(self, mock_get_client):
        """Test getting latest week when database connection fails."""
        mock_get_client.return_value = None
        
        result = games_auto_update.get_latest_week_in_db(self.mock_loader, 2024)
        
        self.assertEqual(result, 0)

    @patch('games_auto_update.get_supabase_client')
    def test_get_latest_week_in_db_exception(self, mock_get_client):
        """Test getting latest week when exception occurs."""
        mock_get_client.side_effect = Exception("Database error")
        
        result = games_auto_update.get_latest_week_in_db(self.mock_loader, 2024)
        
        self.assertEqual(result, 0)

    @patch('games_auto_update.datetime')
    def test_get_current_nfl_season_september(self, mock_datetime):
        """Test getting current NFL season in September (start of season)."""
        mock_now = Mock()
        mock_now.year = 2024
        mock_now.month = 9
        mock_datetime.now.return_value = mock_now
        
        result = games_auto_update.get_current_nfl_season()
        
        self.assertEqual(result, 2024)

    @patch('games_auto_update.datetime')
    def test_get_current_nfl_season_december(self, mock_datetime):
        """Test getting current NFL season in December (middle of season)."""
        mock_now = Mock()
        mock_now.year = 2024
        mock_now.month = 12
        mock_datetime.now.return_value = mock_now
        
        result = games_auto_update.get_current_nfl_season()
        
        self.assertEqual(result, 2024)

    @patch('games_auto_update.datetime')
    def test_get_current_nfl_season_february(self, mock_datetime):
        """Test getting current NFL season in February (end of season)."""
        mock_now = Mock()
        mock_now.year = 2024
        mock_now.month = 2
        mock_datetime.now.return_value = mock_now
        
        result = games_auto_update.get_current_nfl_season()
        
        self.assertEqual(result, 2023)  # Previous year's season

    @patch('games_auto_update.datetime')
    def test_get_current_nfl_season_august(self, mock_datetime):
        """Test getting current NFL season in August (before season starts)."""
        mock_now = Mock()
        mock_now.year = 2024
        mock_now.month = 8
        mock_datetime.now.return_value = mock_now
        
        result = games_auto_update.get_current_nfl_season()
        
        self.assertEqual(result, 2023)  # Previous year's season

    @patch('games_auto_update.setup_cli_logging')
    @patch('games_auto_update.print_results')
    @patch('games_auto_update.GamesDataLoader')
    @patch('games_auto_update.get_latest_week_in_db')
    @patch('games_auto_update.get_current_nfl_season')
    def test_main_no_existing_data(self, mock_get_season, mock_get_latest_week, 
                                   mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when no existing data."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_latest_week.return_value = 0
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 10}
        mock_loader.get_game_count.return_value = 10
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = games_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        mock_get_season.assert_called_once()
        mock_get_latest_week.assert_called_once_with(mock_loader, 2024)
        
        # Should load week 1 only
        mock_loader.load_data.assert_called_once_with(
            season=2024,
            week=1,
            dry_run=False,
            clear_table=False
        )

    @patch('games_auto_update.setup_cli_logging')
    @patch('games_auto_update.print_results')
    @patch('games_auto_update.GamesDataLoader')
    @patch('games_auto_update.get_latest_week_in_db')
    @patch('games_auto_update.get_current_nfl_season')
    def test_main_with_existing_data(self, mock_get_season, mock_get_latest_week, 
                                     mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when existing data exists."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_latest_week.return_value = 5
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 10}
        mock_loader.get_game_count.return_value = 100
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = games_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        
        # Should load weeks 5 and 6
        expected_calls = [
            unittest.mock.call(season=2024, week=5, dry_run=False, clear_table=False),
            unittest.mock.call(season=2024, week=6, dry_run=False, clear_table=False)
        ]
        mock_loader.load_data.assert_has_calls(expected_calls)

    @patch('games_auto_update.setup_cli_logging')
    @patch('games_auto_update.print_results')
    @patch('games_auto_update.GamesDataLoader')
    @patch('games_auto_update.get_latest_week_in_db')
    @patch('games_auto_update.get_current_nfl_season')
    def test_main_skip_invalid_weeks(self, mock_get_season, mock_get_latest_week, 
                                     mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function skips weeks beyond 22."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_latest_week.return_value = 22  # At the end of season
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 10}
        mock_loader.get_game_count.return_value = 500
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = games_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        
        # Should only load week 22, skip week 23
        mock_loader.load_data.assert_called_once_with(
            season=2024,
            week=22,
            dry_run=False,
            clear_table=False
        )

    @patch('games_auto_update.setup_cli_logging')
    @patch('games_auto_update.print_results')
    @patch('games_auto_update.GamesDataLoader')
    @patch('games_auto_update.get_latest_week_in_db')
    @patch('games_auto_update.get_current_nfl_season')
    def test_main_partial_failure(self, mock_get_season, mock_get_latest_week, 
                                  mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when some operations fail."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_latest_week.return_value = 5
        
        mock_loader = Mock()
        # First call succeeds, second fails
        mock_loader.load_data.side_effect = [
            {"success": True, "affected_rows": 10},
            {"success": False, "error": "Network error"}
        ]
        mock_loader.get_game_count.return_value = 100
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = games_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 1)  # 1 = failure due to partial failure

    @patch('games_auto_update.setup_cli_logging')
    @patch('games_auto_update.get_current_nfl_season')
    def test_main_exception_handling(self, mock_get_season, mock_setup_logging):
        """Test main function handles exceptions gracefully."""
        # Setup mock to raise exception
        mock_get_season.side_effect = Exception("Unexpected error")
        
        # Run main
        result = games_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 1)  # 1 = failure due to exception


if __name__ == '__main__':
    unittest.main()
