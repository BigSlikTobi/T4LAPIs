"""Tests for the player weekly stats auto-update script."""

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
    import player_weekly_stats_auto_update
except ImportError:
    # Fallback for different test environments
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
    import player_weekly_stats_auto_update


class TestPlayerWeeklyStatsAutoUpdate(unittest.TestCase):
    """Test cases for player weekly stats auto-update script."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = Mock()
        self.mock_supabase = Mock()

    @patch('player_weekly_stats_auto_update.get_supabase_client')
    def test_get_latest_week_in_games_table_success(self, mock_get_client):
        """Test getting latest week from games table successfully."""
        # Mock successful database response
        mock_response = Mock()
        mock_response.data = [{'week': 8}]
        
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        self.mock_supabase.table.return_value = mock_table
        mock_get_client.return_value = self.mock_supabase
        
        result = player_weekly_stats_auto_update.get_latest_week_in_games_table(2024)
        
        self.assertEqual(result, 8)
        mock_get_client.assert_called_once()
        self.mock_supabase.table.assert_called_once_with("games")

    @patch('player_weekly_stats_auto_update.get_supabase_client')
    def test_get_latest_week_in_games_table_no_data(self, mock_get_client):
        """Test getting latest week from games table when no data exists."""
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        self.mock_supabase.table.return_value = mock_table
        mock_get_client.return_value = self.mock_supabase
        
        result = player_weekly_stats_auto_update.get_latest_week_in_games_table(2024)
        
        self.assertEqual(result, 0)

    @patch('player_weekly_stats_auto_update.get_supabase_client')
    def test_get_latest_week_in_stats_table_success(self, mock_get_client):
        """Test getting latest week from stats table successfully."""
        # Mock successful database response
        mock_response = Mock()
        mock_response.data = [{'week': 6}]
        
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        self.mock_supabase.table.return_value = mock_table
        mock_get_client.return_value = self.mock_supabase
        
        result = player_weekly_stats_auto_update.get_latest_week_in_stats_table(2024)
        
        self.assertEqual(result, 6)
        mock_get_client.assert_called_once()
        self.mock_supabase.table.assert_called_once_with("player_weekly_stats")

    @patch('player_weekly_stats_auto_update.get_supabase_client')
    def test_get_latest_week_in_stats_table_no_data(self, mock_get_client):
        """Test getting latest week from stats table when no data exists."""
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        mock_table = Mock()
        mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        self.mock_supabase.table.return_value = mock_table
        mock_get_client.return_value = self.mock_supabase
        
        result = player_weekly_stats_auto_update.get_latest_week_in_stats_table(2024)
        
        self.assertEqual(result, 0)

    @patch('player_weekly_stats_auto_update.get_supabase_client')
    def test_get_latest_week_connection_failure(self, mock_get_client):
        """Test getting latest week when database connection fails."""
        mock_get_client.return_value = None
        
        result = player_weekly_stats_auto_update.get_latest_week_in_games_table(2024)
        
        self.assertEqual(result, 0)

    @patch('player_weekly_stats_auto_update.get_supabase_client')
    def test_get_latest_week_exception(self, mock_get_client):
        """Test getting latest week when exception occurs."""
        mock_get_client.side_effect = Exception("Database error")
        
        result = player_weekly_stats_auto_update.get_latest_week_in_stats_table(2024)
        
        self.assertEqual(result, 0)

    @patch('player_weekly_stats_auto_update.datetime')
    def test_get_current_nfl_season_various_months(self, mock_datetime):
        """Test getting current NFL season for various months."""
        test_cases = [
            (2024, 9, 2024),   # September - current season starts
            (2024, 12, 2024),  # December - middle of season
            (2025, 2, 2024),   # February - still previous season
            (2024, 8, 2023),   # August - still previous season
        ]
        
        for year, month, expected_season in test_cases:
            with self.subTest(year=year, month=month):
                mock_now = Mock()
                mock_now.year = year
                mock_now.month = month
                mock_datetime.now.return_value = mock_now
                
                result = player_weekly_stats_auto_update.get_current_nfl_season()
                
                self.assertEqual(result, expected_season)

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.print_results')
    @patch('player_weekly_stats_auto_update.PlayerWeeklyStatsDataLoader')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_games_table')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_stats_table')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_no_games_data(self, mock_get_season, mock_get_stats_week, mock_get_games_week,
                                mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when no games data exists."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_games_week.return_value = 0  # No games data
        mock_get_stats_week.return_value = 0
        
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 1)  # 1 = failure, no games data
        mock_get_season.assert_called_once()
        mock_get_games_week.assert_called_once_with(2024)
        # Should not process any data
        mock_loader.load_data.assert_not_called()

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.print_results')
    @patch('player_weekly_stats_auto_update.PlayerWeeklyStatsDataLoader')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_games_table')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_stats_table')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_no_existing_stats_data(self, mock_get_season, mock_get_stats_week, mock_get_games_week,
                                         mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when no existing stats data."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_games_week.return_value = 8  # Games through week 8
        mock_get_stats_week.return_value = 0  # No stats yet
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 50}
        mock_loader.get_record_count.return_value = 400
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        mock_get_season.assert_called_once()
        mock_get_games_week.assert_called_once_with(2024)
        mock_get_stats_week.assert_called_once_with(2024)
        
        # Should process weeks 1-8
        expected_calls = [
            unittest.mock.call(years=[2024], weeks=[week], dry_run=False, clear_table=False)
            for week in range(1, 9)
        ]
        mock_loader.load_data.assert_has_calls(expected_calls)

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.print_results')
    @patch('player_weekly_stats_auto_update.PlayerWeeklyStatsDataLoader')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_games_table')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_stats_table')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_with_existing_stats_data(self, mock_get_season, mock_get_stats_week, mock_get_games_week,
                                           mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when existing stats data exists."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_games_week.return_value = 8  # Games through week 8
        mock_get_stats_week.return_value = 6  # Stats through week 6
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 25}
        mock_loader.get_record_count.return_value = 350
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        
        # Should process weeks 5-8 (week 5 for updates, 6-8 for new/updates)
        expected_calls = [
            unittest.mock.call(years=[2024], weeks=[week], dry_run=False, clear_table=False)
            for week in range(5, 9)
        ]
        mock_loader.load_data.assert_has_calls(expected_calls)

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.print_results')
    @patch('player_weekly_stats_auto_update.PlayerWeeklyStatsDataLoader')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_games_table')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_stats_table')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_stats_up_to_date(self, mock_get_season, mock_get_stats_week, mock_get_games_week,
                                   mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when stats are up to date."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_games_week.return_value = 8  # Games through week 8
        mock_get_stats_week.return_value = 8  # Stats also through week 8
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 10}
        mock_loader.get_record_count.return_value = 400
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        
        # Should process weeks 7-8 (last 2 weeks for updates)
        expected_calls = [
            unittest.mock.call(years=[2024], weeks=[7], dry_run=False, clear_table=False),
            unittest.mock.call(years=[2024], weeks=[8], dry_run=False, clear_table=False)
        ]
        mock_loader.load_data.assert_has_calls(expected_calls)

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.print_results')
    @patch('player_weekly_stats_auto_update.PlayerWeeklyStatsDataLoader')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_games_table')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_stats_table')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_skip_invalid_weeks(self, mock_get_season, mock_get_stats_week, mock_get_games_week,
                                     mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function skips weeks beyond 22."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_games_week.return_value = 25  # Invalid week (beyond 22)
        mock_get_stats_week.return_value = 22
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 5}
        mock_loader.get_record_count.return_value = 500
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        
        # Should only process weeks 21-22, skip weeks beyond 22
        expected_calls = [
            unittest.mock.call(years=[2024], weeks=[21], dry_run=False, clear_table=False),
            unittest.mock.call(years=[2024], weeks=[22], dry_run=False, clear_table=False)
        ]
        mock_loader.load_data.assert_has_calls(expected_calls)

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.print_results')
    @patch('player_weekly_stats_auto_update.PlayerWeeklyStatsDataLoader')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_games_table')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_stats_table')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_partial_failure(self, mock_get_season, mock_get_stats_week, mock_get_games_week,
                                  mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function when some operations fail."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_games_week.return_value = 8
        mock_get_stats_week.return_value = 6
        
        mock_loader = Mock()
        # Some calls succeed, some fail
        mock_loader.load_data.side_effect = [
            {"success": True, "affected_rows": 25},   # Week 5 succeeds
            {"success": True, "affected_rows": 30},   # Week 6 succeeds
            {"success": False, "error": "API error"}, # Week 7 fails
            {"success": True, "affected_rows": 20}    # Week 8 succeeds
        ]
        mock_loader.get_record_count.return_value = 350
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 1)  # 1 = failure due to partial failure

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_exception_handling(self, mock_get_season, mock_setup_logging):
        """Test main function handles exceptions gracefully."""
        # Setup mock to raise exception
        mock_get_season.side_effect = Exception("Unexpected error")
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 1)  # 1 = failure due to exception

    @patch('player_weekly_stats_auto_update.setup_cli_logging')
    @patch('player_weekly_stats_auto_update.print_results')
    @patch('player_weekly_stats_auto_update.PlayerWeeklyStatsDataLoader')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_games_table')
    @patch('player_weekly_stats_auto_update.get_latest_week_in_stats_table')
    @patch('player_weekly_stats_auto_update.get_current_nfl_season')
    def test_main_edge_case_week_1_stats(self, mock_get_season, mock_get_stats_week, mock_get_games_week,
                                         mock_loader_class, mock_print_results, mock_setup_logging):
        """Test main function edge case when stats exist for week 1."""
        # Setup mocks
        mock_get_season.return_value = 2024
        mock_get_games_week.return_value = 3  # Games through week 3
        mock_get_stats_week.return_value = 1  # Stats only for week 1
        
        mock_loader = Mock()
        mock_loader.load_data.return_value = {"success": True, "affected_rows": 15}
        mock_loader.get_record_count.return_value = 100
        mock_loader_class.return_value = mock_loader
        
        # Run main
        result = player_weekly_stats_auto_update.main()
        
        # Assertions (handle_cli_errors decorator returns 0 for success, 1 for failure)
        self.assertEqual(result, 0)  # 0 = success
        
        # Should process weeks 1-3 (start from max(1, 1-1) = 1)
        expected_calls = [
            unittest.mock.call(years=[2024], weeks=[week], dry_run=False, clear_table=False)
            for week in range(1, 4)
        ]
        mock_loader.load_data.assert_has_calls(expected_calls)


if __name__ == '__main__':
    unittest.main()
