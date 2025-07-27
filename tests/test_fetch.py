"""
Tests for the NFL data fetching module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.data.fetch import (
    fetch_team_data,
    fetch_player_data,
    fetch_game_schedule_data,
    fetch_weekly_stats_data,
    fetch_seasonal_roster_data,
    fetch_pbp_data,
    fetch_ngs_data,
    fetch_injury_data,
    fetch_combine_data,
    fetch_draft_data,
)


class TestNFLDataFetch(unittest.TestCase):
    """Test cases for NFL data fetching functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_years = [2023, 2024]
        self.sample_df = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })

    @patch('src.core.data.fetch.nfl.import_team_desc')
    def test_fetch_team_data_success(self, mock_import_team_desc):
        """Test successful fetching of team data."""
        mock_import_team_desc.return_value = self.sample_df
        
        result = fetch_team_data()
        
        mock_import_team_desc.assert_called_once()
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_team_desc')
    def test_fetch_team_data_failure(self, mock_import_team_desc):
        """Test handling of team data fetch failure."""
        mock_import_team_desc.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception) as context:
            fetch_team_data()
        
        self.assertIn("API Error", str(context.exception))

    @patch('src.core.data.fetch.nfl.import_seasonal_rosters')
    def test_fetch_player_data_success(self, mock_import_seasonal_rosters):
        """Test successful fetching of player data."""
        mock_import_seasonal_rosters.return_value = self.sample_df
        
        result = fetch_player_data(self.sample_years)
        
        mock_import_seasonal_rosters.assert_called_once_with(self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_seasonal_rosters')
    def test_fetch_player_data_failure(self, mock_import_seasonal_rosters):
        """Test handling of player data fetch failure."""
        mock_import_seasonal_rosters.side_effect = Exception("Connection Error")
        
        with self.assertRaises(Exception) as context:
            fetch_player_data(self.sample_years)
        
        self.assertIn("Connection Error", str(context.exception))

    @patch('src.core.data.fetch.nfl.import_schedules')
    def test_fetch_game_schedule_data_success(self, mock_import_schedules):
        """Test successful fetching of game schedule data."""
        mock_import_schedules.return_value = self.sample_df
        
        result = fetch_game_schedule_data(self.sample_years)
        
        mock_import_schedules.assert_called_once_with(self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_schedules')
    def test_fetch_game_schedule_data_failure(self, mock_import_schedules):
        """Test handling of game schedule data fetch failure."""
        mock_import_schedules.side_effect = Exception("Data Error")
        
        with self.assertRaises(Exception) as context:
            fetch_game_schedule_data(self.sample_years)
        
        self.assertIn("Data Error", str(context.exception))

    @patch('src.core.data.fetch.nfl.import_weekly_data')
    def test_fetch_weekly_stats_data_success(self, mock_import_weekly_data):
        """Test successful fetching of weekly stats data."""
        mock_import_weekly_data.return_value = self.sample_df
        
        result = fetch_weekly_stats_data(self.sample_years)
        
        mock_import_weekly_data.assert_called_once_with(years=self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_weekly_data')
    def test_fetch_weekly_stats_data_failure(self, mock_import_weekly_data):
        """Test handling of weekly stats data fetch failure."""
        mock_import_weekly_data.side_effect = Exception("Network Error")
        
        with self.assertRaises(Exception) as context:
            fetch_weekly_stats_data(self.sample_years)
        
        self.assertIn("Network Error", str(context.exception))

    @patch('src.core.data.fetch.nfl.import_seasonal_rosters')
    def test_fetch_seasonal_roster_data_success(self, mock_import_seasonal_rosters):
        """Test successful fetching of seasonal roster data."""
        mock_import_seasonal_rosters.return_value = self.sample_df
        
        result = fetch_seasonal_roster_data(self.sample_years)
        
        mock_import_seasonal_rosters.assert_called_once_with(self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_pbp_data')
    def test_fetch_pbp_data_success(self, mock_import_pbp_data):
        """Test successful fetching of play-by-play data."""
        mock_import_pbp_data.return_value = self.sample_df
        
        result = fetch_pbp_data(self.sample_years, downsampling=True)
        
        mock_import_pbp_data.assert_called_once_with(self.sample_years, downsampling=True)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_ngs_data')
    def test_fetch_ngs_data_success(self, mock_import_ngs_data):
        """Test successful fetching of NGS data."""
        mock_import_ngs_data.return_value = self.sample_df
        
        result = fetch_ngs_data('passing', self.sample_years)
        
        mock_import_ngs_data.assert_called_once_with('passing', self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_injuries')
    def test_fetch_injury_data_success(self, mock_import_injuries):
        """Test successful fetching of injury data."""
        mock_import_injuries.return_value = self.sample_df
        
        result = fetch_injury_data(self.sample_years)
        
        mock_import_injuries.assert_called_once_with(self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_combine_data')
    def test_fetch_combine_data_success_with_years(self, mock_import_combine_data):
        """Test successful fetching of combine data with specific years."""
        mock_import_combine_data.return_value = self.sample_df
        
        result = fetch_combine_data(self.sample_years)
        
        mock_import_combine_data.assert_called_once_with(self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_combine_data')
    def test_fetch_combine_data_success_all_years(self, mock_import_combine_data):
        """Test successful fetching of combine data for all years."""
        mock_import_combine_data.return_value = self.sample_df
        
        result = fetch_combine_data(None)
        
        mock_import_combine_data.assert_called_once_with()
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_draft_picks')
    def test_fetch_draft_data_success_with_years(self, mock_import_draft_picks):
        """Test successful fetching of draft data with specific years."""
        mock_import_draft_picks.return_value = self.sample_df
        
        result = fetch_draft_data(self.sample_years)
        
        mock_import_draft_picks.assert_called_once_with(self.sample_years)
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('src.core.data.fetch.nfl.import_draft_picks')
    def test_fetch_draft_data_success_all_years(self, mock_import_draft_picks):
        """Test successful fetching of draft data for all years."""
        mock_import_draft_picks.return_value = self.sample_df
        
        result = fetch_draft_data(None)
        
        mock_import_draft_picks.assert_called_once_with()
        pd.testing.assert_frame_equal(result, self.sample_df)


if __name__ == '__main__':
    unittest.main()
