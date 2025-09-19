"""Tests for the rosters data loader."""

import unittest
from unittest.mock import Mock, patch
import pandas as pd

from src.core.data.loaders.rosters import RostersDataLoader


class TestRostersDataLoader(unittest.TestCase):
    """Test suite for RostersDataLoader."""

    def setUp(self) -> None:
        self.sample_df = pd.DataFrame(
            [
                {
                    "team": "LAR",
                    "player_id": "00-0036654",
                    "player_name": "Puka Nacua",
                }
            ]
        )

    @patch("src.core.data.loaders.base.DatabaseManager")
    def test_load_data_success(self, mock_db_manager_class) -> None:
        """Successfully load roster data into Supabase."""
        mock_db_manager = Mock()
        mock_db_manager.upsert_records.return_value = {"success": True, "affected_rows": 1}
        mock_db_manager.clear_table.return_value = True
        mock_db_manager_class.return_value = mock_db_manager

        loader = RostersDataLoader()

        with patch.object(loader, "fetch_raw_data", return_value=self.sample_df):
            result = loader.load_data(season=2025)

        self.assertTrue(result["success"])
        self.assertEqual(result["total_validated"], 1)
        mock_db_manager.upsert_records.assert_called_once()
        args, kwargs = mock_db_manager.upsert_records.call_args
        self.assertEqual(kwargs.get("on_conflict"), "team,player")
        payload = args[0]
        self.assertEqual(payload[0]["team"], "LA")
        self.assertEqual(payload[0]["player"], "00-0036654")

    @patch("src.core.data.loaders.base.DatabaseManager")
    def test_load_data_dry_run(self, mock_db_manager_class) -> None:
        """Dry run should not upsert data but report counts."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager

        loader = RostersDataLoader()

        with patch.object(loader, "fetch_raw_data", return_value=self.sample_df):
            result = loader.load_data(season=2025, dry_run=True, clear_table=True, include_records=True)

        self.assertTrue(result["success"])
        self.assertTrue(result["dry_run"])
        self.assertTrue(result["would_clear"])
        self.assertEqual(result["would_upsert"], 1)
        self.assertIn("records", result)
        mock_db_manager.upsert_records.assert_not_called()
        mock_db_manager.clear_table.assert_not_called()

    @patch("src.core.data.loaders.base.DatabaseManager")
    def test_load_data_no_records(self, mock_db_manager_class) -> None:
        """Return failure when transformation yields no records."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager

        loader = RostersDataLoader()

        empty_df = pd.DataFrame(
            [
                {
                    "player_name": "Unknown",
                    "team": "XXX",
                    "player_id": "00-0000000",
                }
            ]
        )

        with patch.object(loader, "fetch_raw_data", return_value=empty_df):
            result = loader.load_data(season=2025)

        self.assertFalse(result["success"])
        self.assertIn("message", result)
        mock_db_manager.upsert_records.assert_not_called()


if __name__ == "__main__":
    unittest.main()
