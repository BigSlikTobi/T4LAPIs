"""Tests for the roster data transformer."""

import unittest
import pandas as pd

from src.core.data.transform import RosterDataTransformer


class TestRosterDataTransformer(unittest.TestCase):
    """Test cases for RosterDataTransformer."""

    def setUp(self) -> None:
        self.transformer = RosterDataTransformer()

    def test_transform_successful_record(self) -> None:
        """Transform a valid roster row into database payload."""
        df = pd.DataFrame(
            [
                {
                    "team": "KC",
                    "player_id": "00-0028830",
                    "player_name": "Travis Kelce",
                }
            ]
        )

        records = self.transformer.transform(df)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["team"], "KC")
        self.assertEqual(record["player"], "00-0028830")

    def test_transform_skips_unknown_team(self) -> None:
        """Skip records whose team cannot be mapped."""
        df = pd.DataFrame(
            [
                {
                    "team": "XXX",
                    "player_id": "00-0000000",
                    "player_name": "Unknown Player",
                }
            ]
        )

        records = self.transformer.transform(df)

        self.assertEqual(records, [])
        self.assertEqual(len(self.transformer.skipped_records), 1)
        self.assertEqual(self.transformer.skipped_records[0]["reason"], "invalid_team")

    def test_transform_deduplicates_team_player_pairs(self) -> None:
        """Ensure duplicate team/player pairs only appear once."""
        df = pd.DataFrame(
            [
                {
                    "team": "KC",
                    "player_id": "00-0011111",
                },
                {
                    "team": "KC",
                    "player_id": "00-0011111",
                },
            ]
        )

        records = self.transformer.transform(df)

        self.assertEqual(len(records), 1)
        duplicate_entries = [entry for entry in self.transformer.skipped_records if entry["reason"] == "duplicate_pair"]
        self.assertEqual(len(duplicate_entries), 1)


if __name__ == "__main__":
    unittest.main()
