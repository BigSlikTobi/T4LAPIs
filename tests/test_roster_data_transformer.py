"""Tests for the roster data transformer."""

import unittest
import pandas as pd

from src.core.data.transform import RosterDataTransformer


class TestRosterDataTransformer(unittest.TestCase):
    """Test cases for RosterDataTransformer."""

    def setUp(self) -> None:
        self.team_mapping = {"KC": 101, "LAR": 202}
        self.transformer = RosterDataTransformer(self.team_mapping, version=3)

    def test_transform_successful_record(self) -> None:
        """Transform a valid roster row into database payload."""
        df = pd.DataFrame(
            [
                {
                    "player_name": "Travis Kelce",
                    "position": "TE",
                    "team": "KC",
                    "jersey_number": 87,
                    "status": "Active",
                    "height": "77",
                    "weight": 250,
                    "college": "Cincinnati",
                    "headshot_url": "http://example.com/headshot.jpg",
                    "age": 34,
                    "years_exp": 11,
                }
            ]
        )

        records = self.transformer.transform(df)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["name"], "Travis Kelce")
        self.assertEqual(record["position"], "TE")
        self.assertEqual(record["teamId"], 101)
        self.assertEqual(record["version"], 3)
        self.assertEqual(record["number"], 87)
        self.assertTrue(record["slug"].startswith("te-travis-kelce-kc"))

    def test_transform_skips_unknown_team(self) -> None:
        """Skip records whose team cannot be mapped."""
        df = pd.DataFrame(
            [
                {
                    "player_name": "Unknown Player",
                    "position": "QB",
                    "team": "XXX",
                }
            ]
        )

        records = self.transformer.transform(df)

        self.assertEqual(records, [])
        self.assertEqual(len(self.transformer.skipped_records), 1)
        self.assertEqual(self.transformer.skipped_records[0]["reason"], "team_fk_not_found")

    def test_transform_deduplicates_slug(self) -> None:
        """Ensure duplicate slugs keep the richer record and note the discard."""
        df = pd.DataFrame(
            [
                {
                    "player_name": "Duplicate Player",
                    "position": "WR",
                    "team": "KC",
                    "jersey_number": 11,
                    "status": "Active",
                },
                {
                    "player_name": "Duplicate Player",
                    "position": "WR",
                    "team": "KC",
                    "jersey_number": 11,
                    "status": None,
                    "college": "Kansas State",
                    "weight": 210,
                },
            ]
        )

        records = self.transformer.transform(df)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["college"], "Kansas State")
        self.assertEqual(record["weight"], 210)
        duplicate_entries = [entry for entry in self.transformer.skipped_records if entry["reason"] == "duplicate_slug"]
        self.assertEqual(len(duplicate_entries), 1)


if __name__ == "__main__":
    unittest.main()
