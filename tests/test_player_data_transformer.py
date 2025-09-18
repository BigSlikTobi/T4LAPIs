"""Tests for PlayerDataTransformer specific transformations."""

import unittest
import pandas as pd

from src.core.data.transform import PlayerDataTransformer


class TestPlayerDataTransformer(unittest.TestCase):
    """Test cases for PlayerDataTransformer transformations."""

    def test_transform_includes_sleeper_id(self):
        """Sleeper ID from the source data is preserved in the transformed record."""
        df = pd.DataFrame([
            {
                'player_id': '00-1234567',
                'player_name': 'Test Player',
                'first_name': 'Test',
                'last_name': 'Player',
                'birth_date': '1990-01-01',
                'height': 74,
                'weight': 210,
                'college': 'Test University',
                'position': 'QB',
                'rookie_year': 2015,
                'season': 2024,
                'football_name': 'T. Player',
                'jersey_number': 12,
                'status': 'Active',
                'team': 'KC',
                'years_exp': 10,
                'headshot_url': None,
                'sleeper_id': 123456.0,
                'draft_number': 10,
                'draft_club': 'KC'
            }
        ])

        transformer = PlayerDataTransformer()
        transformed_records = transformer.transform(df)

        self.assertEqual(len(transformed_records), 1)
        record = transformed_records[0]
        self.assertIn('sleeper_id', record)
        self.assertEqual(record['sleeper_id'], '123456')


if __name__ == '__main__':
    unittest.main()
