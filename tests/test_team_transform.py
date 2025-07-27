"""
Tests for the teams data transformation module.
"""
import unittest
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.core.data.transform import TeamDataTransformer


class TestTeamTransform(unittest.TestCase):
    """Test cases for team data transformation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.transformer = TeamDataTransformer()
        
        self.sample_raw_data = pd.DataFrame([
            {
                'team_abbr': 'ARI',
                'team_name': 'Arizona Cardinals',
                'team_id': 3800,
                'team_nick': 'Cardinals',
                'team_conf': 'NFC',
                'team_division': 'NFC West',
                'team_color': '#97233F',
                'team_color2': '#000000',
                'team_color3': '#ffb612',
                'team_color4': '#a5acaf'
            },
            {
                'team_abbr': 'KC',
                'team_name': 'Kansas City Chiefs',
                'team_id': 2520,
                'team_nick': 'Chiefs',
                'team_conf': 'AFC',
                'team_division': 'AFC West',
                'team_color': '#E31837',
                'team_color2': '#FFB81C',
                'team_color3': '#000000',
                'team_color4': '#ffffff'
            }
        ])
        
        self.expected_transformed = [
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

    def test_transform_team_data_success(self):
        """Test successful transformation of team data."""
        result = self.transformer.transform(self.sample_raw_data)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result, self.expected_transformed)

    def test_transform_team_data_missing_columns(self):
        """Test transformation fails with missing required columns."""
        incomplete_data = self.sample_raw_data.drop(columns=['team_name'])
        
        with self.assertRaises(ValueError) as context:
            self.transformer.transform(incomplete_data)
        
        self.assertIn("Missing required columns", str(context.exception))

    def test_transform_team_data_with_nulls(self):
        """Test transformation handles null values correctly."""
        data_with_nulls = self.sample_raw_data.copy()
        data_with_nulls.loc[0, 'team_name'] = None
        data_with_nulls.loc[1, 'team_abbr'] = ''
        
        result = self.transformer.transform(data_with_nulls)
        
        # Should skip records with null/empty critical fields
        self.assertEqual(len(result), 0)

    def test_validate_team_record_valid(self):
        """Test validation of valid team record."""
        valid_record = self.expected_transformed[0]
        
        self.assertTrue(self.transformer._validate_record(valid_record))

    def test_validate_team_record_missing_required_field(self):
        """Test validation fails for missing required fields."""
        invalid_record = self.expected_transformed[0].copy()
        del invalid_record['team_abbr']
        
        self.assertFalse(self.transformer._validate_record(invalid_record))

    def test_validate_team_record_empty_required_field(self):
        """Test validation fails for empty required fields."""
        invalid_record = self.expected_transformed[0].copy()
        invalid_record['team_name'] = ''
        
        self.assertFalse(self.transformer._validate_record(invalid_record))

    def test_validate_team_record_invalid_abbr_length(self):
        """Test validation fails for invalid team abbreviation length."""
        invalid_record = self.expected_transformed[0].copy()
        invalid_record['team_abbr'] = 'TOOLONG'
        
        self.assertFalse(self.transformer._validate_record(invalid_record))

    def test_validate_team_record_invalid_conference(self):
        """Test validation fails for invalid conference."""
        invalid_record = self.expected_transformed[0].copy()
        invalid_record['team_conference'] = 'INVALID'
        
        self.assertFalse(self.transformer._validate_record(invalid_record))

    def test_validate_team_record_invalid_id(self):
        """Test validation fails for invalid team ID."""
        invalid_record = self.expected_transformed[0].copy()
        invalid_record['team_nfl_data_py_id'] = -1
        
        self.assertFalse(self.transformer._validate_record(invalid_record))

    def test_sanitize_team_data_valid_records(self):
        """Test sanitization keeps valid records."""
        result = self.transformer._validate_and_sanitize_records(self.expected_transformed)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result, self.expected_transformed)

    def test_sanitize_team_data_filters_invalid(self):
        """Test sanitization filters out invalid records."""
        mixed_data = self.expected_transformed.copy()
        mixed_data.append({
            'team_abbr': '',  # Invalid - empty
            'team_name': 'Invalid Team',
            'team_conference': 'NFC',
            'team_division': 'NFC West',
            'team_nfl_data_py_id': 1234,
            'team_nick': 'Invalid'
        })
        
        result = self.transformer._validate_and_sanitize_records(mixed_data)
        
        # Should filter out the invalid record
        self.assertEqual(len(result), 2)
        self.assertEqual(result, self.expected_transformed)

    def test_sanitize_team_data_handles_null_colors(self):
        """Test sanitization handles null color values."""
        data_with_null_colors = self.expected_transformed.copy()
        data_with_null_colors[0]['team_color'] = None
        data_with_null_colors[0]['team_color2'] = pd.NA
        
        result = self.transformer._validate_and_sanitize_records(data_with_null_colors)
        
        self.assertEqual(len(result), 2)
        # Should replace null colors with default
        self.assertEqual(result[0]['team_color'], '#000000')
        self.assertEqual(result[0]['team_color2'], '#000000')


if __name__ == '__main__':
    unittest.main()
