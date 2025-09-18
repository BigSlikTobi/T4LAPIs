"""Test module for InjuriesDataLoader."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.data.loaders.injuries import InjuriesDataLoader
from src.core.data.transform import InjuryDataTransformer


class TestInjuriesDataLoader:
    """Test cases for InjuriesDataLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.core.data.loaders.base.DatabaseManager') as mock_db_manager:
            # Mock the database manager to avoid Supabase connection issues in tests
            mock_db_manager.return_value = Mock()
            self.loader = InjuriesDataLoader()
            self.loader.db_manager = Mock()  # Ensure it's properly mocked
        
    def test_loader_initialization(self):
        """Test that the loader initializes correctly."""
        assert self.loader.table_name == "injuries"
        assert self.loader.transformer_class == InjuryDataTransformer
        
    @patch('src.core.data.loaders.injuries.fetch_injury_data')
    def test_fetch_raw_data(self, mock_fetch):
        """Test fetching raw injury data."""
        # Mock data
        mock_df = pd.DataFrame({
            'season': [2024],
            'team': ['SEA'],
            'week': [1],
            'gsis_id': ['00-0012345'],
            'position': ['QB'],
            'full_name': ['Test Player'],
            'report_status': ['Out'],
            'date_modified': ['2024-01-01 10:00:00']
        })
        mock_fetch.return_value = mock_df
        
        # Test
        result = self.loader.fetch_raw_data([2024])
        
        # Assertions
        mock_fetch.assert_called_once_with([2024])
        assert len(result) == 1
        assert result.iloc[0]['full_name'] == 'Test Player'
        
    @patch('src.core.data.loaders.injuries.fetch_injury_data')
    def test_load_injuries_success(self, mock_fetch):
        """Test successful injury data loading."""
        # Mock data
        mock_df = pd.DataFrame({
            'season': [2024, 2024],
            'team': ['SEA', 'KC'],
            'week': [1, 1],
            'gsis_id': ['00-0012345', '00-0012346'],
            'position': ['QB', 'RB'],
            'full_name': ['Test Player 1', 'Test Player 2'],
            'report_status': ['Out', 'Questionable'],
            'date_modified': ['2024-01-01 10:00:00', '2024-01-01 11:00:00']
        })
        mock_fetch.return_value = mock_df
        
        # Mock database manager
        mock_result = Mock()
        mock_result.inserted_count = 2
        mock_result.updated_count = 0
        mock_result.errors_count = 0
        self.loader.db_manager.upsert_data = Mock(return_value=mock_result)
        
        # Test
        result = self.loader.load_injuries(years=[2024], version=1)
        
        # Assertions
        assert result["success"] is True
        assert result["inserted"] == 2
        assert result["updated"] == 0
        assert result["errors"] == 0
        assert result["records_processed"] == 2
        
    @patch('src.core.data.loaders.injuries.fetch_injury_data')
    def test_load_injuries_dry_run(self, mock_fetch):
        """Test dry run mode."""
        # Mock data
        mock_df = pd.DataFrame({
            'season': [2024],
            'team': ['SEA'],
            'week': [1],
            'gsis_id': ['00-0012345'],
            'position': ['QB'],
            'full_name': ['Test Player'],
            'report_status': ['Out'],
            'date_modified': ['2024-01-01 10:00:00']
        })
        mock_fetch.return_value = mock_df
        
        # Test
        result = self.loader.load_injuries(years=[2024], dry_run=True)
        
        # Assertions
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["records_processed"] == 1
        
    @patch('src.core.data.loaders.injuries.fetch_injury_data')
    def test_load_injuries_no_data(self, mock_fetch):
        """Test loading when no data is available."""
        # Mock empty data
        mock_fetch.return_value = pd.DataFrame()
        
        # Test
        result = self.loader.load_injuries(years=[2024])
        
        # Assertions
        assert result["success"] is False
        assert "No data found" in result["error"]
        
    def test_get_next_version_with_data(self):
        """Test getting next version when data exists."""
        # Mock database response
        mock_response = Mock()
        mock_response.data = [{'version': 5}]
        self.loader.db_manager.client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        # Test
        version = self.loader.get_next_version()
        
        # Assertions
        assert version == 6
        
    def test_get_next_version_no_data(self):
        """Test getting next version when no data exists."""
        # Mock database response
        mock_response = Mock()
        mock_response.data = []
        self.loader.db_manager.client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        # Test
        version = self.loader.get_next_version()
        
        # Assertions
        assert version == 1
        
    def test_get_next_version_error(self):
        """Test getting next version when database error occurs."""
        # Mock database error
        self.loader.db_manager.client.table.side_effect = Exception("Database error")
        
        # Test
        version = self.loader.get_next_version()
        
        # Assertions
        assert version == 1  # Should return default


class TestInjuryDataTransformer:
    """Test cases for InjuryDataTransformer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = InjuryDataTransformer()
        
    def test_required_columns(self):
        """Test that required columns are defined correctly."""
        required = self.transformer._get_required_columns()
        expected = [
            'season', 'team', 'week', 'gsis_id', 'position', 'full_name',
            'report_status', 'date_modified'
        ]
        assert required == expected
        
    def test_transform_single_record_success(self):
        """Test successful transformation of a single record."""
        # Sample row
        row = pd.Series({
            'season': 2024,
            'game_type': 'REG',
            'team': 'SEA',
            'week': 1,
            'gsis_id': '00-0012345',
            'position': 'QB',
            'full_name': 'Test Player',
            'first_name': 'Test',
            'last_name': 'Player',
            'report_primary_injury': 'Knee',
            'report_secondary_injury': None,
            'report_status': 'Out',
            'practice_primary_injury': 'Knee',
            'practice_secondary_injury': None,
            'practice_status': 'Did Not Participate',
            'date_modified': '2024-01-01 10:00:00'
        })
        
        # Transform
        result = self.transformer._transform_single_record(row)
        
        # Assertions
        assert result is not None
        assert result['season'] == 2024
        assert result['team'] == 'SEA'  # Should be normalized
        assert result['week'] == 1
        assert result['gsis_id'] == '00-0012345'
        assert result['position'] == 'QB'
        assert result['full_name'] == 'Test Player'
        assert result['report_status'] == 'Out'
        assert result['date_modified'] is not None
        
    def test_transform_single_record_missing_critical_data(self):
        """Test transformation with missing critical data."""
        # Row with missing gsis_id
        row = pd.Series({
            'season': 2024,
            'team': 'SEA',
            'week': 1,
            'gsis_id': None,  # Missing critical field
            'position': 'QB',
            'full_name': 'Test Player',
            'report_status': 'Out',
            'date_modified': '2024-01-01 10:00:00'
        })
        
        # Transform
        result = self.transformer._transform_single_record(row)
        
        # Should return None for invalid record
        assert result is None
        
    def test_validate_record_success(self):
        """Test successful record validation."""
        record = {
            'gsis_id': '00-0012345',
            'season': 2024,
            'week': 1,
            'team': 'SEA',
            'full_name': 'Test Player',
            'report_status': 'Out'
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should pass validation
        assert result is True
        
    def test_validate_record_missing_required_field(self):
        """Test validation with missing required field."""
        record = {
            'gsis_id': '00-0012345',
            'season': 2024,
            'week': 1,
            'team': 'SEA',
            # Missing 'full_name'
            'report_status': 'Out'
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False
        
    def test_validate_record_invalid_season(self):
        """Test validation with invalid season."""
        record = {
            'gsis_id': '00-0012345',
            'season': 1800,  # Invalid season
            'week': 1,
            'team': 'SEA',
            'full_name': 'Test Player',
            'report_status': 'Out'
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False
        
    def test_validate_record_invalid_week(self):
        """Test validation with invalid week."""
        record = {
            'gsis_id': '00-0012345',
            'season': 2024,
            'week': 25,  # Invalid week
            'team': 'SEA',
            'full_name': 'Test Player',
            'report_status': 'Out'
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False