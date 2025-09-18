"""Test module for RostersDataLoader."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.data.loaders.rosters import RostersDataLoader
from src.core.data.transform import RosterDataTransformer


class TestRostersDataLoader:
    """Test cases for RostersDataLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.core.data.loaders.base.DatabaseManager') as mock_db_manager:
            # Mock the database manager to avoid Supabase connection issues in tests
            mock_db_manager.return_value = Mock()
            self.loader = RostersDataLoader()
            self.loader.db_manager = Mock()  # Ensure it's properly mocked
        
    def test_loader_initialization(self):
        """Test that the loader initializes correctly."""
        assert self.loader.table_name == "rosters"
        assert self.loader.transformer_class == RosterDataTransformer
        
    @patch('src.core.data.loaders.rosters.fetch_roster_data')
    def test_fetch_raw_data(self, mock_fetch):
        """Test fetching raw roster data."""
        # Mock data
        mock_df = pd.DataFrame({
            'season': [2024],
            'team': ['SEA'],
            'player_id': ['00-0012345'],
            'player_name': ['Test Player'],
            'position': ['QB'],
            'jersey_number': [12],
            'height': [76],
            'weight': [225]
        })
        mock_fetch.return_value = mock_df
        
        # Test
        result = self.loader.fetch_raw_data([2024])
        
        # Assertions
        mock_fetch.assert_called_once_with([2024])
        assert len(result) == 1
        assert result.iloc[0]['player_name'] == 'Test Player'
        
    @patch('src.core.data.loaders.rosters.fetch_roster_data')
    def test_load_rosters_success(self, mock_fetch):
        """Test successful roster data loading."""
        # Mock data
        mock_df = pd.DataFrame({
            'season': [2024, 2024],
            'team': ['SEA', 'KC'],
            'player_id': ['00-0012345', '00-0012346'],
            'player_name': ['Test Player 1', 'Test Player 2'],
            'position': ['QB', 'RB'],
            'jersey_number': [12, 25],
            'height': [76, 70],
            'weight': [225, 200]
        })
        mock_fetch.return_value = mock_df
        
        # Mock database manager
        mock_result = Mock()
        mock_result.inserted_count = 2
        mock_result.updated_count = 0
        mock_result.errors_count = 0
        self.loader.db_manager.upsert_data = Mock(return_value=mock_result)
        
        # Test
        result = self.loader.load_rosters(years=[2024], version=1)
        
        # Assertions
        assert result["success"] is True
        assert result["inserted"] == 2
        assert result["updated"] == 0
        assert result["errors"] == 0
        assert result["records_processed"] == 2
        
    @patch('src.core.data.loaders.rosters.fetch_roster_data')
    def test_load_rosters_dry_run(self, mock_fetch):
        """Test dry run mode."""
        # Mock data
        mock_df = pd.DataFrame({
            'season': [2024],
            'team': ['SEA'],
            'player_id': ['00-0012345'],
            'player_name': ['Test Player'],
            'position': ['QB'],
            'jersey_number': [12],
            'height': [76],
            'weight': [225]
        })
        mock_fetch.return_value = mock_df
        
        # Test
        result = self.loader.load_rosters(years=[2024], dry_run=True)
        
        # Assertions
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["records_processed"] == 1
        
    @patch('src.core.data.loaders.rosters.fetch_roster_data')
    def test_load_rosters_no_data(self, mock_fetch):
        """Test loading when no data is available."""
        # Mock empty data
        mock_fetch.return_value = pd.DataFrame()
        
        # Test
        result = self.loader.load_rosters(years=[2024])
        
        # Assertions
        assert result["success"] is False
        assert "No data found" in result["error"]
        
    def test_get_next_version_with_data(self):
        """Test getting next version when data exists."""
        # Mock database response
        mock_response = Mock()
        mock_response.data = [{'version': 3}]
        self.loader.db_manager.client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        # Test
        version = self.loader.get_next_version()
        
        # Assertions
        assert version == 4
        
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


class TestRosterDataTransformer:
    """Test cases for RosterDataTransformer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = RosterDataTransformer()
        
    def test_required_columns(self):
        """Test that required columns are defined correctly."""
        required = self.transformer._get_required_columns()
        expected = [
            'season', 'team', 'player_id', 'player_name', 'position'
        ]
        assert required == expected
        
    def test_transform_single_record_success(self):
        """Test successful transformation of a single record."""
        # Sample row
        row = pd.Series({
            'season': 2024,
            'team': 'SEA',
            'position': 'QB',
            'depth_chart_position': 'QB',
            'jersey_number': 12.0,
            'status': 'ACT',
            'player_name': 'Test Player',
            'first_name': 'Test',
            'last_name': 'Player',
            'birth_date': '1990-01-01',
            'height': 76.0,
            'weight': 225.0,
            'college': 'Test University',
            'player_id': '00-0012345',
            'years_exp': 5.0,
            'age': 34.0
        })
        
        # Transform
        result = self.transformer._transform_single_record(row)
        
        # Assertions
        assert result is not None
        assert result['season'] == 2024
        assert result['team'] == 'SEA'
        assert result['position'] == 'QB'
        assert result['player_id'] == '00-0012345'
        assert result['player_name'] == 'Test Player'
        assert result['jersey_number'] == 12
        assert result['height'] == 76
        assert result['weight'] == 225
        assert result['birth_date'] == '1990-01-01'
        
    def test_transform_single_record_missing_critical_data(self):
        """Test transformation with missing critical data."""
        # Row with missing player_id
        row = pd.Series({
            'season': 2024,
            'team': 'SEA',
            'position': 'QB',
            'player_id': None,  # Missing critical field
            'player_name': 'Test Player',
            'jersey_number': 12.0,
            'height': 76.0,
            'weight': 225.0
        })
        
        # Transform
        result = self.transformer._transform_single_record(row)
        
        # Should return None for invalid record
        assert result is None
        
    def test_validate_record_success(self):
        """Test successful record validation."""
        record = {
            'player_id': '00-0012345',
            'season': 2024,
            'team': 'SEA',
            'player_name': 'Test Player',
            'position': 'QB',
            'jersey_number': 12,
            'height': 76,
            'weight': 225
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should pass validation
        assert result is True
        
    def test_validate_record_missing_required_field(self):
        """Test validation with missing required field."""
        record = {
            'player_id': '00-0012345',
            'season': 2024,
            'team': 'SEA',
            # Missing 'player_name'
            'position': 'QB'
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False
        
    def test_validate_record_invalid_season(self):
        """Test validation with invalid season."""
        record = {
            'player_id': '00-0012345',
            'season': 1800,  # Invalid season
            'team': 'SEA',
            'player_name': 'Test Player',
            'position': 'QB'
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False
        
    def test_validate_record_invalid_jersey_number(self):
        """Test validation with invalid jersey number."""
        record = {
            'player_id': '00-0012345',
            'season': 2024,
            'team': 'SEA',
            'player_name': 'Test Player',
            'position': 'QB',
            'jersey_number': 150  # Invalid jersey number
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False
        
    def test_validate_record_invalid_height(self):
        """Test validation with invalid height."""
        record = {
            'player_id': '00-0012345',
            'season': 2024,
            'team': 'SEA',
            'player_name': 'Test Player',
            'position': 'QB',
            'height': 45  # Invalid height (too short)
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False
        
    def test_validate_record_invalid_weight(self):
        """Test validation with invalid weight."""
        record = {
            'player_id': '00-0012345',
            'season': 2024,
            'team': 'SEA',
            'player_name': 'Test Player',
            'position': 'QB',
            'weight': 50  # Invalid weight (too light)
        }
        
        # Validate
        result = self.transformer._validate_record(record)
        
        # Should fail validation
        assert result is False