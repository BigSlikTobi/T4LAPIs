"""Tests for the base data loader module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from abc import ABC
import pandas as pd

from src.core.data.loaders.base import BaseDataLoader


class TestableDataLoader(BaseDataLoader):
    """Concrete implementation of BaseDataLoader for testing."""
    
    def fetch_raw_data(self, **kwargs):
        """Test implementation of fetch_raw_data."""
        # Return a DataFrame with test data
        return pd.DataFrame([{"test": "data"}])
    
    @property
    def transformer_class(self):
        """Test transformer class."""
        mock_transformer_class = Mock()
        mock_transformer = Mock()
        mock_transformer.transform.return_value = [{"transformed": "data"}]
        mock_transformer_class.return_value = mock_transformer
        return mock_transformer_class


class TestBaseDataLoader:
    """Test cases for BaseDataLoader class."""

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_init_success(self, mock_db_manager_class):
        """Test successful initialization of BaseDataLoader."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TestableDataLoader("test_table")
        
        assert loader.table_name == "test_table"
        assert loader.db_manager == mock_db_manager
        mock_db_manager_class.assert_called_once_with("test_table")

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_success(self, mock_db_manager_class):
        """Test successful data loading."""
        mock_db_manager = Mock()
        mock_db_manager.upsert_records.return_value = {
            'success': True,
            'affected_rows': 1
        }
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TestableDataLoader("test_table")
        
        result = loader.load_data()
        
        assert result['success'] is True
        assert result['total_fetched'] == 1
        assert result['total_validated'] == 1
        assert result['upsert_result']['affected_rows'] == 1

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_dry_run(self, mock_db_manager_class):
        """Test dry run mode."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TestableDataLoader("test_table")
        
        result = loader.load_data(dry_run=True)
        
        assert result['success'] is True
        assert result['dry_run'] is True
        mock_db_manager.upsert_records.assert_not_called()

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_clear_table(self, mock_db_manager_class):
        """Test clearing table before loading."""
        mock_db_manager = Mock()
        mock_db_manager.clear_table.return_value = {'success': True}
        mock_db_manager.upsert_records.return_value = {
            'success': True,
            'affected_rows': 1
        }
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TestableDataLoader("test_table")
        
        result = loader.load_data(clear_table=True)
        
        assert result['success'] is True
        mock_db_manager.clear_table.assert_called_once()

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_no_data(self, mock_db_manager_class):
        """Test loading when no data is fetched."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TestableDataLoader("test_table")
        
        # Override fetch_raw_data to return empty DataFrame
        with patch.object(loader, 'fetch_raw_data', return_value=pd.DataFrame()):
            result = loader.load_data()
            
            assert result['success'] is False

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_load_data_transform_exception(self, mock_db_manager_class):
        """Test handling of transformation exceptions."""
        mock_db_manager = Mock()
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TestableDataLoader("test_table")
        
        # Override the transformer to raise an exception
        original_transform = loader.transform_data
        
        def failing_transform(raw_data):
            raise Exception("Transform error")
            
        with patch.object(loader, 'transform_data', side_effect=failing_transform):
            result = loader.load_data()
            
            assert result['success'] is False
            assert "Transform error" in str(result.get('error', ''))

    @patch('src.core.data.loaders.base.DatabaseManager')
    def test_get_record_count(self, mock_db_manager_class):
        """Test getting record count."""
        mock_db_manager = Mock()
        mock_db_manager.get_record_count.return_value = 42
        mock_db_manager_class.return_value = mock_db_manager
        
        loader = TestableDataLoader("test_table")
        count = loader.get_record_count()
        
        assert count == 42
        mock_db_manager.get_record_count.assert_called_once_with(None)

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        # This should fail because we can't instantiate an abstract class
        with pytest.raises(TypeError):
            # Can't instantiate abstract class directly
            BaseDataLoader("test")
        with pytest.raises(TypeError):
            BaseDataLoader("test_table")
