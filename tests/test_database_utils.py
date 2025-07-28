"""Tests for database utilities."""

import unittest
from unittest.mock import Mock, patch, MagicMock

from src.core.utils.database import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class."""

    @patch('src.core.utils.database.get_supabase_client')
    def test_init_success(self, mock_get_client):
        """Test successful DatabaseManager initialization."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        
        self.assertEqual(db_manager.supabase, mock_client)
        self.assertEqual(db_manager.table_name, "test_table")

    @patch('src.core.utils.database.get_supabase_client')
    def test_init_no_client(self, mock_get_client):
        """Test DatabaseManager initialization with no client."""
        mock_get_client.return_value = None
        
        with self.assertRaises(RuntimeError) as context:
            DatabaseManager("test_table")
        
        self.assertIn("Could not initialize Supabase client", str(context.exception))

    @patch('src.core.utils.database.get_supabase_client')
    def test_get_record_count_success(self, mock_get_client):
        """Test successful record count retrieval."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.count = 42
        mock_response.error = None
        mock_client.table.return_value.select.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        count = db_manager.get_record_count()
        
        self.assertEqual(count, 42)
        mock_client.table.assert_called_with("test_table")

    @patch('src.core.utils.database.get_supabase_client')
    def test_get_record_count_exception(self, mock_get_client):
        """Test record count with exception."""
        mock_client = Mock()
        mock_client.table.return_value.select.return_value.execute.side_effect = Exception("DB Error")
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        count = db_manager.get_record_count()
        
        self.assertEqual(count, 0)

    @patch('src.core.utils.database.get_supabase_client')
    def test_clear_table_success(self, mock_get_client):
        """Test successful table clearing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = None
        mock_client.table.return_value.delete.return_value.neq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        result = db_manager.clear_table()
        
        self.assertTrue(result)

    @patch('src.core.utils.database.get_supabase_client')
    def test_clear_table_error(self, mock_get_client):
        """Test table clearing with error."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.error = "Delete failed"
        mock_client.table.return_value.delete.return_value.neq.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        result = db_manager.clear_table()
        
        self.assertFalse(result)

    @patch('src.core.utils.database.get_supabase_client')
    def test_upsert_records_success(self, mock_get_client):
        """Test successful data upsert."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{"id": 1}, {"id": 2}]
        mock_response.error = None
        mock_client.table.return_value.upsert.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        test_data = [{"test": "data1"}, {"test": "data2"}]
        result = db_manager.upsert_records(test_data)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['affected_rows'], 2)
        mock_client.table.assert_called_with("test_table")

    @patch('src.core.utils.database.get_supabase_client')
    def test_upsert_records_with_conflict_column(self, mock_get_client):
        """Test data upsert with conflict column specified."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{"id": 1}]
        mock_response.error = None
        mock_client.table.return_value.upsert.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        test_data = [{"test": "data"}]
        result = db_manager.upsert_records(test_data, on_conflict="unique_id")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['affected_rows'], 1)
        # Verify that upsert was called with the correct parameters
        mock_client.table.return_value.upsert.assert_called_with(json=test_data, on_conflict="unique_id")

    @patch('src.core.utils.database.get_supabase_client')
    def test_insert_records_success(self, mock_get_client):
        """Test successful data insertion."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{"id": 1}, {"id": 2}]
        mock_response.error = None
        mock_client.table.return_value.insert.return_value.execute.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        db_manager = DatabaseManager("test_table")
        test_data = [{"test": "data1"}, {"test": "data2"}]
        result = db_manager.insert_records(test_data)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['affected_rows'], 2)
        mock_client.table.assert_called_with("test_table")


if __name__ == '__main__':
    unittest.main()
