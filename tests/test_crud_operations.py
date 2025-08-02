"""
Comprehensive tests for User & Preference API CRUD operations

This module tests all CRUD operations for users and preferences including:
- User creation, deletion
- Preference creation, reading, updating, deletion (individual and bulk)
"""

import pytest
import uuid
from unittest.mock import Mock, patch

# Import test configuration
from test_config import (
    VALID_UUID, VALID_PREFERENCE_ID, INVALID_UUID,
    PREFERENCE_UPDATE_DATA, MULTIPLE_PREFERENCES
)

# Import FastAPI testing tools
try:
    from fastapi.testclient import TestClient
    # Import the FastAPI app
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from api.main import app
    client = TestClient(app)
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available for testing. Install with: pip install fastapi")
    FASTAPI_AVAILABLE = False
    client = None

# Skip all tests if FastAPI is not available
pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")


class TestUserCRUD:
    """Test CRUD operations for users."""

    @patch('api.main.users_db')
    def test_create_user_success(self, mock_users_db):
        """Test successful user creation."""
        # Mock successful database insertion
        mock_users_db.insert_records.return_value = {'success': True}
        
        response = client.post("/users/")
        
        assert response.status_code == 201
        data = response.json()
        assert "user_id" in data
        assert "created_at" in data
        
        # Verify UUID format
        user_uuid = uuid.UUID(data["user_id"])
        assert str(user_uuid) == data["user_id"]

    @patch('api.main.users_db')
    def test_delete_user_success(self, mock_users_db):
        """Test successful user deletion."""
        # Mock user exists check
        mock_check_result = Mock()
        mock_check_result.data = [{"user_id": VALID_UUID}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_check_result
        
        # Mock successful deletion
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_users_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        response = client.delete(f"/users/{VALID_UUID}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == VALID_UUID
        assert "deleted successfully" in data["message"]

    @patch('api.main.users_db')
    def test_delete_user_not_found(self, mock_users_db):
        """Test user deletion when user doesn't exist."""
        # Mock user doesn't exist
        mock_check_result = Mock()
        mock_check_result.data = []
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_check_result
        
        response = client.delete(f"/users/{VALID_UUID}")
        
        assert response.status_code == 404
        data = response.json()
        assert "User not found" in data["detail"]["error"]

    def test_delete_user_invalid_uuid(self):
        """Test user deletion with invalid UUID format."""
        response = client.delete(f"/users/{INVALID_UUID}")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid user ID format" in data["detail"]["error"]


class TestPreferenceCRUD:
    """Test CRUD operations for user preferences."""

    @patch('api.main.preferences_db')
    @patch('api.main.users_db')
    def test_update_preference_success(self, mock_users_db, mock_preferences_db):
        """Test successful preference update."""
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        test_preference_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preference exists
        mock_pref_result = Mock()
        mock_pref_result.data = [{"preference_id": test_preference_id, "user_id": test_user_id}]
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_pref_result
        
        # Mock successful update
        mock_update_result = Mock()
        mock_update_result.error = None
        mock_preferences_db.supabase.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = mock_update_result
        
        update_data = {
            "entity_id": "TB",
            "type": "team"
        }
        
        response = client.put(f"/users/{test_user_id}/preferences/{test_preference_id}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["entity_id"] == "TB"
        assert data["entity_type"] == "team"

    @patch('api.main.preferences_db')
    @patch('api.main.users_db')
    def test_update_preference_not_found(self, mock_users_db, mock_preferences_db):
        """Test preference update when preference doesn't exist."""
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        test_preference_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preference doesn't exist
        mock_pref_result = Mock()
        mock_pref_result.data = []
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_pref_result
        
        update_data = {
            "entity_id": "TB",
            "type": "team"
        }
        
        response = client.put(f"/users/{test_user_id}/preferences/{test_preference_id}", json=update_data)
        
        assert response.status_code == 404
        data = response.json()
        assert "Preference not found" in data["detail"]["error"]

    @patch('api.main.preferences_db')
    @patch('api.main.users_db')
    def test_delete_all_preferences_success(self, mock_users_db, mock_preferences_db):
        """Test successful deletion of all user preferences."""
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preferences count
        mock_count_result = Mock()
        mock_count_result.data = [{"preference_id": "1"}, {"preference_id": "2"}]
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        
        # Mock successful deletion
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_preferences_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        response = client.delete(f"/users/{test_user_id}/preferences")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["preferences_deleted"] == 2
        assert "Successfully deleted 2 preferences" in data["message"]

    @patch('api.main.preferences_db')
    @patch('api.main.users_db')
    def test_delete_specific_preference_success(self, mock_users_db, mock_preferences_db):
        """Test successful deletion of a specific preference."""
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        test_preference_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preference exists
        mock_pref_result = Mock()
        mock_pref_result.data = [{
            "preference_id": test_preference_id,
            "user_id": test_user_id,
            "entity_id": "KC",
            "entity_type": "team"
        }]
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_pref_result
        
        # Mock successful deletion
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_preferences_db.supabase.table.return_value.delete.return_value.eq.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        response = client.delete(f"/users/{test_user_id}/preferences/{test_preference_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["preference_id"] == test_preference_id
        assert data["deleted_entity"]["entity_id"] == "KC"
        assert data["deleted_entity"]["entity_type"] == "team"

    @patch('api.main.preferences_db')
    @patch('api.main.users_db')
    def test_delete_specific_preference_not_found(self, mock_users_db, mock_preferences_db):
        """Test deletion of non-existent preference."""
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        test_preference_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preference doesn't exist
        mock_pref_result = Mock()
        mock_pref_result.data = []
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_pref_result
        
        response = client.delete(f"/users/{test_user_id}/preferences/{test_preference_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert "Preference not found" in data["detail"]["error"]

    def test_update_preference_invalid_type(self):
        """Test preference update with invalid entity type."""
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        test_preference_id = "123e4567-e89b-12d3-a456-426614174000"
        
        update_data = {
            "entity_id": "TB",
            "type": "invalid_type"
        }
        
        response = client.put(f"/users/{test_user_id}/preferences/{test_preference_id}", json=update_data)
        
        assert response.status_code == 422  # Pydantic validation error


class TestCompleteCRUDWorkflow:
    """Test complete CRUD workflows combining multiple operations."""

    @patch('api.main.preferences_db')
    @patch('api.main.users_db')
    def test_complete_user_lifecycle(self, mock_users_db, mock_preferences_db):
        """Test complete user lifecycle: create, add preferences, update, delete."""
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        # 1. Create user
        mock_users_db.insert_records.return_value = {'success': True}
        
        create_response = client.post("/users/")
        assert create_response.status_code == 201
        
        # 2. Set preferences (mocking the set preferences workflow)
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_preferences_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        mock_preferences_db.insert_records.return_value = {'success': True}
        
        preferences_data = {
            "entities": [
                {"entity_id": "KC", "type": "team"},
                {"entity_id": "00-0033873", "type": "player"}
            ]
        }
        
        set_response = client.post(f"/users/{test_user_id}/preferences", json=preferences_data)
        assert set_response.status_code == 200
        
        # 3. Delete all preferences
        mock_count_result = Mock()
        mock_count_result.data = [{"preference_id": "1"}, {"preference_id": "2"}]
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        
        delete_prefs_response = client.delete(f"/users/{test_user_id}/preferences")
        assert delete_prefs_response.status_code == 200
        
        # 4. Delete user
        mock_users_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        delete_user_response = client.delete(f"/users/{test_user_id}")
        assert delete_user_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
