"""
Test module for User & Preference API endpoints.

This module contains tests for the user management and preference endpoints
including creation, retrieval, and error handling scenarios.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch
import uuid
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api'))

from fastapi.testclient import TestClient


class TestUserEndpoints:
    """Test user management endpoints."""
    
    @pytest.fixture
    def mock_database_managers(self):
        """Mock database managers for testing."""
        with patch('api.main.DatabaseManager') as mock_db:
            mock_instance = Mock()
            mock_instance.supabase = Mock()
            mock_instance.insert_records = Mock(return_value={'success': True})
            mock_db.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def app_client(self, mock_database_managers):
        """Create test client with mocked dependencies."""
        import api.main as main_module
        
        # Set the mocked database managers
        main_module.users_db = mock_database_managers
        main_module.preferences_db = mock_database_managers
        
        return TestClient(main_module.app)
    
    def test_create_user_success(self, app_client, mock_database_managers):
        """Test successful user creation."""
        response = app_client.post("/users/")
        assert response.status_code == 201
        
        data = response.json()
        assert "user_id" in data
        assert "created_at" in data
        
        # Validate UUID format
        user_uuid = uuid.UUID(data["user_id"])
        assert str(user_uuid) == data["user_id"]
        
        # Verify database insert was called
        mock_database_managers.insert_records.assert_called_once()
        call_args = mock_database_managers.insert_records.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["user_id"] == data["user_id"]
    
    def test_create_user_database_failure(self, app_client, mock_database_managers):
        """Test user creation when database fails."""
        mock_database_managers.insert_records.return_value = {'success': False, 'error': 'DB Error'}
        
        response = app_client.post("/users/")
        assert response.status_code == 500
        
        data = response.json()
        assert data["detail"]["error"] == "Failed to create user"


class TestPreferenceEndpoints:
    """Test preference management endpoints."""
    
    @pytest.fixture
    def mock_database_managers(self):
        """Mock database managers for testing."""
        mock_users_db = Mock()
        mock_preferences_db = Mock()
        
        # Mock successful database operations
        mock_users_db.supabase = Mock()
        mock_preferences_db.supabase = Mock()
        mock_preferences_db.insert_records = Mock(return_value={'success': True})
        
        return mock_users_db, mock_preferences_db
    
    @pytest.fixture
    def app_client(self, mock_database_managers):
        """Create test client with mocked dependencies."""
        import api.main as main_module
        
        mock_users_db, mock_preferences_db = mock_database_managers
        main_module.users_db = mock_users_db
        main_module.preferences_db = mock_preferences_db
        
        return TestClient(main_module.app)
    
    def test_set_preferences_success(self, app_client, mock_database_managers):
        """Test successful preference setting."""
        mock_users_db, mock_preferences_db = mock_database_managers
        
        # Mock user exists check
        user_check_response = Mock()
        user_check_response.data = [{"user_id": "550e8400-e29b-41d4-a716-446655440000"}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = user_check_response
        
        # Mock preference deletion
        delete_response = Mock()
        delete_response.error = None
        mock_preferences_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = delete_response
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        preferences_data = {
            "entities": [
                {"entity_id": "00-0033873", "type": "player"},
                {"entity_id": "KC", "type": "team"}
            ]
        }
        
        response = app_client.post(f"/users/{user_id}/preferences", json=preferences_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == user_id
        assert data["preferences_set"] == 2
        
        # Verify database insert was called
        mock_preferences_db.insert_records.assert_called_once()
        call_args = mock_preferences_db.insert_records.call_args[0][0]
        assert len(call_args) == 2
    
    def test_set_preferences_invalid_user_id(self, app_client):
        """Test setting preferences with invalid user ID format."""
        preferences_data = {
            "entities": [{"entity_id": "00-0033873", "type": "player"}]
        }
        
        response = app_client.post("/users/invalid-uuid/preferences", json=preferences_data)
        assert response.status_code == 400
        
        data = response.json()
        assert data["detail"]["error"] == "Invalid user ID format"
    
    def test_set_preferences_user_not_found(self, app_client, mock_database_managers):
        """Test setting preferences for non-existent user."""
        mock_users_db, mock_preferences_db = mock_database_managers
        
        # Mock user doesn't exist
        user_check_response = Mock()
        user_check_response.data = []
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = user_check_response
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        preferences_data = {
            "entities": [{"entity_id": "00-0033873", "type": "player"}]
        }
        
        response = app_client.post(f"/users/{user_id}/preferences", json=preferences_data)
        assert response.status_code == 404
        
        data = response.json()
        assert data["detail"]["error"] == "User not found"
    
    def test_set_preferences_invalid_entity_type(self, app_client, mock_database_managers):
        """Test setting preferences with invalid entity type."""
        mock_users_db, mock_preferences_db = mock_database_managers
        
        # Mock user exists
        user_check_response = Mock()
        user_check_response.data = [{"user_id": "550e8400-e29b-41d4-a716-446655440000"}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = user_check_response
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        preferences_data = {
            "entities": [{"entity_id": "00-0033873", "type": "invalid_type"}]
        }
        
        response = app_client.post(f"/users/{user_id}/preferences", json=preferences_data)
        assert response.status_code == 422  # Pydantic validation error
        
        data = response.json()
        # Pydantic validation returns a different error structure
        assert "detail" in data
        # Check that it's related to enum validation
        assert any("Input should be" in str(detail) or "enum" in str(detail).lower() for detail in data["detail"])
    
    def test_get_preferences_success(self, app_client, mock_database_managers):
        """Test successful preference retrieval."""
        mock_users_db, mock_preferences_db = mock_database_managers
        
        # Mock user exists check
        user_check_response = Mock()
        user_check_response.data = [{"user_id": "550e8400-e29b-41d4-a716-446655440000"}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = user_check_response
        
        # Mock preferences data
        preferences_response = Mock()
        preferences_response.error = None
        preferences_response.data = [
            {
                "preference_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "entity_id": "00-0033873",
                "entity_type": "player",
                "created_at": "2025-08-02T10:30:00Z",
                "updated_at": "2025-08-02T10:30:00Z"
            }
        ]
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = preferences_response
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        response = app_client.get(f"/users/{user_id}/preferences")
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == user_id
        assert data["total_count"] == 1
        assert len(data["preferences"]) == 1
        assert data["preferences"][0]["entity_id"] == "00-0033873"
        assert data["preferences"][0]["entity_type"] == "player"
    
    def test_get_preferences_user_not_found(self, app_client, mock_database_managers):
        """Test getting preferences for non-existent user."""
        mock_users_db, mock_preferences_db = mock_database_managers
        
        # Mock user doesn't exist
        user_check_response = Mock()
        user_check_response.data = []
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = user_check_response
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        response = app_client.get(f"/users/{user_id}/preferences")
        assert response.status_code == 404
        
        data = response.json()
        assert data["detail"]["error"] == "User not found"
    
    def test_get_preferences_empty_list(self, app_client, mock_database_managers):
        """Test getting preferences when user has none."""
        mock_users_db, mock_preferences_db = mock_database_managers
        
        # Mock user exists
        user_check_response = Mock()
        user_check_response.data = [{"user_id": "550e8400-e29b-41d4-a716-446655440000"}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = user_check_response
        
        # Mock empty preferences
        preferences_response = Mock()
        preferences_response.error = None
        preferences_response.data = []
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = preferences_response
        
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        response = app_client.get(f"/users/{user_id}/preferences")
        assert response.status_code == 200
        
        data = response.json()
        assert data["user_id"] == user_id
        assert data["total_count"] == 0
        assert len(data["preferences"]) == 0


class TestEndpointIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest.fixture
    def mock_database_managers(self):
        """Mock database managers for integration testing."""
        mock_users_db = Mock()
        mock_preferences_db = Mock()
        
        mock_users_db.supabase = Mock()
        mock_preferences_db.supabase = Mock()
        mock_users_db.insert_records = Mock(return_value={'success': True})
        mock_preferences_db.insert_records = Mock(return_value={'success': True})
        
        return mock_users_db, mock_preferences_db
    
    @pytest.fixture
    def app_client(self, mock_database_managers):
        """Create test client with mocked dependencies."""
        import api.main as main_module
        
        mock_users_db, mock_preferences_db = mock_database_managers
        main_module.users_db = mock_users_db
        main_module.preferences_db = mock_preferences_db
        
        return TestClient(main_module.app)
    
    def test_full_user_workflow(self, app_client, mock_database_managers):
        """Test complete workflow: create user -> set preferences -> get preferences."""
        mock_users_db, mock_preferences_db = mock_database_managers
        
        # Step 1: Create user
        create_response = app_client.post("/users/")
        assert create_response.status_code == 201
        user_data = create_response.json()
        user_id = user_data["user_id"]
        
        # Step 2: Set preferences (mock user exists)
        user_check_response = Mock()
        user_check_response.data = [{"user_id": user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = user_check_response
        
        delete_response = Mock()
        delete_response.error = None
        mock_preferences_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = delete_response
        
        preferences_data = {
            "entities": [
                {"entity_id": "00-0033873", "type": "player"},
                {"entity_id": "KC", "type": "team"}
            ]
        }
        
        set_response = app_client.post(f"/users/{user_id}/preferences", json=preferences_data)
        assert set_response.status_code == 200
        
        # Step 3: Get preferences (mock preferences exist)
        preferences_response = Mock()
        preferences_response.error = None
        preferences_response.data = [
            {
                "preference_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": user_id,
                "entity_id": "00-0033873",
                "entity_type": "player",
                "created_at": "2025-08-02T10:30:00Z",
                "updated_at": "2025-08-02T10:30:00Z"
            },
            {
                "preference_id": "456e7890-e12b-34d5-a678-901234567890",
                "user_id": user_id,
                "entity_id": "KC",
                "entity_type": "team",
                "created_at": "2025-08-02T10:35:00Z",
                "updated_at": "2025-08-02T10:35:00Z"
            }
        ]
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = preferences_response
        
        get_response = app_client.get(f"/users/{user_id}/preferences")
        assert get_response.status_code == 200
        
        get_data = get_response.json()
        assert get_data["user_id"] == user_id
        assert get_data["total_count"] == 2
        assert len(get_data["preferences"]) == 2


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
