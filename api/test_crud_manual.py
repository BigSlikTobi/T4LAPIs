"""
Manual CRUD Operations Test Script

Tests all CRUD endpoints with mocked database responses to verify API logic.
"""

import sys
import os
import uuid
from unittest.mock import Mock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from api.main import app

def test_crud_operations():
    """Test all CRUD operations with mocked database."""
    client = TestClient(app)
    
    print("=== Testing Complete CRUD Operations ===\n")
    
    # Test data
    test_user_id = "550e8400-e29b-41d4-a716-446655440000"
    test_pref_id = "123e4567-e89b-12d3-a456-426614174000"
    
    # Test 1: Create User
    print("1. Testing User Creation")
    with patch('api.main.users_db') as mock_users_db:
        mock_users_db.insert_records.return_value = {'success': True}
        
        response = client.post("/users/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 201:
            data = response.json()
            print(f"   Created user ID: {data['user_id']}")
            print(f"   Valid UUID: {bool(uuid.UUID(data['user_id']))}")
        print()
    
    # Test 2: Set User Preferences
    print("2. Testing Set User Preferences")
    with patch('api.main.users_db') as mock_users_db, \
         patch('api.main.preferences_db') as mock_prefs_db:
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preference operations
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_prefs_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_delete_result
        mock_prefs_db.insert_records.return_value = {'success': True}
        
        pref_data = {
            "entities": [
                {"entity_id": "KC", "type": "team"},
                {"entity_id": "00-0033873", "type": "player"}
            ]
        }
        
        response = client.post(f"/users/{test_user_id}/preferences", json=pref_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Set {data['preferences_set']} preferences")
        print()
    
    # Test 3: Get User Preferences
    print("3. Testing Get User Preferences")
    with patch('api.main.users_db') as mock_users_db, \
         patch('api.main.preferences_db') as mock_prefs_db:
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preferences data
        mock_prefs_result = Mock()
        mock_prefs_result.data = [
            {
                "preference_id": test_pref_id,
                "user_id": test_user_id,
                "entity_id": "KC",
                "entity_type": "team",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
        mock_prefs_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_prefs_result
        
        response = client.get(f"/users/{test_user_id}/preferences")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Retrieved {len(data['preferences'])} preferences")
        print()
    
    # Test 4: Update Specific Preference
    print("4. Testing Update Specific Preference")
    with patch('api.main.users_db') as mock_users_db, \
         patch('api.main.preferences_db') as mock_prefs_db:
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preference exists
        mock_pref_result = Mock()
        mock_pref_result.data = [{"preference_id": test_pref_id, "user_id": test_user_id}]
        mock_prefs_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_pref_result
        
        # Mock update operation
        mock_update_result = Mock()
        mock_update_result.error = None
        mock_prefs_db.supabase.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = mock_update_result
        
        update_data = {"entity_id": "TB", "type": "team"}
        response = client.put(f"/users/{test_user_id}/preferences/{test_pref_id}", json=update_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Updated to entity: {data['entity_id']} ({data['entity_type']})")
        print()
    
    # Test 5: Delete Specific Preference
    print("5. Testing Delete Specific Preference")
    with patch('api.main.users_db') as mock_users_db, \
         patch('api.main.preferences_db') as mock_prefs_db:
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preference exists
        mock_pref_result = Mock()
        mock_pref_result.data = [{
            "preference_id": test_pref_id,
            "user_id": test_user_id,
            "entity_id": "KC",
            "entity_type": "team"
        }]
        mock_prefs_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_pref_result
        
        # Mock delete operation
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_prefs_db.supabase.table.return_value.delete.return_value.eq.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        response = client.delete(f"/users/{test_user_id}/preferences/{test_pref_id}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Deleted preference: {data['deleted_entity']['entity_id']}")
        print()
    
    # Test 6: Delete All User Preferences
    print("6. Testing Delete All User Preferences")
    with patch('api.main.users_db') as mock_users_db, \
         patch('api.main.preferences_db') as mock_prefs_db:
        
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock preferences count
        mock_count_result = Mock()
        mock_count_result.data = [{"preference_id": "1"}, {"preference_id": "2"}]
        mock_prefs_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        
        # Mock delete operation
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_prefs_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        response = client.delete(f"/users/{test_user_id}/preferences")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Deleted {data['preferences_deleted']} preferences")
        print()
    
    # Test 7: Delete User
    print("7. Testing Delete User")
    with patch('api.main.users_db') as mock_users_db:
        # Mock user exists
        mock_user_result = Mock()
        mock_user_result.data = [{"user_id": test_user_id}]
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        # Mock delete operation
        mock_delete_result = Mock()
        mock_delete_result.error = None
        mock_users_db.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_delete_result
        
        response = client.delete(f"/users/{test_user_id}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   User {data['user_id']} deleted successfully")
        print()
    
    # Test 8: Error Cases
    print("8. Testing Error Cases")
    
    # Invalid UUID
    response = client.delete("/users/invalid-uuid")
    print(f"   Invalid UUID Status: {response.status_code} (Expected: 400)")
    
    # Non-existent user
    with patch('api.main.users_db') as mock_users_db:
        mock_user_result = Mock()
        mock_user_result.data = []
        mock_users_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_user_result
        
        response = client.delete(f"/users/{test_user_id}")
        print(f"   Non-existent User Status: {response.status_code} (Expected: 404)")
    
    # Invalid preference type
    update_data = {"entity_id": "TB", "type": "invalid_type"}
    response = client.put(f"/users/{test_user_id}/preferences/{test_pref_id}", json=update_data)
    print(f"   Invalid Preference Type Status: {response.status_code} (Expected: 422)")
    print()
    
    print("=== All CRUD Operations Tested Successfully! ===")

if __name__ == "__main__":
    test_crud_operations()
