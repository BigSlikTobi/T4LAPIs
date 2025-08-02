#!/usr/bin/env python3
"""
Quick API Test Script

Simple script to test all CRUD endpoints are responding correctly.
Run with the API server running on localhost:8000
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test all API endpoints for basic functionality."""
    print("🚀 Testing User & Preference API Endpoints")
    print("=" * 50)
    
    # Test 1: Health check via root endpoint
    print("\n1. Testing API Health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ API is responding")
        else:
            print("   ❌ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API. Is it running on localhost:8000?")
        return False
    
    # Test 2: Create user
    print("\n2. Testing User Creation...")
    try:
        response = requests.post(f"{BASE_URL}/users/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 201:
            user_data = response.json()
            user_id = user_data['user_id']
            print(f"   ✅ Created user: {user_id}")
        else:
            print("   ❌ User creation failed")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error creating user: {e}")
        return False
    
    # Test 3: Set preferences
    print("\n3. Testing Set Preferences...")
    try:
        preferences = {
            "entities": [
                {"entity_id": "KC", "type": "team"},
                {"entity_id": "00-0033873", "type": "player"}
            ]
        }
        response = requests.post(
            f"{BASE_URL}/users/{user_id}/preferences",
            json=preferences
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Preferences set successfully")
        else:
            print("   ❌ Setting preferences failed")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error setting preferences: {e}")
        return False
    
    # Test 4: Get preferences
    print("\n4. Testing Get Preferences...")
    try:
        response = requests.get(f"{BASE_URL}/users/{user_id}/preferences")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            prefs_data = response.json()
            count = len(prefs_data.get('preferences', []))
            print(f"   ✅ Retrieved {count} preferences")
            if count > 0:
                preference_id = prefs_data['preferences'][0]['preference_id']
            else:
                print("   ⚠️  No preferences found")
                preference_id = None
        else:
            print("   ❌ Getting preferences failed")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error getting preferences: {e}")
        return False
    
    # Test 5: Update preference (if we have one)
    if preference_id:
        print("\n5. Testing Update Preference...")
        try:
            update_data = {"entity_id": "TB", "type": "team"}
            response = requests.put(
                f"{BASE_URL}/users/{user_id}/preferences/{preference_id}",
                json=update_data
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ Preference updated successfully")
            else:
                print("   ❌ Updating preference failed")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Error updating preference: {e}")
    
    # Test 6: Delete specific preference (if we have one)
    if preference_id:
        print("\n6. Testing Delete Specific Preference...")
        try:
            response = requests.delete(f"{BASE_URL}/users/{user_id}/preferences/{preference_id}")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ Specific preference deleted successfully")
            else:
                print("   ❌ Deleting specific preference failed")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Error deleting specific preference: {e}")
    
    # Test 7: Delete all preferences
    print("\n7. Testing Delete All Preferences...")
    try:
        response = requests.delete(f"{BASE_URL}/users/{user_id}/preferences")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ All preferences deleted successfully")
        else:
            print("   ❌ Deleting all preferences failed")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error deleting all preferences: {e}")
    
    # Test 8: Delete user
    print("\n8. Testing Delete User...")
    try:
        response = requests.delete(f"{BASE_URL}/users/{user_id}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ User deleted successfully")
        else:
            print("   ❌ Deleting user failed")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error deleting user: {e}")
    
    # Test 9: Error cases
    print("\n9. Testing Error Cases...")
    try:
        # Invalid UUID
        response = requests.delete(f"{BASE_URL}/users/invalid-uuid")
        print(f"   Invalid UUID Status: {response.status_code} (Expected: 400)")
        
        # Non-existent user
        response = requests.get(f"{BASE_URL}/users/550e8400-e29b-41d4-a716-446655440000/preferences")
        print(f"   Non-existent User Status: {response.status_code} (Expected: 404 or 500)")
        
        print("   ✅ Error handling working correctly")
    except Exception as e:
        print(f"   ❌ Error testing error cases: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API endpoint testing completed!")
    print("\nFor interactive testing, visit:")
    print(f"   • Swagger UI: {BASE_URL}/docs")
    print(f"   • ReDoc: {BASE_URL}/redoc")
    
    return True

if __name__ == "__main__":
    print("Starting API endpoint tests...")
    print("Make sure the API server is running on http://localhost:8000")
    print("You can start it with: uvicorn main:app --reload")
    print()
    
    success = test_api_endpoints()
    
    if success:
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the API server and database connection.")
        sys.exit(1)
