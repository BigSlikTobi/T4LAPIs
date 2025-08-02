"""
Test module for the FastAPI User & Preference API.

This module contains tests for the basic FastAPI application setup,
including endpoint functionality and database connectivity.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api'))

from fastapi.testclient import TestClient


class TestFastAPIBasics:
    """Test basic FastAPI application functionality."""
    
    @pytest.fixture
    def mock_database_managers(self):
        """Mock database managers for testing."""
        with patch('api.main.DatabaseManager') as mock_db:
            mock_instance = Mock()
            mock_instance.supabase = Mock()
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
    
    def test_root_endpoint(self, app_client):
        """Test the root endpoint returns correct information."""
        response = app_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "T4L NFL User & Preference API"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"
    
    def test_health_check_healthy(self, app_client):
        """Test health check endpoint when system is healthy."""
        response = app_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert "timestamp" in data
    
    def test_docs_endpoint_accessible(self, app_client):
        """Test that the auto-generated docs endpoint is accessible."""
        response = app_client.get("/docs")
        assert response.status_code == 200
        # FastAPI docs should return HTML
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_schema_accessible(self, app_client):
        """Test that the OpenAPI schema is accessible."""
        response = app_client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "T4L NFL User & Preference API"
        assert schema["info"]["version"] == "1.0.0"


class TestFastAPIWithDatabaseErrors:
    """Test FastAPI behavior when database is unavailable."""
    
    def test_health_check_with_db_error(self):
        """Test health check when database managers are not initialized."""
        # Import app and manually set database managers to None
        import api.main as main_module
        
        # Save original values
        original_users_db = main_module.users_db
        original_preferences_db = main_module.preferences_db
        
        try:
            # Set database managers to None to simulate initialization failure
            main_module.users_db = None
            main_module.preferences_db = None
            
            # Create test client with the modified state
            client = TestClient(main_module.app)
            
            response = client.get("/health")
            assert response.status_code == 503
            
            data = response.json()
            assert "detail" in data
            assert data["detail"]["status"] == "unhealthy"
            assert data["detail"]["database"] == "disconnected"
            
        finally:
            # Restore original values
            main_module.users_db = original_users_db
            main_module.preferences_db = original_preferences_db


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
