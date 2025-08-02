"""
FastAPI application for User & Preference API (FR-01)

This module provides REST API endpoints for managing users and their preferences
for NFL teams and players. It integrates with the existing database structure
to provide user management capabilities.
"""

import sys
import os
from typing import Dict, Any, List
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import database utilities
from src.core.utils.database import DatabaseManager
from src.core.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Global variables for database managers
users_db = None
preferences_db = None


# Enum for entity types
class EntityType(str, Enum):
    """Enum for valid entity types."""
    player = "player"
    team = "team"


# Pydantic models for API request/response schemas
class EntityPreference(BaseModel):
    """Model for a single entity preference (team or player)."""
    entity_id: str = Field(..., description="ID of the entity (player_id or team abbreviation)")
    type: EntityType = Field(..., description="Type of entity: 'player' or 'team'")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_id": "00-0033873",
                "type": "player"
            }
        }
    }


class PreferencesRequest(BaseModel):
    """Model for setting user preferences."""
    entities: List[EntityPreference] = Field(..., description="List of entity preferences")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "entities": [
                    {"entity_id": "00-0033873", "type": "player"},
                    {"entity_id": "KC", "type": "team"}
                ]
            }
        }
    }


class UserResponse(BaseModel):
    """Model for user creation response."""
    user_id: str = Field(..., description="Unique identifier for the created user")
    created_at: str = Field(..., description="Timestamp when the user was created")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2025-08-02T10:30:00Z"
            }
        }
    }


class PreferenceResponse(BaseModel):
    """Model for a user preference response."""
    preference_id: str = Field(..., description="Unique identifier for the preference")
    user_id: str = Field(..., description="User ID this preference belongs to")
    entity_id: str = Field(..., description="Entity ID (player_id or team abbreviation)")
    entity_type: str = Field(..., description="Type of entity: 'player' or 'team'")
    created_at: str = Field(..., description="Timestamp when the preference was created")
    updated_at: str = Field(..., description="Timestamp when the preference was last updated")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "preference_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "entity_id": "00-0033873",
                "entity_type": "player",
                "created_at": "2025-08-02T10:30:00Z",
                "updated_at": "2025-08-02T10:30:00Z"
            }
        }
    }


class PreferencesListResponse(BaseModel):
    """Model for listing user preferences."""
    user_id: str = Field(..., description="User ID")
    preferences: List[PreferenceResponse] = Field(..., description="List of user preferences")
    total_count: int = Field(..., description="Total number of preferences")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "total_count": 2,
                "preferences": [
                    {
                        "preference_id": "123e4567-e89b-12d3-a456-426614174000",
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "entity_id": "00-0033873",
                        "entity_type": "player",
                        "created_at": "2025-08-02T10:30:00Z",
                        "updated_at": "2025-08-02T10:30:00Z"
                    },
                    {
                        "preference_id": "456e7890-e12b-34d5-a678-901234567890",
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "entity_id": "KC",
                        "entity_type": "team",
                        "created_at": "2025-08-02T10:35:00Z",
                        "updated_at": "2025-08-02T10:35:00Z"
                    }
                ]
            }
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting T4L NFL User & Preference API...")
    
    global users_db, preferences_db
    
    # Initialize database managers
    try:
        users_db = DatabaseManager("users")
        preferences_db = DatabaseManager("user_preferences")
        logger.info("Database managers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database managers: {e}")
        users_db = None
        preferences_db = None
    
    # Verify database connections
    if users_db is None or preferences_db is None:
        logger.error("Database managers not initialized - API may not function correctly")
    else:
        logger.info("API ready to serve requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down T4L NFL User & Preference API...")


# Create FastAPI application instance with lifespan
app = FastAPI(
    title="T4L NFL User & Preference API",
    description="API for managing users and their NFL team/player preferences",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint providing API information."""
    return {
        "message": "T4L NFL User & Preference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring and load balancers."""
    try:
        # Test database connectivity
        if users_db is None or preferences_db is None:
            raise Exception("Database managers not initialized")
        
        # Simple database connectivity test
        # We'll just check if we can access the supabase client
        if not hasattr(users_db, 'supabase') or not hasattr(preferences_db, 'supabase'):
            raise Exception("Database connections not available")
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )


# User & Preference API Endpoints

@app.post("/users/", response_model=UserResponse, status_code=201)
async def create_user() -> UserResponse:
    """
    Create a new user.
    
    Takes no input and creates a new entry in the users table.
    Returns the user_id of the newly created user.
    """
    try:
        # Generate unique user ID
        user_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Prepare user record
        user_record = {
            'user_id': user_id,
            'created_at': current_time,
            'updated_at': current_time
        }
        
        # Insert user into database
        result = users_db.insert_records([user_record])
        
        if not result.get('success', False):
            logger.error(f"Failed to create user: {result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to create user",
                    "message": "Database insertion failed"
                }
            )
        
        logger.info(f"Created new user with ID: {user_id}")
        
        return UserResponse(
            user_id=user_id,
            created_at=current_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to create user"
            }
        )


@app.post("/users/{user_id}/preferences", response_model=Dict[str, Any])
async def set_user_preferences(user_id: str, preferences: PreferencesRequest) -> Dict[str, Any]:
    """
    Set preferences for a user.
    
    Takes a user_id in the path and a request body with a list of entities.
    The endpoint upserts these preferences into the user_preferences table for the given user_id.
    """
    try:
        # Validate user_id format (UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid user ID format",
                    "message": "User ID must be a valid UUID"
                }
            )
        
        # Check if user exists
        user_check = users_db.supabase.table("users").select("user_id").eq("user_id", user_id).execute()
        
        if not hasattr(user_check, 'data') or not user_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "User not found",
                    "message": f"User with ID {user_id} does not exist"
                }
            )
        
        # Clear existing preferences for this user first (upsert behavior)
        delete_result = preferences_db.supabase.table("user_preferences").delete().eq("user_id", user_id).execute()
        
        if hasattr(delete_result, 'error') and delete_result.error:
            logger.warning(f"Warning: Could not clear existing preferences for user {user_id}: {delete_result.error}")
        
        current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        preference_records = []
        
        # Prepare preference records for upsert
        for entity in preferences.entities:
            preference_record = {
                'preference_id': str(uuid.uuid4()),
                'user_id': user_id,
                'entity_id': entity.entity_id,
                'entity_type': entity.type.value,  # Convert enum to string
                'created_at': current_time,
                'updated_at': current_time
            }
            preference_records.append(preference_record)
        
        if hasattr(delete_result, 'error') and delete_result.error:
            logger.warning(f"Warning: Could not clear existing preferences for user {user_id}: {delete_result.error}")
        
        # Insert new preferences
        if preference_records:
            result = preferences_db.insert_records(preference_records)
            
            if not result.get('success', False):
                logger.error(f"Failed to set preferences for user {user_id}: {result.get('error', 'Unknown error')}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Failed to set preferences",
                        "message": "Database insertion failed"
                    }
                )
        
        logger.info(f"Set {len(preference_records)} preferences for user {user_id}")
        
        return {
            "success": True,
            "user_id": user_id,
            "preferences_set": len(preference_records),
            "message": f"Successfully set {len(preference_records)} preferences"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error setting preferences for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to set preferences"
            }
        )


@app.get("/users/{user_id}/preferences", response_model=PreferencesListResponse)
async def get_user_preferences(user_id: str) -> PreferencesListResponse:
    """
    Get preferences for a user.
    
    Takes a user_id in the path and returns a list of the user's current preferences 
    from the user_preferences table.
    """
    try:
        # Validate user_id format (UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid user ID format",
                    "message": "User ID must be a valid UUID"
                }
            )
        
        # Check if user exists
        user_check = users_db.supabase.table("users").select("user_id").eq("user_id", user_id).execute()
        
        if not hasattr(user_check, 'data') or not user_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "User not found",
                    "message": f"User with ID {user_id} does not exist"
                }
            )
        
        # Get user preferences
        preferences_result = preferences_db.supabase.table("user_preferences").select(
            "preference_id, user_id, entity_id, entity_type, created_at, updated_at"
        ).eq("user_id", user_id).execute()
        
        if hasattr(preferences_result, 'error') and preferences_result.error:
            logger.error(f"Database error getting preferences for user {user_id}: {preferences_result.error}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database error",
                    "message": "Failed to retrieve preferences"
                }
            )
        
        preferences_data = preferences_result.data if hasattr(preferences_result, 'data') else []
        
        # Convert to response format
        preferences_list = []
        for pref in preferences_data:
            preferences_list.append(PreferenceResponse(
                preference_id=pref['preference_id'],
                user_id=pref['user_id'],
                entity_id=pref['entity_id'],
                entity_type=pref['entity_type'],
                created_at=pref['created_at'],
                updated_at=pref['updated_at']
            ))
        
        logger.info(f"Retrieved {len(preferences_list)} preferences for user {user_id}")
        
        return PreferencesListResponse(
            user_id=user_id,
            preferences=preferences_list,
            total_count=len(preferences_list)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting preferences for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to retrieve preferences"
            }
        )


@app.delete("/users/{user_id}")
async def delete_user(user_id: str) -> Dict[str, Any]:
    """
    Delete a user and all their preferences.
    
    Takes a user_id in the path and deletes the user from the users table.
    All associated preferences are also deleted due to CASCADE constraint.
    """
    try:
        # Validate user_id format (UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid user ID format",
                    "message": "User ID must be a valid UUID"
                }
            )
        
        # Check if user exists
        user_check = users_db.supabase.table("users").select("user_id").eq("user_id", user_id).execute()
        
        if not hasattr(user_check, 'data') or not user_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "User not found",
                    "message": f"User with ID {user_id} does not exist"
                }
            )
        
        # Delete user (preferences will be deleted automatically due to CASCADE)
        delete_result = users_db.supabase.table("users").delete().eq("user_id", user_id).execute()
        
        if hasattr(delete_result, 'error') and delete_result.error:
            logger.error(f"Database error deleting user {user_id}: {delete_result.error}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database error",
                    "message": "Failed to delete user"
                }
            )
        
        logger.info(f"Deleted user {user_id} and all associated preferences")
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "User and all associated preferences deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to delete user"
            }
        )


@app.put("/users/{user_id}/preferences/{preference_id}")
async def update_user_preference(user_id: str, preference_id: str, preference: EntityPreference) -> Dict[str, Any]:
    """
    Update a specific user preference.
    
    Takes a user_id and preference_id in the path and updates the specific preference.
    """
    try:
        # Validate user_id format (UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid user ID format",
                    "message": "User ID must be a valid UUID"
                }
            )
        
        # Validate preference_id format (UUID)
        try:
            uuid.UUID(preference_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid preference ID format",
                    "message": "Preference ID must be a valid UUID"
                }
            )
        
        # Check if user exists
        user_check = users_db.supabase.table("users").select("user_id").eq("user_id", user_id).execute()
        
        if not hasattr(user_check, 'data') or not user_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "User not found",
                    "message": f"User with ID {user_id} does not exist"
                }
            )
        
        # Check if preference exists and belongs to the user
        pref_check = preferences_db.supabase.table("user_preferences").select("preference_id, user_id").eq("preference_id", preference_id).eq("user_id", user_id).execute()
        
        if not hasattr(pref_check, 'data') or not pref_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Preference not found",
                    "message": f"Preference with ID {preference_id} not found for user {user_id}"
                }
            )
        
        current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Update the preference
        update_data = {
            'entity_id': preference.entity_id,
            'entity_type': preference.type.value,  # Convert enum to string
            'updated_at': current_time
        }
        
        update_result = preferences_db.supabase.table("user_preferences").update(update_data).eq("preference_id", preference_id).eq("user_id", user_id).execute()
        
        if hasattr(update_result, 'error') and update_result.error:
            logger.error(f"Database error updating preference {preference_id}: {update_result.error}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database error",
                    "message": "Failed to update preference"
                }
            )
        
        logger.info(f"Updated preference {preference_id} for user {user_id}")
        
        return {
            "success": True,
            "user_id": user_id,
            "preference_id": preference_id,
            "entity_id": preference.entity_id,
            "entity_type": preference.type.value,  # Convert enum to string
            "message": "Preference updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating preference {preference_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to update preference"
            }
        )


@app.delete("/users/{user_id}/preferences")
async def delete_all_user_preferences(user_id: str) -> Dict[str, Any]:
    """
    Delete all preferences for a user.
    
    Takes a user_id in the path and deletes all preferences for that user.
    """
    try:
        # Validate user_id format (UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid user ID format",
                    "message": "User ID must be a valid UUID"
                }
            )
        
        # Check if user exists
        user_check = users_db.supabase.table("users").select("user_id").eq("user_id", user_id).execute()
        
        if not hasattr(user_check, 'data') or not user_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "User not found",
                    "message": f"User with ID {user_id} does not exist"
                }
            )
        
        # Get count of current preferences
        count_result = preferences_db.supabase.table("user_preferences").select("preference_id").eq("user_id", user_id).execute()
        
        preferences_count = len(count_result.data) if hasattr(count_result, 'data') and count_result.data else 0
        
        # Delete all preferences for the user
        delete_result = preferences_db.supabase.table("user_preferences").delete().eq("user_id", user_id).execute()
        
        if hasattr(delete_result, 'error') and delete_result.error:
            logger.error(f"Database error deleting preferences for user {user_id}: {delete_result.error}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database error",
                    "message": "Failed to delete preferences"
                }
            )
        
        logger.info(f"Deleted {preferences_count} preferences for user {user_id}")
        
        return {
            "success": True,
            "user_id": user_id,
            "preferences_deleted": preferences_count,
            "message": f"Successfully deleted {preferences_count} preferences"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting preferences for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to delete preferences"
            }
        )


@app.delete("/users/{user_id}/preferences/{preference_id}")
async def delete_user_preference(user_id: str, preference_id: str) -> Dict[str, Any]:
    """
    Delete a specific user preference.
    
    Takes a user_id and preference_id in the path and deletes the specific preference.
    """
    try:
        # Validate user_id format (UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid user ID format",
                    "message": "User ID must be a valid UUID"
                }
            )
        
        # Validate preference_id format (UUID)
        try:
            uuid.UUID(preference_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid preference ID format",
                    "message": "Preference ID must be a valid UUID"
                }
            )
        
        # Check if user exists
        user_check = users_db.supabase.table("users").select("user_id").eq("user_id", user_id).execute()
        
        if not hasattr(user_check, 'data') or not user_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "User not found",
                    "message": f"User with ID {user_id} does not exist"
                }
            )
        
        # Check if preference exists and belongs to the user
        pref_check = preferences_db.supabase.table("user_preferences").select("preference_id, user_id, entity_id, entity_type").eq("preference_id", preference_id).eq("user_id", user_id).execute()
        
        if not hasattr(pref_check, 'data') or not pref_check.data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Preference not found",
                    "message": f"Preference with ID {preference_id} not found for user {user_id}"
                }
            )
        
        preference_data = pref_check.data[0]
        
        # Delete the specific preference
        delete_result = preferences_db.supabase.table("user_preferences").delete().eq("preference_id", preference_id).eq("user_id", user_id).execute()
        
        if hasattr(delete_result, 'error') and delete_result.error:
            logger.error(f"Database error deleting preference {preference_id}: {delete_result.error}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database error",
                    "message": "Failed to delete preference"
                }
            )
        
        logger.info(f"Deleted preference {preference_id} for user {user_id}")
        
        return {
            "success": True,
            "user_id": user_id,
            "preference_id": preference_id,
            "deleted_entity": {
                "entity_id": preference_data['entity_id'],
                "entity_type": preference_data['entity_type']
            },
            "message": "Preference deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting preference {preference_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "Failed to delete preference"
            }
        )


if __name__ == "__main__":
    # Configure logging for development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application with uvicorn
    logger.info("Starting FastAPI application in development mode...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
