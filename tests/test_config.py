"""
Test configuration and utilities for API testing
"""

import pytest
import os
import sys

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test configuration
VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"
VALID_PREFERENCE_ID = "123e4567-e89b-12d3-a456-426614174000"
INVALID_UUID = "not-a-valid-uuid"

# Test data templates
VALID_USER_PREFERENCE = {
    "entity_id": "KC",
    "type": "team"
}

VALID_PLAYER_PREFERENCE = {
    "entity_id": "00-0033873",
    "type": "player"
}

INVALID_PREFERENCE = {
    "entity_id": "KC",
    "type": "invalid_type"
}

PREFERENCE_UPDATE_DATA = {
    "entity_id": "TB",
    "type": "team"
}

MULTIPLE_PREFERENCES = {
    "entities": [
        VALID_USER_PREFERENCE,
        VALID_PLAYER_PREFERENCE
    ]
}

# Mock response templates
MOCK_USER_RESPONSE = {
    "user_id": VALID_UUID,
    "created_at": "2024-01-01T00:00:00Z"
}

MOCK_PREFERENCE_RESPONSE = {
    "preference_id": VALID_PREFERENCE_ID,
    "user_id": VALID_UUID,
    "entity_id": "KC",
    "entity_type": "team",
    "created_at": "2024-01-01T00:00:00Z"
}
