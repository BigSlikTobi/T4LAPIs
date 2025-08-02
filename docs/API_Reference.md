# API Endpoints Summary

## Complete CRUD Operations for User & Preference API

### Overview
This API provides comprehensive CRUD (Create, Read, Update, Delete) operations for managing users and their NFL data preferences.

### Base URL
- Local Development: `http://localhost:8000`
- Docker: `http://localhost:8080`

### API Endpoints

#### User Management

1. **Create User**
   - **POST** `/users/`
   - **Description**: Creates a new user with a unique UUID
   - **Response**: `201 Created`
   ```json
   {
     "user_id": "550e8400-e29b-41d4-a716-446655440000",
     "created_at": "2024-01-01T00:00:00Z"
   }
   ```

2. **Delete User**
   - **DELETE** `/users/{user_id}`
   - **Description**: Deletes a user and all associated preferences (CASCADE)
   - **Response**: `200 OK`
   ```json
   {
     "success": true,
     "user_id": "550e8400-e29b-41d4-a716-446655440000",
     "message": "User deleted successfully"
   }
   ```

#### Preference Management

3. **Set User Preferences**
   - **POST** `/users/{user_id}/preferences`
   - **Description**: Sets/replaces all preferences for a user
   - **Request Body**:
   ```json
   {
     "entities": [
       {"entity_id": "KC", "type": "team"},
       {"entity_id": "00-0033873", "type": "player"}
     ]
   }
   ```
   - **Response**: `200 OK`

4. **Get User Preferences**
   - **GET** `/users/{user_id}/preferences`
   - **Description**: Retrieves all preferences for a user
   - **Response**: `200 OK`
   ```json
   {
     "user_id": "550e8400-e29b-41d4-a716-446655440000",
     "preferences": [
       {
         "preference_id": "123e4567-e89b-12d3-a456-426614174000",
         "entity_id": "KC",
         "entity_type": "team",
         "created_at": "2024-01-01T00:00:00Z"
       }
     ],
     "total_count": 1
   }
   ```

5. **Update Specific Preference**
   - **PUT** `/users/{user_id}/preferences/{preference_id}`
   - **Description**: Updates a specific user preference
   - **Request Body**:
   ```json
   {
     "entity_id": "TB",
     "type": "team"
   }
   ```
   - **Response**: `200 OK`

6. **Delete All User Preferences**
   - **DELETE** `/users/{user_id}/preferences`
   - **Description**: Deletes all preferences for a user
   - **Response**: `200 OK`
   ```json
   {
     "success": true,
     "user_id": "550e8400-e29b-41d4-a716-446655440000",
     "preferences_deleted": 3,
     "message": "Successfully deleted 3 preferences for user"
   }
   ```

7. **Delete Specific Preference**
   - **DELETE** `/users/{user_id}/preferences/{preference_id}`
   - **Description**: Deletes a specific user preference
   - **Response**: `200 OK`
   ```json
   {
     "success": true,
     "preference_id": "123e4567-e89b-12d3-a456-426614174000",
     "deleted_entity": {
       "entity_id": "KC",
       "entity_type": "team"
     },
     "message": "Preference deleted successfully"
   }
   ```

### Entity Types
- `team`: NFL team (entity_id should be team abbreviation like "KC", "TB")
- `player`: NFL player (entity_id should be player ID like "00-0033873")

### Error Responses

#### 400 Bad Request
```json
{
  "detail": {
    "error": "Invalid user ID format",
    "provided_id": "not-a-valid-uuid"
  }
}
```

#### 404 Not Found
```json
{
  "detail": {
    "error": "User not found",
    "user_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

#### 422 Validation Error
```json
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body", "entities", 0, "type"],
      "msg": "Input should be 'team' or 'player'",
      "input": "invalid_type"
    }
  ]
}
```

#### 500 Internal Server Error
```json
{
  "detail": {
    "error": "Database connection failed",
    "message": "Please check database configuration"
  }
}
```

### Testing

#### Manual Testing with curl

1. **Create a user**:
```bash
curl -X POST http://localhost:8000/users/
```

2. **Set preferences**:
```bash
curl -X POST http://localhost:8000/users/{user_id}/preferences \
  -H "Content-Type: application/json" \
  -d '{"entities": [{"entity_id": "KC", "type": "team"}]}'
```

3. **Get preferences**:
```bash
curl http://localhost:8000/users/{user_id}/preferences
```

4. **Update preference**:
```bash
curl -X PUT http://localhost:8000/users/{user_id}/preferences/{preference_id} \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "TB", "type": "team"}'
```

5. **Delete specific preference**:
```bash
curl -X DELETE http://localhost:8000/users/{user_id}/preferences/{preference_id}
```

6. **Delete all preferences**:
```bash
curl -X DELETE http://localhost:8000/users/{user_id}/preferences
```

7. **Delete user**:
```bash
curl -X DELETE http://localhost:8000/users/{user_id}
```

#### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Features

✅ **Complete CRUD Operations**
- User creation and deletion
- Preference management (create, read, update, delete)
- Bulk operations (set all preferences, delete all preferences)

✅ **Data Validation**
- UUID format validation for all IDs
- Entity type validation (team/player)
- Request body validation with Pydantic models

✅ **Error Handling**
- Comprehensive error responses with appropriate HTTP status codes
- Database error handling with fallback messages
- User-friendly error messages with context

✅ **Database Integration**
- Supabase integration via existing DatabaseManager
- CASCADE deletion (deleting user removes all preferences)
- Transaction-like operations for data consistency

✅ **API Documentation**
- Auto-generated OpenAPI documentation
- Interactive testing interface
- Comprehensive endpoint descriptions

### Production Ready Features
- Docker containerization
- Health checks
- Structured logging
- Security best practices
- Comprehensive test coverage
