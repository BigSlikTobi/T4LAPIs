# Player Data Duplicate Key Resolution

## Problem Summary

The database was experiencing PostgreSQL errors when loading player data:
```
ERROR: ON CONFLICT DO UPDATE command cannot affect row a second time
```

This error occurred because:
1. NFL player data from `nfl_data_py` contains multiple records for the same player (due to roster changes, status updates, etc.)
2. Multiple player records with the same `player_id` were being sent to the database in a single batch
3. The composite key strategy (`stat_id`) was causing conflicts when the same player appeared multiple times

## Solution Implemented

### 1. Application-Level Deduplication

**File: `src/core/data/transform.py`**

Added deduplication logic to the `BaseDataTransformer` class:
- Added `_deduplicate_records()` method to the transformation pipeline
- Default implementation returns all records (no deduplication)
- Subclasses can override for custom deduplication logic

**PlayerDataTransformer Enhancement:**
- Override `_deduplicate_records()` to handle player duplicates
- Keep only the most recent record per `player_id` based on `last_active_season`
- Removes duplicate players before sending to database
- Logs the number of duplicates removed

### 2. Database-Level Conflict Resolution

**File: `src/core/data/loaders/base.py`**

Enhanced the base loader to use proper conflict resolution:
- For players table: Use `on_conflict="player_id"` parameter
- This enables PostgreSQL `ON CONFLICT (player_id) DO UPDATE` behavior
- If a player already exists in database, overwrite with latest information

**File: `src/core/utils/database.py`**

The database manager already supported the `on_conflict` parameter for upsert operations, but needed a fix for the Supabase Python client API.

**Fixed Supabase API Usage:**
- The `on_conflict` parameter is passed directly to the `upsert()` method
- Previous implementation incorrectly tried to chain `.on_conflict()` method
- Now uses: `supabase.table(table_name).upsert(json=records, on_conflict=column_name)`

## How It Works

### Step 1: Deduplication During Transformation
```python
def _deduplicate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate player records by player_id, keeping the latest record for each player.
    """
    deduplicated = {}
    
    for record in records:
        player_id = record.get('player_id')
        if player_id not in deduplicated:
            deduplicated[player_id] = record
        else:
            # Compare seasons and keep the more recent one
            existing_season = deduplicated[player_id].get('last_active_season') or 0
            current_season = record.get('last_active_season') or 0
            
            if current_season >= existing_season:
                deduplicated[player_id] = record
    
    return list(deduplicated.values())
```

### Step 2: Database Conflict Resolution
```python
if self.table_name == "players":
    # For players, use player_id as conflict resolution to implement overwrite strategy
    upsert_result = self.db_manager.upsert_records(transformed_records, on_conflict="player_id")
```

## Benefits

1. **Eliminates PostgreSQL Errors**: No more "cannot affect row a second time" errors
2. **Data Quality**: Ensures only the most recent player information is stored
3. **Performance**: Reduces database load by removing duplicates before insertion
4. **Backwards Compatible**: All existing tests pass, no breaking changes
5. **Overwrite Strategy**: If a player exists in database, it gets overwritten with latest data

## Fix Applied

### Issue Encountered
When testing the solution, encountered a Supabase API error:
```
'SyncQueryRequestBuilder' object has no attribute 'on_conflict'
```

### Resolution
Updated the database manager to use the correct Supabase Python client API:
- **Before:** `query.on_conflict(column_name)` (method chaining - incorrect)
- **After:** `upsert(json=records, on_conflict=column_name)` (parameter passing - correct)

This ensures the PostgreSQL `ON CONFLICT (player_id) DO UPDATE` behavior works properly.

## Testing

- ✅ All 130 existing tests pass
- ✅ Deduplication logic tested and verified
- ✅ Keeps most recent record based on season
- ✅ Handles players with no duplicates correctly

## Usage

The solution works automatically when loading player data:

```python
from src.core.data.loaders.players import PlayersDataLoader

loader = PlayersDataLoader()
result = loader.load_data(season=2024)  # Deduplication happens automatically
```

No changes required to existing scripts or workflows. The GitHub Actions workflows will now run without database conflicts.
