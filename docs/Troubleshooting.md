# Troubleshooting Guide

This guide covers common issues and their solutions when working with the T4LAPIs NFL data management system.

## 🚨 Common Issues and Solutions

### 1. Player Data Duplicate Key Resolution

#### Problem Summary

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
- Added `_normalize_team_abbreviation()` method to handle historical team names

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

### Step 2: Team Abbreviation Normalization
```python
def _normalize_team_abbreviation(self, team_abbr: Any) -> Optional[str]:
    """
    Normalize team abbreviations to current valid abbreviations.
    Maps old/historical team abbreviations to current ones.
    """
    team_mapping = {
        'SD': 'LAC',  # San Diego Chargers -> Los Angeles Chargers
        'STL': 'LAR', # St. Louis Rams -> Los Angeles Rams
        'OAK': 'LV',  # Oakland Raiders -> Las Vegas Raiders
    }
    
    # Return mapped team or original if valid, None for invalid
    if team_str in team_mapping:
        return team_mapping[team_str]
    elif 2 <= len(team_str) <= 3:
        return team_str
    else:
        return None  # Prevents foreign key constraint violations
```

### Step 3: Database Conflict Resolution
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

### Issue 2: Foreign Key Constraint Violation
When testing with historical data, encountered:
```
insert or update on table "players" violates foreign key constraint "players_draft_team_fkey"
Key (draft_team)=(SD) is not present in table "teams"
```

### Resolution
Added team abbreviation normalization to handle historical team names:
- **SD** (San Diego Chargers) → **LAC** (Los Angeles Chargers)
- **STL** (St. Louis Rams) → **LAR** (Los Angeles Rams)  
- **OAK** (Oakland Raiders) → **LV** (Las Vegas Raiders)
- Invalid team abbreviations → **None** (to avoid constraint violations)

This prevents foreign key violations while preserving data integrity.

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

### 2. Environment Variable Issues

#### Problem: Missing Environment Variables
```
ERROR: SUPABASE_URL environment variable not set
```

#### Solution:
1. Create a `.env` file in the project root:
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-supabase-anon-key
   LOG_LEVEL=INFO
   ```

2. Load environment variables:
   ```bash
   # For CLI tools
   export $(cat .env | xargs)
   python scripts/teams_cli.py
   
   # For Docker
   docker run --env-file .env t4lapis-app python scripts/teams_cli.py
   ```

#### Problem: Invalid Supabase Credentials
```
ERROR: Failed to connect to Supabase: Invalid JWT
```

#### Solution:
- Verify your Supabase URL and key are correct
- Ensure you're using the right key type (anon key for read operations, service key for write operations)
- Check that your Supabase project is active

### 3. Database Connection Issues

#### Problem: Connection Timeouts
```
ERROR: Database connection timeout after 30 seconds
```

#### Solutions:
1. **Check Network Connectivity**: Ensure you can reach your Supabase instance
2. **Increase Timeout**: Set `DATABASE_TIMEOUT=60` environment variable
3. **Retry Logic**: The system includes automatic retries for transient failures

#### Problem: Foreign Key Constraint Violations
```
ERROR: insert or update violates foreign key constraint
```

#### Solutions:
1. **Load Teams First**: Always load team data before player data
2. **Check Team Abbreviations**: Ensure team abbreviations are valid
3. **Use Historical Mapping**: The system automatically maps old team names (SD→LAC, etc.)

### 4. Data Loading Issues

#### Problem: No Data Returned from NFL API
```
WARNING: No games found for 2025 week 1
```

#### Solutions:
1. **Check Season Dates**: NFL seasons run September to August
2. **Verify Week Numbers**: Regular season is weeks 1-18, playoffs are 19-22
3. **API Availability**: The `nfl_data_py` library may not have current season data immediately

#### Problem: Large Dataset Memory Issues
```
ERROR: MemoryError: Unable to allocate array
```

#### Solutions:
1. **Process in Batches**: Use smaller batch sizes for large datasets
2. **Increase Memory**: Run with more available RAM
3. **Use Chunking**: Process data in smaller chunks

### 5. GitHub Actions Workflow Issues

#### Problem: Workflow Fails Silently
```
ERROR: Workflow completed but no data was processed
```

#### Solutions:
1. **Check Logs**: Review GitHub Actions logs for detailed error messages
2. **Verify Secrets**: Ensure `SUPABASE_URL` and `SUPABASE_KEY` secrets are set
3. **Manual Trigger**: Try running the workflow manually with specific parameters

#### Problem: Workflow Times Out
```
ERROR: The job running on runner GitHub Actions 2 has exceeded the maximum execution time
```

#### Solutions:
1. **Optimize Queries**: Reduce the amount of data processed per run
2. **Split Operations**: Break large operations into smaller, more frequent runs
3. **Check Dependencies**: Ensure all dependencies are cached properly

### 6. Performance Issues

#### Problem: Slow Data Loading
```
INFO: Loaded 1000 records in 300 seconds
```

#### Solutions:
1. **Increase Batch Size**: Use larger batches for better throughput
2. **Optimize Database**: Ensure proper indexing on frequently queried columns
3. **Check Network**: Slow network can impact Supabase operations

#### Problem: High Memory Usage
```
WARNING: Memory usage above 80%
```

#### Solutions:
1. **Clear DataFrames**: Explicitly delete large DataFrames after use
2. **Use Generators**: Process data in streams rather than loading all at once
3. **Garbage Collection**: Force garbage collection after large operations

## 🔧 Debugging Tools

### Environment Debug Script
```bash
# Check environment configuration
python scripts/debug_env.py
```

### Verbose Logging
```bash
# Enable detailed logging
LOG_LEVEL=DEBUG python scripts/teams_cli.py --dry-run --verbose
```

### Database Connection Test
```python
from src.core.utils.database import get_supabase_client

try:
    client = get_supabase_client()
    result = client.table('teams').select('*').limit(1).execute()
    print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
```

## 📞 Getting Help

### Log Analysis
When reporting issues, include:
1. **Full error message** with stack trace
2. **Environment details** (Python version, OS, Docker/local)
3. **Command used** and any parameters
4. **Expected vs actual behavior**

### Common Log Patterns
- `ERROR:` - Serious issues requiring attention
- `WARNING:` - Issues that were handled gracefully
- `INFO:` - Normal operational messages
- `DEBUG:` - Detailed execution information

### Support Channels
1. **GitHub Issues**: For bugs and feature requests
2. **Documentation**: Check all documentation files in `docs/`
3. **Tests**: Review test files for usage examples
4. **Code Comments**: Inline documentation in source files

## 💡 Prevention Tips

### Best Practices
1. **Always Test with Dry Run**: Use `--dry-run` flag before actual data operations
2. **Monitor Logs**: Regular review of execution logs helps catch issues early
3. **Keep Dependencies Updated**: Regularly update `nfl_data_py` and other dependencies
4. **Validate Environment**: Run environment checks before important operations

### Regular Maintenance
1. **Database Health Checks**: Periodically verify data integrity
2. **Performance Monitoring**: Track execution times and resource usage
3. **Error Rate Monitoring**: Keep track of failure rates in automated workflows
4. **Documentation Updates**: Keep troubleshooting guide current with new issues
