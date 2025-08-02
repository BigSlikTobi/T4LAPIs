# Data Loaders

This document describes the data loading system that handles inserting NFL data into the Supabase database with proper conflict resolution and error handling.

## 🔧 Loader Architecture

The data loading system follows a modular design with base classes and specialized implementations for different data types.

### Base Loader (`src/core/data/loaders/base.py`)

The `BaseDataLoader` class provides common functionality for all data loaders:

```python
class BaseDataLoader:
    """Base class for all data loaders with common functionality"""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.supabase = get_supabase_client()
        
    def load(self, df: pd.DataFrame, dry_run: bool = False, 
             clear_table: bool = False, on_conflict: str = None) -> bool:
        """Main loading pipeline with options"""
        
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate data before loading"""
        
    def _clear_table(self) -> bool:
        """Clear existing table data"""
        
    def _insert_data(self, records: List[Dict], on_conflict: str = None) -> bool:
        """Insert data with conflict resolution"""
```

### Design Patterns

1. **Template Method**: Base class defines the loading pipeline, subclasses customize validation
2. **Strategy Pattern**: Different conflict resolution strategies for different data types
3. **Command Pattern**: Loaders encapsulate loading operations with undo capability (clear table)

## 📊 Specialized Loaders

### 1. Teams Loader (`teams.py`)

Handles NFL team data loading with upsert behavior.

**Database Schema:**
```json
{
    "team_abbr": "ARI",
    "team_name": "Arizona Cardinals", 
    "team_conference": "NFC",
    "team_division": "NFC West",
    "team_nfl_data_py_id": 3800,
    "team_nick": "Cardinals",
    "team_color": "#97233F",
    "team_color2": "#000000",
    "team_color3": "#ffb612", 
    "team_color4": "#a5acaf"
}
```

**Features:**
- ✅ **Conflict Resolution**: Uses `team_abbr` as primary key
- ✅ **Data Validation**: Validates required fields and data types
- ✅ **Color Normalization**: Ensures hex color format
- ✅ **Historical Support**: Handles team relocations and name changes

**Usage:**
```python
from src.core.data.loaders.teams import TeamsLoader

loader = TeamsLoader()
success = loader.load(teams_df, dry_run=True)
```

### 2. Players Loader (`players.py`)

Handles NFL player roster data with advanced duplicate resolution.

**Key Features:**
- ✅ **Duplicate Handling**: Resolves multiple records per player
- ✅ **Team Normalization**: Handles historical team abbreviations
- ✅ **Season Tracking**: Maintains last active season information
- ✅ **Conflict Resolution**: Uses `player_id` as primary key

**Duplicate Resolution Strategy:**
```python
def _deduplicate_records(self, df: pd.DataFrame) -> pd.DataFrame:
    """Keep most recent record per player based on last_active_season"""
    return df.sort_values('last_active_season').drop_duplicates(
        subset=['player_id'], keep='last'
    )
```

### 3. Games Loader (`games.py`)

Handles NFL game schedule and results data.

**Features:**
- ✅ **Smart Updates**: Distinguishes between scheduled games and completed games
- ✅ **Score Validation**: Validates score data for completed games
- ✅ **Week Boundaries**: Validates week numbers (1-22)
- ✅ **Team Validation**: Ensures referenced teams exist

**Conflict Resolution:**
```python
# Games use upsert to handle score updates
loader.load(games_df, on_conflict="game_id")
```

### 4. Player Weekly Stats Loader (`player_weekly_stats.py`)

Handles weekly player performance statistics.

**Features:**
- ✅ **Composite Keys**: Uses `player_id_season_week` format
- ✅ **Statistics Validation**: Validates numeric statistics
- ✅ **Player Linking**: Ensures player references are valid
- ✅ **Performance Optimization**: Batch processing for large datasets

**Composite Key Generation:**
```python
def _generate_stat_id(self, row):
    """Generate composite key: player_id_season_week"""
    return f"{row['player_id']}_{row['season']}_{row['week']}"
```

## 🔄 Loading Operations

### Standard Loading Workflow

1. **Data Validation**: Check required fields and data types
2. **Data Transformation**: Apply loader-specific transformations
3. **Conflict Resolution**: Handle duplicate records appropriately
4. **Database Operation**: Insert or upsert data
5. **Error Handling**: Log errors and continue processing when possible

### Loading Options

#### Dry Run Mode
```python
# Test loading without database changes
loader.load(df, dry_run=True)
```

#### Clear and Reload
```python
# Replace all existing data
loader.load(df, clear_table=True)
```

#### Conflict Resolution
```python
# Upsert with specific conflict column
loader.load(df, on_conflict="primary_key_column")
```

## 🚨 Error Handling

### Error Categories

1. **Validation Errors**: Invalid data format or missing required fields
   - **Action**: Log warning, skip invalid records, continue processing
   
2. **Database Errors**: Connection issues or constraint violations
   - **Action**: Retry transient errors, fail on permanent errors
   
3. **Transformation Errors**: Issues during data processing
   - **Action**: Log error details, skip problematic records

### Error Recovery Strategies

```python
try:
    loader.load(data)
except ValidationError as e:
    logger.warning(f"Validation failed: {e}")
    # Continue with valid records
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    # Retry or fail depending on error type
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Log and re-raise unexpected errors
```

## 🔧 Configuration

### Database Connection

Loaders use the centralized database configuration:

```python
# Environment variables
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-key"

# Connection management
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
```

### Loader Settings

```python
# Default settings for all loaders
DEFAULT_BATCH_SIZE = 500      # Records per batch
DEFAULT_TIMEOUT = 30          # Database timeout (seconds)
DEFAULT_RETRY_ATTEMPTS = 3    # Retry count for transient errors
```

## 📊 Performance Optimization

### Batch Processing

```python
# Process large datasets in batches
batch_size = 500
for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    try:
        insert_batch(batch)
    except Exception as e:
        logger.error(f"Batch {i//batch_size + 1} failed: {e}")
        continue  # Continue with next batch
```

### Memory Management

```python
# Clear large DataFrames after processing
def load(self, df: pd.DataFrame) -> bool:
    try:
        records = df.to_dict('records')
        result = self._insert_data(records)
        del records  # Free memory
        return result
    finally:
        gc.collect()  # Force garbage collection
```

## 🧪 Testing

### Loader Testing Strategy

Each loader includes comprehensive tests covering:

1. **Normal Operations**: Successful loading scenarios
2. **Error Conditions**: Various failure modes
3. **Edge Cases**: Boundary conditions and special cases
4. **Performance**: Large dataset handling

### Test Examples

```python
def test_teams_loader_success():
    """Test successful team data loading"""
    loader = TeamsLoader()
    result = loader.load(sample_teams_df, dry_run=True)
    assert result is True

def test_players_loader_duplicates():
    """Test duplicate player handling"""
    # Create DataFrame with duplicate player_ids
    # Verify only latest record is kept
    
def test_games_loader_conflict_resolution():
    """Test game score updates via upsert"""
    # Load initial game data
    # Update scores and reload
    # Verify scores are updated, not duplicated
```

## 🔗 Integration Points

### CLI Integration

```python
# CLI tools use loaders directly
from src.core.data.loaders.teams import TeamsLoader

def main():
    args = parse_arguments()
    loader = TeamsLoader()
    
    # Fetch and transform data
    df = fetch_and_transform_data()
    
    # Load with CLI options
    success = loader.load(
        df, 
        dry_run=args.dry_run,
        clear_table=args.clear
    )
```

### Auto-Update Integration

```python
# Auto-update scripts use loaders with smart options
from src.core.data.loaders.games import GamesLoader

def auto_update_games():
    loader = GamesLoader()
    
    # Smart upsert for score updates
    current_week_games = fetch_current_week()
    loader.load(current_week_games, on_conflict="game_id")
    
    # Insert new games
    next_week_games = fetch_next_week()
    loader.load(next_week_games)
```

## 📋 Command-Line Usage

Each loader can be run independently for testing and manual operations:

### Teams Loader
```bash
# Dry run
python src/core/data/loaders/teams.py --dry-run

# Load data
python src/core/data/loaders/teams.py

# Clear and reload
python src/core/data/loaders/teams.py --clear
```

### Players Loader
```bash
# Load specific seasons
python src/core/data/loaders/players.py 2023 2024 --dry-run

# Clear and reload
python src/core/data/loaders/players.py 2024 --clear --dry-run
```

## 🔍 Monitoring and Logging

### Logging Output

```
2024-01-15 10:30:00 - TeamsLoader - INFO - Starting teams data load
2024-01-15 10:30:01 - TeamsLoader - INFO - Validated 32 team records
2024-01-15 10:30:02 - TeamsLoader - INFO - Successfully loaded 32 teams
2024-01-15 10:30:02 - TeamsLoader - INFO - Load completed in 2.1 seconds
```

### Metrics Tracked

- **Records Processed**: Count of total records
- **Records Inserted**: Count of successful insertions
- **Records Updated**: Count of updated records (upserts)
- **Records Skipped**: Count of invalid/duplicate records
- **Processing Time**: Total execution duration
- **Error Count**: Number of errors encountered

## 💡 Best Practices

### Development Guidelines

1. **Always Test with Dry Run**: Verify data before loading
2. **Handle Errors Gracefully**: Don't let single record failures stop entire loads
3. **Log Comprehensively**: Provide detailed logging for debugging
4. **Validate Input Data**: Check data quality before processing
5. **Use Appropriate Conflict Resolution**: Choose the right strategy for each data type

### Production Usage

1. **Monitor Loading Performance**: Track execution times and success rates
2. **Handle Large Datasets**: Use batch processing for large data loads
3. **Implement Retry Logic**: Handle transient database errors
4. **Validate Results**: Check record counts and data integrity after loading

For more information:
- [CLI Tools Guide](CLI_Tools_Guide.md) - Using loaders via CLI
- [Technical Details](Technical_Details.md) - Implementation details
- [Testing Guide](Testing_Guide.md) - Testing loader functionality
