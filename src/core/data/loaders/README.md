# Teams Data Loader

This module provides functionality to load NFL team data from `nfl_data_py` into your Supabase database.

## Features

- ✅ **Modular Design**: Separate concerns for fetching, transforming, and loading
- ✅ **Data Validation**: Comprehensive validation of team records
- ✅ **Error Handling**: Robust error handling and logging
- ✅ **Flexible Options**: Support for dry-run, clearing existing data
- ✅ **Comprehensive Tests**: 100% test coverage with unit tests
- ✅ **Command Line Interface**: Easy-to-use CLI with options

## Database Schema

The loader populates the `teams` table with the following structure:

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

## Usage

### 1. Command Line Interface

#### Dry Run (Safe Testing)
```bash
# Test the data loading without actually inserting into database
python3 src/core/data/loaders/teams.py --dry-run
```

#### Load Fresh Data
```bash
# Load teams data (fails if teams already exist)
python3 src/core/data/loaders/teams.py
```

#### Replace Existing Data
```bash
# Clear existing teams and load fresh data
python3 src/core/data/loaders/teams.py --clear
```

#### Custom Table Name
```bash
# Load into a different table
python3 src/core/data/loaders/teams.py --table my_teams_table
```

### 2. Programmatic Usage

```python
from src.core.data.loaders.teams import TeamsDataLoader

# Create loader instance
loader = TeamsDataLoader()

# Check existing data
existing_count = loader.get_existing_teams_count()
print(f"Found {existing_count} existing teams")

# Load teams (with error handling)
try:
    success = loader.load_teams(clear_existing=False)
    if success:
        print("✅ Teams loaded successfully")
    else:
        print("❌ Failed to load teams")
except Exception as e:
    print(f"Error: {e}")
```

### 3. CLI Script

Use the comprehensive CLI script:

```bash
# Load teams data
python3 scripts/teams_cli.py

# Dry run to test without loading
python3 scripts/teams_cli.py --dry-run

# Clear existing data and reload
python3 scripts/teams_cli.py --clear
```

## Data Flow

```
NFL Data (nfl_data_py) 
    ↓
fetch_team_data() 
    ↓
TeamDataTransformer.transform() 
    ↓
Supabase Database
```

## Architecture

### 📁 File Structure

```
src/core/data/
├── fetch.py              # Data fetching from nfl_data_py
├── transform.py          # Data transformation and validation  
├── loaders/
│   ├── __init__.py
│   └── teams.py          # Teams data loader class
tests/
├── test_team_transform.py  # Transform function tests
└── test_teams_loader.py    # Loader class tests
scripts/
└── teams_cli.py           # CLI script for loading teams
```

### 🧩 Components

#### 1. **Data Fetching** (`fetch.py`)
- `fetch_team_data()` - Fetches raw team data from nfl_data_py

#### 2. **Data Transformation** (`transform.py`)
- `TeamDataTransformer` class - Handles all team data transformation, validation, and sanitization
- `PlayerDataTransformer` class - Handles all player data transformation, validation, and sanitization
- `BaseDataTransformer` abstract class - Provides common transformation framework

#### 3. **Data Loading** (`loaders/teams.py`)
- `TeamsDataLoader` class - Orchestrates the complete loading process
- Database operations (insert, clear, count)
- Error handling and logging

## Data Mapping

| nfl_data_py Field | Database Field | Notes |
|-------------------|----------------|-------|
| `team_abbr` | `team_abbr` | 2-3 character team abbreviation |
| `team_name` | `team_name` | Full team name |
| `team_conf` | `team_conference` | AFC or NFC |
| `team_division` | `team_division` | Division name |
| `team_id` | `team_nfl_data_py_id` | Converted to integer |
| `team_nick` | `team_nick` | Team nickname |
| `team_color` | `team_color` | Primary team color |
| `team_color2` | `team_color2` | Secondary team color |
| `team_color3` | `team_color3` | Tertiary team color |
| `team_color4` | `team_color4` | Quaternary team color |

## Validation Rules

- ✅ `team_abbr`: Must be 2-3 characters
- ✅ `team_name`: Cannot be empty
- ✅ `team_conference`: Must be "AFC" or "NFC"
- ✅ `team_division`: Cannot be empty
- ✅ `team_nfl_data_py_id`: Must be positive integer
- ✅ Color fields: Null values replaced with `#000000`

## Testing

Run the comprehensive test suite:

```bash
# Test data transformation functions
python3 -m pytest tests/test_team_transform.py -v

# Test loader class
python3 -m pytest tests/test_teams_loader.py -v

# Run all tests
python3 -m pytest tests/test_team* -v
```

### Test Coverage

- ✅ **12 tests** for data transformation
- ✅ **13 tests** for loader functionality
- ✅ **100% coverage** of core functionality
- ✅ **Error scenarios** tested
- ✅ **Edge cases** covered

## Error Handling

The loader includes comprehensive error handling:

1. **Connection Errors**: Validates Supabase client initialization
2. **Data Validation**: Filters invalid records with detailed logging
3. **Database Errors**: Handles insert/delete failures gracefully
4. **Missing Data**: Provides clear error messages for missing fields

## Logging

All operations are logged with appropriate levels:

```
INFO  - Normal operations and progress
WARN  - Invalid data records (skipped)
ERROR - Failures and exceptions
```

## Example Output

```
2025-07-27 17:06:36,656 - INFO - Fetching team data using nfl.import_team_desc()
2025-07-27 17:06:37,460 - INFO - Successfully fetched 36 team records
2025-07-27 17:06:37,462 - INFO - Transforming 36 team records
2025-07-27 17:06:37,464 - INFO - Successfully transformed 36 team records
2025-07-27 17:06:37,464 - INFO - Sanitizing 36 team records
2025-07-27 17:06:37,464 - INFO - Sanitized 36 valid team records
2025-07-27 17:06:37,465 - INFO - ✅ Teams data loading completed successfully
```

## Integration

This loader is designed to integrate seamlessly with your existing project:

```python
# Use with your existing database connection
from src.core.db.database_init import get_supabase_client
from src.core.data.loaders.teams import TeamsDataLoader

# Loader automatically uses your configured Supabase client
loader = TeamsDataLoader()
loader.load_teams()
```

## Next Steps

This teams loader serves as a template for creating loaders for other NFL data:

- 🔄 **Players Loader** - Load player roster data
- 🏟️ **Games Loader** - Load game schedule data  
- 📊 **Stats Loader** - Load player statistics
- 🏥 **Injuries Loader** - Load injury reports

The same modular architecture can be applied to all data types.
