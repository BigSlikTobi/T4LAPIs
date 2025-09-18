# NFL Injury and Roster Data Loaders

This directory contains the new unified data loaders for NFL injury and roster data, replacing the standalone scripts in `injury_updates/` and `roster_updates/`.

## Features

- **Standardized Architecture**: Follows the same patterns as other core data loaders (games, players, teams)
- **nfl_data_py Integration**: Uses the canonical data source for consistency
- **Comprehensive Validation**: Field validation, data type checking, and error handling
- **Versioning Support**: Track data snapshots over time
- **Batch Processing**: Configurable batch sizes for large datasets
- **Dry Run Mode**: Test operations without affecting the database
- **CLI Interface**: Consistent command-line interface across all loaders

## Usage

### Loading Injury Data

```bash
# Load current season injury data
python scripts/load_injuries.py --years 2024

# Load multiple years with specific version
python scripts/load_injuries.py --years 2023 2024 --version 5

# Dry run to see what would be loaded
python scripts/load_injuries.py --years 2024 --dry-run

# Clear existing data and load fresh
python scripts/load_injuries.py --years 2024 --clear

# Custom batch size for large datasets
python scripts/load_injuries.py --years 2024 --batch-size 500
```

### Loading Roster Data

```bash
# Load current season roster data
python scripts/load_rosters.py --years 2024

# Load multiple years with specific version
python scripts/load_rosters.py --years 2023 2024 --version 3

# Dry run to see what would be loaded
python scripts/load_rosters.py --years 2024 --dry-run

# Verbose logging for debugging
python scripts/load_rosters.py --years 2024 --verbose
```

## Data Sources

Both loaders use `nfl_data_py` as the canonical data source:

- **Injuries**: `nfl.import_injuries(years)`
- **Rosters**: `nfl.import_seasonal_rosters(years)`

This ensures consistency with other data loaders and eliminates the need for custom API handling, SSL configuration, or manual request management.

## Versioning

Both loaders support versioning to track data snapshots:

- If no version is specified, the system auto-calculates the next version (current max + 1)
- Versions allow you to track changes over time and support incremental updates
- Use `--version N` to specify a specific version number

## Migration from Old Scripts

### From `injury_updates/main.py`

**Old:**
```bash
cd injury_updates
python main.py
```

**New:**
```bash
python scripts/load_injuries.py --years 2024
```

### From `roster_updates/main.py`

**Old:**
```bash
cd roster_updates  
python main.py
```

**New:**
```bash
python scripts/load_rosters.py --years 2024
```

## Database Schema

The loaders work with the existing database schema:

### Injuries Table
- Primary fields: `gsis_id`, `season`, `week`, `team`
- Supports both report and practice injury status
- Includes injury types, player information, and modification dates

### Rosters Table  
- Primary fields: `player_id`, `season`, `team`
- Comprehensive player information including physical stats
- Multiple ID mappings (ESPN, Yahoo, PFF, etc.)
- Position and status information

## Error Handling

The loaders include comprehensive error handling:

- **Field Validation**: Ensures required fields are present and properly formatted
- **Data Type Checking**: Validates numeric ranges, date formats, etc.
- **Graceful Degradation**: Skips invalid records with detailed logging
- **Transaction Safety**: Uses proper upsert operations with conflict resolution

## Testing

Unit tests are provided for both loaders:

```bash
# Test injury loader and transformer
python -m pytest tests/test_injuries_loader.py -v

# Test roster loader and transformer  
python -m pytest tests/test_rosters_loader.py -v
```

The tests cover:
- Data transformation logic
- Validation rules
- Error handling scenarios
- Dry run functionality