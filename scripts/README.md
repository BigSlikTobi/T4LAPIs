# Scripts Directory

This directory contains simple utility scripts for loading NFL data into the database.

## Available Scripts

### `load_teams.py`
Loads NFL teams data into the database.

```bash
python scripts/load_teams.py
```

**Features:**
- Automatically checks for existing team data
- Won't overwrite if teams already exist
- Uses the `TeamsDataLoader` class internally
- Provides clear logging and status messages

### `load_players.py`
Loads NFL player data for the current season (2024) into the database.

```bash
python scripts/load_players.py
```

**Features:**
- Loads player data for season 2024
- Uses upsert functionality to handle player updates
- Provides detailed progress information
- Shows final player count in database

## Usage Notes

### Prerequisites
- Ensure your `.env` file is configured with Supabase credentials
- Run from the project root directory

### For Advanced Usage
For more control over the loading process (dry runs, clearing data, specific seasons), use the CLI interfaces directly:

```bash
# Teams with advanced options
python -m src.core.data.loaders.teams --clear --dry-run

# Players with advanced options  
python -m src.core.data.loaders.players 2023 --clear --dry-run --verbose
```

### Data Flow
Both scripts follow the same pattern:
1. Initialize Supabase client
2. Create appropriate data loader
3. Check existing data
4. Transform and validate data using class-based transformers
5. Load data into database
6. Report results

## Error Handling
- Scripts will exit with status code 1 on failure
- All operations are logged with timestamps
- Database connection issues are handled gracefully
