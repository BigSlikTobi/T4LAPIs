# Scripts Directory

This directory contains CLI utility scripts for loading NFL data into the database.

## Available CLI Scripts

### `teams_cli.py`
Loads NFL teams data into the database.

```bash
python scripts/teams_cli.py [--dry-run] [--clear] [--verbose]
```

**Features:**
- Automatically checks for existing team data
- Won't overwrite if teams already exist (unless --clear is used)
- Supports dry-run mode to preview actions
- Uses the `TeamsDataLoader` class internally
- Provides clear logging and status messages

### `players_cli.py`
Loads NFL player data for a specific season into the database.

```bash
python scripts/players_cli.py 2024 [--dry-run] [--clear] [--verbose]
```

**Features:**
- Loads player data for specified season
- Uses upsert functionality to handle player updates
- Supports dry-run mode for testing
- Provides detailed progress information
- Shows final player count in database

### `games_cli.py`
Loads NFL game data into the database with advanced filtering options.

```bash
python scripts/games_cli.py 2024 [--week 1] [--dry-run] [--clear] [--verbose]
```

**Features:**
- Load games for entire season or specific week
- Supports dry-run mode for testing
- Clear existing data option
- Detailed progress and result reporting
- Shows sample data in dry-run mode

### `player_weekly_stats_cli.py`
Loads NFL player weekly stats data into the database.

```bash
python scripts/player_weekly_stats_cli.py 2024 [--weeks 1 2 3] [--batch-size 100] [--dry-run] [--clear]
```

**Features:**
- Load stats for multiple years and specific weeks
- Configurable batch size for large datasets
- Supports dry-run mode for testing
- Uses upsert functionality for stat updates
- Detailed progress tracking

## Common Options

All CLI scripts support these common options:

- `--dry-run`: Show what would be done without actually doing it
- `--clear`: Clear existing data before loading new data
- `--verbose, -v`: Enable verbose logging
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set specific logging level

## Usage Notes

### Prerequisites
- Ensure your `.env` file is configured with Supabase credentials
- Run from the project root directory

### Example Workflows

```bash
# Load teams (one-time setup)
python scripts/teams_cli.py --dry-run  # Preview first
python scripts/teams_cli.py           # Actually load

# Load players for current season
python scripts/players_cli.py 2024 --dry-run
python scripts/players_cli.py 2024

# Load games for specific week
python scripts/games_cli.py 2024 --week 1 --dry-run
python scripts/games_cli.py 2024 --week 1

# Load player stats for multiple weeks
python scripts/player_weekly_stats_cli.py 2024 --weeks 1 2 3 --dry-run
python scripts/player_weekly_stats_cli.py 2024 --weeks 1 2 3
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
