# NFL Data Scripts

This directory contains command-line interface (CLI) scripts for fetching and loading NFL data into the database.

## Available Scripts

### Core Data Loading Scripts
- `games_cli.py` - Load NFL game data
- `players_cli.py` - Load NFL player data  
- `teams_cli.py` - Load NFL team data
- `player_weekly_stats_cli.py` - Load player weekly statistics

### Helper Scripts
- `debug_env.py` - Debug environment variable configuration
- `helper_scripts/explore_nfl_data.py` - Explore available NFL data
- `helper_scripts/generate_nfl_docs.py` - Generate documentation for NFL data

## Usage

### Local Environment
```bash
# Basic usage
python3 scripts/games_cli.py 2024 --week 1 --dry-run

# With verbose logging
python3 scripts/games_cli.py 2024 --week 1 --dry-run --verbose

# Clear table before loading
python3 scripts/games_cli.py 2024 --clear --dry-run
```

### Docker Environment
All scripts can be run in Docker using the `--env-file` option to load environment variables:

```bash
# Build the Docker image
docker build -t t4lapis-app .

# Run games CLI in Docker
docker run --rm --env-file .env t4lapis-app python3 scripts/games_cli.py 2022 --week 3 --dry-run

# Debug environment variables in Docker
docker run --rm --env-file .env t4lapis-app python3 scripts/debug_env.py

# Run with interactive shell for debugging
docker run --rm -it --env-file .env t4lapis-app bash
```

## Environment Variables

The scripts require the following environment variables to be set:

```bash
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-service-role-key"
```

### Docker Environment Variable Handling

When using Docker, environment variables should be defined in a `.env` file and loaded using `--env-file .env`. The application automatically handles:

- Quoted environment variables (both single and double quotes)
- URL validation and cleaning
- API key validation
- Detailed error messages for debugging

If you encounter issues with environment variables in Docker, use the debug script:
```bash
docker run --rm --env-file .env t4lapis-app python3 scripts/debug_env.py
```

## Common Arguments

All CLI scripts support these common arguments:

- `--dry-run` - Show what would be done without actually doing it
- `--clear` - Clear existing data before loading
- `--verbose` or `-v` - Enable verbose logging
- `--log-level` - Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Examples

### Load full season data
```bash
python3 scripts/games_cli.py 2024

# In Docker
docker run --rm --env-file .env t4lapis-app python3 scripts/games_cli.py 2024
```

### Load specific week with clearing
```bash
python3 scripts/games_cli.py 2024 --week 1 --clear --verbose

# In Docker
docker run --rm --env-file .env t4lapis-app python3 scripts/games_cli.py 2024 --week 1 --clear --verbose
```

### Debug environment setup
```bash
python3 scripts/debug_env.py

# In Docker
docker run --rm --env-file .env t4lapis-app python3 scripts/debug_env.py
```

## Troubleshooting

### "Invalid URL" Error in Docker
This usually means environment variables contain quotes. The application automatically handles this, but if you see this error:

1. Check your `.env` file format
2. Run the debug script: `docker run --rm --env-file .env t4lapis-app python3 scripts/debug_env.py`
3. Ensure your URLs start with `https://` and are properly formatted

### "Could not initialize Supabase client" Error
1. Verify your `.env` file contains valid `SUPABASE_URL` and `SUPABASE_KEY`
2. Check that you're using `--env-file .env` when running Docker
3. Run the debug script to verify environment variable loading
4. Ensure your Supabase credentials are not expired

### CI/Testing Mode
The application automatically detects CI environments and handles database connections accordingly. Set `CI=true` or `GITHUB_ACTIONS=true` to enable CI mode.

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
