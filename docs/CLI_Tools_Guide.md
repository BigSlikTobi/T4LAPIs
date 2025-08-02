# CLI Tools Guide

This guide covers all command-line interface tools available in the T4LAPIs system for manual NFL data operations.

## 📁 Available CLI Tools

All CLI tools are located in the `scripts/` directory and provide consistent interfaces for data operations.

### Core Data Loading Scripts
- **`teams_cli.py`** - Load NFL team data
- **`players_cli.py`** - Load NFL player data  
- **`games_cli.py`** - Load NFL game data
- **`player_weekly_stats_cli.py`** - Load player weekly statistics

### Auto-Update Scripts
- **`games_auto_update.py`** - Smart game data updates (used by GitHub Actions)
- **`player_weekly_stats_auto_update.py`** - Smart stats updates (used by GitHub Actions)

### Helper Scripts
- **`debug_env.py`** - Debug environment variable configuration
- **`helper_scripts/explore_nfl_data.py`** - Explore available NFL data
- **`helper_scripts/generate_nfl_docs.py`** - Generate documentation for NFL data

## 🛠️ CLI Tool Usage

### Common Options

All CLI tools support these common options:

- `--dry-run` - Test without actually inserting data into database
- `--verbose` - Enable verbose logging output
- `--clear` - Clear existing table data before loading (destructive!)

### Environment Setup

CLI tools require environment variables:

```bash
# Load from .env file (recommended)
export $(cat .env | xargs)

# Or set directly
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_anon_key"
export LOG_LEVEL="INFO"  # Optional: DEBUG, INFO, WARNING, ERROR
```

## 📋 Individual Tool Documentation

### 1. Teams CLI (`teams_cli.py`)

Load NFL team information into the database.

```bash
# Basic usage - dry run (safe testing)
python scripts/teams_cli.py --dry-run

# Load team data
python scripts/teams_cli.py

# Replace existing data
python scripts/teams_cli.py --clear

# Verbose output
python scripts/teams_cli.py --dry-run --verbose
```

**Features:**
- Loads 32 NFL team records
- Includes team colors, divisions, conferences
- Safe to run multiple times (upsert behavior)

### 2. Players CLI (`players_cli.py`)

Load NFL player roster data for specified seasons.

```bash
# Load players for 2024 season (dry run)
python scripts/players_cli.py 2024 --dry-run

# Load multiple seasons
python scripts/players_cli.py 2023 2024 --dry-run

# Load with table clearing
python scripts/players_cli.py 2024 --clear --dry-run

# Verbose logging
python scripts/players_cli.py 2024 --dry-run --verbose
```

**Arguments:**
- `seasons` - One or more NFL season years (required)

**Features:**
- Handles player duplicates automatically
- Normalizes team abbreviations
- Comprehensive player information including position, college, physical stats

### 3. Games CLI (`games_cli.py`)

Load NFL game schedule and results data.

```bash
# Load 2024 season games (dry run)
python scripts/games_cli.py 2024 --dry-run

# Load specific week only
python scripts/games_cli.py 2024 --week 1 --dry-run

# Load multiple seasons
python scripts/games_cli.py 2023 2024 --dry-run

# Clear and reload
python scripts/games_cli.py 2024 --clear --dry-run
```

**Arguments:**
- `seasons` - One or more NFL season years (required)

**Options:**
- `--week WEEK` - Load only specific week number (1-22)

**Features:**
- Game schedules and results
- Team matchups, scores, dates
- Playoff and regular season games

### 4. Player Weekly Stats CLI (`player_weekly_stats_cli.py`)

Load weekly player performance statistics.

```bash
# Load 2024 season stats (dry run)
python scripts/player_weekly_stats_cli.py 2024 --dry-run

# Load specific weeks
python scripts/player_weekly_stats_cli.py 2024 --weeks 1 2 3 --dry-run

# Load multiple seasons
python scripts/player_weekly_stats_cli.py 2023 2024 --dry-run

# Clear and reload
python scripts/player_weekly_stats_cli.py 2024 --clear --dry-run
```

**Arguments:**
- `seasons` - One or more NFL season years (required)

**Options:**
- `--weeks WEEKS` - Specific week numbers to load (space-separated)

**Features:**
- Comprehensive player statistics (passing, rushing, receiving, etc.)
- Weekly granularity for season-long analysis
- Links to player and game data

## 🐳 Docker Usage

All CLI tools can be run in Docker for consistent environments:

```bash
# Build the Docker image
docker build -t t4lapis-app .

# Run any CLI tool in Docker
docker run --rm --env-file .env t4lapis-app python scripts/teams_cli.py --dry-run

# Interactive shell for debugging
docker run --rm -it --env-file .env t4lapis-app bash
```

### Docker Examples

```bash
# Teams data
docker run --rm --env-file .env t4lapis-app python scripts/teams_cli.py --dry-run

# Players for 2024
docker run --rm --env-file .env t4lapis-app python scripts/players_cli.py 2024 --dry-run

# Games for specific week
docker run --rm --env-file .env t4lapis-app python scripts/games_cli.py 2024 --week 3 --dry-run

# Debug environment
docker run --rm --env-file .env t4lapis-app python scripts/debug_env.py
```

## 🔍 Helper Scripts

### Debug Environment (`debug_env.py`)

Validates your environment configuration:

```bash
python scripts/debug_env.py
```

**Output includes:**
- Environment variable status
- Supabase connection test
- Database access verification

### Explore NFL Data (`helper_scripts/explore_nfl_data.py`)

Interactive exploration of available NFL datasets:

```bash
python scripts/helper_scripts/explore_nfl_data.py
```

**Features:**
- Lists all available nfl_data_py functions
- Shows data shapes and column information
- Test data fetching capabilities

### Generate Documentation (`helper_scripts/generate_nfl_docs.py`)

Generates comprehensive documentation about NFL data:

```bash
python scripts/helper_scripts/generate_nfl_docs.py
```

**Output:**
- Complete function reference
- Column details for all datasets
- Data shape information

## ⚡ Auto-Update Scripts

These scripts provide intelligent data updating logic used by GitHub Actions:

### Games Auto-Update (`games_auto_update.py`)

Smart game data updating with automatic week detection:

```bash
python scripts/games_auto_update.py
```

**Features:**
- Automatically detects current NFL season
- Finds latest week in database
- Upserts current week + inserts next week
- Handles season boundaries intelligently

### Player Stats Auto-Update (`player_weekly_stats_auto_update.py`)

Intelligent player statistics updating:

```bash
python scripts/player_weekly_stats_auto_update.py
```

**Features:**
- Compares games table vs stats table
- Identifies missing weeks
- Updates recent weeks for late corrections
- Processes only new data

## 🚨 Error Handling

All CLI tools include comprehensive error handling:

- **Connection Issues**: Clear error messages for database connectivity
- **Data Validation**: Validation of input parameters and data integrity
- **Partial Failures**: Continue processing on recoverable errors
- **Logging**: Detailed logs for debugging and monitoring

### Common Error Scenarios

1. **Missing Environment Variables**:
   ```bash
   ERROR: SUPABASE_URL environment variable not set
   ```
   
2. **Database Connection Failure**:
   ```bash
   ERROR: Failed to connect to Supabase
   ```

3. **Invalid Season Year**:
   ```bash
   ERROR: Invalid season year: 2025 (must be between 1999-2024)
   ```

## 💡 Best Practices

### Development Workflow

1. **Always start with dry-run**:
   ```bash
   python scripts/teams_cli.py --dry-run --verbose
   ```

2. **Test with small datasets first**:
   ```bash
   python scripts/games_cli.py 2024 --week 1 --dry-run
   ```

3. **Use verbose logging for debugging**:
   ```bash
   python scripts/players_cli.py 2024 --dry-run --verbose
   ```

### Production Usage

1. **Use environment files**:
   ```bash
   # Set up .env file first
   python scripts/teams_cli.py
   ```

2. **Monitor logs**:
   ```bash
   python scripts/games_cli.py 2024 2>&1 | tee games_load.log
   ```

3. **Validate results**:
   ```bash
   # Check record counts after loading
   python scripts/debug_env.py
   ```

## 🔗 Integration with Automation

These CLI tools serve as the foundation for automated workflows:

- **GitHub Actions** use auto-update scripts for scheduled runs
- **Local development** uses standard CLI tools for testing
- **Docker deployments** provide consistent environments
- **Manual operations** use CLI tools for one-off tasks

For automation details, see: [Automation Workflows](Automation_Workflows.md)
