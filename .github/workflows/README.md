# GitHub Workflows for NFL Data API

This directory contains GitHub Actions workflows that automatically update NFL data in the database.

## Workflows Overview

### 1. Games Data Update (`games-data-update.yml`)
**Schedule**: Thursday, Friday, Saturday, Sunday, Monday, Tuesday at 5:00 AM UTC

This is the most frequent workflow as games happen throughout the week during NFL season.

**Features**:
- Automatically detects the latest week in the database
- Upserts the current week (to catch score updates) 
- Inserts the next week (for upcoming games)
- Uses the smart `games_auto_update.py` script

**Manual Trigger Options**:
- `season`: Specify a particular season
- `week`: Specify a particular week

### 2. Player Weekly Stats Update (`player-weekly-stats-update.yml`)
**Schedule**: Tuesday at 6:00 AM UTC (after Monday Night Football)

Updates player statistics after all games for the week are completed.

**Smart Features**:
- ✅ **Compares games table vs stats table** to find gaps
- ✅ **Updates recent weeks** (last 1-2 weeks for late stat corrections)
- ✅ **Processes only new weeks** that have games but no stats yet
- ✅ **Efficient processing** - doesn't reprocess entire seasons

**Manual Trigger Options**:
- `years`: NFL season years (space-separated)
- `weeks`: Specific weeks (optional)

### 3. Players Data Update (`players-data-update.yml`)
**Schedule**: Wednesday at 3:00 AM UTC (weekly roster updates)

Updates player roster information to catch trades, signings, and releases.

**Manual Trigger Options**:
- `season`: NFL season year

### 4. Teams Data Update (`teams-data-update.yml`)
**Schedule**: 1st of every month at 2:00 AM UTC

Updates team information (rarely changes but good to keep current).

**Manual Trigger**: No additional options needed

### 5. Complete Data Update (`complete-data-update.yml`)
**Schedule**: Manual trigger only

Runs a complete data refresh for a given season. Useful for:
- Initial setup
- Full data refresh
- Recovering from errors

**Manual Trigger Options**:
- `season`: NFL season year (required)
- `include_teams`: Update teams data (default: true)
- `include_players`: Update players data (default: true)
- `include_games`: Update games data (default: true)
- `include_stats`: Update player weekly stats (default: true)

## Required Secrets

Make sure the following secrets are configured in your GitHub repository:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key

## Schedule Summary

| Day | Time (UTC) | Workflow | Purpose |
|-----|------------|----------|---------|
| Monday | 05:00 | Games Data | Update Monday games |
| Tuesday | 05:00 | Games Data | Update Tuesday games (rare) |
| Tuesday | 06:00 | Player Stats | Update weekly stats after MNF |
| Wednesday | 03:00 | Players | Update rosters after waivers |
| Thursday | 05:00 | Games Data | Update Thursday games |
| Friday | 05:00 | Games Data | Update Friday games (rare) |
| Saturday | 05:00 | Games Data | Update Saturday games |
| Sunday | 05:00 | Games Data | Update Sunday games |
| 1st of month | 02:00 | Teams | Update team info |

## Smart Update Logic

### Games Auto-Update (`games_auto_update.py`)
The games workflow uses intelligent update logic:

1. **Auto-detect current NFL season** based on date
2. **Find latest week** in database for that season
3. **Upsert latest week** (catches score updates, status changes)
4. **Insert next week** (adds upcoming games)
5. **Skip invalid weeks** (beyond week 22)

Example:
- Database has: Season 2024, Week 5
- Script will: Upsert Week 5 AND Insert Week 6

### Player Stats Auto-Update (`player_weekly_stats_auto_update.py`)
The player stats workflow uses cross-table comparison:

1. **Check games table** for latest available week with games
2. **Check stats table** for latest week with player stats
3. **Identify gaps** between the two tables
4. **Update recent weeks** (last 1-2 weeks for stat corrections)
5. **Process new weeks** that have games but no stats yet

Example:
- Games table has: Weeks 1-6
- Stats table has: Weeks 1-4
- Script will: Update Weeks 4-6 (Week 4 for corrections, Weeks 5-6 as new data)

## Manual Triggers

All workflows can be manually triggered via the GitHub Actions UI:

1. Go to Actions tab in your repository
2. Select the workflow you want to run
3. Click "Run workflow"
4. Fill in any required inputs
5. Click "Run workflow"

## Monitoring

- Check the Actions tab for workflow run status
- Failed workflows will show error messages
- Each workflow includes descriptive logging
- Notification steps alert on failures

## Troubleshooting

Common issues:

1. **Missing secrets**: Ensure SUPABASE_URL and SUPABASE_KEY are set
2. **Python dependencies**: Check requirements.txt is up to date
3. **Database connection**: Verify Supabase credentials and permissions
4. **Data source issues**: nfl_data_py might have temporary outages
5. **Invalid seasons/weeks**: Check that requested data actually exists

## Development

To test workflows locally:

```bash
# Test the smart auto-update scripts
python scripts/games_auto_update.py
python scripts/player_weekly_stats_auto_update.py

# Test individual scripts
python scripts/games_cli.py 2024 --week 5
python scripts/players_cli.py 2024
python scripts/teams_cli.py
python scripts/player_weekly_stats_cli.py 2024 --weeks 1 2 3

# Run the test suites
python -m pytest tests/test_games_auto_update.py -v
python -m pytest tests/test_player_weekly_stats_auto_update.py -v

# Run all tests with coverage
python -m pytest tests/ -v --cov=src --cov=scripts
```

## Test Coverage

The auto-update scripts have comprehensive test coverage:

### Games Auto-Update Tests (`test_games_auto_update.py`)
- ✅ Database connection handling
- ✅ Season detection logic (all months)
- ✅ Latest week detection
- ✅ No existing data scenario
- ✅ Existing data scenarios
- ✅ Week boundary validation (skips > 22)
- ✅ Partial failure handling
- ✅ Exception handling

### Player Stats Auto-Update Tests (`test_player_weekly_stats_auto_update.py`)
- ✅ Cross-table comparisons (games vs stats)
- ✅ Gap detection logic
- ✅ Various data state scenarios
- ✅ Recent week update logic
- ✅ Edge cases (week 1, end of season)
- ✅ Invalid week handling
- ✅ Comprehensive error scenarios

Both test suites mock external dependencies (database, nfl_data_py) for reliable, fast testing.
