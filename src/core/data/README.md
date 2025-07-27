# NFL Data Fetching Module

This module provides clean, separated functions for fetching NFL data using the `nfl_data_py` library. The goal is to separate data fetching from data transformation, providing a clear interface for retrieving raw data.

## Core Functions

### Required Functions (as requested)

#### 1. `fetch_team_data() -> pd.DataFrame`
Fetches team description data using `nfl.import_team_desc()`.

```python
from src.core.data.fetch import fetch_team_data

# Fetch all NFL team data
team_df = fetch_team_data()
print(f"Fetched {len(team_df)} teams")
```

#### 2. `fetch_player_data(years: List[int]) -> pd.DataFrame`
Fetches player roster data using `nfl.import_seasonal_rosters(years)`.

```python
from src.core.data.fetch import fetch_player_data

# Fetch player data for specific years
years = [2023, 2024]
player_df = fetch_player_data(years)
print(f"Fetched {len(player_df)} player records")
```

#### 3. `fetch_game_schedule_data(years: List[int]) -> pd.DataFrame`
Fetches game schedule data using `nfl.import_schedules(years)`.

```python
from src.core.data.fetch import fetch_game_schedule_data

# Fetch game schedules for specific years
years = [2023, 2024]
schedule_df = fetch_game_schedule_data(years)
print(f"Fetched {len(schedule_df)} games")
```

#### 4. `fetch_weekly_stats_data(years: List[int]) -> pd.DataFrame`
Fetches weekly player statistics using `nfl.import_weekly_data(years)`.

```python
from src.core.data.fetch import fetch_weekly_stats_data

# Fetch weekly stats for specific years
years = [2023, 2024]
weekly_stats_df = fetch_weekly_stats_data(years)
print(f"Fetched {len(weekly_stats_df)} weekly stat records")
```

## Additional Functions

The module also includes additional functions for comprehensive NFL data fetching:

### Roster Data
- `fetch_seasonal_roster_data(years)` - Seasonal roster data
- `fetch_weekly_roster_data(years)` - Weekly roster data

### Advanced Stats
- `fetch_pbp_data(years, downsampling=True)` - Play-by-play data
- `fetch_ngs_data(stat_type, years)` - Next Gen Stats data

### Injury and Draft Data
- `fetch_injury_data(years)` - Injury data
- `fetch_combine_data(years=None)` - NFL Combine data
- `fetch_draft_data(years=None)` - Draft pick data

## Usage Examples

### Basic Usage
```python
from src.core.data.fetch import (
    fetch_team_data,
    fetch_player_data,
    fetch_game_schedule_data,
    fetch_weekly_stats_data
)

# Fetch data for recent years
years = [2023, 2024]

# Get all required datasets
teams = fetch_team_data()
players = fetch_player_data(years)
games = fetch_game_schedule_data(years)
weekly_stats = fetch_weekly_stats_data(years)

print(f"Data fetched:")
print(f"- {len(teams)} teams")
print(f"- {len(players)} players")
print(f"- {len(games)} games")
print(f"- {len(weekly_stats)} weekly stats")
```

### Advanced Usage
```python
from src.core.data.fetch import fetch_ngs_data, fetch_pbp_data

# Fetch Next Gen Stats for passing
ngs_passing = fetch_ngs_data('passing', [2023])

# Fetch play-by-play data with downsampling
pbp_data = fetch_pbp_data([2023], downsampling=True)
```

## Error Handling

All functions include comprehensive error handling and logging:

```python
import logging

# Configure logging to see fetch progress
logging.basicConfig(level=logging.INFO)

try:
    team_df = fetch_team_data()
except Exception as e:
    print(f"Failed to fetch team data: {e}")
```

## Testing

Run the test suite to verify all functions work correctly:

```bash
python3 -m pytest tests/test_fetch.py -v
```

## Demo Scripts

Try the demo scripts to see the functions in action:

```bash
# Run the main demo (requires nfl_data_py to download data)
python3 examples/main_fetch_demo.py

# Run the comprehensive example
python3 examples/fetch_data_example.py
```

## File Structure

```
src/core/data/
├── __init__.py          # Module exports
├── fetch.py             # Main fetching functions
examples/
├── main_fetch_demo.py   # Demo of the 4 main functions
└── fetch_data_example.py # Comprehensive example
tests/
└── test_fetch.py        # Unit tests
```

## Dependencies

- `pandas` - Data manipulation
- `nfl_data_py` - NFL data source
- `logging` - Error handling and progress tracking

## Design Principles

1. **Separation of Concerns**: Fetching is separated from transformation
2. **Clear Interface**: Each function has a specific purpose
3. **Error Handling**: Comprehensive logging and exception handling
4. **Testability**: All functions are unit tested
5. **Documentation**: Clear docstrings and examples

## Integration with Existing Project

These functions can be easily integrated with your existing roster updates and injury updates:

```python
# In your existing scripts, replace direct nfl_data_py calls with:
from src.core.data.fetch import fetch_player_data, fetch_team_data

# Instead of: nfl.import_seasonal_rosters([2024])
player_data = fetch_player_data([2024])

# Instead of: nfl.import_team_desc()
team_data = fetch_team_data()
```
