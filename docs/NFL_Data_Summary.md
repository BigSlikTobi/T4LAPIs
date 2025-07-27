# NFL Data Python - Tables and Columns Summary

*Generated on: 2025-07-26 22:14:56*

## Overview

The `nfl_data_py` library provides access to **24 different NFL datasets** with comprehensive coverage of teams, players, games, statistics, and more. All functions successfully return data, providing rich datasets for analysis.

## 📊 Dataset Summary

| Category | Count | Total Columns | Key Functions |
|----------|-------|---------------|---------------|
| **Core Tables** | 4 | 152 | Teams, Players, Games, Stats |
| **Advanced Stats** | 8 | 425 | NGS, PBP, PFF, Seasonal/Weekly |
| **Draft & Personnel** | 6 | 158 | Draft, Combine, Contracts, Depth Charts |
| **Specialized Data** | 6 | 106 | Injuries, Officials, Lines, FTN |

**Total: 841+ columns across 24 datasets**

## 🎯 Core Tables (Your Primary Functions)

### 1. Teams Table - `import_team_desc()`
- **Shape:** 36 teams × 16 columns
- **Usage:** `nfl.import_team_desc()`
- **Key Columns:** `team_abbr`, `team_name`, `team_id`, `team_conf`, `team_division`, `team_color`

### 2. Players Table - `import_seasonal_rosters(years)`
- **Shape:** 3,089 players × 37 columns (2023 data)
- **Usage:** `nfl.import_seasonal_rosters([2023, 2024])`
- **Key Columns:** `player_name`, `team`, `position`, `jersey_number`, `height`, `weight`, `college`, `player_id`

### 3. Games Table - `import_schedules(years)`
- **Shape:** 285 games × 46 columns (2023 data)
- **Usage:** `nfl.import_schedules([2023])`
- **Key Columns:** `game_id`, `season`, `week`, `gameday`, `away_team`, `home_team`, `away_score`, `home_score`

### 4. Player Weekly Stats - `import_weekly_data(years)`
- **Shape:** 5,653 records × 53 columns (2023 data)
- **Usage:** `nfl.import_weekly_data([2023])`
- **Key Columns:** `player_id`, `player_name`, `recent_team`, `season`, `week`, `passing_yards`, `rushing_yards`, `receiving_yards`

## 📈 Advanced Statistics Tables

### Play-by-Play Data - `import_pbp_data(years)`
- **Shape:** 49,665 plays × 390 columns
- **Description:** Complete play-by-play data with advanced metrics
- **Key Features:** Down/distance, field position, EPA, WPA, air yards

### Next Gen Stats - `import_ngs_data(stat_type, years)`
- **Shape:** 620+ records × 29 columns
- **Stat Types:** 'passing', 'rushing', 'receiving'
- **Key Features:** Speed, acceleration, separation, completion probability

### Snap Counts - `import_snap_counts(years)`
- **Shape:** 26,513 records × 16 columns
- **Description:** Player snap count data by game

## 👥 Personnel & Draft Tables

### Draft Picks - `import_draft_picks()`
- **Shape:** 12,670 picks × 36 columns
- **Coverage:** Complete draft history with career stats

### NFL Combine - `import_combine_data()`
- **Shape:** 8,649 results × 18 columns
- **Metrics:** 40-yard dash, bench press, vertical jump, broad jump

### Contracts - `import_contracts()`
- **Shape:** 46,841 contracts × 25 columns
- **Details:** Contract values, guarantees, salary cap percentages

### Depth Charts - `import_depth_charts(years)`
- **Shape:** 37,327 entries × 15 columns
- **Description:** Weekly team depth chart positions

## 🏥 Injury & Health Data

### Injuries - `import_injuries(years)`
- **Shape:** 5,599 reports × 16 columns
- **Details:** Injury reports, status, body parts affected

## 🎮 Advanced Analytics

### Football Study Hall (FTN) - `import_ftn_data(years)`
- **Shape:** 48,225 plays × 29 columns
- **Features:** Formation data, personnel groupings

### Pro Football Reference - `import_seasonal_pfr(s_type, years)` / `import_weekly_pfr(s_type, years)`
- **Stat Types:** 'pass', 'rec', 'rush'
- **Description:** Advanced PFR statistics

## 📋 Complete Function List

| Function | Rows | Cols | Description |
|----------|------|------|-------------|
| `import_team_desc()` | 36 | 16 | Team information and colors |
| `import_seasonal_rosters(years)` | 3,089 | 37 | End-of-season rosters |
| `import_weekly_rosters(years)` | 45,655 | 37 | Weekly roster changes |
| `import_schedules(years)` | 285 | 46 | Game schedules and results |
| `import_weekly_data(years)` | 5,653 | 53 | Weekly player statistics |
| `import_seasonal_data(years)` | 588 | 58 | Season-long player stats |
| `import_pbp_data(years)` | 49,665 | 390 | Play-by-play data |
| `import_ngs_data(type, years)` | 620 | 29 | Next Gen Stats |
| `import_snap_counts(years)` | 26,513 | 16 | Player snap counts |
| `import_injuries(years)` | 5,599 | 16 | Injury reports |
| `import_draft_picks()` | 12,670 | 36 | Draft pick history |
| `import_combine_data()` | 8,649 | 18 | NFL Combine results |
| `import_contracts()` | 46,841 | 25 | Player contracts |
| `import_depth_charts(years)` | 37,327 | 15 | Team depth charts |
| `import_officials(years)` | 1,993 | 5 | Game officials |
| `import_qbr(years)` | 82 | 23 | ESPN QBR data |
| `import_ftn_data(years)` | 48,225 | 29 | Formation/personnel data |
| `import_players()` | 24,509 | 39 | Player database |
| `import_ids()` | 12,075 | 35 | Player ID mappings |
| `import_seasonal_pfr(type, years)` | 104 | 28 | PFR seasonal stats |
| `import_weekly_pfr(type, years)` | 700 | 24 | PFR weekly stats |
| `import_draft_values()` | 0 | 6 | Draft pick value models |
| `import_sc_lines(years)` | 0 | 7 | Sports betting lines |
| `import_win_totals(years)` | 0 | 9 | Season win totals |

## 🚀 Integration Examples

### Your Existing Fetch Functions
```python
from src.core.data.fetch import (
    fetch_team_data,           # -> nfl.import_team_desc()
    fetch_player_data,         # -> nfl.import_seasonal_rosters(years)
    fetch_game_schedule_data,  # -> nfl.import_schedules(years)
    fetch_weekly_stats_data    # -> nfl.import_weekly_data(years)
)

# Fetch core data
teams = fetch_team_data()
players = fetch_player_data([2023, 2024])
games = fetch_game_schedule_data([2023])
stats = fetch_weekly_stats_data([2023])
```

### Direct nfl_data_py Usage
```python
import nfl_data_py as nfl

# Core tables
teams = nfl.import_team_desc()
players = nfl.import_seasonal_rosters([2023])
games = nfl.import_schedules([2023])
weekly_stats = nfl.import_weekly_data([2023])

# Advanced analytics
pbp = nfl.import_pbp_data([2023])
ngs_passing = nfl.import_ngs_data('passing', [2023])
injuries = nfl.import_injuries([2023])
```

## 📝 Key Insights

1. **Comprehensive Coverage**: 841+ columns across 24 datasets provide complete NFL data coverage
2. **Real-time Updates**: Most datasets are updated regularly throughout the season
3. **Rich Relationships**: Player IDs, team abbreviations, and game IDs link datasets together
4. **Multiple Granularities**: Data available at team, player, game, play, and snap levels
5. **Historical Depth**: Many datasets go back multiple decades

## 🔗 Database Schema Recommendations

For your Supabase database, consider these primary relationships:

```sql
Teams (team_abbr) ←→ Players (team)
Teams (team_abbr) ←→ Games (home_team, away_team)
Players (player_id) ←→ WeeklyStats (player_id)
Games (game_id) ←→ PlayByPlay (game_id)
```

This structure would support efficient queries across all major NFL data dimensions.
