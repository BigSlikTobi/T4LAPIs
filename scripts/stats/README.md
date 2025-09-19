# Statistics CLI Suite

This folder contains organized CLI tools for loading various NFL statistics into the database via `nfl_data_py` loaders in `src/core/data/loaders`.

Each CLI supports a standard set of flags (`--dry-run`, `--clear`, `--verbose|--log-level`) and dataset-specific filters (e.g., `--week`, `--player-id`). All CLIs return exit code 0 on success and 1 on failure.

## Common behaviors
- Week filtering: When a week is not provided for weekly datasets, the loader defaults to the latest week available per season.
- Player filtering: Where applicable, loaders accept `--player-id`, supporting multiple IDs (repeat or comma-separated).
- Dry run: Shows what would be upserted without touching the database, including count and a sample record.

## Tools

### PFR (Pro Football Reference)
- Script: `pfr_cli.py`
- Loader: `ProFootballReferenceDataLoader`
- Modes: seasonal (default) and weekly (`--weekly`)
- Filters:
  - `--week` (weekly only): Specific week, otherwise latest week per season
  - `--player-id`: Filter by GSIS or PFR player ID(s); GSIS ids are mapped to PFR internally
- Example:
  - Weekly, latest week: `scripts/stats/pfr_cli.py pass 2024 --weekly --dry-run`
  - Weekly, week 3: `scripts/stats/pfr_cli.py rec 2023 2024 --weekly --week 3 --dry-run`
  - Filter players: `scripts/stats/pfr_cli.py rush 2024 --weekly --player-id BradTo00,EmmiDo00 --dry-run`
  - Filter by GSIS ID (mapped to PFR): `scripts/stats/pfr_cli.py pass 2024 --weekly --week 1 --player-id 00-0033873 --dry-run`  # Patrick Mahomes (maps to MahoPa00)

### FTN (Football Study Hall)
- Script: `ftn_cli.py`
- Loader: `FootballStudyHallDataLoader`
- Notes: Play-level dataset; includes seasonal/week columns.
- Filters:
  - `--week`: Specific week, otherwise latest week per season
  - `--player-id`: Attempts to filter using available player columns (e.g., player_id, pfr_player_id, or passer/receiver/rusher ids)
- Example:
  - Latest week: `scripts/stats/ftn_cli.py 2024 --dry-run`
  - Week 1: `scripts/stats/ftn_cli.py 2024 --week 1 --dry-run`
  - Player filtered: `scripts/stats/ftn_cli.py 2024 --player-id BrowSp00 --week 2 --dry-run`

### QBR (ESPN Total QBR)
  - `--week`: Specific week, otherwise latest week per season
  - `--player-id`: Filter by available id column; QBR typically provides an ESPN `player_id`

Notes:
- Data availability can lag for the current season. If a year returns 0 rows (e.g., 2024 at the time of writing), the CLI will report "No data found". Try a previous season (e.g., 2023) or rerun later.
- Some sources expose season aggregates as "Season Total" without a numeric week; the loader normalizes that to week 0.
- Example:
  - Latest week: `scripts/stats/qbr_cli.py 2024 --dry-run`
  - Week 5: `scripts/stats/qbr_cli.py 2024 --week 5 --dry-run`

## Conventions
- Exit codes: 0 (success), 1 (failure)
- Logging: controlled by `--verbose` or `--log-level`
- DB operations: upsert by default (except special cases as noted in loader base)

## Troubleshooting
- No records after filtering: Ensure the provided player IDs match the dataset's ID type (GSIS vs PFR vs ESPN). Some loaders attempt mapping but not all conversions are possible.
- Week not found: If you specify a week outside the available range for the season, you may get no records.
- Very large datasets: Consider `--dry-run` first to gauge record counts and sample structure.
- Player-week has no rows: Even with a valid player mapping, some stat categories or specific weeks may yield no rows for that player (e.g., bye weeks or no attempts for that stat). Try another week or stat type.
