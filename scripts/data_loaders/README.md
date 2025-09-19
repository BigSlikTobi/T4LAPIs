# Data Loaders

Purpose
- Ingest core NFL datasets (teams, players, games, weekly stats, NGS, PBP, snap counts, depth charts, PFR, QBR, FTN).

Key Scripts
- Core: `teams_cli.py`, `players_cli.py`, `games_cli.py`, `player_weekly_stats_cli.py`, `snap_counts_cli.py`, `depth_charts_cli.py`, `pbp_cli.py`, `ngs_cli.py`.
- External stats: `stats/pfr_cli.py`, `stats/qbr_cli.py`, `stats/ftn_cli.py`.

Usage
- Teams: `python scripts/data_loaders/teams_cli.py --dry-run`
- Games: `python scripts/data_loaders/games_cli.py 2024 --week 5`
- Player weekly: `python scripts/data_loaders/player_weekly_stats_cli.py 2024 --week 6`
- NGS: `python scripts/data_loaders/ngs_cli.py passing 2024 2023`

Guidance
- Prefer `--dry-run` first. Set Supabase env vars to persist.
- Some loaders accept `--clear` to replace tables; use sparingly.
