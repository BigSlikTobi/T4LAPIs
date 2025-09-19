# Auto Updates

Purpose
- Automate periodic updates for time-sensitive datasets in CI or cron.

Key Scripts
- `games_auto_update.py` — detects current season/week and upserts accordingly.
- `player_weekly_stats_auto_update.py` — compares games vs stats to fill gaps and refresh recent weeks.

Usage
- Games: `python scripts/auto_updates/games_auto_update.py`
- Player stats: `python scripts/auto_updates/player_weekly_stats_auto_update.py`

Guidance
- Intended for automation; ensure Supabase env vars are set in runners.
- Weeks > 22 are skipped by design; idempotent on re-runs.
