"""
Standalone module matching tests' import path `player_weekly_stats_auto_update`.

Implements small wrappers so patching in tests works without network/DB calls.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.core.data.loaders.player_weekly_stats import PlayerWeeklyStatsDataLoader
from src.core.utils.cli import setup_cli_logging, print_results, handle_cli_errors
from src.core.db.database_init import get_supabase_client


def get_latest_week_in_games_table(season: int) -> int:
    try:
        supabase = get_supabase_client()
        if not supabase:
            logging.error("Could not connect to database")
            return 0
        response = (
            supabase.table("games")
            .select("week")
            .eq("season", season)
            .order("week", desc=True)
            .limit(1)
            .execute()
        )
        if getattr(response, "data", None):
            return response.data[0]["week"]
        logging.info(f"No games found for season {season}")
        return 0
    except Exception as e:
        logging.error(f"Error getting latest week from games table: {e}")
        return 0


def get_latest_week_in_stats_table(season: int) -> int:
    try:
        supabase = get_supabase_client()
        if not supabase:
            logging.error("Could not connect to database")
            return 0
        response = (
            supabase.table("player_weekly_stats")
            .select("week")
            .eq("season", season)
            .order("week", desc=True)
            .limit(1)
            .execute()
        )
        if getattr(response, "data", None):
            return response.data[0]["week"]
        logging.info(f"No player stats found for season {season}")
        return 0
    except Exception as e:
        logging.error(f"Error getting latest week from stats table: {e}")
        return 0


def get_current_nfl_season() -> int:
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


@handle_cli_errors
def main():
    class Args:
        verbose = True
        quiet = False

    setup_cli_logging(Args())
    logger = logging.getLogger(__name__)

    try:
        print("📊 Automated NFL Player Weekly Stats Update")
        season = get_current_nfl_season()
        print(f"Current NFL season: {season}")

        loader = PlayerWeeklyStatsDataLoader()
        latest_games_week = get_latest_week_in_games_table(season)
        latest_stats_week = get_latest_week_in_stats_table(season)

        print(f"Latest week in games table for season {season}: {latest_games_week}")
        print(f"Latest week in stats table for season {season}: {latest_stats_week}")

        if latest_games_week == 0:
            print("No games data found. Player stats update requires games data first.")
            return False

        if latest_stats_week == 0:
            weeks = list(range(1, latest_games_week + 1))
            print(f"No existing stats data found. Processing weeks 1 through {latest_games_week}.")
        else:
            start_week = max(1, latest_stats_week - 1)
            weeks = list(range(start_week, latest_games_week + 1))
            print(f"Will update weeks {start_week} through {latest_games_week} (recent + new data)")

        if not weeks:
            print("No weeks to process. Stats are up to date.")
            return True

        results = []
        for week in weeks:
            if week > 22:
                print(f"Week {week} is beyond NFL season, skipping")
                continue
            print(f"\n📅 Processing season {season}, week {week} stats")
            result = loader.load_data(years=[season], weeks=[week], dry_run=False, clear_table=False)
            results.append(result)
            print_results(result, f"player weekly stats for season {season}, week {week}", False)
            print("✅ Successfully processed week {week}" if result.get("success") else f"❌ Failed to process week {week}")

        print(f"\n📊 Total player weekly stat records in database: {loader.get_record_count()}")
        return all(r.get("success") for r in results)
    except Exception as e:
        logger.error(f"Unexpected error in automated stats update: {e}")
        print(f"❌ Unexpected error: {e}")
        return False
