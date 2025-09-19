"""
Standalone module matching tests' import path `games_auto_update`.

Implements small wrappers so patching in tests works without network/DB calls.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.core.data.loaders.games import GamesDataLoader
from src.core.utils.cli import setup_cli_logging, print_results, handle_cli_errors
from src.core.db.database_init import get_supabase_client


def get_latest_week_in_db(loader: GamesDataLoader, season: int) -> int:
    """Return max week for `season` from the `games` table; 0 on failure."""
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
        logging.error(f"Error getting latest week: {e}")
        return 0


def get_current_nfl_season() -> int:
    """Return current NFL season year based on today's date."""
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


@handle_cli_errors
def main():
    """Automated games data update.

    Returns True on success for decorator to convert to exit code 0.
    """
    class Args:
        verbose = True
        quiet = False

    setup_cli_logging(Args())
    logger = logging.getLogger(__name__)

    try:
        print("🏈 Automated NFL Games Data Update")
        season = get_current_nfl_season()
        print(f"Current NFL season: {season}")

        loader = GamesDataLoader()
        latest = get_latest_week_in_db(loader, season)
        print(f"Latest week in database for season {season}: {latest}")

        weeks = [1] if latest == 0 else [latest, latest + 1]
        results = []
        for week in weeks:
            if week > 22:
                print(f"Week {week} is beyond NFL season, skipping")
                continue
            print(f"\n📅 Processing season {season}, week {week}")
            result = loader.load_data(season=season, week=week, dry_run=False, clear_table=False)
            results.append(result)
            print_results(result, f"games data for season {season}, week {week}", False)
            print("✅ Successfully processed week {week}" if result.get("success") else f"❌ Failed to process week {week}")

        print(f"\n📊 Total games in database: {loader.get_game_count()}")
        return all(r.get("success") for r in results)
    except Exception as e:
        logger.error(f"Unexpected error in automated update: {e}")
        print(f"❌ Unexpected error: {e}")
        return False
