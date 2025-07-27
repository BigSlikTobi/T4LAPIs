"""Games table data loader with upsert functionality."""

import logging
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from ..fetch import fetch_game_schedule_data
from ..transform import GameDataTransformer
from src.core.db.database_init import get_supabase_client


logger = logging.getLogger(__name__)


class GamesDataLoader:
    """Loads game data into the database with upsert functionality.
    
    Games need upsert capability since scores and other details might be updated
    as games progress or after final scores are confirmed.
    """
    
    def __init__(self):
        """Initialize the games data loader."""
        self.table_name = "games"
        self.supabase = get_supabase_client()
        
        if not self.supabase:
            raise RuntimeError("Could not initialize Supabase client")
        
    def load_games(self, season: int, week: Optional[int] = None, dry_run: bool = False, clear_table: bool = False) -> Dict[str, Any]:
        """Load game data for a specific season and optionally a specific week.
        
        Args:
            season: NFL season year
            week: Specific week number (if None, loads entire season)
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        if week:
            logger.info(f"Starting games data load for season {season}, week {week}")
        else:
            logger.info(f"Starting games data load for entire season {season}")
        
        try:
            # Fetch raw game data
            logger.info("Fetching game schedule data...")
            raw_games = fetch_game_schedule_data([season])
            
            if raw_games.empty:
                logger.warning(f"No game data found for season {season}")
                return {"success": False, "message": f"No data found for season {season}"}
            
            # Filter by week if specified
            if week is not None:
                raw_games = raw_games[raw_games['week'] == week]
                if raw_games.empty:
                    logger.warning(f"No game data found for season {season}, week {week}")
                    return {"success": False, "message": f"No data found for season {season}, week {week}"}
            
            logger.info(f"Fetched {len(raw_games)} game records")
            
            # Transform the data using the new class
            logger.info("Transforming game data...")
            transformer = GameDataTransformer()
            validated_games = transformer.transform(raw_games)
            
            if not validated_games:
                logger.error("No valid game records after validation")
                return {"success": False, "message": "No valid records after validation"}
            
            logger.info(f"Validated {len(validated_games)} game records")
            
            if dry_run:
                logger.info("DRY RUN - Would process the following operations:")
                logger.info(f"- Clear table: {clear_table}")
                logger.info(f"- Upsert {len(validated_games)} game records")
                
                # Show sample of what would be processed
                if validated_games:
                    sample_record = validated_games[0]
                    logger.info(f"Sample record: {sample_record}")
                
                return {
                    "success": True,
                    "dry_run": True,
                    "would_clear": clear_table,
                    "would_upsert": len(validated_games),
                    "sample_record": validated_games[0] if validated_games else None
                }
            
            # Clear table if requested
            if clear_table:
                logger.info("Clearing existing game data...")
                self._clear_table()
            
            # Upsert game data
            logger.info(f"Upserting {len(validated_games)} game records...")
            upsert_result = self._upsert_games(validated_games)
            
            logger.info("Games data load completed successfully")
            return {
                "success": True,
                "season": season,
                "week": week,
                "total_fetched": len(raw_games),
                "total_validated": len(validated_games),
                "upsert_result": upsert_result,
                "cleared_table": clear_table
            }
            
        except Exception as e:
            logger.error(f"Error loading games data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _clear_table(self) -> None:
        """Clear all data from the games table."""
        try:
            result = self.supabase.table(self.table_name).delete().neq('game_id', '').execute()
            logger.info(f"Cleared games table")
        except Exception as e:
            logger.error(f"Error clearing games table: {str(e)}")
            raise
    
    def _upsert_games(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert game records into the database.
        
        Args:
            games: List of validated game dictionaries
            
        Returns:
            Dictionary with upsert results
        """
        try:
            # Use upsert to handle score updates and other changes
            result = self.supabase.table(self.table_name).upsert(
                games,
                on_conflict="game_id"  # Use game_id as the conflict resolution key
            ).execute()
            
            return {
                "operation": "upsert",
                "affected_rows": len(result.data) if result.data else 0,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error upserting games: {str(e)}")
            raise
    
    def get_game_count(self, season: Optional[int] = None, week: Optional[int] = None) -> int:
        """Get the current number of games in the database.
        
        Args:
            season: Filter by season (optional)
            week: Filter by week (optional)
            
        Returns:
            Number of games in the database
        """
        try:
            query = self.supabase.table(self.table_name).select("game_id", count="exact")
            
            if season:
                query = query.eq("season", season)
            if week:
                query = query.eq("week", week)
                
            result = query.execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error getting game count: {str(e)}")
            return 0
    
    def get_games_by_team(self, team_abbr: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all games for a specific team.
        
        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF')
            season: Filter by season (optional)
            
        Returns:
            List of game dictionaries
        """
        try:
            query = self.supabase.table(self.table_name).select("*").or_(
                f"home_team_abbr.eq.{team_abbr},away_team_abbr.eq.{team_abbr}"
            )
            
            if season:
                query = query.eq("season", season)
                
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting games for team {team_abbr}: {str(e)}")
            return []
    
    def get_games_by_week(self, season: int, week: int) -> List[Dict[str, Any]]:
        """Get all games for a specific season and week.
        
        Args:
            season: NFL season year
            week: Week number
            
        Returns:
            List of game dictionaries
        """
        try:
            result = self.supabase.table(self.table_name).select("*").eq("season", season).eq("week", week).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting games for season {season}, week {week}: {str(e)}")
            return []


def main():
    """CLI interface for the games data loader."""
    import argparse
    import sys
    import os
    
    # Add project root to path for CLI usage
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import here for CLI to avoid relative import issues
    from src.core.data.loaders.games import GamesDataLoader
    
    parser = argparse.ArgumentParser(description="Load NFL game data into the database")
    parser.add_argument("season", type=int, help="NFL season year (e.g., 2024)")
    parser.add_argument("--week", type=int, help="Specific week number (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    parser.add_argument("--clear", action="store_true", help="Clear existing game data before loading")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create loader and run
        loader = GamesDataLoader()
        result = loader.load_games(
            season=args.season,
            week=args.week,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        if result["success"]:
            if args.dry_run:
                if args.week:
                    print(f"DRY RUN - Would upsert {result['would_upsert']} game records for season {args.season}, week {args.week}")
                else:
                    print(f"DRY RUN - Would upsert {result['would_upsert']} game records for entire season {args.season}")
                if result.get("sample_record"):
                    print(f"Sample record: {result['sample_record']}")
            else:
                if args.week:
                    print(f"Successfully loaded game data for season {args.season}, week {args.week}")
                else:
                    print(f"Successfully loaded game data for entire season {args.season}")
                print(f"Fetched: {result['total_fetched']} records")
                print(f"Validated: {result['total_validated']} records")
                print(f"Upserted: {result['upsert_result']['affected_rows']} records")
                
                # Show current game count
                total_games = loader.get_game_count()
                print(f"Total games in database: {total_games}")
        else:
            print(f"Error: {result.get('error', result.get('message', 'Unknown error'))}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
