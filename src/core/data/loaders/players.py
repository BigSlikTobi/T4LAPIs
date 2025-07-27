"""Players table data loader with upsert functionality."""

import logging
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from ..fetch import fetch_player_data
from ..transform import PlayerDataTransformer
from src.core.db.database_init import get_supabase_client


logger = logging.getLogger(__name__)


class PlayersDataLoader:
    """Loads player data into the database with upsert functionality.
    
    Players need upsert capability since they might change teams or positions during the season.
    """
    
    def __init__(self):
        """Initialize the players data loader."""
        self.table_name = "players"
        self.supabase = get_supabase_client()
        
        if not self.supabase:
            raise RuntimeError("Could not initialize Supabase client")
        
    def load_players(self, season: int, dry_run: bool = False, clear_table: bool = False) -> Dict[str, Any]:
        """Load player data for a specific season.
        
        Args:
            season: NFL season year
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        logger.info(f"Starting players data load for season {season}")
        
        try:
            # Fetch raw player data
            logger.info("Fetching player data...")
            raw_players = fetch_player_data([season])  # Pass as list
            
            if raw_players.empty:
                logger.warning(f"No player data found for season {season}")
                return {"success": False, "message": f"No data found for season {season}"}
            
            logger.info(f"Fetched {len(raw_players)} player records")
            
            # Transform the data using the new class
            logger.info("Transforming player data...")
            transformer = PlayerDataTransformer()
            validated_players = transformer.transform(raw_players)
            
            if not validated_players:
                logger.error("No valid player records after validation")
                return {"success": False, "message": "No valid records after validation"}
            
            logger.info(f"Validated {len(validated_players)} player records")
            
            if dry_run:
                logger.info("DRY RUN - Would process the following operations:")
                logger.info(f"- Clear table: {clear_table}")
                logger.info(f"- Upsert {len(validated_players)} player records")
                
                # Show sample of what would be processed
                if validated_players:
                    sample_record = validated_players[0]
                    logger.info(f"Sample record: {sample_record}")
                
                return {
                    "success": True,
                    "dry_run": True,
                    "would_clear": clear_table,
                    "would_upsert": len(validated_players),
                    "sample_record": validated_players[0] if validated_players else None
                }
            
            # Clear table if requested
            if clear_table:
                logger.info("Clearing existing player data...")
                self._clear_table()
            
            # Upsert player data
            logger.info(f"Upserting {len(validated_players)} player records...")
            upsert_result = self._upsert_players(validated_players)
            
            logger.info("Players data load completed successfully")
            return {
                "success": True,
                "season": season,
                "total_fetched": len(raw_players),
                "total_validated": len(validated_players),
                "upsert_result": upsert_result,
                "cleared_table": clear_table
            }
            
        except Exception as e:
            logger.error(f"Error loading players data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _clear_table(self) -> None:
        """Clear all data from the players table."""
        try:
            result = self.supabase.table(self.table_name).delete().neq('player_id', '').execute()
            logger.info(f"Cleared players table")
        except Exception as e:
            logger.error(f"Error clearing players table: {str(e)}")
            raise
    
    def _upsert_players(self, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert player records into the database.
        
        Args:
            players: List of validated player dictionaries
            
        Returns:
            Dictionary with upsert results
        """
        try:
            # Use upsert to handle players changing teams/positions
            result = self.supabase.table(self.table_name).upsert(
                players,
                on_conflict="player_id"  # Use player_id as the conflict resolution key
            ).execute()
            
            return {
                "operation": "upsert",
                "affected_rows": len(result.data) if result.data else 0,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error upserting players: {str(e)}")
            raise
    
    def get_player_count(self) -> int:
        """Get the current number of players in the database.
        
        Returns:
            Number of players in the database
        """
        try:
            result = self.supabase.table(self.table_name).select("player_id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error getting player count: {str(e)}")
            return 0
    
    def get_players_by_team(self, team_abbr: str) -> List[Dict[str, Any]]:
        """Get all players for a specific team.
        
        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF')
            
        Returns:
            List of player dictionaries
        """
        try:
            result = self.supabase.table(self.table_name).select("*").eq("latest_team", team_abbr).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting players for team {team_abbr}: {str(e)}")
            return []


def main():
    """CLI interface for the players data loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load NFL player data into the database")
    parser.add_argument("season", type=int, help="NFL season year (e.g., 2024)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    parser.add_argument("--clear", action="store_true", help="Clear existing player data before loading")
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
        loader = PlayersDataLoader()
        result = loader.load_players(
            season=args.season,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        if result["success"]:
            if args.dry_run:
                print(f"DRY RUN - Would upsert {result['would_upsert']} player records for season {args.season}")
                if result.get("sample_record"):
                    print(f"Sample record: {result['sample_record']}")
            else:
                print(f"Successfully loaded player data for season {args.season}")
                print(f"Fetched: {result['total_fetched']} records")
                print(f"Validated: {result['total_validated']} records")
                print(f"Upserted: {result['upsert_result']['affected_rows']} records")
                
                # Show current player count
                total_players = loader.get_player_count()
                print(f"Total players in database: {total_players}")
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
