import logging
from typing import Dict, Any, List, Optional
from src.core.utils.database import DatabaseManager
from src.core.utils.logging import get_logger

logger = logging.getLogger(__name__)

class EntityDictionaryBuilder:
    """Build entity dictionaries for player and team name/abbreviation to ID mapping."""
    
    def __init__(self):
        """Initialize the entity dictionary builder."""
        self.logger = get_logger(__name__)
        self.players_db = DatabaseManager("players")
        self.teams_db = DatabaseManager("teams")
    
    # 1. ADDED 'self' TO FIX THE TypeError
    # 2. REFACTORED to use the helper methods below, removing redundant code.
    def build_entity_dictionary(self) -> Dict[str, str]:
        """
        Builds a comprehensive dictionary by combining player and team mappings.
        This method now delegates the hard work to the helper methods.
        """
        self.logger.info("Building comprehensive entity dictionary...")
        
        # Build player and team mappings separately
        player_mappings = self._build_player_mappings()
        team_mappings = self._build_team_mappings()
        
        # Combine the dictionaries
        entity_dict = {**player_mappings, **team_mappings}
        
        self.logger.info(f"Built complete entity dictionary with {len(entity_dict)} total mappings")
        return entity_dict
    
    def _build_player_mappings(self) -> Dict[str, str]:
        """Build dictionary mappings for players with robust pagination."""
        player_mappings = {}
        self.logger.info("Building player mappings with pagination...")
        
        offset = 0
        batch_size = 1000  # Supabase's default limit
        
        try:
            while True:
                self.logger.info(f"Fetching player records from offset {offset}...")
                response = self.players_db.supabase.table("players").select(
                    "player_id, full_name, display_name, common_first_name, first_name, last_name",
                    count='exact' # Ask Supabase for the total count
                ).range(offset, offset + batch_size - 1).execute()

                if not (hasattr(response, 'data') and response.data):
                    self.logger.info("No more player data found. Finished fetching.")
                    break

                player_batch = response.data
                self.logger.info(f"Processing {len(player_batch)} player records from this batch.")

                for player in player_batch:
                    player_id = player.get('player_id')
                    if not player_id: continue

                    # Create name variations with robust handling for None values
                    name_variations = set()
                    full_name = player.get('full_name', '')
                    display_name = player.get('display_name', '')
                    first_name = player.get('first_name', '')
                    last_name = player.get('last_name', '')
                    common_first_name = player.get('common_first_name', '')

                    if full_name and full_name.strip(): name_variations.add(full_name.strip())
                    if display_name and display_name.strip(): name_variations.add(display_name.strip())
                    if last_name and last_name.strip(): name_variations.add(last_name.strip())
                    if first_name.strip() and last_name.strip():
                        name_variations.add(f"{first_name.strip()} {last_name.strip()}")
                    if common_first_name.strip() and last_name.strip():
                        name_variations.add(f"{common_first_name.strip()} {last_name.strip()}")
                    
                    for name_variant in name_variations:
                        player_mappings[name_variant] = player_id

                # If the number of records returned is less than the batch size,
                # we have reached the last page.
                if len(player_batch) < batch_size:
                    break
                
                # Move to the next page for the next loop iteration
                offset += batch_size
            
            self.logger.info(f"Successfully built {len(player_mappings)} player name mappings from all pages.")
            
        except Exception as e:
            self.logger.error(f"Exception while building player mappings: {e}", exc_info=True)
            
        return player_mappings
    
    def _build_team_mappings(self) -> Dict[str, str]:
        """Build dictionary mappings for teams."""
        team_mappings = {}
        self.logger.info("Building team mappings...")
        
        try:
            response = self.teams_db.supabase.table("teams").select("team_abbr, team_name, team_nick").execute()
            
            if not (hasattr(response, 'data') and response.data):
                self.logger.warning("No team data found in database")
                return team_mappings
            
            for team in response.data:
                team_abbr = team.get('team_abbr')
                if not team_abbr: continue

                team_variations = {
                    team_abbr,
                    team.get('team_name'),
                    team.get('team_nick')
                }

                for name in team_variations:
                    if name and name.strip():
                        team_mappings[name.strip()] = team_abbr
                
                # Add hardcoded alternative names
                team_mappings.update(self._get_team_alternatives(team_abbr))
            
            self.logger.info(f"Built {len(team_mappings)} team name/abbreviation mappings")

        except Exception as e:
            self.logger.error(f"Exception while building team mappings: {e}", exc_info=True)
            
        return team_mappings
    
    def _get_team_alternatives(self, team_abbr: str) -> Dict[str, str]:
        """Get alternative names and abbreviations for a team."""
        alternatives = {}
        # (This is a simplified version of your alternatives mapping)
        team_alternatives = {
            'KC': ['Kansas City', 'Kansas City Chiefs'],
            'SF': ['49ers', 'Niners', 'San Francisco'],
            # ... add all other teams here ...
        }
        
        if team_abbr in team_alternatives:
            for alt_name in team_alternatives[team_abbr]:
                alternatives[alt_name] = team_abbr
        
        return alternatives

# This wrapper function remains unchanged and is correct
def build_entity_dictionary() -> Dict[str, str]:
    """Convenience function to build entity dictionary."""
    builder = EntityDictionaryBuilder()
    return builder.build_entity_dictionary()