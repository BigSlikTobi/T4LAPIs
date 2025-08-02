"""Entity linking utilities for building dictionaries of players and teams."""

import logging
from typing import Dict, Any, List, Optional
from src.core.utils.database import DatabaseManager
from src.core.utils.logging import get_logger


class EntityDictionaryBuilder:
    """Build entity dictionaries for player and team name/abbreviation to ID mapping."""
    
    def __init__(self):
        """Initialize the entity dictionary builder."""
        self.logger = get_logger(__name__)
        self.players_db = DatabaseManager("players")
        self.teams_db = DatabaseManager("teams")
    
    def build_entity_dictionary(self) -> Dict[str, str]:
        """Build a comprehensive dictionary mapping names and abbreviations to unique IDs.
        
        Returns:
            Dictionary mapping entity names/abbreviations to their unique IDs.
            Example: {
                'Patrick Mahomes': '00-0033873',
                'Christian McCaffrey': '00-0035710', 
                'Chiefs': 'KC',
                'Kansas City Chiefs': 'KC',
                '49ers': 'SF'
            }
        """
        entity_dict = {}
        
        self.logger.info("Building entity dictionary from players and teams tables")
        
        # Add player mappings
        try:
            player_mappings = self._build_player_mappings()
            entity_dict.update(player_mappings)
            self.logger.info(f"Added {len(player_mappings)} player name mappings")
        except Exception as e:
            self.logger.error(f"Failed to build player mappings: {e}")
            
        # Add team mappings
        try:
            team_mappings = self._build_team_mappings()
            entity_dict.update(team_mappings)
            self.logger.info(f"Added {len(team_mappings)} team name/abbreviation mappings")
        except Exception as e:
            self.logger.error(f"Failed to build team mappings: {e}")
        
        self.logger.info(f"Built complete entity dictionary with {len(entity_dict)} total mappings")
        return entity_dict
    
    def _build_player_mappings(self) -> Dict[str, str]:
        """Build dictionary mappings for players.
        
        Returns:
            Dictionary mapping player names to player IDs
        """
        player_mappings = {}
        
        try:
            # Query players table for all players with available name fields
            response = self.players_db.supabase.table("players").select(
                "player_id, full_name, display_name, common_first_name, first_name, last_name"
            ).execute()
            
            if hasattr(response, 'error') and response.error:
                self.logger.error(f"Error querying players table: {response.error}")
                return player_mappings
            
            if not hasattr(response, 'data') or not response.data:
                self.logger.warning("No player data found in database")
                return player_mappings
            
            for player in response.data:
                player_id = player.get('player_id')
                
                if not player_id:
                    continue
                
                # Collect all possible name variations for this player
                name_variations = []
                
                # Add full name (primary identifier)
                full_name = player.get('full_name')
                if full_name:
                    clean_name = str(full_name).strip()
                    if clean_name and clean_name.lower() != 'none':
                        name_variations.append(clean_name)
                
                # Add display name (often used in articles)
                display_name = player.get('display_name')
                if display_name:
                    clean_name = str(display_name).strip()
                    if clean_name and clean_name.lower() != 'none':
                        name_variations.append(clean_name)
                
                # Add first name + last name combination
                first_name = player.get('first_name')
                last_name = player.get('last_name')
                if first_name and last_name:
                    full_name_combo = f"{str(first_name).strip()} {str(last_name).strip()}"
                    if full_name_combo.strip() and full_name_combo.lower() != 'none none':
                        name_variations.append(full_name_combo)
                
                # Add common first name + last name combination
                common_first_name = player.get('common_first_name')
                if common_first_name and last_name:
                    common_name_combo = f"{str(common_first_name).strip()} {str(last_name).strip()}"
                    if common_name_combo.strip() and common_name_combo.lower() != 'none none':
                        name_variations.append(common_name_combo)
                
                # Add last name only (for unique surnames)
                if last_name:
                    clean_last_name = str(last_name).strip()
                    if clean_last_name and clean_last_name.lower() != 'none':
                        # Only add last name if it's reasonably unique (length > 3)
                        if len(clean_last_name) > 3:
                            name_variations.append(clean_last_name)
                
                # Add all unique name variations to mappings
                for name_variant in set(name_variations):  # Use set to avoid duplicates
                    if name_variant:  # Final check for non-empty names
                        player_mappings[name_variant] = player_id
            
            self.logger.info(f"Built {len(player_mappings)} player name mappings")
            return player_mappings
            
        except Exception as e:
            self.logger.error(f"Exception while building player mappings: {e}")
            return player_mappings
    
    def _build_team_mappings(self) -> Dict[str, str]:
        """Build dictionary mappings for teams.
        
        Returns:
            Dictionary mapping team names and abbreviations to team abbreviations
        """
        team_mappings = {}
        
        try:
            # Query teams table for all teams
            response = self.teams_db.supabase.table("teams").select(
                "team_abbr, team_name, team_nick"
            ).execute()
            
            if hasattr(response, 'error') and response.error:
                self.logger.error(f"Error querying teams table: {response.error}")
                return team_mappings
            
            if not hasattr(response, 'data') or not response.data:
                self.logger.warning("No team data found in database")
                return team_mappings
            
            for team in response.data:
                team_abbr = team.get('team_abbr')
                team_name = team.get('team_name')
                team_nick = team.get('team_nick')
                
                if team_abbr:
                    # Map abbreviation to itself (for consistency)
                    team_mappings[team_abbr] = team_abbr
                    
                    # Map full team name to abbreviation
                    if team_name:
                        clean_name = str(team_name).strip()
                        if clean_name and clean_name.lower() != 'none':
                            team_mappings[clean_name] = team_abbr
                    
                    # Map team nickname to abbreviation
                    if team_nick:
                        clean_nick = str(team_nick).strip()
                        if clean_nick and clean_nick.lower() != 'none':
                            team_mappings[clean_nick] = team_abbr
                    
                    # Add common alternative names/abbreviations
                    team_mappings.update(self._get_team_alternatives(team_abbr, team_name, team_nick))
            
            self.logger.info(f"Built {len(team_mappings)} team name/abbreviation mappings")
            return team_mappings
            
        except Exception as e:
            self.logger.error(f"Exception while building team mappings: {e}")
            return team_mappings
    
    def _get_team_alternatives(self, team_abbr: str, team_name: Optional[str], 
                             team_nick: Optional[str]) -> Dict[str, str]:
        """Get alternative names and abbreviations for a team.
        
        Args:
            team_abbr: Official team abbreviation (e.g., 'SF')
            team_name: Full team name (e.g., 'San Francisco 49ers')
            team_nick: Team nickname (e.g., '49ers')
            
        Returns:
            Dictionary of alternative names mapping to team abbreviation
        """
        alternatives = {}
        
        if not team_abbr:
            return alternatives
        
        # Common alternative abbreviations and names
        team_alternatives = {
            'ARI': ['Cardinals', 'Arizona', 'AZ Cards'],
            'ATL': ['Falcons', 'Atlanta', 'ATL Falcons'],
            'BAL': ['Ravens', 'Baltimore', 'BAL Ravens'],
            'BUF': ['Bills', 'Buffalo', 'BUF Bills'],
            'CAR': ['Panthers', 'Carolina', 'CAR Panthers'],
            'CHI': ['Bears', 'Chicago', 'CHI Bears'],
            'CIN': ['Bengals', 'Cincinnati', 'CIN Bengals'],
            'CLE': ['Browns', 'Cleveland', 'CLE Browns'],
            'DAL': ['Cowboys', 'Dallas', 'DAL Cowboys'],
            'DEN': ['Broncos', 'Denver', 'DEN Broncos'],
            'DET': ['Lions', 'Detroit', 'DET Lions'],
            'GB': ['Packers', 'Green Bay', 'Green Bay Packers', 'GNB'],
            'HOU': ['Texans', 'Houston', 'HOU Texans'],
            'IND': ['Colts', 'Indianapolis', 'IND Colts'],
            'JAX': ['Jaguars', 'Jacksonville', 'JAX Jaguars', 'JAC'],
            'KC': ['Chiefs', 'Kansas City', 'Kansas City Chiefs', 'KAN'],
            'LV': ['Raiders', 'Las Vegas', 'Las Vegas Raiders', 'LVR', 'OAK'],
            'LAC': ['Chargers', 'Los Angeles Chargers', 'LA Chargers', 'LAC Chargers', 'SD'],
            'LAR': ['Rams', 'Los Angeles Rams', 'LA Rams', 'LAR Rams', 'STL'],
            'MIA': ['Dolphins', 'Miami', 'MIA Dolphins'],
            'MIN': ['Vikings', 'Minnesota', 'MIN Vikings'],
            'NE': ['Patriots', 'New England', 'New England Patriots', 'NEP'],
            'NO': ['Saints', 'New Orleans', 'New Orleans Saints', 'NOR'],
            'NYG': ['Giants', 'New York Giants', 'NY Giants', 'NYG Giants'],
            'NYJ': ['Jets', 'New York Jets', 'NY Jets', 'NYJ Jets'],
            'PHI': ['Eagles', 'Philadelphia', 'PHI Eagles'],
            'PIT': ['Steelers', 'Pittsburgh', 'PIT Steelers'],
            'SF': ['49ers', 'Niners', 'San Francisco', 'San Francisco 49ers', 'SFO'],
            'SEA': ['Seahawks', 'Seattle', 'SEA Seahawks'],
            'TB': ['Buccaneers', 'Bucs', 'Tampa Bay', 'Tampa Bay Buccaneers', 'TAM'],
            'TEN': ['Titans', 'Tennessee', 'TEN Titans'],
            'WAS': ['Commanders', 'Washington', 'Washington Commanders', 'WSH']
        }
        
        if team_abbr in team_alternatives:
            for alt_name in team_alternatives[team_abbr]:
                alternatives[alt_name] = team_abbr
        
        return alternatives


def build_entity_dictionary() -> Dict[str, str]:
    """Convenience function to build entity dictionary.
    
    Returns:
        Dictionary mapping entity names/abbreviations to their unique IDs
    """
    builder = EntityDictionaryBuilder()
    return builder.build_entity_dictionary()
