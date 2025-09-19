"""
Trending Topic Detector Script - Epic 2 Task 3

This script analyzes articles created in the last N hours to find the players and teams
mentioned most frequently, identifying trending entities based on article creation dates.

Usage:
    python scripts/trending_topic_detector.py [--hours 24] [--top-n 10] [--output-format json]

Features:
- Analyzes articles by creation date (SourceArticles.created_at)
- Configurable lookback period (default: 24 hours)
- Configurable number of top entities to return (default: 10)
- Multiple output formats (json, csv, plain)
- Separate tracking for players and teams
- Frequency and trend analysis
"""

import logging
import argparse
import json
import csv
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.utils.database import DatabaseManager
from src.core.utils.logging import get_logger


@dataclass
class TrendingEntity:
    """Represents a trending entity with its metrics."""
    entity_id: str
    entity_type: str  # 'player' or 'team'
    mention_count: int
    entity_name: Optional[str] = None
    team_abbr: Optional[str] = None  # For players
    position: Optional[str] = None   # For players
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'mention_count': self.mention_count,
            'entity_name': self.entity_name,
            'team_abbr': self.team_abbr,
            'position': self.position
        }


class TrendingTopicDetector:
    """Detects trending NFL entities based on article mention frequency."""
    
    def __init__(self, lookback_hours: int = 24):
        """Initialize the trending topic detector.
        
        Args:
            lookback_hours: Number of hours to look back for trending analysis
        """
        self.lookback_hours = lookback_hours
        self.logger = get_logger(__name__)
        
        # Initialize database connections
        try:
            self.links_db = DatabaseManager("article_entity_links")
            self.players_db = DatabaseManager("players")
            self.teams_db = DatabaseManager("teams")
            
            self.logger.info(f"Trending Topic Detector initialized with {lookback_hours}h lookback")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    def get_trending_entities(self, top_n: int = 10) -> Tuple[List[TrendingEntity], List[TrendingEntity]]:
        """Get the top trending players and teams.
        
        Args:
            top_n: Number of top entities to return for each type
            
        Returns:
            Tuple of (trending_players, trending_teams)
        """
        self.logger.info(f"Analyzing trending entities from articles created in the last {self.lookback_hours} hours")
        
        # Calculate the cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        cutoff_iso = cutoff_time.isoformat()
        
        try:
            # Query for entity mentions from articles created since cutoff time
            # We need to join article_entity_links with SourceArticles to get the creation date
            response = self.links_db.supabase.table("article_entity_links").select(
                "entity_id, entity_type, SourceArticles!inner(created_at)"
            ).gte("SourceArticles.created_at", cutoff_iso).execute()
            
            if not (hasattr(response, 'data') and response.data):
                self.logger.warning("No entity links found for articles created in the specified time period")
                return [], []
            
            # Count mentions by entity type
            player_mentions = defaultdict(int)
            team_mentions = defaultdict(int)
            
            for link in response.data:
                entity_id = link['entity_id']
                entity_type = link['entity_type']
                
                # Skip sentinel records used for marking processed articles
                if entity_id == "PROCESSED_EMPTY":
                    continue
                
                if entity_type == 'player':
                    player_mentions[entity_id] += 1
                elif entity_type == 'team':
                    team_mentions[entity_id] += 1
            
            self.logger.info(f"Found {len(player_mentions)} trending players and {len(team_mentions)} trending teams")
            
            # Get top trending players
            trending_players = self._build_trending_players(player_mentions, top_n)
            
            # Get top trending teams  
            trending_teams = self._build_trending_teams(team_mentions, top_n)
            
            return trending_players, trending_teams
            
        except Exception as e:
            self.logger.error(f"Error analyzing trending entities: {e}")
            raise
    
    def _build_trending_players(self, mentions: Dict[str, int], top_n: int) -> List[TrendingEntity]:
        """Build trending player entities with enriched data."""
        if not mentions:
            return []
        
        # Get top players by mention count
        top_player_ids = sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        trending_players = []
        
        for player_id, mention_count in top_player_ids:
            try:
                # Get player details from database
                player_response = self.players_db.supabase.table("players").select(
                    "full_name, latest_team, position"
                ).eq("player_id", player_id).limit(1).execute()
                
                player_name = None
                team_abbr = None
                position = None
                
                if hasattr(player_response, 'data') and player_response.data:
                    player_data = player_response.data[0]
                    player_name = player_data.get('full_name')
                    team_abbr = player_data.get('latest_team')
                    position = player_data.get('position')
                
                trending_player = TrendingEntity(
                    entity_id=player_id,
                    entity_type='player',
                    mention_count=mention_count,
                    entity_name=player_name,
                    team_abbr=team_abbr,
                    position=position
                )
                
                trending_players.append(trending_player)
                
            except Exception as e:
                self.logger.warning(f"Could not enrich data for player {player_id}: {e}")
                # Add without enrichment
                trending_players.append(TrendingEntity(
                    entity_id=player_id,
                    entity_type='player',
                    mention_count=mention_count
                ))
        
        return trending_players
    
    def _build_trending_teams(self, mentions: Dict[str, int], top_n: int) -> List[TrendingEntity]:
        """Build trending team entities with enriched data."""
        if not mentions:
            return []
        
        # Get top teams by mention count
        top_team_ids = sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        trending_teams = []
        
        for team_abbr, mention_count in top_team_ids:
            try:
                # Get team details from database
                team_response = self.teams_db.supabase.table("teams").select(
                    "team_name"
                ).eq("team_abbr", team_abbr).limit(1).execute()
                
                team_name = None
                
                if hasattr(team_response, 'data') and team_response.data:
                    team_data = team_response.data[0]
                    team_name = team_data.get('team_name')
                
                trending_team = TrendingEntity(
                    entity_id=team_abbr,
                    entity_type='team',
                    mention_count=mention_count,
                    entity_name=team_name
                )
                
                trending_teams.append(trending_team)
                
            except Exception as e:
                self.logger.warning(f"Could not enrich data for team {team_abbr}: {e}")
                # Add without enrichment
                trending_teams.append(TrendingEntity(
                    entity_id=team_abbr,
                    entity_type='team',
                    mention_count=mention_count
                ))
        
        return trending_teams
    
    def get_trending_entity_ids(self, top_n: int = 10) -> List[str]:
        """Get a simple list of trending entity IDs (both players and teams).
        
        This method is designed for use by the trending summary generator.
        
        Args:
            top_n: Total number of entities to return (split between players and teams)
            
        Returns:
            List of entity IDs sorted by mention frequency
        """
        players_count = top_n // 2
        teams_count = top_n - players_count
        
        trending_players, trending_teams = self.get_trending_entities(max(players_count, teams_count))
        
        # Combine and sort by mention count
        all_entities = trending_players[:players_count] + trending_teams[:teams_count]
        all_entities.sort(key=lambda x: x.mention_count, reverse=True)
        
        return [entity.entity_id for entity in all_entities[:top_n]]


def output_results(trending_players: List[TrendingEntity], 
                  trending_teams: List[TrendingEntity], 
                  output_format: str = 'json',
                  output_file: Optional[str] = None) -> None:
    """Output the trending results in the specified format."""
    
    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'trending_players': [player.to_dict() for player in trending_players],
        'trending_teams': [team.to_dict() for team in trending_teams],
        'summary': {
            'total_trending_players': len(trending_players),
            'total_trending_teams': len(trending_teams),
            'top_player': trending_players[0].to_dict() if trending_players else None,
            'top_team': trending_teams[0].to_dict() if trending_teams else None
        }
    }
    
    if output_format.lower() == 'json':
        output_content = json.dumps(results, indent=2)
    elif output_format.lower() == 'csv':
        output_content = _format_as_csv(trending_players, trending_teams)
    elif output_format.lower() == 'plain':
        output_content = _format_as_plain_text(trending_players, trending_teams)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_content)
        print(f"Results written to {output_file}")
    else:
        print(output_content)


def _format_as_csv(players: List[TrendingEntity], teams: List[TrendingEntity]) -> str:
    """Format results as CSV."""
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['entity_type', 'entity_id', 'entity_name', 'mention_count', 'team_abbr', 'position'])
    
    # Players
    for player in players:
        writer.writerow([
            player.entity_type,
            player.entity_id, 
            player.entity_name or '',
            player.mention_count,
            player.team_abbr or '',
            player.position or ''
        ])
    
    # Teams
    for team in teams:
        writer.writerow([
            team.entity_type,
            team.entity_id,
            team.entity_name or '',
            team.mention_count,
            '',
            ''
        ])
    
    return output.getvalue()


def _format_as_plain_text(players: List[TrendingEntity], teams: List[TrendingEntity]) -> str:
    """Format results as plain text."""
    lines = []
    lines.append("TRENDING NFL ENTITIES")
    lines.append("=" * 50)
    lines.append("")
    
    if players:
        lines.append("TOP TRENDING PLAYERS:")
        lines.append("-" * 25)
        for i, player in enumerate(players, 1):
            name = player.entity_name or player.entity_id
            team = f" ({player.team_abbr})" if player.team_abbr else ""
            pos = f" - {player.position}" if player.position else ""
            lines.append(f"{i:2d}. {name}{team}{pos} - {player.mention_count} mentions")
        lines.append("")
    
    if teams:
        lines.append("TOP TRENDING TEAMS:")
        lines.append("-" * 25)
        for i, team in enumerate(teams, 1):
            name = team.entity_name or team.entity_id
            lines.append(f"{i:2d}. {name} - {team.mention_count} mentions")
        lines.append("")
    
    lines.append(f"Analysis timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    return "\n".join(lines)


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Detect trending NFL entities from article mentions')
    parser.add_argument('--hours', type=int, default=24, 
                       help='Number of hours to look back (default: 24)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top entities to return per type (default: 10)')
    parser.add_argument('--output-format', choices=['json', 'csv', 'plain'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--output-file', type=str,
                       help='Output file path (default: stdout)')
    parser.add_argument('--entity-ids-only', action='store_true',
                       help='Output only entity IDs (for use by trending summary generator)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize detector
        detector = TrendingTopicDetector(lookback_hours=args.hours)
        
        if args.entity_ids_only:
            # Simple output for integration with trending summary generator
            entity_ids = detector.get_trending_entity_ids(top_n=args.top_n)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write('\n'.join(entity_ids))
            else:
                for entity_id in entity_ids:
                    print(entity_id)
        else:
            # Full analysis output
            trending_players, trending_teams = detector.get_trending_entities(top_n=args.top_n)
            
            output_results(trending_players, trending_teams, 
                         output_format=args.output_format,
                         output_file=args.output_file)
    
    except Exception as e:
        logging.error(f"Error during trending analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
