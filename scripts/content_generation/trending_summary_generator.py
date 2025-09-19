#!/usr/bin/env python3
"""
Trending Summary Generator Script - Epic 2 Task 4

This script generates comprehensive summaries for trending NFL entities identified by the trending topic detector.
Key features:
- Takes trending entity IDs as input (from trending_topic_detector.py)
- Gathers recent articles and stats for each trending entity
- Generates AI-powered comprehensive summaries using LLM
- Stores results in trending_entity_updates table
- Supports multiple input methods (CLI args, file, pipe from detector)

The generator uses:
- Trending entity IDs from the trending topic detector
- Recent articles from SourceArticles via article_entity_links
- Player stats from player_weekly_stats
- Gemini/DeepSeek LLM for summary generation
- trending_entity_updates table for storage

Usage:
    python scripts/trending_summary_generator.py --entity-ids "00-0033873,KC,NYJ"
    python scripts/trending_summary_generator.py --input-file trending_entities.txt
    python scripts/trending_topic_detector.py --entity-ids-only | python scripts/trending_summary_generator.py --from-stdin
"""

import os
import sys
from pathlib import Path
import logging
import argparse
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path for imports (robust to nesting)
def _repo_root() -> str:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / 'src').exists() and (p / 'README.md').exists():
            return str(p)
    return str(start.parents[0])

project_root = _repo_root()
sys.path.insert(0, project_root)

from src.core.utils.database import DatabaseManager
from src.core.llm.llm_setup import initialize_model, generate_content_with_model


@dataclass
class TrendingSummary:
    """Represents a generated trending summary with metadata."""
    entity_id: str
    entity_type: str
    entity_name: Optional[str]
    summary_content: str
    source_article_count: int
    source_stat_count: int
    generated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'entity_name': self.entity_name,
            'summary_content': self.summary_content,
            'source_article_count': self.source_article_count,
            'source_stat_count': self.source_stat_count,
            'generated_at': self.generated_at
        }


class TrendingSummaryGenerator:
    """Generates comprehensive summaries for trending NFL entities."""
    
    def __init__(self, lookback_hours: int = 72, dry_run: bool = False, 
                 preferred_llm_provider: str = 'gemini'):
        """Initialize the trending summary generator.
        
        Args:
            lookback_hours: How many hours to look back for content (default: 72 for trending context)
            dry_run: If True, don't save summaries to database
            preferred_llm_provider: Preferred LLM provider ('gemini' or 'deepseek')
        """
        self.logger = logging.getLogger(__name__)
        self.lookback_hours = lookback_hours
        self.dry_run = dry_run
        self.preferred_llm_provider = preferred_llm_provider
        
        # Initialize database managers
        self.trending_updates_db = DatabaseManager("trending_entity_updates")
        self.entity_links_db = DatabaseManager("article_entity_links")
        self.players_db = DatabaseManager("players")
        self.teams_db = DatabaseManager("teams")
        self.stats_db = DatabaseManager("player_weekly_stats")
        
        # Initialize LLM
        self.llm_config = None
        self.llm_processing_time = 0.0
        
        # Initialize statistics tracking
        self.stats = {
            'entities_processed': 0,
            'summaries_generated': 0,
            'articles_analyzed': 0,
            'stats_analyzed': 0,
            'errors': 0,
            'processing_time': 0.0,
            'llm_time': 0.0
        }
        
        mode_msg = "DRY RUN mode" if self.dry_run else "LIVE mode"
        self.logger.info(f"TrendingSummaryGenerator initialized with {lookback_hours}h lookback, {mode_msg}")
    
    def initialize_llm(self) -> bool:
        """Initialize the LLM client with preferred provider and fallback."""
        try:
            self.logger.info(f"Initializing {self.preferred_llm_provider.title()} LLM...")
            
            if self.preferred_llm_provider == 'gemini':
                # Initialize Gemini model with grounding for better NFL knowledge
                self.llm_config = initialize_model(
                    provider="gemini",
                    model_type="flash",  # Fast and capable
                    grounding_enabled=True  # Enable Google Search for real-time NFL info
                )
                self.logger.info("Gemini LLM client initialized successfully")
                return True
            elif self.preferred_llm_provider == 'deepseek':
                # Initialize DeepSeek directly
                self.llm_config = initialize_model(
                    provider="deepseek",
                    model_type="chat",
                    grounding_enabled=False
                )
                self.logger.info("DeepSeek LLM client initialized successfully")
                return True
            else:
                raise ValueError(f"Unknown LLM provider: {self.preferred_llm_provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.preferred_llm_provider.title()} LLM: {e}")
            
            # Try fallback if primary provider fails
            if self.preferred_llm_provider != 'deepseek':
                self.logger.info("Falling back to DeepSeek...")
                try:
                    self.llm_config = initialize_model(
                        provider="deepseek",
                        model_type="chat",
                        grounding_enabled=False
                    )
                    self.logger.info("DeepSeek LLM client initialized successfully as fallback")
                    return True
                except Exception as e2:
                    self.logger.error(f"Failed to initialize fallback DeepSeek: {e2}")
            
            return False
    
    def determine_entity_type(self, entity_id: str) -> str:
        """Determine if entity_id is a player or team.
        
        Args:
            entity_id: Entity ID to classify
            
        Returns:
            'player' or 'team'
        """
        # Player IDs typically follow pattern: 00-0033873
        # Team IDs are typically 2-3 letter abbreviations: KC, NYJ, etc.
        if len(entity_id) >= 8 and '-' in entity_id:
            return 'player'
        else:
            return 'team'
    
    def get_entity_name(self, entity_id: str, entity_type: str) -> Optional[str]:
        """Get the display name for an entity.
        
        Args:
            entity_id: Entity ID
            entity_type: 'player' or 'team'
            
        Returns:
            Entity display name or None if not found
        """
        try:
            if entity_type == 'player':
                response = self.players_db.supabase.table("players").select(
                    "full_name, latest_team, position"
                ).eq("player_id", entity_id).limit(1).execute()
                
                if hasattr(response, 'data') and response.data:
                    player_data = response.data[0]
                    name = player_data.get('full_name', entity_id)
                    team = player_data.get('latest_team', '')
                    position = player_data.get('position', '')
                    
                    # Format: "Patrick Mahomes (KC - QB)"
                    if team and position:
                        return f"{name} ({team} - {position})"
                    elif team:
                        return f"{name} ({team})"
                    else:
                        return name
                        
            elif entity_type == 'team':
                response = self.teams_db.supabase.table("teams").select(
                    "team_name"
                ).eq("team_abbr", entity_id).limit(1).execute()
                
                if hasattr(response, 'data') and response.data:
                    team_data = response.data[0]
                    return team_data.get('team_name', entity_id)
            
            return entity_id  # Fallback to ID if name not found
            
        except Exception as e:
            self.logger.warning(f"Could not get name for {entity_type} {entity_id}: {e}")
            return entity_id
    
    def get_recent_articles_for_entity(self, entity_id: str, entity_type: str) -> List[Dict[str, Any]]:
        """Get recent articles mentioning the entity.
        
        Args:
            entity_id: Entity ID
            entity_type: 'player' or 'team'
            
        Returns:
            List of article dictionaries
        """
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
            cutoff_time_str = cutoff_time.isoformat()
            
            # Get articles mentioning this entity in the timeframe
            response = self.entity_links_db.supabase.table("article_entity_links").select(
                "article_id, SourceArticles!inner(id, headline, Content, Author, created_at, source, publishedAt)"
            ).eq("entity_id", entity_id).eq("entity_type", entity_type).gte("SourceArticles.created_at", cutoff_time_str).execute()
            
            if not hasattr(response, 'data') or not response.data:
                return []
            
            # Extract articles from response
            articles = []
            for link in response.data:
                article = link.get('SourceArticles')
                if article:
                    articles.append(article)
            
            self.logger.debug(f"Found {len(articles)} recent articles for {entity_type} {entity_id}")
            return articles
            
        except Exception as e:
            self.logger.warning(f"Error fetching articles for {entity_id}: {e}")
            return []
    
    def get_recent_stats_for_entity(self, entity_id: str, entity_type: str) -> List[Dict[str, Any]]:
        """Get recent stats for the entity (players only).
        
        Args:
            entity_id: Entity ID
            entity_type: 'player' or 'team'
            
        Returns:
            List of stats dictionaries
        """
        try:
            # Only get stats for players
            if entity_type != 'player':
                return []
            
            # Get recent season stats (current season)
            current_season = datetime.now().year
            if datetime.now().month < 9:  # Before September
                current_season -= 1
            
            # Get all stats for the player from current season
            response = self.stats_db.supabase.table("player_weekly_stats").select(
                "*"
            ).eq("player_id", entity_id).execute()
            
            if hasattr(response, 'data') and response.data:
                # Filter for current season and recent weeks
                recent_stats = []
                for stat in response.data:
                    try:
                        # Parse game_id to extract season: {season}_{week}_{team1}_{team2}
                        game_parts = stat['game_id'].split('_')
                        if len(game_parts) >= 2:
                            season = int(game_parts[0])
                            
                            # Only include current season stats
                            if season == current_season:
                                recent_stats.append(stat)
                    except (ValueError, IndexError):
                        continue
                
                # Sort by week descending and limit to last 5 weeks for trending context
                recent_stats.sort(key=lambda x: int(x['game_id'].split('_')[1]), reverse=True)
                recent_stats = recent_stats[:5]
                
                self.logger.debug(f"Found {len(recent_stats)} recent stats for player {entity_id}")
                return recent_stats
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Error fetching stats for {entity_id}: {e}")
            return []
    
    def create_trending_summary_prompt(self, entity_id: str, entity_type: str, entity_name: str,
                                     articles: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> str:
        """Create the prompt for generating a trending entity summary.
        
        Args:
            entity_id: Entity ID
            entity_type: 'player' or 'team'
            entity_name: Display name of entity
            articles: List of recent articles
            stats: List of recent stats
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Create a trending NFL summary for {entity_name} ({entity_type}).

RECENT NEWS ({min(len(articles), 5)} articles):
"""
        
        # Limit to 5 most recent articles with shorter content
        for i, article in enumerate(articles[:5]):
            title = article.get('headline', 'No title')
            content = article.get('Content', '')[:200]  # Shorter content to avoid token limits
            
            prompt += f"""
{i+1}. {title}
   {content}...

"""
        
        if stats and entity_type == 'player':
            prompt += f"""RECENT STATS ({len(stats)} games):
"""
            for stat in stats:
                # Extract week from game_id
                game_parts = stat.get('game_id', '').split('_')
                week = game_parts[1] if len(game_parts) >= 2 else '?'
                
                # Get key stats
                pass_yds = stat.get('passing_yards', 0)
                pass_tds = stat.get('passing_tds', 0)
                rush_yds = stat.get('rushing_yards', 0)
                rec_yds = stat.get('receiving_yards', 0)
                
                prompt += f"""Week {week}: {pass_yds} pass yds, {pass_tds} pass TDs, {rush_yds} rush yds, {rec_yds} rec yds
"""
        
        prompt += f"""
Write a 400-word trending summary explaining:
1. Why {entity_name} is trending now
2. Key developments from recent articles
3. Performance highlights (if player)
4. What NFL fans should know

Use an engaging, journalistic tone. Start with a compelling headline."""
        
        return prompt
    
    def generate_summary_with_llm(self, prompt: str) -> Optional[str]:
        """Generate a summary using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated summary or None if failed
        """
        if not self.llm_config:
            self.logger.error("LLM not initialized")
            return None
        
        try:
            start_time = datetime.now()
            
            # Create messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert NFL analyst and sports journalist who creates comprehensive, engaging trending summaries. Use Google Search for the most current NFL information. Always provide well-structured, informative content that explains why an entity is trending and what it means for NFL fans. Format with clear headings and engaging narrative flow."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Generate content
            summary = generate_content_with_model(
                model_config=self.llm_config,
                messages=messages,
                temperature=0.7,  # Balanced creativity and accuracy
                max_tokens=1500   # Allow for comprehensive summaries
            )
            
            llm_time = (datetime.now() - start_time).total_seconds()
            self.llm_processing_time += llm_time
            
            if summary and summary.strip():
                cleaned_summary = summary.strip()
                self.logger.debug(f"Generated trending summary ({len(cleaned_summary)} chars) in {llm_time:.2f}s using {self.llm_config['provider']}")
                return cleaned_summary
            
            self.logger.warning("LLM returned empty response")
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating summary with LLM: {e}")
            return None
    
    def save_trending_summary(self, trending_summary: TrendingSummary) -> bool:
        """Save the trending summary to the database.
        
        Args:
            trending_summary: TrendingSummary object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create record matching the actual table schema
            record = {
                'trending_content': trending_summary.summary_content,
                'source_articles': [],  # Empty array as shown in schema
                'source_starts': [],    # Empty array as shown in schema  
                'player_ids': [trending_summary.entity_id] if trending_summary.entity_type == 'player' else [],
                'team_ids': [trending_summary.entity_id] if trending_summary.entity_type == 'team' else []
            }
            
            if self.dry_run:
                self.logger.debug(f"DRY RUN: Would save trending summary for {trending_summary.entity_type} {trending_summary.entity_id}")
                self.logger.debug(f"Summary preview: {trending_summary.summary_content[:150]}...")
                return True
            
            result = self.trending_updates_db.insert_records([record])
            
            if result.get('success', False):
                self.logger.debug(f"Saved trending summary for {trending_summary.entity_type} {trending_summary.entity_id}")
                return True
            else:
                self.logger.error(f"Failed to save trending summary: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving trending summary: {e}")
            return False
    
    def process_trending_entity(self, entity_id: str) -> Optional[TrendingSummary]:
        """Process a single trending entity and generate its summary.
        
        Args:
            entity_id: Entity ID to process
            
        Returns:
            TrendingSummary object or None if failed
        """
        try:
            # Determine entity type
            entity_type = self.determine_entity_type(entity_id)
            self.logger.info(f"Processing trending {entity_type}: {entity_id}")
            
            # Get entity name
            entity_name = self.get_entity_name(entity_id, entity_type)
            
            # Get recent content
            articles = self.get_recent_articles_for_entity(entity_id, entity_type)
            stats = self.get_recent_stats_for_entity(entity_id, entity_type)
            
            # Check if we have content to summarize
            if not articles and not stats:
                self.logger.warning(f"No recent content found for {entity_type} {entity_id}, skipping")
                return None
            
            self.logger.info(f"Found {len(articles)} articles and {len(stats)} stats for {entity_id}")
            
            # Update analytics
            self.stats['articles_analyzed'] += len(articles)
            self.stats['stats_analyzed'] += len(stats)
            
            # Create summary prompt
            prompt = self.create_trending_summary_prompt(entity_id, entity_type, entity_name, articles, stats)
            
            # Generate summary
            summary_content = self.generate_summary_with_llm(prompt)
            
            if not summary_content:
                self.logger.error(f"Failed to generate summary for {entity_type} {entity_id}")
                return None
            
            # Create trending summary object
            trending_summary = TrendingSummary(
                entity_id=entity_id,
                entity_type=entity_type,
                entity_name=entity_name,
                summary_content=summary_content,
                source_article_count=len(articles),
                source_stat_count=len(stats),
                generated_at=datetime.now(timezone.utc).isoformat()
            )
            
            return trending_summary
            
        except Exception as e:
            self.logger.error(f"Error processing trending entity {entity_id}: {e}")
            self.stats['errors'] += 1
            return None
    
    def generate_trending_summaries(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Generate summaries for a list of trending entities.
        
        Args:
            entity_ids: List of entity IDs to process
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = datetime.now()
        self.logger.info(f"Starting trending summary generation for {len(entity_ids)} entities...")
        
        try:
            # Initialize LLM
            if not self.initialize_llm():
                return {
                    'success': False,
                    'error': 'Failed to initialize LLM client',
                    'stats': self.stats
                }
            
            if not entity_ids:
                self.logger.warning("No entity IDs provided")
                return {
                    'success': True,
                    'message': 'No entity IDs to process',
                    'stats': self.stats
                }
            
            # Process each entity
            generated_summaries = []
            for entity_id in entity_ids:
                trending_summary = self.process_trending_entity(entity_id.strip())
                
                if trending_summary:
                    # Save to database
                    if self.save_trending_summary(trending_summary):
                        generated_summaries.append(trending_summary)
                        self.stats['summaries_generated'] += 1
                        self.logger.info(f"✅ Generated trending summary for {trending_summary.entity_type} {entity_id}")
                    else:
                        self.logger.error(f"Failed to save summary for {entity_id}")
                
                self.stats['entities_processed'] += 1
            
            # Calculate final statistics
            self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.stats['llm_time'] = self.llm_processing_time
            
            self.logger.info("Trending summary generation completed")
            self.logger.info(f"Processed {self.stats['entities_processed']} entities")
            self.logger.info(f"Generated {self.stats['summaries_generated']} summaries")
            self.logger.info(f"Analyzed {self.stats['articles_analyzed']} articles")
            self.logger.info(f"Analyzed {self.stats['stats_analyzed']} stat records")
            self.logger.info(f"Encountered {self.stats['errors']} errors")
            self.logger.info(f"Total processing time: {self.stats['processing_time']:.2f}s")
            self.logger.info(f"LLM processing time: {self.llm_processing_time:.2f}s")
            
            return {
                'success': True,
                'summaries': [summary.to_dict() for summary in generated_summaries],
                'stats': self.stats
            }
            
        except Exception as e:
            self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.stats['llm_time'] = self.llm_processing_time
            self.logger.error(f"Error during trending summary generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }


def parse_entity_ids_from_input(input_str: str) -> List[str]:
    """Parse entity IDs from various input formats.
    
    Args:
        input_str: Input string containing entity IDs
        
    Returns:
        List of clean entity IDs
    """
    if not input_str:
        return []
    
    # Split by common delimiters and clean
    import re
    # Split on comma, space, newline, or tab
    entity_ids = re.split(r'[,\s\n\t]+', input_str.strip())
    
    # Filter out empty strings
    return [eid.strip() for eid in entity_ids if eid.strip()]


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive summaries for trending NFL entities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --entity-ids "00-0033873,KC,NYJ"        # Generate summaries for specific entities
  %(prog)s --input-file trending_entities.txt      # Read entity IDs from file
  %(prog)s --from-stdin                             # Read entity IDs from stdin (pipe)
  %(prog)s --entity-ids "00-0033873" --dry-run     # Preview without saving
  %(prog)s --hours 48 --entity-ids "KC"            # Use 48-hour lookback period
  
  # Pipeline with trending detector:
  python scripts/trending_topic_detector.py --entity-ids-only | python scripts/trending_summary_generator.py --from-stdin
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--entity-ids', type=str,
                            help='Comma-separated list of entity IDs to process')
    input_group.add_argument('--input-file', type=str,
                            help='File containing entity IDs (one per line or comma-separated)')
    input_group.add_argument('--from-stdin', action='store_true',
                            help='Read entity IDs from stdin (for piping from trending detector)')
    
    # Configuration options
    parser.add_argument('--hours', type=int, default=72,
                       help='Number of hours to look back for content (default: 72)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview generation without saving to database')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--llm-provider', choices=['gemini', 'deepseek'], default='gemini',
                       help='LLM provider to use (default: gemini with deepseek fallback)')
    parser.add_argument('--output-format', choices=['summary', 'json'], default='summary',
                       help='Output format (default: summary)')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Trending Summary Generator")
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No summaries will be saved to database")
    
    try:
        # Get entity IDs from specified input source
        entity_ids = []
        
        if args.entity_ids:
            entity_ids = parse_entity_ids_from_input(args.entity_ids)
            logger.info(f"Processing {len(entity_ids)} entity IDs from command line")
            
        elif args.input_file:
            try:
                with open(args.input_file, 'r') as f:
                    file_content = f.read()
                entity_ids = parse_entity_ids_from_input(file_content)
                logger.info(f"Processing {len(entity_ids)} entity IDs from file: {args.input_file}")
            except FileNotFoundError:
                logger.error(f"Input file not found: {args.input_file}")
                return 1
            except Exception as e:
                logger.error(f"Error reading input file: {e}")
                return 1
                
        elif args.from_stdin:
            try:
                stdin_content = sys.stdin.read()
                entity_ids = parse_entity_ids_from_input(stdin_content)
                logger.info(f"Processing {len(entity_ids)} entity IDs from stdin")
            except Exception as e:
                logger.error(f"Error reading from stdin: {e}")
                return 1
        
        if not entity_ids:
            logger.error("No entity IDs found to process")
            return 1
        
        # Initialize and run generator
        generator = TrendingSummaryGenerator(
            lookback_hours=args.hours,
            dry_run=args.dry_run,
            preferred_llm_provider=args.llm_provider
        )
        
        result = generator.generate_trending_summaries(entity_ids)
        
        if result['success']:
            logger.info("✅ Trending summary generation completed successfully!")
            stats = result['stats']
            
            if args.output_format == 'json':
                # Output JSON format
                output = {
                    'success': True,
                    'summaries': result.get('summaries', []),
                    'stats': stats
                }
                print(json.dumps(output, indent=2))
            else:
                # Output summary format
                print(f"\n📊 Trending Summary Generation Results:")
                print(f"   Entities processed: {stats['entities_processed']}")
                print(f"   Summaries generated: {stats['summaries_generated']}")
                print(f"   Articles analyzed: {stats['articles_analyzed']}")
                print(f"   Stats analyzed: {stats['stats_analyzed']}")
                print(f"   Errors encountered: {stats['errors']}")
                print(f"   Processing time: {stats['processing_time']:.2f}s")
                print(f"   LLM time: {stats['llm_time']:.2f}s")
                
                if stats['entities_processed'] > 0:
                    success_rate = (stats['summaries_generated'] / stats['entities_processed']) * 100
                    print(f"   Success rate: {success_rate:.1f}%")
                
                if args.dry_run:
                    print(f"\n🔍 DRY RUN - No summaries were saved to database")
        else:
            logger.error(f"❌ Trending summary generation failed: {result.get('error', 'Unknown error')}")
            if args.output_format == 'json':
                print(json.dumps({'success': False, 'error': result.get('error', 'Unknown error')}))
            else:
                print(f"\n📊 Stats: {result['stats']}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
