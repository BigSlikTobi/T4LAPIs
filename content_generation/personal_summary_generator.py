#!/usr/bin/env python3
"""
Personalized Summary Generator for NFL Content

This module provides AI-powered personalized summary generation for NFL teams and players.
Key features:
- Loops through users and their preferences (teams/players)
- Gathers recent articles and stats for each entity
- Generates personalized summaries using Gemini LLM with Google Search grounding
- Stores results in generated_updates table
- Handles rolling summary updates and prevents duplicates

The generator uses:
- Gemini 2.5 Flash model with Google Search grounding for enhanced accuracy
- Recent articles from SourceArticles via article_entity_links
- Player stats from player_weekly_stats
- User preferences from user_preferences table
"""

import os
import sys
import logging
import uuid
import json
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.utils.database import DatabaseManager
from src.core.llm.llm_setup import initialize_model, generate_content_with_model


class PersonalizedSummaryGenerator:
    """Generates personalized NFL summaries for users based on their preferences."""
    
    def __init__(self, lookback_hours: int = 24, dry_run: bool = False, 
                 target_user_id: Optional[str] = None, preferred_llm_provider: str = 'gemini'):
        """Initialize the summary generator.
        
        Args:
            lookback_hours: How many hours to look back for new content
            dry_run: If True, don't save summaries to database
            target_user_id: If provided, only generate summaries for this user
            preferred_llm_provider: Preferred LLM provider ('gemini' or 'deepseek')
        """
        self.logger = logging.getLogger(__name__)
        self.lookback_hours = lookback_hours
        self.dry_run = dry_run
        self.target_user_id = target_user_id
        self.preferred_llm_provider = preferred_llm_provider
        
        # Initialize database managers for different tables
        self.users_db = DatabaseManager("users")
        self.preferences_db = DatabaseManager("user_preferences") 
        self.generated_updates_db = DatabaseManager("generated_updates")
        self.entity_links_db = DatabaseManager("article_entity_links")
        self.stats_db = DatabaseManager("player_weekly_stats")
        
        # Initialize Gemini LLM with grounding
        self.llm_config = None
        self.llm_processing_time = 0.0
        
        # Initialize statistics tracking
        self.stats = {
            'users_processed': 0,
            'preferences_processed': 0,
            'summaries_generated': 0,
            'errors': 0,
            'processing_time': 0.0,
            'llm_time': 0.0
        }
        
        mode_msg = "DRY RUN mode" if self.dry_run else "LIVE mode"
        user_msg = f" (user: {self.target_user_id})" if self.target_user_id else " (all users)"
        self.logger.info(f"PersonalizedSummaryGenerator initialized with {lookback_hours}h lookback, {mode_msg}{user_msg}")
    
    def initialize_llm(self):
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
    
    def get_all_users_with_preferences(self) -> List[Dict[str, Any]]:
        """Get all users who have preferences set.
        
        Returns:
            List of user dictionaries with their preferences
        """
        try:
            self.logger.info("Fetching users with preferences...")
            
            # Get users (filter by target_user_id if specified)
            if self.target_user_id:
                users_response = self.users_db.supabase.table("users").select("user_id").eq("user_id", self.target_user_id).execute()
                self.logger.info(f"Filtering for specific user: {self.target_user_id}")
            else:
                users_response = self.users_db.supabase.table("users").select("user_id").execute()
            
            if not hasattr(users_response, 'data') or not users_response.data:
                if self.target_user_id:
                    self.logger.warning(f"User {self.target_user_id} not found in database")
                else:
                    self.logger.warning("No users found in database")
                return []
            
            users_with_preferences = []
            
            for user in users_response.data:
                user_id = user['user_id']
                
                # Get preferences for this user
                prefs_response = self.preferences_db.supabase.table("user_preferences").select(
                    "preference_id, entity_id, entity_type, created_at"
                ).eq("user_id", user_id).execute()
                
                if hasattr(prefs_response, 'data') and prefs_response.data:
                    users_with_preferences.append({
                        'user_id': user_id,
                        'preferences': prefs_response.data
                    })
            
            self.logger.info(f"Found {len(users_with_preferences)} users with preferences")
            return users_with_preferences
            
        except Exception as e:
            self.logger.error(f"Error fetching users with preferences: {e}")
            return []
    
    def get_previous_summary(self, user_id: str, entity_id: str, entity_type: str) -> Optional[str]:
        """Get the most recent summary for a user-entity combination.
        
        Args:
            user_id: User ID
            entity_id: Entity ID (player_id or team abbreviation)
            entity_type: Type of entity ('player' or 'team')
            
        Returns:
            Previous summary content or None if not found
        """
        try:
            # Note: generated_updates table doesn't have entity_id/entity_type columns yet
            # For now, we'll skip this check and generate fresh summaries
            # TODO: Add entity_id and entity_type columns to generated_updates table
            self.logger.info(f"Skipping previous summary check - table schema update needed")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting previous summary for {user_id}-{entity_id}: {e}")
            return None
    
    def get_new_articles_for_entity(self, entity_id: str, entity_type: str, since_hours: int = 24) -> List[Dict[str, Any]]:
        """Get new articles mentioning the entity since the specified time.
        
        Filters articles by their creation date (SourceArticles.created_at) for accurate timeline analysis.
        
        Args:
            entity_id: Entity ID to search for
            entity_type: Type of entity ('player' or 'team')
            since_hours: Number of hours to look back
            
        Returns:
            List of article dictionaries
        """
        try:
            # Calculate the cutoff time
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            cutoff_time_str = cutoff_time.isoformat()
            
            # Get articles that mention this entity in the specified timeframe
            # Filter by SourceArticles.created_at at the database level for better performance
            response = self.entity_links_db.supabase.table("article_entity_links").select(
                "article_id, SourceArticles!inner(id, headline, Content, Author, created_at, source)"
            ).eq("entity_id", entity_id).eq("entity_type", entity_type).gte("SourceArticles.created_at", cutoff_time_str).execute()
            
            if not hasattr(response, 'data') or not response.data:
                return []
            
            # Extract articles from the response (already filtered by database)
            new_articles = []
            for link in response.data:
                article = link.get('SourceArticles')
                if article:
                    new_articles.append(article)
            
            self.logger.debug(f"Found {len(new_articles)} new articles for {entity_type} {entity_id}")
            return new_articles
            
        except Exception as e:
            self.logger.warning(f"Error fetching new articles for {entity_id}: {e}")
            return []
    
    def get_new_stats_for_entity(self, entity_id: str, entity_type: str, since_hours: int = 24) -> List[Dict[str, Any]]:
        """Get new player stats since the specified time.
        
        Args:
            entity_id: Entity ID (only applicable for players)
            entity_type: Type of entity ('player' or 'team')
            since_hours: Number of hours to look back
            
        Returns:
            List of stats dictionaries
        """
        try:
            # Only get stats for players
            if entity_type != 'player':
                return []
            
            # For simplicity, we'll get stats from the last few weeks
            # since player stats are weekly and don't have hour-level timestamps
            current_season = datetime.now().year
            if datetime.now().month < 9:  # Before September
                current_season -= 1
            
            # Since we don't have season/week columns, we'll filter by game_id pattern
            # game_id format: {season}_{week}_{team1}_{team2}
            # Get all stats for the player and filter by recent weeks
            response = self.stats_db.supabase.table("player_weekly_stats").select(
                "*"
            ).eq("player_id", entity_id).execute()
            
            if hasattr(response, 'data') and response.data:
                # Parse all entries, grouping by season
                by_season = {}
                for stat in response.data:
                    try:
                        parts = stat.get('game_id', '').split('_')
                        if len(parts) >= 2:
                            season = int(parts[0])
                            by_season.setdefault(season, []).append(stat)
                    except (ValueError, IndexError):
                        continue

                # Prefer current season; if not available, fallback to latest available season
                season_stats = by_season.get(current_season)
                if not season_stats and by_season:
                    season_stats = by_season[max(by_season.keys())]

                if not season_stats:
                    return []

                # Sort by week desc and take last 3
                def _wk(s: Dict[str, Any]) -> int:
                    try:
                        return int(s.get('game_id', '0_0').split('_')[1])
                    except Exception:
                        return -1

                season_stats.sort(key=_wk, reverse=True)
                season_stats = season_stats[:3]

                self.logger.debug(f"Found {len(season_stats)} recent stats for player {entity_id}")
                return season_stats
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Error fetching new stats for {entity_id}: {e}")
            return []
    
    def create_summary_prompt(self, entity_id: str, entity_type: str, articles: List[Dict[str, Any]], 
                            stats: List[Dict[str, Any]], previous_summary: Optional[str] = None) -> str:
        """Create the prompt for generating a personalized summary.
        
        Args:
            entity_id: Entity ID
            entity_type: Type of entity ('player' or 'team')
            articles: List of new articles
            stats: List of new stats
            previous_summary: Previous summary for context
            
        Returns:
            Formatted prompt string
        """
        entity_name = entity_id  # We'll use the ID as name for now, could be enhanced
        
        prompt = f"""You are an expert NFL analyst creating personalized updates for a fan. Create a comprehensive summary about {entity_name} ({entity_type}).

CONTEXT:
Entity: {entity_name} ({entity_type})
Time Period: Last 24 hours
"""
        
        if previous_summary:
            prompt += f"""
Previous Summary (for context):
{previous_summary[:500]}...
"""
        
        prompt += f"""
NEW INFORMATION:

Recent Articles ({len(articles)} found):
"""
        
        for i, article in enumerate(articles[:5]):  # Limit to 5 articles
            title = article.get('headline', 'No title')
            content = article.get('Content', '')[:300]  # Truncate content
            source = article.get('source', 'Unknown')
            created = article.get('created_at', 'Unknown date')
            
            prompt += f"""
Article {i+1}:
Title: {title}
Source: {source}
Date: {created}
Content: {content}...
"""
        
        if stats and entity_type == 'player':
            prompt += f"""
Recent Stats ({len(stats)} weeks):
"""
            for stat in stats:
                # Extract week and season from game_id: {season}_{week}_{team1}_{team2}
                game_parts = stat.get('game_id', '').split('_')
                week = game_parts[1] if len(game_parts) >= 2 else 'Unknown'
                season = game_parts[0] if len(game_parts) >= 1 else 'Unknown'
                passing_yards = stat.get('passing_yards', 0)
                rushing_yards = stat.get('rushing_yards', 0)
                receiving_yards = stat.get('receiving_yards', 0)
                
                prompt += f"""
Week {week}, {season}: Passing: {passing_yards} yds, Rushing: {rushing_yards} yds, Receiving: {receiving_yards} yds
"""
        
        prompt += f"""
INSTRUCTIONS:
1. Create a comprehensive but concise summary (200-400 words)
2. Focus on the most important and recent developments
3. Include key statistics if this is a player
4. Mention any notable games, performances, or news
5. If this is an update to a previous summary, highlight what's new
6. Use an engaging, informative tone suitable for a dedicated fan
7. Structure the response with clear paragraphs

Generate the personalized summary now:"""
        
        return prompt
    
    def generate_summary_with_llm(self, prompt: str) -> Optional[str]:
        """Generate a summary using the Gemini LLM with grounding.
        
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
                    "content": "You are an expert NFL analyst who creates personalized, engaging summaries for football fans. Use Google Search to find the most current and accurate NFL information. Always provide comprehensive but concise updates focusing on the most important recent developments. Format your response with clear headings and bullet points for easy reading."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Generate content using the new unified interface
            summary = generate_content_with_model(
                model_config=self.llm_config,
                messages=messages,
                temperature=0.7,  # Slightly higher for more engaging content
                max_tokens=1000   # Allow for longer summaries
            )
            
            llm_time = (datetime.now() - start_time).total_seconds()
            self.llm_processing_time += llm_time
            
            if summary and summary.strip():
                cleaned_summary = summary.strip()
                self.logger.debug(f"Generated summary ({len(cleaned_summary)} chars) in {llm_time:.2f}s using {self.llm_config['provider']}")
                return cleaned_summary
            
            self.logger.warning("LLM returned empty response")
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating summary with LLM: {e}")
            return None
    
    def save_generated_summary(self, user_id: str, entity_id: str, entity_type: str, 
                             summary: str, source_article_ids: List[int], 
                             source_stat_ids: List[str]) -> bool:
        """Save the generated summary to the database.
        
        Args:
            user_id: User ID
            entity_id: Entity ID
            entity_type: Entity type
            summary: Generated summary content
            source_article_ids: List of article IDs used
            source_stat_ids: List of stat IDs used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            
            record = {
                'update_id': str(uuid.uuid4()),
                'user_id': user_id,
                'created_at': current_time,
                'update_content': summary,
                'source_article_ids': source_article_ids + [entity_id],  # Include entity_id in source_article_ids for now
                'source_stat_ids': source_stat_ids
                # TODO: Add entity_id and entity_type columns to generated_updates table
                # 'entity_id': entity_id,
                # 'entity_type': entity_type,
            }
            
            if self.dry_run:
                self.logger.debug(f"DRY RUN: Would save summary for user {user_id}, entity {entity_id}")
                self.logger.debug(f"Summary preview: {summary[:100]}...")
                return True
            
            result = self.generated_updates_db.insert_records([record])
            
            if result.get('success', False):
                self.logger.debug(f"Saved summary for user {user_id}, entity {entity_id}")
                return True
            else:
                self.logger.error(f"Failed to save summary: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving generated summary: {e}")
            return False
    
    def preview_generation_stats(self) -> Dict[str, int]:
        """Preview statistics about what would be generated without actually running.
        
        Returns:
            Dictionary with preview statistics
        """
        try:
            # Get users with preferences
            users_with_preferences = self.get_all_users_with_preferences()
            
            total_preferences = sum(len(user_data['preferences']) for user_data in users_with_preferences)
            
            # Estimate how many summaries would be generated
            # For now, assume all preferences will generate summaries
            estimated_summaries = total_preferences
            
            return {
                'users_with_preferences': len(users_with_preferences),
                'total_preferences': total_preferences,
                'estimated_summaries': estimated_summaries
            }
            
        except Exception as e:
            self.logger.error(f"Error getting preview stats: {e}")
            return {
                'users_with_preferences': 0,
                'total_preferences': 0,
                'estimated_summaries': 0
            }
    
    def process_user_preference(self, user_id: str, preference: Dict[str, Any]) -> bool:
        """Process a single user preference and generate summary if needed.
        
        Args:
            user_id: User ID
            preference: Preference dictionary
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            entity_id = preference['entity_id']
            entity_type = preference['entity_type']
            
            self.logger.info(f"Processing preference: user {user_id}, {entity_type} {entity_id}")
            
            # Get previous summary for context
            previous_summary = self.get_previous_summary(user_id, entity_id, entity_type)
            
            # Get new articles and stats
            new_articles = self.get_new_articles_for_entity(entity_id, entity_type, self.lookback_hours)
            new_stats = self.get_new_stats_for_entity(entity_id, entity_type, self.lookback_hours)
            
            # Check if we have new content to summarize
            if not new_articles and not new_stats:
                self.logger.debug(f"No new content for {entity_type} {entity_id}, skipping")
                return True
            
            self.logger.info(f"Found {len(new_articles)} articles and {len(new_stats)} stats for {entity_id}")
            
            # Create the summary prompt
            prompt = self.create_summary_prompt(entity_id, entity_type, new_articles, new_stats, previous_summary)
            
            # Generate summary with LLM
            summary = self.generate_summary_with_llm(prompt)
            
            if not summary:
                self.logger.error(f"Failed to generate summary for {entity_type} {entity_id}")
                return False
            
            # Prepare source IDs
            source_article_ids = [article['id'] for article in new_articles if article.get('id')]
            source_stat_ids = [stat.get('stat_id', '') for stat in new_stats if stat.get('stat_id')]
            
            # Save the summary
            if self.save_generated_summary(user_id, entity_id, entity_type, summary, 
                                         source_article_ids, source_stat_ids):
                self.stats['summaries_generated'] += 1
                self.logger.info(f"Successfully generated summary for user {user_id}, {entity_type} {entity_id}")
                return True
            else:
                self.logger.error(f"Failed to save summary for {entity_type} {entity_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing user preference: {e}")
            self.stats['errors'] += 1
            return False
    
    def run_personalized_summary_generation(self) -> Dict[str, Any]:
        """Run the complete personalized summary generation process.
        
        Returns:
            Dictionary with results and statistics
        """
        start_time = datetime.now()
        self.logger.info("Starting personalized summary generation...")
        
        try:
            # Initialize LLM
            if not self.initialize_llm():
                return {
                    'success': False,
                    'error': 'Failed to initialize LLM client',
                    'stats': self.stats
                }
            
            # Get all users with preferences
            users_with_preferences = self.get_all_users_with_preferences()
            
            if not users_with_preferences:
                self.logger.warning("No users with preferences found")
                return {
                    'success': True,
                    'message': 'No users with preferences found',
                    'stats': self.stats
                }
            
            # Process each user
            for user_data in users_with_preferences:
                user_id = user_data['user_id']
                preferences = user_data['preferences']
                
                self.logger.info(f"Processing user {user_id} with {len(preferences)} preferences")
                
                # Process each preference
                for preference in preferences:
                    self.process_user_preference(user_id, preference)
                    self.stats['preferences_processed'] += 1
                
                self.stats['users_processed'] += 1
            
            # Calculate final statistics
            self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.stats['llm_time'] = self.llm_processing_time
            
            self.logger.info("Personalized summary generation completed successfully")
            self.logger.info(f"Processed {self.stats['users_processed']} users")
            self.logger.info(f"Processed {self.stats['preferences_processed']} preferences")
            self.logger.info(f"Generated {self.stats['summaries_generated']} summaries")
            self.logger.info(f"Encountered {self.stats['errors']} errors")
            self.logger.info(f"Total processing time: {self.stats['processing_time']:.2f}s")
            self.logger.info(f"LLM processing time: {self.llm_processing_time:.2f}s")
            
            return {
                'success': True,
                'stats': self.stats
            }
            
        except Exception as e:
            self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.stats['llm_time'] = self.llm_processing_time
            self.logger.error(f"Error during personalized summary generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }


def main():
    """Main function to run the personalized summary generator."""
    parser = argparse.ArgumentParser(
        description='Generate personalized NFL summaries for users based on their preferences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Generate summaries for last 24 hours
  %(prog)s --hours 48                   # Generate summaries for last 48 hours
  %(prog)s --user-id user123            # Generate summaries for specific user only
  %(prog)s --verbose                    # Enable detailed logging
  %(prog)s --dry-run                    # Preview what would be generated without saving
        """
    )
    
    parser.add_argument('--hours', type=int, default=24,
                       help='Number of hours to look back for new content (default: 24)')
    parser.add_argument('--user-id', type=str,
                       help='Generate summaries for specific user only (default: all users)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview generation without saving to database')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--llm-provider', choices=['gemini', 'deepseek'], default='gemini',
                       help='LLM provider to use (default: gemini with deepseek fallback)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics without generating summaries')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Personalized Summary Generator")
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No summaries will be saved to database")
    
    try:
        # Initialize and run the generator
        generator = PersonalizedSummaryGenerator(
            lookback_hours=args.hours,
            dry_run=args.dry_run,
            target_user_id=args.user_id,
            preferred_llm_provider=args.llm_provider
        )
        
        if args.stats_only:
            # Just show preview statistics
            stats = generator.preview_generation_stats()
            print(f"\n📊 Generation Preview:")
            print(f"   Users with preferences: {stats.get('users_with_preferences', 0)}")
            print(f"   Total preferences: {stats.get('total_preferences', 0)}")
            print(f"   Estimated summaries: {stats.get('estimated_summaries', 0)}")
            return 0
        
        result = generator.run_personalized_summary_generation()
        
        if result['success']:
            logger.info("✅ Personalized summary generation completed successfully!")
            stats = result['stats']
            print(f"\n📊 Generation Results:")
            print(f"   Users processed: {stats['users_processed']}")
            print(f"   Preferences processed: {stats['preferences_processed']}")
            print(f"   Summaries generated: {stats['summaries_generated']}")
            print(f"   Errors encountered: {stats['errors']}")
            print(f"   Processing time: {stats['processing_time']:.2f}s")
            print(f"   LLM time: {stats['llm_time']:.2f}s")
            
            if stats['preferences_processed'] > 0:
                success_rate = (stats['summaries_generated'] / stats['preferences_processed']) * 100
                print(f"   Success rate: {success_rate:.1f}%")
                
            if args.dry_run:
                print(f"\n🔍 DRY RUN - No summaries were saved to database")
        else:
            logger.error(f"❌ Personalized summary generation failed: {result.get('error', 'Unknown error')}")
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
