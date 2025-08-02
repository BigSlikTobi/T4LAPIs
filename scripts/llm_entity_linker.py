"""
LLM-Enhanced Entity Linker Script - Enhanced Task 2

This script uses DeepSeek LLM for more accurate entity extraction from NFL articles,
then links the extracted entities to the existing entity dictionary for validation
and creates links in the article_entity_links table.
"""

import logging
import json
import time
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
import uuid

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.utils.database import DatabaseManager
from src.core.data.entity_linking import build_entity_dictionary
from src.core.llm.llm_init import get_deepseek_client
from src.core.utils.logging import get_logger


@dataclass
class LLMEntityMatch:
    """Represents an entity match found by LLM."""
    entity_name: str
    entity_id: str
    entity_type: str  # 'player' or 'team'
    confidence: str = "high"  # LLM-extracted entities have high confidence


class LLMEntityLinker:
    """LLM-enhanced entity linker using DeepSeek for entity extraction."""
    
    def __init__(self, batch_size: int = 50):
        """Initialize the LLM entity linker.
        
        Args:
            batch_size: Number of articles to process per batch
        """
        self.batch_size = batch_size
        self.logger = get_logger(__name__)
        
        # Initialize database managers
        self.articles_db = DatabaseManager("SourceArticles")
        self.links_db = DatabaseManager("article_entity_links")
        
        # Initialize LLM client
        self.llm_client = None
        
        # Entity dictionary for validation
        self.entity_dict = {}
        
        # Statistics tracking
        self.stats = {
            'articles_processed': 0,
            'llm_calls': 0,
            'entities_extracted': 0,
            'entities_validated': 0,
            'links_created': 0,
            'processing_time': 0.0,
            'llm_time': 0.0
        }
        
        self.logger.info(f"LLM Entity Linker initialized with batch size {batch_size}")
    
    def initialize_llm_and_entities(self) -> bool:
        """Initialize LLM client and build entity dictionary.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing LLM client and entity dictionary...")
            
            # Initialize DeepSeek LLM client
            self.llm_client = get_deepseek_client()
            
            # Test LLM connection
            if not self.llm_client.test_connection():
                self.logger.error("Failed to connect to DeepSeek LLM")
                return False
            
            # Build entity dictionary for validation
            self.logger.info("Building entity dictionary for validation...")
            self.entity_dict = build_entity_dictionary()
            
            if not self.entity_dict:
                self.logger.error("Failed to build entity dictionary")
                return False
            
            self.logger.info(f"Initialization successful:")
            self.logger.info(f"  - LLM client: Connected to DeepSeek")
            self.logger.info(f"  - Entity dictionary: {len(self.entity_dict)} patterns loaded")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM and entities: {e}")
            return False
    
    def get_unlinked_articles(self, batch_size: int) -> List[Dict[str, Any]]:
        """Fetch articles that don't have any entity links yet.
        
        Args:
            batch_size: Maximum number of articles to fetch
            
        Returns:
            List of article records with id and Content fields
        """
        try:
            self.logger.info(f"Fetching up to {batch_size} unlinked articles...")
            
            # Try RPC function first
            try:
                response = self.articles_db.supabase.rpc(
                    'get_unlinked_articles',
                    {'batch_limit': batch_size}
                ).execute()
                
                if hasattr(response, 'error') and response.error:
                    raise Exception("RPC function not available")
                
                if hasattr(response, 'data') and response.data:
                    articles = response.data
                    # Ensure the data has the Content field
                    for article in articles:
                        if 'text' in article and 'Content' not in article:
                            article['Content'] = article['text']
                    
                    self.logger.info(f"Found {len(articles)} unlinked articles via RPC")
                    return articles
                    
            except Exception:
                self.logger.info("RPC function not available, using fallback query...")
                
                # Fallback: get articles that might not have links
                response = self.articles_db.supabase.table("SourceArticles").select(
                    "id, Content"
                ).filter('Content', 'neq', '').limit(batch_size).execute()
                
                if hasattr(response, 'error') and response.error:
                    self.logger.error(f"Fallback query failed: {response.error}")
                    return []
                
                if not hasattr(response, 'data') or not response.data:
                    self.logger.info("No articles found in fallback query")
                    return []
                
                articles = response.data
                self.logger.info(f"Found {len(articles)} articles via fallback query")
                return articles
            
        except Exception as e:
            self.logger.error(f"Exception while fetching unlinked articles: {e}")
            return []
    
    def extract_entities_with_llm(self, article_text: str) -> Tuple[List[str], List[str]]:
        """Extract entities from article text using LLM.
        
        Args:
            article_text: Article text to analyze
            
        Returns:
            Tuple of (player_names, team_names) lists
        """
        if not self.llm_client or not article_text:
            return [], []
        
        try:
            start_time = time.time()
            
            self.logger.debug("Calling LLM for entity extraction...")
            entities = self.llm_client.extract_entities(article_text)
            
            llm_time = time.time() - start_time
            self.stats['llm_time'] += llm_time
            self.stats['llm_calls'] += 1
            
            players = entities.get('players', [])
            teams = entities.get('teams', [])
            
            self.logger.debug(f"LLM extracted {len(players)} players and {len(teams)} teams in {llm_time:.2f}s")
            
            return players, teams
            
        except Exception as e:
            self.logger.error(f"Error during LLM entity extraction: {e}")
            return [], []
    
    def validate_and_link_entities(self, players: List[str], teams: List[str]) -> List[LLMEntityMatch]:
        """Validate extracted entities against the entity dictionary and create matches.
        
        Args:
            players: List of player names from LLM
            teams: List of team names from LLM
            
        Returns:
            List of validated entity matches
        """
        matches = []
        
        # Validate players
        for player_name in players:
            if not player_name or not player_name.strip():
                continue
                
            player_name = player_name.strip()
            self.stats['entities_extracted'] += 1
            
            # Try exact match first
            if player_name in self.entity_dict:
                entity_id = self.entity_dict[player_name]
                # Check if it's a player (not a team)
                if len(entity_id) > 3 or not entity_id.isupper():
                    matches.append(LLMEntityMatch(
                        entity_name=player_name,
                        entity_id=entity_id,
                        entity_type="player"
                    ))
                    self.stats['entities_validated'] += 1
                    continue
            
            # Try case-insensitive match
            player_lower = player_name.lower()
            for dict_name, entity_id in self.entity_dict.items():
                if dict_name.lower() == player_lower:
                    # Check if it's a player (not a team)
                    if len(entity_id) > 3 or not entity_id.isupper():
                        matches.append(LLMEntityMatch(
                            entity_name=dict_name,  # Use dictionary name for consistency
                            entity_id=entity_id,
                            entity_type="player"
                        ))
                        self.stats['entities_validated'] += 1
                        break
        
        # Validate teams
        for team_name in teams:
            if not team_name or not team_name.strip():
                continue
                
            team_name = team_name.strip()
            self.stats['entities_extracted'] += 1
            
            # Try exact match first
            if team_name in self.entity_dict:
                entity_id = self.entity_dict[team_name]
                # Check if it's a team (short uppercase code)
                if len(entity_id) <= 3 and entity_id.isupper():
                    matches.append(LLMEntityMatch(
                        entity_name=team_name,
                        entity_id=entity_id,
                        entity_type="team"
                    ))
                    self.stats['entities_validated'] += 1
                    continue
            
            # Try case-insensitive match
            team_lower = team_name.lower()
            for dict_name, entity_id in self.entity_dict.items():
                if dict_name.lower() == team_lower:
                    # Check if it's a team (short uppercase code)
                    if len(entity_id) <= 3 and entity_id.isupper():
                        matches.append(LLMEntityMatch(
                            entity_name=dict_name,  # Use dictionary name for consistency
                            entity_id=entity_id,
                            entity_type="team"
                        ))
                        self.stats['entities_validated'] += 1
                        break
        
        return matches
    
    def create_entity_links(self, article_id: int, matches: List[LLMEntityMatch]) -> bool:
        """Create entity links in the database.
        
        Args:
            article_id: ID of the article
            matches: List of validated entity matches
            
        Returns:
            True if successful, False otherwise
        """
        if not matches:
            return True
        
        try:
            # Remove duplicates (same entity_id and entity_type for same article)
            unique_matches = {}
            for match in matches:
                key = (match.entity_id, match.entity_type)
                if key not in unique_matches:
                    unique_matches[key] = match
            
            matches = list(unique_matches.values())
            
            if not matches:
                return True
            
            # Prepare records for insertion
            records = []
            for match in matches:
                record = {
                    'link_id': str(uuid.uuid4()),
                    'article_id': article_id,
                    'entity_id': match.entity_id,
                    'entity_type': match.entity_type
                }
                records.append(record)
            
            # Insert records
            result = self.links_db.insert_records(records)
            
            if result.get('success', False):
                self.stats['links_created'] += len(records)
                self.logger.debug(f"Created {len(records)} entity links for article {article_id}")
                return True
            else:
                self.logger.error(f"Failed to create entity links: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception while creating entity links: {e}")
            return False
    
    def process_article_batch(self, articles: List[Dict[str, Any]]) -> int:
        """Process a batch of articles with LLM entity extraction.
        
        Args:
            articles: List of article records
            
        Returns:
            Number of articles successfully processed
        """
        processed_count = 0
        
        for article in articles:
            try:
                article_id = article.get('id')
                content = article.get('Content', '') or article.get('text', '')
                
                if not article_id or not content:
                    self.logger.warning(f"Skipping article with missing ID or content: {article_id}")
                    continue
                
                self.logger.debug(f"Processing article {article_id}")
                
                # Extract entities using LLM
                players, teams = self.extract_entities_with_llm(content)
                
                if not players and not teams:
                    self.logger.debug(f"No entities extracted for article {article_id}")
                    processed_count += 1
                    self.stats['articles_processed'] += 1
                    continue
                
                # Validate entities and create matches
                matches = self.validate_and_link_entities(players, teams)
                
                if not matches:
                    self.logger.debug(f"No valid entities found for article {article_id}")
                    processed_count += 1
                    self.stats['articles_processed'] += 1
                    continue
                
                # Create entity links
                if self.create_entity_links(article_id, matches):
                    self.logger.info(f"Article {article_id}: created {len(matches)} entity links")
                    processed_count += 1
                else:
                    self.logger.error(f"Failed to create entity links for article {article_id}")
                
                self.stats['articles_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing article {article.get('id', 'unknown')}: {e}")
                continue
        
        return processed_count
    
    def run_llm_entity_linking(self, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """Main entry point to run the LLM-enhanced entity linking process.
        
        Args:
            max_batches: Maximum number of batches to process. If None, processes all.
        
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting LLM-enhanced entity linking process...")
            
            # Initialize LLM and entity dictionary
            if not self.initialize_llm_and_entities():
                return {
                    'success': False,
                    'error': 'Failed to initialize LLM and entity dictionary',
                    'stats': self.stats
                }
            
            total_processed = 0
            batch_count = 0
            
            while True:
                batch_count += 1
                self.logger.info(f"Processing batch {batch_count}...")
                
                # Check if we've reached the maximum number of batches
                if max_batches is not None and batch_count > max_batches:
                    self.logger.info(f"Reached maximum batch limit of {max_batches}")
                    break
                
                # Fetch next batch of unlinked articles
                articles = self.get_unlinked_articles(self.batch_size)
                
                if not articles:
                    self.logger.info("No more unlinked articles to process")
                    break
                
                # Process the batch
                batch_processed = self.process_article_batch(articles)
                total_processed += batch_processed
                
                self.logger.info(f"Batch {batch_count}: processed {batch_processed}/{len(articles)} articles")
                
                # If we processed fewer articles than the batch size, we might be done
                if len(articles) < self.batch_size:
                    self.logger.info("Reached end of available articles")
                    break
            
            # Calculate final statistics
            self.stats['processing_time'] = time.time() - start_time
            
            self.logger.info("LLM entity linking process completed successfully")
            self.logger.info(f"Total articles processed: {self.stats['articles_processed']}")
            self.logger.info(f"Total LLM calls: {self.stats['llm_calls']}")
            self.logger.info(f"Total entities extracted: {self.stats['entities_extracted']}")
            self.logger.info(f"Total entities validated: {self.stats['entities_validated']}")
            self.logger.info(f"Total links created: {self.stats['links_created']}")
            self.logger.info(f"Processing time: {self.stats['processing_time']:.2f}s")
            self.logger.info(f"LLM time: {self.stats['llm_time']:.2f}s")
            
            validation_rate = (self.stats['entities_validated'] / self.stats['entities_extracted'] * 100) if self.stats['entities_extracted'] > 0 else 0
            self.logger.info(f"Entity validation rate: {validation_rate:.1f}%")
            
            return {
                'success': True,
                'total_processed': total_processed,
                'stats': self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Error during LLM entity linking process: {e}")
            self.stats['processing_time'] = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the LLM entity linker
    linker = LLMEntityLinker(batch_size=3)
    result = linker.run_llm_entity_linking(max_batches=1)
    
    if result['success']:
        print("✅ LLM Entity Linking completed successfully!")
        print(f"📊 Articles processed: {result['stats']['articles_processed']}")
        print(f"🤖 LLM calls: {result['stats']['llm_calls']}")
        print(f"🔍 Entities extracted: {result['stats']['entities_extracted']}")
        print(f"✅ Entities validated: {result['stats']['entities_validated']}")
        print(f"🔗 Links created: {result['stats']['links_created']}")
        print(f"⏱️  Processing time: {result['stats']['processing_time']:.2f}s")
        print(f"🤖 LLM time: {result['stats']['llm_time']:.2f}s")
        
        if result['stats']['entities_extracted'] > 0:
            validation_rate = result['stats']['entities_validated'] / result['stats']['entities_extracted'] * 100
            print(f"📈 Validation rate: {validation_rate:.1f}%")
    else:
        print(f"❌ LLM Entity Linking failed: {result['error']}")
        print(f"📊 Stats: {result['stats']}")
