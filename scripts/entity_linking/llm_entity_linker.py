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
        """
        Initializes the LLM client and builds the entity dictionary for validation.

        This function sets up the necessary components for the entity linking process.
        It creates an instance of the LLM client and then calls a helper function
        to build a comprehensive dictionary of all known players and teams from the database.
        
        A second dictionary with lowercased keys is also created to make the
        validation process case-insensitive and more efficient.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        self.logger.info("Initializing LLM client and building entity dictionary...")
        try:
            # 1. Initialize the LLM Client
            self.llm_client = get_deepseek_client()
            self.logger.info("DeepSeek LLM client initialized successfully.")

            # 2. Build the primary entity dictionary from the database
            # This dictionary maps entity names (e.g., "Patrick Mahomes", "Kansas City Chiefs")
            # to their unique database IDs (e.g., "mahpa00", "KC").
            self.entity_dict = build_entity_dictionary()
            if not self.entity_dict:
                self.logger.error("Entity dictionary could not be built or is empty. Aborting.")
                return False
            
            # 3. Create a lowercased version for robust, case-insensitive matching.
            # This is the key optimization: we create a new dictionary where all keys
            # (player and team names) are lowercased. This avoids repeated string
            # lowercasing inside the validation loop and centralizes the logic.
            self.entity_dict_lower = {k.lower(): v for k, v in self.entity_dict.items()}

            self.logger.info(f"Entity dictionary built successfully with {len(self.entity_dict)} entries.")
            return True

        except Exception as e:
            self.logger.critical(f"A critical error occurred during initialization: {e}", exc_info=True)
            return False
    
    def get_unlinked_articles(self, batch_size: int) -> List[Dict[str, Any]]:
        """Fetch articles that don't have any entity links yet."""
        self.logger.info(f"Fetching up to {batch_size} unlinked articles...")
        
        # --- Tier 1: Try RPC function first (No change here) ---
        try:
            response = self.articles_db.supabase.rpc('get_unlinked_articles', {'batch_limit': batch_size}).execute()
            if hasattr(response, 'data') and response.data:
                articles = response.data
                self.logger.info(f"Found {len(articles)} unlinked articles via RPC")
                return articles
            # If there's an error, it will fall through to the next block
            if hasattr(response, 'error') and response.error:
                raise ConnectionError("RPC function not available or failed.")
        except Exception:
            self.logger.info("RPC function not available, using fallback query...")

        # --- Tier 2: Fall back to LEFT JOIN query ---
        try:
            self.logger.info("Attempting to fetch with corrected LEFT JOIN query...")
            # The syntax is changed to correctly perform the join before filtering
            response = self.articles_db.supabase.from_("SourceArticles").select(
                "id, Content, contentType, article_entity_links!left(article_id)"
            ).filter(
                "article_entity_links.article_id", "is", "null"
            ).filter(
                'Content', 'neq', ''
            ).in_(
                'contentType', ['news_article', 'news-round-up', 'topic_collection']
            ).limit(batch_size).execute()

            if hasattr(response, 'error') and response.error:
                raise ConnectionError(f"LEFT JOIN query failed: {response.error.message}")

            if hasattr(response, 'data'):
                articles = response.data
                # Remove the empty 'article_entity_links' field from the results
                for article in articles:
                    article.pop('article_entity_links', None)
                self.logger.info(f"SUCCESS: Found {len(articles)} unlinked articles via LEFT JOIN.")
                return articles

        except Exception as e:
            self.logger.warning(f"{e}. Falling back to inefficient manual filtering...")

        # --- Tier 3: Manual filtering as final fallback ---
        self.logger.info("Using manual filtering as a last resort...")
        # (Your existing manual filtering code remains here as the final backup)
        # ... your code to get linked_article_ids and loop through SourceArticles ...
        links_response = self.links_db.supabase.table("article_entity_links").select("article_id").execute()
        linked_article_ids = {link['article_id'] for link in links_response.data} if hasattr(links_response, 'data') and links_response.data else set()
        
        unlinked_articles = []
        offset = 0
        fetch_size = max(batch_size * 5, 100)
        
        for _ in range(10): # Max 10 attempts
            if len(unlinked_articles) >= batch_size: break
            articles_response = self.articles_db.supabase.table("SourceArticles").select("id, Content, contentType").filter('Content', 'neq', '').in_('contentType', ['news_article', 'news-round-up', 'topic_collection']).range(offset, offset + fetch_size - 1).execute()
            if not (hasattr(articles_response, 'data') and articles_response.data): break
            
            batch_articles = articles_response.data
            unlinked_articles.extend([a for a in batch_articles if a['id'] not in linked_article_ids])
            offset += fetch_size
            if len(batch_articles) < fetch_size: break
            
        self.logger.info(f"Found {len(unlinked_articles[:batch_size])} unlinked articles via manual filtering.")
        return unlinked_articles[:batch_size]
    
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
            
            raw_players = entities.get('players', [])
            raw_teams = entities.get('teams', [])
            
            # Extract names from structured LLM response
            players = []
            for player in raw_players:
                if isinstance(player, dict) and 'name' in player:
                    # LLM returned structured data with confidence
                    confidence = player.get('confidence', 0.0)
                    if confidence >= 0.5:  # Only include high-confidence entities
                        players.append(player['name'])
                elif isinstance(player, str):
                    # LLM returned simple string
                    players.append(player)
            
            teams = []
            for team in raw_teams:
                if isinstance(team, dict) and 'name' in team:
                    # LLM returned structured data with confidence
                    confidence = team.get('confidence', 0.0)
                    if confidence >= 0.5:  # Only include high-confidence entities
                        teams.append(team['name'])
                elif isinstance(team, str):
                    # LLM returned simple string
                    teams.append(team)
            
            self.logger.debug(f"LLM extracted {len(players)} players and {len(teams)} teams in {llm_time:.2f}s")
            
            return players, teams
            
        except Exception as e:
            self.logger.error(f"Error during LLM entity extraction: {e}")
            return [], []
    
    def validate_and_link_entities(self, players: List[str], teams: List[str]) -> List[LLMEntityMatch]:
        """
        Validate extracted entities against the entity dictionary using an efficient,
        case-insensitive lookup.
        """
        matches = []
        # 1. Consolidate players and teams into one structure
        all_entities = {'player': players, 'team': teams}
        self.logger.debug(f"Starting validation for {len(players)} players and {len(teams)} teams.")

        # 2. Use a single, unified loop
        for entity_type, entity_names in all_entities.items():
            for name in entity_names:
                name_clean = name.strip()
                if not name_clean:
                    continue

                self.stats['entities_extracted'] += 1
                name_lower = name_clean.lower()

                # 3. Use the pre-built lowercase dictionary for a direct, fast lookup
                if name_lower in self.entity_dict_lower:
                    # The name exists in our database, case-insensitively
                    entity_id = self.entity_dict_lower[name_lower]

                    # 4. Perform the type check (player vs. team)
                    is_player = len(entity_id) > 3 or not entity_id.isupper()
                    is_team = len(entity_id) <= 3 and entity_id.isupper()

                    # 5. Add the match if the type is correct
                    if (entity_type == 'player' and is_player) or \
                    (entity_type == 'team' and is_team):
                        
                        # Retrieve the original capitalized name for consistency
                        original_name = next((k for k in self.entity_dict if k.lower() == name_lower), name_clean)

                        matches.append(LLMEntityMatch(
                            entity_name=original_name,
                            entity_id=entity_id,
                            entity_type=entity_type
                        ))
                        self.stats['entities_validated'] += 1
                        self.logger.debug(f"Validated '{name_clean}' as {entity_type} -> {entity_id}")
                    else:
                        self.logger.debug(f"'{name_clean}' found but is wrong type (expected {entity_type})")
                else:
                    self.logger.debug(f"Entity '{name_clean}' not found in dictionary.")

        self.logger.debug(f"Validation complete: {len(matches)} valid matches.")
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
        """
        Process a batch of articles, creating entity links for valid matches
        and creating a sentinel record for articles with no matches.
        """
        processed_count = 0
        
        for article in articles:
            article_id = article.get('id')
            content = article.get('Content', '')
            
            if not article_id or not content:
                self.logger.warning(f"Skipping article with missing ID or content: {article_id}")
                continue
            
            try:
                # 1. Extract entities using the LLM
                players, teams = self.extract_entities_with_llm(content)
                
                # 2. Validate the extracted entities against the dictionary
                matches = self.validate_and_link_entities(players, teams)
                
                # 3. ---> This is the new logic block <---
                # If no valid matches are found, create a sentinel record to mark as processed.
                if not matches:
                    self.logger.info(f"Article {article_id}: No valid entities found. Marking as processed with a sentinel record.")
                    
                    # Create a special match object for our sentinel record.
                    empty_match = LLMEntityMatch(
                        entity_name="processed_empty",
                        entity_id="PROCESSED_EMPTY", # The unique ID for our marker
                        entity_type="status"         # A special type to distinguish it
                    )
                    
                    # Call the existing link creation function with this special match.
                    if self.create_entity_links(article_id, [empty_match]):
                        self.stats['articles_processed'] += 1
                        processed_count += 1
                    
                    # Continue to the next article in the batch.
                    continue
                # --- End of new logic ---

                # 4. If there were real matches, create the links as before.
                if self.create_entity_links(article_id, matches):
                    self.logger.info(f"Article {article_id}: Created {len(matches)} entity links.")
                    self.stats['articles_processed'] += 1
                    processed_count += 1
                else:
                    self.logger.error(f"Failed to create entity links for article {article_id}")

            except Exception as e:
                self.logger.error(f"An error occurred while processing article {article_id}: {e}", exc_info=True)
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
