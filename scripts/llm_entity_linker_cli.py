#!/usr/bin/env python3
"""
Command-line interface for LLM-enhanced entity linking.

This script provides options for testing and running the LLM-enhanced entity linking functionality
using DeepSeek LLM for more accurate entity extraction.

Examples:
    # Test LLM entity extraction on sample text
    python llm_entity_linker_cli.py test --text "Patrick Mahomes threw for 300 yards as the Chiefs beat the 49ers."
    
    # Run LLM entity linking on next 10 unlinked articles
    python llm_entity_linker_cli.py run --batch-size 10 --max-batches 1
    
    # Show LLM entity linking statistics
    python llm_entity_linker_cli.py stats
"""

import argparse
import sys
import os
import time
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.llm_entity_linker import LLMEntityLinker
from src.core.llm.llm_init import get_deepseek_client
from src.core.data.entity_linking import build_entity_dictionary
from src.core.utils.logging import setup_logging


def test_llm_extraction(text: str) -> None:
    """Test LLM entity extraction on provided text."""
    print(f"\n🤖 Testing LLM entity extraction on text:")
    print(f"Text: '{text}'")
    print("-" * 80)
    
    try:
        # Initialize LLM client
        print("⚙️  Initializing DeepSeek LLM client...")
        llm_client = get_deepseek_client()
        
        if not llm_client.test_connection():
            print("❌ Failed to connect to DeepSeek LLM")
            return
        
        print("✅ LLM client connected successfully")
        
        # Extract entities using LLM
        print("\n🔍 Extracting entities with LLM...")
        start_time = time.time()
        entities = llm_client.extract_entities(text)
        extraction_time = time.time() - start_time
        
        players = entities.get('players', [])
        teams = entities.get('teams', [])
        
        print(f"⏱️  LLM extraction completed in {extraction_time:.2f} seconds")
        print(f"🎯 Found {len(players)} players and {len(teams)} teams")
        
        if players:
            print(f"\n👥 Players:")
            for i, player in enumerate(players, 1):
                if isinstance(player, dict) and 'name' in player:
                    print(f"  {i}. {player['name']} (confidence: {player.get('confidence', 'N/A')})")
                else:
                    print(f"  {i}. {player}")
        
        if teams:
            print(f"\n🏈 Teams:")
            for i, team in enumerate(teams, 1):
                if isinstance(team, dict) and 'name' in team:
                    print(f"  {i}. {team['name']} (confidence: {team.get('confidence', 'N/A')})")
                else:
                    print(f"  {i}. {team}")
        
        if not players and not teams:
            print("❌ No entities extracted by LLM")
        
        if not players and not teams:
            print("❌ No entities extracted by LLM")
        
        # Validate against entity dictionary
        print("\n✅ Validating against entity dictionary...")
        entity_dict = build_entity_dictionary()
        
        validated_players = 0
        validated_teams = 0
        
        # Extract names from structured entities
        player_names = []
        for player in players:
            if isinstance(player, dict) and 'name' in player:
                player_names.append(player['name'])
            elif isinstance(player, str):
                player_names.append(player)
        
        team_names = []
        for team in teams:
            if isinstance(team, dict) and 'name' in team:
                team_names.append(team['name'])
            elif isinstance(team, str):
                team_names.append(team)
        
        for player_name in player_names:
            if player_name in entity_dict or player_name.lower() in [k.lower() for k in entity_dict.keys()]:
                validated_players += 1
        
        for team_name in team_names:
            if team_name in entity_dict or team_name.lower() in [k.lower() for k in entity_dict.keys()]:
                validated_teams += 1
        
        total_extracted = len(player_names) + len(team_names)
        total_validated = validated_players + validated_teams
        validation_rate = (total_validated / total_extracted * 100) if total_extracted > 0 else 0
        
        print(f"📊 Validation results:")
        print(f"   Players validated: {validated_players}/{len(player_names)}")
        print(f"   Teams validated: {validated_teams}/{len(team_names)}")
        print(f"   Overall validation rate: {validation_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error during LLM entity extraction: {e}")


def run_llm_entity_linking(batch_size: int, max_batches: int = None) -> None:
    """Run LLM-enhanced entity linking on unlinked articles."""
    print(f"\n🤖 Running LLM-enhanced entity linking with batch size {batch_size}")
    if max_batches:
        print(f"📊 Maximum batches: {max_batches}")
    print("-" * 80)
    
    # Initialize LLM entity linker
    linker = LLMEntityLinker(batch_size=batch_size)
    
    # Run LLM entity linking
    start_time = time.time()
    result = linker.run_llm_entity_linking(max_batches=max_batches)
    end_time = time.time()
    
    if result['success']:
        print(f"\\n✅ LLM entity linking completed successfully!")
        print(f"📊 Total articles processed: {result['total_processed']}")
        print(f"⏱️  Total processing time: {end_time - start_time:.2f} seconds")
        
        # Show detailed stats
        stats = result['stats']
        print(f"\\n📈 Detailed Statistics:")
        print(f"   Articles processed: {stats['articles_processed']}")
        print(f"   LLM calls: {stats['llm_calls']}")
        print(f"   Entities extracted: {stats['entities_extracted']}")
        print(f"   Entities validated: {stats['entities_validated']}")
        print(f"   Links created: {stats['links_created']}")
        print(f"   Processing time: {stats['processing_time']:.2f} seconds")
        print(f"   LLM processing time: {stats['llm_time']:.2f} seconds")
        
        if stats['articles_processed'] > 0:
            avg_llm_calls = stats['llm_calls'] / stats['articles_processed']
            avg_entities = stats['entities_extracted'] / stats['articles_processed']
            avg_links = stats['links_created'] / stats['articles_processed']
            print(f"   Average LLM calls per article: {avg_llm_calls:.2f}")
            print(f"   Average entities per article: {avg_entities:.2f}")
            print(f"   Average links per article: {avg_links:.2f}")
        
        if stats['entities_extracted'] > 0:
            validation_rate = stats['entities_validated'] / stats['entities_extracted'] * 100
            print(f"   Entity validation rate: {validation_rate:.1f}%")
    else:
        print(f"\\n❌ LLM entity linking failed: {result['error']}")


def debug_unlinked_articles(batch_size: int = 10) -> None:
    """Debug function to check article counts and unlinked article detection."""
    print(f"\n🔍 Debugging unlinked articles detection with batch size {batch_size}")
    print("-" * 80)
    
    try:
        # Initialize database managers
        from src.core.utils.database import DatabaseManager
        articles_db = DatabaseManager("SourceArticles")
        links_db = DatabaseManager("article_entity_links")
        
        # Check total articles
        print("📊 Database Statistics:")
        
        # Total articles
        result = articles_db.supabase.table('SourceArticles').select('id', count='exact').execute()
        total_articles = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   Total SourceArticles: {total_articles}")
        
        # Articles with valid content types
        result = articles_db.supabase.table('SourceArticles').select('id', count='exact').in_(
            'contentType', ['news_article', 'news-round-up', 'topic_collection']
        ).execute()
        valid_content_type_articles = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   Articles with valid contentType: {valid_content_type_articles}")
        
        # Articles with content AND valid content type
        result = articles_db.supabase.table('SourceArticles').select('id', count='exact').neq('Content', '').in_(
            'contentType', ['news_article', 'news-round-up', 'topic_collection']
        ).execute()
        articles_with_content = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   Articles with Content + valid contentType: {articles_with_content}")
        
        # Show breakdown by content type
        for content_type in ['news_article', 'news-round-up', 'topic_collection']:
            result = articles_db.supabase.table('SourceArticles').select('id', count='exact').eq('contentType', content_type).execute()
            type_count = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
            print(f"   Articles with contentType='{content_type}': {type_count}")
        
        # Articles with null content
        result = articles_db.supabase.table('SourceArticles').select('id', count='exact').is_('Content', 'null').execute()
        articles_null_content = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   Articles with NULL Content: {articles_null_content}")
        
        # Articles with empty string content
        result = articles_db.supabase.table('SourceArticles').select('id', count='exact').eq('Content', '').execute()
        articles_empty_content = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   Articles with empty Content: {articles_empty_content}")
        
        # Total entity links
        result = links_db.supabase.table('article_entity_links').select('article_id', count='exact').execute()
        total_links = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   Total entity links: {total_links}")
        
        # Unique linked articles
        result = links_db.supabase.table('article_entity_links').select('article_id').execute()
        unique_linked_articles = len(set([link['article_id'] for link in result.data])) if result.data else 0
        print(f"   Unique linked articles: {unique_linked_articles}")
        
        # Calculate unlinked articles estimate
        estimated_unlinked = articles_with_content - unique_linked_articles
        print(f"   Estimated unlinked articles: {estimated_unlinked}")
        
        print(f"\n🔬 Testing unlinked article detection methods:")
        
        # Test the LLMEntityLinker's get_unlinked_articles method
        linker = LLMEntityLinker(batch_size=batch_size)
        unlinked_articles = linker.get_unlinked_articles(batch_size)
        
        print(f"   LLMEntityLinker.get_unlinked_articles() returned: {len(unlinked_articles)} articles")
        
        if unlinked_articles:
            print(f"   Sample article IDs: {[article['id'] for article in unlinked_articles[:5]]}")
            
            # Check if these articles actually have content
            print(f"\n📝 Content check for first few articles:")
            for i, article in enumerate(unlinked_articles[:3]):
                content_length = len(article.get('Content', '')) if article.get('Content') else 0
                print(f"   Article {article['id']}: Content length = {content_length}")
                if content_length > 0:
                    preview = article['Content'][:100] + "..." if len(article['Content']) > 100 else article['Content']
                    print(f"     Preview: {preview}")
                    
                    # Check if this article actually has links
                    links_result = links_db.supabase.table('article_entity_links').select('*').eq('article_id', article['id']).execute()
                    link_count = len(links_result.data) if links_result.data else 0
                    print(f"     Existing links: {link_count}")
        else:
            print("   No unlinked articles found!")
            print("\n🔧 Let's try manual detection:")
            
            # Manual check: get some articles with content
            result = articles_db.supabase.table('SourceArticles').select('id, Content, contentType').neq('Content', '').in_(
                'contentType', ['news_article', 'news-round-up', 'topic_collection']
            ).limit(10).execute()
            if result.data:
                print(f"   Found {len(result.data)} articles with content")
                for article in result.data[:3]:
                    # Check if this article has links
                    links_result = links_db.supabase.table('article_entity_links').select('*').eq('article_id', article['id']).execute()
                    link_count = len(links_result.data) if links_result.data else 0
                    content_length = len(article.get('Content', '')) if article.get('Content') else 0
                    content_type = article.get('contentType', 'unknown')
                    print(f"     Article {article['id']}: ContentType={content_type}, Content={content_length} chars, Links={link_count}")
                    
                    if link_count == 0 and content_length > 0:
                        print(f"       ✅ This article should be processable!")
            
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()


def show_llm_stats() -> None:
    """Show statistics about LLM entity linking capabilities."""
    print("\\n📊 LLM Entity Linking Statistics")
    print("-" * 80)
    
    try:
        # Test LLM connection
        print("⚙️  Testing LLM connection...")
        llm_client = get_deepseek_client()
        
        if llm_client.test_connection():
            print("✅ DeepSeek LLM connection successful")
        else:
            print("❌ DeepSeek LLM connection failed")
            return
        
        # Get entity dictionary stats
        print("\\n📚 Loading entity dictionary...")
        entity_dict = build_entity_dictionary()
        
        if not entity_dict:
            print("❌ No entities found in dictionary")
            return
        
        print(f"✅ Entity dictionary loaded with {len(entity_dict)} patterns")
        
        # Count by type
        player_count = 0
        team_count = 0
        
        for entity_name, entity_id in entity_dict.items():
            if len(entity_id) <= 3 and entity_id.isupper():
                team_count += 1
            else:
                player_count += 1
        
        print(f"\\n📈 Entity Dictionary Breakdown:")
        print(f"   Player patterns: {player_count}")
        print(f"   Team patterns: {team_count}")
        
        # Test LLM extraction on sample texts
        print("\\n🧪 Testing LLM extraction on sample texts...")
        test_texts = [
            "Patrick Mahomes threw for 300 yards as the Kansas City Chiefs defeated the San Francisco 49ers.",
            "Lamar Jackson and the Baltimore Ravens lost to the Buffalo Bills in overtime.",
            "Cooper Kupp caught 8 passes for 120 yards in the Los Angeles Rams victory over Seattle."
        ]
        
        total_extracted = 0
        total_validated = 0
        total_time = 0
        
        for i, text in enumerate(test_texts, 1):
            print(f"\\n   Test {i}: Processing sample text...")
            start_time = time.time()
            entities = llm_client.extract_entities(text)
            extraction_time = time.time() - start_time
            total_time += extraction_time
            
            players = entities.get('players', [])
            teams = entities.get('teams', [])
            extracted_count = len(players) + len(teams)
            total_extracted += extracted_count
            
            # Validate entities
            validated_count = 0
            for player in players:
                if player in entity_dict or player.lower() in [k.lower() for k in entity_dict.keys()]:
                    validated_count += 1
            for team in teams:
                if team in entity_dict or team.lower() in [k.lower() for k in entity_dict.keys()]:
                    validated_count += 1
            
            total_validated += validated_count
            
            print(f"      Extracted: {len(players)} players, {len(teams)} teams")
            print(f"      Validated: {validated_count}/{extracted_count}")
            print(f"      Time: {extraction_time:.2f}s")
        
        # Summary statistics
        avg_time = total_time / len(test_texts)
        validation_rate = (total_validated / total_extracted * 100) if total_extracted > 0 else 0
        
        print(f"\\n📊 LLM Performance Summary:")
        print(f"   Total entities extracted: {total_extracted}")
        print(f"   Total entities validated: {total_validated}")
        print(f"   Average validation rate: {validation_rate:.1f}%")
        print(f"   Average extraction time: {avg_time:.2f}s per text")
        
    except Exception as e:
        print(f"❌ Error getting LLM statistics: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="LLM-Enhanced Entity Linking CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test LLM entity extraction on sample text')
    test_parser.add_argument('--text', required=True, 
                           help='Text to test LLM entity extraction on')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run LLM entity linking on unlinked articles')
    run_parser.add_argument('--batch-size', type=int, default=5,
                          help='Number of articles to process per batch (default: 5)')
    run_parser.add_argument('--max-batches', type=int,
                          help='Maximum number of batches to process (default: unlimited)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show LLM entity linking statistics')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug unlinked articles detection')
    debug_parser.add_argument('--batch-size', type=int, default=10,
                             help='Batch size to test (default: 10)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging()
    
    # Execute command
    try:
        if args.command == 'test':
            test_llm_extraction(args.text)
        
        elif args.command == 'run':
            run_llm_entity_linking(args.batch_size, args.max_batches)
        
        elif args.command == 'stats':
            show_llm_stats()
        
        elif args.command == 'debug':
            debug_unlinked_articles(args.batch_size)
        
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
