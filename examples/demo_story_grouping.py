#!/usr/bin/env python3
"""Demo script for story grouping URL context extraction.

Demonstrates tasks 3.1, 3.2, and 3.3 functionality.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone

# Add the src directory to Python path for local execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nfl_news_pipeline.models import ProcessedNewsItem
from nfl_news_pipeline.story_grouping import URLContextExtractor, ContextCache


def setup_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def demo_url_context_extraction():
    """Demonstrate URL context extraction functionality."""
    print("🧪 Story Grouping URL Context Extraction Demo")
    print("=" * 50)
    
    # Create sample news items
    sample_items = [
        ProcessedNewsItem(
            url="https://espn.com/chiefs-mahomes-injury-update",
            title="Patrick Mahomes suffers ankle injury in Chiefs victory over Bills",
            publication_date=datetime.now(timezone.utc),
            source_name="espn",
            publisher="ESPN",
            description="Kansas City Chiefs quarterback Patrick Mahomes injured his ankle during the fourth quarter but continued playing to lead his team to victory.",
            relevance_score=0.95,
            filter_method="rule_based",
            entities=["Patrick Mahomes", "Kansas City Chiefs", "Buffalo Bills"],
            categories=["injury", "quarterback"]
        ),
        ProcessedNewsItem(
            url="https://nfl.com/trade-deadline-moves",
            title="NFL Trade Deadline: Chiefs acquire wide receiver",
            publication_date=datetime.now(timezone.utc),
            source_name="nfl",
            publisher="NFL.com",
            description="The Kansas City Chiefs made a move at the trade deadline to strengthen their receiving corps.",
            relevance_score=0.88,
            filter_method="llm",
            entities=["Kansas City Chiefs"],
            categories=["trade", "wide receiver"]
        ),
        ProcessedNewsItem(
            url="https://example.com/invalid-article",
            title="Test Article with No Real Content",
            publication_date=datetime.now(timezone.utc),
            source_name="test",
            publisher="Test Publisher"
        )
    ]
    
    # Initialize cache and extractor
    print("\n📦 Initializing cache and extractor...")
    cache = ContextCache(ttl_hours=1, enable_memory_cache=True, enable_disk_cache=True)
    extractor = URLContextExtractor(
        preferred_provider="openai",  # Will fallback since no API keys in demo
        cache=cache,
        enable_caching=True
    )
    
    print("✅ Cache and extractor initialized")
    print(f"   Cache TTL: {cache.ttl_hours} hours")
    print(f"   Memory cache enabled: {cache.memory_cache is not None}")
    
    # Process each news item
    print("\n🔍 Extracting context from news items...")
    for i, item in enumerate(sample_items, 1):
        print(f"\n--- Processing Item {i} ---")
        print(f"URL: {item.url}")
        print(f"Title: {item.title}")
        
        # Extract context
        summary = await extractor.extract_context(item)
        
        print(f"✅ Context extracted successfully")
        print(f"   Model used: {summary.llm_model}")
        print(f"   Fallback used: {summary.fallback_used}")
        print(f"   Confidence: {summary.confidence_score:.2f}")
        print(f"   Summary: {summary.summary_text[:100]}...")
        
        if summary.entities:
            if summary.entities.get("players"):
                print(f"   Players: {', '.join(summary.entities['players'][:3])}")
            if summary.entities.get("teams"):
                print(f"   Teams: {', '.join(summary.entities['teams'][:3])}")
        
        if summary.key_topics:
            print(f"   Topics: {', '.join(summary.key_topics[:3])}")
    
    # Demonstrate caching by processing the first item again
    print(f"\n🔄 Testing cache functionality...")
    print("Re-processing first item to test cache hit...")
    
    cached_summary = await extractor.extract_context(sample_items[0])
    
    # Display cache statistics
    stats = cache.get_cache_stats()
    print(f"✅ Cache statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
    
    # Demonstrate team name normalization
    print(f"\n🔧 Testing team name normalization...")
    test_teams = ["chiefs", "49ers", "KC", "San Francisco", "Unknown Team"]
    normalized = extractor._normalize_team_names(test_teams)
    
    print(f"Original teams: {test_teams}")
    print(f"Normalized teams: {normalized}")
    
    # Demonstrate metadata-based fallback
    print(f"\n🔄 Testing metadata fallback...")
    fallback_item = ProcessedNewsItem(
        url="https://example.com/fallback-test",
        title="Player injured during practice session",
        publication_date=datetime.now(timezone.utc),
        source_name="test",
        publisher="Test",
        description="A key player was injured during Tuesday's practice."
    )
    
    fallback_summary = extractor._fallback_to_metadata(fallback_item)
    print(f"✅ Fallback summary generated:")
    print(f"   Summary: {fallback_summary.summary_text}")
    print(f"   Detected topics: {fallback_summary.key_topics}")
    print(f"   Fallback used: {fallback_summary.fallback_used}")


def demo_cache_utilities():
    """Demonstrate cache utility functions."""
    print(f"\n🛠️  Cache Utilities Demo")
    print("=" * 30)
    
    from nfl_news_pipeline.story_grouping import generate_metadata_hash
    
    # Test metadata hash generation
    hash1 = generate_metadata_hash("Patrick Mahomes injury update")
    hash2 = generate_metadata_hash("Patrick Mahomes injury update", "Detailed description")
    hash3 = generate_metadata_hash("Different title")
    
    print(f"Hash 1 (title only): {hash1}")
    print(f"Hash 2 (title + desc): {hash2}")
    print(f"Hash 3 (different): {hash3}")
    
    print(f"\n✅ Hash consistency:")
    print(f"   Same title generates same hash: {hash1 == generate_metadata_hash('Patrick Mahomes injury update')}")
    print(f"   Different titles generate different hashes: {hash1 != hash3}")


async def main():
    """Main demo function."""
    setup_logging()
    
    try:
        await demo_url_context_extraction()
        demo_cache_utilities()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n📋 Summary of implemented features:")
        print(f"   ✅ Task 3.1: LLM URL context extractor with OpenAI/Google support")
        print(f"   ✅ Task 3.2: Fallback mechanism for metadata-based summaries")
        print(f"   ✅ Task 3.3: Caching layer with TTL and cost optimization")
        print(f"   ✅ Entity normalization for teams and players")
        print(f"   ✅ Comprehensive error handling and confidence scoring")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())