#!/usr/bin/env python3
"""Example demonstrating the embedding system usage."""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nfl_news_pipeline.models import ContextSummary
from nfl_news_pipeline.embedding import EmbeddingGenerator, EmbeddingStorageManager, EmbeddingErrorHandler


async def main():
    """Demonstrate the embedding system functionality."""
    print("=== Embedding System Demo ===\n")
    
    # 1. Create a sample context summary
    print("1. Creating sample context summary...")
    context_summary = ContextSummary(
        news_url_id="demo-url-123",
        summary_text="Kansas City Chiefs sign veteran quarterback as backup to Patrick Mahomes",
        llm_model="gpt-4o-nano",
        confidence_score=0.9,
        entities={"teams": ["KC", "Chiefs"], "positions": ["quarterback"], "players": ["Patrick Mahomes"]},
        key_topics=["signing", "quarterback", "backup"],
        fallback_used=False,
        generated_at=datetime.now(timezone.utc)
    )
    print(f"   Summary: {context_summary.summary_text}")
    print(f"   Topics: {context_summary.key_topics}")
    print(f"   Entities: {context_summary.entities}\n")
    
    # 2. Initialize embedding generator (without OpenAI for demo)
    print("2. Initializing embedding generator...")
    generator = EmbeddingGenerator(
        use_openai_primary=False,  # Use sentence transformers for demo
        batch_size=10
    )
    print(f"   Model info: {generator.get_model_info()}\n")
    
    # 3. Initialize error handler
    print("3. Initializing error handler...")
    error_handler = EmbeddingErrorHandler(
        max_retries=2,
        base_delay=0.1,  # Fast for demo
        enable_circuit_breaker=True
    )
    print(f"   Error handler stats: {error_handler.get_error_stats()}\n")
    
    # 4. Demonstrate fallback embedding creation
    print("4. Creating fallback embedding (since we can't download models)...")
    try:
        # This will likely fail due to network restrictions, so we'll use fallback
        embedding = await generator.generate_embedding(context_summary, "demo-url-123")
        print(f"   Embedding generated successfully!")
        print(f"   Model: {embedding.model_name}")
        print(f"   Vector length: {len(embedding.embedding_vector)}")
        print(f"   Confidence: {embedding.confidence_score}")
    except Exception as e:
        print(f"   Embedding generation failed (expected): {e}")
        print("   Creating fallback embedding...")
        
        # Use error handler to create fallback
        fallback_embedding = await error_handler.create_fallback_embedding(
            context_summary, "demo-url-123", "metadata_based"
        )
        if fallback_embedding:
            print(f"   Fallback embedding created!")
            print(f"   Model: {fallback_embedding.model_name}")
            print(f"   Vector length: {len(fallback_embedding.embedding_vector)}")
            print(f"   Confidence: {fallback_embedding.confidence_score}")
            embedding = fallback_embedding
        else:
            print("   Fallback creation also failed")
            return
    
    print("\n5. Validating embedding...")
    try:
        embedding.validate()
        print("   Embedding validation passed!")
    except Exception as e:
        print(f"   Embedding validation failed: {e}")
        return
    
    # 6. Demonstrate storage operations (mock)
    print("\n6. Storage operations demo (using mock)...")
    print("   Note: This would normally connect to Supabase database")
    print(f"   Embedding ready for storage:")
    print(f"   - News URL ID: {embedding.news_url_id}")
    print(f"   - Vector dimension: {len(embedding.embedding_vector)}")
    print(f"   - Model: {embedding.model_name} v{embedding.model_version}")
    print(f"   - Generated at: {embedding.generated_at}")
    
    # 7. Demonstrate batch processing
    print("\n7. Batch processing demo...")
    summaries = [context_summary] * 3  # Simulate 3 similar summaries
    url_ids = ["demo-url-1", "demo-url-2", "demo-url-3"]
    
    print(f"   Processing {len(summaries)} summaries in batch...")
    try:
        batch_embeddings = await generator.generate_embeddings_batch(summaries, url_ids)
        print(f"   Batch processing completed: {len(batch_embeddings)} embeddings generated")
    except Exception as e:
        print(f"   Batch processing failed (expected): {e}")
        print("   This would fall back to individual processing in production")
    
    print("\n=== Demo Complete ===")
    print("The embedding system is ready for production use with:")
    print("- OpenAI text-embedding-3-small (primary)")
    print("- Sentence transformers (fallback)")
    print("- Comprehensive error handling and retry logic")
    print("- Efficient batch processing")
    print("- Supabase storage integration")
    print("- Circuit breaker pattern for reliability")


if __name__ == "__main__":
    asyncio.run(main())