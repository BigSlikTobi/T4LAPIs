#!/usr/bin/env python3
"""
Demo script showing how SimilarityCalculator and GroupCentroidManager work together
for story similarity grouping.

This demonstrates the implemented tasks 5.1 and 5.2 functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from datetime import datetime, timezone

from nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from nfl_news_pipeline.centroid_manager import GroupCentroidManager
from nfl_news_pipeline.models import (
    StoryEmbedding,
    StoryGroup,
    GroupCentroid,
    GroupStatus,
    EMBEDDING_DIM
)


def demo_similarity_calculator():
    """Demonstrate SimilarityCalculator capabilities."""
    print("🔍 SimilarityCalculator Demo")
    print("=" * 50)
    
    # Initialize with cosine similarity and 0.7 threshold
    calc = SimilarityCalculator(
        similarity_threshold=0.7,
        metric=SimilarityMetric.COSINE
    )
    
    print(f"✓ Initialized with threshold: {calc.threshold}")
    print(f"✓ Using metric: {calc.metric.value}")
    
    # Create some example embeddings
    embedding1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    embedding2 = embedding1 + np.random.normal(0, 0.1, EMBEDDING_DIM).astype(np.float32)  # Similar
    embedding3 = np.random.rand(EMBEDDING_DIM).astype(np.float32)  # Different
    
    # Calculate similarities
    sim_similar = calc.calculate_similarity(embedding1, embedding2)
    sim_different = calc.calculate_similarity(embedding1, embedding3)
    
    print(f"\n📊 Similarity Results:")
    print(f"   Similar stories: {sim_similar:.3f}")
    print(f"   Different stories: {sim_different:.3f}")
    
    # Test threshold checking
    print(f"\n🎯 Threshold Checking:")
    print(f"   Similar above threshold? {calc.is_similar(sim_similar)}")
    print(f"   Different above threshold? {calc.is_similar(sim_different)}")
    
    print("\n✅ SimilarityCalculator demo complete!\n")


def demo_centroid_manager():
    """Demonstrate GroupCentroidManager capabilities."""
    print("🎯 GroupCentroidManager Demo")
    print("=" * 50)
    
    manager = GroupCentroidManager()
    
    # Create some story embeddings
    embeddings = []
    story_topics = ["QB trade", "Injury report", "Draft pick"]
    
    for i, topic in enumerate(story_topics):
        # Create embeddings with some similarity (same general direction)
        base_vector = [0.5 + np.random.normal(0, 0.1)] + [np.random.normal(0, 0.05) for _ in range(EMBEDDING_DIM - 1)]
        
        embedding = StoryEmbedding(
            news_url_id=f"story-{i+1}",
            embedding_vector=base_vector,
            model_name="text-embedding-3-small",
            model_version="1.0",
            summary_text=f"NFL {topic} news story",
            confidence_score=0.9
        )
        embeddings.append(embedding)
    
    print(f"✓ Created {len(embeddings)} story embeddings")
    
    # Calculate centroid
    centroid = manager.calculate_centroid(embeddings)
    print(f"✓ Calculated centroid (dimension: {len(centroid)})")
    print(f"   Centroid norm: {np.linalg.norm(centroid):.3f} (should be ~1.0)")
    
    # Create a group and update its centroid
    group = StoryGroup(
        id="group-1",
        member_count=0,
        status=GroupStatus.NEW,
        tags=["nfl", "news"]
    )
    
    update_result = manager.update_group_centroid(group, embeddings)
    print(f"\n📈 Group Update Results:")
    print(f"   Update successful: {update_result.update_successful}")
    print(f"   Member count: {update_result.member_count}")
    print(f"   Group status: {group.status.value}")
    
    # Test incremental centroid updates
    print(f"\n⚡ Incremental Update Demo:")
    original_centroid = np.array(group.centroid_embedding)
    
    # Add a new story
    new_embedding = StoryEmbedding(
        news_url_id="story-new",
        embedding_vector=[0.8] + [0.0] * (EMBEDDING_DIM - 1),
        model_name="text-embedding-3-small",
        model_version="1.0",
        summary_text="New NFL story",
        confidence_score=0.95
    )
    
    updated_centroid = manager.add_embedding_to_centroid(
        original_centroid, len(embeddings), new_embedding
    )
    
    # Check how much the centroid changed
    centroid_change = np.linalg.norm(updated_centroid - original_centroid)
    print(f"   Centroid changed by: {centroid_change:.4f}")
    print(f"   New centroid norm: {np.linalg.norm(updated_centroid):.3f}")
    
    print("\n✅ GroupCentroidManager demo complete!\n")


def demo_integration():
    """Demonstrate how SimilarityCalculator and GroupCentroidManager work together."""
    print("🔗 Integration Demo")
    print("=" * 50)
    
    # Initialize components
    similarity_calc = SimilarityCalculator(similarity_threshold=0.6)
    centroid_manager = GroupCentroidManager()
    
    # Simulate existing groups
    groups_data = [
        {
            "id": "qb-trades",
            "topic": "quarterback trades",
            "base_vector": [1.0, 0.0, 0.0]
        },
        {
            "id": "injuries", 
            "topic": "injury reports",
            "base_vector": [0.0, 1.0, 0.0]
        },
        {
            "id": "draft",
            "topic": "draft news",
            "base_vector": [0.0, 0.0, 1.0]
        }
    ]
    
    # Create group centroids
    centroids = []
    for group_data in groups_data:
        vector = group_data["base_vector"] + [0.0] * (EMBEDDING_DIM - 3)
        centroid = GroupCentroid(
            group_id=group_data["id"],
            centroid_vector=vector,
            member_count=5,
            last_updated=datetime.now(timezone.utc)
        )
        centroids.append(centroid)
    
    print(f"✓ Created {len(centroids)} existing group centroids")
    
    # Test story assignment
    test_stories = [
        {
            "id": "new-qb-story",
            "vector": [0.9, 0.1, 0.0] + [0.0] * (EMBEDDING_DIM - 3),
            "topic": "quarterback trade"
        },
        {
            "id": "new-injury-story", 
            "vector": [0.0, 0.8, 0.2] + [0.0] * (EMBEDDING_DIM - 3),
            "topic": "injury report"
        },
        {
            "id": "random-story",
            "vector": [0.2, 0.3, 0.1] + [0.1] * (EMBEDDING_DIM - 3),
            "topic": "general sports news"
        }
    ]
    
    print(f"\n🔍 Story Assignment Results:")
    
    for story in test_stories:
        story_embedding = StoryEmbedding(
            news_url_id=story["id"],
            embedding_vector=story["vector"],
            model_name="test-model",
            model_version="1.0", 
            summary_text=f"NFL {story['topic']} story",
            confidence_score=0.9
        )
        
        # Find similar groups
        matches = similarity_calc.find_similar_groups(story_embedding, centroids)
        
        if matches:
            best_group, similarity_score = matches[0]
            print(f"   📄 {story['topic']}")
            print(f"      → Matched to: {best_group}")
            print(f"      → Similarity: {similarity_score:.3f}")
        else:
            print(f"   📄 {story['topic']}")
            print(f"      → No matches found (below threshold)")
    
    print("\n✅ Integration demo complete!")


if __name__ == "__main__":
    print("🚀 Story Similarity Grouping Demo")
    print("Tasks 5.1 and 5.2 Implementation")
    print("=" * 60)
    print()
    
    try:
        demo_similarity_calculator()
        demo_centroid_manager()
        demo_integration()
        
        print("\n🎉 All demos completed successfully!")
        print("\nThe SimilarityCalculator and GroupCentroidManager are ready")
        print("to be integrated into the larger story grouping pipeline.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)