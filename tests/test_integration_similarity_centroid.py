"""
Integration test for SimilarityCalculator and GroupCentroidManager.

This test demonstrates how the two components work together in a realistic
story grouping scenario.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
from src.nfl_news_pipeline.models import (
    StoryEmbedding,
    StoryGroup,
    GroupCentroid,
    GroupStatus,
    EMBEDDING_DIM
)


def test_similarity_and_centroid_integration():
    """Test how SimilarityCalculator and GroupCentroidManager work together."""
    
    # Initialize components
    similarity_calc = SimilarityCalculator(
        similarity_threshold=0.7,
        metric=SimilarityMetric.COSINE
    )
    centroid_manager = GroupCentroidManager()
    
    # Create initial story embeddings for the first group
    embeddings_group1 = []
    for i in range(3):
        # Create similar embeddings (all close to [1, 0, 0, ...])
        vector = [0.9 + np.random.normal(0, 0.05)] + [np.random.normal(0, 0.05) for _ in range(EMBEDDING_DIM - 1)]
        embedding = StoryEmbedding(
            news_url_id=f"story-1-{i}",
            embedding_vector=vector,
            model_name="test-model",
            model_version="1.0",
            summary_text=f"NFL quarterback trade story {i}",
            confidence_score=0.9
        )
        embeddings_group1.append(embedding)
    
    # Create first group and its centroid
    group1 = StoryGroup(
        id="group-1",
        member_count=0,
        status=GroupStatus.NEW,
        tags=["quarterback", "trade"]
    )
    
    centroid1 = centroid_manager.create_group_centroid(group1, embeddings_group1)
    
    # Verify group centroid was created properly
    assert centroid1.group_id == "group-1"
    assert centroid1.member_count == 3
    assert len(centroid1.centroid_vector) == EMBEDDING_DIM
    assert group1.centroid_embedding == centroid1.centroid_vector
    
    # Create a new story that should match group 1
    new_story_similar = StoryEmbedding(
        news_url_id="new-story-1",
        embedding_vector=[0.95] + [0.0] * (EMBEDDING_DIM - 1),
        model_name="test-model",
        model_version="1.0",
        summary_text="Another quarterback trade story",
        confidence_score=0.9
    )
    
    # Test similarity calculation
    matches = similarity_calc.find_similar_groups(new_story_similar, [centroid1])
    
    assert len(matches) >= 1
    assert matches[0][0] == "group-1"
    assert matches[0][1] > 0.7  # Should be above threshold
    print(f"✓ Similar story matched with similarity score: {matches[0][1]:.3f}")
    
    # Add the new story to the group and update centroid
    embeddings_group1.append(new_story_similar)
    update_result = centroid_manager.update_group_centroid(group1, embeddings_group1)
    
    assert update_result.update_successful is True
    assert update_result.member_count == 4
    print("✓ Successfully updated group centroid after adding new member")
    
    # Create a different story that should NOT match group 1
    new_story_different = StoryEmbedding(
        news_url_id="new-story-2",
        embedding_vector=[0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3),
        model_name="test-model",
        model_version="1.0",
        summary_text="NFL injury report story",
        confidence_score=0.9
    )
    
    # Test that different story doesn't match
    updated_centroid = GroupCentroid(
        group_id=group1.id,
        centroid_vector=group1.centroid_embedding,
        member_count=group1.member_count,
        last_updated=group1.updated_at
    )
    
    matches_different = similarity_calc.find_similar_groups(new_story_different, [updated_centroid])
    
    if matches_different:
        print(f"✓ Different story similarity score: {matches_different[0][1]:.3f} (below threshold)")
        assert matches_different[0][1] < 0.7  # Should be below threshold
    else:
        print("✓ Different story correctly found no matches")
    
    # Test incremental centroid updates
    original_centroid = np.array(group1.centroid_embedding)
    incremental_centroid = centroid_manager.add_embedding_to_centroid(
        original_centroid, 4, new_story_different
    )
    
    # Verify incremental update produces different centroid
    assert not np.allclose(original_centroid, incremental_centroid)
    print("✓ Incremental centroid update working correctly")
    
    # Test centroid similarity calculation
    similarity_between_centroids = centroid_manager.centroid_similarity(
        centroid1.centroid_vector,
        incremental_centroid.tolist()
    )
    
    assert 0.0 <= similarity_between_centroids <= 1.0
    print(f"✓ Centroid similarity calculation: {similarity_between_centroids:.3f}")
    
    print("\n✓ All integration tests passed successfully!")


def test_multiple_groups_similarity_search():
    """Test similarity search across multiple groups."""
    
    similarity_calc = SimilarityCalculator(similarity_threshold=0.6)
    centroid_manager = GroupCentroidManager()
    
    # Create multiple groups with different centroids
    groups_and_centroids = []
    
    # Group 1: QB trades (around [1, 0, 0, ...])
    group1_centroid = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    centroid1 = GroupCentroid(
        group_id="qb-trades",
        centroid_vector=group1_centroid,
        member_count=5
    )
    groups_and_centroids.append(centroid1)
    
    # Group 2: Injury reports (around [0, 1, 0, ...])
    group2_centroid = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    centroid2 = GroupCentroid(
        group_id="injuries",
        centroid_vector=group2_centroid,
        member_count=3
    )
    groups_and_centroids.append(centroid2)
    
    # Group 3: Draft news (around [0, 0, 1, ...])
    group3_centroid = [0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3)
    centroid3 = GroupCentroid(
        group_id="draft-news",
        centroid_vector=group3_centroid,
        member_count=2
    )
    groups_and_centroids.append(centroid3)
    
    # Test story that should match QB trades
    qb_story = StoryEmbedding(
        news_url_id="qb-story",
        embedding_vector=[0.9, 0.1] + [0.0] * (EMBEDDING_DIM - 2),
        model_name="test-model",
        model_version="1.0",
        summary_text="Quarterback trade news",
        confidence_score=0.9
    )
    
    matches = similarity_calc.find_similar_groups(qb_story, groups_and_centroids)
    
    # Should find the QB trades group as best match
    assert len(matches) >= 1
    assert matches[0][0] == "qb-trades"
    print(f"✓ QB story correctly matched to qb-trades group (similarity: {matches[0][1]:.3f})")
    
    # Test best match functionality
    best_match = similarity_calc.find_best_matching_group(qb_story, groups_and_centroids)
    assert best_match is not None
    assert best_match[0] == "qb-trades"
    print(f"✓ Best match correctly identified: {best_match[0]}")
    
    # Test story that shouldn't match any group (very different)
    random_story = StoryEmbedding(
        news_url_id="random-story",
        embedding_vector=[0.1] * EMBEDDING_DIM,  # Very different pattern
        model_name="test-model",
        model_version="1.0",
        summary_text="Random sports news",
        confidence_score=0.9
    )
    
    # Lower threshold for this test
    similarity_calc.update_threshold(0.9)
    matches_random = similarity_calc.find_similar_groups(random_story, groups_and_centroids)
    
    # Should find no matches with high threshold
    assert len(matches_random) == 0
    print("✓ Random story correctly found no matches with high threshold")
    
    print("\n✓ Multiple groups similarity search test passed!")


if __name__ == "__main__":
    test_similarity_and_centroid_integration()
    test_multiple_groups_similarity_search()