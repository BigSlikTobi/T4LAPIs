"""
Tests for similarity calculation functionality.

Tests cover various similarity metrics, edge cases, and integration with
the story grouping models.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.nfl_news_pipeline.similarity import (
    SimilarityCalculator,
    SimilarityMetric,
    SimilarityResult
)
from src.nfl_news_pipeline.models import (
    StoryEmbedding,
    GroupCentroid,
    EMBEDDING_DIM
)


class TestSimilarityCalculator:
    """Test suite for SimilarityCalculator class."""

    def test_initialization_default(self):
        """Test default initialization of SimilarityCalculator."""
        calc = SimilarityCalculator()
        assert calc.threshold == 0.8
        assert calc.metric == SimilarityMetric.COSINE

    def test_initialization_custom(self):
        """Test custom initialization of SimilarityCalculator."""
        calc = SimilarityCalculator(
            similarity_threshold=0.7,
            metric=SimilarityMetric.EUCLIDEAN
        )
        assert calc.threshold == 0.7
        assert calc.metric == SimilarityMetric.EUCLIDEAN

    def test_initialization_invalid_threshold(self):
        """Test initialization with invalid threshold raises error."""
        with pytest.raises(ValueError, match="similarity_threshold must be in range"):
            SimilarityCalculator(similarity_threshold=-0.1)
        
        with pytest.raises(ValueError, match="similarity_threshold must be in range"):
            SimilarityCalculator(similarity_threshold=1.1)

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        calc = SimilarityCalculator(metric=SimilarityMetric.COSINE)
        vec1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        vec2 = vec1.copy()
        
        similarity = calc.calculate_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        calc = SimilarityCalculator(metric=SimilarityMetric.COSINE)
        
        # Create orthogonal vectors
        vec1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec2[1] = 1.0
        
        similarity = calc.calculate_similarity(vec1, vec2)
        assert abs(similarity - 0.5) < 1e-6  # Orthogonal vectors should give 0.5 in normalized range

    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        calc = SimilarityCalculator(metric=SimilarityMetric.COSINE)
        vec1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        similarity = calc.calculate_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_euclidean_similarity_identical_vectors(self):
        """Test euclidean similarity with identical vectors."""
        calc = SimilarityCalculator(metric=SimilarityMetric.EUCLIDEAN)
        vec1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        vec2 = vec1.copy()
        
        similarity = calc.calculate_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6

    def test_dot_product_similarity(self):
        """Test dot product similarity calculation."""
        calc = SimilarityCalculator(metric=SimilarityMetric.DOT_PRODUCT)
        
        # Normalized vectors with known dot product
        vec1 = np.ones(EMBEDDING_DIM, dtype=np.float32) / np.sqrt(EMBEDDING_DIM)
        vec2 = np.ones(EMBEDDING_DIM, dtype=np.float32) / np.sqrt(EMBEDDING_DIM)
        
        similarity = calc.calculate_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6

    def test_calculate_similarity_dimension_mismatch(self):
        """Test error when embedding dimensions don't match."""
        calc = SimilarityCalculator()
        vec1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        vec2 = np.random.rand(EMBEDDING_DIM - 1).astype(np.float32)
        
        with pytest.raises(ValueError, match=f"Embedding must have dimension {EMBEDDING_DIM}"):
            calc.calculate_similarity(vec1, vec2)

    def test_calculate_similarity_invalid_dimensions(self):
        """Test error when embeddings have wrong dimensions."""
        calc = SimilarityCalculator()
        vec1 = np.random.rand(EMBEDDING_DIM + 10).astype(np.float32)
        vec2 = np.random.rand(EMBEDDING_DIM + 10).astype(np.float32)
        
        with pytest.raises(ValueError, match=f"Embedding must have dimension {EMBEDDING_DIM}"):
            calc.calculate_similarity(vec1, vec2)

    def test_is_similar(self):
        """Test similarity threshold checking."""
        calc = SimilarityCalculator(similarity_threshold=0.8)
        
        assert calc.is_similar(0.9) is True
        assert calc.is_similar(0.8) is True
        assert calc.is_similar(0.7) is False

    def test_find_similar_groups(self):
        """Test finding similar groups from centroids."""
        calc = SimilarityCalculator(similarity_threshold=0.7)
        
        # Create test embedding
        embedding_vector = np.random.rand(EMBEDDING_DIM).tolist()
        story_embedding = StoryEmbedding(
            news_url_id="test-story-1",
            embedding_vector=embedding_vector,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        # Create test centroids (one similar, one not)
        similar_centroid = GroupCentroid(
            group_id="group-1",
            centroid_vector=embedding_vector,  # Same as story embedding
            member_count=2
        )
        
        different_centroid = GroupCentroid(
            group_id="group-2",
            centroid_vector=[0.0] * EMBEDDING_DIM,  # Very different
            member_count=1
        )
        
        centroids = [similar_centroid, different_centroid]
        matches = calc.find_similar_groups(story_embedding, centroids)
        
        # Should find the similar group
        assert len(matches) >= 1
        assert matches[0][0] == "group-1"  # First match should be the similar group
        assert matches[0][1] > 0.7  # Similarity should be above threshold

    def test_find_best_matching_group(self):
        """Test finding the single best matching group."""
        calc = SimilarityCalculator(similarity_threshold=0.5)
        
        embedding_vector = np.random.rand(EMBEDDING_DIM).tolist()
        story_embedding = StoryEmbedding(
            news_url_id="test-story-1",
            embedding_vector=embedding_vector,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        centroids = [
            GroupCentroid(
                group_id="group-1",
                centroid_vector=embedding_vector,  # Exact match
                member_count=2
            ),
            GroupCentroid(
                group_id="group-2",
                centroid_vector=[0.5] * EMBEDDING_DIM,  # Partial match
                member_count=1
            )
        ]
        
        best_match = calc.find_best_matching_group(story_embedding, centroids)
        
        assert best_match is not None
        assert best_match[0] == "group-1"  # Should be the exact match
        assert best_match[1] > 0.9  # High similarity

    def test_find_best_matching_group_no_matches(self):
        """Test when no groups match the threshold."""
        calc = SimilarityCalculator(similarity_threshold=0.99)  # Very high threshold
        
        embedding_vector = np.random.rand(EMBEDDING_DIM).tolist()
        story_embedding = StoryEmbedding(
            news_url_id="test-story-1",
            embedding_vector=embedding_vector,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        centroids = [
            GroupCentroid(
                group_id="group-1",
                centroid_vector=[0.1] * EMBEDDING_DIM,  # Very different
                member_count=1
            )
        ]
        
        best_match = calc.find_best_matching_group(story_embedding, centroids)
        assert best_match is None

    def test_batch_calculate_similarities(self):
        """Test batch similarity calculation."""
        calc = SimilarityCalculator()
        
        embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        centroids = [
            np.random.rand(EMBEDDING_DIM).astype(np.float32),
            np.random.rand(EMBEDDING_DIM).astype(np.float32),
            embedding.copy()  # One identical
        ]
        
        similarities = calc.batch_calculate_similarities(embedding, centroids)
        
        assert len(similarities) == 3
        assert all(0.0 <= sim <= 1.0 for sim in similarities)
        assert abs(similarities[2] - 1.0) < 1e-6  # Identical should be ~1.0

    def test_update_threshold(self):
        """Test updating similarity threshold."""
        calc = SimilarityCalculator(similarity_threshold=0.8)
        
        calc.update_threshold(0.9)
        assert calc.threshold == 0.9
        
        with pytest.raises(ValueError):
            calc.update_threshold(-0.1)

    def test_update_metric(self):
        """Test updating similarity metric."""
        calc = SimilarityCalculator(metric=SimilarityMetric.COSINE)
        
        calc.update_metric(SimilarityMetric.EUCLIDEAN)
        assert calc.metric == SimilarityMetric.EUCLIDEAN
        
        with pytest.raises(ValueError):
            calc.update_metric("invalid")

    def test_validate_embedding_list_input(self):
        """Test that embedding lists are properly converted to numpy arrays."""
        calc = SimilarityCalculator()
        
        vec1_list = [0.5] * EMBEDDING_DIM
        vec2_list = [0.7] * EMBEDDING_DIM
        
        # Should work with list inputs
        similarity = calc.calculate_similarity(vec1_list, vec2_list)
        assert 0.0 <= similarity <= 1.0

    def test_validate_embedding_non_finite_values(self):
        """Test error handling for non-finite values in embeddings."""
        calc = SimilarityCalculator()
        
        vec1 = np.full(EMBEDDING_DIM, np.inf, dtype=np.float32)
        vec2 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        
        with pytest.raises(ValueError, match="Embedding contains non-finite values"):
            calc.calculate_similarity(vec1, vec2)

    def test_validate_embedding_nan_values(self):
        """Test error handling for NaN values in embeddings."""
        calc = SimilarityCalculator()
        
        vec1 = np.full(EMBEDDING_DIM, np.nan, dtype=np.float32)
        vec2 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        
        with pytest.raises(ValueError, match="Embedding contains non-finite values"):
            calc.calculate_similarity(vec1, vec2)


class TestSimilarityResult:
    """Test suite for SimilarityResult class."""

    def test_similarity_result_creation(self):
        """Test creating a valid SimilarityResult."""
        result = SimilarityResult(
            story1_id="story-1",
            story2_id="story-2",
            similarity_score=0.85,
            metric_used=SimilarityMetric.COSINE,
            calculated_at=datetime.now(timezone.utc)
        )
        
        assert result.story1_id == "story-1"
        assert result.story2_id == "story-2"
        assert result.similarity_score == 0.85
        assert result.metric_used == SimilarityMetric.COSINE

    def test_similarity_result_invalid_score(self):
        """Test that invalid similarity scores raise an error."""
        with pytest.raises(ValueError, match="similarity_score must be in range"):
            SimilarityResult(
                story1_id="story-1",
                story2_id="story-2",
                similarity_score=1.5,  # Invalid: > 1.0
                metric_used=SimilarityMetric.COSINE,
                calculated_at=datetime.now(timezone.utc)
            )
        
        with pytest.raises(ValueError, match="similarity_score must be in range"):
            SimilarityResult(
                story1_id="story-1",
                story2_id="story-2",
                similarity_score=-0.1,  # Invalid: < 0.0
                metric_used=SimilarityMetric.COSINE,
                calculated_at=datetime.now(timezone.utc)
            )