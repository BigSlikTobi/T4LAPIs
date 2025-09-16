"""
Tests for group centroid management functionality.

Tests cover centroid calculation, incremental updates, and integration
with the story grouping models.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.nfl_news_pipeline.centroid_manager import (
    GroupCentroidManager,
    CentroidUpdateResult
)
from src.nfl_news_pipeline.models import (
    StoryEmbedding,
    StoryGroup,
    GroupCentroid,
    GroupStatus,
    EMBEDDING_DIM
)


class TestGroupCentroidManager:
    """Test suite for GroupCentroidManager class."""

    def test_initialization(self):
        """Test initialization of GroupCentroidManager."""
        manager = GroupCentroidManager()
        assert manager is not None

    def test_calculate_centroid_single_embedding(self):
        """Test centroid calculation with a single embedding."""
        manager = GroupCentroidManager()
        
        embedding_vector = np.random.rand(EMBEDDING_DIM).tolist()
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=embedding_vector,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        centroid = manager.calculate_centroid([embedding])
        
        # Single embedding centroid should be normalized version of the embedding
        assert len(centroid) == EMBEDDING_DIM
        assert np.isfinite(centroid).all()
        
        # Should be normalized (unit length)
        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 1e-6

    def test_calculate_centroid_multiple_embeddings(self):
        """Test centroid calculation with multiple embeddings."""
        manager = GroupCentroidManager()
        
        embeddings = []
        for i in range(3):
            embedding_vector = np.random.rand(EMBEDDING_DIM).tolist()
            embedding = StoryEmbedding(
                news_url_id=f"story-{i}",
                embedding_vector=embedding_vector,
                model_name="test-model",
                model_version="1.0",
                summary_text=f"Test story {i}",
                confidence_score=0.9
            )
            embeddings.append(embedding)
        
        centroid = manager.calculate_centroid(embeddings)
        
        assert len(centroid) == EMBEDDING_DIM
        assert np.isfinite(centroid).all()
        
        # Should be normalized
        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 1e-6

    def test_calculate_centroid_empty_list(self):
        """Test that empty embedding list raises error."""
        manager = GroupCentroidManager()
        
        with pytest.raises(ValueError, match="Cannot calculate centroid from empty embeddings list"):
            manager.calculate_centroid([])

    def test_calculate_centroid_invalid_embedding(self):
        """Test error handling for invalid embeddings."""
        manager = GroupCentroidManager()
        
        # Embedding with wrong dimension
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[0.5] * (EMBEDDING_DIM - 10),  # Wrong dimension
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="has dimension"):
            manager.calculate_centroid([embedding])

    def test_calculate_centroid_empty_vector(self):
        """Test error handling for embedding with empty vector."""
        manager = GroupCentroidManager()
        
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[],  # Empty vector
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="has no vector data"):
            manager.calculate_centroid([embedding])

    def test_calculate_centroid_non_finite_values(self):
        """Test error handling for non-finite values in embeddings."""
        manager = GroupCentroidManager()
        
        embedding_vector = [np.inf] * EMBEDDING_DIM
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=embedding_vector,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="contains non-finite values"):
            manager.calculate_centroid([embedding])

    def test_calculate_weighted_centroid(self):
        """Test weighted centroid calculation."""
        manager = GroupCentroidManager()
        
        # Create embeddings with known values
        embedding1 = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story 1",
            confidence_score=0.9
        )
        
        embedding2 = StoryEmbedding(
            news_url_id="story-2",
            embedding_vector=[0.0] + [1.0] + [0.0] * (EMBEDDING_DIM - 2),
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story 2",
            confidence_score=0.9
        )
        
        embeddings = [embedding1, embedding2]
        weights = [0.8, 0.2]  # Favor first embedding
        
        centroid = manager.calculate_weighted_centroid(embeddings, weights)
        
        assert len(centroid) == EMBEDDING_DIM
        assert np.isfinite(centroid).all()
        
        # Should be normalized
        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 1e-6

    def test_calculate_weighted_centroid_mismatched_lengths(self):
        """Test error when embeddings and weights have different lengths."""
        manager = GroupCentroidManager()
        
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[0.5] * EMBEDDING_DIM,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="Number of embeddings must match number of weights"):
            manager.calculate_weighted_centroid([embedding], [0.5, 0.5])

    def test_calculate_weighted_centroid_negative_weights(self):
        """Test error when weights are negative."""
        manager = GroupCentroidManager()
        
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[0.5] * EMBEDDING_DIM,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            manager.calculate_weighted_centroid([embedding], [-0.5])

    def test_calculate_weighted_centroid_zero_total_weight(self):
        """Test error when total weight is zero."""
        manager = GroupCentroidManager()
        
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[0.5] * EMBEDDING_DIM,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="Total weight cannot be zero"):
            manager.calculate_weighted_centroid([embedding], [0.0])

    def test_update_group_centroid_success(self):
        """Test successful group centroid update."""
        manager = GroupCentroidManager()
        
        group = StoryGroup(
            id="group-1",
            member_count=1,
            status=GroupStatus.NEW,
            centroid_embedding=[0.0] * EMBEDDING_DIM
        )
        
        embeddings = [
            StoryEmbedding(
                news_url_id="story-1",
                embedding_vector=np.random.rand(EMBEDDING_DIM).tolist(),
                model_name="test-model",
                model_version="1.0",
                summary_text="Test story 1",
                confidence_score=0.9
            ),
            StoryEmbedding(
                news_url_id="story-2",
                embedding_vector=np.random.rand(EMBEDDING_DIM).tolist(),
                model_name="test-model",
                model_version="1.0",
                summary_text="Test story 2",
                confidence_score=0.9
            )
        ]
        
        result = manager.update_group_centroid(group, embeddings)
        
        assert isinstance(result, CentroidUpdateResult)
        assert result.group_id == "group-1"
        assert result.update_successful is True
        assert result.member_count == 2
        assert result.error_message is None
        assert len(result.new_centroid) == EMBEDDING_DIM
        
        # Group should be updated
        assert group.member_count == 2
        assert group.centroid_embedding == result.new_centroid
        assert group.updated_at is not None

    def test_update_group_centroid_empty_members(self):
        """Test updating centroid with no members."""
        manager = GroupCentroidManager()
        
        group = StoryGroup(
            id="group-1",
            member_count=1,
            status=GroupStatus.NEW,
            centroid_embedding=[0.5] * EMBEDDING_DIM
        )
        
        result = manager.update_group_centroid(group, [])
        
        assert result.update_successful is True
        assert result.member_count == 0
        assert result.new_centroid == [0.0] * EMBEDDING_DIM
        assert "no members" in result.error_message

    def test_create_group_centroid_with_embeddings(self):
        """Test creating GroupCentroid from group and embeddings."""
        manager = GroupCentroidManager()
        
        group = StoryGroup(
            id="group-1",
            member_count=0,
            status=GroupStatus.NEW
        )
        
        embeddings = [
            StoryEmbedding(
                news_url_id="story-1",
                embedding_vector=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
                model_name="test-model",
                model_version="1.0",
                summary_text="Test story",
                confidence_score=0.9
            )
        ]
        
        centroid = manager.create_group_centroid(group, embeddings)
        
        assert isinstance(centroid, GroupCentroid)
        assert centroid.group_id == "group-1"
        assert centroid.member_count == 1
        assert len(centroid.centroid_vector) == EMBEDDING_DIM
        assert centroid.last_updated is not None
        
        # Group should be updated
        assert group.centroid_embedding == centroid.centroid_vector
        assert group.member_count == 1

    def test_create_group_centroid_empty_embeddings(self):
        """Test creating GroupCentroid with no embeddings."""
        manager = GroupCentroidManager()
        
        group = StoryGroup(
            id="group-1",
            member_count=0,
            status=GroupStatus.NEW
        )
        
        centroid = manager.create_group_centroid(group, [])
        
        assert centroid.group_id == "group-1"
        assert centroid.member_count == 0
        assert centroid.centroid_vector == [0.0] * EMBEDDING_DIM

    def test_create_group_centroid_no_group_id(self):
        """Test error when group has no ID."""
        manager = GroupCentroidManager()
        
        group = StoryGroup(
            member_count=0,
            status=GroupStatus.NEW
        )
        
        with pytest.raises(ValueError, match="Group must have an ID"):
            manager.create_group_centroid(group, [])

    def test_add_embedding_to_centroid_first_embedding(self):
        """Test adding first embedding to empty centroid."""
        manager = GroupCentroidManager()
        
        empty_centroid = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        updated_centroid = manager.add_embedding_to_centroid(empty_centroid, 0, embedding)
        
        assert len(updated_centroid) == EMBEDDING_DIM
        assert np.isfinite(updated_centroid).all()
        
        # Should be normalized
        norm = np.linalg.norm(updated_centroid)
        assert abs(norm - 1.0) < 1e-6

    def test_add_embedding_to_centroid_incremental(self):
        """Test incremental centroid update."""
        manager = GroupCentroidManager()
        
        # Start with a centroid
        current_centroid = np.array([1.0] + [0.0] * (EMBEDDING_DIM - 1), dtype=np.float32)
        current_count = 1
        
        # Add another embedding
        new_embedding = StoryEmbedding(
            news_url_id="story-2",
            embedding_vector=[0.0] + [1.0] + [0.0] * (EMBEDDING_DIM - 2),
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story 2",
            confidence_score=0.9
        )
        
        updated_centroid = manager.add_embedding_to_centroid(current_centroid, current_count, new_embedding)
        
        assert len(updated_centroid) == EMBEDDING_DIM
        assert np.isfinite(updated_centroid).all()
        
        # Should be normalized
        norm = np.linalg.norm(updated_centroid)
        assert abs(norm - 1.0) < 1e-6

    def test_add_embedding_negative_count(self):
        """Test error when current count is negative."""
        manager = GroupCentroidManager()
        
        centroid = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[0.5] * EMBEDDING_DIM,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="Current count must be non-negative"):
            manager.add_embedding_to_centroid(centroid, -1, embedding)

    def test_remove_embedding_from_centroid_last_member(self):
        """Test removing the last member from a group."""
        manager = GroupCentroidManager()
        
        current_centroid = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=current_centroid.tolist(),
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        result = manager.remove_embedding_from_centroid(current_centroid, 1, embedding)
        
        assert result is None  # Group becomes empty

    def test_remove_embedding_from_centroid_incremental(self):
        """Test incremental centroid update when removing an embedding."""
        manager = GroupCentroidManager()
        
        # Start with a centroid representing 2 members
        current_centroid = np.array([0.5, 0.5] + [0.0] * (EMBEDDING_DIM - 2), dtype=np.float32)
        current_count = 2
        
        # Remove one embedding
        removed_embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2),
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        updated_centroid = manager.remove_embedding_from_centroid(current_centroid, current_count, removed_embedding)
        
        assert updated_centroid is not None
        assert len(updated_centroid) == EMBEDDING_DIM
        assert np.isfinite(updated_centroid).all()

    def test_remove_embedding_empty_group(self):
        """Test error when trying to remove from empty group."""
        manager = GroupCentroidManager()
        
        centroid = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        embedding = StoryEmbedding(
            news_url_id="story-1",
            embedding_vector=[0.5] * EMBEDDING_DIM,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test story",
            confidence_score=0.9
        )
        
        with pytest.raises(ValueError, match="Cannot remove from empty group"):
            manager.remove_embedding_from_centroid(centroid, 0, embedding)

    def test_validate_centroid_valid(self):
        """Test validation of valid centroid."""
        manager = GroupCentroidManager()
        
        valid_centroid = [0.5] * EMBEDDING_DIM
        assert manager.validate_centroid(valid_centroid) is True

    def test_validate_centroid_invalid_type(self):
        """Test validation of invalid centroid type."""
        manager = GroupCentroidManager()
        
        assert manager.validate_centroid("not a list") is False
        assert manager.validate_centroid(42) is False

    def test_validate_centroid_wrong_dimension(self):
        """Test validation of wrong dimension centroid."""
        manager = GroupCentroidManager()
        
        wrong_dim_centroid = [0.5] * (EMBEDDING_DIM + 10)
        assert manager.validate_centroid(wrong_dim_centroid) is False

    def test_validate_centroid_non_finite(self):
        """Test validation of centroid with non-finite values."""
        manager = GroupCentroidManager()
        
        invalid_centroid = [np.inf] * EMBEDDING_DIM
        assert manager.validate_centroid(invalid_centroid) is False
        
        nan_centroid = [np.nan] * EMBEDDING_DIM
        assert manager.validate_centroid(nan_centroid) is False

    def test_centroid_similarity(self):
        """Test similarity calculation between centroids."""
        manager = GroupCentroidManager()
        
        centroid1 = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        centroid2 = [1.0] + [0.0] * (EMBEDDING_DIM - 1)  # Identical
        
        similarity = manager.centroid_similarity(centroid1, centroid2)
        assert abs(similarity - 1.0) < 1e-6

    def test_centroid_similarity_orthogonal(self):
        """Test similarity between orthogonal centroids."""
        manager = GroupCentroidManager()
        
        centroid1 = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        centroid2 = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
        
        similarity = manager.centroid_similarity(centroid1, centroid2)
        assert abs(similarity - 0.5) < 1e-6  # Should be 0.5 in normalized range

    def test_centroid_similarity_invalid_centroids(self):
        """Test error when centroids are invalid."""
        manager = GroupCentroidManager()
        
        valid_centroid = [0.5] * EMBEDDING_DIM
        invalid_centroid = [0.5] * (EMBEDDING_DIM - 10)
        
        with pytest.raises(ValueError, match="Invalid centroid vectors"):
            manager.centroid_similarity(valid_centroid, invalid_centroid)