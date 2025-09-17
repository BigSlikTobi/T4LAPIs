"""
Similarity calculation engine for story grouping.

This module provides similarity calculation capabilities for semantic embeddings,
supporting multiple distance metrics and efficient similarity search operations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional

from .models import StoryEmbedding, GroupCentroid, EMBEDDING_DIM


class SimilarityMetric(Enum):
    """Supported similarity metrics for embedding comparison."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class SimilarityResult:
    """Result of a similarity calculation between two stories or embeddings."""
    story1_id: str
    story2_id: str
    similarity_score: float
    metric_used: SimilarityMetric
    calculated_at: datetime

    def __post_init__(self):
        """Validate similarity result data."""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError("similarity_score must be in range [0.0, 1.0]")


class SimilarityCalculator:
    """
    Calculate similarity scores between embeddings using various metrics.
    
    Supports cosine similarity, euclidean distance, and dot product similarity.
    Provides configurable thresholds for similarity matching and efficient
    batch operations for comparing against multiple group centroids.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        cosine_scaling_gamma: float = 1.0,
    ):
        """
        Initialize similarity calculator.
        
        Args:
            similarity_threshold: Minimum similarity score to consider as similar (0.0-1.0)
            metric: Distance metric to use for similarity calculations
        """
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in range [0.0, 1.0]")
        
        if cosine_scaling_gamma <= 0:
            raise ValueError("cosine_scaling_gamma must be positive")

        self.threshold = similarity_threshold
        self.metric = metric
        self._cosine_gamma = float(cosine_scaling_gamma)

    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate similarity score between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0.0 and 1.0
            
        Raises:
            ValueError: If embeddings have different dimensions or invalid values
        """
        # Validate inputs
        embedding1 = self._validate_embedding(embedding1)
        embedding2 = self._validate_embedding(embedding2)
        
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same dimensions")
        
        # Calculate similarity based on metric
        if self.metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(embedding1, embedding2)
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_similarity(embedding1, embedding2)
        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            return self._dot_product_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")

    def find_similar_groups(
        self,
        new_embedding: StoryEmbedding,
        group_centroids: List[GroupCentroid]
    ) -> List[Tuple[str, float]]:
        """
        Find groups with similarity above threshold for a new story embedding.
        
        Args:
            new_embedding: Story embedding to find matches for
            group_centroids: List of group centroids to compare against
            
        Returns:
            List of (group_id, similarity_score) tuples for matches above threshold,
            sorted by similarity score in descending order
        """
        new_vector = self._embedding_to_array(new_embedding.embedding_vector)
        matches = []
        
        for centroid in group_centroids:
            centroid_vector = self._embedding_to_array(centroid.centroid_vector)
            similarity = self.calculate_similarity(new_vector, centroid_vector)
            
            if self.is_similar(similarity):
                matches.append((centroid.group_id, similarity))
        
        # Sort by similarity score in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def find_best_matching_group(
        self,
        new_embedding: StoryEmbedding,
        group_centroids: List[GroupCentroid]
    ) -> Optional[Tuple[str, float]]:
        """
        Find the single best matching group for a new story embedding.
        
        Args:
            new_embedding: Story embedding to find best match for
            group_centroids: List of group centroids to compare against
            
        Returns:
            (group_id, similarity_score) tuple for best match, or None if no match above threshold
        """
        matches = self.find_similar_groups(new_embedding, group_centroids)
        return matches[0] if matches else None

    def is_similar(self, similarity_score: float) -> bool:
        """
        Check if similarity score exceeds threshold.
        
        Args:
            similarity_score: Similarity score to check
            
        Returns:
            True if score is above threshold, False otherwise
        """
        return similarity_score >= self.threshold

    def batch_calculate_similarities(
        self,
        embedding: np.ndarray,
        centroid_vectors: List[np.ndarray]
    ) -> List[float]:
        """
        Calculate similarities between one embedding and multiple centroids efficiently.
        
        Args:
            embedding: Single embedding vector
            centroid_vectors: List of centroid vectors to compare against
            
        Returns:
            List of similarity scores in same order as input centroids
        """
        embedding = self._validate_embedding(embedding)
        similarities = []
        
        for centroid in centroid_vectors:
            centroid = self._validate_embedding(centroid)
            similarity = self.calculate_similarity(embedding, centroid)
            similarities.append(similarity)
            
        return similarities

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Handle zero vectors to avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        # Ensure result is in [0, 1] range (cosine similarity can be [-1, 1])
        similarity = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))
        if self._cosine_gamma != 1.0:
            # Optional scaling of cosine similarity in [0,1];
            # By default gamma=1.0 (no scaling) to match tests' expectations.
            similarity = similarity ** self._cosine_gamma
        return similarity

    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate euclidean distance-based similarity between two vectors."""
        # Calculate euclidean distance and convert to similarity (0-1 range)
        distance = np.linalg.norm(vec1 - vec2)
        
        # Convert distance to similarity using exponential decay
        # This ensures similarity is in [0, 1] range where 0 distance = 1.0 similarity
        max_distance = np.sqrt(2 * EMBEDDING_DIM)  # Theoretical max distance for normalized vectors
        normalized_distance = min(distance / max_distance, 1.0)
        
        return 1.0 - normalized_distance

    def _dot_product_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate dot product-based similarity between two vectors."""
        # For normalized vectors, dot product is equivalent to cosine similarity
        dot_product = np.dot(vec1, vec2)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, (dot_product + 1.0) / 2.0))

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Validate and ensure embedding is proper numpy array.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If embedding is invalid
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.ndim != 1:
            raise ValueError("Embedding must be a 1-dimensional array")
        
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding must have dimension {EMBEDDING_DIM}")
        
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains non-finite values")
        
        return embedding.astype(np.float32)

    def _embedding_to_array(self, embedding_list: List[float]) -> np.ndarray:
        """Convert embedding list to numpy array with validation."""
        return self._validate_embedding(np.array(embedding_list, dtype=np.float32))

    def update_threshold(self, new_threshold: float) -> None:
        """
        Update similarity threshold.
        
        Args:
            new_threshold: New threshold value (0.0-1.0)
            
        Raises:
            ValueError: If threshold is not in valid range
        """
        if not (0.0 <= new_threshold <= 1.0):
            raise ValueError("Threshold must be in range [0.0, 1.0]")
        self.threshold = new_threshold

    def update_metric(self, new_metric: SimilarityMetric) -> None:
        """
        Update similarity metric.
        
        Args:
            new_metric: New similarity metric to use
        """
        if not isinstance(new_metric, SimilarityMetric):
            raise ValueError("Metric must be a SimilarityMetric enum value")
        self.metric = new_metric
