"""
Group centroid management for story grouping.

This module provides functionality for calculating, updating, and managing
group centroids from member embeddings. Centroids are used for efficient
similarity comparison when assigning new stories to groups.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import logging

from .models import StoryEmbedding, StoryGroup, GroupCentroid, EMBEDDING_DIM


logger = logging.getLogger(__name__)


@dataclass
class CentroidUpdateResult:
    """Result of a centroid update operation."""
    group_id: str
    old_centroid: Optional[List[float]]
    new_centroid: List[float]
    member_count: int
    update_successful: bool
    error_message: Optional[str] = None


class GroupCentroidManager:
    """
    Manage group centroids for story similarity grouping.
    
    Provides methods to calculate centroids from member embeddings, update
    centroids when group membership changes, and maintain centroid consistency.
    Uses weighted averaging and handles edge cases like empty groups.
    """

    def __init__(self):
        """Initialize centroid manager."""
        pass

    def calculate_centroid(self, embeddings: List[StoryEmbedding]) -> np.ndarray:
        """
        Calculate centroid from a list of story embeddings.
        
        Args:
            embeddings: List of story embeddings to calculate centroid from
            
        Returns:
            Centroid vector as numpy array
            
        Raises:
            ValueError: If embeddings list is empty or contains invalid embeddings
        """
        if not embeddings:
            raise ValueError("Cannot calculate centroid from empty embeddings list")
        
        # Convert embeddings to numpy arrays and validate
        embedding_arrays = []
        for embedding in embeddings:
            if not embedding.embedding_vector:
                raise ValueError(f"Embedding for story {embedding.news_url_id} has no vector data")
            
            if len(embedding.embedding_vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding for story {embedding.news_url_id} has dimension "
                    f"{len(embedding.embedding_vector)}, expected {EMBEDDING_DIM}"
                )
            
            embedding_array = np.array(embedding.embedding_vector, dtype=np.float32)
            if not np.isfinite(embedding_array).all():
                raise ValueError(f"Embedding for story {embedding.news_url_id} contains non-finite values")
            
            embedding_arrays.append(embedding_array)
        
        # Calculate mean centroid
        centroid_matrix = np.stack(embedding_arrays)
        centroid = np.mean(centroid_matrix, axis=0)
        
        # Normalize centroid to unit length for consistency
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid

    def calculate_weighted_centroid(
        self,
        embeddings: List[StoryEmbedding],
        weights: List[float]
    ) -> np.ndarray:
        """
        Calculate weighted centroid from embeddings and weights.
        
        Args:
            embeddings: List of story embeddings
            weights: List of weights corresponding to each embedding
            
        Returns:
            Weighted centroid vector as numpy array
            
        Raises:
            ValueError: If inputs are invalid or mismatched lengths
        """
        if not embeddings:
            raise ValueError("Cannot calculate centroid from empty embeddings list")
        
        if len(embeddings) != len(weights):
            raise ValueError("Number of embeddings must match number of weights")
        
        if not all(w >= 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        
        # Convert embeddings to numpy arrays
        embedding_arrays = []
        for embedding in embeddings:
            embedding_array = np.array(embedding.embedding_vector, dtype=np.float32)
            embedding_arrays.append(embedding_array)
        
        # Calculate weighted average
        weighted_sum = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for embedding_array, weight in zip(embedding_arrays, weights):
            weighted_sum += embedding_array * weight
        
        centroid = weighted_sum / total_weight
        
        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid

    def update_group_centroid(
        self,
        group: StoryGroup,
        member_embeddings: List[StoryEmbedding]
    ) -> CentroidUpdateResult:
        """
        Update a group's centroid based on current member embeddings.
        
        Args:
            group: StoryGroup to update centroid for
            member_embeddings: Current embeddings of all group members
            
        Returns:
            CentroidUpdateResult with update details
        """
        old_centroid = group.centroid_embedding.copy() if group.centroid_embedding else None
        
        try:
            if not member_embeddings:
                # Handle empty group case
                new_centroid = np.zeros(EMBEDDING_DIM, dtype=np.float32).tolist()
                error_msg = "Group has no members, centroid set to zero vector"
                logger.warning(f"Group {group.id} has no members, setting centroid to zero vector")
                
                return CentroidUpdateResult(
                    group_id=group.id or "",
                    old_centroid=old_centroid,
                    new_centroid=new_centroid,
                    member_count=0,
                    update_successful=True,
                    error_message=error_msg
                )
            
            # Calculate new centroid
            new_centroid_array = self.calculate_centroid(member_embeddings)
            new_centroid = new_centroid_array.tolist()
            
            # Update group centroid
            group.centroid_embedding = new_centroid
            group.member_count = len(member_embeddings)
            group.updated_at = datetime.now(timezone.utc)
            
            logger.info(f"Updated centroid for group {group.id} with {len(member_embeddings)} members")
            
            return CentroidUpdateResult(
                group_id=group.id or "",
                old_centroid=old_centroid,
                new_centroid=new_centroid,
                member_count=len(member_embeddings),
                update_successful=True
            )
            
        except Exception as e:
            error_msg = f"Failed to update centroid for group {group.id}: {str(e)}"
            logger.error(error_msg)
            
            return CentroidUpdateResult(
                group_id=group.id or "",
                old_centroid=old_centroid,
                new_centroid=old_centroid or [0.0] * EMBEDDING_DIM,
                member_count=len(member_embeddings),
                update_successful=False,
                error_message=error_msg
            )

    def create_group_centroid(
        self,
        group: StoryGroup,
        initial_embeddings: List[StoryEmbedding]
    ) -> GroupCentroid:
        """
        Create a GroupCentroid object from a StoryGroup and its member embeddings.
        
        Args:
            group: StoryGroup to create centroid for
            initial_embeddings: Initial member embeddings
            
        Returns:
            GroupCentroid object
            
        Raises:
            ValueError: If group or embeddings are invalid
        """
        if not group.id:
            raise ValueError("Group must have an ID")
        
        if not initial_embeddings:
            # Create empty centroid for group with no members
            centroid_vector = [0.0] * EMBEDDING_DIM
        else:
            # Calculate centroid from embeddings
            centroid_array = self.calculate_centroid(initial_embeddings)
            centroid_vector = centroid_array.tolist()
            
            # Update group with calculated centroid
            group.centroid_embedding = centroid_vector
            group.member_count = len(initial_embeddings)
        
        return GroupCentroid(
            group_id=group.id,
            centroid_vector=centroid_vector,
            member_count=len(initial_embeddings),
            last_updated=datetime.now(timezone.utc)
        )

    def add_embedding_to_centroid(
        self,
        current_centroid: np.ndarray,
        current_count: int,
        new_embedding: StoryEmbedding
    ) -> np.ndarray:
        """
        Incrementally update centroid when adding a new embedding.
        
        This is more efficient than recalculating from all embeddings.
        
        Args:
            current_centroid: Current group centroid
            current_count: Current number of members in group
            new_embedding: New embedding to add
            
        Returns:
            Updated centroid vector
            
        Raises:
            ValueError: If inputs are invalid
        """
        if current_count < 0:
            raise ValueError("Current count must be non-negative")
        
        if current_count == 0:
            # First embedding becomes the centroid
            new_array = np.array(new_embedding.embedding_vector, dtype=np.float32)
            norm = np.linalg.norm(new_array)
            return new_array / norm if norm > 0 else new_array
        
        # Validate current centroid
        if len(current_centroid) != EMBEDDING_DIM:
            raise ValueError(f"Current centroid has dimension {len(current_centroid)}, expected {EMBEDDING_DIM}")
        
        # Validate new embedding
        if len(new_embedding.embedding_vector) != EMBEDDING_DIM:
            raise ValueError(f"New embedding has dimension {len(new_embedding.embedding_vector)}, expected {EMBEDDING_DIM}")
        
        new_array = np.array(new_embedding.embedding_vector, dtype=np.float32)
        
        # Calculate incremental update: new_centroid = (old_centroid * count + new_embedding) / (count + 1)
        updated_centroid = (current_centroid * current_count + new_array) / (current_count + 1)
        
        # Normalize updated centroid
        norm = np.linalg.norm(updated_centroid)
        if norm > 0:
            updated_centroid = updated_centroid / norm
        
        return updated_centroid

    def remove_embedding_from_centroid(
        self,
        current_centroid: np.ndarray,
        current_count: int,
        removed_embedding: StoryEmbedding
    ) -> Optional[np.ndarray]:
        """
        Incrementally update centroid when removing an embedding.
        
        Args:
            current_centroid: Current group centroid
            current_count: Current number of members in group
            removed_embedding: Embedding being removed
            
        Returns:
            Updated centroid vector, or None if group becomes empty
            
        Raises:
            ValueError: If inputs are invalid
        """
        if current_count <= 0:
            raise ValueError("Cannot remove from empty group")
        
        if current_count == 1:
            # Group becomes empty after removal
            return None
        
        # Validate inputs
        if len(current_centroid) != EMBEDDING_DIM:
            raise ValueError(f"Current centroid has dimension {len(current_centroid)}, expected {EMBEDDING_DIM}")
        
        if len(removed_embedding.embedding_vector) != EMBEDDING_DIM:
            raise ValueError(f"Removed embedding has dimension {len(removed_embedding.embedding_vector)}, expected {EMBEDDING_DIM}")
        
        removed_array = np.array(removed_embedding.embedding_vector, dtype=np.float32)
        
        # Calculate incremental update: new_centroid = (old_centroid * count - removed_embedding) / (count - 1)
        updated_centroid = (current_centroid * current_count - removed_array) / (current_count - 1)
        
        # Normalize updated centroid
        norm = np.linalg.norm(updated_centroid)
        if norm > 0:
            updated_centroid = updated_centroid / norm
        
        return updated_centroid

    def validate_centroid(self, centroid: List[float]) -> bool:
        """
        Validate that a centroid vector is properly formatted.
        
        Args:
            centroid: Centroid vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(centroid, list):
                return False
            
            if len(centroid) != EMBEDDING_DIM:
                return False
            
            centroid_array = np.array(centroid, dtype=np.float32)
            if not np.isfinite(centroid_array).all():
                return False
            
            return True
            
        except Exception:
            return False

    def centroid_similarity(self, centroid1: List[float], centroid2: List[float]) -> float:
        """
        Calculate similarity between two centroids using cosine similarity.
        
        Args:
            centroid1: First centroid vector
            centroid2: Second centroid vector
            
        Returns:
            Cosine similarity score between 0.0 and 1.0
            
        Raises:
            ValueError: If centroids are invalid
        """
        if not self.validate_centroid(centroid1) or not self.validate_centroid(centroid2):
            raise ValueError("Invalid centroid vectors")
        
        vec1 = np.array(centroid1, dtype=np.float32)
        vec2 = np.array(centroid2, dtype=np.float32)
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        # Convert to 0-1 range
        return max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))