"""
Group Manager for story similarity grouping.

This module implements the core logic for assigning stories to groups,
managing group lifecycle, and handling group membership operations.
Implements tasks 6.1, 6.2, and 6.3 from the story grouping specification.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from uuid import uuid4

from .models import (
    StoryEmbedding, 
    StoryGroup, 
    StoryGroupMember, 
    ContextSummary,
    GroupCentroid,
    GroupStatus,
    EMBEDDING_DIM
)
from .similarity import SimilarityCalculator, SimilarityMetric
from .centroid_manager import GroupCentroidManager


logger = logging.getLogger(__name__)


@dataclass
class GroupAssignmentResult:
    """Result of assigning a story to a group."""
    news_url_id: str
    group_id: str
    similarity_score: float
    is_new_group: bool
    assignment_successful: bool
    error_message: Optional[str] = None


@dataclass
class GroupMembershipValidation:
    """Result of validating group membership."""
    is_valid: bool
    error_message: Optional[str] = None
    duplicate_found: bool = False
    group_at_capacity: bool = False


class GroupStorageManager:
    """
    Storage manager for group-related database operations.
    
    Handles database operations for story groups, embeddings, and memberships
    using the existing Supabase client patterns.
    """
    
    def __init__(self, supabase_client: Any):
        """Initialize storage manager with Supabase client."""
        self.client = supabase_client
        self.table_embeddings = os.getenv("STORY_EMBEDDINGS_TABLE", "story_embeddings")
        self.table_groups = os.getenv("STORY_GROUPS_TABLE", "story_groups")  
        self.table_members = os.getenv("STORY_GROUP_MEMBERS_TABLE", "story_group_members")
        self.table_summaries = os.getenv("CONTEXT_SUMMARIES_TABLE", "context_summaries")
        
    async def store_embedding(self, embedding: StoryEmbedding) -> bool:
        """Store story embedding in database."""
        try:
            payload = embedding.to_db()
            
            # Check if embedding already exists
            existing = self.client.table(self.table_embeddings)\
                .select("id")\
                .eq("news_url_id", embedding.news_url_id)\
                .execute()
            
            if existing.data:
                # Update existing embedding
                self.client.table(self.table_embeddings)\
                    .update(payload)\
                    .eq("news_url_id", embedding.news_url_id)\
                    .execute()
            else:
                # Insert new embedding
                self.client.table(self.table_embeddings)\
                    .insert(payload)\
                    .execute()
            
            logger.info(f"Stored embedding for story {embedding.news_url_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding for {embedding.news_url_id}: {e}")
            return False

    async def upsert_context_summary(self, summary: ContextSummary) -> bool:
        """Insert or update a context summary record for a story."""
        try:
            summary.validate()

            # Ensure generated_at is set for TTL-based consumers
            if summary.generated_at is None:
                summary.generated_at = datetime.now(timezone.utc)

            payload = summary.to_db()

            existing = self.client.table(self.table_summaries)\
                .select("id")\
                .eq("news_url_id", summary.news_url_id)\
                .execute()

            table = self.client.table(self.table_summaries)
            if existing.data:
                table.update(payload).eq("news_url_id", summary.news_url_id).execute()
            else:
                table.insert(payload).execute()

            logger.info(f"Upserted context summary for {summary.news_url_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert context summary for {summary.news_url_id}: {e}")
            return False
    
    async def store_group(self, group: StoryGroup) -> bool:
        """Store new story group."""
        try:
            payload = group.to_db()
            
            if group.id:
                # Update existing group
                self.client.table(self.table_groups)\
                    .update(payload)\
                    .eq("id", group.id)\
                    .execute()
            else:
                # Insert new group and get ID
                resp = self.client.table(self.table_groups)\
                    .insert(payload)\
                    .execute()
                
                if resp.data and len(resp.data) > 0:
                    group.id = resp.data[0]["id"]
            
            logger.info(f"Stored group {group.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store group {group.id}: {e}")
            return False
    
    async def add_member_to_group(self, group_id: str, news_url_id: str, similarity_score: float) -> bool:
        """Add story to group membership table."""
        try:
            member = StoryGroupMember(
                group_id=group_id,
                news_url_id=news_url_id,
                similarity_score=similarity_score,
                added_at=datetime.now(timezone.utc)
            )
            
            payload = member.to_db()
            self.client.table(self.table_members)\
                .insert(payload)\
                .execute()
            
            logger.info(f"Added story {news_url_id} to group {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add story {news_url_id} to group {group_id}: {e}")
            return False
    
    async def get_group_centroids(self) -> List[GroupCentroid]:
        """Retrieve all group centroids for similarity comparison."""
        try:
            resp = self.client.table(self.table_groups)\
                .select("id,centroid_embedding,member_count,updated_at")\
                .is_("centroid_embedding", "not.null")\
                .execute()
            
            centroids = []
            for row in resp.data:
                centroid = GroupCentroid(
                    group_id=row["id"],
                    centroid_vector=row["centroid_embedding"],
                    member_count=row["member_count"],
                    last_updated=row.get("updated_at")
                )
                centroids.append(centroid)
            
            logger.info(f"Retrieved {len(centroids)} group centroids")
            return centroids
            
        except Exception as e:
            logger.error(f"Failed to retrieve group centroids: {e}")
            return []
    
    async def get_group_embeddings(self, group_id: str) -> List[StoryEmbedding]:
        """Get all embeddings for a specific group."""
        try:
            # Get member URLs for the group
            members_resp = self.client.table(self.table_members)\
                .select("news_url_id")\
                .eq("group_id", group_id)\
                .execute()
            
            if not members_resp.data:
                return []
            
            url_ids = [m["news_url_id"] for m in members_resp.data]
            
            # Get embeddings for these URLs
            embeddings_resp = self.client.table(self.table_embeddings)\
                .select("*")\
                .in_("news_url_id", url_ids)\
                .execute()
            
            embeddings = []
            for row in embeddings_resp.data:
                embedding = StoryEmbedding.from_db(row)
                embeddings.append(embedding)
            
            logger.info(f"Retrieved {len(embeddings)} embeddings for group {group_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings for group {group_id}: {e}")
            return []
    
    async def get_group_by_id(self, group_id: str) -> Optional[StoryGroup]:
        """Get a specific group by ID."""
        try:
            resp = self.client.table(self.table_groups)\
                .select("*")\
                .eq("id", group_id)\
                .execute()
            
            if resp.data and len(resp.data) > 0:
                return StoryGroup.from_db(resp.data[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve group {group_id}: {e}")
            return None
    
    async def update_group_status(self, group_id: str, status: GroupStatus) -> bool:
        """Update group status and timestamp."""
        try:
            self.client.table(self.table_groups)\
                .update({
                    "status": status.value,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })\
                .eq("id", group_id)\
                .execute()
            
            logger.info(f"Updated group {group_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update group {group_id} status: {e}")
            return False
    
    async def get_embedding_by_url_id(self, news_url_id: str) -> Optional[StoryEmbedding]:
        """Check if embedding already exists for a story."""
        try:
            resp = self.client.table(self.table_embeddings)\
                .select("*")\
                .eq("news_url_id", news_url_id)\
                .execute()
            
            if resp.data and len(resp.data) > 0:
                return StoryEmbedding.from_db(resp.data[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding for {news_url_id}: {e}")
            return None
    
    async def check_membership_exists(self, group_id: str, news_url_id: str) -> bool:
        """Check if story is already a member of the group."""
        try:
            resp = self.client.table(self.table_members)\
                .select("id")\
                .eq("group_id", group_id)\
                .eq("news_url_id", news_url_id)\
                .execute()
            
            return bool(resp.data)
            
        except Exception as e:
            logger.error(f"Failed to check membership for {news_url_id} in group {group_id}: {e}")
            return False


class GroupManager:
    """
    Main group manager for story assignment and lifecycle management.
    
    Implements core functionality for:
    - Task 6.1: Group assignment and story processing
    - Task 6.2: Group lifecycle and status management  
    - Task 6.3: Group membership operations with validation
    """
    
    def __init__(
        self,
        storage_manager: GroupStorageManager,
        similarity_calculator: SimilarityCalculator,
        centroid_manager: GroupCentroidManager,
        similarity_threshold: float = 0.8,
        max_group_size: int = 50
    ):
        """
        Initialize GroupManager.
        
        Args:
            storage_manager: Database storage manager
            similarity_calculator: Similarity calculation engine
            centroid_manager: Centroid calculation manager
            similarity_threshold: Minimum similarity for group assignment
            max_group_size: Maximum number of stories per group
        """
        self.storage = storage_manager
        self.similarity_calc = similarity_calculator
        self.centroid_manager = centroid_manager
        self.similarity_threshold = similarity_threshold
        self.max_group_size = max_group_size
        
        # Update similarity calculator threshold
        self.similarity_calc.update_threshold(similarity_threshold)
    
    # Task 6.1: Group assignment and story processing
    
    async def process_new_story(self, embedding: StoryEmbedding) -> GroupAssignmentResult:
        """
        Process new story and assign to group or create new group.
        
        Core implementation of Task 6.1: story assignment logic.
        
        Args:
            embedding: Story embedding to process
            
        Returns:
            GroupAssignmentResult with assignment details
        """
        try:
            logger.info(f"Processing new story: {embedding.news_url_id}")
            
            # Find best matching group
            best_match = await self.find_best_matching_group(embedding)
            
            if best_match:
                group_id, similarity_score = best_match
                logger.info(f"Found matching group {group_id} with similarity {similarity_score:.3f}")
                
                # Add story to existing group
                success = await self.add_story_to_group(group_id, embedding, similarity_score)
                
                return GroupAssignmentResult(
                    news_url_id=embedding.news_url_id,
                    group_id=group_id,
                    similarity_score=similarity_score,
                    is_new_group=False,
                    assignment_successful=success
                )
            else:
                logger.info(f"No matching group found, creating new group")
                
                # Create new group
                group_id = await self.create_new_group(embedding)
                
                return GroupAssignmentResult(
                    news_url_id=embedding.news_url_id,
                    group_id=group_id,
                    similarity_score=1.0,  # Perfect match with itself
                    is_new_group=True,
                    assignment_successful=bool(group_id)
                )
                
        except Exception as e:
            error_msg = f"Failed to process story {embedding.news_url_id}: {str(e)}"
            logger.error(error_msg)
            
            return GroupAssignmentResult(
                news_url_id=embedding.news_url_id,
                group_id="",
                similarity_score=0.0,
                is_new_group=False,
                assignment_successful=False,
                error_message=error_msg
            )
    
    async def find_best_matching_group(self, embedding: StoryEmbedding) -> Optional[Tuple[str, float]]:
        """
        Find the best matching existing group for a story embedding.
        
        Args:
            embedding: Story embedding to find match for
            
        Returns:
            (group_id, similarity_score) tuple for best match, or None if no match above threshold
        """
        try:
            # Get all group centroids
            centroids = await self.storage.get_group_centroids()
            
            if not centroids:
                logger.info("No existing groups found")
                return None
            
            # Find best matching group using similarity calculator
            best_match = self.similarity_calc.find_best_matching_group(embedding, centroids)
            
            if best_match:
                group_id, similarity_score = best_match
                
                # Validate that group isn't at capacity
                group = await self.storage.get_group_by_id(group_id)
                if group and group.member_count >= self.max_group_size:
                    logger.warning(f"Group {group_id} is at capacity ({group.member_count}), skipping")
                    return None
                
                logger.info(f"Best matching group: {group_id} (similarity: {similarity_score:.3f})")
                return (group_id, similarity_score)
            
            logger.info("No groups above similarity threshold")
            return None
            
        except Exception as e:
            logger.error(f"Error finding best matching group: {e}")
            return None
    
    async def create_new_group(self, story_embedding: StoryEmbedding) -> str:
        """
        Create new group with story as initial member.
        
        Args:
            story_embedding: Initial story embedding for the group
            
        Returns:
            Group ID of created group, or empty string if failed
        """
        try:
            # Create new group with centroid from single embedding
            group = StoryGroup(
                id=str(uuid4()),
                member_count=1,
                status=GroupStatus.NEW,
                tags=[],
                centroid_embedding=story_embedding.embedding_vector.copy(),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Store group
            success = await self.storage.store_group(group)
            if not success:
                logger.error(f"Failed to store new group")
                return ""
            
            # Add story as first member
            member_success = await self.storage.add_member_to_group(
                group.id, 
                story_embedding.news_url_id, 
                1.0  # Perfect similarity with itself
            )
            
            if not member_success:
                logger.error(f"Failed to add initial member to group {group.id}")
                return ""
            
            logger.info(f"Created new group {group.id} with initial story {story_embedding.news_url_id}")
            return group.id
            
        except Exception as e:
            logger.error(f"Failed to create new group: {e}")
            return ""
    
    # Task 6.2: Group lifecycle and status management
    
    async def update_group_lifecycle(self, group_id: str, new_member_added: bool = False) -> bool:
        """
        Update group lifecycle status based on activity.
        
        Implements Task 6.2: group status tracking with timestamps.
        
        Args:
            group_id: ID of group to update
            new_member_added: Whether a new member was just added
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            group = await self.storage.get_group_by_id(group_id)
            if not group:
                logger.error(f"Group {group_id} not found for lifecycle update")
                return False
            
            # Determine new status based on activity and current state
            new_status = self._determine_new_status(group, new_member_added)
            
            if new_status != group.status:
                success = await self.storage.update_group_status(group_id, new_status)
                if success:
                    logger.info(f"Updated group {group_id} status from {group.status.value} to {new_status.value}")
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update group lifecycle for {group_id}: {e}")
            return False
    
    def _determine_new_status(self, group: StoryGroup, new_member_added: bool) -> GroupStatus:
        """
        Determine new group status based on current state and activity.
        
        Logic:
        - NEW: Just created or first member
        - UPDATED: Recently modified (new member added)
        - STABLE: No recent changes
        """
        if group.member_count <= 1:
            return GroupStatus.NEW
        elif new_member_added:
            return GroupStatus.UPDATED
        else:
            # Check if enough time has passed to consider stable
            if group.updated_at:
                hours_since_update = (datetime.now(timezone.utc) - group.updated_at).total_seconds() / 3600
                if hours_since_update > 24:  # 24 hours without updates = stable
                    return GroupStatus.STABLE
            
            return group.status  # Keep current status
    
    async def add_group_tags(self, group_id: str, tags: List[str]) -> bool:
        """
        Add tags to a group for categorization.
        
        Part of Task 6.2: group tagging system.
        
        Args:
            group_id: ID of group to tag
            tags: List of tags to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            group = await self.storage.get_group_by_id(group_id)
            if not group:
                logger.error(f"Group {group_id} not found for tagging")
                return False
            
            # Add new tags, avoiding duplicates
            existing_tags = set(group.tags or [])
            new_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            all_tags = list(existing_tags.union(set(new_tags)))
            
            # Update group with new tags
            updated_data = {
                "tags": all_tags,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            try:
                self.storage.client.table(self.storage.table_groups)\
                    .update(updated_data)\
                    .eq("id", group_id)\
                    .execute()
                
                logger.info(f"Added tags {new_tags} to group {group_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update group {group_id} with tags: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding tags to group {group_id}: {e}")
            return False
    
    # Task 6.3: Group membership operations with validation
    
    async def add_story_to_group(self, group_id: str, story_embedding: StoryEmbedding, similarity_score: float) -> bool:
        """
        Add story to existing group with validation.
        
        Implements Task 6.3: membership operations with validation and duplicate prevention.
        
        Args:
            group_id: ID of group to add story to
            story_embedding: Story embedding to add
            similarity_score: Similarity score between story and group
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Validate membership
            validation = await self.validate_group_membership(group_id, story_embedding.news_url_id)
            if not validation.is_valid:
                logger.warning(f"Membership validation failed: {validation.error_message}")
                return False
            
            # Store embedding if not already stored
            await self.storage.store_embedding(story_embedding)
            
            # Add to membership table
            member_success = await self.storage.add_member_to_group(
                group_id, 
                story_embedding.news_url_id, 
                similarity_score
            )
            
            if not member_success:
                return False
            
            # Update group centroid and member count
            await self.update_group_centroid(group_id)
            
            # Update group lifecycle status
            await self.update_group_lifecycle(group_id, new_member_added=True)
            
            logger.info(f"Successfully added story {story_embedding.news_url_id} to group {group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add story {story_embedding.news_url_id} to group {group_id}: {e}")
            return False
    
    async def validate_group_membership(self, group_id: str, news_url_id: str) -> GroupMembershipValidation:
        """
        Validate that a story can be added to a group.
        
        Implements Task 6.3: membership validation and duplicate prevention.
        
        Args:
            group_id: ID of group to validate membership for
            news_url_id: ID of story to validate
            
        Returns:
            GroupMembershipValidation with validation results
        """
        try:
            # Check if group exists
            group = await self.storage.get_group_by_id(group_id)
            if not group:
                return GroupMembershipValidation(
                    is_valid=False,
                    error_message=f"Group {group_id} does not exist"
                )
            
            # Check for duplicate membership
            duplicate_exists = await self.storage.check_membership_exists(group_id, news_url_id)
            if duplicate_exists:
                return GroupMembershipValidation(
                    is_valid=False,
                    error_message=f"Story {news_url_id} is already a member of group {group_id}",
                    duplicate_found=True
                )
            
            # Check group capacity
            if group.member_count >= self.max_group_size:
                return GroupMembershipValidation(
                    is_valid=False,
                    error_message=f"Group {group_id} is at capacity ({group.member_count}/{self.max_group_size})",
                    group_at_capacity=True
                )
            
            return GroupMembershipValidation(is_valid=True)
            
        except Exception as e:
            return GroupMembershipValidation(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    async def update_group_centroid(self, group_id: str) -> bool:
        """
        Update group centroid based on current member embeddings.
        
        Part of Task 6.3: member count tracking and centroid management.
        
        Args:
            group_id: ID of group to update centroid for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current group
            group = await self.storage.get_group_by_id(group_id)
            if not group:
                logger.error(f"Group {group_id} not found for centroid update")
                return False
            
            # Get all member embeddings
            member_embeddings = await self.storage.get_group_embeddings(group_id)
            
            # Update centroid using centroid manager
            update_result = self.centroid_manager.update_group_centroid(group, member_embeddings)
            
            if update_result.update_successful:
                # Store updated group
                success = await self.storage.store_group(group)
                if success:
                    logger.info(f"Updated centroid for group {group_id} with {len(member_embeddings)} members")
                return success
            else:
                logger.error(f"Centroid update failed: {update_result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update centroid for group {group_id}: {e}")
            return False
    
    async def get_group_member_count(self, group_id: str) -> int:
        """
        Get current member count for a group.
        
        Part of Task 6.3: member count tracking.
        
        Args:
            group_id: ID of group to get count for
            
        Returns:
            Number of members in group, or 0 if error
        """
        try:
            group = await self.storage.get_group_by_id(group_id)
            if group:
                return group.member_count
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get member count for group {group_id}: {e}")
            return 0
    
    # Utility methods
    
    async def get_group_analytics(self, group_id: str) -> Dict[str, Any]:
        """
        Get analytics and metadata for a group.
        
        Args:
            group_id: ID of group to analyze
            
        Returns:
            Dictionary with group analytics data
        """
        try:
            group = await self.storage.get_group_by_id(group_id)
            if not group:
                return {}
            
            member_embeddings = await self.storage.get_group_embeddings(group_id)
            
            # Calculate analytics
            analytics = {
                "group_id": group_id,
                "member_count": group.member_count,
                "status": group.status.value,
                "tags": group.tags or [],
                "created_at": group.created_at.isoformat() if group.created_at else None,
                "updated_at": group.updated_at.isoformat() if group.updated_at else None,
                "has_embeddings": len(member_embeddings),
                "avg_confidence": sum(e.confidence_score for e in member_embeddings) / len(member_embeddings) if member_embeddings else 0.0
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get analytics for group {group_id}: {e}")
