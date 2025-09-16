"""Storage and retrieval operations for story groups and group membership."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from supabase import Client

from ..models import StoryGroup, StoryGroupMember, GroupCentroid, GroupStatus

logger = logging.getLogger(__name__)


class GroupStorageManager:
    """Manage storage and retrieval of story groups and membership in Supabase.
    
    Extends existing storage patterns to handle group-specific operations
    including vector similarity search, transaction management, and error handling.
    Implements Tasks 7.1, 7.2, and 7.3 requirements.
    """

    def __init__(self, supabase_client: Client):
        """Initialize with Supabase client.
        
        Args:
            supabase_client: Authenticated Supabase client
        """
        self.supabase = supabase_client
        self.table_groups = "story_groups"
        self.table_members = "story_group_members"
        self.table_embeddings = "story_embeddings"
        
        logger.info("GroupStorageManager initialized")

    # ===== Task 7.1: CRUD Operations =====

    async def create_group(self, group: StoryGroup) -> Optional[str]:
        """Create a new story group.
        
        Args:
            group: The StoryGroup to create
            
        Returns:
            Group ID if successful, None otherwise
            
        Raises:
            ValueError: If group validation fails
            Exception: For database operation errors
        """
        try:
            group.validate()
            
            insert_data = group.to_db()
            # Remove id to let database generate it
            insert_data.pop("id", None)
            
            response = self.supabase.table(self.table_groups).insert(insert_data).execute()
            
            if response.data and len(response.data) > 0:
                group_id = response.data[0].get("id")
                logger.info(f"Successfully created group: {group_id}")
                return group_id
            else:
                logger.error("No data returned when creating group")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create group: {e}")
            raise

    async def get_group(self, group_id: str) -> Optional[StoryGroup]:
        """Retrieve a story group by ID.
        
        Args:
            group_id: The group ID to look up
            
        Returns:
            StoryGroup if found, None otherwise
        """
        try:
            response = self.supabase.table(self.table_groups).select("*").eq(
                "id", group_id
            ).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                return StoryGroup.from_db(response.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve group {group_id}: {e}")
            raise

    async def update_group(self, group: StoryGroup) -> bool:
        """Update an existing story group.
        
        Args:
            group: The updated StoryGroup (must have id)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not group.id:
                raise ValueError("Group ID is required for update")
                
            group.validate()
            update_data = group.to_db()
            
            # Remove id from update data
            group_id = update_data.pop("id")
            
            response = self.supabase.table(self.table_groups).update(update_data).eq(
                "id", group_id
            ).execute()
            
            if response.data:
                logger.debug(f"Successfully updated group: {group_id}")
                return True
            else:
                logger.error(f"No data returned when updating group {group_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update group {group.id}: {e}")
            raise

    async def delete_group(self, group_id: str) -> bool:
        """Delete a story group and all its memberships.
        
        Args:
            group_id: The group ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete group (memberships cascade automatically)
            response = self.supabase.table(self.table_groups).delete().eq(
                "id", group_id
            ).execute()
            
            if response.data:
                logger.info(f"Successfully deleted group: {group_id}")
                return True
            else:
                logger.warning(f"No group found to delete: {group_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete group {group_id}: {e}")
            raise

    async def add_member_to_group(self, membership: StoryGroupMember) -> bool:
        """Add a story to a group.
        
        Args:
            membership: The StoryGroupMember to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            membership.validate()
            
            insert_data = membership.to_db()
            # Remove id to let database generate it
            insert_data.pop("id", None)
            
            response = self.supabase.table(self.table_members).insert(insert_data).execute()
            
            if response.data:
                logger.debug(f"Successfully added member to group {membership.group_id}")
                return True
            else:
                logger.error(f"No data returned when adding member to group {membership.group_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add member to group {membership.group_id}: {e}")
            raise

    async def remove_member_from_group(self, group_id: str, news_url_id: str) -> bool:
        """Remove a story from a group.
        
        Args:
            group_id: The group ID
            news_url_id: The news URL ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.supabase.table(self.table_members).delete().eq(
                "group_id", group_id
            ).eq("news_url_id", news_url_id).execute()
            
            if response.data:
                logger.debug(f"Successfully removed member from group {group_id}")
                return True
            else:
                logger.warning(f"No membership found to remove: {group_id}/{news_url_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove member from group {group_id}: {e}")
            raise

    async def get_group_members(self, group_id: str) -> List[StoryGroupMember]:
        """Get all members of a group.
        
        Args:
            group_id: The group ID
            
        Returns:
            List of StoryGroupMember objects
        """
        try:
            response = self.supabase.table(self.table_members).select("*").eq(
                "group_id", group_id
            ).order("added_at").execute()
            
            members = []
            if response.data:
                for row in response.data:
                    try:
                        members.append(StoryGroupMember.from_db(row))
                    except Exception as e:
                        logger.error(f"Failed to parse member from DB row: {e}")
                        continue
                        
            logger.debug(f"Retrieved {len(members)} members for group {group_id}")
            return members
            
        except Exception as e:
            logger.error(f"Failed to retrieve members for group {group_id}: {e}")
            raise

    # ===== Task 7.2: Vector Similarity Search =====

    async def find_similar_groups(
        self, 
        embedding_vector: List[float], 
        similarity_threshold: float = 0.8,
        max_results: int = 10,
        distance_metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """Find groups similar to the given embedding vector.
        
        Args:
            embedding_vector: The embedding vector to compare against
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            max_results: Maximum number of results to return
            distance_metric: Distance metric ("cosine", "l2", "inner_product")
            
        Returns:
            List of (group_id, similarity_score) tuples, ordered by similarity
        """
        try:
            # Map distance metrics to Supabase operators
            metric_operators = {
                "cosine": "<=>",
                "l2": "<->", 
                "inner_product": "<#>"
            }
            
            if distance_metric not in metric_operators:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")
                
            operator = metric_operators[distance_metric]
            
            # Build the vector similarity query
            # For cosine similarity, we want 1 - distance to get similarity score
            if distance_metric == "cosine":
                select_expr = f"id, (1 - (centroid_embedding {operator} '[{','.join(map(str, embedding_vector))}]')) as similarity"
            else:
                # For other metrics, use negative distance as similarity approximation
                select_expr = f"id, (-(centroid_embedding {operator} '[{','.join(map(str, embedding_vector))}]')) as similarity"
            
            response = self.supabase.table(self.table_groups).select(select_expr).not_(
                "centroid_embedding", "is", None
            ).order("similarity", desc=True).limit(max_results).execute()
            
            results = []
            if response.data:
                for row in response.data:
                    similarity = float(row.get("similarity", 0.0))
                    
                    # Apply threshold filter
                    if similarity >= similarity_threshold:
                        results.append((row["id"], similarity))
                        
            logger.debug(f"Found {len(results)} similar groups above threshold {similarity_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar groups: {e}")
            raise

    async def get_all_group_centroids(self) -> List[GroupCentroid]:
        """Retrieve all group centroids for similarity comparison.
        
        Returns:
            List of GroupCentroid objects with non-null centroids
        """
        try:
            response = self.supabase.table(self.table_groups).select(
                "id, centroid_embedding, member_count, updated_at"
            ).not_("centroid_embedding", "is", None).execute()
            
            centroids = []
            if response.data:
                for row in response.data:
                    try:
                        from ..models import GroupCentroid
                        
                        centroid = GroupCentroid(
                            group_id=row["id"],
                            centroid_vector=list(row["centroid_embedding"]),
                            member_count=int(row["member_count"]),
                            last_updated=self._parse_datetime(row.get("updated_at"))
                        )
                        centroids.append(centroid)
                    except Exception as e:
                        logger.error(f"Failed to parse centroid from DB row: {e}")
                        continue
                        
            logger.debug(f"Retrieved {len(centroids)} group centroids")
            return centroids
            
        except Exception as e:
            logger.error(f"Failed to retrieve group centroids: {e}")
            raise

    async def search_groups_by_status(
        self, 
        status: GroupStatus, 
        limit: Optional[int] = None
    ) -> List[StoryGroup]:
        """Search groups by status.
        
        Args:
            status: The GroupStatus to filter by
            limit: Maximum number of results (optional)
            
        Returns:
            List of StoryGroup objects matching the status
        """
        try:
            query = self.supabase.table(self.table_groups).select("*").eq(
                "status", status.value
            ).order("updated_at", desc=True)
            
            if limit:
                query = query.limit(limit)
                
            response = query.execute()
            
            groups = []
            if response.data:
                for row in response.data:
                    try:
                        groups.append(StoryGroup.from_db(row))
                    except Exception as e:
                        logger.error(f"Failed to parse group from DB row: {e}")
                        continue
                        
            logger.debug(f"Found {len(groups)} groups with status {status.value}")
            return groups
            
        except Exception as e:
            logger.error(f"Failed to search groups by status {status.value}: {e}")
            raise

    # ===== Task 7.3: Transaction Management and Error Handling =====

    async def create_group_with_member(
        self, 
        group: StoryGroup, 
        initial_member: StoryGroupMember
    ) -> Optional[str]:
        """Create a new group with its first member in a transaction.
        
        Args:
            group: The StoryGroup to create
            initial_member: The initial member to add
            
        Returns:
            Group ID if successful, None otherwise
            
        Raises:
            ValueError: If validation fails
            Exception: For transaction failures
        """
        try:
            # Validate inputs
            group.validate()
            initial_member.validate()
            
            # Note: Supabase doesn't support explicit transactions in the Python client
            # We'll implement compensating actions for rollback
            
            # Step 1: Create the group
            group_data = group.to_db()
            group_data.pop("id", None)
            
            group_response = self.supabase.table(self.table_groups).insert(group_data).execute()
            
            if not group_response.data or len(group_response.data) == 0:
                logger.error("Failed to create group in transaction")
                return None
                
            group_id = group_response.data[0]["id"]
            logger.debug(f"Created group {group_id} in transaction")
            
            try:
                # Step 2: Add the initial member with the created group ID
                member_data = initial_member.to_db()
                member_data["group_id"] = group_id  # Ensure correct group ID
                member_data.pop("id", None)
                
                member_response = self.supabase.table(self.table_members).insert(member_data).execute()
                
                if not member_response.data:
                    # Rollback: delete the group
                    logger.error(f"Failed to add initial member, rolling back group {group_id}")
                    await self._rollback_group_creation(group_id)
                    return None
                    
                logger.info(f"Successfully created group {group_id} with initial member")
                return group_id
                
            except Exception as member_error:
                # Rollback: delete the group
                logger.error(f"Member creation failed, rolling back group {group_id}: {member_error}")
                await self._rollback_group_creation(group_id)
                raise
                
        except Exception as e:
            logger.error(f"Failed to create group with member: {e}")
            raise

    async def update_group_centroid_with_members(
        self, 
        group_id: str, 
        new_centroid: List[float],
        member_count: int
    ) -> bool:
        """Update group centroid and member count atomically.
        
        Args:
            group_id: The group ID to update
            new_centroid: The new centroid embedding
            member_count: The new member count
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from datetime import datetime, timezone
            
            update_data = {
                "centroid_embedding": new_centroid,
                "member_count": member_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": GroupStatus.UPDATED.value  # Mark as updated when centroid changes
            }
            
            response = self.supabase.table(self.table_groups).update(update_data).eq(
                "id", group_id
            ).execute()
            
            if response.data:
                logger.debug(f"Successfully updated centroid for group {group_id}")
                return True
            else:
                logger.error(f"No data returned when updating centroid for group {group_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update group centroid {group_id}: {e}")
            raise

    async def batch_update_group_status(
        self, 
        group_ids: List[str], 
        status: GroupStatus
    ) -> Tuple[int, int]:
        """Update status for multiple groups efficiently.
        
        Args:
            group_ids: List of group IDs to update
            status: New status to set
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not group_ids:
            return 0, 0
            
        successful_count = 0
        failed_count = 0
        
        from datetime import datetime, timezone
        update_data = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Update in batches to avoid large payloads
        batch_size = 100
        for i in range(0, len(group_ids), batch_size):
            batch = group_ids[i:i + batch_size]
            
            try:
                response = self.supabase.table(self.table_groups).update(update_data).in_(
                    "id", batch
                ).execute()
                
                if response.data:
                    successful_count += len(response.data)
                else:
                    failed_count += len(batch)
                    
            except Exception as e:
                logger.error(f"Batch update failed for {len(batch)} groups: {e}")
                failed_count += len(batch)
                
        logger.info(f"Batch status update: {successful_count} successful, {failed_count} failed")
        return successful_count, failed_count

    # ===== Private Helper Methods =====

    async def _rollback_group_creation(self, group_id: str) -> None:
        """Rollback helper to delete a created group.
        
        Args:
            group_id: The group ID to delete
        """
        try:
            self.supabase.table(self.table_groups).delete().eq("id", group_id).execute()
            logger.debug(f"Rolled back group creation: {group_id}")
        except Exception as rollback_error:
            logger.error(f"Failed to rollback group {group_id}: {rollback_error}")

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse a datetime value from the database.
        
        Args:
            value: The datetime value (string or datetime object)
            
        Returns:
            Parsed datetime or None
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    # ===== Statistics and Monitoring =====

    async def get_group_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about groups and memberships.
        
        Returns:
            Dictionary with group statistics
        """
        try:
            # Total group count
            groups_response = self.supabase.table(self.table_groups).select("id", count="exact").execute()
            total_groups = groups_response.count if hasattr(groups_response, 'count') else 0
            
            # Status distribution
            status_response = self.supabase.table(self.table_groups).select("status").execute()
            status_counts = {}
            if status_response.data:
                for row in status_response.data:
                    status = row.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
            
            # Member count distribution
            member_response = self.supabase.table(self.table_groups).select("member_count").execute()
            member_counts = []
            if member_response.data:
                member_counts = [row.get("member_count", 0) for row in member_response.data]
            
            # Total memberships
            memberships_response = self.supabase.table(self.table_members).select("id", count="exact").execute()
            total_memberships = memberships_response.count if hasattr(memberships_response, 'count') else 0
            
            stats = {
                "total_groups": total_groups,
                "total_memberships": total_memberships,
                "status_distribution": status_counts,
                "avg_group_size": sum(member_counts) / len(member_counts) if member_counts else 0,
                "max_group_size": max(member_counts) if member_counts else 0,
                "min_group_size": min(member_counts) if member_counts else 0,
            }
            
            logger.debug(f"Retrieved group statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to retrieve group statistics: {e}")
            raise