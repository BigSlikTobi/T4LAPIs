"""Test cases for GroupStorageManager functionality.

Tests cover Tasks 7.1, 7.2, and 7.3:
- CRUD operations for groups and memberships
- Vector similarity search capabilities
- Transaction management and error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from datetime import datetime, timezone
from typing import List, Any, Dict

from src.nfl_news_pipeline.storage.group_manager import GroupStorageManager
from src.nfl_news_pipeline.models import (
    StoryGroup, StoryGroupMember, GroupCentroid, GroupStatus, EMBEDDING_DIM
)


class TestGroupStorageManager:
    """Test suite for GroupStorageManager implementation."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Create a mock Supabase client."""
        client = Mock()
        client.table = Mock()
        return client

    @pytest.fixture
    def group_storage_manager(self, mock_supabase_client):
        """Create GroupStorageManager instance with mocked client."""
        return GroupStorageManager(mock_supabase_client)

    @pytest.fixture
    def sample_group(self):
        """Create a sample StoryGroup for testing."""
        return StoryGroup(
            id="test-group-id",
            member_count=3,
            status=GroupStatus.NEW,
            tags=["breaking", "trade"],
            centroid_embedding=[0.1] * EMBEDDING_DIM,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_member(self):
        """Create a sample StoryGroupMember for testing."""
        return StoryGroupMember(
            id="test-member-id",
            group_id="test-group-id",
            news_url_id="test-news-id",
            similarity_score=0.85,
            added_at=datetime.now(timezone.utc)
        )

    # ===== Task 7.1: CRUD Operations Tests =====

    @pytest.mark.asyncio
    async def test_create_group_success(self, group_storage_manager, sample_group, mock_supabase_client):
        """Test successful group creation."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[{"id": "new-group-id"}])

        # Execute
        result = await group_storage_manager.create_group(sample_group)

        # Verify
        assert result == "new-group-id"
        mock_supabase_client.table.assert_called_with("story_groups")
        mock_table.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_group_validation_error(self, group_storage_manager):
        """Test group creation with validation error."""
        # Create invalid group (negative member count)
        invalid_group = StoryGroup(
            member_count=-1,
            status=GroupStatus.NEW
        )

        # Execute and verify exception
        with pytest.raises(ValueError):
            await group_storage_manager.create_group(invalid_group)

    @pytest.mark.asyncio
    async def test_get_group_success(self, group_storage_manager, mock_supabase_client):
        """Test successful group retrieval."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[{
            "id": "test-group-id",
            "member_count": 3,
            "status": "new",
            "tags": ["test"],
            "centroid_embedding": [0.1] * EMBEDDING_DIM,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }])

        # Execute
        result = await group_storage_manager.get_group("test-group-id")

        # Verify
        assert result is not None
        assert result.id == "test-group-id"
        assert result.member_count == 3
        assert result.status == GroupStatus.NEW

    @pytest.mark.asyncio
    async def test_get_group_not_found(self, group_storage_manager, mock_supabase_client):
        """Test group retrieval when group doesn't exist."""
        # Setup mock response with no data
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[])

        # Execute
        result = await group_storage_manager.get_group("nonexistent-id")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_update_group_success(self, group_storage_manager, sample_group, mock_supabase_client):
        """Test successful group update."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[{"id": "test-group-id"}])

        # Execute
        result = await group_storage_manager.update_group(sample_group)

        # Verify
        assert result is True
        mock_table.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_member_to_group_success(self, group_storage_manager, sample_member, mock_supabase_client):
        """Test successful member addition."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[{"id": "new-member-id"}])

        # Execute
        result = await group_storage_manager.add_member_to_group(sample_member)

        # Verify
        assert result is True
        mock_supabase_client.table.assert_called_with("story_group_members")

    @pytest.mark.asyncio
    async def test_get_group_members_success(self, group_storage_manager, mock_supabase_client):
        """Test successful retrieval of group members."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[
            {
                "id": "member-1",
                "group_id": "test-group-id",
                "news_url_id": "news-1",
                "similarity_score": 0.9,
                "added_at": "2024-01-01T00:00:00Z"
            },
            {
                "id": "member-2",
                "group_id": "test-group-id",
                "news_url_id": "news-2",
                "similarity_score": 0.8,
                "added_at": "2024-01-01T01:00:00Z"
            }
        ])

        # Execute
        result = await group_storage_manager.get_group_members("test-group-id")

        # Verify
        assert len(result) == 2
        assert all(isinstance(member, StoryGroupMember) for member in result)
        assert result[0].group_id == "test-group-id"

    # ===== Task 7.2: Vector Similarity Search Tests =====

    @pytest.mark.asyncio
    async def test_find_similar_groups_cosine(self, group_storage_manager, mock_supabase_client):
        """Test vector similarity search with cosine distance."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.not_.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[
            {"id": "group-1", "similarity": 0.95},
            {"id": "group-2", "similarity": 0.85},
            {"id": "group-3", "similarity": 0.75}  # Below threshold
        ])

        # Execute
        embedding = [0.1] * EMBEDDING_DIM
        result = await group_storage_manager.find_similar_groups(
            embedding, 
            similarity_threshold=0.8,
            max_results=10,
            distance_metric="cosine"
        )

        # Verify
        assert len(result) == 2  # Only 2 above threshold
        assert result[0] == ("group-1", 0.95)
        assert result[1] == ("group-2", 0.85)

    @pytest.mark.asyncio
    async def test_find_similar_groups_invalid_metric(self, group_storage_manager):
        """Test vector similarity search with invalid distance metric."""
        embedding = [0.1] * EMBEDDING_DIM
        
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            await group_storage_manager.find_similar_groups(
                embedding, 
                distance_metric="invalid"
            )

    @pytest.mark.asyncio
    async def test_get_all_group_centroids_success(self, group_storage_manager, mock_supabase_client):
        """Test retrieval of all group centroids."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.not_.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[
            {
                "id": "group-1",
                "centroid_embedding": [0.1] * EMBEDDING_DIM,
                "member_count": 3,
                "updated_at": "2024-01-01T00:00:00Z"
            },
            {
                "id": "group-2",
                "centroid_embedding": [0.2] * EMBEDDING_DIM,
                "member_count": 5,
                "updated_at": "2024-01-01T01:00:00Z"
            }
        ])

        # Execute
        result = await group_storage_manager.get_all_group_centroids()

        # Verify
        assert len(result) == 2
        assert all(isinstance(centroid, GroupCentroid) for centroid in result)
        assert result[0].group_id == "group-1"
        assert len(result[0].centroid_vector) == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_search_groups_by_status(self, group_storage_manager, mock_supabase_client):
        """Test searching groups by status."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.order.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[
            {
                "id": "group-1",
                "member_count": 3,
                "status": "new",
                "tags": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        ])

        # Execute
        result = await group_storage_manager.search_groups_by_status(GroupStatus.NEW, limit=10)

        # Verify
        assert len(result) == 1
        assert result[0].status == GroupStatus.NEW

    # ===== Task 7.3: Transaction Management Tests =====

    @pytest.mark.asyncio
    async def test_create_group_with_member_success(self, group_storage_manager, sample_group, sample_member, mock_supabase_client):
        """Test successful transactional group creation with initial member."""
        # Setup mock responses for both operations
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        
        # Mock group creation response
        mock_table.insert.return_value = mock_table
        
        # Create separate mock objects for different execute calls
        group_execute_mock = Mock()
        group_execute_mock.data = [{"id": "new-group-id"}]
        
        member_execute_mock = Mock()
        member_execute_mock.data = [{"id": "new-member-id"}]
        
        # Configure execute to return different responses based on call order
        mock_table.execute.side_effect = [group_execute_mock, member_execute_mock]

        # Execute
        result = await group_storage_manager.create_group_with_member(sample_group, sample_member)

        # Verify
        assert result == "new-group-id"
        assert mock_table.insert.call_count == 2  # Group + Member

    @pytest.mark.asyncio
    async def test_create_group_with_member_rollback(self, group_storage_manager, sample_group, sample_member, mock_supabase_client):
        """Test rollback when member creation fails."""
        # Setup mock responses
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.delete.return_value = mock_table
        mock_table.eq.return_value = mock_table
        
        # Group creation succeeds, member creation fails
        group_execute_mock = Mock()
        group_execute_mock.data = [{"id": "new-group-id"}]
        
        member_execute_mock = Mock()
        member_execute_mock.data = []  # Empty data = failure
        
        rollback_execute_mock = Mock()
        rollback_execute_mock.data = [{"id": "new-group-id"}]
        
        mock_table.execute.side_effect = [group_execute_mock, member_execute_mock, rollback_execute_mock]

        # Execute
        result = await group_storage_manager.create_group_with_member(sample_group, sample_member)

        # Verify
        assert result is None  # Should return None due to rollback
        assert mock_table.delete.call_count == 1  # Rollback delete called

    @pytest.mark.asyncio
    async def test_update_group_centroid_with_members(self, group_storage_manager, mock_supabase_client):
        """Test atomic centroid and member count update."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[{"id": "test-group-id"}])

        # Execute
        new_centroid = [0.5] * EMBEDDING_DIM
        result = await group_storage_manager.update_group_centroid_with_members(
            "test-group-id", new_centroid, 5
        )

        # Verify
        assert result is True
        mock_table.update.assert_called_once()
        
        # Verify update data contains expected fields
        update_call_args = mock_table.update.call_args[0][0]
        assert update_call_args["centroid_embedding"] == new_centroid
        assert update_call_args["member_count"] == 5
        assert update_call_args["status"] == GroupStatus.UPDATED.value

    @pytest.mark.asyncio
    async def test_batch_update_group_status(self, group_storage_manager, mock_supabase_client):
        """Test batch status update for multiple groups."""
        # Setup mock response
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.in_.return_value = mock_table
        mock_table.execute.return_value = Mock(data=[
            {"id": "group-1"}, {"id": "group-2"}, {"id": "group-3"}
        ])

        # Execute
        group_ids = ["group-1", "group-2", "group-3"]
        successful, failed = await group_storage_manager.batch_update_group_status(
            group_ids, GroupStatus.STABLE
        )

        # Verify
        assert successful == 3
        assert failed == 0

    @pytest.mark.asyncio
    async def test_get_group_statistics(self, group_storage_manager, mock_supabase_client):
        """Test retrieval of comprehensive group statistics."""
        # Setup mock responses for different statistics queries
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        
        # Mock different responses for different select calls
        mock_table.select.return_value = mock_table
        mock_table.execute.side_effect = [
            Mock(count=10, data=None),  # Total groups count
            Mock(data=[{"status": "new"}, {"status": "new"}, {"status": "updated"}]),  # Status distribution
            Mock(data=[{"member_count": 3}, {"member_count": 5}, {"member_count": 2}]),  # Member counts
            Mock(count=15, data=None)  # Total memberships count
        ]

        # Execute
        result = await group_storage_manager.get_group_statistics()

        # Verify
        assert result["total_groups"] == 10
        assert result["total_memberships"] == 15
        assert result["status_distribution"]["new"] == 2
        assert result["status_distribution"]["updated"] == 1
        assert result["avg_group_size"] == pytest.approx(3.33, rel=1e-2)
        assert result["max_group_size"] == 5
        assert result["min_group_size"] == 2

    # ===== Error Handling Tests =====

    @pytest.mark.asyncio
    async def test_database_error_handling(self, group_storage_manager, sample_group, mock_supabase_client):
        """Test proper error handling when database operations fail."""
        # Setup mock to raise exception
        mock_table = Mock()
        mock_supabase_client.table.return_value = mock_table
        mock_table.insert.side_effect = Exception("Database connection error")

        # Execute and verify exception propagation
        with pytest.raises(Exception, match="Database connection error"):
            await group_storage_manager.create_group(sample_group)

    @pytest.mark.asyncio
    async def test_empty_group_ids_batch_update(self, group_storage_manager):
        """Test batch update with empty group IDs list."""
        # Execute
        successful, failed = await group_storage_manager.batch_update_group_status(
            [], GroupStatus.STABLE
        )

        # Verify
        assert successful == 0
        assert failed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])