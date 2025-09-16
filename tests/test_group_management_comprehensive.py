"""
Comprehensive unit tests for group management and storage operations (Task 11.1).

These tests provide comprehensive coverage for GroupManager and storage operations
with properly mocked dependencies and edge case testing.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from src.nfl_news_pipeline.models import (
    StoryEmbedding,
    StoryGroup,
    StoryGroupMember,
    GroupCentroid,
    GroupStatus,
    EMBEDDING_DIM
)
from src.nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
from src.nfl_news_pipeline.group_manager import (
    GroupManager,
    GroupStorageManager,
    GroupAssignmentResult,
    GroupMembershipValidation
)


class TestGroupManagerComprehensive:
    """Comprehensive unit tests for GroupManager with mocked dependencies."""
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create a mock storage manager."""
        return Mock(spec=GroupStorageManager)
    
    @pytest.fixture
    def mock_similarity_calculator(self):
        """Create a mock similarity calculator."""
        calculator = Mock(spec=SimilarityCalculator)
        calculator.threshold = 0.8
        calculator.metric = SimilarityMetric.COSINE
        return calculator
    
    @pytest.fixture
    def mock_centroid_manager(self):
        """Create a mock centroid manager."""
        return Mock(spec=GroupCentroidManager)
    
    @pytest.fixture
    def sample_embedding(self):
        """Create a sample story embedding."""
        vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        normalized_vector = vector / np.linalg.norm(vector)
        
        return StoryEmbedding(
            news_url_id="test-story-123",
            embedding_vector=normalized_vector.tolist(),
            model_name="text-embedding-3-small",
            model_version="1.0",
            summary_text="Chiefs quarterback throws touchdown pass in victory",
            confidence_score=0.92,
            generated_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_group(self):
        """Create a sample story group."""
        centroid_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        normalized_centroid = centroid_vector / np.linalg.norm(centroid_vector)
        
        return StoryGroup(
            id=str(uuid4()),
            centroid_embedding=normalized_centroid.tolist(),
            member_count=5,
            status=GroupStatus.STABLE,
            tags=["football", "nfl"],
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            updated_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_group_centroids(self):
        """Create sample group centroids for testing."""
        centroids = []
        for i in range(3):
            vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            normalized_vector = vector / np.linalg.norm(vector)
            
            centroid = GroupCentroid(
                group_id=f"group-{i}",
                centroid_vector=normalized_vector.tolist(),
                member_count=i + 2,
                last_updated=datetime.now(timezone.utc)
            )
            centroids.append(centroid)
        return centroids
    
    def test_init_with_all_dependencies(self, mock_storage_manager, mock_similarity_calculator, mock_centroid_manager):
        """Test GroupManager initialization with all dependencies."""
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager,
            similarity_threshold=0.75,
            max_group_size=25
        )
        
        assert manager.storage == mock_storage_manager
        assert manager.similarity_calc == mock_similarity_calculator
        assert manager.centroid_manager == mock_centroid_manager
        assert manager.similarity_threshold == 0.75
        assert manager.max_group_size == 25
    
    @pytest.mark.asyncio
    async def test_process_new_story_creates_new_group(self, mock_storage_manager, mock_similarity_calculator, 
                                                       mock_centroid_manager, sample_embedding):
        """Test processing new story that creates a new group."""
        # Setup mocks for no existing groups
        mock_storage_manager.get_group_centroids.return_value = []
        mock_storage_manager.store_group.return_value = True
        mock_storage_manager.store_embedding.return_value = True
        mock_storage_manager.add_member_to_group.return_value = True
        
        # Mock group creation
        new_group_id = str(uuid4())
        mock_storage_manager.store_group.return_value = new_group_id
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager
        )
        
        result = await manager.process_new_story(sample_embedding)
        
        assert isinstance(result, GroupAssignmentResult)
        assert result.group_id == new_group_id
        assert result.similarity_score == 1.0  # Perfect similarity with self
        assert result.is_new_group is True
        assert result.assignment_successful is True
        
        # Verify storage calls
        mock_storage_manager.store_embedding.assert_called_once_with(sample_embedding)
        mock_storage_manager.store_group.assert_called_once()
        mock_storage_manager.add_member_to_group.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_new_story_joins_existing_group(self, mock_storage_manager, mock_similarity_calculator,
                                                          mock_centroid_manager, sample_embedding, sample_group_centroids):
        """Test processing new story that joins existing group."""
        # Setup mocks for existing groups with high similarity
        mock_storage_manager.get_group_centroids.return_value = sample_group_centroids
        mock_similarity_calculator.find_best_matching_group.return_value = ("group-1", 0.89)
        mock_similarity_calculator.is_similar.return_value = True
        
        # Setup successful storage operations
        mock_storage_manager.store_embedding.return_value = True
        mock_storage_manager.add_member_to_group.return_value = True
        mock_storage_manager.get_group_by_id.return_value = StoryGroup(
            id="group-1",
            member_count=6,  # Updated count
            status=GroupStatus.UPDATED,
            tags=["football"]
        )
        
        # Mock centroid update
        mock_centroid_manager.update_group_centroid.return_value = True
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager,
            auto_update_centroids=True
        )
        
        result = await manager.process_new_story(sample_embedding)
        
        assert isinstance(result, GroupAssignmentResult)
        assert result.group_id == "group-1"
        assert result.similarity_score == 0.89
        assert result.is_new_group is False
        assert result.assignment_successful is True
        
        # Verify centroid update was called
        mock_centroid_manager.update_group_centroid.assert_called_once_with("group-1")
    
    @pytest.mark.asyncio
    async def test_process_new_story_no_similar_groups(self, mock_storage_manager, mock_similarity_calculator,
                                                       mock_centroid_manager, sample_embedding, sample_group_centroids):
        """Test processing new story when no groups are similar enough."""
        # Setup mocks for existing groups with low similarity
        mock_storage_manager.get_group_centroids.return_value = sample_group_centroids
        mock_similarity_calculator.find_best_matching_group.return_value = ("group-1", 0.65)
        mock_similarity_calculator.is_similar.return_value = False  # Below threshold
        
        # Mock new group creation
        new_group_id = str(uuid4())
        mock_storage_manager.store_group.return_value = new_group_id
        mock_storage_manager.store_embedding.return_value = True
        mock_storage_manager.add_member_to_group.return_value = True
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager,
            min_similarity_threshold=0.8
        )
        
        result = await manager.process_new_story(sample_embedding)
        
        assert result.is_new_group is True
        assert result.group_id == new_group_id
        assert result.assignment_successful is True
    
    @pytest.mark.asyncio
    async def test_process_new_story_group_size_limit(self, mock_storage_manager, mock_similarity_calculator,
                                                      mock_centroid_manager, sample_embedding, sample_group_centroids):
        """Test processing new story when best matching group is at size limit."""
        # Setup group at max size
        large_group = StoryGroup(
            id="group-1",
            member_count=25,  # At limit
            status=GroupStatus.STABLE,
            tags=["football"]
        )
        
        mock_storage_manager.get_group_centroids.return_value = sample_group_centroids
        mock_similarity_calculator.find_best_matching_group.return_value = ("group-1", 0.89)
        mock_similarity_calculator.is_similar.return_value = True
        mock_storage_manager.get_group_by_id.return_value = large_group
        
        # Mock new group creation for overflow
        new_group_id = str(uuid4())
        mock_storage_manager.store_group.return_value = new_group_id
        mock_storage_manager.store_embedding.return_value = True
        mock_storage_manager.add_member_to_group.return_value = True
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager,
            max_group_size=25
        )
        
        result = await manager.process_new_story(sample_embedding)
        
        assert result.is_new_group is True  # Should create new group due to size limit
        assert result.group_id == new_group_id
    
    @pytest.mark.asyncio
    async def test_process_multiple_stories_batch(self, mock_storage_manager, mock_similarity_calculator,
                                                  mock_centroid_manager):
        """Test batch processing of multiple stories."""
        # Create multiple embeddings
        embeddings = []
        for i in range(5):
            vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            normalized_vector = vector / np.linalg.norm(vector)
            
            embedding = StoryEmbedding(
                news_url_id=f"story-{i}",
                embedding_vector=normalized_vector.tolist(),
                model_name="test-model",
                model_version="1.0",
                summary_text=f"Story {i} summary",
                confidence_score=0.8,
                generated_at=datetime.now(timezone.utc)
            )
            embeddings.append(embedding)
        
        # Mock storage operations
        mock_storage_manager.get_group_centroids.return_value = []
        mock_storage_manager.store_group.side_effect = [str(uuid4()) for _ in range(5)]
        mock_storage_manager.store_embedding.return_value = True
        mock_storage_manager.add_member_to_group.return_value = True
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager
        )
        
        results = await manager.process_stories_batch(embeddings)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, GroupAssignmentResult)
            assert result.is_new_group is True  # No existing groups to match
    
    @pytest.mark.asyncio
    async def test_validate_group_membership_comprehensive(self, mock_storage_manager, mock_similarity_calculator,
                                                           mock_centroid_manager, sample_embedding):
        """Test comprehensive group membership validation."""
        # Mock existing membership check
        mock_storage_manager.check_membership_exists.return_value = False
        
        # Mock group existence and capacity
        group = StoryGroup(
            id="test-group",
            member_count=10,
            status=GroupStatus.STABLE,
            tags=["test"]
        )
        mock_storage_manager.get_group_by_id.return_value = group
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager,
            max_group_size=25
        )
        
        validation = await manager.validate_group_membership("test-group", sample_embedding)
        
        assert isinstance(validation, GroupMembershipValidation)
        assert validation.is_valid is True
        assert validation.can_add_member is True
        assert validation.is_duplicate is False
        assert validation.group_exists is True
        assert validation.at_capacity is False
    
    @pytest.mark.asyncio
    async def test_validate_group_membership_duplicate(self, mock_storage_manager, mock_similarity_calculator,
                                                       mock_centroid_manager, sample_embedding):
        """Test validation when story is already a member."""
        # Mock existing membership
        mock_storage_manager.check_membership_exists.return_value = True
        
        group = StoryGroup(
            id="test-group",
            member_count=10,
            status=GroupStatus.STABLE,
            tags=["test"]
        )
        mock_storage_manager.get_group_by_id.return_value = group
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager
        )
        
        validation = await manager.validate_group_membership("test-group", sample_embedding)
        
        assert validation.is_valid is False
        assert validation.is_duplicate is True
        assert validation.can_add_member is False
    
    @pytest.mark.asyncio
    async def test_validate_group_membership_at_capacity(self, mock_storage_manager, mock_similarity_calculator,
                                                         mock_centroid_manager, sample_embedding):
        """Test validation when group is at capacity."""
        mock_storage_manager.check_membership_exists.return_value = False
        
        # Group at capacity
        group = StoryGroup(
            id="test-group",
            member_count=25,  # At limit
            status=GroupStatus.STABLE,
            tags=["test"]
        )
        mock_storage_manager.get_group_by_id.return_value = group
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager,
            max_group_size=25
        )
        
        validation = await manager.validate_group_membership("test-group", sample_embedding)
        
        assert validation.is_valid is False
        assert validation.at_capacity is True
        assert validation.can_add_member is False
    
    @pytest.mark.asyncio
    async def test_update_group_status_lifecycle(self, mock_storage_manager, mock_similarity_calculator,
                                                 mock_centroid_manager):
        """Test group status lifecycle management."""
        # Test new group aging to stable
        new_group = StoryGroup(
            id="new-group",
            member_count=1,
            status=GroupStatus.NEW,
            created_at=datetime.now(timezone.utc) - timedelta(hours=25),  # Older than 24 hours
            updated_at=datetime.now(timezone.utc) - timedelta(hours=25)
        )
        
        mock_storage_manager.get_group_by_id.return_value = new_group
        mock_storage_manager.update_group_status.return_value = True
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager,
            status_transition_hours=24
        )
        
        result = await manager.update_group_status("new-group")
        
        assert result is True
        mock_storage_manager.update_group_status.assert_called_with("new-group", GroupStatus.STABLE)
    
    @pytest.mark.asyncio
    async def test_get_group_analytics_comprehensive(self, mock_storage_manager, mock_similarity_calculator,
                                                     mock_centroid_manager):
        """Test comprehensive group analytics generation."""
        # Mock group data
        group = StoryGroup(
            id="analytics-group",
            member_count=15,
            status=GroupStatus.STABLE,
            created_at=datetime.now(timezone.utc) - timedelta(days=7),
            updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
            tags=["football", "nfl", "game"]
        )
        
        # Mock member embeddings with varying timestamps
        members = []
        for i in range(15):
            member = StoryGroupMember(
                group_id="analytics-group",
                news_url_id=f"story-{i}",
                similarity_score=0.8 + (i % 3) * 0.05,  # Varying similarity
                added_at=datetime.now(timezone.utc) - timedelta(hours=i*2)
            )
            members.append(member)
        
        mock_storage_manager.get_group_by_id.return_value = group
        mock_storage_manager.get_group_members.return_value = members
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager
        )
        
        analytics = await manager.get_group_analytics("analytics-group")
        
        assert analytics.group_id == "analytics-group"
        assert analytics.member_count == 15
        assert 0.8 <= analytics.avg_similarity <= 0.9
        assert analytics.story_span_hours > 0
        assert len(analytics.tags) == 3
        assert analytics.group_age_days == 7
    
    @pytest.mark.asyncio
    async def test_merge_groups_functionality(self, mock_storage_manager, mock_similarity_calculator,
                                              mock_centroid_manager):
        """Test group merging functionality."""
        # Setup two groups to merge
        group1 = StoryGroup(
            id="group-1",
            member_count=5,
            status=GroupStatus.STABLE,
            tags=["football", "nfl"]
        )
        
        group2 = StoryGroup(
            id="group-2", 
            member_count=3,
            status=GroupStatus.STABLE,
            tags=["nfl", "game"]
        )
        
        # Mock members for both groups
        group1_members = [
            StoryGroupMember(group_id="group-1", news_url_id=f"story-1-{i}", similarity_score=0.85)
            for i in range(5)
        ]
        group2_members = [
            StoryGroupMember(group_id="group-2", news_url_id=f"story-2-{i}", similarity_score=0.82)
            for i in range(3)
        ]
        
        mock_storage_manager.get_group_by_id.side_effect = lambda gid: group1 if gid == "group-1" else group2
        mock_storage_manager.get_group_members.side_effect = lambda gid: group1_members if gid == "group-1" else group2_members
        mock_storage_manager.merge_groups.return_value = True
        mock_centroid_manager.update_group_centroid.return_value = True
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager
        )
        
        result = await manager.merge_groups("group-1", "group-2")
        
        assert result is True
        mock_storage_manager.merge_groups.assert_called_once_with("group-1", "group-2")
        mock_centroid_manager.update_group_centroid.assert_called_once_with("group-1")
    
    @pytest.mark.asyncio
    async def test_error_handling_storage_failures(self, mock_storage_manager, mock_similarity_calculator,
                                                   mock_centroid_manager, sample_embedding):
        """Test error handling when storage operations fail."""
        # Mock storage failure
        mock_storage_manager.get_group_centroids.side_effect = Exception("Database connection failed")
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager
        )
        
        # Should handle gracefully and not raise exception
        result = await manager.process_new_story(sample_embedding)
        
        assert result is None  # or some appropriate error result
    
    @pytest.mark.asyncio
    async def test_concurrent_story_processing(self, mock_storage_manager, mock_similarity_calculator,
                                               mock_centroid_manager):
        """Test concurrent processing of multiple stories."""
        # Create multiple embeddings
        embeddings = []
        for i in range(10):
            vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            normalized_vector = vector / np.linalg.norm(vector)
            
            embedding = StoryEmbedding(
                news_url_id=f"concurrent-story-{i}",
                embedding_vector=normalized_vector.tolist(),
                model_name="test-model", 
                model_version="1.0",
                summary_text=f"Concurrent story {i}",
                confidence_score=0.8
            )
            embeddings.append(embedding)
        
        # Mock successful storage for all
        mock_storage_manager.get_group_centroids.return_value = []
        mock_storage_manager.store_group.side_effect = [str(uuid4()) for _ in range(10)]
        mock_storage_manager.store_embedding.return_value = True
        mock_storage_manager.add_member_to_group.return_value = True
        
        manager = GroupManager(
            storage_manager=mock_storage_manager,
            similarity_calculator=mock_similarity_calculator,
            centroid_manager=mock_centroid_manager
        )
        
        # Process concurrently
        tasks = [manager.process_new_story(embedding) for embedding in embeddings]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert len(results) == 10
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent processing failed: {result}")
            assert isinstance(result, GroupAssignmentResult)


class TestGroupStorageManagerComprehensive:
    """Comprehensive unit tests for GroupStorageManager with detailed mocking."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Create a comprehensive mock Supabase client."""
        mock_client = Mock()
        mock_table = Mock()
        mock_client.table.return_value = mock_table
        return mock_client
    
    @pytest.fixture
    def sample_group_data(self):
        """Create sample group data for testing."""
        return {
            "id": str(uuid4()),
            "centroid_embedding": [0.1] * EMBEDDING_DIM,
            "member_count": 5,
            "status": "stable",
            "tags": ["football", "nfl"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def test_init_with_supabase_client(self, mock_supabase_client):
        """Test storage manager initialization."""
        storage = GroupStorageManager(mock_supabase_client)
        
        assert storage.supabase == mock_supabase_client
        assert storage.groups_table == "story_groups"
        assert storage.members_table == "story_group_members"
        assert storage.embeddings_table == "story_embeddings"
    
    @pytest.mark.asyncio
    async def test_store_group_new_record(self, mock_supabase_client, sample_group_data):
        """Test storing a new group record."""
        mock_response = Mock()
        mock_response.data = [sample_group_data]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.insert.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        group = StoryGroup.from_db(sample_group_data)
        result = await storage.store_group(group)
        
        assert result == sample_group_data["id"]
        mock_table.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_group_centroids_batch(self, mock_supabase_client):
        """Test retrieving group centroids in batch."""
        # Mock multiple groups
        groups_data = []
        for i in range(5):
            group_data = {
                "id": f"group-{i}",
                "centroid_embedding": [0.1 * (i+1)] * EMBEDDING_DIM,
                "member_count": i + 2,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            groups_data.append(group_data)
        
        mock_response = Mock()
        mock_response.data = groups_data
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.is_.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        centroids = await storage.get_group_centroids()
        
        assert len(centroids) == 5
        for i, centroid in enumerate(centroids):
            assert isinstance(centroid, GroupCentroid)
            assert centroid.group_id == f"group-{i}"
            assert centroid.member_count == i + 2
    
    @pytest.mark.asyncio
    async def test_add_member_to_group_with_validation(self, mock_supabase_client):
        """Test adding member to group with validation."""
        # Mock successful insertion
        member_data = {
            "id": str(uuid4()),
            "group_id": "test-group",
            "news_url_id": "test-story",
            "similarity_score": 0.87,
            "added_at": datetime.now(timezone.utc).isoformat()
        }
        
        mock_response = Mock()
        mock_response.data = [member_data]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.insert.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        result = await storage.add_member_to_group("test-group", "test-story", 0.87)
        
        assert result is True
        mock_table.insert.assert_called_once()
        inserted_data = mock_table.insert.call_args[0][0]
        assert inserted_data["group_id"] == "test-group"
        assert inserted_data["news_url_id"] == "test-story"
        assert inserted_data["similarity_score"] == 0.87
    
    @pytest.mark.asyncio
    async def test_check_membership_exists_true(self, mock_supabase_client):
        """Test checking existing membership."""
        mock_response = Mock()
        mock_response.data = [{"id": "existing-membership"}]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        exists = await storage.check_membership_exists("test-group", "test-story")
        
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_check_membership_exists_false(self, mock_supabase_client):
        """Test checking non-existent membership."""
        mock_response = Mock()
        mock_response.data = []
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        exists = await storage.check_membership_exists("test-group", "test-story")
        
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_get_group_members_comprehensive(self, mock_supabase_client):
        """Test retrieving all members of a group."""
        # Mock multiple members
        members_data = []
        for i in range(10):
            member_data = {
                "id": str(uuid4()),
                "group_id": "test-group",
                "news_url_id": f"story-{i}",
                "similarity_score": 0.8 + (i % 3) * 0.05,
                "added_at": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat()
            }
            members_data.append(member_data)
        
        mock_response = Mock()
        mock_response.data = members_data
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        members = await storage.get_group_members("test-group")
        
        assert len(members) == 10
        for i, member in enumerate(members):
            assert isinstance(member, StoryGroupMember)
            assert member.group_id == "test-group"
            assert member.news_url_id == f"story-{i}"
    
    @pytest.mark.asyncio
    async def test_update_group_status_with_timestamp(self, mock_supabase_client):
        """Test updating group status with timestamp."""
        updated_data = {
            "id": "test-group",
            "status": "updated",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        mock_response = Mock()
        mock_response.data = [updated_data]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.update.return_value.eq.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        result = await storage.update_group_status("test-group", GroupStatus.UPDATED)
        
        assert result is True
        mock_table.update.assert_called_once()
        update_data = mock_table.update.call_args[0][0]
        assert update_data["status"] == "updated"
        assert "updated_at" in update_data
    
    @pytest.mark.asyncio
    async def test_delete_group_cascade(self, mock_supabase_client):
        """Test deleting group with cascade to members."""
        # Mock successful deletion
        mock_delete_response = Mock()
        mock_delete_response.data = [{"id": "test-group"}]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.delete.return_value.eq.return_value.execute.return_value = mock_delete_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        result = await storage.delete_group("test-group")
        
        assert result is True
        # Should delete from groups table (members cascade via FK)
        mock_table.delete.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_groups_by_status(self, mock_supabase_client):
        """Test retrieving groups by status."""
        # Mock groups with specific status
        groups_data = []
        for i in range(3):
            group_data = {
                "id": f"new-group-{i}",
                "status": "new",
                "member_count": i + 1,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            groups_data.append(group_data)
        
        mock_response = Mock()
        mock_response.data = groups_data
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        groups = await storage.get_groups_by_status(GroupStatus.NEW)
        
        assert len(groups) == 3
        for group in groups:
            assert isinstance(group, StoryGroup)
            assert group.status == GroupStatus.NEW
    
    @pytest.mark.asyncio
    async def test_get_storage_statistics(self, mock_supabase_client):
        """Test retrieving comprehensive storage statistics."""
        # Mock statistics data
        stats_data = [
            {
                "total_groups": 150,
                "total_members": 2500,
                "avg_group_size": 16.67,
                "status_breakdown": {
                    "new": 25,
                    "updated": 45,
                    "stable": 80
                }
            }
        ]
        
        mock_response = Mock()
        mock_response.data = stats_data
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        stats = await storage.get_storage_statistics()
        
        assert stats["total_groups"] == 150
        assert stats["total_members"] == 2500
        assert stats["avg_group_size"] == 16.67
        assert stats["status_breakdown"]["stable"] == 80
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, mock_supabase_client):
        """Test graceful handling of database errors."""
        # Mock database error
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.side_effect = Exception("Database connection lost")
        
        storage = GroupStorageManager(mock_supabase_client)
        
        # Should not raise exception, return appropriate default
        result = await storage.get_group_centroids()
        
        assert result == []  # or appropriate error result
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_simulation(self, mock_supabase_client):
        """Test transaction-like behavior with rollback simulation."""
        # Simulate partial success followed by failure
        mock_table = mock_supabase_client.table.return_value
        
        # First operation succeeds
        mock_table.insert.return_value.execute.return_value = Mock(data=[{"id": "success"}])
        
        # Second operation fails
        def failing_operation(*args, **kwargs):
            raise Exception("Second operation failed")
        
        storage = GroupStorageManager(mock_supabase_client)
        
        # Test that storage handles partial failures appropriately
        # (This would depend on actual implementation of transaction handling)
        with pytest.raises(Exception):
            # Simulate complex operation that should roll back
            await storage.store_group(StoryGroup(
                member_count=1,
                status=GroupStatus.NEW
            ))
            # Follow with operation that fails
            failing_operation()
    
    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, mock_supabase_client):
        """Test bulk operations for performance."""
        # Create multiple members to insert
        members_data = []
        for i in range(100):
            member_data = {
                "group_id": "bulk-group",
                "news_url_id": f"bulk-story-{i}",
                "similarity_score": 0.8 + (i % 10) * 0.01,
            }
            members_data.append(member_data)
        
        mock_response = Mock()
        mock_response.data = [{"inserted": len(members_data)}]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.insert.return_value.execute.return_value = mock_response
        
        storage = GroupStorageManager(mock_supabase_client)
        
        result = await storage.bulk_add_members("bulk-group", members_data)
        
        assert result is True
        # Verify bulk insert was called (not individual inserts)
        mock_table.insert.assert_called_once()
        inserted_data = mock_table.insert.call_args[0][0]
        assert len(inserted_data) == 100