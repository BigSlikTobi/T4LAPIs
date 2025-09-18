"""
Tests for GroupManager - Tasks 6.1, 6.2, and 6.3 implementation.

Comprehensive test suite covering:
- Task 6.1: Group assignment and story processing
- Task 6.2: Group lifecycle and status management  
- Task 6.3: Group membership operations with validation
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
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


@pytest.fixture
def sample_embedding():
    """Create a sample story embedding for testing."""
    return StoryEmbedding(
        news_url_id="test-story-1",
        embedding_vector=[0.1] * EMBEDDING_DIM,
        model_name="test-model",
        model_version="1.0",
        summary_text="Test story about NFL trade",
        confidence_score=0.9,
        generated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_group():
    """Create a sample story group for testing."""
    return StoryGroup(
        id="test-group-1",
        member_count=2,
        status=GroupStatus.NEW,
        tags=["trade", "quarterback"],
        centroid_embedding=[0.2] * EMBEDDING_DIM,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_centroid():
    """Create a sample group centroid for testing."""
    return GroupCentroid(
        group_id="test-group-1",
        centroid_vector=[0.2] * EMBEDDING_DIM,
        member_count=2,
        last_updated=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client for testing."""
    client = Mock()
    
    # Mock table method to return a mock table object
    mock_table = Mock()
    client.table.return_value = mock_table
    
    # Mock common query methods
    mock_table.select.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.in_.return_value = mock_table
    mock_table.is_.return_value = mock_table
    mock_table.execute.return_value = Mock(data=[])
    
    return client


@pytest.fixture
def storage_manager(mock_supabase_client):
    """Create a GroupStorageManager with mocked client."""
    return GroupStorageManager(mock_supabase_client)


@pytest.fixture
def similarity_calculator():
    """Create a SimilarityCalculator for testing."""
    return SimilarityCalculator(similarity_threshold=0.8, metric=SimilarityMetric.COSINE)


@pytest.fixture
def centroid_manager():
    """Create a GroupCentroidManager for testing."""
    return GroupCentroidManager()


@pytest.fixture
def group_manager(storage_manager, similarity_calculator, centroid_manager):
    """Create a GroupManager with all dependencies."""
    return GroupManager(
        storage_manager=storage_manager,
        similarity_calculator=similarity_calculator,
        centroid_manager=centroid_manager,
        similarity_threshold=0.8,
        max_group_size=50
    )


class TestGroupStorageManager:
    """Test GroupStorageManager database operations."""
    
    @pytest.mark.asyncio
    async def test_store_embedding_new(self, storage_manager, sample_embedding, mock_supabase_client):
        """Test storing a new embedding."""
        # Mock no existing embedding
        mock_supabase_client.table().select().eq().execute.return_value = Mock(data=[])
        
        result = await storage_manager.store_embedding(sample_embedding)
        
        assert result is True
        mock_supabase_client.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_store_embedding_update_existing(self, storage_manager, sample_embedding, mock_supabase_client):
        """Test updating an existing embedding."""
        # Mock existing embedding
        mock_supabase_client.table().select().eq().execute.return_value = Mock(data=[{"id": "existing-id"}])
        
        result = await storage_manager.store_embedding(sample_embedding)
        
        assert result is True
        mock_supabase_client.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_store_group_new(self, storage_manager, sample_group, mock_supabase_client):
        """Test storing a new group."""
        sample_group.id = None  # New group without ID
        mock_supabase_client.table().insert().execute.return_value = Mock(data=[{"id": "new-group-id"}])
        
        result = await storage_manager.store_group(sample_group)
        
        assert result is True
        assert sample_group.id == "new-group-id"
    
    @pytest.mark.asyncio
    async def test_add_member_to_group(self, storage_manager, mock_supabase_client):
        """Test adding a member to a group."""
        result = await storage_manager.add_member_to_group("group-1", "story-1", 0.85)
        
        assert result is True
        mock_supabase_client.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_group_centroids(self, storage_manager, mock_supabase_client):
        """Test retrieving group centroids."""
        # Mock centroid data
        mock_data = [
            {
                "id": "group-1",
                "centroid_embedding": [0.1] * EMBEDDING_DIM,
                "member_count": 3,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(data=mock_data)
        
        centroids = await storage_manager.get_group_centroids()
        
        assert len(centroids) == 1
        assert centroids[0].group_id == "group-1"
        assert centroids[0].member_count == 3
    
    @pytest.mark.asyncio
    async def test_get_group_embeddings(self, storage_manager, mock_supabase_client):
        """Test retrieving embeddings for a group."""
        # Mock member data
        members_data = [{"news_url_id": "story-1"}, {"news_url_id": "story-2"}]
        
        # Mock embedding data
        embeddings_data = [
            {
                "id": "emb-1",
                "news_url_id": "story-1",
                "embedding_vector": [0.1] * EMBEDDING_DIM,
                "model_name": "test-model",
                "model_version": "1.0",
                "summary_text": "Test summary",
                "confidence_score": 0.9
            }
        ]
        
        # Setup mock responses
        mock_supabase_client.table().select().eq().execute.side_effect = [
            Mock(data=members_data),  # Members query
            Mock(data=embeddings_data)  # Embeddings query
        ]
        
        embeddings = await storage_manager.get_group_embeddings("group-1")
        
        assert len(embeddings) == 1
        assert embeddings[0].news_url_id == "story-1"
    
    @pytest.mark.asyncio
    async def test_check_membership_exists_true(self, storage_manager, mock_supabase_client):
        """Test checking membership when it exists."""
        mock_supabase_client.table().select().eq().eq().execute.return_value = Mock(data=[{"id": "member-1"}])
        
        exists = await storage_manager.check_membership_exists("group-1", "story-1")
        
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_check_membership_exists_false(self, storage_manager, mock_supabase_client):
        """Test checking membership when it doesn't exist."""
        mock_supabase_client.table().select().eq().eq().execute.return_value = Mock(data=[])
        
        exists = await storage_manager.check_membership_exists("group-1", "story-1")
        
        assert exists is False


class TestGroupManager:
    """Test GroupManager core functionality for Tasks 6.1, 6.2, and 6.3."""
    
    # Task 6.1 Tests: Group assignment and story processing
    
    @pytest.mark.asyncio
    async def test_process_new_story_creates_new_group(self, group_manager, sample_embedding):
        """Test processing story when no matching groups exist."""
        # Mock no existing centroids
        group_manager.storage.get_group_centroids = AsyncMock(return_value=[])
        group_manager.create_new_group = AsyncMock(return_value="new-group-id")
        
        result = await group_manager.process_new_story(sample_embedding)
        
        assert result.assignment_successful is True
        assert result.is_new_group is True
        assert result.group_id == "new-group-id"
        assert result.similarity_score == 1.0
    
    @pytest.mark.asyncio
    async def test_process_new_story_assigns_to_existing_group(self, group_manager, sample_embedding, sample_centroid):
        """Test processing story when matching group exists."""
        # Mock existing centroid
        group_manager.storage.get_group_centroids = AsyncMock(return_value=[sample_centroid])
        group_manager.similarity_calc.find_best_matching_group = Mock(return_value=("group-1", 0.85))
        group_manager.add_story_to_group = AsyncMock(return_value=True)
        
        result = await group_manager.process_new_story(sample_embedding)
        
        assert result.assignment_successful is True
        assert result.is_new_group is False
        assert result.group_id == "group-1"
        assert result.similarity_score == 0.85
    
    @pytest.mark.asyncio
    async def test_find_best_matching_group_with_match(self, group_manager, sample_embedding, sample_centroid, sample_group):
        """Test finding best matching group when match exists."""
        group_manager.storage.get_group_centroids = AsyncMock(return_value=[sample_centroid])
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.similarity_calc.find_best_matching_group = Mock(return_value=("group-1", 0.85))
        
        result = await group_manager.find_best_matching_group(sample_embedding)
        
        assert result is not None
        assert result[0] == "group-1"
        assert result[1] == 0.85
    
    @pytest.mark.asyncio
    async def test_find_best_matching_group_no_match(self, group_manager, sample_embedding):
        """Test finding best matching group when no match exists."""
        group_manager.storage.get_group_centroids = AsyncMock(return_value=[])
        
        result = await group_manager.find_best_matching_group(sample_embedding)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_find_best_matching_group_at_capacity(self, group_manager, sample_embedding, sample_centroid, sample_group):
        """Test finding best matching group when group is at capacity."""
        # Set group at capacity
        sample_group.member_count = 50
        
        group_manager.storage.get_group_centroids = AsyncMock(return_value=[sample_centroid])
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.similarity_calc.find_best_matching_group = Mock(return_value=("group-1", 0.85))
        
        result = await group_manager.find_best_matching_group(sample_embedding)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_create_new_group_success(self, group_manager, sample_embedding):
        """Test successful creation of new group."""
        group_manager.storage.store_group = AsyncMock(return_value=True)
        group_manager.storage.add_member_to_group = AsyncMock(return_value=True)
        
        group_id = await group_manager.create_new_group(sample_embedding)
        
        assert group_id != ""
        assert len(group_id) > 0  # Should be a valid UUID
    
    @pytest.mark.asyncio
    async def test_create_new_group_storage_failure(self, group_manager, sample_embedding):
        """Test new group creation when storage fails."""
        group_manager.storage.store_group = AsyncMock(return_value=False)
        
        group_id = await group_manager.create_new_group(sample_embedding)
        
        assert group_id == ""
    
    # Task 6.2 Tests: Group lifecycle and status management
    
    @pytest.mark.asyncio
    async def test_update_group_lifecycle_new_member(self, group_manager, sample_group):
        """Test lifecycle update when new member is added."""
        sample_group.status = GroupStatus.NEW
        sample_group.member_count = 2
        
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.storage.update_group_status = AsyncMock(return_value=True)
        
        result = await group_manager.update_group_lifecycle("group-1", new_member_added=True)
        
        assert result is True
        # Should update to UPDATED status when member added
        group_manager.storage.update_group_status.assert_called_with("group-1", GroupStatus.UPDATED)
    
    @pytest.mark.asyncio
    async def test_determine_new_status_first_member(self, group_manager, sample_group):
        """Test status determination for first member."""
        sample_group.member_count = 1
        
        status = group_manager._determine_new_status(sample_group, False)
        
        assert status == GroupStatus.NEW
    
    @pytest.mark.asyncio
    async def test_determine_new_status_new_member_added(self, group_manager, sample_group):
        """Test status determination when new member is added."""
        sample_group.member_count = 3
        
        status = group_manager._determine_new_status(sample_group, new_member_added=True)
        
        assert status == GroupStatus.UPDATED
    
    @pytest.mark.asyncio
    async def test_determine_new_status_stable_after_time(self, group_manager, sample_group):
        """Test status determination after group becomes stable."""
        sample_group.member_count = 3
        sample_group.status = GroupStatus.UPDATED
        # Set updated time to more than 24 hours ago
        from datetime import timedelta
        sample_group.updated_at = datetime.now(timezone.utc) - timedelta(hours=25)
        
        status = group_manager._determine_new_status(sample_group, new_member_added=False)
        
        assert status == GroupStatus.STABLE
    
    @pytest.mark.asyncio
    async def test_add_group_tags_success(self, group_manager, sample_group, mock_supabase_client):
        """Test successful addition of tags to group."""
        sample_group.tags = ["existing-tag"]
        
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        
        result = await group_manager.add_group_tags("group-1", ["new-tag", "another-tag"])
        
        assert result is True
        mock_supabase_client.table.assert_called()
    
    @pytest.mark.asyncio
    async def test_add_group_tags_duplicate_prevention(self, group_manager, sample_group, mock_supabase_client):
        """Test that duplicate tags are not added."""
        sample_group.tags = ["existing-tag"]
        
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        
        result = await group_manager.add_group_tags("group-1", ["existing-tag", "new-tag"])
        
        assert result is True
        # Should only add the new tag, not duplicate existing
    
    # Task 6.3 Tests: Group membership operations with validation
    
    @pytest.mark.asyncio
    async def test_add_story_to_group_success(self, group_manager, sample_embedding):
        """Test successful addition of story to group."""
        group_manager.validate_group_membership = AsyncMock(
            return_value=GroupMembershipValidation(is_valid=True)
        )
        group_manager.storage.store_embedding = AsyncMock(return_value=True)
        group_manager.storage.add_member_to_group = AsyncMock(return_value=True)
        group_manager.update_group_centroid = AsyncMock(return_value=True)
        group_manager.update_group_lifecycle = AsyncMock(return_value=True)
        
        result = await group_manager.add_story_to_group("group-1", sample_embedding, 0.85)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_add_story_to_group_validation_failure(self, group_manager, sample_embedding):
        """Test story addition when validation fails."""
        group_manager.validate_group_membership = AsyncMock(
            return_value=GroupMembershipValidation(
                is_valid=False, 
                error_message="Duplicate member"
            )
        )
        
        result = await group_manager.add_story_to_group("group-1", sample_embedding, 0.85)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_group_membership_success(self, group_manager, sample_group):
        """Test successful membership validation."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.storage.check_membership_exists = AsyncMock(return_value=False)
        
        result = await group_manager.validate_group_membership("group-1", "story-1")
        
        assert result.is_valid is True
        assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_validate_group_membership_group_not_found(self, group_manager):
        """Test validation when group doesn't exist."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=None)
        
        result = await group_manager.validate_group_membership("group-1", "story-1")
        
        assert result.is_valid is False
        assert "does not exist" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_group_membership_duplicate_member(self, group_manager, sample_group):
        """Test validation when member already exists."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.storage.check_membership_exists = AsyncMock(return_value=True)
        
        result = await group_manager.validate_group_membership("group-1", "story-1")
        
        assert result.is_valid is False
        assert result.duplicate_found is True
        assert "already a member" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_group_membership_at_capacity(self, group_manager, sample_group):
        """Test validation when group is at capacity."""
        sample_group.member_count = 50  # At max capacity
        
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.storage.check_membership_exists = AsyncMock(return_value=False)
        
        result = await group_manager.validate_group_membership("group-1", "story-1")
        
        assert result.is_valid is False
        assert result.group_at_capacity is True
        assert "at capacity" in result.error_message
    
    @pytest.mark.asyncio
    async def test_update_group_centroid_success(self, group_manager, sample_group, sample_embedding):
        """Test successful centroid update."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.storage.get_group_embeddings = AsyncMock(return_value=[sample_embedding])
        group_manager.storage.store_group = AsyncMock(return_value=True)
        
        # Mock centroid manager update
        update_result = Mock()
        update_result.update_successful = True
        group_manager.centroid_manager.update_group_centroid = Mock(return_value=update_result)
        
        result = await group_manager.update_group_centroid("group-1")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_update_group_centroid_group_not_found(self, group_manager):
        """Test centroid update when group doesn't exist."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=None)
        
        result = await group_manager.update_group_centroid("group-1")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_group_member_count(self, group_manager, sample_group):
        """Test getting group member count."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        
        count = await group_manager.get_group_member_count("group-1")
        
        assert count == sample_group.member_count
    
    @pytest.mark.asyncio
    async def test_get_group_member_count_group_not_found(self, group_manager):
        """Test getting member count when group doesn't exist."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=None)
        
        count = await group_manager.get_group_member_count("group-1")
        
        assert count == 0
    
    # Integration and analytics tests
    
    @pytest.mark.asyncio
    async def test_get_group_analytics(self, group_manager, sample_group, sample_embedding):
        """Test getting group analytics."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=sample_group)
        group_manager.storage.get_group_embeddings = AsyncMock(return_value=[sample_embedding])
        
        analytics = await group_manager.get_group_analytics("group-1")
        
        assert analytics["group_id"] == "group-1"
        assert analytics["member_count"] == sample_group.member_count
        assert analytics["status"] == sample_group.status.value
        assert analytics["tags"] == sample_group.tags
        assert analytics["has_embeddings"] == 1
        assert analytics["avg_confidence"] == sample_embedding.confidence_score
    
    @pytest.mark.asyncio
    async def test_get_group_analytics_group_not_found(self, group_manager):
        """Test getting analytics when group doesn't exist."""
        group_manager.storage.get_group_by_id = AsyncMock(return_value=None)
        
        analytics = await group_manager.get_group_analytics("group-1")
        
        assert analytics == {}


class TestGroupAssignmentResult:
    """Test GroupAssignmentResult data structure."""
    
    def test_assignment_result_creation(self):
        """Test creating assignment result."""
        result = GroupAssignmentResult(
            news_url_id="story-1",
            group_id="group-1",
            similarity_score=0.85,
            is_new_group=False,
            assignment_successful=True
        )
        
        assert result.news_url_id == "story-1"
        assert result.group_id == "group-1"
        assert result.similarity_score == 0.85
        assert result.is_new_group is False
        assert result.assignment_successful is True
        assert result.error_message is None


class TestGroupMembershipValidation:
    """Test GroupMembershipValidation data structure."""
    
    def test_validation_success(self):
        """Test successful validation result."""
        validation = GroupMembershipValidation(is_valid=True)
        
        assert validation.is_valid is True
        assert validation.error_message is None
        assert validation.duplicate_found is False
        assert validation.group_at_capacity is False
    
    def test_validation_failure_with_details(self):
        """Test validation failure with detailed information."""
        validation = GroupMembershipValidation(
            is_valid=False,
            error_message="Group is at capacity",
            duplicate_found=False,
            group_at_capacity=True
        )
        
        assert validation.is_valid is False
        assert validation.error_message == "Group is at capacity"
        assert validation.duplicate_found is False
        assert validation.group_at_capacity is True
