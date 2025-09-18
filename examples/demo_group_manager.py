#!/usr/bin/env python3
"""
Demo script for GroupManager Tasks 6.1, 6.2, and 6.3.

This script demonstrates the core functionality implemented for story grouping:
- Task 6.1: Group assignment and story processing  
- Task 6.2: Group lifecycle and status management
- Task 6.3: Group membership operations with validation

Run with: python demo_group_manager.py
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import Mock

from src.nfl_news_pipeline.models import (
    StoryEmbedding,
    StoryGroup,
    GroupStatus,
    EMBEDDING_DIM
)
from src.nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
from src.nfl_news_pipeline.group_manager import GroupManager, GroupStorageManager


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_mock_supabase_client():
    """Create a mock Supabase client for demo purposes."""
    client = Mock()
    
    # Track state for demo
    client._groups = {}
    client._embeddings = {}
    client._members = {}
    
    # Mock table method
    mock_table = Mock()
    client.table.return_value = mock_table
    
    # Mock responses - in real implementation these would query actual database
    mock_table.select.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.update.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.in_.return_value = mock_table
    mock_table.is_.return_value = mock_table
    mock_table.execute.return_value = Mock(data=[])
    
    return client


def create_sample_embeddings():
    """Create sample story embeddings for demo."""
    embeddings = []
    
    # Similar NFL trade stories (should group together)
    embeddings.append(StoryEmbedding(
        news_url_id="trade-story-1",
        embedding_vector=[0.8, 0.2] + [0.1] * (EMBEDDING_DIM - 2),
        model_name="demo-model",
        model_version="1.0",
        summary_text="NFL quarterback trade between teams",
        confidence_score=0.9,
        generated_at=datetime.now(timezone.utc)
    ))
    
    embeddings.append(StoryEmbedding(
        news_url_id="trade-story-2",
        embedding_vector=[0.82, 0.18] + [0.05] * (EMBEDDING_DIM - 2),
        model_name="demo-model",
        model_version="1.0",
        summary_text="Star quarterback traded to new team",
        confidence_score=0.85,
        generated_at=datetime.now(timezone.utc)
    ))
    
    # Different story about injuries (should create new group)
    embeddings.append(StoryEmbedding(
        news_url_id="injury-story-1",
        embedding_vector=[0.1, 0.8] + [0.05] * (EMBEDDING_DIM - 2),
        model_name="demo-model",
        model_version="1.0",
        summary_text="NFL player suffers knee injury",
        confidence_score=0.88,
        generated_at=datetime.now(timezone.utc)
    ))
    
    # Another trade story (should join first group)
    embeddings.append(StoryEmbedding(
        news_url_id="trade-story-3",
        embedding_vector=[0.78, 0.22] + [0.08] * (EMBEDDING_DIM - 2),
        model_name="demo-model", 
        model_version="1.0",
        summary_text="Major trade deal shakes up NFL",
        confidence_score=0.92,
        generated_at=datetime.now(timezone.utc)
    ))
    
    return embeddings


async def demo_group_manager():
    """Demonstrate GroupManager functionality."""
    print("=" * 60)
    print("NFL Story Grouping Demo - Tasks 6.1, 6.2, 6.3")
    print("=" * 60)
    
    # Initialize components
    mock_client = create_mock_supabase_client()
    storage_manager = GroupStorageManager(mock_client)
    similarity_calculator = SimilarityCalculator(similarity_threshold=0.8, metric=SimilarityMetric.COSINE)
    centroid_manager = GroupCentroidManager()
    
    group_manager = GroupManager(
        storage_manager=storage_manager,
        similarity_calculator=similarity_calculator,
        centroid_manager=centroid_manager,
        similarity_threshold=0.8,
        max_group_size=50
    )
    
    # Mock storage operations for demo
    from unittest.mock import AsyncMock
    storage_manager.get_group_centroids = AsyncMock(return_value=[])
    storage_manager.store_group = AsyncMock(return_value=True)
    storage_manager.add_member_to_group = AsyncMock(return_value=True)
    storage_manager.get_group_by_id = AsyncMock(return_value=None)
    storage_manager.store_embedding = AsyncMock(return_value=True)
    storage_manager.check_membership_exists = AsyncMock(return_value=False)
    storage_manager.get_group_embeddings = AsyncMock(return_value=[])
    
    print("\n📋 Creating sample story embeddings...")
    sample_embeddings = create_sample_embeddings()
    
    for i, embedding in enumerate(sample_embeddings, 1):
        print(f"  {i}. {embedding.news_url_id}: {embedding.summary_text}")
    
    print(f"\n🔧 Configured GroupManager:")
    print(f"  - Similarity threshold: {group_manager.similarity_threshold}")
    print(f"  - Max group size: {group_manager.max_group_size}")
    print(f"  - Similarity metric: {similarity_calculator.metric.value}")
    
    # Task 6.1: Process stories and assign to groups
    print(f"\n{'='*60}")
    print("Task 6.1: Group Assignment and Story Processing")
    print("="*60)
    
    group_assignments = []
    
    for i, embedding in enumerate(sample_embeddings, 1):
        print(f"\n📰 Processing Story {i}: {embedding.news_url_id}")
        
        # Simulate finding centroids for existing groups
        if group_assignments:
            # For demo, create mock centroids from previous assignments
            mock_centroids = []
            for assignment in group_assignments:
                if assignment.is_new_group and assignment.assignment_successful:
                    from src.nfl_news_pipeline.models import GroupCentroid
                    centroid = GroupCentroid(
                        group_id=assignment.group_id,
                        centroid_vector=sample_embeddings[0].embedding_vector,  # Simplified
                        member_count=1,
                        last_updated=datetime.now(timezone.utc)
                    )
                    mock_centroids.append(centroid)
            
            storage_manager.get_group_centroids = AsyncMock(return_value=mock_centroids)
        
        result = await group_manager.process_new_story(embedding)
        group_assignments.append(result)
        
        print(f"  ✅ Result: {'New group created' if result.is_new_group else 'Assigned to existing group'}")
        print(f"  📊 Group ID: {result.group_id}")
        print(f"  🎯 Similarity Score: {result.similarity_score:.3f}")
        print(f"  ✔️ Success: {result.assignment_successful}")
        
        if result.error_message:
            print(f"  ❌ Error: {result.error_message}")
    
    # Task 6.2: Group lifecycle and status management
    print(f"\n{'='*60}")
    print("Task 6.2: Group Lifecycle and Status Management")
    print("="*60)
    
    # Simulate group lifecycle updates
    for assignment in group_assignments:
        if assignment.assignment_successful:
            print(f"\n📊 Managing lifecycle for group: {assignment.group_id}")
            
            # Mock a group for lifecycle testing
            sample_group = StoryGroup(
                id=assignment.group_id,
                member_count=2 if not assignment.is_new_group else 1,
                status=GroupStatus.NEW if assignment.is_new_group else GroupStatus.UPDATED,
                tags=[],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            storage_manager.get_group_by_id = AsyncMock(return_value=sample_group)
            storage_manager.update_group_status = AsyncMock(return_value=True)
            
            # Update lifecycle
            success = await group_manager.update_group_lifecycle(
                assignment.group_id, 
                new_member_added=not assignment.is_new_group
            )
            
            print(f"  📈 Lifecycle updated: {success}")
            print(f"  🏷️ Current status: {sample_group.status.value}")
            
            # Add tags to group
            tags = ["nfl", "trade"] if "trade" in assignment.news_url_id else ["nfl", "injury"]
            tag_success = await group_manager.add_group_tags(assignment.group_id, tags)
            print(f"  🏷️ Tags added: {tag_success} - {tags}")
    
    # Task 6.3: Group membership operations with validation
    print(f"\n{'='*60}")
    print("Task 6.3: Group Membership Operations with Validation")
    print("="*60)
    
    # Test membership validation
    if group_assignments:
        test_group_id = group_assignments[0].group_id
        test_story_id = "test-duplicate-story"
        
        print(f"\n🔍 Testing membership validation for group: {test_group_id}")
        
        # Test valid membership
        validation = await group_manager.validate_group_membership(test_group_id, test_story_id)
        print(f"  ✅ Valid membership: {validation.is_valid}")
        
        if not validation.is_valid:
            print(f"  ❌ Validation error: {validation.error_message}")
            print(f"  🔄 Duplicate found: {validation.duplicate_found}")
            print(f"  📊 Group at capacity: {validation.group_at_capacity}")
        
        # Test group member count
        count = await group_manager.get_group_member_count(test_group_id)
        print(f"  👥 Current member count: {count}")
        
        # Test group analytics
        analytics = await group_manager.get_group_analytics(test_group_id)
        if analytics:
            print(f"  📈 Group Analytics:")
            print(f"    - Member count: {analytics.get('member_count', 'N/A')}")
            print(f"    - Status: {analytics.get('status', 'N/A')}")
            print(f"    - Tags: {analytics.get('tags', [])}")
            print(f"    - Average confidence: {analytics.get('avg_confidence', 0.0):.3f}")
    
    print(f"\n{'='*60}")
    print("Demo Complete! 🎉")
    print("="*60)
    
    print(f"\n📊 Summary:")
    print(f"  - Stories processed: {len(sample_embeddings)}")
    print(f"  - Successful assignments: {sum(1 for a in group_assignments if a.assignment_successful)}")
    print(f"  - New groups created: {sum(1 for a in group_assignments if a.is_new_group)}")
    print(f"  - Existing group assignments: {sum(1 for a in group_assignments if not a.is_new_group)}")
    
    print(f"\n✅ Tasks 6.1, 6.2, and 6.3 implementation demonstrated successfully!")
    
    return group_assignments


async def main():
    """Main demo function."""
    try:
        await demo_group_manager()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    print("🚀 Starting NFL Story Grouping Demo...")
    asyncio.run(main())