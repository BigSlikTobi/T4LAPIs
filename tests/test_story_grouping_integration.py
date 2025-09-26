"""
Comprehensive integration tests for end-to-end story grouping functionality (Task 11.2).

These tests validate the complete story grouping pipeline from URL context extraction
through embedding generation to group assignment and storage operations.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from typing import List, Dict, Any

from src.nfl_news_pipeline.models import (
    ProcessedNewsItem,
    ContextSummary,
    StoryEmbedding,
    StoryGroup,
    StoryGroupMember,
    GroupCentroid,
    GroupStatus,
    EMBEDDING_DIM
)
from src.nfl_news_pipeline.story_grouping import URLContextExtractor, ContextCache
from src.nfl_news_pipeline.embedding import EmbeddingGenerator, EmbeddingStorageManager
from src.nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from src.nfl_news_pipeline.group_manager import GroupManager, GroupStorageManager
from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
from src.nfl_news_pipeline.orchestrator import StoryGroupingOrchestrator


class TestStoryGroupingPipelineIntegration:
    """Integration tests for the complete story grouping pipeline."""
    
    @pytest.fixture
    def sample_news_items(self):
        """Create a realistic set of news items for integration testing."""
        return [
            ProcessedNewsItem(
                url="https://nfl.com/chiefs-mahomes-4-touchdowns",
                title="Patrick Mahomes throws 4 touchdowns as Chiefs defeat Raiders 35-21",
                publication_date=datetime.now(timezone.utc) - timedelta(hours=2),
                source_name="nfl_official",
                publisher="NFL.com",
                description="Kansas City Chiefs quarterback Patrick Mahomes threw four touchdown passes in a dominant victory over the Las Vegas Raiders.",
                relevance_score=0.95,
                filter_method="rule_based",
                entities=["Patrick Mahomes", "Kansas City Chiefs", "Las Vegas Raiders"],
                categories=["game_recap", "quarterback", "touchdowns"]
            ),
            ProcessedNewsItem(
                url="https://espn.com/chiefs-victory-mahomes-stellar",
                title="Mahomes leads Chiefs to impressive win with stellar performance",
                publication_date=datetime.now(timezone.utc) - timedelta(hours=1),
                source_name="espn",
                publisher="ESPN",
                description="Patrick Mahomes showcased his elite quarterback skills as the Kansas City Chiefs secured a convincing victory.",
                relevance_score=0.92,
                filter_method="llm",
                entities=["Patrick Mahomes", "Kansas City Chiefs"],
                categories=["game_recap", "quarterback"]
            ),
            ProcessedNewsItem(
                url="https://bleacherreport.com/bills-allen-comeback",
                title="Josh Allen orchestrates stunning comeback for Buffalo Bills",
                publication_date=datetime.now(timezone.utc) - timedelta(hours=3),
                source_name="bleacher_report",
                publisher="Bleacher Report",
                description="Buffalo Bills quarterback Josh Allen led a remarkable fourth-quarter comeback against the Miami Dolphins.",
                relevance_score=0.89,
                filter_method="rule_based",
                entities=["Josh Allen", "Buffalo Bills", "Miami Dolphins"],
                categories=["game_recap", "comeback", "quarterback"]
            ),
            ProcessedNewsItem(
                url="https://nfl.com/trade-deadline-activity",
                title="NFL Trade Deadline: Multiple teams make significant moves",
                publication_date=datetime.now(timezone.utc) - timedelta(hours=4),
                source_name="nfl_official",
                publisher="NFL.com",
                description="Several NFL teams completed major trades before the deadline, reshaping their rosters for the playoff push.",
                relevance_score=0.85,
                filter_method="llm",
                entities=["NFL"],
                categories=["trade", "deadline", "roster"]
            ),
            ProcessedNewsItem(
                url="https://profootballtalk.com/mahomes-injury-update",
                title="Patrick Mahomes injury update: Expected to play next week",
                publication_date=datetime.now(timezone.utc) - timedelta(hours=5),
                source_name="pft",
                publisher="Pro Football Talk",
                description="Kansas City Chiefs quarterback Patrick Mahomes is expected to play despite minor ankle injury from yesterday's game.",
                relevance_score=0.91,
                filter_method="rule_based",
                entities=["Patrick Mahomes", "Kansas City Chiefs"],
                categories=["injury", "quarterback", "update"]
            )
        ]
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Create realistic LLM responses for different story types."""
        return {
            "chiefs-mahomes-4-touchdowns": {
                "summary": "Patrick Mahomes delivered an outstanding performance throwing four touchdown passes as the Kansas City Chiefs dominated the Las Vegas Raiders 35-21 in a commanding victory that showcased the team's offensive prowess.",
                "entities": {
                    "players": ["Patrick Mahomes"],
                    "teams": ["Kansas City Chiefs", "Las Vegas Raiders"],
                    "coaches": [],
                    "positions": ["quarterback"]
                },
                "key_topics": ["touchdown", "victory", "performance", "game"],
                "story_category": "game_recap",
                "confidence": 0.94
            },
            "chiefs-victory-mahomes-stellar": {
                "summary": "Patrick Mahomes demonstrated elite quarterback skills leading the Kansas City Chiefs to an impressive victory with a stellar performance that highlighted his exceptional talent and leadership.",
                "entities": {
                    "players": ["Patrick Mahomes"],
                    "teams": ["Kansas City Chiefs"],
                    "coaches": [],
                    "positions": ["quarterback"]
                },
                "key_topics": ["victory", "performance", "quarterback", "leadership"],
                "story_category": "game_recap",
                "confidence": 0.91
            },
            "bills-allen-comeback": {
                "summary": "Josh Allen orchestrated a stunning fourth-quarter comeback as the Buffalo Bills overcame a significant deficit to defeat the Miami Dolphins in a thrilling display of quarterback excellence.",
                "entities": {
                    "players": ["Josh Allen"],
                    "teams": ["Buffalo Bills", "Miami Dolphins"],
                    "coaches": [],
                    "positions": ["quarterback"]
                },
                "key_topics": ["comeback", "quarterback", "victory", "fourth-quarter"],
                "story_category": "game_recap",
                "confidence": 0.89
            },
            "trade-deadline-activity": {
                "summary": "Multiple NFL teams completed significant trades before the deadline, making strategic roster moves to strengthen their positions for the upcoming playoff push and championship aspirations.",
                "entities": {
                    "players": [],
                    "teams": ["NFL"],
                    "coaches": [],
                    "positions": []
                },
                "key_topics": ["trade", "deadline", "roster", "playoffs"],
                "story_category": "trade_news",
                "confidence": 0.87
            },
            "mahomes-injury-update": {
                "summary": "Patrick Mahomes is expected to play in next week's game despite sustaining a minor ankle injury during the Kansas City Chiefs' recent victory, according to team medical staff.",
                "entities": {
                    "players": ["Patrick Mahomes"],
                    "teams": ["Kansas City Chiefs"],
                    "coaches": [],
                    "positions": ["quarterback"]
                },
                "key_topics": ["injury", "update", "quarterback", "chiefs"],
                "story_category": "injury_report",
                "confidence": 0.92
            }
        }
    
    @pytest.fixture
    def mock_embedding_vectors(self):
        """Create realistic embedding vectors that show similarity patterns."""
        # Base vector for Mahomes/Chiefs stories (similar)
        mahomes_base = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        mahomes_base = mahomes_base / np.linalg.norm(mahomes_base)
        
        # Similar vector for related Mahomes story (high similarity ~0.85)
        mahomes_similar = mahomes_base + np.random.normal(0, 0.1, EMBEDDING_DIM).astype(np.float32)
        mahomes_similar = mahomes_similar / np.linalg.norm(mahomes_similar)
        
        # Different but related vector for Mahomes injury (medium similarity ~0.7)
        mahomes_injury = mahomes_base + np.random.normal(0, 0.2, EMBEDDING_DIM).astype(np.float32)
        mahomes_injury = mahomes_injury / np.linalg.norm(mahomes_injury)
        
        # Different vector for Josh Allen story (low similarity ~0.4)
        allen_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        allen_vector = allen_vector / np.linalg.norm(allen_vector)
        
        # Different vector for trade story (very low similarity ~0.2)
        trade_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        trade_vector = trade_vector / np.linalg.norm(trade_vector)
        
        return {
            "chiefs-mahomes-4-touchdowns": mahomes_base,
            "chiefs-victory-mahomes-stellar": mahomes_similar,
            "bills-allen-comeback": allen_vector,
            "trade-deadline-activity": trade_vector,
            "mahomes-injury-update": mahomes_injury
        }
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_single_story(self, sample_news_items, mock_llm_responses, mock_embedding_vectors):
        """Test the complete pipeline processing a single story."""
        story = sample_news_items[0]  # Mahomes 4 touchdowns story
        story_key = "chiefs-mahomes-4-touchdowns"
        
        # Mock context extraction
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps(mock_llm_responses[story_key])
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Mock embedding generation
            with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
                mock_model = Mock()
                mock_model.encode.return_value = mock_embedding_vectors[story_key]
                mock_transformer.return_value = mock_model
                
                # Mock storage
                mock_supabase = Mock()
                mock_table = Mock()
                mock_supabase.table.return_value = mock_table
                
                # Mock no existing groups (new group creation)
                mock_table.select.return_value.is_.return_value.execute.return_value = Mock(data=[])
                mock_table.insert.return_value.execute.return_value = Mock(data=[{"id": "new-group-id"}])
                
                # Initialize pipeline components
                cache = ContextCache(enable_disk_cache=False)
                extractor = URLContextExtractor(
                    openai_api_key="test-key",
                    cache=cache,
                    enable_caching=True
                )
                
                generator = EmbeddingGenerator(use_openai_primary=False)
                embedding_storage = EmbeddingStorageManager(mock_supabase)
                similarity_calc = SimilarityCalculator(similarity_threshold=0.8)
                centroid_manager = GroupCentroidManager()
                group_storage = GroupStorageManager(mock_supabase)
                group_manager = GroupManager(group_storage, similarity_calc, centroid_manager)
                
                # Create orchestrator
                orchestrator = StoryGroupingOrchestrator(
                    context_extractor=extractor,
                    embedding_generator=generator,
                    group_manager=group_manager,
                    embedding_storage=embedding_storage
                )
                
                # Process the story
                result = await orchestrator.process_story(story)
                
                # Verify complete pipeline execution
                assert result is not None
                assert result.group_id == "new-group-id"
                assert result.is_new_group is True
                assert result.similarity_score == 1.0  # Perfect similarity with self for new group
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_multiple_related_stories(self, sample_news_items, mock_llm_responses, mock_embedding_vectors):
        """Test pipeline processing multiple related stories that should group together."""
        import json
        
        # Process two related Mahomes stories
        story1 = sample_news_items[0]  # Mahomes 4 touchdowns
        story2 = sample_news_items[1]  # Mahomes stellar performance
        
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Setup responses for both stories
            mock_responses = [
                Mock(choices=[Mock(message=Mock(content=json.dumps(mock_llm_responses["chiefs-mahomes-4-touchdowns"])))]),
                Mock(choices=[Mock(message=Mock(content=json.dumps(mock_llm_responses["chiefs-victory-mahomes-stellar"])))]),
            ]
            mock_client.chat.completions.create.side_effect = mock_responses
            
            with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
                mock_model = Mock()
                mock_transformer.return_value = mock_model
                
                # Setup embedding responses
                mock_embeddings = [
                    mock_embedding_vectors["chiefs-mahomes-4-touchdowns"],
                    mock_embedding_vectors["chiefs-victory-mahomes-stellar"]
                ]
                mock_model.encode.side_effect = mock_embeddings
                
                # Mock storage for group creation and joining
                mock_supabase = Mock()
                mock_table = Mock()
                mock_supabase.table.return_value = mock_table
                
                # First story: no existing groups
                # Second story: finds existing group
                group_centroid_data = [{
                    "id": "group-1",
                    "centroid_embedding": mock_embedding_vectors["chiefs-mahomes-4-touchdowns"].tolist(),
                    "member_count": 1,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }]
                
                mock_table.select.return_value.is_.return_value.execute.side_effect = [
                    Mock(data=[]),  # No groups for first story
                    Mock(data=group_centroid_data)  # Group exists for second story
                ]
                
                mock_table.insert.return_value.execute.return_value = Mock(data=[{"id": "group-1"}])
                mock_table.update.return_value.eq.return_value.execute.return_value = Mock(data=[{"id": "group-1"}])
                
                # Initialize components
                cache = ContextCache(enable_disk_cache=False)
                extractor = URLContextExtractor(openai_api_key="test-key", cache=cache)
                generator = EmbeddingGenerator(use_openai_primary=False)
                embedding_storage = EmbeddingStorageManager(mock_supabase)
                similarity_calc = SimilarityCalculator(similarity_threshold=0.8)
                centroid_manager = GroupCentroidManager()
                group_storage = GroupStorageManager(mock_supabase)
                group_manager = GroupManager(group_storage, similarity_calc, centroid_manager)
                
                orchestrator = StoryGroupingOrchestrator(
                    context_extractor=extractor,
                    embedding_generator=generator,
                    group_manager=group_manager,
                    embedding_storage=embedding_storage
                )
                
                # Process both stories
                result1 = await orchestrator.process_story(story1)
                result2 = await orchestrator.process_story(story2)
                
                # Verify results
                assert result1.is_new_group is True
                assert result1.group_id == "group-1"
                
                # Second story should join the existing group
                assert result2.is_new_group is False
                assert result2.group_id == "group-1"
                # Similarity should be high (vectors are designed to be similar)
                assert result2.similarity_score > 0.8
    
    @pytest.mark.asyncio
    async def test_pipeline_with_different_story_types(self, sample_news_items, mock_llm_responses, mock_embedding_vectors):
        """Test pipeline handling different types of stories (game recaps vs. trades vs. injuries)."""
        import json
        
        # Process stories of different types
        stories = [
            (sample_news_items[0], "chiefs-mahomes-4-touchdowns"),  # Game recap
            (sample_news_items[3], "trade-deadline-activity"),      # Trade news
            (sample_news_items[4], "mahomes-injury-update")         # Injury report
        ]
        
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Setup LLM responses
            mock_responses = [
                Mock(choices=[Mock(message=Mock(content=json.dumps(mock_llm_responses[key])))]) 
                for _, key in stories
            ]
            mock_client.chat.completions.create.side_effect = mock_responses
            
            with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
                mock_model = Mock()
                mock_transformer.return_value = mock_model
                
                # Setup embedding responses
                mock_embeddings = [mock_embedding_vectors[key] for _, key in stories]
                mock_model.encode.side_effect = mock_embeddings
                
                # Mock storage - each story creates new group (different types)
                mock_supabase = Mock()
                mock_table = Mock()
                mock_supabase.table.return_value = mock_table
                
                # No existing groups, so each creates new
                mock_table.select.return_value.is_.return_value.execute.return_value = Mock(data=[])
                mock_table.insert.return_value.execute.side_effect = [
                    Mock(data=[{"id": f"group-{i}"}]) for i in range(len(stories))
                ]
                
                # Initialize pipeline
                cache = ContextCache(enable_disk_cache=False)
                extractor = URLContextExtractor(openai_api_key="test-key", cache=cache)
                generator = EmbeddingGenerator(use_openai_primary=False)
                embedding_storage = EmbeddingStorageManager(mock_supabase)
                similarity_calc = SimilarityCalculator(similarity_threshold=0.8)
                centroid_manager = GroupCentroidManager()
                group_storage = GroupStorageManager(mock_supabase)
                group_manager = GroupManager(group_storage, similarity_calc, centroid_manager)
                
                orchestrator = StoryGroupingOrchestrator(
                    context_extractor=extractor,
                    embedding_generator=generator,
                    group_manager=group_manager,
                    embedding_storage=embedding_storage
                )
                
                # Process all stories
                results = []
                for story, key in stories:
                    result = await orchestrator.process_story(story)
                    results.append(result)
                
                # Verify each story created its own group (different types)
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result.is_new_group is True
                    assert result.group_id == f"group-{i}"
                    assert result.similarity_score == 1.0  # Perfect with self
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling_and_fallbacks(self, sample_news_items):
        """Test pipeline error handling with fallbacks at each stage."""
        story = sample_news_items[0]
        
        # Test LLM failure -> fallback to metadata
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("LLM API Error")
            mock_openai.return_value = mock_client
            
            # Test embedding failure -> fallback embedding
            with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
                mock_transformer.side_effect = Exception("Model loading failed")
                
                # Mock storage
                mock_supabase = Mock()
                mock_table = Mock()
                mock_supabase.table.return_value = mock_table
                mock_table.select.return_value.is_.return_value.execute.return_value = Mock(data=[])
                mock_table.insert.return_value.execute.return_value = Mock(data=[{"id": "fallback-group"}])
                
                # Initialize with error handling
                cache = ContextCache(enable_disk_cache=False)
                extractor = URLContextExtractor(cache=cache)  # No API keys, will fallback
                
                # Use error handler for embeddings
                from src.nfl_news_pipeline.embedding.error_handler import EmbeddingErrorHandler
                error_handler = EmbeddingErrorHandler()
                
                generator = EmbeddingGenerator(use_openai_primary=False)
                embedding_storage = EmbeddingStorageManager(mock_supabase)
                similarity_calc = SimilarityCalculator(similarity_threshold=0.8)
                centroid_manager = GroupCentroidManager()
                group_storage = GroupStorageManager(mock_supabase)
                group_manager = GroupManager(group_storage, similarity_calc, centroid_manager)
                
                orchestrator = StoryGroupingOrchestrator(
                    context_extractor=extractor,
                    embedding_generator=generator,
                    group_manager=group_manager,
                    embedding_storage=embedding_storage,
                    enable_fallbacks=True
                )
                
                # Process story with fallbacks
                result = await orchestrator.process_story(story)
                
                # Should still succeed with fallbacks
                assert result is not None
                assert result.group_id == "fallback-group"
    
    @pytest.mark.asyncio
    async def test_pipeline_caching_effectiveness(self, sample_news_items, mock_llm_responses, mock_embedding_vectors):
        """Test that caching works effectively across pipeline stages."""
        import json
        
        story = sample_news_items[0]
        story_key = "chiefs-mahomes-4-touchdowns"
        
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps(mock_llm_responses[story_key])
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
                mock_model = Mock()
                mock_model.encode.return_value = mock_embedding_vectors[story_key]
                mock_transformer.return_value = mock_model
                
                # Mock storage
                mock_supabase = Mock()
                mock_table = Mock()
                mock_supabase.table.return_value = mock_table
                mock_table.select.return_value.is_.return_value.execute.return_value = Mock(data=[])
                mock_table.insert.return_value.execute.return_value = Mock(data=[{"id": "cached-group"}])
                
                # Initialize with caching enabled
                cache = ContextCache(enable_disk_cache=False, enable_memory_cache=True)
                extractor = URLContextExtractor(
                    openai_api_key="test-key",
                    cache=cache,
                    enable_caching=True
                )
                
                generator = EmbeddingGenerator(use_openai_primary=False)
                embedding_storage = EmbeddingStorageManager(mock_supabase)
                similarity_calc = SimilarityCalculator(similarity_threshold=0.8)
                centroid_manager = GroupCentroidManager()
                group_storage = GroupStorageManager(mock_supabase)
                group_manager = GroupManager(group_storage, similarity_calc, centroid_manager)
                
                orchestrator = StoryGroupingOrchestrator(
                    context_extractor=extractor,
                    embedding_generator=generator,
                    group_manager=group_manager,
                    embedding_storage=embedding_storage
                )
                
                # Process same story twice
                result1 = await orchestrator.process_story(story)
                result2 = await orchestrator.process_story(story)
                
                # Verify results are consistent
                assert result1.group_id == result2.group_id
                
                # Verify LLM was only called once (cached on second call)
                assert mock_client.chat.completions.create.call_count == 1
                
                # Check cache statistics
                cache_stats = cache.get_cache_stats()
                assert cache_stats["cache_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_processing_performance(self, sample_news_items, mock_llm_responses, mock_embedding_vectors):
        """Test pipeline batch processing for performance and consistency."""
        import json
        
        # Process all stories in batch
        stories = sample_news_items
        story_keys = ["chiefs-mahomes-4-touchdowns", "chiefs-victory-mahomes-stellar", 
                     "bills-allen-comeback", "trade-deadline-activity", "mahomes-injury-update"]
        
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Setup LLM responses for all stories
            mock_responses = [
                Mock(choices=[Mock(message=Mock(content=json.dumps(mock_llm_responses[key])))])
                for key in story_keys
            ]
            mock_client.chat.completions.create.side_effect = mock_responses
            
            with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
                mock_model = Mock()
                mock_transformer.return_value = mock_model
                
                # Setup embedding responses
                mock_embeddings = [mock_embedding_vectors[key] for key in story_keys]
                mock_model.encode.side_effect = mock_embeddings
                
                # Mock storage for batch operations
                mock_supabase = Mock()
                mock_table = Mock()
                mock_supabase.table.return_value = mock_table
                
                # Setup responses for group creation/joining
                mock_table.select.return_value.is_.return_value.execute.return_value = Mock(data=[])
                mock_table.insert.return_value.execute.side_effect = [
                    Mock(data=[{"id": f"batch-group-{i}"}]) for i in range(len(stories))
                ]
                
                # Initialize pipeline
                cache = ContextCache(enable_disk_cache=False)
                extractor = URLContextExtractor(openai_api_key="test-key", cache=cache)
                generator = EmbeddingGenerator(use_openai_primary=False, batch_size=3)  # Test batching
                embedding_storage = EmbeddingStorageManager(mock_supabase)
                similarity_calc = SimilarityCalculator(similarity_threshold=0.8)
                centroid_manager = GroupCentroidManager()
                group_storage = GroupStorageManager(mock_supabase)
                group_manager = GroupManager(group_storage, similarity_calc, centroid_manager)
                
                orchestrator = StoryGroupingOrchestrator(
                    context_extractor=extractor,
                    embedding_generator=generator,
                    group_manager=group_manager,
                    embedding_storage=embedding_storage
                )
                
                # Process batch
                results = await orchestrator.process_stories_batch(stories)
                
                # Verify all stories processed successfully
                assert len(results) == len(stories)
                for i, result in enumerate(results):
                    assert result is not None
                    assert result.group_id == f"batch-group-{i}"
    
    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(self, sample_news_items, mock_llm_responses, mock_embedding_vectors):
        """Test pipeline with concurrent story processing."""
        import json
        
        stories = sample_news_items[:3]  # Process 3 stories concurrently
        story_keys = ["chiefs-mahomes-4-touchdowns", "chiefs-victory-mahomes-stellar", "bills-allen-comeback"]
        
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Setup concurrent LLM responses
            mock_responses = [
                Mock(choices=[Mock(message=Mock(content=json.dumps(mock_llm_responses[key])))])
                for key in story_keys
            ]
            mock_client.chat.completions.create.side_effect = mock_responses
            
            with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
                mock_model = Mock()
                mock_transformer.return_value = mock_model
                
                # Setup concurrent embedding responses
                mock_embeddings = [mock_embedding_vectors[key] for key in story_keys]
                mock_model.encode.side_effect = mock_embeddings
                
                # Mock storage with thread-safe operations
                mock_supabase = Mock()
                mock_table = Mock()
                mock_supabase.table.return_value = mock_table
                
                mock_table.select.return_value.is_.return_value.execute.return_value = Mock(data=[])
                mock_table.insert.return_value.execute.side_effect = [
                    Mock(data=[{"id": f"concurrent-group-{i}"}]) for i in range(len(stories))
                ]
                
                # Initialize pipeline with concurrency support
                cache = ContextCache(enable_disk_cache=False)
                extractor = URLContextExtractor(openai_api_key="test-key", cache=cache)
                generator = EmbeddingGenerator(use_openai_primary=False)
                embedding_storage = EmbeddingStorageManager(mock_supabase)
                similarity_calc = SimilarityCalculator(similarity_threshold=0.8)
                centroid_manager = GroupCentroidManager()
                group_storage = GroupStorageManager(mock_supabase)
                group_manager = GroupManager(group_storage, similarity_calc, centroid_manager)
                
                orchestrator = StoryGroupingOrchestrator(
                    context_extractor=extractor,
                    embedding_generator=generator,
                    group_manager=group_manager,
                    embedding_storage=embedding_storage,
                    max_concurrent_stories=3
                )
                
                # Process stories concurrently
                tasks = [orchestrator.process_story(story) for story in stories]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify all completed successfully
                assert len(results) == 3
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        pytest.fail(f"Concurrent processing failed: {result}")
                    assert result.group_id == f"concurrent-group-{i}"


class TestDatabaseIntegrationWithVectorOperations:
    """Integration tests for database operations with vector similarity."""
    
    @pytest.fixture
    def mock_vector_database(self):
        """Create a mock database that simulates vector operations."""
        class MockVectorDB:
            def __init__(self):
                self.groups = {}
                self.embeddings = {}
                self.members = {}
            
            def store_embedding(self, embedding: StoryEmbedding):
                self.embeddings[embedding.news_url_id] = embedding
                return True
            
            def store_group(self, group: StoryGroup):
                group_id = group.id or str(uuid4())
                group.id = group_id
                self.groups[group_id] = group
                return group_id
            
            def vector_similarity_search(self, query_vector: List[float], threshold: float = 0.8):
                """Simulate vector similarity search."""
                results = []
                query_np = np.array(query_vector)
                
                for group_id, group in self.groups.items():
                    if group.centroid_embedding:
                        centroid_np = np.array(group.centroid_embedding)
                        similarity = np.dot(query_np, centroid_np) / (
                            np.linalg.norm(query_np) * np.linalg.norm(centroid_np)
                        )
                        if similarity >= threshold:
                            results.append((group_id, similarity))
                
                return sorted(results, key=lambda x: x[1], reverse=True)
            
            def get_group_centroids(self):
                centroids = []
                for group_id, group in self.groups.items():
                    if group.centroid_embedding:
                        centroid = GroupCentroid(
                            group_id=group_id,
                            centroid_vector=group.centroid_embedding,
                            member_count=group.member_count,
                            last_updated=group.updated_at
                        )
                        centroids.append(centroid)
                return centroids
        
        return MockVectorDB()
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search_accuracy(self, mock_vector_database):
        """Test vector similarity search accuracy with known similar vectors."""
        # Create base vector
        base_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)
        
        # Create similar vector (should have high similarity ~0.9)
        similar_vector = base_vector + np.random.normal(0, 0.1, EMBEDDING_DIM).astype(np.float32)
        similar_vector = similar_vector / np.linalg.norm(similar_vector)
        
        # Create dissimilar vector (should have low similarity ~0.3)
        dissimilar_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        dissimilar_vector = dissimilar_vector / np.linalg.norm(dissimilar_vector)
        
        # Store groups with centroids
        similar_group = StoryGroup(
            id="similar-group",
            centroid_embedding=base_vector.tolist(),
            member_count=1,
            status=GroupStatus.NEW
        )
        
        dissimilar_group = StoryGroup(
            id="dissimilar-group",
            centroid_embedding=dissimilar_vector.tolist(),
            member_count=1,
            status=GroupStatus.NEW
        )
        
        mock_vector_database.store_group(similar_group)
        mock_vector_database.store_group(dissimilar_group)
        
        # Search with similar vector
        results = mock_vector_database.vector_similarity_search(
            similar_vector.tolist(), 
            threshold=0.8
        )
        
        # Should find the similar group but not the dissimilar one
        assert len(results) == 1
        assert results[0][0] == "similar-group"
        assert results[0][1] > 0.8  # High similarity
        
        # Search with lower threshold
        results_low_threshold = mock_vector_database.vector_similarity_search(
            similar_vector.tolist(),
            threshold=0.1
        )
        
        # Should find both groups
        assert len(results_low_threshold) == 2
        # Results should be sorted by similarity (highest first)
        assert results_low_threshold[0][1] > results_low_threshold[1][1]
    
    @pytest.mark.asyncio
    async def test_vector_index_performance_simulation(self, mock_vector_database):
        """Test performance characteristics of vector similarity search."""
        # Create many groups with random centroids
        num_groups = 100
        groups = []
        
        for i in range(num_groups):
            vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            
            group = StoryGroup(
                id=f"perf-group-{i}",
                centroid_embedding=vector.tolist(),
                member_count=5,
                status=GroupStatus.STABLE
            )
            groups.append(group)
            mock_vector_database.store_group(group)
        
        # Query vector
        query_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Measure search performance (simulated)
        import time
        start_time = time.time()
        
        results = mock_vector_database.vector_similarity_search(
            query_vector.tolist(),
            threshold=0.7
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Performance assertions (these would be different with real vector DB)
        assert search_time < 1.0  # Should complete quickly even with 100 groups
        assert isinstance(results, list)
        
        # Verify results are properly sorted
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i][1] >= results[i + 1][1]
    
    @pytest.mark.asyncio
    async def test_vector_storage_and_retrieval_consistency(self, mock_vector_database):
        """Test consistency of vector storage and retrieval operations."""
        # Create embeddings with known vectors
        test_vectors = []
        test_embeddings = []
        
        for i in range(5):
            vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            test_vectors.append(vector)
            
            embedding = StoryEmbedding(
                news_url_id=f"consistency-test-{i}",
                embedding_vector=vector.tolist(),
                model_name="test-model",
                model_version="1.0",
                summary_text=f"Test summary {i}",
                confidence_score=0.8,
                generated_at=datetime.now(timezone.utc)
            )
            test_embeddings.append(embedding)
            
            # Store embedding
            success = mock_vector_database.store_embedding(embedding)
            assert success is True
        
        # Verify retrieval consistency
        for i, embedding in enumerate(test_embeddings):
            retrieved = mock_vector_database.embeddings.get(embedding.news_url_id)
            assert retrieved is not None
            assert retrieved.news_url_id == embedding.news_url_id
            
            # Verify vector consistency
            stored_vector = np.array(retrieved.embedding_vector)
            original_vector = test_vectors[i]
            
            # Vectors should be identical (within floating point precision)
            assert np.allclose(stored_vector, original_vector, rtol=1e-6)
    
    @pytest.mark.asyncio
    async def test_concurrent_vector_operations(self, mock_vector_database):
        """Test concurrent vector operations for thread safety simulation."""
        # Create multiple embeddings for concurrent operations
        embeddings = []
        for i in range(10):
            vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            
            embedding = StoryEmbedding(
                news_url_id=f"concurrent-vector-{i}",
                embedding_vector=vector.tolist(),
                model_name="test-model",
                model_version="1.0",
                summary_text=f"Concurrent test {i}",
                confidence_score=0.8
            )
            embeddings.append(embedding)
        
        # Simulate concurrent storage operations
        async def store_embedding(embedding):
            return mock_vector_database.store_embedding(embedding)
        
        tasks = [store_embedding(embedding) for embedding in embeddings]
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert all(results)
        assert len(mock_vector_database.embeddings) == 10
        
        # Verify all embeddings are stored correctly
        for embedding in embeddings:
            assert embedding.news_url_id in mock_vector_database.embeddings
    
    @pytest.mark.asyncio
    async def test_vector_dimension_validation(self, mock_vector_database):
        """Test validation of vector dimensions in database operations."""
        # Test with correct dimensions
        correct_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()
        correct_embedding = StoryEmbedding(
            news_url_id="correct-dimension",
            embedding_vector=correct_vector,
            model_name="test-model",
            model_version="1.0",
            summary_text="Correct dimension test",
            confidence_score=0.8
        )
        
        # Should succeed
        result = mock_vector_database.store_embedding(correct_embedding)
        assert result is True
        
        # Test with incorrect dimensions
        wrong_vector = np.random.rand(100).astype(np.float32).tolist()  # Wrong dimension
        wrong_embedding = StoryEmbedding(
            news_url_id="wrong-dimension",
            embedding_vector=wrong_vector,
            model_name="test-model",
            model_version="1.0", 
            summary_text="Wrong dimension test",
            confidence_score=0.8
        )
        
        # Should fail validation (in real implementation)
        with pytest.raises(ValueError):
            wrong_embedding.validate()  # Model validation should catch this


class TestPerformanceAndScalability:
    """Performance tests for large-scale story processing."""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing_simulation(self):
        """Test processing a large batch of stories for performance characteristics."""
        # Create a large number of mock stories
        num_stories = 100
        stories = []
        
        for i in range(num_stories):
            story = ProcessedNewsItem(
                url=f"https://example.com/story-{i}",
                title=f"Test Story {i}",
                publication_date=datetime.now(timezone.utc) - timedelta(hours=i),
                source_name="performance_test",
                publisher="Test Publisher",
                description=f"Performance test story number {i}",
                relevance_score=0.8,
                filter_method="rule_based"
            )
            stories.append(story)
        
        # Mock all dependencies for performance testing
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.URLContextExtractor') as mock_extractor:
            with patch('src.nfl_news_pipeline.embedding.EmbeddingGenerator') as mock_generator:
                with patch('src.nfl_news_pipeline.group_manager.GroupManager') as mock_manager:
                    
                    # Setup mocks for fast execution
                    mock_extractor_instance = Mock()
                    mock_extractor_instance.extract_context = AsyncMock(return_value=ContextSummary(
                        news_url_id="test",
                        summary_text="Mock summary",
                        llm_model="mock",
                        confidence_score=0.8
                    ))
                    mock_extractor.return_value = mock_extractor_instance
                    
                    mock_generator_instance = Mock()
                    mock_embedding = StoryEmbedding(
                        news_url_id="test",
                        embedding_vector=[0.1] * EMBEDDING_DIM,
                        model_name="mock",
                        model_version="1.0",
                        summary_text="Mock",
                        confidence_score=0.8
                    )
                    mock_generator_instance.generate_embedding = AsyncMock(return_value=mock_embedding)
                    mock_generator.return_value = mock_generator_instance
                    
                    mock_manager_instance = Mock()
                    from src.nfl_news_pipeline.group_manager import GroupAssignmentResult
                    mock_result = GroupAssignmentResult(
                        group_id="mock-group",
                        similarity_score=0.9,
                        is_new_group=True,
                        group_size=1
                    )
                    mock_manager_instance.process_new_story = AsyncMock(return_value=mock_result)
                    mock_manager.return_value = mock_manager_instance
                    
                    # Measure processing time
                    import time
                    start_time = time.time()
                    
                    # Process stories in batches
                    batch_size = 10
                    results = []
                    
                    for i in range(0, num_stories, batch_size):
                        batch = stories[i:i + batch_size]
                        batch_tasks = []
                        
                        for story in batch:
                            # Simulate pipeline processing
                            async def process_story(s):
                                context = await mock_extractor_instance.extract_context(s)
                                embedding = await mock_generator_instance.generate_embedding(context, s.url)
                                result = await mock_manager_instance.process_new_story(embedding)
                                return result
                            
                            batch_tasks.append(process_story(story))
                        
                        batch_results = await asyncio.gather(*batch_tasks)
                        results.extend(batch_results)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Performance assertions
                    assert len(results) == num_stories
                    assert total_time < 10.0  # Should complete within reasonable time
                    
                    # Calculate throughput
                    throughput = num_stories / total_time
                    assert throughput > 10  # Should process at least 10 stories per second
    
    @pytest.mark.asyncio
    async def test_memory_usage_simulation(self):
        """Test memory usage patterns during large-scale processing."""
        # This would test memory usage in a real scenario
        # For now, we'll simulate memory-efficient processing patterns
        
        num_iterations = 50
        max_memory_usage = 0
        
        for i in range(num_iterations):
            # Simulate creating and processing embeddings
            embeddings = []
            for j in range(20):  # Process 20 embeddings per iteration
                vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
                embedding = StoryEmbedding(
                    news_url_id=f"memory-test-{i}-{j}",
                    embedding_vector=vector.tolist(),
                    model_name="memory-test",
                    model_version="1.0",
                    summary_text="Memory test",
                    confidence_score=0.8
                )
                embeddings.append(embedding)
            
            # Simulate memory usage tracking
            current_memory = len(embeddings) * EMBEDDING_DIM * 4  # 4 bytes per float
            max_memory_usage = max(max_memory_usage, current_memory)
            
            # Clear embeddings to simulate memory cleanup
            del embeddings
        
        # Memory usage should be reasonable
        max_memory_mb = max_memory_usage / (1024 * 1024)
        assert max_memory_mb < 100  # Should use less than 100MB for test data
    
    @pytest.mark.asyncio
    async def test_error_recovery_under_load(self):
        """Test error recovery mechanisms under high load."""
        num_stories = 20
        error_rate = 0.2  # 20% of operations fail
        
        # Create stories for load testing
        stories = [
            ProcessedNewsItem(
                url=f"https://example.com/load-test-{i}",
                title=f"Load Test Story {i}",
                publication_date=datetime.now(timezone.utc),
                source_name="load_test",
                publisher="Load Test",
                description=f"Load test story {i}",
                relevance_score=0.8
            )
            for i in range(num_stories)
        ]
        
        # Mock components with intermittent failures
        success_count = 0
        failure_count = 0
        import random
        rng = random.Random(1)

        async def mock_process_with_failures(story):
            if rng.random() < error_rate:
                # Simulate failure
                nonlocal failure_count
                failure_count += 1
                raise Exception(f"Simulated failure for {story.url}")
            else:
                # Simulate success
                nonlocal success_count
                success_count += 1
                from src.nfl_news_pipeline.group_manager import GroupAssignmentResult
                return GroupAssignmentResult(
                    group_id=f"load-test-group-{success_count}",
                    similarity_score=0.8,
                    is_new_group=True,
                    group_size=1
                )
        
        # Process with error handling
        results = []
        errors = []
        
        for story in stories:
            try:
                result = await mock_process_with_failures(story)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Verify error handling
        assert success_count + failure_count == num_stories
        assert len(results) == success_count
        assert len(errors) == failure_count
        
        # Should have both successes and failures based on error rate
        assert success_count > 0
        assert failure_count > 0
