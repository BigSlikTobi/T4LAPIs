"""Tests for story grouping pipeline integration."""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nfl_news_pipeline.orchestrator.pipeline import NFLNewsPipeline
from src.nfl_news_pipeline.models import (
    NewsItem, ProcessedNewsItem, FeedConfig, DefaultsConfig, PipelineConfig
)


class MockStorageManager:
    """Mock storage manager for testing pipeline integration."""
    
    def __init__(self):
        self.stored_items = []
        self.watermarks = {}
        self.ids_by_url = {}
        
    def check_duplicate_urls(self, urls):
        return {}
    
    def store_news_items(self, items):
        self.stored_items.extend(items)
        # Generate mock IDs for items
        ids_by_url = {item.url: f"id_{i}" for i, item in enumerate(items)}
        self.ids_by_url.update(ids_by_url)
        
        # Mock storage result
        result = Mock()
        result.inserted_count = len(items)
        result.updated_count = 0
        result.errors_count = 0
        result.ids_by_url = ids_by_url
        return result
    
    def get_watermark(self, source_name):
        return self.watermarks.get(source_name)
    
    def update_watermark(self, source_name, **kwargs):
        if 'last_processed_date' in kwargs:
            self.watermarks[source_name] = kwargs['last_processed_date']
        return True


class MockAuditLogger:
    """Mock audit logger for testing."""
    
    def __init__(self):
        self.events = []
        
    def log_fetch_start(self, source_name):
        self.events.append(("fetch_start", source_name))
        
    def log_fetch_end(self, source_name, **kwargs):
        self.events.append(("fetch_end", source_name, kwargs))
        
    def log_filter_summary(self, **kwargs):
        self.events.append(("filter_summary", kwargs))
        
    def log_store_summary(self, **kwargs):
        self.events.append(("store_summary", kwargs))
        
    def log_pipeline_summary(self, **kwargs):
        self.events.append(("pipeline_summary", kwargs))
        
    def log_event(self, event_type, **kwargs):
        self.events.append(("event", event_type, kwargs))
        
    def log_error(self, **kwargs):
        self.events.append(("error", kwargs))


class TestStoryGroupingPipelineIntegration(unittest.TestCase):
    """Test cases for story grouping integration with the main pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.storage = MockStorageManager()
        self.audit = MockAuditLogger()
        
        # Create mock news items
        self.test_items = [
            NewsItem(
                url=f"https://example.com/story{i}",
                title=f"Test Story {i}",
                publication_date=datetime.now(timezone.utc),
                source_name="test_source",
                publisher="Test Publisher",
                description=f"Description for story {i}",
                raw_metadata={}
            )
            for i in range(3)
        ]

    @patch('src.nfl_news_pipeline.orchestrator.pipeline.ConfigManager')
    def test_pipeline_without_story_grouping(self, mock_config_manager):
        """Test that pipeline works normally when story grouping is disabled."""
        # Setup config manager to return disabled story grouping
        mock_cm = Mock()
        mock_defaults = DefaultsConfig()
        mock_defaults.enable_story_grouping = False
        mock_cm.get_defaults.return_value = mock_defaults
        mock_cm.get_enabled_sources.return_value = []
        mock_config_manager.return_value = mock_cm
        
        # Create pipeline
        pipeline = NFLNewsPipeline(
            config_path="dummy.yaml",
            storage=self.storage,
            audit=self.audit
        )
        
        # Test that story grouping is not enabled
        self.assertFalse(pipeline._should_run_story_grouping())
        
        # Test that orchestrator is None when story grouping is disabled
        orchestrator = pipeline._get_story_grouping_orchestrator()
        self.assertIsNone(orchestrator)

    @patch.dict(os.environ, {'NEWS_PIPELINE_ENABLE_STORY_GROUPING': '1'})
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.ConfigManager')
    def test_pipeline_with_story_grouping_env_override(self, mock_config_manager):
        """Test that environment variable can enable story grouping."""
        # Setup config manager with story grouping disabled in config
        mock_cm = Mock()
        mock_defaults = DefaultsConfig()
        mock_defaults.enable_story_grouping = False
        mock_cm.get_defaults.return_value = mock_defaults
        mock_config_manager.return_value = mock_cm
        
        # Create pipeline
        pipeline = NFLNewsPipeline(
            config_path="dummy.yaml",
            storage=self.storage,
            audit=self.audit
        )
        
        # Test that story grouping is enabled via environment variable
        self.assertTrue(pipeline._should_run_story_grouping())

    @patch('src.nfl_news_pipeline.orchestrator.pipeline.ConfigManager')
    def test_pipeline_with_story_grouping_config_enabled(self, mock_config_manager):
        """Test that story grouping can be enabled via configuration."""
        # Setup config manager with story grouping enabled
        mock_cm = Mock()
        mock_defaults = DefaultsConfig()
        mock_defaults.enable_story_grouping = True
        mock_defaults.story_grouping_max_parallelism = 3
        mock_defaults.story_grouping_max_stories_per_run = 100
        mock_defaults.story_grouping_reprocess_existing = False
        mock_cm.get_defaults.return_value = mock_defaults
        mock_config_manager.return_value = mock_cm
        
        # Create pipeline
        pipeline = NFLNewsPipeline(
            config_path="dummy.yaml",
            storage=self.storage,
            audit=self.audit
        )
        
        # Test that story grouping is enabled
        self.assertTrue(pipeline._should_run_story_grouping())

    @patch('src.nfl_news_pipeline.orchestrator.pipeline.StoryGroupingOrchestrator')
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.GroupManager')
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.EmbeddingGenerator')
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.SimilarityCalculator')
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.URLContextExtractor')
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.EmbeddingErrorHandler')
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.ConfigManager')
    def test_story_grouping_orchestrator_initialization(self, 
                                                       mock_config_manager,
                                                       mock_error_handler,
                                                       mock_context_extractor,
                                                       mock_similarity_calculator,
                                                       mock_embedding_generator,
                                                       mock_group_manager,
                                                       mock_story_grouping_orchestrator):
        """Test that story grouping orchestrator is properly initialized."""
        # Setup config manager
        mock_cm = Mock()
        mock_defaults = DefaultsConfig()
        mock_defaults.enable_story_grouping = True
        mock_defaults.story_grouping_max_parallelism = 4
        mock_defaults.story_grouping_max_stories_per_run = 200
        mock_defaults.story_grouping_reprocess_existing = True
        mock_cm.get_defaults.return_value = mock_defaults
        mock_config_manager.return_value = mock_cm
        
        # Setup mocks
        mock_orchestrator_instance = Mock()
        mock_story_grouping_orchestrator.return_value = mock_orchestrator_instance
        
        # Create pipeline
        pipeline = NFLNewsPipeline(
            config_path="dummy.yaml",
            storage=self.storage,
            audit=self.audit
        )
        
        # Get orchestrator
        orchestrator = pipeline._get_story_grouping_orchestrator()
        
        # Verify orchestrator was created with correct settings
        self.assertIsNotNone(orchestrator)
        mock_story_grouping_orchestrator.assert_called_once()
        
        # Verify components were initialized
        mock_context_extractor.assert_called_once()
        mock_embedding_generator.assert_called_once()
        mock_similarity_calculator.assert_called_once()
        mock_error_handler.assert_called_once()
        mock_group_manager.assert_called_once_with(self.storage)

    @patch('asyncio.run')
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.ConfigManager')
    def test_run_story_grouping_for_items(self, mock_config_manager, mock_asyncio_run):
        """Test that story grouping is executed for stored items."""
        # Setup config manager
        mock_cm = Mock()
        mock_defaults = DefaultsConfig()
        mock_defaults.enable_story_grouping = True
        mock_cm.get_defaults.return_value = mock_defaults
        mock_config_manager.return_value = mock_cm
        
        # Setup mock orchestrator
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_metrics = Mock()
        mock_metrics.processed_stories = 2
        mock_metrics.new_groups_created = 1
        mock_metrics.existing_groups_updated = 1
        mock_result.metrics = mock_metrics
        mock_orchestrator.process_batch = AsyncMock(return_value=mock_result)
        
        # Create pipeline and set orchestrator
        pipeline = NFLNewsPipeline(
            config_path="dummy.yaml",
            storage=self.storage,
            audit=self.audit
        )
        pipeline._story_grouping_orchestrator = mock_orchestrator
        
        # Create test items
        processed_items = [
            ProcessedNewsItem(
                url="https://example.com/story1",
                title="Test Story 1",
                publication_date=datetime.now(timezone.utc),
                source_name="test_source",
                publisher="Test Publisher",
                relevance_score=0.9
            ),
            ProcessedNewsItem(
                url="https://example.com/story2",
                title="Test Story 2",
                publication_date=datetime.now(timezone.utc),
                source_name="test_source",
                publisher="Test Publisher",
                relevance_score=0.8
            )
        ]
        
        ids_by_url = {
            "https://example.com/story1": "id_1",
            "https://example.com/story2": "id_2"
        }
        
        # Mock asyncio.run to call the coroutine directly
        def run_mock(coro):
            return mock_result
        mock_asyncio_run.side_effect = run_mock
        
        # Run story grouping
        pipeline._run_story_grouping_for_items(processed_items, ids_by_url)
        
        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()
        
        # Verify audit logging
        story_grouping_events = [
            event for event in self.audit.events 
            if len(event) >= 2 and event[1] == "story_grouping"
        ]
        self.assertTrue(len(story_grouping_events) > 0)

    @patch.dict(os.environ, {'NEWS_PIPELINE_DEBUG': '1'})
    @patch('src.nfl_news_pipeline.orchestrator.pipeline.ConfigManager')
    def test_story_grouping_debug_output(self, mock_config_manager):
        """Test that debug output is generated when debug mode is enabled."""
        # Setup config manager
        mock_cm = Mock()
        mock_defaults = DefaultsConfig()
        mock_defaults.enable_story_grouping = True
        mock_cm.get_defaults.return_value = mock_defaults
        mock_config_manager.return_value = mock_cm
        
        # Capture print output
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Create pipeline
            pipeline = NFLNewsPipeline(
                config_path="dummy.yaml",
                storage=self.storage,
                audit=self.audit
            )
            
            # Test items without IDs (should return early)
            processed_items = [
                ProcessedNewsItem(
                    url="https://example.com/story1",
                    title="Test Story 1",
                    publication_date=datetime.now(timezone.utc),
                    source_name="test_source",
                    publisher="Test Publisher"
                )
            ]
            
            ids_by_url = {}  # Empty mapping
            
            # This should return early and not produce debug output about processing
            pipeline._run_story_grouping_for_items(processed_items, ids_by_url)
            
            # No debug output should be produced since no items have IDs
            output = captured_output.getvalue()
            self.assertEqual(output, "")
            
        finally:
            sys.stdout = old_stdout

    def test_story_grouping_error_handling(self):
        """Test that story grouping errors are handled gracefully."""
        # Create pipeline with mock orchestrator that raises an exception
        pipeline = NFLNewsPipeline(
            config_path="dummy.yaml",
            storage=self.storage,
            audit=self.audit
        )
        
        # Mock orchestrator initialization to return None (failure case)
        with patch.object(pipeline, '_get_story_grouping_orchestrator', return_value=None):
            # This should not raise an exception
            processed_items = [
                ProcessedNewsItem(
                    url="https://example.com/story1",
                    title="Test Story 1",
                    publication_date=datetime.now(timezone.utc),
                    source_name="test_source",
                    publisher="Test Publisher"
                )
            ]
            
            ids_by_url = {"https://example.com/story1": "id_1"}
            
            # Should complete without error
            pipeline._run_story_grouping_for_items(processed_items, ids_by_url)

    def test_story_grouping_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # Create pipeline
        pipeline = NFLNewsPipeline(
            config_path="dummy.yaml",
            storage=self.storage,
            audit=self.audit
        )
        
        # Mock import error
        with patch('src.nfl_news_pipeline.orchestrator.pipeline.StoryGroupingOrchestrator', 
                  side_effect=ImportError("Module not found")):
            
            orchestrator = pipeline._get_story_grouping_orchestrator()
            self.assertIsNone(orchestrator)
            
            # Verify error was logged
            error_events = [event for event in self.audit.events if event[0] == "error"]
            self.assertTrue(len(error_events) > 0)


if __name__ == '__main__':
    unittest.main()