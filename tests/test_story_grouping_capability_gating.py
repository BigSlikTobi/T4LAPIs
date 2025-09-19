"""Tests for story grouping capability-based gating logic."""

import unittest
from unittest.mock import Mock, patch
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nfl_news_pipeline.orchestrator.pipeline import NFLNewsPipeline
from src.nfl_news_pipeline.storage.protocols import (
    StoryGroupingCapable, has_story_grouping_capability, get_grouping_client
)
from src.nfl_news_pipeline.models import FeedConfig, DefaultsConfig


class MockAuditLogger:
    """Mock audit logger for testing."""
    
    def __init__(self):
        self.events = []
        
    def log_event(self, event_type: str, message: str, data: dict = None):
        self.events.append({
            'event_type': event_type,
            'message': message,
            'data': data or {}
        })
        
    def log_error(self, context: str, exc: Exception):
        self.events.append({
            'event_type': 'error',
            'context': context,
            'exception': str(exc)
        })


class StorageWithClient:
    """Mock storage with traditional .client attribute."""
    
    def __init__(self, client=None):
        self.client = client or Mock(name="supabase_client")
        
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        result = Mock()
        result.inserted_count = len(items)
        result.updated_count = 0
        result.errors_count = 0
        result.ids_by_url = {item.url: f"id_{i}" for i, item in enumerate(items)}
        return result
        
    def get_watermark(self, source_name):
        return None
        
    def update_watermark(self, source_name, **kwargs):
        return True


class StorageWithCapability(StoryGroupingCapable):
    """Mock storage that implements StoryGroupingCapable protocol."""
    
    def __init__(self, client=None):
        self._client = client or Mock(name="supabase_client")
        
    def get_grouping_client(self) -> Any:
        return self._client
        
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        result = Mock()
        result.inserted_count = len(items)
        result.updated_count = 0
        result.errors_count = 0
        result.ids_by_url = {item.url: f"id_{i}" for i, item in enumerate(items)}
        return result
        
    def get_watermark(self, source_name):
        return None
        
    def update_watermark(self, source_name, **kwargs):
        return True


class StorageWithoutCapability:
    """Mock storage without any grouping capability."""
    
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        result = Mock()
        result.inserted_count = len(items)
        result.updated_count = 0
        result.errors_count = 0
        result.ids_by_url = {item.url: f"id_{i}" for i, item in enumerate(items)}
        return result
        
    def get_watermark(self, source_name):
        return None
        
    def update_watermark(self, source_name, **kwargs):
        return True


class StorageWithMethodOnly:
    """Mock storage with get_grouping_client method but no client attribute."""
    
    def __init__(self, client=None):
        self._client = client or Mock(name="supabase_client")
        
    def get_grouping_client(self) -> Any:
        return self._client
        
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        result = Mock()
        result.inserted_count = len(items)
        result.updated_count = 0
        result.errors_count = 0
        result.ids_by_url = {item.url: f"id_{i}" for i, item in enumerate(items)}
        return result
        
    def get_watermark(self, source_name):
        return None
        
    def update_watermark(self, source_name, **kwargs):
        return True


class TestCapabilityDetection(unittest.TestCase):
    """Test capability detection functions."""
    
    def test_storage_with_client_attribute(self):
        """Storage with .client attribute should be capable."""
        storage = StorageWithClient()
        self.assertTrue(has_story_grouping_capability(storage))
        client = get_grouping_client(storage)
        self.assertIsNotNone(client)
        self.assertEqual(client, storage.client)
    
    def test_storage_with_protocol_implementation(self):
        """Storage implementing StoryGroupingCapable should be capable."""
        storage = StorageWithCapability()
        self.assertTrue(has_story_grouping_capability(storage))
        client = get_grouping_client(storage)
        self.assertIsNotNone(client)
    
    def test_storage_with_method_only(self):
        """Storage with get_grouping_client method should be capable."""
        storage = StorageWithMethodOnly()
        self.assertTrue(has_story_grouping_capability(storage))
        client = get_grouping_client(storage)
        self.assertIsNotNone(client)
    
    def test_storage_without_capability(self):
        """Storage without any capability should not be capable."""
        storage = StorageWithoutCapability()
        self.assertFalse(has_story_grouping_capability(storage))
        
        with self.assertRaises(ValueError) as cm:
            get_grouping_client(storage)
        self.assertIn("does not support story grouping capabilities", str(cm.exception))
    
    def test_storage_with_none_client(self):
        """Storage with None client should not be capable."""
        storage = StorageWithClient()
        storage.client = None  # Explicitly set to None after creation
        self.assertFalse(has_story_grouping_capability(storage))


class TestStoryGroupingGating(unittest.TestCase):
    """Test story grouping gating logic in orchestrator."""
    
    def setUp(self):
        # Clear environment variables
        for key in ['NEWS_PIPELINE_ENABLE_STORY_GROUPING', 'NEWS_PIPELINE_DISABLE_STORY_GROUPING']:
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        # Clean up environment variables
        for key in ['NEWS_PIPELINE_ENABLE_STORY_GROUPING', 'NEWS_PIPELINE_DISABLE_STORY_GROUPING']:
            if key in os.environ:
                del os.environ[key]
    
    def test_enabled_with_supabase_storage(self):
        """Story grouping should be enabled with traditional Supabase storage."""
        storage = StorageWithClient()
        audit_logger = MockAuditLogger()
        
        # Mock config manager
        config_manager = Mock()
        defaults = DefaultsConfig()
        defaults.enable_story_grouping = True
        config_manager.get_defaults.return_value = defaults
        
        pipeline = NFLNewsPipeline(config_path="fake_config.yaml", storage=storage, audit=audit_logger)
        pipeline.cm = config_manager
        
        # Should be enabled
        self.assertTrue(pipeline._should_run_story_grouping())
        
        # No error events should be logged
        error_events = [e for e in audit_logger.events if e['event_type'] == 'story_grouping_disabled']
        self.assertEqual(len(error_events), 0)
    
    def test_enabled_with_capability_storage(self):
        """Story grouping should be enabled with protocol-implementing storage."""
        storage = StorageWithCapability()
        audit_logger = MockAuditLogger()
        
        # Mock config manager
        config_manager = Mock()
        defaults = DefaultsConfig()
        defaults.enable_story_grouping = True
        config_manager.get_defaults.return_value = defaults
        
        pipeline = NFLNewsPipeline(config_path="fake_config.yaml", storage=storage, audit=audit_logger)
        pipeline.cm = config_manager
        
        # Should be enabled
        self.assertTrue(pipeline._should_run_story_grouping())
        
        # No error events should be logged
        error_events = [e for e in audit_logger.events if e['event_type'] == 'story_grouping_disabled']
        self.assertEqual(len(error_events), 0)
    
    def test_enabled_with_method_only_storage(self):
        """Story grouping should be enabled with storage having get_grouping_client method."""
        storage = StorageWithMethodOnly()
        audit_logger = MockAuditLogger()
        
        # Mock config manager
        config_manager = Mock()
        defaults = DefaultsConfig()
        defaults.enable_story_grouping = True
        config_manager.get_defaults.return_value = defaults
        
        pipeline = NFLNewsPipeline(config_path="fake_config.yaml", storage=storage, audit=audit_logger)
        pipeline.cm = config_manager
        
        # Should be enabled
        self.assertTrue(pipeline._should_run_story_grouping())
    
    def test_disabled_with_incapable_storage(self):
        """Story grouping should be disabled with storage lacking capabilities."""
        storage = StorageWithoutCapability()
        audit_logger = MockAuditLogger()
        
        # Mock config manager
        config_manager = Mock()
        defaults = DefaultsConfig()
        defaults.enable_story_grouping = True
        config_manager.get_defaults.return_value = defaults
        
        pipeline = NFLNewsPipeline(config_path="fake_config.yaml", storage=storage, audit=audit_logger)
        pipeline.cm = config_manager
        
        # Should be disabled
        self.assertFalse(pipeline._should_run_story_grouping())
        
        # Should log appropriate event
        error_events = [e for e in audit_logger.events if e['event_type'] == 'story_grouping_disabled']
        self.assertEqual(len(error_events), 1)
        self.assertIn("storage lacks required capabilities", error_events[0]['message'])
        self.assertEqual(error_events[0]['data']['storage_type'], 'StorageWithoutCapability')
    
    def test_disabled_by_config(self):
        """Story grouping should be disabled when config disables it."""
        storage = StorageWithClient()
        audit_logger = MockAuditLogger()
        
        # Mock config manager with grouping disabled
        config_manager = Mock()
        defaults = DefaultsConfig()
        defaults.enable_story_grouping = False
        config_manager.get_defaults.return_value = defaults
        
        pipeline = NFLNewsPipeline(config_path="fake_config.yaml", storage=storage, audit=audit_logger)
        pipeline.cm = config_manager
        
        # Should be disabled
        self.assertFalse(pipeline._should_run_story_grouping())
    
    def test_force_disabled_by_env_var(self):
        """Story grouping should be disabled when environment variable forces it."""
        os.environ['NEWS_PIPELINE_DISABLE_STORY_GROUPING'] = '1'
        
        storage = StorageWithClient()
        audit_logger = MockAuditLogger()
        
        # Mock config manager with grouping enabled
        config_manager = Mock()
        defaults = DefaultsConfig()
        defaults.enable_story_grouping = True
        config_manager.get_defaults.return_value = defaults
        
        pipeline = NFLNewsPipeline(config_path="fake_config.yaml", storage=storage, audit=audit_logger)
        pipeline.cm = config_manager
        
        # Should be disabled due to environment variable
        self.assertFalse(pipeline._should_run_story_grouping())
    
    def test_force_enabled_by_env_var(self):
        """Story grouping should be enabled when environment variable forces it."""
        os.environ['NEWS_PIPELINE_ENABLE_STORY_GROUPING'] = '1'
        
        storage = StorageWithClient()
        audit_logger = MockAuditLogger()
        
        # Mock config manager with grouping disabled
        config_manager = Mock()
        defaults = DefaultsConfig()
        defaults.enable_story_grouping = False
        config_manager.get_defaults.return_value = defaults
        
        pipeline = NFLNewsPipeline(config_path="fake_config.yaml", storage=storage, audit=audit_logger)
        pipeline.cm = config_manager
        
        # Should be enabled due to environment variable override
        self.assertTrue(pipeline._should_run_story_grouping())


if __name__ == '__main__':
    unittest.main()