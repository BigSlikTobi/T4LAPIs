"""Test story grouping gating logic improvements for Issue 52."""

from __future__ import annotations

import os
import pytest
from unittest.mock import Mock
from typing import Protocol

from src.nfl_news_pipeline.orchestrator.pipeline import NFLNewsPipeline
from src.nfl_news_pipeline.models import FeedConfig, DefaultsConfig


class StoryGroupingStorage(Protocol):
    """Protocol defining the storage interface required for story grouping."""
    
    def persist_group(self, *args, **kwargs) -> any: ...
    def persist_context(self, *args, **kwargs) -> any: ...
    def persist_embedding(self, *args, **kwargs) -> any: ...


class MockStorageWithClient:
    """Mock storage that has a .client attribute (traditional Supabase storage)."""
    
    def __init__(self):
        self.client = Mock()  # Simulates Supabase client
        
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        return Mock(inserted_count=len(items), updated_count=0, errors_count=0, ids_by_url={})


class MockStorageWithCapabilities:
    """Mock storage without .client but with required story grouping capabilities."""
    
    def __init__(self):
        # No .client attribute
        pass
        
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        return Mock(inserted_count=len(items), updated_count=0, errors_count=0, ids_by_url={})
        
    def persist_group(self, *args, **kwargs):
        """Required for story grouping."""
        return True
        
    def persist_context(self, *args, **kwargs):
        """Required for story grouping."""
        return True
        
    def persist_embedding(self, *args, **kwargs):
        """Required for story grouping."""
        return True


class MockStorageWithoutCapabilities:
    """Mock storage without .client and without story grouping capabilities."""
    
    def __init__(self):
        # No .client attribute
        pass
        
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        return Mock(inserted_count=len(items), updated_count=0, errors_count=0, ids_by_url={})


class MockConfigManager:
    """Mock config manager for testing."""
    
    def __init__(self, enable_story_grouping=True):
        self._defaults = DefaultsConfig(enable_story_grouping=enable_story_grouping)
        
    def get_defaults(self):
        return self._defaults
        
    def get_enabled_sources(self):
        return []


def test_story_grouping_enabled_with_client_attribute(monkeypatch):
    """Test that story grouping is enabled when storage has .client attribute."""
    monkeypatch.delenv("NEWS_PIPELINE_DISABLE_STORY_GROUPING", raising=False)
    monkeypatch.delenv("NEWS_PIPELINE_ENABLE_STORY_GROUPING", raising=False)
    
    storage = MockStorageWithClient()
    
    pipeline = NFLNewsPipeline("", storage=storage)
    # Mock the config manager
    pipeline.cm = MockConfigManager(enable_story_grouping=True)
    
    # Should be enabled because storage has .client
    assert pipeline._should_run_story_grouping() is True


def test_story_grouping_enabled_with_capabilities_no_client(monkeypatch):
    """Test that story grouping is enabled when storage has capabilities but no .client."""
    monkeypatch.delenv("NEWS_PIPELINE_DISABLE_STORY_GROUPING", raising=False)
    monkeypatch.delenv("NEWS_PIPELINE_ENABLE_STORY_GROUPING", raising=False)
    
    storage = MockStorageWithCapabilities()
    
    pipeline = NFLNewsPipeline("", storage=storage)
    # Mock the config manager
    pipeline.cm = MockConfigManager(enable_story_grouping=True)
    
    # Should be enabled because storage has required capabilities
    # This test will fail with current implementation, pass after fix
    assert pipeline._should_run_story_grouping() is True


def test_story_grouping_disabled_without_capabilities(monkeypatch):
    """Test that story grouping is disabled when storage lacks required capabilities."""
    monkeypatch.delenv("NEWS_PIPELINE_DISABLE_STORY_GROUPING", raising=False)
    monkeypatch.delenv("NEWS_PIPELINE_ENABLE_STORY_GROUPING", raising=False)
    
    storage = MockStorageWithoutCapabilities()
    
    pipeline = NFLNewsPipeline("", storage=storage)
    # Mock the config manager
    pipeline.cm = MockConfigManager(enable_story_grouping=True)
    
    # Should be disabled because storage lacks capabilities
    assert pipeline._should_run_story_grouping() is False


def test_story_grouping_disabled_by_environment_variable(monkeypatch):
    """Test that environment variable override works regardless of storage capabilities."""
    monkeypatch.setenv("NEWS_PIPELINE_DISABLE_STORY_GROUPING", "true")
    
    storage = MockStorageWithClient()
    
    pipeline = NFLNewsPipeline("", storage=storage)
    # Mock the config manager
    pipeline.cm = MockConfigManager(enable_story_grouping=True)
    
    # Should be disabled due to environment variable
    assert pipeline._should_run_story_grouping() is False


def test_story_grouping_enabled_by_environment_variable(monkeypatch):
    """Test that environment variable can force enable story grouping."""
    monkeypatch.setenv("NEWS_PIPELINE_ENABLE_STORY_GROUPING", "true")
    
    storage = MockStorageWithCapabilities()
    
    pipeline = NFLNewsPipeline("", storage=storage)
    # Mock the config manager
    pipeline.cm = MockConfigManager(enable_story_grouping=False)
    
    # Should be enabled due to environment variable override
    assert pipeline._should_run_story_grouping() is True


if __name__ == "__main__":
    # Run the tests to demonstrate current vs expected behavior
    pytest.main([__file__, "-v"])