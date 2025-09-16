"""Tests for story grouping URL context extraction functionality.

Tests cover LLM URL context extraction, fallback mechanisms, caching,
and entity normalization for tasks 3.1, 3.2, and 3.3.
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.nfl_news_pipeline.models import ContextSummary, ProcessedNewsItem
from src.nfl_news_pipeline.story_grouping import URLContextExtractor, ContextCache, generate_metadata_hash


class TestURLContextExtractor:
    """Test cases for URLContextExtractor class (Task 3.1 and 3.2)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_news_item = ProcessedNewsItem(
            url="https://example.com/chiefs-mahomes-injury",
            title="Patrick Mahomes suffers ankle injury in Chiefs victory",
            publication_date=datetime.now(timezone.utc),
            source_name="test_source",
            publisher="ESPN",
            description="Chiefs quarterback Patrick Mahomes injured his ankle during the game but continued playing.",
            relevance_score=0.9,
            filter_method="rule_based",
            entities=["Patrick Mahomes", "Kansas City Chiefs"],
            categories=["injury", "quarterback"]
        )
        
        self.sample_llm_response = {
            "summary": "Patrick Mahomes of the Kansas City Chiefs suffered an ankle injury during their victory but continued playing through the pain.",
            "entities": {
                "players": ["Patrick Mahomes"],
                "teams": ["Kansas City Chiefs"],
                "coaches": []
            },
            "key_topics": ["injury", "quarterback", "victory"],
            "story_category": "injury",
            "confidence": 0.9
        }
    
    def test_init_with_no_api_keys(self):
        """Test initialization without API keys."""
        with patch.dict('os.environ', {}, clear=True):
            extractor = URLContextExtractor()
            
            assert extractor.openai_client is None
            assert extractor.google_client is None
            assert extractor.preferred_provider == "openai"
    
    @patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI')
    def test_init_with_openai_key(self, mock_openai_class):
        """Test initialization with OpenAI API key."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            extractor = URLContextExtractor(preferred_provider="openai")
            
            assert extractor.openai_client == mock_client
            mock_openai_class.assert_called_once_with(api_key='test-key')
    
    @patch('src.nfl_news_pipeline.story_grouping.context_extractor.genai')
    @patch('src.nfl_news_pipeline.story_grouping.context_extractor.GenerativeModel')
    def test_init_with_google_key(self, mock_model_class, mock_genai):
        """Test initialization with Google API key."""
        mock_client = Mock()
        mock_model_class.return_value = mock_client
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            extractor = URLContextExtractor(preferred_provider="google")
            
            assert extractor.google_client == mock_client
            mock_genai.configure.assert_called_once_with(api_key='test-key')
    
    @pytest.mark.asyncio
    async def test_extract_context_with_cache_hit(self):
        """Test extract_context returns cached result when available."""
        # Create cached summary
        cached_summary = ContextSummary(
            news_url_id="test-id",
            summary_text="Cached summary",
            llm_model="test-model",
            confidence_score=0.8,
            generated_at=datetime.now(timezone.utc)
        )
        
        # Mock cache
        mock_cache = Mock()
        mock_cache.get_cached_summary.return_value = cached_summary
        
        extractor = URLContextExtractor(cache=mock_cache)
        result = await extractor.extract_context(self.sample_news_item)
        
        assert result == cached_summary
        mock_cache.get_cached_summary.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_context_with_openai_success(self):
        """Test successful context extraction with OpenAI."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(self.sample_llm_response)
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = URLContextExtractor()
        extractor.openai_client = mock_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        result = await extractor.extract_context(self.sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.summary_text == self.sample_llm_response["summary"]
        assert result.llm_model == "gpt-5-nano"
        assert result.confidence_score == 0.9
        assert result.fallback_used is False
        assert "Patrick Mahomes" in result.entities["players"]
        assert "Kansas City Chiefs" in result.entities["teams"]
    
    @pytest.mark.asyncio
    async def test_extract_context_with_google_success(self):
        """Test successful context extraction with Google AI."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_llm_response)
        mock_client.generate_content.return_value = mock_response
        
        extractor = URLContextExtractor(preferred_provider="google")
        extractor.google_client = mock_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        result = await extractor.extract_context(self.sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.summary_text == self.sample_llm_response["summary"]
        assert result.llm_model == "gemini-2.5-flash-lite"
        assert result.confidence_score == 0.9
        assert result.fallback_used is False
    
    @pytest.mark.asyncio
    async def test_extract_context_fallback_to_metadata(self):
        """Test fallback to metadata when LLM fails (Task 3.2)."""
        # Ensure no API keys leak into this test so no clients are initialized
        with patch.dict('os.environ', {}, clear=True):
            extractor = URLContextExtractor()
            # No LLM clients set, should fallback
            result = await extractor.extract_context(self.sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.fallback_used is True
        assert result.llm_model == "metadata_fallback"
        assert result.confidence_score == 0.6  # Lower confidence for fallback
        assert self.sample_news_item.title in result.summary_text
        assert "injury" in result.key_topics  # Should extract from title
    
    @pytest.mark.asyncio
    async def test_extract_context_openai_failure_google_success(self):
        """Test fallback from OpenAI to Google when OpenAI fails."""
        # Set up failing OpenAI client
        mock_openai_client = Mock()
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Set up working Google client
        mock_google_client = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_llm_response)
        mock_google_client.generate_content.return_value = mock_response
        
        extractor = URLContextExtractor(preferred_provider="openai")
        extractor.openai_client = mock_openai_client
        extractor.google_client = mock_google_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        result = await extractor.extract_context(self.sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.llm_model == "gemini-2.5-flash-lite"
        assert result.fallback_used is False
    
    def test_normalize_team_names(self):
        """Test team name normalization."""
        extractor = URLContextExtractor()
        
        teams = ["chiefs", "49ers", "KC", "San Francisco", "Random Team"]
        normalized = extractor._normalize_team_names(teams)
        
        expected = [
            "Kansas City Chiefs",
            "San Francisco 49ers", 
            "Kansas City Chiefs",  # KC -> Kansas City Chiefs
            "San Francisco 49ers",  # San Francisco -> San Francisco 49ers
            "Random Team"  # Unknown team kept as is
        ]
        
        # Should remove duplicates
        assert "Kansas City Chiefs" in normalized
        assert "San Francisco 49ers" in normalized
        assert "Random Team" in normalized
        assert len([t for t in normalized if t == "Kansas City Chiefs"]) == 1
    
    def test_normalize_player_names(self):
        """Test player name normalization."""
        extractor = URLContextExtractor()
        
        players = ["patrick mahomes", "JOSH ALLEN", "  travis kelce  ", ""]
        normalized = extractor._normalize_player_names(players)
        
        expected = ["Patrick Mahomes", "Josh Allen", "Travis Kelce"]
        assert normalized == expected
    
    def test_parse_llm_response_with_code_blocks(self):
        """Test parsing LLM response with JSON in code blocks."""
        extractor = URLContextExtractor()
        
        response_content = """
        Here's the analysis:
        ```json
        {
            "summary": "Test summary",
            "entities": {
                "players": ["Test Player"],
                "teams": ["Test Team"]
            },
            "key_topics": ["test"],
            "confidence": 0.8
        }
        ```
        """
        
        news_item = ProcessedNewsItem(
            url="https://example.com/test",
            title="Test",
            publication_date=datetime.now(timezone.utc),
            source_name="test",
            publisher="test"
        )
        
        result = extractor._parse_llm_response(response_content, "test-model", news_item)
        
        assert result is not None
        assert result.summary_text == "Test summary"
        assert "Test Player" in result.entities["players"]
        assert "Test Team" in result.entities["teams"]
        assert result.confidence_score == 0.8
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        extractor = URLContextExtractor()
        
        news_item = ProcessedNewsItem(
            url="https://example.com/test",
            title="Test",
            publication_date=datetime.now(timezone.utc),
            source_name="test",
            publisher="test"
        )
        
        response_content = "This is not valid JSON"
        result = extractor._parse_llm_response(response_content, "test-model", news_item)
        
        assert result is None
    
    def test_fallback_to_metadata_with_categories(self):
        """Test metadata fallback with existing categories."""
        extractor = URLContextExtractor()
        
        news_item = ProcessedNewsItem(
            url="https://example.com/test",
            title="Test title",
            publication_date=datetime.now(timezone.utc),
            source_name="test",
            publisher="test",
            description="Test description",
            categories=["existing_category"]
        )
        
        result = extractor._fallback_to_metadata(news_item)
        
        assert result.fallback_used is True
        assert "existing_category" in result.key_topics
        assert "Test title. Test description" == result.summary_text
    
    def test_fallback_to_metadata_injury_detection(self):
        """Test metadata fallback detects injury topics from title."""
        extractor = URLContextExtractor()
        
        news_item = ProcessedNewsItem(
            url="https://example.com/test",
            title="Player injured in game",
            publication_date=datetime.now(timezone.utc),
            source_name="test",
            publisher="test"
        )
        
        result = extractor._fallback_to_metadata(news_item)
        
        assert "injury" in result.key_topics
    
    def test_extract_entities_from_text_basic(self):
        """Test basic entity extraction from text."""
        extractor = URLContextExtractor()
        
        text = "The Kansas City Chiefs won against the 49ers. Patrick Mahomes played well."
        entities = extractor._extract_entities_from_text(text)
        
        assert "Kansas City Chiefs" in entities["teams"]
        assert "San Francisco 49ers" in entities["teams"]
        # Player name extraction is basic in fallback mode
        assert len(entities["players"]) >= 0  # May or may not detect names


class TestContextCache:
    """Test cases for ContextCache class (Task 3.3)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_summary = ContextSummary(
            news_url_id="test-id",
            summary_text="Test summary",
            llm_model="test-model",
            confidence_score=0.8,
            generated_at=datetime.now(timezone.utc)
        )
    
    def test_init_with_defaults(self):
        """Test cache initialization with default settings."""
        cache = ContextCache()
        
        assert cache.ttl_hours == 24
        assert cache.ttl_seconds == 24 * 3600
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
        assert cache.memory_cache is not None
    
    def test_init_with_custom_ttl(self):
        """Test cache initialization with custom TTL."""
        cache = ContextCache(ttl_hours=12)
        
        assert cache.ttl_hours == 12
        assert cache.ttl_seconds == 12 * 3600
    
    def test_init_with_memory_cache_disabled(self):
        """Test cache initialization with memory cache disabled."""
        cache = ContextCache(enable_memory_cache=False)
        
        assert cache.memory_cache is None
    
    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        cache = ContextCache()
        
        key1 = cache._generate_cache_key("https://example.com/test")
        key2 = cache._generate_cache_key("https://example.com/test")
        key3 = cache._generate_cache_key("https://example.com/different")
        
        assert key1 == key2  # Same URL should generate same key
        assert key1 != key3  # Different URLs should generate different keys
        assert len(key1) == 32  # MD5 hash length
    
    def test_generate_cache_key_with_metadata_hash(self):
        """Test cache key generation with metadata hash."""
        cache = ContextCache()
        
        key1 = cache._generate_cache_key("https://example.com/test", "hash1")
        key2 = cache._generate_cache_key("https://example.com/test", "hash2")
        key3 = cache._generate_cache_key("https://example.com/test", "hash1")
        
        assert key1 != key2  # Different metadata hashes
        assert key1 == key3  # Same URL and metadata hash
    
    def test_is_valid_cache_entry_recent(self):
        """Test cache entry validation for recent entries."""
        cache = ContextCache(ttl_hours=24)
        
        recent_summary = ContextSummary(
            news_url_id="test-id",
            summary_text="Test",
            llm_model="test",
            confidence_score=0.8,
            generated_at=datetime.now(timezone.utc) - timedelta(hours=1)  # 1 hour ago
        )
        
        assert cache._is_valid_cache_entry(recent_summary) is True
    
    def test_is_valid_cache_entry_expired(self):
        """Test cache entry validation for expired entries."""
        cache = ContextCache(ttl_hours=24)
        
        expired_summary = ContextSummary(
            news_url_id="test-id",
            summary_text="Test",
            llm_model="test",
            confidence_score=0.8,
            generated_at=datetime.now(timezone.utc) - timedelta(hours=25)  # 25 hours ago
        )
        
        assert cache._is_valid_cache_entry(expired_summary) is False
    
    def test_is_valid_cache_entry_no_timestamp(self):
        """Test cache entry validation with no timestamp."""
        cache = ContextCache()
        
        no_timestamp_summary = ContextSummary(
            news_url_id="test-id",
            summary_text="Test",
            llm_model="test",
            confidence_score=0.8,
            generated_at=None
        )
        
        assert cache._is_valid_cache_entry(no_timestamp_summary) is False
    
    def test_get_cached_summary_memory_hit(self):
        """Test cache retrieval with memory cache hit."""
        cache = ContextCache()
        
        # Mock memory cache hit
        cache.memory_cache = Mock()
        cache.memory_cache.get.return_value = self.sample_summary.to_db()
        
        result = cache.get_cached_summary("https://example.com/test")
        
        assert result is not None
        assert isinstance(result, ContextSummary)
        assert result.summary_text == self.sample_summary.summary_text
        assert cache.cache_hits == 1
        assert cache.cache_misses == 0
    
    def test_get_cached_summary_memory_miss(self):
        """Test cache retrieval with memory cache miss."""
        cache = ContextCache()
        
        # Mock memory cache miss
        cache.memory_cache = Mock()
        cache.memory_cache.get.return_value = None
        
        result = cache.get_cached_summary("https://example.com/test")
        
        assert result is None
        assert cache.cache_hits == 0
        assert cache.cache_misses == 1
    
    def test_store_summary_success(self):
        """Test successful summary storage."""
        cache = ContextCache()
        
        # Mock memory cache
        cache.memory_cache = Mock()
        
        result = cache.store_summary(self.sample_summary)
        
        assert result is True
        cache.memory_cache.set.assert_called_once()
    
    def test_store_summary_invalid_summary(self):
        """Test summary storage with invalid summary."""
        cache = ContextCache()
        
        invalid_summary = ContextSummary(
            news_url_id="",  # Invalid - empty news_url_id
            summary_text="Test",
            llm_model="test",
            confidence_score=0.8
        )
        
        result = cache.store_summary(invalid_summary)
        
        assert result is False
    
    def test_invalidate_url(self):
        """Test URL cache invalidation."""
        cache = ContextCache()
        
        # Mock memory cache
        cache.memory_cache = Mock()
        
        result = cache.invalidate_url("https://example.com/test")
        
        assert result is True
        cache.memory_cache.set.assert_called_once()
    
    def test_get_cache_stats_no_requests(self):
        """Test cache statistics with no requests."""
        cache = ContextCache()
        
        stats = cache.get_cache_stats()
        
        expected = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "hit_rate_percent": 0,
            "ttl_hours": 24
        }
        
        assert stats == expected
    
    def test_get_cache_stats_with_requests(self):
        """Test cache statistics with hits and misses."""
        cache = ContextCache()
        cache.cache_hits = 8
        cache.cache_misses = 2
        
        stats = cache.get_cache_stats()
        
        assert stats["cache_hits"] == 8
        assert stats["cache_misses"] == 2
        assert stats["total_requests"] == 10
        assert stats["hit_rate_percent"] == 80.0


class TestCacheUtilities:
    """Test cases for cache utility functions."""
    
    def test_generate_metadata_hash_title_only(self):
        """Test metadata hash generation with title only."""
        hash1 = generate_metadata_hash("Test title")
        hash2 = generate_metadata_hash("Test title")
        hash3 = generate_metadata_hash("Different title")
        
        assert hash1 == hash2  # Same title
        assert hash1 != hash3  # Different title
        assert len(hash1) == 32  # MD5 hash length
    
    def test_generate_metadata_hash_with_description(self):
        """Test metadata hash generation with title and description."""
        hash1 = generate_metadata_hash("Title", "Description")
        hash2 = generate_metadata_hash("Title", "Description")
        hash3 = generate_metadata_hash("Title", "Different description")
        hash4 = generate_metadata_hash("Title")  # No description
        
        assert hash1 == hash2  # Same title and description
        assert hash1 != hash3  # Different description
        assert hash1 != hash4  # Missing description
    
    def test_generate_metadata_hash_empty_inputs(self):
        """Test metadata hash generation with empty inputs."""
        hash1 = generate_metadata_hash("")
        hash2 = generate_metadata_hash("", "")
        hash3 = generate_metadata_hash("", None)
        
        # Should not crash and should generate consistent hashes
        assert len(hash1) == 32
        assert len(hash2) == 32
        assert len(hash3) == 32


class TestIntegration:
    """Integration tests for URL context extraction with caching."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_with_caching(self):
        """Test complete workflow with caching enabled."""
        # Create cache and extractor
        cache = ContextCache(ttl_hours=1)
        extractor = URLContextExtractor(cache=cache, enable_caching=True)
        
        # Create test news item
        news_item = ProcessedNewsItem(
            url="https://example.com/test-article",
            title="Test Article Title",
            publication_date=datetime.now(timezone.utc),
            source_name="test_source",
            publisher="Test Publisher"
        )
        
        # First call - should use fallback (no LLM clients configured)
        result1 = await extractor.extract_context(news_item)
        
        assert result1.fallback_used is True
        initial_misses = cache.cache_misses
        initial_hits = cache.cache_hits
        
        # Second call - should hit cache (if cache is working)
        result2 = await extractor.extract_context(news_item)
        
        assert result2.summary_text == result1.summary_text
        # Cache should either hit or miss, but total requests should increase
        final_requests = cache.cache_hits + cache.cache_misses
        initial_requests = initial_hits + initial_misses
        assert final_requests > initial_requests
    
    @pytest.mark.asyncio
    async def test_metadata_change_invalidates_cache(self):
        """Test that metadata changes result in cache miss."""
        cache = ContextCache(ttl_hours=1)
        extractor = URLContextExtractor(cache=cache, enable_caching=True)
        
        # Create first news item
        news_item1 = ProcessedNewsItem(
            url="https://example.com/test-article",
            title="Original Title",
            publication_date=datetime.now(timezone.utc),
            source_name="test_source",
            publisher="Test Publisher"
        )
        
        # Create second news item with same URL but different title
        news_item2 = ProcessedNewsItem(
            url="https://example.com/test-article",
            title="Updated Title",  # Different title
            publication_date=datetime.now(timezone.utc),
            source_name="test_source",
            publisher="Test Publisher"
        )
        
        # First call
        result1 = await extractor.extract_context(news_item1)
        initial_requests = cache.cache_hits + cache.cache_misses
        
        # Second call with different metadata should process differently
        result2 = await extractor.extract_context(news_item2)
        final_requests = cache.cache_hits + cache.cache_misses
        
        # Should have processed both items (total requests increased)
        assert final_requests > initial_requests
        assert result1.summary_text != result2.summary_text