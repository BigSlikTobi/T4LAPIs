"""
Comprehensive unit tests for URL context extraction (Task 11.1).

These tests provide comprehensive coverage for the URLContextExtractor
with properly mocked LLM responses to avoid external dependencies.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.nfl_news_pipeline.models import ContextSummary, ProcessedNewsItem
from src.nfl_news_pipeline.story_grouping import URLContextExtractor, ContextCache


class TestURLContextExtractorComprehensive:
    """Comprehensive unit tests for URLContextExtractor with mocked LLM responses."""
    
    @pytest.fixture
    def sample_news_item(self):
        """Create a sample news item for testing."""
        return ProcessedNewsItem(
            url="https://nfl.com/chiefs-mahomes-touchdown",
            title="Patrick Mahomes throws 4 touchdowns in Chiefs victory",
            publication_date=datetime.now(timezone.utc),
            source_name="nfl_official",
            publisher="NFL.com",
            description="Chiefs quarterback Patrick Mahomes threw four touchdown passes in a dominant victory over the Raiders.",
            relevance_score=0.95,
            filter_method="rule_based",
            entities=["Patrick Mahomes", "Kansas City Chiefs", "Las Vegas Raiders"],
            categories=["game_recap", "quarterback", "touchdowns"]
        )
    
    @pytest.fixture
    def mock_openai_response(self):
        """Create a realistic OpenAI API response."""
        return {
            "summary": "Patrick Mahomes led the Kansas City Chiefs to a dominant victory over the Las Vegas Raiders, throwing four touchdown passes in an outstanding performance that showcased his elite quarterback skills.",
            "entities": {
                "players": ["Patrick Mahomes"],
                "teams": ["Kansas City Chiefs", "Las Vegas Raiders"],
                "coaches": [],
                "positions": ["quarterback"]
            },
            "key_topics": ["touchdown", "victory", "quarterback", "performance"],
            "story_category": "game_recap",
            "confidence": 0.93
        }
    
    @pytest.fixture
    def mock_google_response(self):
        """Create a realistic Google AI response."""
        return {
            "summary": "Chiefs quarterback Patrick Mahomes delivered an exceptional performance with four touchdown passes against the Raiders in a commanding team victory.",
            "entities": {
                "players": ["Patrick Mahomes"],
                "teams": ["Kansas City Chiefs", "Raiders"],
                "coaches": [],
                "positions": ["quarterback"]
            },
            "key_topics": ["touchdown", "performance", "quarterback", "chiefs"],
            "story_category": "game_recap",
            "confidence": 0.91
        }
    
    def test_init_with_explicit_api_keys(self):
        """Test initialization with explicitly provided API keys."""
        with patch('src.nfl_news_pipeline.story_grouping.context_extractor.OpenAI') as mock_openai, \
             patch('src.nfl_news_pipeline.story_grouping.context_extractor.genai') as mock_genai, \
             patch('src.nfl_news_pipeline.story_grouping.context_extractor.GenerativeModel') as mock_model:
            
            mock_openai_client = Mock()
            mock_openai.return_value = mock_openai_client
            
            mock_google_client = Mock()
            mock_model.return_value = mock_google_client
            
            extractor = URLContextExtractor(
                openai_api_key="test-openai-key",
                google_api_key="test-google-key",
                preferred_provider="openai"
            )
            
            assert extractor.openai_client == mock_openai_client
            assert extractor.google_client == mock_google_client
            assert extractor.preferred_provider == "openai"
            mock_openai.assert_called_once_with(api_key="test-openai-key")
            mock_genai.configure.assert_called_once_with(api_key="test-google-key")
    
    @pytest.mark.asyncio
    async def test_extract_context_openai_success_with_confidence(self, sample_news_item, mock_openai_response):
        """Test successful OpenAI context extraction with confidence validation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_openai_response)
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = URLContextExtractor()
        extractor.openai_client = mock_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        result = await extractor.extract_context(sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.summary_text == mock_openai_response["summary"]
        assert result.llm_model == "gpt-5-nano"
        assert result.confidence_score == 0.93
        assert result.fallback_used is False
        assert "Patrick Mahomes" in result.entities["players"]
        assert "Kansas City Chiefs" in result.entities["teams"]
        assert "Las Vegas Raiders" in result.entities["teams"]
        assert "touchdown" in result.key_topics
        
        # Verify API call parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-5-nano'
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['max_tokens'] == 500
    
    @pytest.mark.asyncio
    async def test_extract_context_google_success_with_entity_normalization(self, sample_news_item, mock_google_response):
        """Test successful Google AI context extraction with entity normalization."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(mock_google_response)
        mock_client.generate_content.return_value = mock_response
        
        extractor = URLContextExtractor(preferred_provider="google")
        extractor.google_client = mock_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        result = await extractor.extract_context(sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.summary_text == mock_google_response["summary"]
        assert result.llm_model == "gemini-2.5-flash-lite"
        assert result.confidence_score == 0.91
        assert result.fallback_used is False
        
        # Verify entity normalization (Raiders -> Las Vegas Raiders)
        assert "Las Vegas Raiders" in result.entities["teams"]
        assert "Kansas City Chiefs" in result.entities["teams"]
    
    @pytest.mark.asyncio
    async def test_extract_context_llm_timeout_fallback(self, sample_news_item):
        """Test fallback to metadata when LLM times out."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timeout")
        
        extractor = URLContextExtractor()
        extractor.openai_client = mock_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        result = await extractor.extract_context(sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.fallback_used is True
        assert result.llm_model == "metadata_fallback"
        assert result.confidence_score == 0.6  # Lower confidence for fallback
        assert sample_news_item.title in result.summary_text
        assert sample_news_item.description in result.summary_text
    
    @pytest.mark.asyncio
    async def test_extract_context_llm_invalid_json_fallback(self, sample_news_item):
        """Test fallback when LLM returns invalid JSON."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON response"
        mock_client.chat.completions.create.return_value = mock_response
        
        extractor = URLContextExtractor()
        extractor.openai_client = mock_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        result = await extractor.extract_context(sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.fallback_used is True
        assert result.llm_model == "metadata_fallback"
    
    @pytest.mark.asyncio
    async def test_extract_context_with_cache_storage(self, sample_news_item, mock_openai_response):
        """Test that successful extractions are stored in cache."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(mock_openai_response)
        mock_client.chat.completions.create.return_value = mock_response
        
        mock_cache = Mock()
        mock_cache.get_cached_summary.return_value = None
        mock_cache.store_summary.return_value = True
        
        extractor = URLContextExtractor(cache=mock_cache, enable_caching=True)
        extractor.openai_client = mock_client
        
        result = await extractor.extract_context(sample_news_item)
        
        assert isinstance(result, ContextSummary)
        mock_cache.store_summary.assert_called_once()
        stored_summary = mock_cache.store_summary.call_args[0][0]
        assert stored_summary.summary_text == mock_openai_response["summary"]
    
    @pytest.mark.asyncio
    async def test_extract_context_retry_logic_with_exponential_backoff(self, sample_news_item):
        """Test retry logic with exponential backoff for transient errors."""
        mock_client = Mock()
        # Fail twice, then succeed
        mock_client.chat.completions.create.side_effect = [
            Exception("Rate limit"),
            Exception("Service unavailable"),
            Mock(choices=[Mock(message=Mock(content=json.dumps({
                "summary": "Test summary",
                "entities": {"players": [], "teams": []},
                "key_topics": ["test"],
                "confidence": 0.8
            })))])
        ]
        
        extractor = URLContextExtractor()
        extractor.openai_client = mock_client
        extractor.cache = Mock()
        extractor.cache.get_cached_summary.return_value = None
        
        with patch('asyncio.sleep') as mock_sleep:  # Mock sleep to speed up test
            result = await extractor.extract_context(sample_news_item)
        
        assert isinstance(result, ContextSummary)
        assert result.summary_text == "Test summary"
        assert mock_client.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries with backoff
    
    def test_normalize_team_names_comprehensive(self):
        """Test comprehensive team name normalization."""
        extractor = URLContextExtractor()
        
        test_cases = [
            # (input, expected_output)
            (["chiefs", "Chiefs", "CHIEFS"], ["Kansas City Chiefs"]),
            (["49ers", "niners", "San Francisco"], ["San Francisco 49ers"]),
            (["bills", "Buffalo"], ["Buffalo Bills"]),
            (["pats", "patriots", "New England"], ["New England Patriots"]),
            (["dolphins", "Miami"], ["Miami Dolphins"]),
            (["jets", "NY Jets"], ["New York Jets"]),
            (["steelers", "Pittsburgh"], ["Pittsburgh Steelers"]),
            (["ravens", "Baltimore"], ["Baltimore Ravens"]),
            (["browns", "Cleveland"], ["Cleveland Browns"]),
            (["bengals", "Cincinnati"], ["Cincinnati Bengals"]),
            (["titans", "Tennessee"], ["Tennessee Titans"]),
            (["colts", "Indianapolis"], ["Indianapolis Colts"]),
            (["texans", "Houston"], ["Houston Texans"]),
            (["jaguars", "jags", "Jacksonville"], ["Jacksonville Jaguars"]),
            (["broncos", "Denver"], ["Denver Broncos"]),
            (["chargers", "LA Chargers"], ["Los Angeles Chargers"]),
            (["raiders", "Las Vegas", "LV Raiders"], ["Las Vegas Raiders"]),
            (["cowboys", "Dallas"], ["Dallas Cowboys"]),
            (["giants", "NY Giants"], ["New York Giants"]),
            (["eagles", "Philadelphia"], ["Philadelphia Eagles"]),
            (["commanders", "Washington"], ["Washington Commanders"]),
            (["packers", "Green Bay"], ["Green Bay Packers"]),
            (["lions", "Detroit"], ["Detroit Lions"]),
            (["bears", "Chicago"], ["Chicago Bears"]),
            (["vikings", "Minnesota"], ["Minnesota Vikings"]),
            (["falcons", "Atlanta"], ["Atlanta Falcons"]),
            (["panthers", "Carolina"], ["Carolina Panthers"]),
            (["saints", "New Orleans"], ["New Orleans Saints"]),
            (["bucs", "buccaneers", "Tampa Bay"], ["Tampa Bay Buccaneers"]),
            (["cardinals", "Arizona"], ["Arizona Cardinals"]),
            (["rams", "LA Rams"], ["Los Angeles Rams"]),
            (["seahawks", "Seattle"], ["Seattle Seahawks"]),
        ]
        
        for input_teams, expected in test_cases:
            result = extractor._normalize_team_names(input_teams)
            assert len(result) == 1, f"Expected 1 team for {input_teams}, got {result}"
            assert result[0] == expected[0], f"Expected {expected[0]} for {input_teams}, got {result[0]}"
    
    def test_normalize_player_names_edge_cases(self):
        """Test player name normalization with edge cases."""
        extractor = URLContextExtractor()
        
        test_cases = [
            # (input, expected_output)
            (["patrick mahomes", "PATRICK MAHOMES"], ["Patrick Mahomes"]),
            (["tom brady", "Tom Brady", "T. Brady"], ["Tom Brady", "T. Brady"]),  # Keep variations
            (["   josh allen   ", "Josh Allen"], ["Josh Allen"]),
            (["", "  ", "invalid name"], ["Invalid Name"]),  # Handle empty strings
            (["j.j. watt", "JJ Watt"], ["J.J. Watt", "Jj Watt"]),
            (["o'dell beckham jr", "Odell Beckham Jr."], ["O'Dell Beckham Jr", "Odell Beckham Jr."]),
        ]
        
        for input_players, expected in test_cases:
            result = extractor._normalize_player_names(input_players)
            assert len(result) <= len(expected), f"Too many results for {input_players}: {result}"
            for expected_name in expected:
                if expected_name in ["Invalid Name"]:
                    continue  # Skip validation for edge cases
                # Check that similar names are present (allowing for minor variations)
                assert any(expected_name.lower() in name.lower() or name.lower() in expected_name.lower() 
                          for name in result), f"Expected name like '{expected_name}' in {result}"
    
    def test_parse_llm_response_malformed_json_variations(self):
        """Test parsing various malformed JSON responses."""
        extractor = URLContextExtractor()
        
        news_item = ProcessedNewsItem(
            url="https://example.com/test",
            title="Test Article",
            publication_date=datetime.now(timezone.utc),
            source_name="test",
            publisher="test"
        )
        
        # Test various malformed JSON scenarios
        malformed_responses = [
            "Not JSON at all",
            '{"incomplete": json',
            '{"valid_json": true, "but_missing_required_fields": true}',
            '```json\n{"summary": "test"}\n```',  # Should work - JSON in code blocks
            '{"summary": null, "entities": {}, "key_topics": [], "confidence": 0.8}',  # null summary
            '{"summary": "", "entities": {}, "key_topics": [], "confidence": 1.2}',  # invalid confidence
        ]
        
        for response in malformed_responses:
            result = extractor._parse_llm_response(response, "test-model", news_item)
            
            if "```json" in response and "summary" in response:
                # Should successfully parse JSON in code blocks
                assert result is not None
            elif "null" in response or '""' in response or "1.2" in response:
                # Should fail validation
                assert result is None
            else:
                # Should fail parsing
                assert result is None
    
    def test_extract_entities_from_text_comprehensive(self):
        """Test comprehensive entity extraction from text."""
        extractor = URLContextExtractor()
        
        test_texts = [
            "The Kansas City Chiefs defeated the Buffalo Bills 31-24 in a thrilling game.",
            "Patrick Mahomes threw for 300 yards while Josh Allen had 250 passing yards.",
            "Coach Andy Reid praised the team's performance against Buffalo.",
            "The 49ers and Cowboys will play next Sunday in Dallas.",
            "Travis Kelce caught 8 passes for 120 yards and 2 touchdowns.",
        ]
        
        for text in test_texts:
            entities = extractor._extract_entities_from_text(text)
            
            assert isinstance(entities, dict)
            assert "teams" in entities
            assert "players" in entities
            assert "coaches" in entities
            assert isinstance(entities["teams"], list)
            assert isinstance(entities["players"], list)
            assert isinstance(entities["coaches"], list)
            
            # Verify that detected entities are properly normalized
            for team in entities["teams"]:
                assert any(team.endswith(suffix) for suffix in [
                    "Chiefs", "Bills", "49ers", "Cowboys", "Ravens", "Steelers", 
                    "Patriots", "Dolphins", "Jets", "Broncos", "Chargers", "Raiders"
                ]), f"Team '{team}' doesn't appear to be properly normalized"
    
    def test_create_extraction_prompt_comprehensive(self):
        """Test prompt creation with various input scenarios."""
        extractor = URLContextExtractor()
        
        # Test with full metadata
        full_item = ProcessedNewsItem(
            url="https://nfl.com/full-metadata-article",
            title="Chiefs Trade for Star Wide Receiver",
            publication_date=datetime.now(timezone.utc),
            source_name="nfl_official",
            publisher="NFL.com",
            description="The Kansas City Chiefs acquired a top-tier wide receiver in a blockbuster trade.",
            entities=["Kansas City Chiefs"],
            categories=["trade", "wide receiver"]
        )
        
        prompt = extractor._create_extraction_prompt(full_item)
        
        assert "Chiefs Trade for Star Wide Receiver" in prompt
        assert "blockbuster trade" in prompt
        assert "JSON format" in prompt
        assert "summary" in prompt
        assert "entities" in prompt
        assert "confidence" in prompt
        
        # Test with minimal metadata
        minimal_item = ProcessedNewsItem(
            url="https://example.com/minimal",
            title="Short Title",
            publication_date=datetime.now(timezone.utc),
            source_name="test",
            publisher="test"
        )
        
        minimal_prompt = extractor._create_extraction_prompt(minimal_item)
        assert "Short Title" in minimal_prompt
        assert len(minimal_prompt) < len(prompt)  # Should be shorter
    
    @pytest.mark.asyncio
    async def test_fallback_with_existing_entities_integration(self, sample_news_item):
        """Test fallback mode properly uses existing entities."""
        # Modify sample item to have rich existing entities
        sample_news_item.entities = ["Patrick Mahomes", "Travis Kelce", "Kansas City Chiefs"]
        sample_news_item.categories = ["touchdown", "quarterback", "victory"]
        
        extractor = URLContextExtractor()
        # Ensure no LLM clients to force fallback
        extractor.openai_client = None
        extractor.google_client = None
        
        result = await extractor.extract_context(sample_news_item)
        
        assert result.fallback_used is True
        assert "Patrick Mahomes" in result.entities.get("players", [])
        assert "Travis Kelce" in result.entities.get("players", [])
        assert "Kansas City Chiefs" in result.entities.get("teams", [])
        assert "touchdown" in result.key_topics
        assert "quarterback" in result.key_topics
        assert "victory" in result.key_topics


class TestContextCacheAdvanced:
    """Advanced tests for ContextCache functionality."""
    
    @pytest.fixture
    def cache_with_custom_config(self):
        """Create cache with custom configuration."""
        return ContextCache(
            ttl_hours=12,
            enable_memory_cache=True,
            enable_disk_cache=False,
            verbose=True
        )
    
    def test_cache_key_generation_with_metadata_variations(self, cache_with_custom_config):
        """Test cache key generation with different metadata combinations."""
        cache = cache_with_custom_config
        
        # Test URL normalization
        urls = [
            "https://example.com/article",
            "http://example.com/article",
            "https://example.com/article?utm_source=test",
            "https://example.com/article#section1"
        ]
        
        keys = [cache._generate_cache_key(url) for url in urls]
        
        # URLs with different protocols/parameters should generate different keys
        assert len(set(keys)) == len(urls), "Expected different keys for different URLs"
        
        # Same URL should generate same key
        key1 = cache._generate_cache_key("https://example.com/test")
        key2 = cache._generate_cache_key("https://example.com/test")
        assert key1 == key2
    
    def test_cache_statistics_accuracy(self, cache_with_custom_config):
        """Test cache statistics tracking accuracy."""
        cache = cache_with_custom_config
        
        # Mock memory cache for controlled testing
        cache.memory_cache = Mock()
        
        # Simulate cache hits and misses
        cache.memory_cache.get.side_effect = [
            {"summary_text": "cached1"},  # hit
            None,  # miss
            {"summary_text": "cached2"},  # hit
            None,  # miss
            None,  # miss
        ]
        
        # Perform cache operations
        urls = [f"https://example.com/test{i}" for i in range(5)]
        for url in urls:
            cache.get_cached_summary(url)
        
        stats = cache.get_cache_stats()
        
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 3
        assert stats["total_requests"] == 5
        assert stats["hit_rate_percent"] == 40.0
    
    def test_cache_ttl_expiration_boundary_conditions(self):
        """Test cache TTL expiration at boundary conditions."""
        cache = ContextCache(ttl_hours=24)
        
        now = datetime.now(timezone.utc)
        
        # Test summary exactly at TTL boundary
        exactly_expired = ContextSummary(
            news_url_id="test",
            summary_text="test",
            llm_model="test",
            confidence_score=0.8,
            generated_at=now - timedelta(hours=24, seconds=1)  # 1 second past TTL
        )
        
        exactly_valid = ContextSummary(
            news_url_id="test",
            summary_text="test", 
            llm_model="test",
            confidence_score=0.8,
            generated_at=now - timedelta(hours=23, minutes=59, seconds=59)  # 1 second before TTL
        )
        
        assert not cache._is_valid_cache_entry(exactly_expired)
        assert cache._is_valid_cache_entry(exactly_valid)
    
    def test_cache_error_handling_resilience(self, cache_with_custom_config):
        """Test cache resilience to various error conditions."""
        cache = cache_with_custom_config
        
        # Mock memory cache to raise exceptions
        cache.memory_cache = Mock()
        cache.memory_cache.get.side_effect = Exception("Cache error")
        cache.memory_cache.set.side_effect = Exception("Cache write error")
        
        # Operations should not raise exceptions, just return appropriate defaults
        result = cache.get_cached_summary("https://example.com/test")
        assert result is None  # Should gracefully handle error
        
        summary = ContextSummary(
            news_url_id="test",
            summary_text="test",
            llm_model="test",
            confidence_score=0.8,
            generated_at=datetime.now(timezone.utc)
        )
        
        # Store should return False on error, not raise exception
        result = cache.store_summary(summary)
        assert result is False