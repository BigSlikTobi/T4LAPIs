#!/usr/bin/env python3
"""
T# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the module directly since it's a script
sys.path.insert(0, os.path.join(project_root, 'scripts'))
import trending_summary_generator
from trending_summary_generator import (
    TrendingSummary,
    TrendingSummaryGenerator,
    parse_entity_ids_from_input,
    main
)rending Summary Generator Script - Epic 2 Task 4

Test coverage:
- TrendingSummary dataclass functionality
- TrendingSummaryGenerator initialization and configuration
- Entity type determination and name retrieval
- Article and stats fetching
- Prompt generation
- LLM integration and summary generation
- Database storage
- CLI argument parsing and input methods
- Error handling and edge cases
- Pipeline integration

Run with: python -m pytest tests/test_content_generation.trending_summary_generator.py -v
"""

import pytest
import json
import tempfile
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from io import StringIO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from content_generation.trending_summary_generator import (
    TrendingSummary,
    TrendingSummaryGenerator,
    parse_entity_ids_from_input,
    main
)


class TestTrendingSummary:
    """Test TrendingSummary dataclass functionality."""
    
    def test_trending_summary_creation(self):
        """Test TrendingSummary object creation."""
        summary = TrendingSummary(
            entity_id="00-0033873",
            entity_type="player",
            entity_name="Patrick Mahomes (KC - QB)",
            summary_content="Test summary content",
            source_article_count=5,
            source_stat_count=3,
            generated_at="2025-08-03T18:00:00Z"
        )
        
        assert summary.entity_id == "00-0033873"
        assert summary.entity_type == "player"
        assert summary.entity_name == "Patrick Mahomes (KC - QB)"
        assert summary.summary_content == "Test summary content"
        assert summary.source_article_count == 5
        assert summary.source_stat_count == 3
        assert summary.generated_at == "2025-08-03T18:00:00Z"
    
    def test_trending_summary_to_dict(self):
        """Test TrendingSummary to_dict conversion."""
        summary = TrendingSummary(
            entity_id="KC",
            entity_type="team",
            entity_name="Kansas City Chiefs",
            summary_content="Team summary",
            source_article_count=10,
            source_stat_count=0,
            generated_at="2025-08-03T18:00:00Z"
        )
        
        expected_dict = {
            'entity_id': "KC",
            'entity_type': "team",
            'entity_name': "Kansas City Chiefs",
            'summary_content': "Team summary",
            'source_article_count': 10,
            'source_stat_count': 0,
            'generated_at': "2025-08-03T18:00:00Z"
        }
        
        assert summary.to_dict() == expected_dict


class TestTrendingSummaryGenerator:
    """Test TrendingSummaryGenerator class functionality."""
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_generator_initialization(self, mock_db_manager):
        """Test generator initialization with default parameters."""
        generator = TrendingSummaryGenerator()
        
        assert generator.lookback_hours == 72
        assert generator.dry_run is False
        assert generator.preferred_llm_provider == 'gemini'
        assert generator.llm_config is None
        assert generator.llm_processing_time == 0.0
        assert len(generator.stats) == 7
        
        # Verify database managers were created
        assert mock_db_manager.call_count == 5
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_generator_custom_initialization(self, mock_db_manager):
        """Test generator initialization with custom parameters."""
        generator = TrendingSummaryGenerator(
            lookback_hours=48,
            dry_run=True,
            preferred_llm_provider='deepseek'
        )
        
        assert generator.lookback_hours == 48
        assert generator.dry_run is True
        assert generator.preferred_llm_provider == 'deepseek'
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_determine_entity_type(self, mock_db_manager):
        """Test entity type determination logic."""
        generator = TrendingSummaryGenerator()
        
        # Test player ID patterns
        assert generator.determine_entity_type("00-0033873") == "player"
        assert generator.determine_entity_type("00-0029263") == "player"
        assert generator.determine_entity_type("12-3456789") == "player"
        
        # Test team ID patterns
        assert generator.determine_entity_type("KC") == "team"
        assert generator.determine_entity_type("NYJ") == "team"
        assert generator.determine_entity_type("NE") == "team"
        assert generator.determine_entity_type("SF") == "team"
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_get_entity_name_player_success(self, mock_db_manager):
        """Test getting player entity name successfully."""
        # Mock the database response
        mock_response = Mock()
        mock_response.data = [{
            'full_name': 'Patrick Mahomes',
            'latest_team': 'KC',
            'position': 'QB'
        }]
        
        mock_db = Mock()
        mock_db.supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        generator = TrendingSummaryGenerator()
        generator.players_db = mock_db
        
        result = generator.get_entity_name("00-0033873", "player")
        
        assert result == "Patrick Mahomes (KC - QB)"
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_get_entity_name_team_success(self, mock_db_manager):
        """Test getting team entity name successfully."""
        # Mock the database response
        mock_response = Mock()
        mock_response.data = [{
            'team_name': 'Kansas City Chiefs'
        }]
        
        mock_db = Mock()
        mock_db.supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        generator = TrendingSummaryGenerator()
        generator.teams_db = mock_db
        
        result = generator.get_entity_name("KC", "team")
        
        assert result == "Kansas City Chiefs"
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_get_entity_name_not_found(self, mock_db_manager):
        """Test getting entity name when not found in database."""
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        mock_db = Mock()
        mock_db.supabase.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        generator = TrendingSummaryGenerator()
        generator.players_db = mock_db
        
        result = generator.get_entity_name("unknown-player", "player")
        
        assert result == "unknown-player"
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_get_recent_articles_success(self, mock_db_manager):
        """Test getting recent articles successfully."""
        # Mock database response with articles
        mock_response = Mock()
        mock_response.data = [
            {
                'SourceArticles': {
                    'id': 1,
                    'headline': 'Test Article 1',
                    'Content': 'Article content 1',
                    'Author': 'Author 1',
                    'created_at': '2025-08-03T12:00:00Z',
                    'source': 'ESPN'
                }
            },
            {
                'SourceArticles': {
                    'id': 2,
                    'headline': 'Test Article 2',
                    'Content': 'Article content 2',
                    'Author': 'Author 2',
                    'created_at': '2025-08-03T13:00:00Z',
                    'source': 'NFL.com'
                }
            }
        ]
        
        mock_db = Mock()
        mock_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.gte.return_value.execute.return_value = mock_response
        
        generator = TrendingSummaryGenerator(lookback_hours=24)
        generator.entity_links_db = mock_db
        
        articles = generator.get_recent_articles_for_entity("00-0033873", "player")
        
        assert len(articles) == 2
        assert articles[0]['headline'] == 'Test Article 1'
        assert articles[1]['headline'] == 'Test Article 2'
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_get_recent_stats_success(self, mock_db_manager):
        """Test getting recent stats successfully."""
        # Mock database response with stats
        mock_response = Mock()
        mock_response.data = [
            {
                'game_id': '2024_18_KC_BUF',
                'passing_yards': 320,
                'passing_tds': 3,
                'rushing_yards': 15,
                'receiving_yards': 0
            },
            {
                'game_id': '2024_17_KC_HOU',
                'passing_yards': 280,
                'passing_tds': 2,
                'rushing_yards': 8,
                'receiving_yards': 0
            }
        ]
        
        mock_db = Mock()
        mock_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        generator = TrendingSummaryGenerator()
        generator.stats_db = mock_db
        
        stats = generator.get_recent_stats_for_entity("00-0033873", "player")
        
        assert len(stats) == 2
        assert stats[0]['passing_yards'] == 320
        assert stats[1]['passing_yards'] == 280
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_get_recent_stats_team_empty(self, mock_db_manager):
        """Test getting stats for team returns empty list."""
        generator = TrendingSummaryGenerator()
        
        stats = generator.get_recent_stats_for_entity("KC", "team")
        
        assert stats == []
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_create_trending_summary_prompt_player(self, mock_db_manager):
        """Test prompt creation for player with articles and stats."""
        generator = TrendingSummaryGenerator()
        
        articles = [
            {
                'headline': 'Mahomes leads Chiefs to victory',
                'Content': 'Patrick Mahomes threw for 350 yards and 3 touchdowns...'
            }
        ]
        
        stats = [
            {
                'game_id': '2024_18_KC_BUF',
                'passing_yards': 350,
                'passing_tds': 3,
                'rushing_yards': 12,
                'receiving_yards': 0
            }
        ]
        
        prompt = generator.create_trending_summary_prompt(
            "00-0033873", "player", "Patrick Mahomes (KC - QB)", articles, stats
        )
        
        assert "Patrick Mahomes (KC - QB)" in prompt
        assert "Mahomes leads Chiefs to victory" in prompt
        assert "RECENT STATS" in prompt
        assert "Week 18: 350 pass yds, 3 pass TDs" in prompt
        assert "Write a 400-word trending summary" in prompt
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_create_trending_summary_prompt_team(self, mock_db_manager):
        """Test prompt creation for team with articles only."""
        generator = TrendingSummaryGenerator()
        
        articles = [
            {
                'headline': 'Chiefs sign new defensive coordinator',
                'Content': 'The Kansas City Chiefs have announced...'
            }
        ]
        
        prompt = generator.create_trending_summary_prompt(
            "KC", "team", "Kansas City Chiefs", articles, []
        )
        
        assert "Kansas City Chiefs" in prompt
        assert "Chiefs sign new defensive coordinator" in prompt
        assert "RECENT STATS" not in prompt
        assert "Write a 400-word trending summary" in prompt
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    @patch('content_generation.trending_summary_generator.generate_content_with_model')
    def test_generate_summary_with_llm_success(self, mock_generate, mock_db_manager):
        """Test successful LLM summary generation."""
        mock_generate.return_value = "Generated trending summary content"
        
        generator = TrendingSummaryGenerator()
        generator.llm_config = {'provider': 'gemini'}
        
        summary = generator.generate_summary_with_llm("Test prompt")
        
        assert summary == "Generated trending summary content"
        assert generator.llm_processing_time > 0
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    @patch('content_generation.trending_summary_generator.generate_content_with_model')
    def test_generate_summary_with_llm_empty_response(self, mock_generate, mock_db_manager):
        """Test LLM returning empty response."""
        mock_generate.return_value = ""
        
        generator = TrendingSummaryGenerator()
        generator.llm_config = {'provider': 'gemini'}
        
        summary = generator.generate_summary_with_llm("Test prompt")
        
        assert summary is None
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_generate_summary_with_llm_no_config(self, mock_db_manager):
        """Test LLM generation without initialized config."""
        generator = TrendingSummaryGenerator()
        generator.llm_config = None
        
        summary = generator.generate_summary_with_llm("Test prompt")
        
        assert summary is None
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_save_trending_summary_success(self, mock_db_manager):
        """Test successful database save."""
        mock_db = Mock()
        mock_db.insert_records.return_value = {'success': True}
        
        generator = TrendingSummaryGenerator()
        generator.trending_updates_db = mock_db
        
        summary = TrendingSummary(
            entity_id="00-0033873",
            entity_type="player",
            entity_name="Patrick Mahomes",
            summary_content="Test summary",
            source_article_count=5,
            source_stat_count=3,
            generated_at="2025-08-03T18:00:00Z"
        )
        
        result = generator.save_trending_summary(summary)
        
        assert result is True
        mock_db.insert_records.assert_called_once()
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_save_trending_summary_dry_run(self, mock_db_manager):
        """Test database save in dry run mode."""
        generator = TrendingSummaryGenerator(dry_run=True)
        
        summary = TrendingSummary(
            entity_id="00-0033873",
            entity_type="player",
            entity_name="Patrick Mahomes",
            summary_content="Test summary",
            source_article_count=5,
            source_stat_count=3,
            generated_at="2025-08-03T18:00:00Z"
        )
        
        result = generator.save_trending_summary(summary)
        
        assert result is True


class TestInputParsing:
    """Test input parsing functionality."""
    
    def test_parse_entity_ids_comma_separated(self):
        """Test parsing comma-separated entity IDs."""
        input_str = "00-0033873,KC,NYJ"
        result = parse_entity_ids_from_input(input_str)
        
        assert result == ["00-0033873", "KC", "NYJ"]
    
    def test_parse_entity_ids_space_separated(self):
        """Test parsing space-separated entity IDs."""
        input_str = "00-0033873 KC NYJ"
        result = parse_entity_ids_from_input(input_str)
        
        assert result == ["00-0033873", "KC", "NYJ"]
    
    def test_parse_entity_ids_newline_separated(self):
        """Test parsing newline-separated entity IDs."""
        input_str = "00-0033873\nKC\nNYJ"
        result = parse_entity_ids_from_input(input_str)
        
        assert result == ["00-0033873", "KC", "NYJ"]
    
    def test_parse_entity_ids_mixed_delimiters(self):
        """Test parsing with mixed delimiters."""
        input_str = "00-0033873, KC\nNYJ\t00-0029263"
        result = parse_entity_ids_from_input(input_str)
        
        assert result == ["00-0033873", "KC", "NYJ", "00-0029263"]
    
    def test_parse_entity_ids_empty_input(self):
        """Test parsing empty input."""
        assert parse_entity_ids_from_input("") == []
        assert parse_entity_ids_from_input(None) == []
        assert parse_entity_ids_from_input("   ") == []
    
    def test_parse_entity_ids_with_extra_whitespace(self):
        """Test parsing with extra whitespace."""
        input_str = "  00-0033873  ,  KC  ,  NYJ  "
        result = parse_entity_ids_from_input(input_str)
        
        assert result == ["00-0033873", "KC", "NYJ"]


class TestCLIIntegration:
    """Test CLI argument parsing and main function."""
    
    @patch('content_generation.trending_summary_generator.TrendingSummaryGenerator')
    @patch('sys.argv', ['content_generation.trending_summary_generator.py', '--entity-ids', 'KC,NYJ', '--dry-run'])
    def test_cli_entity_ids_argument(self, mock_generator_class):
        """Test CLI with entity IDs argument."""
        mock_generator = Mock()
        mock_generator.generate_trending_summaries.return_value = {
            'success': True,
            'summaries': [],
            'stats': {
                'entities_processed': 2,
                'summaries_generated': 2,
                'articles_analyzed': 10,
                'stats_analyzed': 0,
                'errors': 0,
                'processing_time': 5.0,
                'llm_time': 3.0
            }
        }
        mock_generator_class.return_value = mock_generator
        
        with patch('sys.stdout', new_callable=StringIO):
            result = main()
        
        assert result == 0
        mock_generator_class.assert_called_once_with(
            lookback_hours=72,
            dry_run=True,
            preferred_llm_provider='gemini'
        )
    
    @patch('content_generation.trending_summary_generator.TrendingSummaryGenerator')
    @patch('sys.argv', ['content_generation.trending_summary_generator.py', '--input-file', 'test.txt'])
    @patch('builtins.open', create=True)
    def test_cli_input_file_argument(self, mock_open, mock_generator_class):
        """Test CLI with input file argument."""
        mock_open.return_value.__enter__.return_value.read.return_value = "KC\nNYJ"
        
        mock_generator = Mock()
        mock_generator.generate_trending_summaries.return_value = {
            'success': True,
            'summaries': [],
            'stats': {
                'entities_processed': 2,
                'summaries_generated': 2,
                'articles_analyzed': 10,
                'stats_analyzed': 0,
                'errors': 0,
                'processing_time': 5.0,
                'llm_time': 3.0
            }
        }
        mock_generator_class.return_value = mock_generator
        
        with patch('sys.stdout', new_callable=StringIO):
            result = main()
        
        assert result == 0
        mock_open.assert_called_once_with('test.txt', 'r')
    
    @patch('content_generation.trending_summary_generator.TrendingSummaryGenerator')
    @patch('sys.argv', ['content_generation.trending_summary_generator.py', '--from-stdin'])
    @patch('sys.stdin', StringIO("KC\nNYJ"))
    def test_cli_stdin_argument(self, mock_generator_class):
        """Test CLI with stdin input."""
        mock_generator = Mock()
        mock_generator.generate_trending_summaries.return_value = {
            'success': True,
            'summaries': [],
            'stats': {
                'entities_processed': 2,
                'summaries_generated': 2,
                'articles_analyzed': 10,
                'stats_analyzed': 0,
                'errors': 0,
                'processing_time': 5.0,
                'llm_time': 3.0
            }
        }
        mock_generator_class.return_value = mock_generator
        
        with patch('sys.stdout', new_callable=StringIO):
            result = main()
        
        assert result == 0
    
    @patch('content_generation.trending_summary_generator.TrendingSummaryGenerator')
    @patch('sys.argv', [
        'content_generation.trending_summary_generator.py', 
        '--entity-ids', 'KC', 
        '--hours', '48', 
        '--llm-provider', 'deepseek',
        '--output-format', 'json'
    ])
    def test_cli_custom_options(self, mock_generator_class):
        """Test CLI with custom options."""
        mock_generator = Mock()
        mock_generator.generate_trending_summaries.return_value = {
            'success': True,
            'summaries': [],
            'stats': {
                'entities_processed': 1,
                'summaries_generated': 1,
                'articles_analyzed': 5,
                'stats_analyzed': 0,
                'errors': 0,
                'processing_time': 3.0,
                'llm_time': 2.0
            }
        }
        mock_generator_class.return_value = mock_generator
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = main()
        
        assert result == 0
        mock_generator_class.assert_called_once_with(
            lookback_hours=48,
            dry_run=False,
            preferred_llm_provider='deepseek'
        )
        
        # Check JSON output
        output = mock_stdout.getvalue()
        assert '"success": true' in output


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('content_generation.trending_summary_generator.DatabaseManager')
    def test_database_connection_error(self, mock_db_manager):
        """Test handling database connection errors."""
        mock_db_manager.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception):
            TrendingSummaryGenerator()
    
    @patch('content_generation.trending_summary_generator.TrendingSummaryGenerator')
    @patch('sys.argv', ['content_generation.trending_summary_generator.py', '--entity-ids', ''])
    def test_empty_entity_ids(self, mock_generator_class):
        """Test handling empty entity IDs."""
        mock_generator = Mock()
        mock_generator.generate_trending_summaries.return_value = {
            'success': True,
            'message': 'No entity IDs to process',
            'stats': {
                'entities_processed': 0,
                'summaries_generated': 0,
                'articles_analyzed': 0,
                'stats_analyzed': 0,
                'errors': 0,
                'processing_time': 0.0,
                'llm_time': 0.0
            }
        }
        mock_generator_class.return_value = mock_generator
        
        with patch('sys.stdout', new_callable=StringIO):
            result = main()
        
        assert result == 1  # Should return error code for no entity IDs
    
    @patch('content_generation.trending_summary_generator.TrendingSummaryGenerator')
    @patch('sys.argv', ['content_generation.trending_summary_generator.py', '--input-file', 'nonexistent.txt'])
    def test_file_not_found(self, mock_generator_class):
        """Test handling file not found error."""
        with patch('sys.stdout', new_callable=StringIO):
            result = main()
        
        assert result == 1
    
    @patch('content_generation.trending_summary_generator.TrendingSummaryGenerator')
    @patch('sys.argv', ['content_generation.trending_summary_generator.py', '--entity-ids', 'KC'])
    def test_llm_initialization_failure(self, mock_generator_class):
        """Test handling LLM initialization failure."""
        mock_generator = Mock()
        mock_generator.generate_trending_summaries.return_value = {
            'success': False,
            'error': 'Failed to initialize LLM client',
            'stats': {
                'entities_processed': 0,
                'summaries_generated': 0,
                'articles_analyzed': 0,
                'stats_analyzed': 0,
                'errors': 0,
                'processing_time': 0.0,
                'llm_time': 0.0
            }
        }
        mock_generator_class.return_value = mock_generator
        
        with patch('sys.stdout', new_callable=StringIO):
            result = main()
        
        assert result == 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
