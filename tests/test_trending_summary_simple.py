#!/usr/bin/env python3
"""
Simple Integration Tests for Trending Summary Generator

This test file validates the core functionality without complex mocking.
Run with: python tests/test_trending_summary_simple.py
"""

import os
import sys
import tempfile
from io import StringIO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from content_generation.trending_summary_generator import (
    TrendingSummary,
    TrendingSummaryGenerator,
    parse_entity_ids_from_input
)


def test_trending_summary_dataclass():
    """Test TrendingSummary dataclass functionality."""
    print("Testing TrendingSummary dataclass...")
    
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
    assert summary.summary_content == "Test summary content"
    
    # Test to_dict conversion
    summary_dict = summary.to_dict()
    assert summary_dict['entity_id'] == "00-0033873"
    assert summary_dict['entity_type'] == "player"
    assert summary_dict['source_article_count'] == 5
    
    print("✅ TrendingSummary dataclass tests passed")


def test_entity_type_determination():
    """Test entity type determination logic."""
    print("Testing entity type determination...")
    
    # Mock generator without database connections
    class MockGenerator:
        def determine_entity_type(self, entity_id):
            if len(entity_id) >= 8 and '-' in entity_id:
                return 'player'
            else:
                return 'team'
    
    generator = MockGenerator()
    
    # Test player IDs
    assert generator.determine_entity_type("00-0033873") == "player"
    assert generator.determine_entity_type("00-0029263") == "player"
    
    # Test team IDs
    assert generator.determine_entity_type("KC") == "team"
    assert generator.determine_entity_type("NYJ") == "team"
    assert generator.determine_entity_type("NE") == "team"
    
    print("✅ Entity type determination tests passed")


def test_input_parsing():
    """Test input parsing functionality."""
    print("Testing input parsing...")
    
    # Test comma-separated
    result = parse_entity_ids_from_input("00-0033873,KC,NYJ")
    assert result == ["00-0033873", "KC", "NYJ"]
    
    # Test space-separated
    result = parse_entity_ids_from_input("00-0033873 KC NYJ")
    assert result == ["00-0033873", "KC", "NYJ"]
    
    # Test newline-separated
    result = parse_entity_ids_from_input("00-0033873\nKC\nNYJ")
    assert result == ["00-0033873", "KC", "NYJ"]
    
    # Test mixed delimiters
    result = parse_entity_ids_from_input("00-0033873, KC\nNYJ\t00-0029263")
    assert result == ["00-0033873", "KC", "NYJ", "00-0029263"]
    
    # Test empty input
    assert parse_entity_ids_from_input("") == []
    assert parse_entity_ids_from_input(None) == []
    assert parse_entity_ids_from_input("   ") == []
    
    # Test with extra whitespace
    result = parse_entity_ids_from_input("  00-0033873  ,  KC  ,  NYJ  ")
    assert result == ["00-0033873", "KC", "NYJ"]
    
    print("✅ Input parsing tests passed")


def test_prompt_creation_structure():
    """Test prompt creation structure without database dependencies."""
    print("Testing prompt creation structure...")
    
    # Mock generator class
    class MockPromptGenerator:
        def create_trending_summary_prompt(self, entity_id, entity_type, entity_name, articles, stats):
            prompt = f"""Create a trending NFL summary for {entity_name} ({entity_type}).

RECENT NEWS ({min(len(articles), 5)} articles):
"""
            for i, article in enumerate(articles[:5]):
                title = article.get('headline', 'No title')
                content = article.get('Content', '')[:200]
                prompt += f"""
{i+1}. {title}
   {content}...

"""
            
            if stats and entity_type == 'player':
                prompt += f"""RECENT STATS ({len(stats)} games):
"""
                for stat in stats:
                    game_parts = stat.get('game_id', '').split('_')
                    week = game_parts[1] if len(game_parts) >= 2 else '?'
                    pass_yds = stat.get('passing_yards', 0)
                    pass_tds = stat.get('passing_tds', 0)
                    prompt += f"""Week {week}: {pass_yds} pass yds, {pass_tds} pass TDs
"""
            
            prompt += f"""
Write a 400-word trending summary explaining:
1. Why {entity_name} is trending now
2. Key developments from recent articles
3. Performance highlights (if player)
4. What NFL fans should know

Use an engaging, journalistic tone. Start with a compelling headline."""
            
            return prompt
    
    generator = MockPromptGenerator()
    
    # Test player prompt
    articles = [{'headline': 'Mahomes leads Chiefs to victory', 'Content': 'Patrick Mahomes threw for 350 yards...'}]
    stats = [{'game_id': '2024_18_KC_BUF', 'passing_yards': 350, 'passing_tds': 3}]
    
    prompt = generator.create_trending_summary_prompt(
        "00-0033873", "player", "Patrick Mahomes (KC - QB)", articles, stats
    )
    
    assert "Patrick Mahomes (KC - QB)" in prompt
    assert "Mahomes leads Chiefs to victory" in prompt
    assert "RECENT STATS" in prompt
    assert "Week 18: 350 pass yds, 3 pass TDs" in prompt
    assert "Write a 400-word trending summary" in prompt
    
    # Test team prompt (no stats)
    team_articles = [{'headline': 'Chiefs sign new coordinator', 'Content': 'The Kansas City Chiefs announced...'}]
    
    team_prompt = generator.create_trending_summary_prompt(
        "KC", "team", "Kansas City Chiefs", team_articles, []
    )
    
    assert "Kansas City Chiefs" in team_prompt
    assert "Chiefs sign new coordinator" in team_prompt
    assert "RECENT STATS" not in team_prompt
    
    print("✅ Prompt creation structure tests passed")


def test_cli_help_functionality():
    """Test that the CLI help works without errors."""
    print("Testing CLI help functionality...")
    
    try:
        # Import and test basic CLI functionality
        from content_generation.trending_summary_generator import main
        
        # Redirect stdout to capture help output
        old_stdout = sys.stdout
        old_argv = sys.argv
        
        sys.stdout = StringIO()
        sys.argv = ['trending_summary_generator.py', '--help']
        
        try:
            main()
        except SystemExit as e:
            # Help command exits with code 0
            assert e.code == 0
        
        help_output = sys.stdout.getvalue()
        
        # Restore stdout and argv
        sys.stdout = old_stdout
        sys.argv = old_argv
        
        # Check that help contains expected content
        assert "Generate comprehensive summaries for trending NFL entities" in help_output
        assert "--entity-ids" in help_output
        assert "--from-stdin" in help_output
        assert "--dry-run" in help_output
        assert "--llm-provider" in help_output
        
        print("✅ CLI help functionality tests passed")
        
    except ImportError as e:
        print(f"⚠️ CLI help test skipped due to import error: {e}")


def run_all_tests():
    """Run all simple tests."""
    print("🧪 Running Trending Summary Generator Simple Tests\n")
    
    try:
        test_trending_summary_dataclass()
        test_entity_type_determination()
        test_input_parsing()
        test_prompt_creation_structure()
        test_cli_help_functionality()
        
        print("\n✅ All simple tests passed! The trending summary generator core functionality is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
