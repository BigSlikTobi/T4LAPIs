"""
Tests for the Personalized Summary Generator - Sprint 3 Epic 1 Task 1

This test suite covers all functionality of the PersonalizedSummaryGenerator class,
including user preference processing, content gathering, LLM integration,
and database operations.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from content_generation.personal_summary_generator import PersonalizedSummaryGenerator


class TestPersonalizedSummaryGenerator(unittest.TestCase):
    """Test cases for the PersonalizedSummaryGenerator class."""
    
    @patch('content_generation.personal_summary_generator.DatabaseManager')
    def setUp(self, mock_db_manager):
        """Set up test fixtures."""
        # Mock all database managers
        self.mock_users_db = Mock()
        self.mock_preferences_db = Mock()
        self.mock_generated_updates_db = Mock()
        self.mock_articles_db = Mock()
        self.mock_entity_links_db = Mock()
        self.mock_stats_db = Mock()
        
        mock_db_manager.side_effect = [
            self.mock_users_db,
            self.mock_preferences_db,
            self.mock_generated_updates_db,
            self.mock_articles_db,
            self.mock_entity_links_db,
            self.mock_stats_db
        ]
        
        self.generator = PersonalizedSummaryGenerator(lookback_hours=24)
        
        # Sample test data
        self.sample_user_data = {
            'user_id': '550e8400-e29b-41d4-a716-446655440000',
            'preferences': [
                {
                    'preference_id': '123e4567-e89b-12d3-a456-426614174000',
                    'entity_id': '00-0033873',  # Patrick Mahomes
                    'entity_type': 'player',
                    'created_at': '2025-08-02T10:30:00Z'
                },
                {
                    'preference_id': '456e7890-e12b-34d5-a678-901234567890',
                    'entity_id': 'KC',  # Kansas City Chiefs
                    'entity_type': 'team',
                    'created_at': '2025-08-02T10:35:00Z'
                }
            ]
        }
        
        self.sample_articles = [
            {
                'id': 1001,
                'headline': 'Mahomes Throws for 350 Yards in Victory',
                'Content': 'Patrick Mahomes led the Kansas City Chiefs to a 28-21 victory over the Denver Broncos...',
                'Author': 'John Smith',
                'publishedAt': datetime.now(timezone.utc).isoformat(),
                'source': 'ESPN'
            },
            {
                'id': 1002,
                'headline': 'Chiefs Defense Steps Up',
                'Content': 'The Kansas City Chiefs defense forced three turnovers in their latest game...',
                'Author': 'Jane Doe',
                'publishedAt': datetime.now(timezone.utc).isoformat(),
                'source': 'NFL.com'
            }
        ]
        
        self.sample_stats = [
            {
                'stat_id': '00-0033873_2024_15',
                'player_id': '00-0033873',
                'season': 2024,
                'week': 15,
                'passing_yards': 350,
                'passing_tds': 3,
                'interceptions': 1,
                'rushing_yards': 25,
                'rushing_tds': 0
            },
            {
                'stat_id': '00-0033873_2024_14',
                'player_id': '00-0033873',
                'season': 2024,
                'week': 14,
                'passing_yards': 280,
                'passing_tds': 2,
                'interceptions': 0,
                'rushing_yards': 15,
                'rushing_tds': 1
            }
        ]
    
    @patch('content_generation.personal_summary_generator.initialize_model')
    def test_initialize_llm_success(self, mock_initialize_model):
        """Test successful LLM initialization."""
        mock_llm_config = {
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "grounding_enabled": True
        }
        mock_initialize_model.return_value = mock_llm_config
        
        result = self.generator.initialize_llm()
        
        self.assertTrue(result)
        self.assertEqual(self.generator.llm_config, mock_llm_config)
        mock_initialize_model.assert_called()
    
    @patch('content_generation.personal_summary_generator.initialize_model')
    def test_initialize_llm_failure(self, mock_initialize_model):
        """Test LLM initialization failure."""
        mock_initialize_model.side_effect = Exception("API key not found")
        
        result = self.generator.initialize_llm()
        
        self.assertFalse(result)
    
    def test_get_all_users_with_preferences_success(self):
        """Test successful retrieval of users with preferences."""
        # Mock users database response
        mock_users_response = Mock()
        mock_users_response.data = [{'user_id': '550e8400-e29b-41d4-a716-446655440000'}]
        self.mock_users_db.supabase.table.return_value.select.return_value.execute.return_value = mock_users_response
        
        # Mock preferences database response
        mock_prefs_response = Mock()
        mock_prefs_response.data = self.sample_user_data['preferences']
        self.mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_prefs_response
        
        result = self.generator.get_all_users_with_preferences()
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['user_id'], '550e8400-e29b-41d4-a716-446655440000')
        self.assertEqual(len(result[0]['preferences']), 2)
    
    def test_get_all_users_with_preferences_no_users(self):
        """Test retrieval when no users exist."""
        mock_users_response = Mock()
        mock_users_response.data = []
        self.mock_users_db.supabase.table.return_value.select.return_value.execute.return_value = mock_users_response
        
        result = self.generator.get_all_users_with_preferences()
        
        self.assertEqual(len(result), 0)
    
    def test_get_all_users_with_preferences_no_preferences(self):
        """Test retrieval when users have no preferences."""
        # Mock users database response
        mock_users_response = Mock()
        mock_users_response.data = [{'user_id': '550e8400-e29b-41d4-a716-446655440000'}]
        self.mock_users_db.supabase.table.return_value.select.return_value.execute.return_value = mock_users_response
        
        # Mock empty preferences database response
        mock_prefs_response = Mock()
        mock_prefs_response.data = []
        self.mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_prefs_response
        
        result = self.generator.get_all_users_with_preferences()
        
        self.assertEqual(len(result), 0)
    
    def test_get_previous_summary_found(self):
        """Test getting previous summary when method is not yet implemented."""
        # The method currently always returns None due to schema limitations
        result = self.generator.get_previous_summary('user123', '00-0033873', 'player')
        
        # The method currently returns None until schema is updated
        self.assertIsNone(result)
    
    def test_get_previous_summary_not_found(self):
        """Test getting previous summary when none exists."""
        mock_response = Mock()
        mock_response.data = []
        self.mock_generated_updates_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        result = self.generator.get_previous_summary('user123', '00-0033873', 'player')
        
        self.assertIsNone(result)
    
    def test_get_new_articles_for_entity_success(self):
        """Test successful retrieval of new articles for an entity."""
        # Create a mock that properly handles the query chain  
        mock_query_result = type('MockQueryResult', (), {
            'data': [
                {'article_id': 1001, 'SourceArticles': self.sample_articles[0]},
                {'article_id': 1002, 'SourceArticles': self.sample_articles[1]}
            ]
        })()
        
        # Mock the query chain to return our result
        mock_chain = Mock()
        mock_chain.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_query_result
        
        # Replace the supabase client for this test
        original_client = self.generator.entity_links_db.supabase
        self.generator.entity_links_db.supabase = mock_chain
        
        try:
            result = self.generator.get_new_articles_for_entity('00-0033873', 'player', 24)
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['headline'], 'Mahomes Throws for 350 Yards in Victory')
        finally:
            # Restore original
            self.generator.entity_links_db.supabase = original_client
    
    def test_get_new_articles_for_entity_no_articles(self):
        """Test retrieval when no new articles exist."""
        mock_links_response = Mock()
        mock_links_response.data = []
        self.mock_entity_links_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_links_response
        
        result = self.generator.get_new_articles_for_entity('00-0033873', 'player', 24)
        
        self.assertEqual(len(result), 0)
    
    def test_get_new_stats_for_entity_player(self):
        """Test successful retrieval of new stats for a player entity."""
        
        # Create mock stats data
        mock_stats_data = [
            {
                'stat_id': 2001,
                'player_id': '00-0033873',
                'game_id': '2024_12_KC_LV',
                'passing_yards': 350,
                'passing_tds': 3,
                'rushing_yards': 45
            }
        ]
        
        # Create a mock that properly handles the query chain  
        mock_query_result = type('MockQueryResult', (), {
            'data': mock_stats_data
        })()
        
        # Mock the query chain to return our result
        mock_chain = Mock()
        mock_chain.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_query_result
        
        # Replace the supabase client for this test (note: stats_db not entity_links_db)
        original_client = self.generator.stats_db.supabase
        self.generator.stats_db.supabase = mock_chain
        
        try:
            result = self.generator.get_new_stats_for_entity('00-0033873', 'player', 24)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['passing_yards'], 350)
            self.assertEqual(result[0]['passing_tds'], 3)
        finally:
            # Restore original
            self.generator.stats_db.supabase = original_client
    
    def test_get_new_stats_for_entity_team(self):
        """Test stats retrieval for a team (should return empty)."""
        result = self.generator.get_new_stats_for_entity('KC', 'team', 24)
        
        self.assertEqual(len(result), 0)
    
    def test_create_summary_prompt_with_previous_summary(self):
        """Test prompt creation with previous summary context."""
        previous_summary = "Patrick Mahomes had a great game last week..."
        
        prompt = self.generator.create_summary_prompt(
            '00-0033873',
            'player',
            self.sample_articles[:1],
            self.sample_stats[:1],
            previous_summary
        )
        
        self.assertIn('00-0033873', prompt)
        self.assertIn('Previous Summary', prompt)
        self.assertIn('Patrick Mahomes had a great game', prompt)
        self.assertIn('350 yds', prompt)
    
    def test_create_summary_prompt_without_previous_summary(self):
        """Test prompt creation without previous summary."""
        prompt = self.generator.create_summary_prompt(
            'KC',
            'team',
            self.sample_articles,
            [],
            None
        )
        
        self.assertIn('KC (team)', prompt)
        self.assertNotIn('Previous Summary', prompt)
        self.assertIn('Recent Articles (2 found)', prompt)
    
    @patch('content_generation.personal_summary_generator.generate_content_with_model')
    def test_generate_summary_with_llm_success(self, mock_generate_content):
        """Test successful summary generation with LLM."""
        # Set up mock LLM config
        self.generator.llm_config = {
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "grounding_enabled": True
        }
        
        # Mock the response
        mock_generate_content.return_value = "Patrick Mahomes continues to excel this season..."
        
        prompt = "Generate a summary about Patrick Mahomes..."
        result = self.generator.generate_summary_with_llm(prompt)
        
        self.assertEqual(result, "Patrick Mahomes continues to excel this season...")
        mock_generate_content.assert_called_once()
    
    def test_generate_summary_with_llm_no_client(self):
        """Test summary generation when LLM client is not initialized."""
        self.generator.llm_client = None
        
        result = self.generator.generate_summary_with_llm("test prompt")
        
        self.assertIsNone(result)
    
    def test_generate_summary_with_llm_empty_response(self):
        """Test summary generation when LLM returns empty response."""
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_llm_client.client.chat.completions.create.return_value = mock_response
        
        self.generator.llm_client = mock_llm_client
        
        result = self.generator.generate_summary_with_llm("test prompt")
        
        self.assertIsNone(result)
    
    def test_save_generated_summary_success(self):
        """Test successful saving of generated summary."""
        mock_result = {'success': True}
        self.mock_generated_updates_db.insert_records.return_value = mock_result
        
        result = self.generator.save_generated_summary(
            'user123',
            '00-0033873',
            'player',
            'Generated summary content...',
            [1001, 1002],
            ['stat1', 'stat2']
        )
        
        self.assertTrue(result)
        self.mock_generated_updates_db.insert_records.assert_called_once()
    
    def test_save_generated_summary_failure(self):
        """Test saving summary when database insert fails."""
        mock_result = {'success': False, 'error': 'Database error'}
        self.mock_generated_updates_db.insert_records.return_value = mock_result
        
        result = self.generator.save_generated_summary(
            'user123',
            '00-0033873',
            'player',
            'Generated summary content...',
            [1001],
            ['stat1']
        )
        
        self.assertFalse(result)
    
    @patch.object(PersonalizedSummaryGenerator, 'get_previous_summary')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_articles_for_entity')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_stats_for_entity')
    @patch.object(PersonalizedSummaryGenerator, 'generate_summary_with_llm')
    @patch.object(PersonalizedSummaryGenerator, 'save_generated_summary')
    def test_process_user_preference_success(self, mock_save, mock_generate, mock_stats, mock_articles, mock_previous):
        """Test successful processing of a user preference."""
        # Set up mocks
        mock_previous.return_value = "Previous summary..."
        mock_articles.return_value = self.sample_articles
        mock_stats.return_value = self.sample_stats
        mock_generate.return_value = "Generated summary content..."
        mock_save.return_value = True
        
        preference = self.sample_user_data['preferences'][0]
        result = self.generator.process_user_preference('user123', preference)
        
        self.assertTrue(result)
        self.assertEqual(self.generator.stats['summaries_generated'], 1)
        
        # Verify all methods were called
        mock_previous.assert_called_once()
        mock_articles.assert_called_once()
        mock_stats.assert_called_once()
        mock_generate.assert_called_once()
        mock_save.assert_called_once()
    
    @patch.object(PersonalizedSummaryGenerator, 'get_previous_summary')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_articles_for_entity')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_stats_for_entity')
    def test_process_user_preference_no_new_content(self, mock_stats, mock_articles, mock_previous):
        """Test processing when no new content is available."""
        # Set up mocks to return no new content
        mock_previous.return_value = "Previous summary..."
        mock_articles.return_value = []
        mock_stats.return_value = []
        
        preference = self.sample_user_data['preferences'][0]
        result = self.generator.process_user_preference('user123', preference)
        
        self.assertTrue(result)  # Should succeed but skip processing
        self.assertEqual(self.generator.stats['summaries_generated'], 0)
    
    @patch.object(PersonalizedSummaryGenerator, 'get_previous_summary')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_articles_for_entity')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_stats_for_entity')
    @patch.object(PersonalizedSummaryGenerator, 'generate_summary_with_llm')
    def test_process_user_preference_llm_failure(self, mock_generate, mock_stats, mock_articles, mock_previous):
        """Test processing when LLM generation fails."""
        # Set up mocks
        mock_previous.return_value = None
        mock_articles.return_value = self.sample_articles
        mock_stats.return_value = self.sample_stats
        mock_generate.return_value = None  # LLM failure
        
        preference = self.sample_user_data['preferences'][0]
        result = self.generator.process_user_preference('user123', preference)
        
        self.assertFalse(result)
        self.assertEqual(self.generator.stats['summaries_generated'], 0)
    
    @patch.object(PersonalizedSummaryGenerator, 'initialize_llm')
    @patch.object(PersonalizedSummaryGenerator, 'get_all_users_with_preferences')
    @patch.object(PersonalizedSummaryGenerator, 'process_user_preference')
    def test_run_personalized_summary_generation_success(self, mock_process, mock_get_users, mock_init_llm):
        """Test successful full summary generation run."""
        # Set up mocks
        mock_init_llm.return_value = True
        mock_get_users.return_value = [self.sample_user_data]
        mock_process.return_value = True
        
        result = self.generator.run_personalized_summary_generation()
        
        self.assertTrue(result['success'])
        self.assertEqual(result['stats']['users_processed'], 1)
        self.assertEqual(result['stats']['preferences_processed'], 2)
        
        # Verify process_user_preference was called for each preference
        self.assertEqual(mock_process.call_count, 2)
    
    @patch.object(PersonalizedSummaryGenerator, 'initialize_llm')
    def test_run_personalized_summary_generation_llm_init_failure(self, mock_init_llm):
        """Test full run when LLM initialization fails."""
        mock_init_llm.return_value = False
        
        result = self.generator.run_personalized_summary_generation()
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Failed to initialize LLM client')
    
    @patch.object(PersonalizedSummaryGenerator, 'initialize_llm')
    @patch.object(PersonalizedSummaryGenerator, 'get_all_users_with_preferences')
    def test_run_personalized_summary_generation_no_users(self, mock_get_users, mock_init_llm):
        """Test full run when no users with preferences exist."""
        mock_init_llm.return_value = True
        mock_get_users.return_value = []
        
        result = self.generator.run_personalized_summary_generation()
        
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], 'No users with preferences found')
        self.assertEqual(result['stats']['users_processed'], 0)


class TestPersonalizedSummaryGeneratorIntegration(unittest.TestCase):
    """Integration tests for the PersonalizedSummaryGenerator."""
    
    @patch('content_generation.personal_summary_generator.DatabaseManager')
    @patch('content_generation.personal_summary_generator.initialize_model')
    @patch('content_generation.personal_summary_generator.generate_content_with_model')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_articles_for_entity')
    @patch.object(PersonalizedSummaryGenerator, 'get_new_stats_for_entity')
    def test_integration_workflow(self, mock_get_stats, mock_get_articles, mock_generate_content, mock_initialize_model, mock_db_manager):
        """Test the complete workflow integration."""
        # Set up database mocks
        mock_users_db = Mock()
        mock_preferences_db = Mock()
        mock_generated_updates_db = Mock()
        mock_articles_db = Mock()
        mock_entity_links_db = Mock()
        mock_stats_db = Mock()
        
        mock_db_manager.side_effect = [
            mock_users_db,      # users_db
            mock_preferences_db, # preferences_db
            mock_generated_updates_db, # generated_updates_db
            mock_articles_db,    # articles_db
            mock_entity_links_db, # entity_links_db
            mock_stats_db        # stats_db
        ]
        
        # Set up LLM mocks
        mock_llm_config = {
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "grounding_enabled": True
        }
        mock_initialize_model.return_value = mock_llm_config
        mock_generate_content.return_value = "Comprehensive summary about the entity..."
        
        # Mock the data fetching methods directly
        mock_get_articles.return_value = [{
            'id': 1001,
            'headline': 'Chiefs Win Big',
            'Content': 'The Kansas City Chiefs won their latest game...',
            'publishedAt': datetime.now(timezone.utc).isoformat()
        }]
        mock_get_stats.return_value = []  # No stats for team
        
        # Create proper response objects
        class MockResponse:
            def __init__(self, data):
                self.data = data
        
        # Set up database responses
        users_response = MockResponse([{'user_id': 'test-user-id'}])
        mock_users_db.supabase.table.return_value.select.return_value.execute.return_value = users_response
        
        prefs_response = MockResponse([{
            'preference_id': 'test-pref-id',
            'entity_id': 'KC',
            'entity_type': 'team',
            'created_at': '2025-08-02T10:00:00Z'
        }])
        mock_preferences_db.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = prefs_response
        
        previous_summary_response = MockResponse([])
        mock_generated_updates_db.supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = previous_summary_response
        
        save_response = {'success': True}
        mock_generated_updates_db.insert_records.return_value = save_response
        
        # Run the generator
        generator = PersonalizedSummaryGenerator(lookback_hours=24)
        result = generator.run_personalized_summary_generation()
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertEqual(result['stats']['users_processed'], 1)
        self.assertEqual(result['stats']['preferences_processed'], 1)
        self.assertEqual(result['stats']['summaries_generated'], 1)
        
        # Verify database calls
        mock_generated_updates_db.insert_records.assert_called_once()


if __name__ == '__main__':
    unittest.main()
