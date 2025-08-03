"""Tests for LLM-enhanced entity linking functionality."""

import unittest
from unittest.mock import Mock, patch
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the modules to test
from scripts.llm_entity_linker import LLMEntityLinker, LLMEntityMatch


class TestLLMEntityLinker(unittest.TestCase):
    """Test cases for the LLMEntityLinker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample data for testing
        self.sample_article = {
            'id': 1,
            'title': 'Test Article',
            'Content': 'Patrick Mahomes threw for 300 yards as the Kansas City Chiefs defeated the San Francisco 49ers 31-20.',
            'url': 'https://example.com/test'
        }
        
        # Mock entity dictionary (actual format from build_entity_dictionary)
        self.mock_entity_dict = {
            'Patrick Mahomes': '00-0033873',  # Player ID (long, not all uppercase)
            'Travis Kelce': '00-0034857',     # Player ID
            'Kansas City Chiefs': 'KC',       # Team ID (short, uppercase)
            'San Francisco 49ers': 'SF'       # Team ID
        }
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    def test_init(self, mock_get_logger, mock_db_manager):
        """Test LLMEntityLinker initialization."""
        # Set up mocks
        mock_articles_db = Mock()
        mock_links_db = Mock()
        mock_db_manager.side_effect = [mock_articles_db, mock_links_db]
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Initialize linker
        linker = LLMEntityLinker(batch_size=25)
        
        # Verify initialization
        self.assertEqual(linker.batch_size, 25)
        self.assertEqual(linker.articles_db, mock_articles_db)
        self.assertEqual(linker.links_db, mock_links_db)
        self.assertEqual(linker.logger, mock_logger)
        self.assertIsNone(linker.llm_client)
        self.assertEqual(linker.entity_dict, {})
        
        # Verify database managers were created correctly
        expected_calls = [
            unittest.mock.call("SourceArticles"),
            unittest.mock.call("article_entity_links")
        ]
        mock_db_manager.assert_has_calls(expected_calls)
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    @patch('scripts.llm_entity_linker.get_deepseek_client')
    @patch('scripts.llm_entity_linker.build_entity_dictionary')
    def test_initialize_llm_and_entities_success(self, mock_build_entity_dict, mock_get_deepseek_client, mock_get_logger, mock_db_manager):
        """Test successful initialization of LLM and entity dictionary."""
        # Set up mocks
        mock_llm_client = Mock()
        mock_get_deepseek_client.return_value = mock_llm_client
        mock_build_entity_dict.return_value = self.mock_entity_dict
        
        linker = LLMEntityLinker()
        result = linker.initialize_llm_and_entities()
        
        self.assertTrue(result)
        self.assertEqual(linker.llm_client, mock_llm_client)
        self.assertEqual(linker.entity_dict, self.mock_entity_dict)
        
        # Verify calls
        mock_get_deepseek_client.assert_called_once()
        mock_build_entity_dict.assert_called_once()
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    @patch('scripts.llm_entity_linker.get_deepseek_client')
    @patch('scripts.llm_entity_linker.build_entity_dictionary')
    def test_initialize_llm_and_entities_llm_failure(self, mock_build_entity_dict, mock_get_deepseek_client, mock_get_logger, mock_db_manager):
        """Test initialization failure when entity dictionary building fails."""
        # Set up mocks - entity dict building fails
        mock_llm_client = Mock()
        mock_get_deepseek_client.return_value = mock_llm_client
        mock_build_entity_dict.return_value = {}  # Empty dict indicates failure
        
        linker = LLMEntityLinker()
        result = linker.initialize_llm_and_entities()
        
        self.assertFalse(result)
        mock_build_entity_dict.assert_called_once()
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    def test_extract_entities_with_llm(self, mock_get_logger, mock_db_manager):
        """Test LLM entity extraction method."""
        linker = LLMEntityLinker()
        
        # Mock the LLM client
        mock_llm_client = Mock()
        mock_llm_client.extract_entities.return_value = {
            'players': ['Patrick Mahomes', 'Travis Kelce'],
            'teams': ['Kansas City Chiefs']
        }
        linker.llm_client = mock_llm_client
        
        players, teams = linker.extract_entities_with_llm("Test article content")
        
        self.assertEqual(players, ['Patrick Mahomes', 'Travis Kelce'])
        self.assertEqual(teams, ['Kansas City Chiefs'])
        mock_llm_client.extract_entities.assert_called_once_with("Test article content")
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    def test_validate_and_link_entities(self, mock_get_logger, mock_db_manager):
        """Test entity validation and linking method."""
        linker = LLMEntityLinker()
        linker.entity_dict = self.mock_entity_dict
        linker.entity_dict_lower = {k.lower(): v for k, v in self.mock_entity_dict.items()}
        
        players = ['Patrick Mahomes', 'Unknown Player']
        teams = ['Kansas City Chiefs', 'Unknown Team']
        
        matches = linker.validate_and_link_entities(players, teams)
        
        # Should only return valid entities
        self.assertEqual(len(matches), 2)
        
        # Check the matches
        entity_names = [match.entity_name for match in matches]
        self.assertIn('Patrick Mahomes', entity_names)
        self.assertIn('Kansas City Chiefs', entity_names)
        
        # Check specific match details
        mahomes_match = next(match for match in matches if match.entity_name == 'Patrick Mahomes')
        self.assertEqual(mahomes_match.entity_id, '00-0033873')
        self.assertEqual(mahomes_match.entity_type, 'player')
        
        chiefs_match = next(match for match in matches if match.entity_name == 'Kansas City Chiefs')
        self.assertEqual(chiefs_match.entity_id, 'KC')
        self.assertEqual(chiefs_match.entity_type, 'team')
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    def test_validate_and_link_entities_empty_dict(self, mock_get_logger, mock_db_manager):
        """Test entity validation with empty dictionary."""
        linker = LLMEntityLinker()
        linker.entity_dict = {}
        linker.entity_dict_lower = {}
        
        players = ['Patrick Mahomes']
        teams = ['Kansas City Chiefs']
        
        matches = linker.validate_and_link_entities(players, teams)
        
        self.assertEqual(matches, [])
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    @patch('scripts.llm_entity_linker.uuid.uuid4')
    def test_create_entity_links(self, mock_uuid, mock_get_logger, mock_db_manager):
        """Test creation of entity links in database."""
        # Set up mocks
        mock_links_db = Mock()
        mock_links_db.insert_records.return_value = {'success': True}
        mock_db_manager.side_effect = [Mock(), mock_links_db]
        mock_uuid.return_value = 'test-uuid-1234'
        
        linker = LLMEntityLinker()
        
        matches = [
            LLMEntityMatch('Patrick Mahomes', '00-0033873', 'player'),
            LLMEntityMatch('Kansas City Chiefs', 'KC', 'team')
        ]
        
        result = linker.create_entity_links(123, matches)
        
        # Verify success
        self.assertTrue(result)
        
        # Verify database insert was called
        mock_links_db.insert_records.assert_called_once()
        
        # Check the records that were inserted
        call_args = mock_links_db.insert_records.call_args[0][0]  # First argument
        self.assertEqual(len(call_args), 2)
        
        # Check first record (player)
        first_record = call_args[0]
        self.assertEqual(first_record['article_id'], 123)
        self.assertEqual(first_record['entity_id'], '00-0033873')
        self.assertEqual(first_record['entity_type'], 'player')
        self.assertEqual(first_record['link_id'], 'test-uuid-1234')
        
        # Check second record (team)
        second_record = call_args[1]
        self.assertEqual(second_record['article_id'], 123)
        self.assertEqual(second_record['entity_id'], 'KC')
        self.assertEqual(second_record['entity_type'], 'team')
        self.assertEqual(second_record['link_id'], 'test-uuid-1234')
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    def test_create_entity_links_empty_list(self, mock_get_logger, mock_db_manager):
        """Test creation of entity links with empty list."""
        linker = LLMEntityLinker()
        
        result = linker.create_entity_links(123, [])
        
        # Should return True for empty list
        self.assertTrue(result)
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    def test_create_entity_links_database_failure(self, mock_get_logger, mock_db_manager):
        """Test creation of entity links when database insert fails."""
        # Set up mocks
        mock_links_db = Mock()
        mock_links_db.insert_records.return_value = {'success': False}
        mock_db_manager.side_effect = [Mock(), mock_links_db]
        
        linker = LLMEntityLinker()
        
        matches = [LLMEntityMatch('Patrick Mahomes', '00-0033873', 'player')]
        
        result = linker.create_entity_links(123, matches)
        
        # Should return False on database failure
        self.assertFalse(result)


class TestLLMEntityMatch(unittest.TestCase):
    """Test cases for the LLMEntityMatch dataclass."""
    
    def test_entity_match_creation(self):
        """Test creating an LLMEntityMatch."""
        match = LLMEntityMatch('Patrick Mahomes', 'mahomes_patrick', 'player')
        
        self.assertEqual(match.entity_name, 'Patrick Mahomes')
        self.assertEqual(match.entity_id, 'mahomes_patrick')
        self.assertEqual(match.entity_type, 'player')
        self.assertEqual(match.confidence, 'high')  # Default value
    
    def test_entity_match_with_confidence(self):
        """Test creating an LLMEntityMatch with custom confidence."""
        match = LLMEntityMatch('Travis Kelce', 'kelce_travis', 'player', 'medium')
        
        self.assertEqual(match.entity_name, 'Travis Kelce')
        self.assertEqual(match.entity_id, 'kelce_travis')
        self.assertEqual(match.entity_type, 'player')
        self.assertEqual(match.confidence, 'medium')


class TestLLMEntityLinkerIntegration(unittest.TestCase):
    """Integration tests for LLM entity linking with mocked external dependencies."""
    
    @patch('scripts.llm_entity_linker.DatabaseManager')
    @patch('scripts.llm_entity_linker.get_logger')
    @patch('scripts.llm_entity_linker.get_deepseek_client')
    @patch('scripts.llm_entity_linker.build_entity_dictionary')
    @patch('scripts.llm_entity_linker.uuid.uuid4')
    def test_integration_workflow(self, mock_uuid, mock_build_entity_dict, mock_get_deepseek_client, mock_get_logger, mock_db_manager):
        """Test integration of multiple LLM entity linking components."""
        # Set up mocks
        mock_uuid.return_value = 'test-uuid-123'
        
        entity_dict = {
            'Patrick Mahomes': '00-0033873',  # Player ID
            'Kansas City Chiefs': 'KC'        # Team ID
        }
        mock_build_entity_dict.return_value = entity_dict
        
        mock_llm_client = Mock()
        mock_llm_client.extract_entities.return_value = {
            'players': ['Patrick Mahomes'],
            'teams': ['Kansas City Chiefs']
        }
        mock_get_deepseek_client.return_value = mock_llm_client
        
        mock_articles_db = Mock()
        mock_links_db = Mock()
        mock_links_db.insert_records.return_value = {'success': True}
        mock_db_manager.side_effect = [mock_articles_db, mock_links_db]
        
        # Run integration test
        linker = LLMEntityLinker()
        
        # 1. Initialize
        self.assertTrue(linker.initialize_llm_and_entities())
        
        # 2. Extract entities
        players, teams = linker.extract_entities_with_llm("Test article content")
        self.assertEqual(players, ['Patrick Mahomes'])
        self.assertEqual(teams, ['Kansas City Chiefs'])
        
        # 3. Validate and link entities
        matches = linker.validate_and_link_entities(players, teams)
        self.assertEqual(len(matches), 2)  # 1 player + 1 team
        
        # 4. Create entity links
        result = linker.create_entity_links(1, matches)
        self.assertTrue(result)
        
        # Verify all components were called correctly
        mock_get_deepseek_client.assert_called_once()
        mock_build_entity_dict.assert_called_once()
        mock_llm_client.extract_entities.assert_called_once_with("Test article content")
        mock_links_db.insert_records.assert_called_once()


if __name__ == '__main__':
    unittest.main()
