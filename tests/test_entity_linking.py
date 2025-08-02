"""Tests for entity linking functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.core.data.entity_linking import EntityDictionaryBuilder, build_entity_dictionary


class TestEntityDictionaryBuilder(unittest.TestCase):
    """Test cases for the EntityDictionaryBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock DatabaseManager to prevent actual database connections
        self.db_patcher = patch('src.core.data.entity_linking.DatabaseManager')
        self.mock_db_manager = self.db_patcher.start()
        
        # Mock the database instances
        self.mock_players_db = Mock()
        self.mock_teams_db = Mock()
        self.mock_db_manager.side_effect = [self.mock_players_db, self.mock_teams_db]
        
        # Sample player data from database
        self.sample_players_data = [
            {
                'player_id': '00-0033873',
                'full_name': 'Patrick Mahomes',
                'display_name': 'P. Mahomes',
                'common_first_name': 'Patrick',
                'first_name': 'Patrick',
                'last_name': 'Mahomes'
            },
            {
                'player_id': '00-0035710',
                'full_name': 'Christian McCaffrey',
                'display_name': 'C. McCaffrey',
                'common_first_name': 'Christian',
                'first_name': 'Christian',
                'last_name': 'McCaffrey'
            },
            {
                'player_id': '00-0036355',
                'full_name': 'Josh Allen',
                'display_name': 'J. Allen',
                'common_first_name': 'Josh',
                'first_name': 'Joshua',
                'last_name': 'Allen'
            }
        ]
        
        # Sample team data from database
        self.sample_teams_data = [
            {
                'team_abbr': 'KC',
                'team_name': 'Kansas City Chiefs',
                'team_nick': 'Chiefs'
            },
            {
                'team_abbr': 'SF',
                'team_name': 'San Francisco 49ers',
                'team_nick': '49ers'
            },
            {
                'team_abbr': 'BUF',
                'team_name': 'Buffalo Bills',
                'team_nick': 'Bills'
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db_patcher.stop()
    
    def test_init_success(self):
        """Test successful initialization of EntityDictionaryBuilder."""
        builder = EntityDictionaryBuilder()
        
        self.assertEqual(builder.players_db, self.mock_players_db)
        self.assertEqual(builder.teams_db, self.mock_teams_db)
        
        # Verify DatabaseManager was called correctly
        expected_calls = [unittest.mock.call("players"), unittest.mock.call("teams")]
        self.mock_db_manager.assert_has_calls(expected_calls)
    
    def test_build_entity_dictionary_success(self):
        """Test successful building of entity dictionary."""
        # Mock database managers
        # Mock database responses
        mock_players_response = Mock()
        mock_players_response.error = None
        mock_players_response.data = self.sample_players_data
        self.mock_players_db.supabase.table.return_value.select.return_value.execute.return_value = mock_players_response
        
        mock_teams_response = Mock()
        mock_teams_response.error = None
        mock_teams_response.data = self.sample_teams_data
        self.mock_teams_db.supabase.table.return_value.select.return_value.execute.return_value = mock_teams_response
        
        builder = EntityDictionaryBuilder()
        result = builder.build_entity_dictionary()
        
        # Verify expected mappings are present
        self.assertIn('Patrick Mahomes', result)
        self.assertEqual(result['Patrick Mahomes'], '00-0033873')
        
        self.assertIn('P. Mahomes', result)
        self.assertEqual(result['P. Mahomes'], '00-0033873')
        
        self.assertIn('Christian McCaffrey', result)
        self.assertEqual(result['Christian McCaffrey'], '00-0035710')
        
        self.assertIn('C. McCaffrey', result)
        self.assertEqual(result['C. McCaffrey'], '00-0035710')
        
        self.assertIn('Josh Allen', result)
        self.assertEqual(result['Josh Allen'], '00-0036355')
        
        self.assertIn('Joshua Allen', result)  # first_name + last_name
        self.assertEqual(result['Joshua Allen'], '00-0036355')
        
        self.assertIn('McCaffrey', result)  # Last name only (length > 3)
        self.assertEqual(result['McCaffrey'], '00-0035710')
        
        # Check team mappings
        self.assertIn('KC', result)
        self.assertEqual(result['KC'], 'KC')
        
        self.assertIn('Kansas City Chiefs', result)
        self.assertEqual(result['Kansas City Chiefs'], 'KC')
        
        self.assertIn('Chiefs', result)
        self.assertEqual(result['Chiefs'], 'KC')
        
        self.assertIn('SF', result)
        self.assertEqual(result['SF'], 'SF')
        
        self.assertIn('San Francisco 49ers', result)
        self.assertEqual(result['San Francisco 49ers'], 'SF')
        
        self.assertIn('49ers', result)
        self.assertEqual(result['49ers'], 'SF')
        
        # Check some alternative team names
        self.assertIn('Niners', result)
        self.assertEqual(result['Niners'], 'SF')
        
        self.assertIn('Kansas City', result)
        self.assertEqual(result['Kansas City'], 'KC')
    
    def test_build_player_mappings_success(self):
        """Test successful building of player mappings."""
        # Mock database response
        mock_response = Mock()
        mock_response.error = None
        mock_response.data = self.sample_players_data
        self.mock_players_db.supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        builder = EntityDictionaryBuilder()
        result = builder._build_player_mappings()
        
        expected_mappings = {
            'Patrick Mahomes': '00-0033873',
            'P. Mahomes': '00-0033873',
            'Christian McCaffrey': '00-0035710',
            'C. McCaffrey': '00-0035710',
            'McCaffrey': '00-0035710',
            'Josh Allen': '00-0036355',
            'J. Allen': '00-0036355',
            'Joshua Allen': '00-0036355'
        }
        
        # Check that all expected mappings are present
        for name, expected_id in expected_mappings.items():
            self.assertIn(name, result)
            self.assertEqual(result[name], expected_id)
        
        # Verify database query was called correctly
        self.mock_players_db.supabase.table.assert_called_with("players")
        self.mock_players_db.supabase.table.return_value.select.assert_called_with(
            "player_id, full_name, display_name, common_first_name, first_name, last_name"
        )
    
    def test_build_player_mappings_database_error(self):
        """Test player mappings building with database error."""
        # Mock database response with error
        mock_response = Mock()
        mock_response.error = "Database connection failed"
        mock_response.data = None
        self.mock_players_db.supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        builder = EntityDictionaryBuilder()
        result = builder._build_player_mappings()
        
        self.assertEqual(result, {})
    
    def test_build_player_mappings_no_data(self):
        """Test player mappings building with no data."""
        # Mock database response with no data
        mock_response = Mock()
        mock_response.error = None
        mock_response.data = []
        self.mock_players_db.supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        builder = EntityDictionaryBuilder()
        result = builder._build_player_mappings()
        
        self.assertEqual(result, {})
    
    def test_build_player_mappings_with_invalid_data(self):
        """Test player mappings building with invalid/incomplete data."""
        # Mock database response with incomplete data
        invalid_players_data = [
            {
                'player_id': '00-0033873',
                'full_name': 'Patrick Mahomes',
                'display_name': 'P. Mahomes',
                'common_first_name': 'Patrick',
                'first_name': 'Patrick',
                'last_name': 'Mahomes'
            },
            {
                'player_id': None,  # Missing player_id
                'full_name': 'Invalid Player',
                'display_name': 'I. Player',
                'common_first_name': 'Invalid',
                'first_name': 'Invalid',
                'last_name': 'Player'
            },
            {
                'player_id': '00-0035710',
                'full_name': None,  # Missing full_name
                'display_name': None,
                'common_first_name': None,
                'first_name': None,
                'last_name': None
            },
            {
                'player_id': '00-0036355',
                'full_name': '   ',  # Whitespace only
                'display_name': '   ',
                'common_first_name': '   ',
                'first_name': '   ',
                'last_name': '   '
            },
            {
                'player_id': '00-0036356',
                'full_name': 'none',  # String 'none'
                'display_name': 'none',
                'common_first_name': 'none',
                'first_name': 'none',
                'last_name': 'none'
            }
        ]
        
        mock_response = Mock()
        mock_response.error = None
        mock_response.data = invalid_players_data
        self.mock_players_db.supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        builder = EntityDictionaryBuilder()
        result = builder._build_player_mappings()
        
        # Only valid player should be included
        expected_mappings = {
            'Patrick Mahomes': '00-0033873',
            'P. Mahomes': '00-0033873',
            'Mahomes': '00-0033873'  # Last name only
        }
        
        # Check that all expected mappings are present
        for name, expected_id in expected_mappings.items():
            self.assertIn(name, result)
            self.assertEqual(result[name], expected_id)
    
    def test_build_team_mappings_success(self):
        """Test successful building of team mappings."""
        # Mock database response
        mock_response = Mock()
        mock_response.error = None
        mock_response.data = self.sample_teams_data
        self.mock_teams_db.supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        builder = EntityDictionaryBuilder()
        result = builder._build_team_mappings()
        
        # Check core mappings
        self.assertIn('KC', result)
        self.assertEqual(result['KC'], 'KC')
        self.assertIn('Kansas City Chiefs', result)
        self.assertEqual(result['Kansas City Chiefs'], 'KC')
        self.assertIn('Chiefs', result)
        self.assertEqual(result['Chiefs'], 'KC')
        
        self.assertIn('SF', result)
        self.assertEqual(result['SF'], 'SF')
        self.assertIn('San Francisco 49ers', result)
        self.assertEqual(result['San Francisco 49ers'], 'SF')
        self.assertIn('49ers', result)
        self.assertEqual(result['49ers'], 'SF')
        
        # Check alternative names
        self.assertIn('Kansas City', result)
        self.assertEqual(result['Kansas City'], 'KC')
        self.assertIn('Niners', result)
        self.assertEqual(result['Niners'], 'SF')
        
        # Verify database query was called correctly
        self.mock_teams_db.supabase.table.assert_called_with("teams")
        self.mock_teams_db.supabase.table.return_value.select.assert_called_with("team_abbr, team_name, team_nick")
    
    def test_build_team_mappings_database_error(self):
        """Test team mappings building with database error."""
        # Mock database response with error
        mock_response = Mock()
        mock_response.error = "Database connection failed"
        mock_response.data = None
        self.mock_teams_db.supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        builder = EntityDictionaryBuilder()
        result = builder._build_team_mappings()
        
        self.assertEqual(result, {})
    
    def test_build_team_mappings_with_invalid_data(self):
        """Test team mappings building with invalid/incomplete data."""
        # Mock database response with incomplete data
        invalid_teams_data = [
            {
                'team_abbr': 'KC',
                'team_name': 'Kansas City Chiefs',
                'team_nick': 'Chiefs'
            },
            {
                'team_abbr': None,  # Missing team_abbr
                'team_name': 'Invalid Team',
                'team_nick': 'Invalid'
            },
            {
                'team_abbr': 'SF',
                'team_name': None,  # Missing team_name
                'team_nick': None   # Missing team_nick
            }
        ]
        
        mock_response = Mock()
        mock_response.error = None
        mock_response.data = invalid_teams_data
        self.mock_teams_db.supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        builder = EntityDictionaryBuilder()
        result = builder._build_team_mappings()
        
        # Should have KC mappings and SF abbreviation mapping
        self.assertIn('KC', result)
        self.assertEqual(result['KC'], 'KC')
        self.assertIn('Kansas City Chiefs', result)
        self.assertEqual(result['Kansas City Chiefs'], 'KC')
        self.assertIn('Chiefs', result)
        self.assertEqual(result['Chiefs'], 'KC')
        
        self.assertIn('SF', result)
        self.assertEqual(result['SF'], 'SF')
        
        # Should not have invalid team
        self.assertNotIn('Invalid Team', result)
    
    def test_get_team_alternatives(self):
        """Test getting alternative team names."""
        builder = EntityDictionaryBuilder()
        
        # Test KC alternatives
        kc_alternatives = builder._get_team_alternatives('KC', 'Kansas City Chiefs', 'Chiefs')
        self.assertIn('Kansas City', kc_alternatives)
        self.assertEqual(kc_alternatives['Kansas City'], 'KC')
        
        # Test SF alternatives
        sf_alternatives = builder._get_team_alternatives('SF', 'San Francisco 49ers', '49ers')
        self.assertIn('Niners', sf_alternatives)
        self.assertEqual(sf_alternatives['Niners'], 'SF')
        self.assertIn('San Francisco', sf_alternatives)
        self.assertEqual(sf_alternatives['San Francisco'], 'SF')
        
        # Test with missing team_abbr
        empty_alternatives = builder._get_team_alternatives(None, 'Some Team', 'Team')
        self.assertEqual(empty_alternatives, {})
        
        # Test with unknown team
        unknown_alternatives = builder._get_team_alternatives('UNK', 'Unknown Team', 'Unknown')
        self.assertEqual(unknown_alternatives, {})
    
    def test_build_entity_dictionary_partial_failure(self):
        """Test entity dictionary building with partial failures."""
        # Mock successful players response
        mock_players_response = Mock()
        mock_players_response.error = None
        mock_players_response.data = self.sample_players_data
        self.mock_players_db.supabase.table.return_value.select.return_value.execute.return_value = mock_players_response
        
        # Mock failed teams response
        mock_teams_response = Mock()
        mock_teams_response.error = "Teams table error"
        mock_teams_response.data = None
        self.mock_teams_db.supabase.table.return_value.select.return_value.execute.return_value = mock_teams_response
        
        builder = EntityDictionaryBuilder()
        result = builder.build_entity_dictionary()
        
        # Should have player data but no team data
        self.assertIn('Patrick Mahomes', result)
        self.assertEqual(result['Patrick Mahomes'], '00-0033873')
        
        self.assertNotIn('KC', result)
        self.assertNotIn('Kansas City Chiefs', result)
    
    def test_build_entity_dictionary_exception_handling(self):
        """Test entity dictionary building with exceptions."""
        # Mock exception in players query
        self.mock_players_db.supabase.table.return_value.select.return_value.execute.side_effect = Exception("Database error")
        
        # Mock successful teams response
        mock_teams_response = Mock()
        mock_teams_response.error = None
        mock_teams_response.data = self.sample_teams_data
        self.mock_teams_db.supabase.table.return_value.select.return_value.execute.return_value = mock_teams_response
        
        builder = EntityDictionaryBuilder()
        result = builder.build_entity_dictionary()
        
        # Should have team data but no player data
        self.assertNotIn('Patrick Mahomes', result)
        
        self.assertIn('KC', result)
        self.assertEqual(result['KC'], 'KC')
        self.assertIn('Kansas City Chiefs', result)
        self.assertEqual(result['Kansas City Chiefs'], 'KC')


class TestConvenienceFunction(unittest.TestCase):
    """Test cases for the convenience function."""
    
    @patch('src.core.data.entity_linking.EntityDictionaryBuilder')
    def test_build_entity_dictionary_function(self, mock_builder_class):
        """Test the convenience function."""
        mock_builder = Mock()
        mock_builder.build_entity_dictionary.return_value = {'test': 'value'}
        mock_builder_class.return_value = mock_builder
        
        result = build_entity_dictionary()
        
        mock_builder_class.assert_called_once()
        mock_builder.build_entity_dictionary.assert_called_once()
        self.assertEqual(result, {'test': 'value'})


if __name__ == '__main__':
    unittest.main()
