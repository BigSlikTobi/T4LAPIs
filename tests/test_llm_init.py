"""Tests for LLM initialization and entity extraction functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from src.core.llm.llm_init import DeepSeekLLM, get_deepseek_client


class TestDeepSeekLLM(unittest.TestCase):
    """Test cases for the DeepSeekLLM class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "test-api-key-12345"
        
        # Sample valid LLM responses
        self.valid_response_json = {
            "players": ["Patrick Mahomes", "Travis Kelce"],
            "teams": ["Kansas City Chiefs", "San Francisco 49ers"]
        }
        
        self.valid_response_text = json.dumps(self.valid_response_json)
        
        # Sample article text
        self.sample_article = "Patrick Mahomes threw for 300 yards as the Kansas City Chiefs defeated the San Francisco 49ers 31-20. Travis Kelce caught 8 passes."
    
    @patch('src.core.llm.llm_init.load_dotenv')
    @patch('src.core.llm.llm_init.OpenAI')
    def test_init_with_api_key(self, mock_openai, mock_load_dotenv):
        """Test successful initialization with provided API key."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        self.assertEqual(llm.api_key, self.test_api_key)
        self.assertEqual(llm.client, mock_client)
        self.assertEqual(llm.model, "deepseek-chat")
        
        # Verify OpenAI client was initialized correctly
        mock_openai.assert_called_once_with(
            api_key=self.test_api_key,
            base_url="https://api.deepseek.com"
        )
    
    @patch('src.core.llm.llm_init.load_dotenv')
    @patch('src.core.llm.llm_init.os.getenv')
    @patch('src.core.llm.llm_init.OpenAI')
    def test_init_with_env_variable(self, mock_openai, mock_getenv, mock_load_dotenv):
        """Test initialization with API key from environment variable."""
        mock_getenv.return_value = self.test_api_key
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM()
        
        self.assertEqual(llm.api_key, self.test_api_key)
        mock_getenv.assert_called_with('DEEPSEEK_API_KEY')
    
    @patch('src.core.llm.llm_init.load_dotenv')
    @patch('src.core.llm.llm_init.os.getenv')
    def test_init_without_api_key(self, mock_getenv, mock_load_dotenv):
        """Test initialization failure when no API key is provided."""
        mock_getenv.return_value = None
        
        with self.assertRaises(ValueError) as context:
            DeepSeekLLM()
        
        self.assertIn("DeepSeek API key not provided", str(context.exception))
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_test_connection_success(self, mock_openai):
        """Test successful connection test."""
        # Mock OpenAI client and response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Connection successful"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        result = llm.test_connection()
        
        self.assertTrue(result)
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_test_connection_failure(self, mock_openai):
        """Test connection test failure."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        result = llm.test_connection()
        
        self.assertFalse(result)
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_extract_entities_success(self, mock_openai):
        """Test successful entity extraction."""
        # Mock OpenAI client and response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.valid_response_text
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        result = llm.extract_entities(self.sample_article)
        
        self.assertEqual(result, self.valid_response_json)
        
        # Verify the API call was made correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        
        self.assertEqual(call_args[1]['model'], 'deepseek-chat')
        self.assertEqual(len(call_args[1]['messages']), 2)
        self.assertEqual(call_args[1]['temperature'], 0.1)
        self.assertEqual(call_args[1]['max_tokens'], 1000)
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_extract_entities_empty_text(self, mock_openai):
        """Test entity extraction with empty text."""
        mock_openai.return_value = Mock()
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        # Test empty string
        result = llm.extract_entities("")
        self.assertEqual(result, {'players': [], 'teams': []})
        
        # Test None
        result = llm.extract_entities(None)
        self.assertEqual(result, {'players': [], 'teams': []})
        
        # Test whitespace only
        result = llm.extract_entities("   ")
        self.assertEqual(result, {'players': [], 'teams': []})
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_extract_entities_api_error(self, mock_openai):
        """Test entity extraction with API error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        result = llm.extract_entities(self.sample_article)
        
        self.assertEqual(result, {'players': [], 'teams': []})
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_extract_entities_with_retries(self, mock_openai):
        """Test entity extraction with retries after failures."""
        # Mock client that fails twice then succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = self.valid_response_text
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            mock_response  # Third attempt succeeds
        ]
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        result = llm.extract_entities(self.sample_article, max_retries=3)
        
        self.assertEqual(result, self.valid_response_json)
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_extract_entities_max_retries_exceeded(self, mock_openai):
        """Test entity extraction when max retries are exceeded."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Persistent API Error")
        mock_openai.return_value = mock_client
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        result = llm.extract_entities(self.sample_article, max_retries=2)
        
        self.assertEqual(result, {'players': [], 'teams': []})
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
    
    def test_create_extraction_prompt(self):
        """Test prompt creation for entity extraction."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        prompt = llm._create_extraction_prompt(self.sample_article)
        
        self.assertIn(self.sample_article, prompt)
        self.assertIn("NFL news article", prompt)
        self.assertIn("JSON", prompt)
        self.assertIn("players", prompt)
        self.assertIn("teams", prompt)
    
    def test_create_extraction_prompt_truncation(self):
        """Test prompt creation with long article text."""
        long_article = "A" * 5000  # Article longer than max_article_length
        
        llm = DeepSeekLLM(api_key=self.test_api_key)
        prompt = llm._create_extraction_prompt(long_article)
        
        # Should be truncated and have ellipsis
        self.assertIn("...", prompt)
        self.assertLess(len(prompt), len(long_article) + 1000)  # Should be significantly shorter
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        result = llm._parse_llm_response(self.valid_response_text)
        
        self.assertEqual(result, self.valid_response_json)
    
    def test_parse_llm_response_json_in_code_block(self):
        """Test parsing JSON wrapped in code blocks."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        # Test with ```json code block
        wrapped_response = f"```json\n{self.valid_response_text}\n```"
        result = llm._parse_llm_response(wrapped_response)
        self.assertEqual(result, self.valid_response_json)
        
        # Test with generic ``` code block
        wrapped_response = f"```\n{self.valid_response_text}\n```"
        result = llm._parse_llm_response(wrapped_response)
        self.assertEqual(result, self.valid_response_json)
    
    def test_parse_llm_response_with_extra_text(self):
        """Test parsing JSON with extra text around it."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        response_with_extra = f"Here is the extraction:\n{self.valid_response_text}\nEnd of response."
        result = llm._parse_llm_response(response_with_extra)
        
        self.assertEqual(result, self.valid_response_json)
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        invalid_json = '{"players": ["John Doe",], "teams": [}'  # Invalid JSON
        result = llm._parse_llm_response(invalid_json)
        
        self.assertIsNone(result)
    
    def test_parse_llm_response_missing_keys(self):
        """Test parsing JSON response with missing required keys."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        # Missing 'teams' key
        incomplete_json = '{"players": ["John Doe"]}'
        result = llm._parse_llm_response(incomplete_json)
        self.assertIsNone(result)
        
        # Missing 'players' key
        incomplete_json = '{"teams": ["Some Team"]}'
        result = llm._parse_llm_response(incomplete_json)
        self.assertIsNone(result)
    
    def test_parse_llm_response_wrong_types(self):
        """Test parsing JSON response with wrong data types."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        # Players should be list, not string
        wrong_type_json = '{"players": "John Doe", "teams": ["Some Team"]}'
        result = llm._parse_llm_response(wrong_type_json)
        self.assertIsNone(result)
        
        # Teams should be list, not string
        wrong_type_json = '{"players": ["John Doe"], "teams": "Some Team"}'
        result = llm._parse_llm_response(wrong_type_json)
        self.assertIsNone(result)
    
    def test_parse_llm_response_cleans_data(self):
        """Test that response parsing cleans the data properly."""
        llm = DeepSeekLLM(api_key=self.test_api_key)
        
        # JSON with empty strings and whitespace
        messy_json = '{"players": ["John Doe", "", "  ", "Jane Smith", null], "teams": ["Team A", "   Team B   ", ""]}'
        result = llm._parse_llm_response(messy_json)
        
        expected = {
            "players": ["John Doe", "Jane Smith"],
            "teams": ["Team A", "Team B"]
        }
        self.assertEqual(result, expected)


class TestGetDeepSeekClient(unittest.TestCase):
    """Test cases for the get_deepseek_client function."""
    
    @patch('src.core.llm.llm_init.DeepSeekLLM')
    def test_get_deepseek_client_with_api_key(self, mock_deepseek_class):
        """Test getting client with provided API key."""
        mock_client = Mock()
        mock_deepseek_class.return_value = mock_client
        
        api_key = "test-key"
        result = get_deepseek_client(api_key=api_key)
        
        mock_deepseek_class.assert_called_once_with(api_key=api_key)
        self.assertEqual(result, mock_client)
    
    @patch('src.core.llm.llm_init.DeepSeekLLM')
    def test_get_deepseek_client_without_api_key(self, mock_deepseek_class):
        """Test getting client without API key (should use environment)."""
        mock_client = Mock()
        mock_deepseek_class.return_value = mock_client
        
        result = get_deepseek_client()
        
        mock_deepseek_class.assert_called_once_with(api_key=None)
        self.assertEqual(result, mock_client)


class TestLLMIntegration(unittest.TestCase):
    """Integration tests for LLM functionality."""
    
    @patch('src.core.llm.llm_init.OpenAI')
    def test_full_extraction_workflow(self, mock_openai):
        """Test the complete entity extraction workflow."""
        # Mock a realistic API response
        api_response_content = '''
        {
            "players": ["Patrick Mahomes", "Travis Kelce", "Tyreek Hill"],
            "teams": ["Kansas City Chiefs", "Miami Dolphins"]
        }
        '''
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = api_response_content
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test the full workflow
        llm = DeepSeekLLM(api_key="test-key")
        
        article = """
        Patrick Mahomes threw three touchdown passes, including two to Travis Kelce,
        as the Kansas City Chiefs defeated the Miami Dolphins 24-17. 
        Tyreek Hill had 120 receiving yards for Miami in the loss.
        """
        
        result = llm.extract_entities(article)
        
        expected = {
            "players": ["Patrick Mahomes", "Travis Kelce", "Tyreek Hill"],
            "teams": ["Kansas City Chiefs", "Miami Dolphins"]
        }
        
        self.assertEqual(result, expected)
        
        # Verify the API call parameters
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], 'deepseek-chat')
        self.assertEqual(call_args[1]['temperature'], 0.1)
        self.assertIn(article, call_args[1]['messages'][1]['content'])


if __name__ == '__main__':
    unittest.main()
