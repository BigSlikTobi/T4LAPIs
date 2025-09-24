"""
LLM initialization module for DeepSeek API integration.

This module handles the setup and configuration for the DeepSeek chat model
used in LLM-enhanced entity extraction.
"""

import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass


class DeepSeekLLM:
    """DeepSeek LLM client for entity extraction."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DeepSeek LLM client.
        
        Args:
            api_key: DeepSeek API key. If None, will try to get from environment.
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable.")
        
        # Initialize OpenAI client with DeepSeek configuration
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        
        self.model = "deepseek-chat"
        self.logger.info("DeepSeek LLM client initialized successfully")
    
    def extract_entities(self, article_text: str, max_retries: int = 3) -> Dict[str, Any]:
        """Extract NFL entities from article text using DeepSeek LLM.
        
        Args:
            article_text: The article text to analyze
            max_retries: Maximum number of retry attempts for API calls
            
        Returns:
            Dictionary with 'players' and 'teams' lists, or None if extraction fails
        """
        if not article_text or not article_text.strip():
            self.logger.warning("Empty article text provided")
            return {'players': [], 'teams': []}
        
        # Create the prompt for entity extraction
        prompt = self._create_extraction_prompt(article_text)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Attempting entity extraction (attempt {attempt + 1}/{max_retries})")
                
                token_param = "max_completion_tokens" if self.model.lower().startswith("gpt-5") else "max_tokens"
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert NFL analyst who specializes in extracting player and team names from news articles. You always respond with valid JSON."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,  # Low temperature for consistent results
                    stream=False,
                    **{token_param: 1000},
                )
                
                # Extract the response content
                content = response.choices[0].message.content
                
                if not content:
                    self.logger.warning(f"Empty response from LLM (attempt {attempt + 1})")
                    continue
                
                # Parse the JSON response
                entities = self._parse_llm_response(content)
                
                if entities is not None:
                    self.logger.info(f"Successfully extracted entities: {len(entities.get('players', []))} players, {len(entities.get('teams', []))} teams")
                    return entities
                else:
                    self.logger.warning(f"Failed to parse LLM response (attempt {attempt + 1})")
                    
            except Exception as e:
                self.logger.error(f"Error during entity extraction (attempt {attempt + 1}): {e}")
                
                if attempt == max_retries - 1:
                    self.logger.error("All retry attempts failed for entity extraction")
                    return {'players': [], 'teams': []}
        
        return {'players': [], 'teams': []}
    
    def _create_extraction_prompt(self, article_text: str) -> str:
        """Create the prompt for LLM entity extraction.
        
        Args:
            article_text: The article text to include in the prompt
            
        Returns:
            Formatted prompt string
        """
        # Truncate article if too long to avoid token limits
        max_article_length = 3000  # Conservative limit
        if len(article_text) > max_article_length:
            article_text = article_text[:max_article_length] + "..."
            self.logger.debug("Article text truncated for LLM processing")
        
        prompt = f"""
        From the following NFL news article, extract all mentioned NFL players and teams.

        IMPORTANT RULES:
        1.  **Players**: Extract the most complete name available. If only a last name is used (e.g., "Worthy"), extract the last name. If a full name is used, extract the full name.
        2.  **Teams**: Always extract the full team name (e.g., "Kansas City Chiefs", not "Chiefs").
        3.  **Accuracy**: Do not include non-NFL personnel like coaches or commentators.
        4.  **Format**: Respond with ONLY a valid JSON object containing two lists: "players" and "teams". Do not add duplicates.

        Article:
        '''{article_text}'''

        JSON Output:
        """
        
        return prompt
    
    def _parse_llm_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM JSON response.
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            Parsed dictionary with players and teams, or None if parsing fails
        """
        import json
        
        try:
            # Clean the response content
            cleaned_content = response_content.strip()
            
            # Try to find JSON content if wrapped in other text
            if '```json' in cleaned_content:
                # Extract JSON from code block
                start = cleaned_content.find('```json') + 7
                end = cleaned_content.find('```', start)
                if end > start:
                    cleaned_content = cleaned_content[start:end].strip()
            elif '```' in cleaned_content:
                # Extract JSON from generic code block
                start = cleaned_content.find('```') + 3
                end = cleaned_content.find('```', start)
                if end > start:
                    cleaned_content = cleaned_content[start:end].strip()
            
            # Find JSON object boundaries
            if not cleaned_content.startswith('{'):
                # Look for the first opening brace
                start_brace = cleaned_content.find('{')
                if start_brace >= 0:
                    cleaned_content = cleaned_content[start_brace:]
            
            if not cleaned_content.endswith('}'):
                # Look for the last closing brace
                end_brace = cleaned_content.rfind('}')
                if end_brace >= 0:
                    cleaned_content = cleaned_content[:end_brace + 1]
            
            # Parse JSON
            entities = json.loads(cleaned_content)
            
            # Validate structure
            if not isinstance(entities, dict):
                self.logger.error("LLM response is not a dictionary")
                return None
            
            if 'players' not in entities or 'teams' not in entities:
                self.logger.error("LLM response missing required keys 'players' or 'teams'")
                return None
            
            if not isinstance(entities['players'], list) or not isinstance(entities['teams'], list):
                self.logger.error("LLM response 'players' and 'teams' must be lists")
                return None
            
            # Clean and validate the lists - preserve structure if dictionaries, convert to strings if simple values
            cleaned_players = []
            for player in entities['players']:
                if player:
                    if isinstance(player, dict):
                        # Keep dictionary structure for structured responses
                        cleaned_players.append(player)
                    else:
                        # Convert simple values to strings
                        player_str = str(player).strip()
                        if player_str:
                            cleaned_players.append(player_str)
            
            cleaned_teams = []
            for team in entities['teams']:
                if team:
                    if isinstance(team, dict):
                        # Keep dictionary structure for structured responses
                        cleaned_teams.append(team)
                    else:
                        # Convert simple values to strings
                        team_str = str(team).strip()
                        if team_str:
                            cleaned_teams.append(team_str)
            
            entities['players'] = cleaned_players
            entities['teams'] = cleaned_teams
            
            # Debug: log the final entities structure
            self.logger.debug(f"Cleaned entities structure: players={entities['players']}, teams={entities['teams']}")
            
            return entities
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw response content: {response_content}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing LLM response: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test the connection to DeepSeek API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.logger.info("Testing DeepSeek API connection...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful'."}
                ],
                max_tokens=10,
                stream=False
            )
            
            content = response.choices[0].message.content
            self.logger.info(f"DeepSeek API connection test response: {content}")
            return True
            
        except Exception as e:
            self.logger.error(f"DeepSeek API connection test failed: {e}")
            return False


def get_deepseek_client(api_key: Optional[str] = None) -> DeepSeekLLM:
    """Get a configured DeepSeek LLM client.
    
    Args:
        api_key: Optional API key. If None, will use environment variable.
        
    Returns:
        Configured DeepSeekLLM client
    """
    return DeepSeekLLM(api_key=api_key)


if __name__ == "__main__":
    # Test the LLM connection
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        client = get_deepseek_client()
        
        if client.test_connection():
            print("✅ DeepSeek API connection successful!")
            
            # Test entity extraction with sample text
            sample_text = "Patrick Mahomes threw for 300 yards as the Kansas City Chiefs defeated the San Francisco 49ers 31-20."
            entities = client.extract_entities(sample_text)
            
            print(f"✅ Entity extraction test successful!")
            print(f"Players found: {entities['players']}")
            print(f"Teams found: {entities['teams']}")
        else:
            print("❌ DeepSeek API connection failed!")
            
    except Exception as e:
        print(f"❌ Error testing DeepSeek client: {e}")
