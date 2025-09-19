#TODO: Grounding is not working with light model and not necessary for translation

import os
import logging
from typing import Optional, Dict, Any, Union
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def initialize_model(provider: str, model_type: str = "default", grounding_enabled: bool = True, api_key: Optional[str] = None):
    """
    Initialize a model (Gemini with optional grounding or DeepSeek) using the new client configuration.

    Args:
        provider (str): The provider name. 'gemini' or 'deepseek' are supported.
        model_type (str): The type of model to use ("default", "lite", or "flash" for Gemini; "chat" for DeepSeek).
        grounding_enabled (bool): Whether to enable Google Search grounding (Gemini only).
        api_key (str, optional): API key. If None, will use environment variable.

    Returns:
        dict: A dictionary containing the model configuration and client.

    Raises:
        ValueError: If the provider or model_type is unsupported, or if the API key is missing.
    """
    logger = logging.getLogger(__name__)
    
    if provider.lower() == "gemini":
        return _initialize_gemini(model_type, grounding_enabled, api_key, logger)
    elif provider.lower() == "deepseek":
        return _initialize_deepseek(model_type, api_key, logger)
    else:
        raise ValueError("Unsupported provider. Choose 'gemini' or 'deepseek'.")


def _initialize_gemini(model_type: str = "default", grounding_enabled: bool = True, api_key: Optional[str] = None, logger=None):
    """Initialize Gemini model with optional Google Search grounding."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if model_type == "lite":
        selected_model = "gemini-2.5-flash-lite"  # Use available model
        logger.info("Using Gemini 2.5 Flash Lite model")
    elif model_type == "flash":
        selected_model = "gemini-2.5-flash"
        logger.info("Using Gemini 2.5 Flash model")
    elif model_type == "default":
        selected_model = "gemini-2.5-pro"
        logger.info("Using Gemini 2.5 Pro model")
    else:
        raise ValueError("Unsupported Gemini model type. Choose 'default', 'lite', or 'flash'.")

    logger.info(f"Initializing Gemini model: {selected_model} with Google Search Grounding: {grounding_enabled}")
    # Support both GEMINI_API_KEY (preferred) and GOOGLE_API_KEY (legacy/alias)
    google_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set (GOOGLE_API_KEY is also accepted as an alias).")

    # Configure the Gemini client
    genai.configure(api_key=google_api_key)
    
    # Create model configuration with tools if grounding enabled
    tools = []
    if grounding_enabled:
        # Note: For now, we'll disable grounding as it requires specific setup
        # tools = ['google_search_retrieval']  # This may require special configuration
        logger.warning("Google Search grounding disabled for now - requires additional setup")
        grounding_enabled = False

    return {
        "provider": "gemini",
        "model_name": selected_model,
        "model": genai.GenerativeModel(selected_model),
        "client": genai,
        "grounding_enabled": grounding_enabled,
        "tools": tools
    }


def _initialize_deepseek(model_type: str = "chat", api_key: Optional[str] = None, logger=None):
    """Initialize DeepSeek model."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if model_type != "chat":
        raise ValueError("DeepSeek only supports 'chat' model type.")
    
    selected_model = "deepseek-chat"
    logger.info(f"Initializing DeepSeek model: {selected_model}")
    
    deepseek_api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
    if not deepseek_api_key:
        raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable.")
    
    # Initialize OpenAI client with DeepSeek configuration
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    
    return {
        "provider": "deepseek",
        "model_name": selected_model,
        "model": selected_model,
        "client": client,
        "grounding_enabled": False,
        "tools": []
    }


def generate_content_with_model(model_config: Dict[str, Any], messages: list, temperature: float = 0.1) -> str:
    """Generate content using the configured model.
    
    Args:
        model_config: Model configuration returned by initialize_model
        messages: List of message dictionaries with 'role' and 'content'
        temperature: Sampling temperature
        
    Returns:
        Generated content as string
    """
    logger = logging.getLogger(__name__)
    
    if model_config["provider"] == "gemini":
        return _generate_gemini_content(model_config, messages, temperature, logger)
    elif model_config["provider"] == "deepseek":
        return _generate_deepseek_content(model_config, messages, temperature, logger)
    else:
        raise ValueError(f"Unsupported provider: {model_config['provider']}")


def _generate_gemini_content(model_config: Dict[str, Any], messages: list, temperature: float, max_tokens: int, logger) -> str:
    """Generate content using Gemini model."""
    try:
        # Convert messages to prompt format for Gemini
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                user_messages.append(msg["content"])
            elif msg["role"] == "assistant":
                # For conversation history, we'd handle this differently
                pass
        
        # Create the prompt
        if system_message:
            prompt = f"System: {system_message}\n\nUser: {' '.join(user_messages)}"
        else:
            prompt = ' '.join(user_messages)
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Generate content
        model = model_config["model"]
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating Gemini content: {e}")
        raise


def _generate_deepseek_content(model_config: Dict[str, Any], messages: list, temperature: float, max_tokens: int, logger) -> str:
    """Generate content using DeepSeek model."""
    try:
        response = model_config["client"].chat.completions.create(
            model=model_config["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating DeepSeek content: {e}")
        raise


# Legacy functions for backward compatibility
def initialize_gemini_model(model_type: str = "default", grounding_enabled: bool = True, api_key: Optional[str] = None):
    """Legacy function - use initialize_model('gemini', ...) instead."""
    return initialize_model("gemini", model_type, grounding_enabled, api_key)


def initialize_deepseek_model(api_key: Optional[str] = None):
    """Legacy function - use initialize_model('deepseek', ...) instead."""
    return initialize_model("deepseek", "chat", False, api_key)


class DeepSeekLLM:
    """DeepSeek LLM client for entity extraction and content generation."""
    
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
                    stream=False
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
        """Create the prompt for LLM entity extraction."""
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
        """Parse the LLM JSON response."""
        import json
        
        try:
            # Clean the response content
            cleaned_content = response_content.strip()
            
            # Try to find JSON content if wrapped in other text
            if '```json' in cleaned_content:
                start = cleaned_content.find('```json') + 7
                end = cleaned_content.find('```', start)
                if end > start:
                    cleaned_content = cleaned_content[start:end].strip()
            elif '```' in cleaned_content:
                start = cleaned_content.find('```') + 3
                end = cleaned_content.find('```', start)
                if end > start:
                    cleaned_content = cleaned_content[start:end].strip()
            
            # Find JSON object boundaries
            if not cleaned_content.startswith('{'):
                start_brace = cleaned_content.find('{')
                if start_brace >= 0:
                    cleaned_content = cleaned_content[start_brace:]
            
            if not cleaned_content.endswith('}'):
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
            
            # Clean and validate the lists
            cleaned_players = []
            for player in entities['players']:
                if player:
                    if isinstance(player, dict):
                        cleaned_players.append(player)
                    else:
                        player_str = str(player).strip()
                        if player_str:
                            cleaned_players.append(player_str)
            
            cleaned_teams = []
            for team in entities['teams']:
                if team:
                    if isinstance(team, dict):
                        cleaned_teams.append(team)
                    else:
                        team_str = str(team).strip()
                        if team_str:
                            cleaned_teams.append(team_str)
            
            entities['players'] = cleaned_players
            entities['teams'] = cleaned_teams
            
            return entities
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing LLM response: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test the connection to DeepSeek API."""
        try:
            self.logger.info("Testing DeepSeek API connection...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful'."}
                ],
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
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test Gemini
        print("Testing Gemini model...")
        gemini_config = initialize_model("gemini", "flash", grounding_enabled=True)
        print(f"  ✅ Gemini Model Initialized:")
        print(f"    Model Name: {gemini_config['model_name']}")
        print(f"    Grounding Enabled: {gemini_config['grounding_enabled']}")
        
        # Test DeepSeek
        print("\nTesting DeepSeek model...")
        deepseek_config = initialize_model("deepseek", "chat", grounding_enabled=False)
        print(f"  ✅ DeepSeek Model Initialized:")
        print(f"    Model Name: {deepseek_config['model_name']}")
        print(f"    Grounding Enabled: {deepseek_config['grounding_enabled']}")
        
        # Test content generation with DeepSeek (simpler test)
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, this is a test!' and nothing else."}
        ]
        
        print("\nTesting DeepSeek content generation...")
        response = generate_content_with_model(deepseek_config, test_messages, temperature=0.1)
        print(f"  ✅ DeepSeek Response: {response}")
        
    except Exception as e:
        print(f"❌ Error during initialization test: {e}")
