"""URL context extraction service using LLM URL context capabilities.

Implements LLM URL context extraction with fallback to metadata-based summaries.
Supports OpenAI GPT-4o-mini and Google Gemini models with entity normalization.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from ..models import ContextSummary, ProcessedNewsItem
from .cache import ContextCache, generate_metadata_hash

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
except ImportError:
    genai = None
    GenerativeModel = None


logger = logging.getLogger(__name__)


class URLContextExtractor:
    """Extract contextual summaries from news URLs using LLM URL context capabilities.
    
    Features:
    - LLM URL context analysis with OpenAI GPT-4o-mini and Google Gemini
    - Embedding-friendly summary generation with entity normalization
    - Fallback to metadata-based summaries when LLM fails
    - Confidence scoring and caching integration
    """
    
    # NFL team name mappings for normalization
    NFL_TEAM_MAPPINGS = {
        # Common abbreviations and alternate names
        "chiefs": "Kansas City Chiefs",
        "kc": "Kansas City Chiefs",
        "kansas city": "Kansas City Chiefs",
        "49ers": "San Francisco 49ers",
        "niners": "San Francisco 49ers",
        "sf": "San Francisco 49ers",
        "san francisco": "San Francisco 49ers",
        "ravens": "Baltimore Ravens",
        "bills": "Buffalo Bills",
        "patriots": "New England Patriots",
        "pats": "New England Patriots",
        "dolphins": "Miami Dolphins",
        "jets": "New York Jets",
        "ny jets": "New York Jets",
        "steelers": "Pittsburgh Steelers",
        "browns": "Cleveland Browns",
        "bengals": "Cincinnati Bengals",
        "titans": "Tennessee Titans",
        "colts": "Indianapolis Colts",
        "texans": "Houston Texans",
        "jaguars": "Jacksonville Jaguars",
        "jags": "Jacksonville Jaguars",
        "broncos": "Denver Broncos",
        "chargers": "Los Angeles Chargers",
        "raiders": "Las Vegas Raiders",
        "cowboys": "Dallas Cowboys",
        "giants": "New York Giants",
        "ny giants": "New York Giants",
        "eagles": "Philadelphia Eagles",
        "commanders": "Washington Commanders",
        "washington": "Washington Commanders",
        "packers": "Green Bay Packers",
        "bears": "Chicago Bears",
        "lions": "Detroit Lions",
        "vikings": "Minnesota Vikings",
        "falcons": "Atlanta Falcons",
        "panthers": "Carolina Panthers",
        "saints": "New Orleans Saints",
        "bucs": "Tampa Bay Buccaneers",
        "buccaneers": "Tampa Bay Buccaneers",
        "tb": "Tampa Bay Buccaneers",
        "cardinals": "Arizona Cardinals",
        "rams": "Los Angeles Rams",
        "la rams": "Los Angeles Rams",
        "seahawks": "Seattle Seahawks",
    }
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        preferred_provider: str = "openai",
        cache: Optional[ContextCache] = None,
        enable_caching: bool = True,
    ):
        """Initialize URL context extractor.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            google_api_key: Google AI API key (defaults to GOOGLE_API_KEY env var)
            preferred_provider: Preferred LLM provider ("openai" or "google")
            cache: ContextCache instance for caching summaries
            enable_caching: Enable caching of context summaries
        """
        self.preferred_provider = preferred_provider.lower()
        self.cache = cache
        self.enable_caching = enable_caching
        
        # Initialize OpenAI client
        self.openai_client = None
        if OpenAI:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize Google AI client
        self.google_client = None
        if genai:
            api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.google_client = GenerativeModel('gemini-2.0-flash-exp')
                    logger.info("Google AI client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Google AI client: {e}")
        
        # Validate at least one provider is available
        if not self.openai_client and not self.google_client:
            logger.warning("No LLM providers initialized - will use fallback mode only")
        
        logger.info(f"URLContextExtractor initialized with provider: {preferred_provider}")
    
    async def extract_context(self, news_item: ProcessedNewsItem) -> ContextSummary:
        """Extract contextual summary from news item.
        
        Args:
            news_item: ProcessedNewsItem to analyze
            
        Returns:
            ContextSummary with generated summary and metadata
        """
        try:
            # Check cache first if enabled
            metadata_hash = None
            if self.enable_caching and self.cache:
                metadata_hash = generate_metadata_hash(news_item.title, news_item.description)
                cached_summary = self.cache.get_cached_summary(news_item.url, metadata_hash)
                if cached_summary:
                    logger.debug(f"Using cached summary for: {news_item.url}")
                    return cached_summary
            
            # Try LLM URL context extraction
            summary = await self._try_llm_url_context(news_item)
            
            # Fallback to metadata-based summary if LLM fails
            if not summary:
                logger.debug(f"LLM extraction failed, using fallback for: {news_item.url}")
                summary = self._fallback_to_metadata(news_item)
            
            # Set news_url_id to URL for now (would be actual ID in real implementation)
            summary.news_url_id = news_item.url
            
            # Store in cache if enabled
            if self.enable_caching and self.cache:
                self.cache.store_summary(summary, metadata_hash)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error extracting context for {news_item.url}: {e}")
            # Return fallback summary on any error
            fallback_summary = self._fallback_to_metadata(news_item)
            fallback_summary.news_url_id = news_item.url
            return fallback_summary
    
    async def _try_llm_url_context(self, news_item: ProcessedNewsItem) -> Optional[ContextSummary]:
        """Use LLM URL context to analyze the story.
        
        Args:
            news_item: News item to analyze
            
        Returns:
            ContextSummary if successful, None if failed
        """
        # Try preferred provider first
        if self.preferred_provider == "openai" and self.openai_client:
            result = await self._extract_with_openai(news_item)
            if result:
                return result
            
        if self.preferred_provider == "google" and self.google_client:
            result = await self._extract_with_google(news_item)
            if result:
                return result
        
        # Try alternate provider as fallback
        if self.preferred_provider == "openai" and self.google_client:
            result = await self._extract_with_google(news_item)
            if result:
                return result
                
        if self.preferred_provider == "google" and self.openai_client:
            result = await self._extract_with_openai(news_item)
            if result:
                return result
        
        return None
    
    async def _extract_with_openai(self, news_item: ProcessedNewsItem) -> Optional[ContextSummary]:
        """Extract context using OpenAI GPT-4o-mini.
        
        Args:
            news_item: News item to analyze
            
        Returns:
            ContextSummary if successful, None if failed
        """
        if not self.openai_client:
            return None
        
        try:
            prompt = self._create_llm_prompt(news_item)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert NFL news analyst who creates concise, embedding-friendly summaries from news URLs. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=30
            )
            
            if not response.choices or not response.choices[0].message.content:
                return None
            
            result = self._parse_llm_response(response.choices[0].message.content, "gpt-4o-mini", news_item)
            if result:
                logger.debug(f"OpenAI extraction successful for: {news_item.url}")
                return result
                
        except Exception as e:
            logger.error(f"OpenAI extraction failed for {news_item.url}: {e}")
        
        return None
    
    async def _extract_with_google(self, news_item: ProcessedNewsItem) -> Optional[ContextSummary]:
        """Extract context using Google Gemini.
        
        Args:
            news_item: News item to analyze
            
        Returns:
            ContextSummary if successful, None if failed
        """
        if not self.google_client:
            return None
        
        try:
            prompt = self._create_llm_prompt(news_item)
            
            response = self.google_client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 500,
                }
            )
            
            if not response.text:
                return None
            
            result = self._parse_llm_response(response.text, "gemini-2.0-flash-exp", news_item)
            if result:
                logger.debug(f"Google AI extraction successful for: {news_item.url}")
                return result
                
        except Exception as e:
            logger.error(f"Google AI extraction failed for {news_item.url}: {e}")
        
        return None
    
    def _create_llm_prompt(self, news_item: ProcessedNewsItem) -> str:
        """Create prompt for LLM URL context analysis.
        
        Args:
            news_item: News item to create prompt for
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        Analyze this NFL news URL and available metadata to create a concise, embedding-friendly summary:

        URL: {news_item.url}
        Title: {news_item.title}
        Publisher: {news_item.publisher}
        Description: {news_item.description or "Not available"}
        Publication Date: {news_item.publication_date}

        TASK: Create a contextual summary that captures the core story elements (who, what, when, where, why) for semantic similarity analysis.

        REQUIREMENTS:
        1. Use full team names (e.g., "Kansas City Chiefs", not "Chiefs")
        2. Use complete player names when possible
        3. Include key story categories (injury, trade, performance, etc.)
        4. Keep summary concise but informative (2-3 sentences)
        5. Extract key topics/themes for categorization
        6. Identify main entities (players, teams, coaches)

        RESPONSE FORMAT (JSON only):
        {{
            "summary": "Concise embedding-friendly summary",
            "entities": {{
                "players": ["Full Player Name", ...],
                "teams": ["Full Team Name", ...],
                "coaches": ["Coach Name", ...]
            }},
            "key_topics": ["topic1", "topic2", ...],
            "story_category": "injury|trade|performance|news|analysis",
            "confidence": 0.8
        }}
        """
        
        return prompt.strip()
    
    def _parse_llm_response(self, response_content: str, model_name: str, news_item: ProcessedNewsItem) -> Optional[ContextSummary]:
        """Parse LLM response into ContextSummary.
        
        Args:
            response_content: Raw LLM response
            model_name: Name of the model used
            
        Returns:
            ContextSummary if parsing successful, None otherwise
        """
        try:
            import json
            
            # Clean response content
            content = response_content.strip()
            
            # Extract JSON from code blocks if present
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()
            
            # Find JSON boundaries
            if not content.startswith("{"):
                start_brace = content.find("{")
                if start_brace >= 0:
                    content = content[start_brace:]
            
            if not content.endswith("}"):
                end_brace = content.rfind("}")
                if end_brace >= 0:
                    content = content[:end_brace + 1]
            
            # Parse JSON
            data = json.loads(content)
            
            # Extract and normalize data
            summary_text = data.get("summary", "").strip()
            if not summary_text:
                return None
            
            entities = data.get("entities", {})
            players = self._normalize_player_names(entities.get("players", []))
            teams = self._normalize_team_names(entities.get("teams", []))
            coaches = entities.get("coaches", [])
            
            # Combine all entities for the entities field
            all_entities = {}
            if players:
                all_entities["players"] = players
            if teams:
                all_entities["teams"] = teams
            if coaches:
                all_entities["coaches"] = coaches
            
            key_topics = data.get("key_topics", [])
            if isinstance(key_topics, str):
                key_topics = [key_topics]
            
            confidence = float(data.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            return ContextSummary(
                news_url_id=news_item.url,  # Set the URL as news_url_id
                summary_text=summary_text,
                llm_model=model_name,
                confidence_score=confidence,
                entities=all_entities,
                key_topics=key_topics,
                fallback_used=False,
                generated_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _fallback_to_metadata(self, news_item: ProcessedNewsItem) -> ContextSummary:
        """Generate summary from available title/description when LLM fails.
        
        Args:
            news_item: News item to create fallback summary for
            
        Returns:
            ContextSummary based on metadata
        """
        # Create summary from title and description
        summary_parts = [news_item.title]
        if news_item.description:
            summary_parts.append(news_item.description)
        
        summary_text = ". ".join(part.strip() for part in summary_parts if part.strip())
        
        # Extract basic entities from text
        entities = self._extract_entities_from_text(summary_text)
        
        # Generate basic topics from existing categories or title
        key_topics = []
        if news_item.categories:
            key_topics.extend(news_item.categories)
        else:
            # Extract basic topics from title
            title_lower = news_item.title.lower()
            if any(word in title_lower for word in ["injury", "injured", "hurt"]):
                key_topics.append("injury")
            if any(word in title_lower for word in ["trade", "traded", "signs"]):
                key_topics.append("trade")
            if any(word in title_lower for word in ["score", "touchdown", "win", "loss"]):
                key_topics.append("performance")
        
        return ContextSummary(
            news_url_id=news_item.url,  # Will be set by caller
            summary_text=summary_text,
            llm_model="metadata_fallback",
            confidence_score=0.6,  # Lower confidence for fallback
            entities=entities,
            key_topics=key_topics,
            fallback_used=True,
            generated_at=datetime.now(timezone.utc)
        )
    
    def _normalize_team_names(self, teams: List[str]) -> List[str]:
        """Normalize team names to full official names.
        
        Args:
            teams: List of team names to normalize
            
        Returns:
            List of normalized team names
        """
        normalized = []
        for team in teams:
            if not team:
                continue
            
            team_lower = team.lower().strip()
            
            # Check for exact mapping
            if team_lower in self.NFL_TEAM_MAPPINGS:
                normalized.append(self.NFL_TEAM_MAPPINGS[team_lower])
            elif team.strip():  # Keep original if no mapping found
                normalized.append(team.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for team in normalized:
            if team not in seen:
                seen.add(team)
                result.append(team)
        
        return result
    
    def _normalize_player_names(self, players: List[str]) -> List[str]:
        """Normalize player names for consistent comparison.
        
        Args:
            players: List of player names to normalize
            
        Returns:
            List of normalized player names
        """
        normalized = []
        for player in players:
            if not player:
                continue
            
            # Basic normalization - title case and trim
            player_clean = " ".join(word.capitalize() for word in player.strip().split())
            if player_clean:
                normalized.append(player_clean)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for player in normalized:
            if player not in seen:
                seen.add(player)
                result.append(player)
        
        return result
    
    def _extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract basic entities from text using pattern matching.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {"players": [], "teams": [], "coaches": []}
        
        # Extract team names from text
        text_lower = text.lower()
        for team_key, team_name in self.NFL_TEAM_MAPPINGS.items():
            if team_key in text_lower:
                entities["teams"].append(team_name)
        
        # Simple player name extraction (capitalized words that could be names)
        # This is basic - a full implementation would use NER or more sophisticated methods
        words = text.split()
        potential_names = []
        for i, word in enumerate(words):
            if (word[0].isupper() and len(word) > 2 and 
                i + 1 < len(words) and words[i + 1][0].isupper() and len(words[i + 1]) > 2):
                potential_names.append(f"{word} {words[i + 1]}")
        
        # Filter out obvious non-names (very basic filtering)
        for name in potential_names:
            name_lower = name.lower()
            if not any(team_word in name_lower for team_word in ["chiefs", "49ers", "ravens", "bills"]):
                entities["players"].append(name)
        
        return entities