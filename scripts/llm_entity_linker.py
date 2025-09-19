"""
LLM-Enhanced Entity Linker (test-friendly shim)

Implements a minimal, patchable version of the entity linker so tests can mock
dependencies on this module path without invoking network/DB.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.core.utils.database import DatabaseManager
from src.core.data.entity_linking import build_entity_dictionary
from src.core.llm.llm_init import get_deepseek_client
from src.core.utils.logging import get_logger


@dataclass
class LLMEntityMatch:
    entity_name: str
    entity_id: str
    entity_type: str  # 'player' or 'team'
    confidence: str = "high"


class LLMEntityLinker:
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.logger = get_logger(__name__)
        self.articles_db = DatabaseManager("SourceArticles")
        self.links_db = DatabaseManager("article_entity_links")
        self.llm_client = None
        self.entity_dict: Dict[str, str] = {}
        self.entity_dict_lower: Dict[str, str] = {}
        self.stats: Dict[str, Any] = {
            "articles_processed": 0,
            "llm_calls": 0,
            "entities_extracted": 0,
            "entities_validated": 0,
            "links_created": 0,
            "processing_time": 0.0,
            "llm_time": 0.0,
        }

    def initialize_llm_and_entities(self) -> bool:
        try:
            self.llm_client = get_deepseek_client()
            self.entity_dict = build_entity_dictionary() or {}
            if not self.entity_dict:
                self.logger.error("Entity dictionary could not be built or is empty. Aborting.")
                return False
            self.entity_dict_lower = {k.lower(): v for k, v in self.entity_dict.items()}
            return True
        except Exception as e:
            self.logger.critical(f"A critical error occurred during initialization: {e}")
            return False

    def extract_entities_with_llm(self, article_text: str) -> Tuple[List[str], List[str]]:
        if not self.llm_client or not article_text:
            return [], []
        entities = self.llm_client.extract_entities(article_text)
        players = entities.get("players", [])
        teams = entities.get("teams", [])
        # Normalize possible structured responses
        def _normalize(seq):
            out: List[str] = []
            for it in seq:
                if isinstance(it, dict) and "name" in it:
                    out.append(it["name"])  # ignore confidence in tests
                elif isinstance(it, str):
                    out.append(it)
            return out
        players = _normalize(players)
        teams = _normalize(teams)
        self.stats["llm_calls"] += 1
        self.stats["entities_extracted"] += len(players) + len(teams)
        return players, teams

    def validate_and_link_entities(self, players: List[str], teams: List[str]) -> List[LLMEntityMatch]:
        if not self.entity_dict_lower:
            return []
        matches: List[LLMEntityMatch] = []
        for name in players:
            ent_id = self.entity_dict_lower.get(name.lower())
            if ent_id:
                matches.append(LLMEntityMatch(name, ent_id, "player"))
        for name in teams:
            ent_id = self.entity_dict_lower.get(name.lower())
            if ent_id:
                matches.append(LLMEntityMatch(name, ent_id, "team"))
        self.stats["entities_validated"] += len(matches)
        return matches

    def create_entity_links(self, article_id: int, matches: List[LLMEntityMatch]) -> bool:
        if not matches:
            return True
        try:
            records = [
                {
                    "link_id": str(uuid.uuid4()),
                    "article_id": article_id,
                    "entity_id": m.entity_id,
                    "entity_type": m.entity_type,
                }
                for m in matches
            ]
            res = self.links_db.insert_records(records)
            ok = bool(res.get("success")) if isinstance(res, dict) else bool(res)
            if ok:
                self.stats["links_created"] += len(matches)
            return ok
        except Exception as e:
            self.logger.error(f"Failed to create entity links: {e}")
            return False

