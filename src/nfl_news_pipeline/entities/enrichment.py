from __future__ import annotations

from typing import Optional

from .extraction import EntitiesExtractor
from ..models import ProcessedNewsItem

try:
    from src.core.data.entity_linking import build_entity_dictionary
except Exception:  # pragma: no cover - optional dependency in some environments
    build_entity_dictionary = None  # type: ignore[assignment]


def build_entities_extractor() -> Optional[EntitiesExtractor]:
    """Return an EntitiesExtractor with the best available dictionary, if possible."""
    entity_dict = None
    if build_entity_dictionary is not None:
        try:
            entity_dict = build_entity_dictionary()
        except Exception:
            entity_dict = None
    try:
        return EntitiesExtractor(entity_dict=entity_dict)
    except Exception:
        return None


def enrich_processed_item(item: ProcessedNewsItem, extractor: Optional[EntitiesExtractor]) -> ProcessedNewsItem:
    """Populate entities, categories, and structured tags on the processed item."""
    if extractor is None:
        return item
    try:
        ents = extractor.extract(item)
    except Exception:
        return item

    players = sorted(ents.players)
    teams = sorted(ents.teams)
    topics = sorted(ents.topics)

    entities_list = list(item.entities or [])
    if players:
        entities_list.extend(players)
    if teams:
        entities_list.extend(teams)

    if entities_list:
        # Deduplicate while preserving order preference: existing first, then new
        seen_entities = set()
        deduped_entities = []
        for value in entities_list:
            if value in seen_entities:
                continue
            seen_entities.add(value)
            deduped_entities.append(value)
        item.entities = deduped_entities
    if topics:
        existing_categories = set(item.categories or [])
        item.categories = sorted(existing_categories.union(topics))

    if players or teams or topics:
        enriched_meta = dict(item.raw_metadata or {})
        enriched_meta["entity_tags"] = {
            "players": players,
            "teams": teams,
            "topics": topics,
        }
        item.raw_metadata = enriched_meta

    return item
