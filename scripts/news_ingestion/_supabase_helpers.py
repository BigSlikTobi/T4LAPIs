"""Reusable helpers for Supabase interactions in stage CLIs."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from src.nfl_news_pipeline.models import (
    ContextSummary,
    StoryEmbedding,
    StoryGroup,
    GroupStatus,
    EMBEDDING_DIM,
)
from src.nfl_news_pipeline.group_manager import GroupStorageManager
from src.nfl_news_pipeline.storage.protocols import get_grouping_client

from .pipeline_cli import _build_storage


DEFAULT_FETCH_LIMIT = 50
NEWS_URL_PAGE_SIZE = int(os.getenv("NEWS_URL_PAGE_SIZE", "200"))


def _supabase_client():
    storage = _build_storage(dry_run=False)
    return get_grouping_client(storage)


def load_context_summaries(
    *,
    story_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[ContextSummary]:
    client = _supabase_client()
    table = client.table(os.getenv("CONTEXT_SUMMARIES_TABLE", "context_summaries"))
    query = table.select("*")
    if story_ids:
        query = query.in_("news_url_id", story_ids)
    elif limit:
        query = query.order("generated_at", desc=True).limit(limit)
    else:
        query = query.order("generated_at", desc=True).limit(DEFAULT_FETCH_LIMIT)
    response = query.execute()
    summaries: List[ContextSummary] = []
    for row in response.data or []:
        try:
            summaries.append(ContextSummary.from_db(row))
        except Exception:
            continue
    return summaries


def load_embeddings(
    *,
    story_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[StoryEmbedding]:
    client = _supabase_client()
    table = client.table(os.getenv("STORY_EMBEDDINGS_TABLE", "story_embeddings"))
    query = table.select("*")
    if story_ids:
        query = query.in_("news_url_id", story_ids)
    elif limit:
        query = query.order("generated_at", desc=True).limit(limit)
    else:
        query = query.order("generated_at", desc=True).limit(DEFAULT_FETCH_LIMIT)
    response = query.execute()
    embeddings: List[StoryEmbedding] = []
    for row in response.data or []:
        try:
            embeddings.append(StoryEmbedding.from_db(row))
        except Exception:
            continue
    return embeddings


async def write_context_summaries(summaries: Iterable[ContextSummary]) -> int:
    client = _supabase_client()
    storage = GroupStorageManager(client)
    stored = 0
    for summary in summaries:
        ok = await storage.upsert_context_summary(summary)
        if ok:
            stored += 1
    return stored


async def write_embeddings(embeddings: Iterable[StoryEmbedding]) -> int:
    client = _supabase_client()
    storage = GroupStorageManager(client)
    stored = 0
    for embedding in embeddings:
        ok = await storage.store_embedding(embedding)
        if ok:
            stored += 1
    return stored


async def write_groups(
    groups: Iterable[StoryGroup],
    memberships: Iterable[tuple[str, str, float]],
) -> int:
    client = _supabase_client()
    storage = GroupStorageManager(client)
    stored = 0
    group_map: dict[str, str] = {}
    for group in groups:
        result = await storage.store_group(group)
        group_id = group.id if group.id else (result if isinstance(result, str) else None)
        if group_id:
            stored += 1
            group_map[group.id or group_id] = group_id
    for group_id, news_url_id, similarity in memberships:
        gid = group_map.get(group_id, group_id)
        await storage.add_member_to_group(gid, news_url_id, similarity)
    return stored


def story_ids_from_embeddings(embeddings: Iterable[StoryEmbedding]) -> List[str]:
    return [embedding.news_url_id for embedding in embeddings]


def centroid_from_vectors(vectors: List[List[float]]) -> Optional[List[float]]:
    if not vectors:
        return None
    import numpy as np

    matrix = np.array(vectors, dtype=float)
    if matrix.shape[1] != EMBEDDING_DIM:
        return None
    centroid = matrix.mean(axis=0)
    return centroid.tolist()


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def load_news_without_context(
    *,
    limit: Optional[int] = None,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    client = _supabase_client()
    table = client.table(os.getenv("NEWS_URLS_TABLE", "news_urls"))

    select_fields = fields or [
        "id",
        "url",
        "title",
        "description",
        "publication_date",
        "source_name",
        "publisher",
        "raw_metadata",
    ]
    columns = ",".join(select_fields)

    page_size = NEWS_URL_PAGE_SIZE
    if limit is not None and limit < page_size:
        page_size = max(limit, 1)

    collected: List[Dict[str, object]] = []
    offset = 0
    remaining = limit

    while True:
        batch_size = page_size if remaining is None else max(min(page_size, remaining), 1)
        upper = offset + batch_size - 1
        query = (
            table.select(columns)
            .order("publication_date", desc=True)
            .range(offset, upper)
        )
        response = query.execute()
        rows = response.data or []
        if not rows:
            break

        ids = [str(row.get("id")) for row in rows if row.get("id")]
        existing = set()
        if ids:
            summaries = load_context_summaries(story_ids=ids)
            existing = {summary.news_url_id for summary in summaries}

        for row in rows:
            news_id = row.get("id")
            if not news_id or str(news_id) in existing:
                continue
            # Normalize publication date for downstream consumers
            publication = _parse_datetime(row.get("publication_date"))
            if publication:
                row["publication_date"] = publication
            else:
                row["publication_date"] = datetime.now(timezone.utc)
            collected.append(row)

        if remaining is not None:
            remaining -= len(rows)
            if remaining <= 0:
                break

        if len(rows) < batch_size:
            break

        offset += batch_size

    return collected
