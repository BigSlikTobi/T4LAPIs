#!/usr/bin/env python3
"""CLI to run embedding generation on context summaries."""
from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv

from .context_extraction_cli import _parse_datetime  # reuse helper
from .pipeline_cli import _build_storage

# --------------------------------------------------------------------------------------
# Repository bootstrap
# --------------------------------------------------------------------------------------


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return candidate
    return start.parents[0]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(str(ROOT / ".env"), override=False)

# --------------------------------------------------------------------------------------
# Domain imports (after sys.path adjustments)
# --------------------------------------------------------------------------------------
from src.nfl_news_pipeline.embedding import EmbeddingGenerator
from src.nfl_news_pipeline.group_manager import GroupStorageManager
from src.nfl_news_pipeline.models import ContextSummary, StoryEmbedding, EMBEDDING_DIM
from src.nfl_news_pipeline.storage.protocols import get_grouping_client


DEFAULT_SUPABASE_LIMIT = 50
SUPABASE_CONTEXT_PAGE_SIZE = int(os.getenv("CONTEXT_SUMMARY_PAGE_SIZE", "200"))
EMBEDDING_LOOKUP_CHUNK = int(os.getenv("STORY_EMBEDDING_LOOKUP_CHUNK", "200"))


def _chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _parse_story_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def _get_supabase_client():
    storage = _build_storage(dry_run=False)
    return get_grouping_client(storage)


def _fetch_existing_embedding_ids(story_ids: Iterable[str]) -> set[str]:
    ids = [str(value) for value in story_ids if value]
    if not ids:
        return set()

    client = _get_supabase_client()
    table = client.table(os.getenv("STORY_EMBEDDINGS_TABLE", "story_embeddings"))
    existing: set[str] = set()
    chunk_size = max(1, EMBEDDING_LOOKUP_CHUNK)
    for chunk in _chunked(ids, chunk_size):
        response = table.select("news_url_id").in_("news_url_id", list(chunk)).execute()
        rows = response.data or []
        for row in rows:
            news_id = row.get("news_url_id")
            if news_id:
                existing.add(str(news_id))
    return existing


def _rows_to_summaries(rows: List[Dict[str, object]]) -> List[ContextSummary]:
    summaries: List[ContextSummary] = []
    for row in rows:
        try:
            summaries.append(ContextSummary.from_db(row))
        except Exception:
            continue
    return summaries


def _load_context_from_supabase(
    *,
    story_ids: List[str],
    limit: Optional[int],
) -> List[ContextSummary]:
    client = _get_supabase_client()
    table = client.table(os.getenv("CONTEXT_SUMMARIES_TABLE", "context_summaries"))

    if story_ids:
        response = table.select("*").in_("news_url_id", story_ids).execute()
        rows = response.data or []
        return _rows_to_summaries(rows)

    page_size = SUPABASE_CONTEXT_PAGE_SIZE
    if limit is not None and limit < page_size:
        page_size = max(limit, 1)

    collected: List[ContextSummary] = []
    offset = 0
    remaining = limit

    while True:
        batch_size = page_size if remaining is None else max(min(page_size, remaining), 1)
        upper = offset + batch_size - 1
        query = (
            table.select("*")
            .order("generated_at", desc=True)
            .range(offset, upper)
        )
        response = query.execute()
        rows = response.data or []
        if not rows:
            break

        batch = _rows_to_summaries(rows)
        collected.extend(batch)

        if remaining is not None:
            remaining -= len(batch)
            if remaining <= 0:
                break

        if len(rows) < batch_size:
            break

        offset += batch_size

    return collected


def _filter_missing_embeddings(
    summaries: List[ContextSummary],
    *,
    include_embedded: bool,
) -> List[ContextSummary]:
    if include_embedded:
        return summaries

    ids = [summary.news_url_id for summary in summaries if summary.news_url_id]
    if not ids:
        return summaries

    existing = _fetch_existing_embedding_ids(ids)
    if not existing:
        return summaries

    filtered = [summary for summary in summaries if summary.news_url_id not in existing]
    skipped = len(summaries) - len(filtered)
    if skipped:
        print(f"Skipped {skipped} summaries; embeddings already exist.")
    return filtered


async def _store_embeddings_supabase_async(embeddings: List[StoryEmbedding]) -> int:
    client = _get_supabase_client()
    storage = GroupStorageManager(client)
    stored = 0
    for embedding in embeddings:
        ok = await storage.store_embedding(embedding)
        if ok:
            stored += 1
    return stored


def store_embeddings_supabase(embeddings: List[StoryEmbedding]) -> int:
    if not embeddings:
        return 0
    return asyncio.run(_store_embeddings_supabase_async(embeddings))


def _load_context_payload(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    stories = payload.get("stories", [])
    return stories


def _summary_from_story(story: Dict[str, object]) -> ContextSummary:
    generated_at = _parse_datetime(story.get("generated_at")) or datetime.utcnow()
    return ContextSummary(
        news_url_id=str(story.get("story_id")),
        summary_text=str(story.get("summary_text") or ""),
        llm_model=str(story.get("llm_model") or "unknown"),
        confidence_score=float(story.get("confidence_score", 0.0) or 0.0),
        generated_at=generated_at,
        key_topics=story.get("key_topics") or [],
    )


def _mock_embedding(summary: ContextSummary) -> StoryEmbedding:
    digest = hashlib.sha256(summary.summary_text.encode("utf-8")).digest()
    # Repeat digest to reach EMBEDDING_DIM length
    bytes_needed = EMBEDDING_DIM
    values = []
    while len(values) < bytes_needed:
        for byte in digest:
            normalized = (byte / 255.0) * 2.0 - 1.0
            values.append(normalized)
            if len(values) >= bytes_needed:
                break
    return StoryEmbedding(
        news_url_id=summary.news_url_id,
        embedding_vector=values,
        model_name="mock-embedding",
        model_version="1.0",
        summary_text=summary.summary_text,
        confidence_score=summary.confidence_score,
        generated_at=datetime.utcnow(),
    )


async def _generate_embeddings(
    generator: EmbeddingGenerator,
    summaries: List[ContextSummary],
) -> List[StoryEmbedding]:
    ids = [summary.news_url_id for summary in summaries]
    embeddings = await generator.generate_embeddings_batch(summaries, ids)
    return embeddings


def run_embedding_pipeline(
    *,
    input_path: Optional[str],
    output_path: Optional[str],
    mock: bool,
    from_supabase: bool = False,
    story_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    write_supabase: bool = False,
    include_embedded: bool = False,
) -> int:
    if from_supabase:
        summaries = _load_context_from_supabase(
            story_ids=story_ids or [],
            limit=limit,
        )
        summaries = _filter_missing_embeddings(summaries, include_embedded=include_embedded)
    else:
        if not input_path:
            raise ValueError("--input is required when not reading from Supabase")
        stories = _load_context_payload(input_path)
        if not stories:
            print("No stories found in context payload.")
            return 0
        summaries = [_summary_from_story(story) for story in stories]

    if not summaries:
        print("No context summaries available for embedding generation.")
        return 0

    embeddings: List[StoryEmbedding]
    errors: List[str] = []

    if mock:
        embeddings = [_mock_embedding(summary) for summary in summaries]
    else:
        generator = EmbeddingGenerator(openai_api_key=os.getenv("OPENAI_API_KEY"))
        try:
            embeddings = asyncio.run(_generate_embeddings(generator, summaries))
        except Exception as exc:
            errors.append(str(exc))
            embeddings = []

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stories": [
            {
                "story_id": emb.news_url_id,
                "embedding": emb.embedding_vector,
                "model_name": emb.model_name,
                "model_version": emb.model_version,
                "confidence_score": emb.confidence_score,
            }
            for emb in embeddings
        ],
        "errors": errors,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote embeddings to {output_path}")
    else:
        print(json.dumps(payload, indent=2))

    print(
        "Embedding summary: "
        f"processed={len(embeddings)} errors={len(errors)}"
    )
    if errors:
        for err in errors[:5]:
            print(f"  - {err}")

    if write_supabase and embeddings:
        stored = store_embeddings_supabase(embeddings)
        print(f"Stored {stored} embeddings to Supabase.")

    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate embeddings from context summaries")
    parser.add_argument("--input", help="Path to context summaries JSON")
    parser.add_argument("--output", help="Destination path for embeddings JSON")
    parser.add_argument("--mock", action="store_true", help="Generate deterministic embeddings without external services")
    parser.add_argument("--from-supabase", action="store_true", help="Load context summaries directly from Supabase")
    parser.add_argument("--story-ids", help="Comma-separated list of news_url_id values to process when reading from Supabase")
    parser.add_argument("--limit", type=int, help="Maximum summaries to load from Supabase (default: 50)")
    parser.add_argument("--include-embedded", action="store_true", help="Process summaries even if embeddings already exist")
    parser.add_argument("--write-supabase", action="store_true", help="Persist generated embeddings to Supabase")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from_supabase = bool(getattr(args, "from_supabase", False))
    story_ids = _parse_story_ids(getattr(args, "story_ids", None))
    limit = getattr(args, "limit", None)

    if not from_supabase and not getattr(args, "input", None):
        parser.error("--input is required unless --from-supabase is provided")

    return run_embedding_pipeline(
        input_path=str(args.input) if getattr(args, "input", None) else None,
        output_path=getattr(args, "output", None),
        mock=bool(getattr(args, "mock", False)),
        from_supabase=from_supabase,
        story_ids=story_ids,
        limit=limit,
        write_supabase=bool(getattr(args, "write_supabase", False)),
        include_embedded=bool(getattr(args, "include_embedded", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
