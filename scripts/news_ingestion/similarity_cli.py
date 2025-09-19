#!/usr/bin/env python3
"""CLI to calculate similarity between embeddings."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import json
from dotenv import load_dotenv

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
from src.nfl_news_pipeline.logging import AuditLogger
from src.nfl_news_pipeline.similarity import SimilarityCalculator, SimilarityMetric
from src.nfl_news_pipeline.storage.protocols import get_grouping_client


DEFAULT_SUPABASE_LIMIT = 50


def _parse_story_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def _get_supabase_client():
    storage = _build_storage(dry_run=False)
    return get_grouping_client(storage)


def _load_embeddings(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("stories", [])


def _embedding_array(story: Dict[str, object]) -> np.ndarray:
    vector = story.get("embedding") or []
    if isinstance(vector, str):
        try:
            vector = json.loads(vector)
        except Exception:
            vector = []
    return np.array(vector, dtype=float)


def _load_embeddings_from_supabase(
    *,
    story_ids: List[str],
    limit: Optional[int],
) -> List[Dict[str, object]]:
    client = _get_supabase_client()
    table = client.table(os.getenv("STORY_EMBEDDINGS_TABLE", "story_embeddings"))
    query = table.select(
        "news_url_id, embedding_vector, model_name, model_version, confidence_score, generated_at"
    )

    if story_ids:
        query = query.in_("news_url_id", story_ids)
    elif limit:
        query = query.order("generated_at", desc=True).limit(limit)
    else:
        query = query.order("generated_at", desc=True).limit(DEFAULT_SUPABASE_LIMIT)

    response = query.execute()
    rows = response.data or []

    stories: List[Dict[str, object]] = []
    for row in rows:
        vector = row.get("embedding_vector") or []
        stories.append(
            {
                "story_id": row.get("news_url_id"),
                "embedding": vector,
                "model_name": row.get("model_name"),
                "model_version": row.get("model_version"),
                "confidence_score": row.get("confidence_score"),
            }
        )
    return stories


def write_similarity_to_supabase(payload: Dict[str, object]) -> None:
    storage = _build_storage(dry_run=False)
    audit = AuditLogger(storage=storage)
    audit.log_event(
        "similarity_cli",
        message="pairs_generated",
        data=payload,
    )


def run_similarity_pipeline(
    *,
    input_path: Optional[str],
    output_path: Optional[str],
    threshold: float,
    top_k: Optional[int],
    metric: SimilarityMetric,
    from_supabase: bool = False,
    story_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    write_supabase: bool = False,
) -> int:
    if from_supabase:
        stories = _load_embeddings_from_supabase(
            story_ids=story_ids or [],
            limit=limit,
        )
    else:
        if not input_path:
            raise ValueError("--input is required when not reading from Supabase")
        stories = _load_embeddings(input_path)

    if not stories:
        print("No embeddings found in input payload.")
        return 0

    calculator = SimilarityCalculator(similarity_threshold=threshold, metric=metric)

    pairs: List[Dict[str, object]] = []

    for story_a, story_b in combinations(stories, 2):
        vector_a = _embedding_array(story_a)
        vector_b = _embedding_array(story_b)
        if vector_a.size == 0 or vector_b.size == 0:
            continue
        score = float(calculator.calculate_similarity(vector_a, vector_b))
        if calculator.is_similar(score):
            pairs.append(
                {
                    "story_id_a": story_a.get("story_id"),
                    "story_id_b": story_b.get("story_id"),
                    "similarity": score,
                }
            )

    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    if top_k is not None:
        pairs = pairs[: max(top_k, 0)]

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metric": metric.value,
        "threshold": threshold,
        "pairs": pairs,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote similarity pairs to {output_path}")
    else:
        print(json.dumps(payload, indent=2))

    print(f"Similarity summary: pairs_above_threshold={len(pairs)}")

    if write_supabase and pairs:
        write_similarity_to_supabase(payload)
        print("Logged similarity pairs to Supabase audit log.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate embedding similarity scores")
    parser.add_argument("--input", help="Path to embeddings JSON")
    parser.add_argument("--output", help="Destination path for similarity JSON")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (default: 0.5)")
    parser.add_argument("--top-k", type=int, help="Limit output to top K pairs above threshold")
    parser.add_argument("--metric", choices=[m.value for m in SimilarityMetric], default=SimilarityMetric.COSINE.value)
    parser.add_argument("--from-supabase", action="store_true", help="Load embeddings directly from Supabase")
    parser.add_argument("--story-ids", help="Comma-separated news_url_ids to load when reading from Supabase")
    parser.add_argument("--limit", type=int, help="Maximum embeddings to load from Supabase (default: 50)")
    parser.add_argument("--write-supabase", action="store_true", help="Log similarity results to Supabase audit log")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    metric = SimilarityMetric(args.metric)
    from_supabase = bool(getattr(args, "from_supabase", False))
    story_ids = _parse_story_ids(getattr(args, "story_ids", None))
    limit = getattr(args, "limit", None)

    if not from_supabase and not getattr(args, "input", None):
        parser.error("--input is required unless --from-supabase is provided")

    return run_similarity_pipeline(
        input_path=str(args.input) if getattr(args, "input", None) else None,
        output_path=getattr(args, "output", None),
        threshold=float(getattr(args, "threshold", 0.5)),
        top_k=getattr(args, "top_k", None),
        metric=metric,
        from_supabase=from_supabase,
        story_ids=story_ids,
        limit=limit,
        write_supabase=bool(getattr(args, "write_supabase", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
