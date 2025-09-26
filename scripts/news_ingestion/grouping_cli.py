#!/usr/bin/env python3
"""CLI to form story groups from similarity results."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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
from src.nfl_news_pipeline.group_manager import GroupStorageManager
from src.nfl_news_pipeline.models import StoryGroup, GroupStatus
from src.nfl_news_pipeline.storage.protocols import get_grouping_client


def _get_supabase_client():
    storage = _build_storage(dry_run=False)
    return get_grouping_client(storage)


def _load_similarity(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_embedding(vector: object) -> List[float]:
    if isinstance(vector, str):
        try:
            vector = json.loads(vector)
        except Exception:
            vector = []
    return list(vector or [])


DEFAULT_SUPABASE_EMBEDDING_LIMIT = 200
SUPABASE_NEWS_PAGE_SIZE = int(os.getenv("NEWS_URL_PAGE_SIZE", "200"))
EMBEDDING_FETCH_CHUNK = int(os.getenv("GROUPING_EMBEDDING_LOOKUP_CHUNK", "200"))


def _chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _load_embeddings(path: Optional[str]) -> Dict[str, List[float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    embeddings: Dict[str, List[float]] = {}
    for story in payload.get("stories", []):
        embeddings[str(story.get("story_id"))] = _normalize_embedding(story.get("embedding"))
    return embeddings


def _fetch_embeddings_for_ids(
    story_ids: Iterable[str],
    *,
    limit: Optional[int] = None,
) -> Dict[str, List[float]]:
    client = _get_supabase_client()
    table = client.table(os.getenv("STORY_EMBEDDINGS_TABLE", "story_embeddings"))
    ids = [str(value) for value in story_ids if value]
    embeddings: Dict[str, List[float]] = {}

    if ids:
        chunk_size = max(1, EMBEDDING_FETCH_CHUNK)
        for chunk in _chunked(ids, chunk_size):
            response = (
                table.select("news_url_id, embedding_vector, generated_at")
                .in_("news_url_id", list(chunk))
                .execute()
            )
            rows = response.data or []
            for row in rows:
                news_id = row.get("news_url_id")
                if news_id is None:
                    continue
                embeddings[str(news_id)] = _normalize_embedding(row.get("embedding_vector"))
        return embeddings

    fetch_limit = limit or DEFAULT_SUPABASE_EMBEDDING_LIMIT
    response = (
        table.select("news_url_id, embedding_vector, generated_at")
        .order("generated_at", desc=True)
        .limit(fetch_limit)
        .execute()
    )
    rows = response.data or []
    for row in rows:
        news_id = row.get("news_url_id")
        if news_id is None:
            continue
        embeddings[str(news_id)] = _normalize_embedding(row.get("embedding_vector"))
    return embeddings


def _fetch_all_news_ids(
    *,
    limit: Optional[int],
) -> List[str]:
    client = _get_supabase_client()
    table = client.table(os.getenv("NEWS_URLS_TABLE", "news_urls"))

    page_size = SUPABASE_NEWS_PAGE_SIZE
    if limit is not None and limit < page_size:
        page_size = max(limit, 1)

    collected: List[str] = []
    offset = 0
    remaining = limit

    while True:
        batch_size = page_size if remaining is None else max(min(page_size, remaining), 1)
        upper = offset + batch_size - 1
        query = (
            table.select("id")
            .order("publication_date", desc=True)
            .range(offset, upper)
        )
        response = query.execute()
        rows = response.data or []
        if not rows:
            break

        for row in rows:
            news_id = row.get("id")
            if news_id:
                collected.append(str(news_id))

        if remaining is not None:
            remaining -= len(rows)
            if remaining <= 0:
                break

        if len(rows) < batch_size:
            break

        offset += batch_size

    return collected


def _fetch_existing_memberships(story_ids: Iterable[str]) -> Dict[str, str]:
    ids = [str(value) for value in story_ids if value]
    if not ids:
        return {}

    client = _get_supabase_client()
    table = client.table(os.getenv("STORY_GROUP_MEMBERS_TABLE", "story_group_members"))
    memberships: Dict[str, str] = {}
    chunk_size = max(1, EMBEDDING_FETCH_CHUNK)
    for chunk in _chunked(ids, chunk_size):
        response = (
            table.select("news_url_id, group_id")
            .in_("news_url_id", list(chunk))
            .execute()
        )
        rows = response.data or []
        for row in rows:
            news_id = row.get("news_url_id")
            group_id = row.get("group_id")
            if news_id and group_id:
                memberships[str(news_id)] = str(group_id)
    return memberships


def _fetch_group_metadata(group_ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
    ids = [str(value) for value in group_ids if value]
    if not ids:
        return {}

    client = _get_supabase_client()
    table = client.table(os.getenv("STORY_GROUPS_TABLE", "story_groups"))
    output: Dict[str, Dict[str, object]] = {}
    chunk_size = max(1, EMBEDDING_FETCH_CHUNK)
    for chunk in _chunked(ids, chunk_size):
        response = (
            table.select("id, member_count, centroid_embedding, tags, status")
            .in_("id", list(chunk))
            .execute()
        )
        rows = response.data or []
        for row in rows:
            group_id = row.get("id")
            if not group_id:
                continue
            output[str(group_id)] = {
                "member_count": int(row.get("member_count", 0) or 0),
                "centroid_embedding": _normalize_embedding(row.get("centroid_embedding")),
                "tags": row.get("tags") or [],
                "status": row.get("status"),
            }
    return output


def _merge_centroid(
    *,
    existing_centroid: Optional[List[float]],
    existing_count: int,
    new_vectors: List[List[float]],
) -> Optional[List[float]]:
    vectors = [np.array(vec, dtype=float) for vec in new_vectors if vec]
    if not vectors:
        return existing_centroid

    combined_new = np.sum(vectors, axis=0)

    if existing_centroid and existing_count > 0:
        try:
            base = np.array(existing_centroid, dtype=float)
        except Exception:
            base = None
        if base is not None and base.size:
            numerator = (base * float(existing_count)) + combined_new
            centroid = numerator / float(existing_count + len(vectors))
            return centroid.tolist()

    centroid = combined_new / float(len(vectors))
    return centroid.tolist()


def _union_find_build(pairs: List[Tuple[str, str]], nodes: List[str]) -> Dict[str, str]:
    parent = {node: node for node in nodes}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in pairs:
        if a in parent and b in parent:
            union(a, b)

    return {node: find(node) for node in nodes}


def _compute_centroid(members: List[str], embeddings: Dict[str, List[float]]) -> Optional[List[float]]:
    vectors = []
    for member in members:
        vector = embeddings.get(member)
        if vector:
            vectors.append(np.array(vector, dtype=float))
    if not vectors:
        return None
    centroid = np.mean(vectors, axis=0)
    return centroid.tolist()


def _best_similarity(member: str, members: List[str], pair_scores: Dict[frozenset, float]) -> float:
    best = 1.0
    for other in members:
        if other == member:
            continue
        score = pair_scores.get(frozenset({member, other}))
        if score is not None and score > best:
            best = score
    return best


def _build_pair_score_map(pairs: List[Dict[str, object]]) -> Dict[frozenset, float]:
    pair_scores: Dict[frozenset, float] = {}
    for pair in pairs:
        a = str(pair.get("story_id_a"))
        b = str(pair.get("story_id_b"))
        score = float(pair.get("similarity", 0.0))
        pair_scores[frozenset({a, b})] = score
    return pair_scores


def run_grouping_pipeline(
    *,
    similarity_path: str,
    embeddings_path: Optional[str],
    output_path: Optional[str],
    threshold: Optional[float],
    embeddings_from_supabase: bool = False,
    supabase_limit: Optional[int] = None,
    include_all_stories: bool = False,
    news_limit: Optional[int] = None,
    write_supabase: bool = False,
) -> int:
    similarity_payload = _load_similarity(similarity_path)
    pairs_data = similarity_payload.get("pairs", [])

    pair_scores = _build_pair_score_map(pairs_data)

    # Apply optional threshold override
    filtered_pairs: List[Tuple[str, str]] = []
    discovered_ids: set[str] = set()
    for pair in pairs_data:
        a = str(pair.get("story_id_a"))
        b = str(pair.get("story_id_b"))
        discovered_ids.update({a, b})

        score = float(pair.get("similarity", 0.0))
        if threshold is not None and score < threshold:
            continue
        filtered_pairs.append((a, b))

    embeddings = _load_embeddings(embeddings_path)

    candidate_ids = set(discovered_ids)
    candidate_ids.update(embeddings.keys())

    if include_all_stories:
        news_ids = _fetch_all_news_ids(limit=news_limit)
        candidate_ids.update(news_ids)

    if embeddings_from_supabase:
        if candidate_ids:
            missing_ids = sorted(story_id for story_id in candidate_ids if story_id not in embeddings)
            if missing_ids:
                fetched = _fetch_embeddings_for_ids(missing_ids, limit=supabase_limit)
                embeddings.update(fetched)
        else:
            fetched = _fetch_embeddings_for_ids([], limit=supabase_limit)
            embeddings.update(fetched)
            candidate_ids.update(fetched.keys())

    # Recompute candidate ids in case Supabase added new embeddings
    candidate_ids.update(embeddings.keys())

    existing_memberships: Dict[str, str] = {}
    existing_group_metadata: Dict[str, Dict[str, object]] = {}
    if candidate_ids and (write_supabase or embeddings_from_supabase):
        existing_memberships = _fetch_existing_memberships(candidate_ids)
        existing_group_metadata = _fetch_group_metadata(existing_memberships.values())

    if not candidate_ids:
        print("No stories discovered in similarities or embeddings; nothing to group.")
        return 0

    nodes = sorted(candidate_ids)
    root_map = _union_find_build(filtered_pairs, nodes)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for node, root in root_map.items():
        clusters[root].append(node)

    groups_payload = []
    store_tasks: List[Dict[str, object]] = []
    for index, (root, members) in enumerate(
        sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True), start=1
    ):
        existing_group_counts: Dict[str, int] = {}
        for member in members:
            group_id = existing_memberships.get(member)
            if not group_id:
                continue
            existing_group_counts[group_id] = existing_group_counts.get(group_id, 0) + 1

        assigned_group_id: Optional[str] = None
        if existing_group_counts:
            assigned_group_id = max(existing_group_counts, key=existing_group_counts.get)

        new_members = [member for member in members if member not in existing_memberships]
        centroid = _compute_centroid(members, embeddings)

        groups_payload.append(
            {
                "group_id": assigned_group_id or f"group-{index}",
                "members": members,
                "size": len(members),
                "centroid_embedding": centroid,
                "existing_group_id": assigned_group_id,
                "new_members": new_members,
            }
        )

        if assigned_group_id and new_members:
            store_tasks.append(
                {
                    "mode": "update",
                    "group_id": assigned_group_id,
                    "members": members,
                    "new_members": new_members,
                }
            )
        elif not assigned_group_id and new_members:
            store_tasks.append(
                {
                    "mode": "create",
                    "members": members,
                    "new_members": new_members,
                }
            )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "groups": groups_payload,
        "pair_count": len(filtered_pairs),
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote grouping results to {output_path}")
    else:
        print(json.dumps(payload, indent=2))

    print(
        f"Grouping summary: groups={len(groups_payload)} members={sum(len(g['members']) for g in groups_payload)}"
    )

    if write_supabase and store_tasks:
        stored = store_groups_to_supabase(
            tasks=store_tasks,
            pair_scores=pair_scores,
            embeddings=embeddings,
            existing_memberships=existing_memberships,
            existing_group_metadata=existing_group_metadata,
        )
        print(f"Persisted {stored} group(s) to Supabase.")

    return 0


async def _store_groups_supabase_async(
    *,
    tasks: List[Dict[str, object]],
    pair_scores: Dict[frozenset, float],
    embeddings: Dict[str, List[float]],
    existing_memberships: Dict[str, str],
    existing_group_metadata: Dict[str, Dict[str, object]],
) -> int:
    client = _get_supabase_client()
    storage = GroupStorageManager(client)
    stored = 0

    for task in tasks:
        mode = task.get("mode")
        members: List[str] = task.get("members", [])
        new_members: List[str] = task.get("new_members", [])
        if not new_members:
            continue

        if mode == "update":
            group_id = str(task.get("group_id"))
            if not group_id:
                continue

            meta = existing_group_metadata.get(group_id, {})
            existing_count = int(meta.get("member_count", 0) or 0)
            existing_centroid = meta.get("centroid_embedding")
            tags = meta.get("tags") or []
            raw_status = meta.get("status")
            try:
                status = GroupStatus.from_str(raw_status) if raw_status else GroupStatus.NEW
            except ValueError:
                status = GroupStatus.NEW

            new_vectors = [embeddings.get(member) for member in new_members]
            centroid = _merge_centroid(
                existing_centroid=existing_centroid,
                existing_count=existing_count,
                new_vectors=new_vectors,
            )

            try:
                story_group = StoryGroup(
                    id=group_id,
                    member_count=existing_count + len(new_members),
                    status=status,
                    centroid_embedding=centroid,
                    tags=tags,
                )
            except ValueError:
                story_group = StoryGroup(
                    id=group_id,
                    member_count=existing_count + len(new_members),
                    status=status,
                    centroid_embedding=None,
                    tags=tags,
                )

            await storage.store_group(story_group)
            success = True
            for member in new_members:
                similarity = _best_similarity(member, members, pair_scores)
                ok = await storage.add_member_to_group(group_id, member, similarity)
                if ok:
                    existing_memberships[member] = group_id
                success = success and ok

            if success:
                stored += 1
            continue

        # Default branch creates a new group with the provided members (singletons included)
        centroid = _compute_centroid(new_members, embeddings)
        try:
            story_group = StoryGroup(
                member_count=len(new_members),
                status=GroupStatus.NEW,
                centroid_embedding=centroid,
                tags=[],
            )
        except ValueError:
            story_group = StoryGroup(
                member_count=len(new_members),
                status=GroupStatus.NEW,
                centroid_embedding=None,
                tags=[],
            )

        result = await storage.store_group(story_group)
        group_id = story_group.id if story_group.id else (result if isinstance(result, str) else None)
        if not group_id:
            continue

        success = True
        for member in new_members:
            similarity = _best_similarity(member, members, pair_scores)
            ok = await storage.add_member_to_group(group_id, member, similarity)
            if ok:
                existing_memberships[member] = group_id
            success = success and ok

        if success:
            stored += 1

    return stored


def store_groups_to_supabase(
    *,
    tasks: List[Dict[str, object]],
    pair_scores: Dict[frozenset, float],
    embeddings: Dict[str, List[float]],
    existing_memberships: Dict[str, str],
    existing_group_metadata: Dict[str, Dict[str, object]],
) -> int:
    return asyncio.run(
        _store_groups_supabase_async(
            tasks=tasks,
            pair_scores=pair_scores,
            embeddings=embeddings,
            existing_memberships=existing_memberships,
            existing_group_metadata=existing_group_metadata,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign stories to groups using similarity pairs")
    parser.add_argument("--similarities", required=True, help="Path to similarity pairs JSON")
    parser.add_argument("--embeddings", help="Optional path to embeddings JSON to compute centroids")
    parser.add_argument("--embeddings-from-supabase", action="store_true", help="Load embeddings from Supabase when centroids are required")
    parser.add_argument("--limit", type=int, help="Maximum embeddings to load from Supabase when resolving singleton stories")
    parser.add_argument("--include-all-stories", action="store_true", help="Ensure every news_url_id is assigned to a group, even without similarity pairs")
    parser.add_argument("--news-limit", type=int, help="Maximum news URLs to consider when auto-creating singleton groups")
    parser.add_argument("--output", help="Destination path for grouping JSON")
    parser.add_argument("--threshold", type=float, help="Override similarity threshold for grouping")
    parser.add_argument("--write-supabase", action="store_true", help="Persist generated groups and members to Supabase")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    return run_grouping_pipeline(
        similarity_path=str(args.similarities),
        embeddings_path=getattr(args, "embeddings", None),
        output_path=getattr(args, "output", None),
        threshold=getattr(args, "threshold", None),
        embeddings_from_supabase=bool(getattr(args, "embeddings_from_supabase", False)),
        supabase_limit=getattr(args, "limit", None),
        include_all_stories=bool(getattr(args, "include_all_stories", False)),
        news_limit=getattr(args, "news_limit", None),
        write_supabase=bool(getattr(args, "write_supabase", False)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
