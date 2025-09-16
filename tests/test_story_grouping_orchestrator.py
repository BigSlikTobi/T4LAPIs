from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.nfl_news_pipeline.models import (
    ContextSummary,
    ProcessedNewsItem,
    StoryEmbedding,
    GroupCentroid,
    EMBEDDING_DIM,
)
from src.nfl_news_pipeline.similarity import SimilarityCalculator
from src.nfl_news_pipeline.orchestrator.story_grouping import (
    StoryGroupingOrchestrator,
    StoryGroupingSettings,
)


@pytest.fixture
def sample_processed_item() -> ProcessedNewsItem:
    return ProcessedNewsItem(
        url="https://example.com/story",
        title="Example story",
        publication_date=datetime.now(timezone.utc),
        source_name="example",
        publisher="Example Publisher",
        description="Example description",
        relevance_score=0.9,
    )


def make_summary(url: str) -> ContextSummary:
    return ContextSummary(
        news_url_id=url,
        summary_text="Sample summary",
        llm_model="test-model",
        confidence_score=0.85,
        generated_at=datetime.now(timezone.utc),
    )


def make_embedding(news_url_id: str, vector: Optional[np.ndarray] = None) -> StoryEmbedding:
    if vector is None:
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vector[0] = 1.0
    return StoryEmbedding(
        news_url_id=news_url_id,
        embedding_vector=vector.tolist(),
        model_name="test-embedding",
        model_version="1.0",
        summary_text="summary",
        confidence_score=0.9,
        generated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_process_batch_creates_new_group_when_no_centroids(sample_processed_item):
    storage = SimpleNamespace(
        get_embedding_by_url_id=AsyncMock(return_value=None),
        upsert_context_summary=AsyncMock(return_value=True),
        get_group_centroids=AsyncMock(return_value=[]),
        get_group_embeddings=AsyncMock(return_value=[]),
    )

    group_manager = MagicMock()
    group_manager.similarity_calc = SimilarityCalculator(similarity_threshold=0.75)
    group_manager.storage = storage
    group_manager.add_story_to_group = AsyncMock(return_value=True)
    group_manager.create_new_group = AsyncMock(return_value="group-created")

    context_extractor = MagicMock()
    context_extractor.extract_context = AsyncMock(return_value=make_summary(sample_processed_item.url))

    embedding_generator = MagicMock()
    embedding_generator.generate_embedding = AsyncMock(return_value=make_embedding("news-id-1"))

    orchestrator = StoryGroupingOrchestrator(
        group_manager=group_manager,
        context_extractor=context_extractor,
        embedding_generator=embedding_generator,
    )

    result = await orchestrator.process_batch([sample_processed_item], {sample_processed_item.url: "news-id-1"})

    assert result.metrics.new_groups_created == 1
    assert result.metrics.existing_groups_updated == 0
    assert group_manager.create_new_group.await_count == 1
    assert group_manager.add_story_to_group.await_count == 0
    assert result.outcomes[0].status == "created"
    assert result.outcomes[0].group_id == "group-created"
    stored_summary = storage.upsert_context_summary.await_args_list[0].args[0]
    assert stored_summary.news_url_id == "news-id-1"


@pytest.mark.asyncio
async def test_process_batch_assigns_existing_group_using_candidates(sample_processed_item):
    embedding_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    embedding_vector[0] = 1.0
    story_embedding = make_embedding("news-id-2", embedding_vector)

    centroid_a = GroupCentroid(
        group_id="group-a",
        centroid_vector=embedding_vector.tolist(),
        member_count=2,
        last_updated=datetime.now(timezone.utc),
    )
    centroid_b_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    centroid_b_vector[1] = 1.0
    centroid_b = GroupCentroid(
        group_id="group-b",
        centroid_vector=centroid_b_vector.tolist(),
        member_count=3,
        last_updated=datetime.now(timezone.utc),
    )

    storage = SimpleNamespace(
        get_embedding_by_url_id=AsyncMock(return_value=None),
        upsert_context_summary=AsyncMock(return_value=True),
        get_group_centroids=AsyncMock(return_value=[centroid_a, centroid_b]),
        get_group_embeddings=AsyncMock(side_effect=lambda group_id: [story_embedding] if group_id == "group-a" else []),
    )

    group_manager = MagicMock()
    group_manager.similarity_calc = SimilarityCalculator(similarity_threshold=0.7)
    group_manager.storage = storage
    group_manager.add_story_to_group = AsyncMock(return_value=True)
    group_manager.create_new_group = AsyncMock(return_value="group-new")

    context_extractor = MagicMock()
    context_extractor.extract_context = AsyncMock(return_value=make_summary(sample_processed_item.url))

    embedding_generator = MagicMock()
    embedding_generator.generate_embedding = AsyncMock(return_value=story_embedding)

    settings = StoryGroupingSettings(max_candidates=1)
    orchestrator = StoryGroupingOrchestrator(
        group_manager=group_manager,
        context_extractor=context_extractor,
        embedding_generator=embedding_generator,
        settings=settings,
    )

    result = await orchestrator.process_batch([sample_processed_item], {sample_processed_item.url: "news-id-2"})

    assert group_manager.add_story_to_group.await_count == 1
    called_group_id = group_manager.add_story_to_group.await_args_list[0].args[0]
    called_similarity = group_manager.add_story_to_group.await_args_list[0].args[2]
    assert called_group_id == "group-a"
    assert pytest.approx(called_similarity, rel=1e-6) == 1.0
    assert group_manager.create_new_group.await_count == 0
    assert result.metrics.existing_groups_updated == 1
    assert result.outcomes[0].candidate_groups == ["group-a"]


@pytest.mark.asyncio
async def test_prioritization_respects_limits(sample_processed_item):
    older_item = sample_processed_item
    newer_item = ProcessedNewsItem(
        url="https://example.com/high-priority",
        title="High priority",
        publication_date=datetime.now(timezone.utc) + timedelta(hours=1),
        source_name="example",
        publisher="Example Publisher",
        description="",
        relevance_score=0.95,
    )

    storage = SimpleNamespace(
        get_embedding_by_url_id=AsyncMock(return_value=None),
        upsert_context_summary=AsyncMock(return_value=True),
        get_group_centroids=AsyncMock(return_value=[]),
        get_group_embeddings=AsyncMock(return_value=[]),
    )

    group_manager = MagicMock()
    group_manager.similarity_calc = SimilarityCalculator(similarity_threshold=0.75)
    group_manager.storage = storage
    created_groups: List[str] = []

    async def record_group(embedding: StoryEmbedding) -> str:
        created_groups.append(embedding.news_url_id)
        return f"group-{len(created_groups)}"

    group_manager.add_story_to_group = AsyncMock(return_value=True)
    group_manager.create_new_group = AsyncMock(side_effect=record_group)

    context_extractor = MagicMock()
    context_extractor.extract_context = AsyncMock(side_effect=lambda item: make_summary(item.url))

    embedding_generator = MagicMock()
    embedding_generator.generate_embedding = AsyncMock(side_effect=lambda summary, news_id: make_embedding(news_id))

    settings = StoryGroupingSettings(max_stories_per_run=1)
    orchestrator = StoryGroupingOrchestrator(
        group_manager=group_manager,
        context_extractor=context_extractor,
        embedding_generator=embedding_generator,
        settings=settings,
    )

    result = await orchestrator.process_batch(
        [older_item, newer_item],
        {
            older_item.url: "old-id",
            newer_item.url: "new-id",
        },
    )

    created_outcomes = [o for o in result.outcomes if o.status == "created"]
    assert len(created_outcomes) == 1
    assert created_outcomes[0].news_url_id == "new-id"
    assert created_groups == ["new-id"]
    skipped = [o for o in result.outcomes if o.status == "skipped"]
    assert any(o.reason == "max_stories_limit" for o in skipped)
    assert result.metrics.processed_stories == 1
    assert result.metrics.stories_throttled == 1
