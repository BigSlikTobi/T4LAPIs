import pytest
from datetime import datetime
from src.nfl_news_pipeline.models import (
    EMBEDDING_DIM,
    GroupStatus,
    ContextSummary,
    StoryEmbedding,
    StoryGroup,
    StoryGroupMember,
    GroupCentroid,
)


def test_context_summary_validation_and_roundtrip():
    cs = ContextSummary(
        news_url_id="11111111-1111-1111-1111-111111111111",
        summary_text="QB trade update between teams X and Y",
        llm_model="gpt-4o-nano",
        confidence_score=0.9,
        entities={"teams": ["KC", "BUF"]},
        key_topics=["trade", "quarterback"],
        fallback_used=False,
    )
    cs.validate()
    row = cs.to_db()
    back = ContextSummary.from_db(row)
    assert back.summary_text == cs.summary_text
    assert back.llm_model == cs.llm_model
    assert back.key_topics == cs.key_topics
    assert back.fallback_used == cs.fallback_used


def test_story_embedding_dimension_and_roundtrip():
    vec = [0.0] * EMBEDDING_DIM
    emb = StoryEmbedding(
        news_url_id="22222222-2222-2222-2222-222222222222",
        embedding_vector=vec,
        model_name="text-embedding-3-small",
        model_version="1",
        summary_text="QB trade update between teams X and Y",
        confidence_score=0.85,
    )
    emb.validate()
    row = emb.to_db()
    back = StoryEmbedding.from_db(row)
    assert len(back.embedding_vector) == EMBEDDING_DIM
    assert back.model_name == emb.model_name


def test_story_group_validation_and_roundtrip():
    centroid = [0.0] * EMBEDDING_DIM
    grp = StoryGroup(
        id="33333333-3333-3333-3333-333333333333",
        centroid_embedding=centroid,
        member_count=2,
        status=GroupStatus.NEW,
        tags=["breaking", "trade"],
        created_at=datetime.utcnow(),
    )
    grp.validate()
    row = grp.to_db()
    back = StoryGroup.from_db(row)
    assert back.member_count == grp.member_count
    assert back.status == GroupStatus.NEW


def test_story_group_member_validation_and_roundtrip():
    mem = StoryGroupMember(
        group_id="44444444-4444-4444-4444-444444444444",
        news_url_id="55555555-5555-5555-5555-555555555555",
        similarity_score=0.92,
    )
    mem.validate()
    row = mem.to_db()
    back = StoryGroupMember.from_db(row)
    assert back.group_id == mem.group_id
    assert 0.0 <= back.similarity_score <= 1.0


def test_group_centroid_from_group():
    centroid = [0.0] * EMBEDDING_DIM
    grp = StoryGroup(
        id="66666666-6666-6666-6666-666666666666",
        centroid_embedding=centroid,
        member_count=3,
        status=GroupStatus.UPDATED,
    )
    gc = GroupCentroid.from_group(grp)
    gc.validate()
    assert gc.group_id == grp.id
    assert len(gc.centroid_vector) == EMBEDDING_DIM
