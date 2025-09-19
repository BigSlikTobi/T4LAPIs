import json
from datetime import datetime

from scripts.news_ingestion.embedding_cli import run_embedding_pipeline
from scripts.news_ingestion.similarity_cli import run_similarity_pipeline
from src.nfl_news_pipeline.similarity import SimilarityMetric
from src.nfl_news_pipeline.models import EMBEDDING_DIM
from scripts.news_ingestion.grouping_cli import run_grouping_pipeline
from scripts.news_ingestion import staged_pipeline_cli


def _write_context_payload(tmp_path, stories):
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stories": stories,
    }
    path = tmp_path / "context.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_embedding_similarity_grouping_pipeline(tmp_path):
    context_stories = [
        {
            "story_id": "abc-1",
            "summary_text": "Quarterback signs new contract",
            "llm_model": "mock",
            "confidence_score": 0.9,
        },
        {
            "story_id": "abc-2",
            "summary_text": "Team clinches playoff berth",
            "llm_model": "mock",
            "confidence_score": 0.95,
        },
    ]

    context_path = _write_context_payload(tmp_path, context_stories)
    embeddings_path = tmp_path / "embeddings.json"
    similarities_path = tmp_path / "similarities.json"
    groups_path = tmp_path / "groups.json"

    rc_embed = run_embedding_pipeline(
        input_path=context_path,
        output_path=str(embeddings_path),
        mock=True,
    )
    assert rc_embed == 0

    embeddings_payload = json.loads(embeddings_path.read_text(encoding="utf-8"))
    assert len(embeddings_payload["stories"]) == 2

    rc_sim = run_similarity_pipeline(
        input_path=str(embeddings_path),
        output_path=str(similarities_path),
        threshold=0.0,
        top_k=None,
        metric=SimilarityMetric.COSINE,
    )
    assert rc_sim == 0

    similarities_payload = json.loads(similarities_path.read_text(encoding="utf-8"))
    assert similarities_payload["pairs"]

    rc_group = run_grouping_pipeline(
        similarity_path=str(similarities_path),
        embeddings_path=str(embeddings_path),
        output_path=str(groups_path),
        threshold=None,
    )
    assert rc_group == 0

    groups_payload = json.loads(groups_path.read_text(encoding="utf-8"))
    assert groups_payload["groups"], "Expected at least one group"


def test_embedding_pipeline_supabase_store(monkeypatch, tmp_path):
    context_path = _write_context_payload(
        tmp_path,
        [
            {"story_id": "abc-1", "summary_text": "Sample", "llm_model": "mock", "confidence_score": 0.9}
        ],
    )
    embeddings_path = tmp_path / "embeddings.json"

    captured = {}

    def fake_store(embeddings):
        captured["count"] = len(embeddings)
        return len(embeddings)

    monkeypatch.setattr("scripts.news_ingestion.embedding_cli.store_embeddings_supabase", fake_store)

    rc = run_embedding_pipeline(
        input_path=context_path,
        output_path=str(embeddings_path),
        mock=True,
        write_supabase=True,
    )
    assert rc == 0
    assert captured["count"] == 1


def test_embedding_pipeline_supabase_skips_existing(monkeypatch):
    from scripts.news_ingestion.embedding_cli import ContextSummary

    summaries = [
        ContextSummary(
            news_url_id="existing",
            summary_text="Existing summary",
            llm_model="model",
            confidence_score=0.8,
            generated_at=datetime.utcnow(),
        ),
        ContextSummary(
            news_url_id="fresh",
            summary_text="Fresh summary",
            llm_model="model",
            confidence_score=0.9,
            generated_at=datetime.utcnow(),
        ),
    ]

    monkeypatch.setattr(
        "scripts.news_ingestion.embedding_cli._load_context_from_supabase",
        lambda **_: summaries,
    )
    monkeypatch.setattr(
        "scripts.news_ingestion.embedding_cli._fetch_existing_embedding_ids",
        lambda ids: {"existing"},
    )

    captured = {}

    def fake_store(embeddings):
        captured["count"] = len(embeddings)
        return len(embeddings)

    monkeypatch.setattr(
        "scripts.news_ingestion.embedding_cli.store_embeddings_supabase",
        fake_store,
    )

    rc = run_embedding_pipeline(
        input_path=None,
        output_path=None,
        mock=True,
        from_supabase=True,
        write_supabase=True,
    )

    assert rc == 0
    assert captured["count"] == 1


def test_similarity_pipeline_supabase_log(monkeypatch, tmp_path):
    embeddings_path = tmp_path / "embeddings.json"
    vector = [0.0] * EMBEDDING_DIM
    embeddings_payload = {
        "stories": [
            {"story_id": "a", "embedding": vector},
            {"story_id": "b", "embedding": vector},
        ]
    }
    embeddings_path.write_text(json.dumps(embeddings_payload), encoding="utf-8")

    similarities_path = tmp_path / "similarities.json"
    logged = {}

    def fake_log(payload):
        logged["pairs"] = payload.get("pairs")

    monkeypatch.setattr(
        "scripts.news_ingestion.similarity_cli.write_similarity_to_supabase",
        fake_log,
    )

    rc = run_similarity_pipeline(
        input_path=str(embeddings_path),
        output_path=str(similarities_path),
        threshold=0.0,
        top_k=None,
        metric=SimilarityMetric.COSINE,
        write_supabase=True,
    )
    assert rc == 0
    assert "pairs" in logged


def test_grouping_pipeline_supabase_store(monkeypatch, tmp_path):
    similarities_path = tmp_path / "similarities.json"
    similarities_payload = {
        "pairs": [
            {"story_id_a": "a", "story_id_b": "b", "similarity": 0.9},
        ]
    }
    similarities_path.write_text(json.dumps(similarities_payload), encoding="utf-8")

    stored = {}

    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli._fetch_existing_memberships",
        lambda ids: {},
    )
    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli._fetch_group_metadata",
        lambda ids: {},
    )

    def fake_store(**kwargs):
        stored.update(kwargs)
        return len(kwargs.get("tasks", []))

    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli.store_groups_to_supabase",
        fake_store,
    )

    rc = run_grouping_pipeline(
        similarity_path=str(similarities_path),
        embeddings_path=None,
        output_path=None,
        threshold=None,
        write_supabase=True,
    )
    assert rc == 0
    assert stored.get("tasks")


def test_grouping_pipeline_updates_existing_group(monkeypatch, tmp_path):
    similarities_path = tmp_path / "similarities.json"
    similarities_payload = {
        "pairs": [
            {"story_id_a": "existing", "story_id_b": "fresh", "similarity": 0.92},
        ]
    }
    similarities_path.write_text(json.dumps(similarities_payload), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli._fetch_existing_memberships",
        lambda ids: {"existing": "group-123"},
    )
    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli._fetch_group_metadata",
        lambda ids: {
            "group-123": {
                "member_count": 1,
                "centroid_embedding": [],
                "tags": ["nfl"],
                "status": "new",
            }
        },
    )

    captured = {}

    def fake_store(**kwargs):
        captured.update(kwargs)
        return len(kwargs.get("tasks", []))

    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli.store_groups_to_supabase",
        fake_store,
    )

    rc = run_grouping_pipeline(
        similarity_path=str(similarities_path),
        embeddings_path=None,
        output_path=None,
        threshold=None,
        write_supabase=True,
    )

    assert rc == 0
    tasks = captured.get("tasks")
    assert tasks
    assert tasks[0]["mode"] == "update"
    assert tasks[0]["group_id"] == "group-123"
    assert tasks[0]["new_members"] == ["fresh"]


def test_grouping_pipeline_creates_singletons_from_news(monkeypatch, tmp_path):
    similarities_path = tmp_path / "similarities.json"
    similarities_payload = {"pairs": []}
    similarities_path.write_text(json.dumps(similarities_payload), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli._fetch_all_news_ids",
        lambda limit: ["story-1", "story-2"],
    )
    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli._fetch_existing_memberships",
        lambda ids: {},
    )
    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli._fetch_group_metadata",
        lambda ids: {},
    )

    captured = {}

    def fake_store(**kwargs):
        captured.update(kwargs)
        return len(kwargs.get("tasks", []))

    monkeypatch.setattr(
        "scripts.news_ingestion.grouping_cli.store_groups_to_supabase",
        fake_store,
    )

    rc = run_grouping_pipeline(
        similarity_path=str(similarities_path),
        embeddings_path=None,
        output_path=None,
        threshold=None,
        embeddings_from_supabase=False,
        include_all_stories=True,
        write_supabase=True,
    )

    assert rc == 0
    tasks = captured.get("tasks")
    assert tasks and len(tasks) == 2
    assert all(task["mode"] == "create" for task in tasks)
    assert {task["new_members"][0] for task in tasks} == {"story-1", "story-2"}


def test_staged_pipeline_runs_all_stages(monkeypatch, tmp_path):
    calls = []

    def make_stage(name):
        def _stage(**kwargs):
            calls.append(name)
            return staged_pipeline_cli.StageResult(name, 0)

        return _stage

    monkeypatch.setattr(staged_pipeline_cli, "_run_fetch_stage", make_stage("fetch"))
    monkeypatch.setattr(staged_pipeline_cli, "_run_context_stage", make_stage("context"))
    monkeypatch.setattr(staged_pipeline_cli, "_run_embedding_stage", make_stage("embedding"))
    monkeypatch.setattr(staged_pipeline_cli, "_run_similarity_stage", make_stage("similarity"))
    monkeypatch.setattr(staged_pipeline_cli, "_run_grouping_stage", make_stage("grouping"))

    rc = staged_pipeline_cli.run_sequential_pipeline(
        cfg_path=str(tmp_path / "feeds.yaml"),
        source=None,
        use_watermarks=True,
        ignore_watermark=False,
        context_provider="google",
        context_limit=None,
        embedding_limit=None,
        similarity_threshold=0.8,
        similarity_output=str(tmp_path / "similarities.json"),
        similarity_limit=None,
        similarity_write_supabase=False,
        grouping_threshold=0.8,
        grouping_supabase_limit=None,
        include_all_stories=True,
        grouping_news_limit=None,
        skip_fetch=False,
        skip_context=False,
        skip_embedding=False,
        skip_similarity=False,
        skip_grouping=False,
    )

    assert rc == 0
    assert calls == ["fetch", "context", "embedding", "similarity", "grouping"]


def test_staged_pipeline_stops_on_failure(monkeypatch, tmp_path):
    calls = []

    def failing_stage(**kwargs):
        calls.append("fetch")
        return staged_pipeline_cli.StageResult("fetch", 1)

    def not_called_stage(**kwargs):
        raise AssertionError("Stage should not run after failure")

    monkeypatch.setattr(staged_pipeline_cli, "_run_fetch_stage", failing_stage)
    monkeypatch.setattr(staged_pipeline_cli, "_run_context_stage", not_called_stage)
    monkeypatch.setattr(staged_pipeline_cli, "_run_embedding_stage", not_called_stage)
    monkeypatch.setattr(staged_pipeline_cli, "_run_similarity_stage", not_called_stage)
    monkeypatch.setattr(staged_pipeline_cli, "_run_grouping_stage", not_called_stage)

    rc = staged_pipeline_cli.run_sequential_pipeline(
        cfg_path=str(tmp_path / "feeds.yaml"),
        source=None,
        use_watermarks=False,
        ignore_watermark=False,
        context_provider=None,
        context_limit=None,
        embedding_limit=None,
        similarity_threshold=0.8,
        similarity_output=str(tmp_path / "similarities.json"),
        similarity_limit=None,
        similarity_write_supabase=False,
        grouping_threshold=0.8,
        grouping_supabase_limit=None,
        include_all_stories=False,
        grouping_news_limit=None,
        skip_fetch=False,
        skip_context=False,
        skip_embedding=False,
        skip_similarity=False,
        skip_grouping=False,
    )

    assert rc == 1
    assert calls == ["fetch"]
