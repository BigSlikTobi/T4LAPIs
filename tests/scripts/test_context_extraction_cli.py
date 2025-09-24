import asyncio
from datetime import datetime

from src.nfl_news_pipeline.models import ContextSummary, ProcessedNewsItem

from scripts.news_ingestion.context_extraction_cli import (
    extract_context_batch,
    run_context_from_file,
    dump_context_from_supabase,
)
import json


def _write_fetch_payload(tmp_path, stories):
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stories": stories,
    }
    path = tmp_path / "fetch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class _DummyExtractor:
    def __init__(self) -> None:
        self.calls = []

    async def extract_context(self, item: ProcessedNewsItem) -> ContextSummary:
        self.calls.append(item.url)
        return ContextSummary(
            news_url_id="",  # filled in by helper
            summary_text=f"Summary for {item.url}",
            llm_model="test-model",
            confidence_score=0.9,
            generated_at=datetime.utcnow(),
        )


class _DummyGroupStorage:
    def __init__(self) -> None:
        self.summaries: list[ContextSummary] = []

    async def upsert_context_summary(self, summary: ContextSummary) -> bool:
        self.summaries.append(summary)
        return True


def _make_item(url: str) -> ProcessedNewsItem:
    return ProcessedNewsItem(
        url=url,
        title="title",
        publication_date=datetime.utcnow(),
        source_name="espn",
        publisher="ESPN",
    )


def test_extract_context_batch_stores_summaries():
    extractor = _DummyExtractor()
    storage = _DummyGroupStorage()
    items = [_make_item("https://example.com/a"), _make_item("https://example.com/b")]
    url_id_map = {item.url: f"id_{idx}" for idx, item in enumerate(items)}

    processed, stored, errors, summaries = asyncio.run(
        extract_context_batch(extractor, storage, items, url_id_map, dry_run=False)
    )

    assert processed == 2
    assert stored == 2
    assert not errors
    assert not summaries  # summaries only returned in dry-run mode

    assert [summary.news_url_id for summary in storage.summaries] == ["id_0", "id_1"]
    assert extractor.calls == [item.url for item in items]


def test_run_context_from_file_mock(tmp_path):
    stories = [
        {
            "story_id": "abc-1",
            "url": "https://example.com/a",
            "title": "Example A",
            "source_name": "espn",
            "publisher": "ESPN",
            "publication_date": datetime.utcnow().isoformat() + "Z",
        }
    ]
    fetch_path = _write_fetch_payload(tmp_path, stories)
    output_path = tmp_path / "context.json"

    rc = run_context_from_file(
        input_path=fetch_path,
        output_path=str(output_path),
        mock=True,
        write_supabase=False,
    )
    assert rc == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["stories"][0]["story_id"] == "abc-1"
    assert "summary_text" in payload["stories"][0]


def test_run_context_from_file_writes_supabase(monkeypatch, tmp_path):
    stories = [
        {
            "story_id": "abc-1",
            "url": "https://example.com/a",
            "title": "Example A",
            "source_name": "espn",
            "publisher": "ESPN",
            "publication_date": datetime.utcnow().isoformat() + "Z",
        }
    ]
    fetch_path = _write_fetch_payload(tmp_path, stories)

    stored = {}

    async def fake_store(summaries):
        stored["count"] = len(list(summaries))
        return stored["count"]

    monkeypatch.setattr(
        "scripts.news_ingestion.context_extraction_cli.write_context_summaries",
        lambda summaries: fake_store(summaries),
    )

    rc = run_context_from_file(
        input_path=fetch_path,
        output_path=None,
        mock=True,
        write_supabase=True,
    )
    assert rc == 0
    assert stored.get("count") == 1


def test_dump_context_from_supabase(monkeypatch, tmp_path):
    summaries = [
        ContextSummary(
            news_url_id="abc-1",
            summary_text="Summary",
            llm_model="mock",
            confidence_score=0.9,
            generated_at=datetime.utcnow(),
        )
    ]

    monkeypatch.setattr(
        "scripts.news_ingestion.context_extraction_cli.load_context_summaries",
        lambda story_ids=None, limit=None: summaries,
    )

    output_path = tmp_path / "contexts.json"
    rc = dump_context_from_supabase(
        story_ids=None,
        limit=None,
        output_path=str(output_path),
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["stories"][0]["story_id"] == "abc-1"
