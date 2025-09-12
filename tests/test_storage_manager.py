from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pytest

from src.nfl_news_pipeline.models import ProcessedNewsItem
from src.nfl_news_pipeline.storage import StorageManager


class FakeResponse:
    def __init__(self, data=None):
        self.data = data or []


class FakeTable:
    def __init__(self, name: str, db: "FakeSupabase"):
        self.name = name
        self.db = db
        self._ops: Dict[str, Any] = {}

    # Query builders
    def select(self, cols: str):
        self._ops["select"] = cols
        return self

    def in_(self, col: str, values: List[str]):
        self._ops.setdefault("filters", []).append(("in", col, values))
        return self

    def eq(self, col: str, val: Any):
        self._ops.setdefault("filters", []).append(("eq", col, val))
        return self

    # Mutations
    def insert(self, rows: List[Dict[str, Any]]):
        self._ops["insert"] = rows
        return self

    def update(self, row: Dict[str, Any]):
        self._ops["update"] = row
        return self

    def upsert(self, row: Dict[str, Any]):
        self._ops["upsert"] = row
        return self

    def limit(self, n: int):
        self._ops["limit"] = n
        return self

    def execute(self):
        return self.db.apply(self.name, self._ops)


class FakeSupabase:
    def __init__(self):
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            "news_urls": [],
            "source_watermarks": [],
            "filter_decisions": [],
            "pipeline_audit_log": [],
        }

    def table(self, name: str):
        return FakeTable(name, self)

    # Very small in-memory behaviors sufficient for tests
    def apply(self, table: str, ops: Dict[str, Any]):
        if table == "news_urls":
            return self._apply_news_urls(ops)
        if table == "source_watermarks":
            return self._apply_watermarks(ops)
        if table == "filter_decisions" or table == "pipeline_audit_log":
            return self._apply_simple_append(table, ops)
        return FakeResponse([])

    def _filter_rows(self, rows: List[Dict[str, Any]], filters: List):
        out = rows
        for f in filters or []:
            kind, col, val = f
            if kind == "in":
                out = [r for r in out if r.get(col) in set(val)]
            elif kind == "eq":
                out = [r for r in out if r.get(col) == val]
        return out

    def _apply_news_urls(self, ops: Dict[str, Any]):
        rows = self.tables["news_urls"]
        if "select" in ops:
            subset = self._filter_rows(rows, ops.get("filters"))
            return FakeResponse(subset)
        if "insert" in ops:
            inserted = []
            for r in ops["insert"]:
                # emulate unique url
                existing = next((x for x in rows if x.get("url") == r.get("url")), None)
                if existing:
                    continue
                r = dict(r)
                r.setdefault("id", f"id_{len(rows)+1}")
                rows.append(r)
                inserted.append(r)
            return FakeResponse(inserted)
        if "update" in ops:
            subset = self._filter_rows(rows, ops.get("filters"))
            for r in subset:
                r.update(dict(ops["update"]))
            return FakeResponse(subset)
        return FakeResponse([])

    def _apply_watermarks(self, ops: Dict[str, Any]):
        rows = self.tables["source_watermarks"]
        if "select" in ops:
            subset = self._filter_rows(rows, ops.get("filters"))
            if ops.get("limit"):
                subset = subset[: ops["limit"]]
            return FakeResponse(subset)
        if "upsert" in ops:
            r = ops["upsert"]
            existing = next((x for x in rows if x.get("source_name") == r.get("source_name")), None)
            if existing:
                existing.update(dict(r))
                return FakeResponse([existing])
            rows.append(dict(r))
            return FakeResponse([r])
        return FakeResponse([])

    def _apply_simple_append(self, table: str, ops: Dict[str, Any]):
        if "insert" in ops:
            r = ops["insert"]
            self.tables[table].append(dict(r))
            return FakeResponse([r])
        return FakeResponse([])


def make_item(url: str, title: str = "Title", publisher: str = "Pub", src: str = "src", score: float = 0.9):
    return ProcessedNewsItem(
        url=url,
        title=title,
        publication_date=datetime.now(timezone.utc) - timedelta(hours=1),
        source_name=src,
        publisher=publisher,
        description=None,
        relevance_score=score,
        filter_method="rule_based",
        filter_reasoning="keywords",
    )


def test_dedup_and_store_insert_and_update():
    db = FakeSupabase()
    sm = StorageManager(db)

    items = [make_item("https://a.com/1"), make_item("https://a.com/2")]
    res1 = sm.store_news_items(items)
    assert res1.inserted_count == 2
    assert res1.updated_count == 0
    assert res1.errors_count == 0
    assert set(res1.ids_by_url.keys()) == {"https://a.com/1", "https://a.com/2"}

    # Update one item
    items2 = [make_item("https://a.com/2", title="New Title")]
    res2 = sm.store_news_items(items2)
    assert res2.inserted_count == 0
    assert res2.updated_count == 1
    assert db.tables["news_urls"][1]["title"] == "New Title"


def test_watermarks_get_and_update_and_filter():
    db = FakeSupabase()
    sm = StorageManager(db)
    now = datetime.now(timezone.utc)

    # No watermark yet
    assert sm.get_watermark("feed1") is None

    assert sm.update_watermark("feed1", last_processed_date=now)
    wm = sm.get_watermark("feed1")
    assert wm is not None

    # filter_new_items respects watermark
    old = make_item("https://b.com/old")
    old.publication_date = now - timedelta(days=2)
    new = make_item("https://b.com/new")
    new.publication_date = now + timedelta(seconds=1)
    out = sm.filter_new_items([old, new], "feed1")
    assert [i.url for i in out] == ["https://b.com/new"]


def test_log_filter_decision_and_audit_event():
    db = FakeSupabase()
    sm = StorageManager(db)

    # Need a news url row to reference
    res = sm.store_news_items([make_item("https://c.com/1")])
    url_id = next(iter(res.ids_by_url.values()))

    ok = sm.log_filter_decision(url_id, method="rule_based", stage="rule", confidence=0.8, reasoning="keywords")
    assert ok
    assert len(db.tables["filter_decisions"]) == 1

    ok2 = sm.add_audit_event("filter", source_name="feedX", message="done")
    assert ok2
    assert len(db.tables["pipeline_audit_log"]) == 1
