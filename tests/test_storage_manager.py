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

    def delete(self):
        self._ops["delete"] = True
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

    def range(self, start: int, end: int):
        self._ops["range"] = (start, end)
        return self

    def limit(self, n: int):
        self._ops["limit"] = n
        return self

    def execute(self):
        result = self.db.apply(self.name, self._ops)
        self._ops = {}
        return result


class FakeSupabase:
    def __init__(self):
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            "news_urls": [],
            "source_watermarks": [],
            "filter_decisions": [],
            "pipeline_audit_log": [],
            "players": [],  # Added for player/team disambiguation tests
            "teams": [],
            "news_url_entities": [],
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
        if table == "players":
            return self._apply_players(ops)
        if table == "teams":
            return self._apply_teams(ops)
        if table == "news_url_entities":
            return self._apply_news_url_entities(ops)
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

    def _apply_range(self, rows: List[Dict[str, Any]], ops: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "range" in ops:
            start, end = ops["range"]
            return rows[start : end + 1]
        return rows

    def _apply_news_urls(self, ops: Dict[str, Any]):
        rows = self.tables["news_urls"]
        if "select" in ops:
            subset = self._filter_rows(rows, ops.get("filters"))
            subset = self._apply_range(subset, ops)
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

    def _apply_players(self, ops: Dict[str, Any]):
        rows = self.tables["players"]
        if "select" in ops:
            subset = self._filter_rows(rows, ops.get("filters"))
            subset = self._apply_range(subset, ops)
            return FakeResponse(subset)
        if "insert" in ops:
            inserted = []
            for r in ops["insert"]:
                nr = dict(r)
                rows.append(nr)
                inserted.append(nr)
            return FakeResponse(inserted)
        return FakeResponse([])

    def _apply_teams(self, ops: Dict[str, Any]):
        rows = self.tables["teams"]
        if "select" in ops:
            subset = self._filter_rows(rows, ops.get("filters"))
            subset = self._apply_range(subset, ops)
            return FakeResponse(subset)
        if "insert" in ops:
            inserted = []
            for r in ops["insert"]:
                nr = dict(r)
                rows.append(nr)
                inserted.append(nr)
            return FakeResponse(inserted)
        return FakeResponse([])

    def _apply_news_url_entities(self, ops: Dict[str, Any]):
        rows = self.tables["news_url_entities"]
        if "delete" in ops:
            filters = ops.get("filters") or []
            to_delete = set()
            subset = self._filter_rows(rows, filters)
            for r in subset:
                to_delete.add(id(r))
            self.tables["news_url_entities"] = [r for r in rows if id(r) not in to_delete]
            rows = self.tables["news_url_entities"]
        if "insert" in ops:
            inserted = []
            for r in ops["insert"]:
                nr = dict(r)
                nr.setdefault("id", f"ent_{len(self.tables['news_url_entities']) + 1}")
                self.tables["news_url_entities"].append(nr)
                inserted.append(nr)
            return FakeResponse(inserted)
        if "select" in ops:
            subset = self._filter_rows(rows, ops.get("filters"))
            subset = self._apply_range(subset, ops)
            return FakeResponse(subset)
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


def test_player_disambiguation_with_team_context(monkeypatch):
    """Players with same name but different teams: only the matching team should remain."""
    db = FakeSupabase()
    # Insert two players with identical names but different teams
    db.tables["players"].extend([
        {
            "player_id": "00-1111111",
            "full_name": "Josh Allen",
            "first_name": "Josh",
            "last_name": "Allen",
            "latest_team": "BUF",
        },
        {
            "player_id": "00-2222222",
            "full_name": "Josh Allen",
            "first_name": "Josh",
            "last_name": "Allen",
            "latest_team": "JAX",
        },
    ])

    # Ensure filtering enabled & ambiguity skip behavior
    monkeypatch.setenv("ENABLE_PLAYER_TEAM_FILTER", "1")
    monkeypatch.setenv("KEEP_AMBIGUOUS_PLAYERS", "0")

    sm = StorageManager(db)

    # Build item with both teams present -> ambiguity -> player should be dropped
    it_ambiguous = make_item("https://ambiguous.com/1")
    it_ambiguous.raw_metadata = {"entity_tags": {"players": ["Josh Allen"], "teams": ["BUF", "JAX"]}}

    # Item with only BUF -> should keep Josh Allen (BUF)
    it_buf_only = make_item("https://buf.com/1")
    it_buf_only.raw_metadata = {"entity_tags": {"players": ["Josh Allen"], "teams": ["BUF"]}}

    res = sm.store_news_items([it_ambiguous, it_buf_only])
    # Collect stored entity rows
    # Our FakeSupabase does not persist news_url_entities table; we validate via side effects of filtering.
    # Instead, re-run the filter helper directly for clarity.
    filtered_players_amb = sm._filter_players_by_teams(["Josh Allen"], ["BUF", "JAX"])  # noqa: SLF001
    filtered_players_buf = sm._filter_players_by_teams(["Josh Allen"], ["BUF"])  # noqa: SLF001

    assert filtered_players_amb == []  # Ambiguous -> dropped
    assert filtered_players_buf == ["00-1111111"]  # Unique within BUF resolves to player_id

    # Sanity check: ids mapped for insertion path
    assert len(res.ids_by_url) == 2


def test_store_entities_normalized_links_to_ids(monkeypatch):
    db = FakeSupabase()
    db.tables["players"].extend([
        {
            "player_id": "00-1111111",
            "full_name": "Josh Allen",
            "first_name": "Josh",
            "last_name": "Allen",
            "common_first_name": "Josh",
            "latest_team": "BUF",
        },
        {
            "player_id": "00-2222222",
            "full_name": "Josh Allen",
            "first_name": "Josh",
            "last_name": "Allen",
            "common_first_name": "Josh",
            "latest_team": "JAX",
        },
    ])
    db.tables["teams"].append(
        {
            "team_abbr": "BUF",
            "team_name": "Buffalo Bills",
            "team_nick": "Bills",
            "city": "Buffalo",
        }
    )

    monkeypatch.setenv("ENABLE_PLAYER_TEAM_FILTER", "1")
    monkeypatch.setenv("KEEP_AMBIGUOUS_PLAYERS", "0")

    sm = StorageManager(db)

    item = make_item("https://buf.com/linked")
    item.raw_metadata = {"entity_tags": {"players": ["Josh Allen"], "teams": ["Buffalo Bills"]}}

    res = sm.store_news_items([item])
    url_id = res.ids_by_url[item.url]

    entities = [
        (row["entity_type"], row["entity_value"]) for row in db.tables["news_url_entities"] if row["news_url_id"] == url_id
    ]

    assert ("team", "BUF") in entities
    assert ("player", "00-1111111") in entities


def test_store_entities_preserves_existing_rows_when_no_entities(monkeypatch):
    db = FakeSupabase()
    existing_id = "url_1"
    db.tables["news_urls"].append({
        "id": existing_id,
        "url": "https://a.com/1",
        "title": "Title",
        "publication_date": datetime.now(timezone.utc),
        "source_name": "src",
        "publisher": "Pub",
    })
    db.tables["news_url_entities"].append({
        "id": "ent_1",
        "news_url_id": existing_id,
        "entity_type": "team",
        "entity_value": "BUF",
    })

    sm = StorageManager(db)
    item = make_item("https://a.com/1")
    item.entities = []
    item.raw_metadata = {}

    res = sm.store_news_items([item])
    assert res.updated_count == 1
    # Existing entity row should remain untouched because no new entities were provided
    assert any(row["entity_value"] == "BUF" for row in db.tables["news_url_entities"] if row["news_url_id"] == existing_id)


def test_store_entities_falls_back_to_uppercase_team_when_cache_empty(monkeypatch):
    db = FakeSupabase()
    sm = StorageManager(db)

    item = make_item("https://team.com/1")
    item.raw_metadata = {"entity_tags": {"players": [], "teams": ["BUF"]}}

    res = sm.store_news_items([item])
    url_id = res.ids_by_url[item.url]
    entities = [
        (row["entity_type"], row["entity_value"]) for row in db.tables["news_url_entities"] if row["news_url_id"] == url_id
    ]

    assert ("team", "BUF") in entities


def test_store_players_when_team_column_missing(monkeypatch):
    db = FakeSupabase()
    db.tables["players"].append(
        {
            "player_id": "00-9999999",
            "full_name": "Example Player",
            "first_name": "Example",
            "last_name": "Player",
            "team_abbr": "DAL",
        }
    )

    db.tables["teams"].append({"team_abbr": "DAL", "team_name": "Dallas Cowboys", "team_nick": "Cowboys"})

    sm = StorageManager(db)

    item = make_item("https://player.com/1")
    item.raw_metadata = {"entity_tags": {"players": ["Example Player"], "teams": ["DAL"]}}

    res = sm.store_news_items([item])
    url_id = res.ids_by_url[item.url]
    entities = [
        (row["entity_type"], row["entity_value"]) for row in db.tables["news_url_entities"] if row["news_url_id"] == url_id
    ]

    assert ("player", "00-9999999") in entities
