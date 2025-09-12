from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.nfl_news_pipeline.logging import AuditLogger, categorize_error


class FakeStorage:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add_audit_event(self, event_type: str, *, pipeline_run_id=None, source_name=None, message=None, event_data=None):
        self.events.append(
            {
                "type": event_type,
                "run": pipeline_run_id,
                "source": source_name,
                "message": message,
                "data": event_data,
            }
        )
        return True


def test_categorize_error():
    assert categorize_error(TimeoutError("timeout")) == "network_error"
    class ParseErr(Exception):
        pass
    assert categorize_error(ParseErr("parse")) == "parse_error"
    class DBErr(Exception):
        pass
    assert categorize_error(DBErr("supabase db")) == "db_error"
    class LLMErr(Exception):
        pass
    assert categorize_error(LLMErr("openai")) == "llm_error"


def test_audit_logger_events_and_error():
    fs = FakeStorage()
    al = AuditLogger(storage=fs)
    assert al.log_fetch_start("feed1")
    assert al.log_fetch_end("feed1", items=10, duration_ms=123)
    assert al.log_filter_summary(candidates=20, kept=5)
    assert al.log_store_summary(inserted=4, updated=1, errors=0)
    try:
        raise ValueError("xml parse boom")
    except Exception as e:
        assert al.log_error(context="parsing", exc=e)

    assert len(fs.events) >= 5
    # Ensure pipeline_run_id is set and consistent
    runs = {e["run"] for e in fs.events}
    assert len(runs) == 1


def test_pipeline_summary():
    fs = FakeStorage()
    al = AuditLogger(storage=fs)
    assert al.log_pipeline_summary(sources=3, fetched_items=40, filtered_in=15, errors=1, duration_ms=2222)
    assert fs.events[-1]["message"] == "pipeline_summary"
