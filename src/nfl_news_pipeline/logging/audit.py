from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..storage import StorageManager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def categorize_error(exc: BaseException) -> str:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if any(k in name or k in msg for k in ["http", "timeout", "network", "connection"]):
        return "network_error"
    if any(k in name or k in msg for k in ["xml", "parse", "jsondecode", "valueerror"]):
        return "parse_error"
    if any(k in name or k in msg for k in ["supabase", "database", "db", "psycopg", "sql"]):
        return "db_error"
    if any(k in name or k in msg for k in ["openai", "llm", "ai", "model"]):
        return "llm_error"
    return "unknown_error"


@dataclass
class AuditLogger:
    storage: StorageManager
    pipeline_run_id: Optional[str] = None
    default_source: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------- Context helpers --------
    def ensure_run_id(self) -> str:
        if not self.pipeline_run_id:
            self.pipeline_run_id = str(uuid.uuid4())
        return self.pipeline_run_id

    # -------- Event logging --------
    def log_event(
        self,
        event_type: str,
        *,
        source_name: Optional[str] = None,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        run_id = self.ensure_run_id()
        return self.storage.add_audit_event(
            event_type,
            pipeline_run_id=run_id,
            source_name=source_name or self.default_source,
            message=message,
            event_data=data,
        )

    def log_fetch_start(self, source_name: str) -> bool:
        return self.log_event("fetch", source_name=source_name, message="fetch_start", data={"ts": _now_iso()})

    def log_fetch_end(self, source_name: str, *, items: int, duration_ms: int) -> bool:
        return self.log_event(
            "fetch",
            source_name=source_name,
            message="fetch_end",
            data={"items": items, "duration_ms": duration_ms, "ts": _now_iso()},
        )

    def log_filter_summary(self, *, candidates: int, kept: int) -> bool:
        return self.log_event(
            "filter",
            message="filter_summary",
            data={"candidates": candidates, "kept": kept, "ts": _now_iso()},
        )

    def log_store_summary(self, *, inserted: int, updated: int, errors: int) -> bool:
        return self.log_event(
            "store",
            message="store_summary",
            data={"inserted": inserted, "updated": updated, "errors": errors, "ts": _now_iso()},
        )

    def log_error(self, *, context: str, exc: BaseException, extra: Optional[Dict[str, Any]] = None) -> bool:
        cat = categorize_error(exc)
        tb = traceback.format_exc()
        data = {"category": cat, "error": str(exc), "trace": tb}
        if extra:
            data.update(extra)
        return self.log_event("error", message=context, data=data)

    def log_pipeline_summary(
        self,
        *,
        sources: int,
        fetched_items: int,
        filtered_in: int,
        errors: int,
        duration_ms: int,
    ) -> bool:
        return self.log_event(
            "store",
            message="pipeline_summary",
            data={
                "sources": sources,
                "fetched_items": fetched_items,
                "filtered_in": filtered_in,
                "errors": errors,
                "duration_ms": duration_ms,
                "ts": _now_iso(),
            },
        )
