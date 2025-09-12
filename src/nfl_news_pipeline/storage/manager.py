from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import ProcessedNewsItem


@dataclass
class StorageResult:
    inserted_count: int
    updated_count: int
    errors_count: int
    ids_by_url: Dict[str, str]


class StorageManager:
    """Storage layer for Supabase with URL dedup and watermarks.

    Notes:
    - This class depends on a Supabase client instance (supabase-py >= 2.x).
    - We avoid direct SQL; instead, use PostgREST ops exposed by the client.
    - For migrations, see the SQL files under db/migrations/.
    """

    def __init__(self, client: Any):
        self.client = client
        self.table_news = os.getenv("NEWS_URLS_TABLE", "news_urls")
        self.table_watermarks = os.getenv("SOURCE_WATERMARKS_TABLE", "source_watermarks")
        self.table_audit = os.getenv("PIPELINE_AUDIT_LOG_TABLE", "pipeline_audit_log")
        self.table_filter_decisions = os.getenv("FILTER_DECISIONS_TABLE", "filter_decisions")

    # ------------- Dedup helpers -------------
    def check_duplicate_urls(self, urls: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Return mapping of existing URL -> row for given URLs."""
        url_list = list({u for u in urls if u})
        if not url_list:
            return {}
        resp = self.client.table(self.table_news).select("id,url,publication_date,updated_at").in_("url", url_list).execute()
        rows = getattr(resp, "data", []) or []
        return {r["url"]: r for r in rows if r.get("url")}

    # ------------- Core store -------------
    def store_news_items(self, items: List[ProcessedNewsItem]) -> StorageResult:
        """Insert/update news items by URL.

        - Inserts new URLs
        - Updates existing URLs with latest metadata
        - Returns counts and id mapping
        """
        ids_by_url: Dict[str, str] = {}
        inserted, updated, errors = 0, 0, 0
        if not items:
            return StorageResult(inserted, updated, errors, ids_by_url)

        # Lookup existing by URL
        urls = [i.url for i in items]
        existing_map = self.check_duplicate_urls(urls)

        # Partition new vs existing
        new_rows: List[Dict[str, Any]] = []
        updates: List[Tuple[str, Dict[str, Any]]] = []  # (url, payload)

        for it in items:
            payload = self._item_to_row(it)
            if it.url in existing_map:
                # Update existing
                updates.append((it.url, payload))
            else:
                new_rows.append(payload)

        # Batch insert new
        if new_rows:
            try:
                resp = self.client.table(self.table_news).insert(new_rows).execute()
                rows = getattr(resp, "data", []) or []
                for r in rows:
                    if r.get("url") and r.get("id"):
                        ids_by_url[r["url"]] = r["id"]
                inserted += len(rows)
            except Exception:
                errors += len(new_rows)

        # Update existing one-by-one to keep it simple and predictable
        for url, payload in updates:
            try:
                resp = self.client.table(self.table_news).update(payload).eq("url", url).execute()
                rows = getattr(resp, "data", []) or []
                if rows:
                    r = rows[0]
                    if r.get("url") and r.get("id"):
                        ids_by_url[r["url"]] = r["id"]
                updated += 1
            except Exception:
                errors += 1

        # For any URLs with no id from the responses, fetch ids to complete mapping
        missing = [u for u in urls if u not in ids_by_url]
        if missing:
            try:
                resp = self.client.table(self.table_news).select("id,url").in_("url", missing).execute()
                rows = getattr(resp, "data", []) or []
                for r in rows:
                    if r.get("url") and r.get("id"):
                        ids_by_url[r["url"]] = r["id"]
            except Exception:
                pass

        return StorageResult(inserted, updated, errors, ids_by_url)

    def _item_to_row(self, it: ProcessedNewsItem) -> Dict[str, Any]:
        return {
            "url": it.url,
            "title": it.title,
            "description": it.description,
            "publication_date": self._to_utc_iso(it.publication_date),
            "source_name": it.source_name,
            "publisher": it.publisher,
            "relevance_score": float(it.relevance_score),
            "filter_method": it.filter_method,
            "filter_reasoning": it.filter_reasoning,
            "entities": it.entities or [],
            "categories": it.categories or [],
            "raw_metadata": it.raw_metadata or {},
        }

    # ------------- Filter decision logging -------------
    def log_filter_decision(
        self,
        news_url_id: str,
        *,
        method: str,
        stage: str,
        confidence: float,
        reasoning: str,
        model_id: Optional[str] = None,
    ) -> bool:
        payload = {
            "news_url_id": news_url_id,
            "method": method,
            "stage": stage,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "model_id": model_id,
        }
        try:
            self.client.table(self.table_filter_decisions).insert(payload).execute()
            return True
        except Exception:
            return False

    def add_audit_event(
        self,
        event_type: str,
        *,
        pipeline_run_id: Optional[str] = None,
        source_name: Optional[str] = None,
        message: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        payload = {
            "pipeline_run_id": pipeline_run_id,
            "source_name": source_name,
            "event_type": event_type,
            "message": message,
            "event_data": event_data or {},
        }
        try:
            self.client.table(self.table_audit).insert(payload).execute()
            return True
        except Exception:
            return False

    # ------------- Watermarks -------------
    def get_watermark(self, source_name: str) -> Optional[datetime]:
        try:
            resp = self.client.table(self.table_watermarks).select("last_processed_date").eq("source_name", source_name).limit(1).execute()
            rows = getattr(resp, "data", []) or []
            if not rows:
                return None
            val = rows[0].get("last_processed_date")
            if not val:
                return None
            # Supabase returns ISO strings
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except Exception:
            return None

    def update_watermark(
        self,
        source_name: str,
        *,
        last_processed_date: datetime,
        last_successful_run: Optional[datetime] = None,
        items_processed: Optional[int] = None,
        errors_count: Optional[int] = None,
    ) -> bool:
        payload = {
            "source_name": source_name,
            "last_processed_date": self._to_utc_iso(last_processed_date),
        }
        if last_successful_run is not None:
            payload["last_successful_run"] = self._to_utc_iso(last_successful_run)
        if items_processed is not None:
            payload["items_processed"] = int(items_processed)
        if errors_count is not None:
            payload["errors_count"] = int(errors_count)

        try:
            # Upsert by primary key source_name
            self.client.table(self.table_watermarks).upsert(payload).execute()
            return True
        except Exception:
            return False

    def filter_new_items(self, items: List[ProcessedNewsItem], source_name: str) -> List[ProcessedNewsItem]:
        """Return only items with publication_date > stored watermark."""
        wm = self.get_watermark(source_name)
        if not wm:
            return items
        return [i for i in items if (i.publication_date or datetime.min.replace(tzinfo=timezone.utc)) > wm]

    # ------------- Utils -------------
    @staticmethod
    def _to_utc_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
