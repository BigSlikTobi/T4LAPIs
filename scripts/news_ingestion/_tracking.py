"""Shared helpers for monitoring per-source storage batches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from src.nfl_news_pipeline.models import ProcessedNewsItem


@dataclass
class SourceBatch:
    """Captured storage result for a single source store call."""

    source_name: str
    items: List[ProcessedNewsItem]
    ids_by_url: Dict[str, str]
    inserted_count: int
    updated_count: int
    previous_watermark: Optional[datetime]

    @property
    def is_new_source(self) -> bool:
        return self.previous_watermark is None

    def has_new_material(self) -> bool:
        return (self.inserted_count + self.updated_count) > 0

    def iter_items_with_ids(self) -> Iterable[ProcessedNewsItem]:
        for item in self.items:
            if item.url in self.ids_by_url:
                yield item


class TrackingStorageAdapter:
    """Wrap a storage implementation to record inserted batches by source."""

    def __init__(self, storage) -> None:
        self._storage = storage
        self._watermarks: Dict[str, Optional[datetime]] = {}
        self.batches: List[SourceBatch] = []

    def __getattr__(self, name: str):  # pragma: no cover - passthrough helper
        return getattr(self._storage, name)

    def get_watermark(self, source_name: str):
        watermark = self._storage.get_watermark(source_name)
        self._watermarks[source_name] = watermark
        return watermark

    def store_news_items(self, items: List[ProcessedNewsItem]):
        result = self._storage.store_news_items(items)
        source_name = items[0].source_name if items else "unknown"
        previous_watermark = self._watermarks.get(source_name)
        batch = SourceBatch(
            source_name=source_name,
            items=list(items),
            ids_by_url=dict(result.ids_by_url),
            inserted_count=result.inserted_count,
            updated_count=result.updated_count,
            previous_watermark=previous_watermark,
        )
        self.batches.append(batch)
        return result

    @property
    def client(self):  # pragma: no cover - passthrough helper
        return getattr(self._storage, "client", None)

    def get_grouping_client(self):  # pragma: no cover - passthrough helper
        if hasattr(self._storage, "get_grouping_client"):
            return self._storage.get_grouping_client()
        raise AttributeError("Underlying storage does not expose get_grouping_client")


__all__ = ["SourceBatch", "TrackingStorageAdapter"]

