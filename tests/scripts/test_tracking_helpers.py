from datetime import datetime

from src.nfl_news_pipeline.models import ProcessedNewsItem

from scripts.news_ingestion._tracking import TrackingStorageAdapter


class _DummyResult:
    def __init__(self, items):
        self.inserted_count = len(items)
        self.updated_count = 0
        self.ids_by_url = {item.url: f"id_{idx}" for idx, item in enumerate(items)}


class _DummyStorage:
    def __init__(self) -> None:
        self._watermarks = {}

    def get_watermark(self, source_name: str):
        return self._watermarks.get(source_name)

    def store_news_items(self, items):
        return _DummyResult(items)


def _make_item(source: str, url: str) -> ProcessedNewsItem:
    return ProcessedNewsItem(
        url=url,
        title="title",
        publication_date=datetime.utcnow(),
        source_name=source,
        publisher="publisher",
        description="desc",
    )


def test_tracking_storage_records_new_source_batch():
    storage = TrackingStorageAdapter(_DummyStorage())
    storage.get_watermark("espn")  # simulate lookup with no existing watermark

    items = [_make_item("espn", "https://example.com/a")]
    storage.store_news_items(items)

    assert len(storage.batches) == 1
    batch = storage.batches[0]

    assert batch.source_name == "espn"
    assert batch.is_new_source  # watermark was never set
    assert batch.has_new_material()
    assert list(batch.iter_items_with_ids()) == items
    assert batch.ids_by_url[items[0].url].startswith("id_")
