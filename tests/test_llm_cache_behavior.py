from types import SimpleNamespace
from datetime import datetime

import pytest

from src.nfl_news_pipeline.models import NewsItem
from src.nfl_news_pipeline.filters.llm import LLMFilter
import hashlib, json


class DummyLLMClient:
    """Minimal OpenAI-like client that counts create() calls and returns fixed JSON."""

    def __init__(self):
        self.call_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self._create,
            )
        )

    def _create(self, model, messages, timeout=None):  # signature compatible with code path
        self.call_count += 1
        # Return an object with choices[0].message.content
        content = '{"is_relevant": true, "confidence": 0.9, "reason": "ok"}'
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


def make_item():
    return NewsItem(
        url="https://example.com/sports/nfl/opener",
        title="NFL season opener draws record viewership",
        publication_date=datetime.utcnow(),
        source_name="example",
        publisher="Example Sports",
        description="The first game of the season set a new TV rating high.",
    )


def test_llm_cache_hit_returns_cached_result():
    client = DummyLLMClient()
    # Enable cache with in-memory TTL cache (no sqlite path)
    llm = LLMFilter(client=client, cache_enabled=True, cache_ttl_s=3600, cache_path=None)

    item = make_item()

    first = llm.filter(item)
    # Verify the cache now contains the entry
    payload = {
        "m": llm.model,
        "t": item.title,
        "d": item.description or "",
        "u": item.url,
    }
    cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    cached = llm._cache.get(cache_key) if getattr(llm, "_cache", None) is not None else None
    assert cached is not None, "Expected first LLM call to populate cache"
    second = llm.filter(item)

    # Client should be called only once due to cache hit on the second call
    assert client.call_count == 1
    assert first.is_relevant is True
    assert pytest.approx(first.confidence_score, rel=1e-6) == 0.9
    assert first.method == "llm"

    assert second.is_relevant is True
    assert pytest.approx(second.confidence_score, rel=1e-6) == 0.9
    assert second.method == "llm-cache"


def test_llm_cache_disabled_calls_client_each_time():
    client = DummyLLMClient()
    # Explicitly disable cache regardless of env
    llm = LLMFilter(client=client, cache_enabled=False)

    item = make_item()

    a = llm.filter(item)
    b = llm.filter(item)

    # Without cache, the client should be called twice and method remains 'llm'
    assert client.call_count == 2
    assert a.method == "llm"
    assert b.method == "llm"
