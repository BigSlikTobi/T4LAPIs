from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import feedparser

from ..models import DefaultsConfig, FeedConfig, NewsItem
from ..config import ConfigError

logger = logging.getLogger(__name__)


class RSSProcessor:
    """Fetch and parse RSS feeds into NewsItem objects.

    - Uses httpx with configured headers and timeouts
    - Parses RSS with feedparser from bytes
    - Gracefully handles missing fields
    - Retries with exponential backoff on transient network errors
    """

    def __init__(
        self,
        defaults: DefaultsConfig,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> None:
        self.defaults = defaults
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.headers = {"User-Agent": self.defaults.user_agent}
        self.timeout = httpx.Timeout(self.defaults.timeout_seconds)

    # ---------- Public API (sync wrappers for convenience) ----------
    def fetch_feed(self, feed: FeedConfig) -> List[NewsItem]:
        return asyncio.run(self.fetch_feed_async(feed))

    def fetch_multiple(self, feeds: List[FeedConfig]) -> List[NewsItem]:
        return asyncio.run(self.fetch_multiple_async(feeds))

    # ---------- Async API ----------
    async def fetch_feed_async(self, feed: FeedConfig) -> List[NewsItem]:
        if feed.type != "rss":
            raise ConfigError(f"RSSProcessor only supports 'rss' type, got '{feed.type}'")
        if not feed.url:
            raise ConfigError(f"RSS feed '{feed.name}' missing 'url'")

        raw = await self._http_get_with_retry(feed.url)
        if raw is None:
            logger.warning("RSS fetch failed for %s; returning empty list", feed.url)
            return []

        parsed = feedparser.parse(raw)
        items: List[NewsItem] = []
        for entry in parsed.entries or []:
            try:
                items.append(self._parse_entry(entry, feed))
            except Exception as e:
                logger.debug("Skipping malformed RSS entry from %s: %s", feed.name, e)
                continue
        return items

    async def fetch_multiple_async(self, feeds: List[FeedConfig]) -> List[NewsItem]:
        sem = asyncio.Semaphore(max(1, self.defaults.max_parallel_fetches))

        async def _task(feed: FeedConfig) -> List[NewsItem]:
            async with sem:
                return await self.fetch_feed_async(feed)

        results = await asyncio.gather(*[_task(f) for f in feeds], return_exceptions=True)
        items: List[NewsItem] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("RSS fetch error: %s", r)
                continue
            items.extend(r)
        return items

    # ---------- Internal helpers ----------
    async def _http_get_with_retry(self, url: str) -> Optional[bytes]:
        delay = self.base_delay
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout, follow_redirects=True) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.get(url)
                    if resp.status_code >= 500:
                        raise httpx.HTTPStatusError("server error", request=resp.request, response=resp)
                    resp.raise_for_status()
                    return resp.content
                except (httpx.TimeoutException, httpx.HTTPError) as e:
                    last_err = e
                    logger.debug("HTTP attempt %d failed for %s: %s", attempt, url, e)
                    if attempt == self.max_retries:
                        break
                    await asyncio.sleep(delay)
                    delay *= 2
                except Exception as e:
                    last_err = e
                    logger.debug("Non-HTTP error fetching %s: %s", url, e)
                    break
        logger.warning("Failed to fetch %s after %d attempts: %s", url, self.max_retries, last_err)
        return None

    def _parse_entry(self, entry: Any, feed: FeedConfig) -> NewsItem:
        # URL
        url = (
            getattr(entry, "link", None)
            or getattr(entry, "id", None)
            or (entry.get("link") if isinstance(entry, dict) else None)
        )
        if not url:
            raise ValueError("RSS entry missing link/id")

        # Title
        title = (
            getattr(entry, "title", None)
            or (entry.get("title") if isinstance(entry, dict) else None)
            or url
        )

        # Description / summary
        description = (
            getattr(entry, "summary", None)
            or getattr(entry, "description", None)
            or (entry.get("summary") if isinstance(entry, dict) else None)
            or (entry.get("description") if isinstance(entry, dict) else None)
        )

        # Publication date
        pub_dt = self._parse_pub_date(entry)

        return NewsItem(
            url=url,
            title=title,
            description=description,
            publication_date=pub_dt,
            source_name=feed.name,
            publisher=feed.publisher,
            raw_metadata=self._safe_raw(entry),
        )

    def _parse_pub_date(self, entry: Any) -> datetime:
        # Try best-effort across common fields
        candidates = []
        for key in ("published_parsed", "updated_parsed"):
            val = getattr(entry, key, None) or (entry.get(key) if isinstance(entry, dict) else None)
            if val is not None:
                candidates.append(val)
        # If feedparser returns struct_time
        for c in candidates:
            try:
                # feedparser returns time.struct_time; convert to aware UTC
                import time

                ts = time.mktime(c)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                continue

        # Try string fallback
        for key in ("published", "updated", "pubDate"):
            s = getattr(entry, key, None) or (entry.get(key) if isinstance(entry, dict) else None)
            if s:
                try:
                    from email.utils import parsedate_to_datetime

                    dt = parsedate_to_datetime(str(s))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc)
                except Exception:
                    continue

        # Final fallback: use "now" UTC to avoid None in model
        return datetime.now(timezone.utc)

    @staticmethod
    def _safe_raw(entry: Any) -> Dict[str, Any]:
        try:
            if isinstance(entry, dict):
                return dict(entry)
            # feedparser entries behave like dict-like objects; attempt shallow copy
            return {k: entry.get(k) for k in entry.keys()}  # type: ignore[attr-defined]
        except Exception:
            return {}
