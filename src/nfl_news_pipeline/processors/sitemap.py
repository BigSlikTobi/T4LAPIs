from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from ..models import DefaultsConfig, FeedConfig, NewsItem
from ..config import ConfigError

logger = logging.getLogger(__name__)


class SitemapProcessor:
    """Fetch and parse XML sitemaps into NewsItem objects.

    - Supports dynamic URL construction from url_template using UTC date
    - Extracts <loc> and <lastmod> from <url> entries
    - Gracefully handles missing/invalid dates
    - Applies days_back filter and max_articles limits
    - Uses httpx with configured headers, timeouts, retries
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

    # ---------- Public API ----------
    def fetch_sitemap(self, feed: FeedConfig) -> List[NewsItem]:
        return asyncio.run(self.fetch_sitemap_async(feed))

    async def fetch_sitemap_async(self, feed: FeedConfig, *, now: Optional[datetime] = None) -> List[NewsItem]:
        if feed.type != "sitemap":
            raise ConfigError(f"SitemapProcessor only supports 'sitemap' type, got '{feed.type}'")

        url = self.construct_sitemap_url(feed, now=now)
        raw = await self._http_get_with_retry(url)
        if raw is None:
            logger.warning("Sitemap fetch failed for %s; returning empty list", url)
            return []

        # Handle potential BOM and leading whitespace/newlines before XML declaration
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        raw = raw.lstrip()

        # First try: XML sitemap (<urlset> or <sitemapindex>)
        root: Optional[ET.Element] = None
        try:
            root = ET.fromstring(raw)
        except Exception as e:
            logger.debug("Sitemap XML parse error for %s: %s", url, e)
            root = None

        items: List[NewsItem] = []
        if root is not None and self._localname(root.tag) in {"urlset", "sitemapindex"}:
            # Iterate all elements and pick by localname == 'url' (handles default namespaces)
            for url_el in root.iter():
                if self._localname(url_el.tag) != "url":
                    continue
                try:
                    loc_text: Optional[str] = None
                    lastmod_text: Optional[str] = None
                    for child in url_el:
                        ln = self._localname(child.tag)
                        if ln == "loc":
                            loc_text = (child.text or "").strip()
                        elif ln == "lastmod":
                            lastmod_text = (child.text or "").strip()
                    if not loc_text:
                        continue

                    pub_dt = self._parse_lastmod(lastmod_text) if lastmod_text else self._utc_now()
                    items.append(
                        NewsItem(
                            url=loc_text,
                            title=loc_text,
                            description=None,
                            publication_date=pub_dt,
                            source_name=feed.name,
                            publisher=feed.publisher,
                            raw_metadata={},
                        )
                    )
                except Exception as e:
                    logger.debug("Skipping malformed sitemap entry in %s: %s", feed.name, e)
                    continue
        else:
            # Fallback: HTML monthly index (e.g., NFL.com monthly articles listing)
            try:
                items = self._parse_html_monthly_index(raw, feed)
            except Exception as e:
                logger.debug("HTML sitemap fallback failed for %s: %s", url, e)
                items = []

        # Apply days_back filter
        if feed.days_back is not None:
            cutoff = (now or self._utc_now()) - timedelta(days=int(feed.days_back))
            items = [i for i in items if i.publication_date >= cutoff]

        # Sort by publication_date desc
        items.sort(key=lambda x: x.publication_date, reverse=True)

        # Enforce max_articles
        if feed.max_articles is not None and int(feed.max_articles) >= 0:
            items = items[: int(feed.max_articles)]

        return items

    def construct_sitemap_url(self, feed: FeedConfig, *, now: Optional[datetime] = None) -> str:
        if feed.type != "sitemap":
            raise ConfigError(f"construct_sitemap_url requires sitemap feed, got '{feed.type}'")
        if not feed.url_template:
            raise ConfigError(f"Feed '{feed.name}' missing 'url_template'")
        if now is None:
            now = self._utc_now()
        yyyy = f"{now.year:04d}"
        mm = f"{now.month:02d}"
        dd = f"{now.day:02d}"
        return (
            feed.url_template
            .replace("{YYYY}", yyyy)
            .replace("{MM}", mm)
            .replace("{DD}", dd)
        )

    # ---------- Internals ----------
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
                    if attempt == self.max_retries:
                        break
                    await asyncio.sleep(delay)
                    delay *= 2
                except Exception as e:
                    last_err = e
                    break
        logger.warning("Failed to fetch %s after %d attempts: %s", url, self.max_retries, last_err)
        return None

    @staticmethod
    def _localname(tag: str) -> str:
        if tag.startswith("{"):
            return tag.split("}", 1)[1]
        return tag

    def _strip_namespaces(self, root: ET.Element) -> ET.Element:
        # Retained for optional use; not used in main flow now
        for el in root.iter():
            el.tag = self._localname(el.tag)
        return root

    @staticmethod
    def _ns_uri(root: ET.Element) -> Optional[str]:
        if root.tag.startswith("{"):
            return root.tag.split("}", 1)[0].strip("{}")
        return None

    @staticmethod
    def _parse_lastmod(text: str) -> datetime:
        s = text.strip()
        # Normalize Z to +00:00 for fromisoformat
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        # Try ISO8601
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
        # Try RFC 2822 / HTTP-date
        try:
            from email.utils import parsedate_to_datetime

            dt = parsedate_to_datetime(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    # ---------- HTML Fallback Parsing ----------
    def _parse_html_monthly_index(self, raw: bytes, feed: FeedConfig) -> List[NewsItem]:
        """Parse an HTML monthly index page containing a table of articles.

        Expected structure (simplified):
          <table>
            <tr><th>Published On</th><th>Title</th></tr>
            <tr><td>2025-09-01</td><td><a href="/news/...">Article</a></td></tr>
            ...
          </table>
        """
        text = raw.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(text, "html.parser")
        items: List[NewsItem] = []

        # Try to find rows in the first table; if none, scan all rows
        tables = soup.find_all("table")
        rows = []
        if tables:
            rows = tables[0].find_all("tr")
        if not rows:
            rows = soup.find_all("tr")

        base = "https://www.nfl.com"
        for tr in rows:
            # Skip header rows
            if tr.find("th") is not None:
                continue
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            date_text = tds[0].get_text(strip=True)
            link = tds[1].find("a")
            if not link:
                continue
            href = (link.get("href") or "").strip()
            title = link.get_text(strip=True) or href
            if not href:
                continue
            if href.startswith("/"):
                href = base + href

            # Parse date
            pub_dt = None
            if date_text:
                try:
                    # Expect YYYY-MM-DD; fallback to ISO
                    if len(date_text) == 10:
                        pub_dt = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    else:
                        pub_dt = datetime.fromisoformat(date_text)
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        pub_dt = pub_dt.astimezone(timezone.utc)
                except Exception:
                    pub_dt = self._utc_now()
            else:
                pub_dt = self._utc_now()

            items.append(
                NewsItem(
                    url=href,
                    title=title,
                    description=None,
                    publication_date=pub_dt,
                    source_name=feed.name,
                    publisher=feed.publisher,
                    raw_metadata={"source": "html-index"},
                )
            )

        return items
