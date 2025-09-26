#!/usr/bin/env python3
"""One-off migration script to convert legacy SourceArticles into the modern pipeline schema.

This script performs a minimal backfill to translate legacy SourceArticles rows
into the modern pipeline schema:

1. Ensure every ``SourceArticles`` row has a corresponding entry in
   ``news_urls`` (if one does not already exist).
2. Persist the legacy article body as a ``context_summaries`` record so that
   the standard pipeline can take over downstream processing (embeddings,
   grouping, etc.).

It is designed for a single pragmatic run, but it is idempotent. Re-running the
script will skip rows that already exist in the destination tables.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nfl_news_pipeline.models import ProcessedNewsItem
from src.nfl_news_pipeline.story_grouping.context_extractor import URLContextExtractor

# ------------------------------
# Deterministic identifiers
# ------------------------------
NEWS_URL_NAMESPACE = uuid.UUID("6b7a5ec0-476c-4b1c-a8c4-090e2d0cbc3c")

# ------------------------------
# Simple heuristics / constants
# ------------------------------
SOURCE_ID_MAP: Dict[int, Dict[str, str]] = {
    1: {"source_name": "Legacy - NFL.com", "publisher": "NFL.com"},
    2: {"source_name": "Legacy - ESPN", "publisher": "ESPN"},
    3: {"source_name": "Legacy - Bleacher Report", "publisher": "Bleacher Report"},
    4: {"source_name": "Legacy - FOX Sports", "publisher": "FOX Sports"},
}

SOURCE_ARTICLE_COLUMNS = ",".join(
    [
        "id",
        "created_at",
        "updated_at",
        "uniqueName",
        "source",
        "headline",
        "href",
        "url",
        "publishedAt",
        "isProcessed",
        "contentType",
        "Author",
        "Content",
        "isArticleCreated",
        "isTranslated",
        "duplication_of",
        "cluster_id",
    ]
)

DEFAULT_FILTER_METHOD = "legacy_backfill"
DEFAULT_FILTER_REASON = "Imported from SourceArticles one-off script"
DEFAULT_SUMMARY_MODEL = "legacy-backfill"
DEFAULT_CONFIDENCE = 0.65

# Canonical topic taxonomy (finite list for knowledge graph integration)
TOPIC_TAXONOMY: List[Dict[str, str]] = [
    {"id": "injury_report", "description": "Player injuries, medical updates, rehab timelines"},
    {"id": "trade_news", "description": "Trades, trade rumors, compensation details"},
    {"id": "free_agent_signing", "description": "Free-agent signings, extensions, contract restructures"},
    {"id": "roster_move", "description": "Waivers, releases, activations, practice squad moves"},
    {"id": "depth_chart", "description": "Starter/backup changes, snap count shifts, positional battles"},
    {"id": "coaching_change", "description": "Coaching hires, firings, coordinator or playcaller updates"},
    {"id": "draft_scouting", "description": "Draft prospects, combine reports, scouting analysis"},
    {"id": "legal_discipline", "description": "Legal matters, suspensions, disciplinary actions"},
    {"id": "retirement", "description": "Retirement announcements or considerations"},
    {"id": "record_milestone", "description": "Records, milestones, awards, historic achievements"},
    {"id": "performance_recap", "description": "Game recaps, standout performances, statistical breakdowns"},
    {"id": "strategic_analysis", "description": "Scheme changes, film study, strategic adjustments"},
    {"id": "fantasy_impact", "description": "Fantasy football impact, lineup decisions, waiver advice"},
    {"id": "league_policy", "description": "League rules, policy changes, competition committee updates"},
]

FALLBACK_TOPIC_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("injury_report", re.compile(r"\b(acl|achilles|hamstring|ankle|concussion|injur(?:y|ed)|questionable|doubtful|out for season|sprain|placed on ir)\b", re.I)),
    ("trade_news", re.compile(r"\btrade(?:d|s)?\b|\bacquire(?:d|s)?\b|\bin exchange for\b", re.I)),
    ("free_agent_signing", re.compile(r"\bsign(?:ed|s|ing)\b|\bextension\b|\bcontract\b|\bdeal\b|\bagree(?:d)? to terms\b", re.I)),
    ("roster_move", re.compile(r"\bwaive(?:d|s)?\b|\brelease(?:d|s)?\b|\bcut\b|\bpractice squad\b|\bactivate(?:d|s)?\b|\bpromote(?:d|s)?\b", re.I)),
    ("depth_chart", re.compile(r"\bstarter\b|\bstarting\b|\bbackup\b|\bdepth chart\b|\bsnap count\b|\bfirst-team\b|\bsecond-team\b", re.I)),
    ("coaching_change", re.compile(r"\bcoach\b|\bcoordinator\b|\bplaycaller\b|\bfire(?:d|s)?\b|\bhire(?:d|s)?\b|\bpromote(?:d|s)?\b", re.I)),
    ("draft_scouting", re.compile(r"\bdraft\b|\bcombine\b|\bpro day\b|\bmock draft\b|\bprospect\b", re.I)),
    ("legal_discipline", re.compile(r"\bsuspens(?:ion|ions|ed)\b|\bban(?:ned)?\b|\bdiscipline\b|\binvestigation\b|\barrest(?:ed)?\b|\blawsuit\b|\bcharges?\b", re.I)),
    ("retirement", re.compile(r"\bretire(?:ment|d|s)\b", re.I)),
    ("record_milestone", re.compile(r"\brecord\b|\bmilestone\b|\bfranchise mark\b|\bcareer high\b|\baward\b", re.I)),
    ("performance_recap", re.compile(r"\bbeat\b|\bdefeat\b|\bvictory\b|\bloss?\b|\bwin\b|\btouchdown\b|\bperformance\b|\bstat line\b", re.I)),
    ("strategic_analysis", re.compile(r"\bscheme\b|\bstrategy\b|\badjustment\b|\bfilm\b|\banalysis\b|\bgame plan\b", re.I)),
    ("fantasy_impact", re.compile(r"\bfantasy\b|\bwaiver wire\b|start(?:/|\s*or\s*)sit|\bdfs\b|\blineup advice\b", re.I)),
    ("league_policy", re.compile(r"\bpolicy\b|\brule change\b|\bcollective bargaining\b|\bcba\b|\bmemo\b|\bleague ban\b", re.I)),
]


class LLMTopicExtractor:
    """LLM-backed topic classifier constrained to a finite taxonomy."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        taxonomy: Optional[Sequence[Dict[str, str]]] = None,
        default_limit: int = 5,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        timeout_s: Optional[float] = None,
        max_chars: int = 4000,
    ) -> None:
        self.taxonomy: List[Dict[str, str]] = list(taxonomy or TOPIC_TAXONOMY)
        self.allowed_ids: set[str] = {item["id"] for item in self.taxonomy}
        self.default_limit = max(1, default_limit)
        self.max_retries = max(1, max_retries)
        self.timeout_s = float(timeout_s) if timeout_s is not None else float(os.environ.get("OPENAI_TIMEOUT", "15"))
        self.max_chars = max_chars

        self.model = model or os.environ.get("TOPICS_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
        self._is_gpt5 = self.model.lower().startswith("gpt-5")
        self._token_param = "max_completion_tokens" if self._is_gpt5 else "max_tokens"
        if self._is_gpt5:
            if temperature not in (None, 1.0):  # pragma: no cover - logging side effect
                logging.debug("Ignoring custom topic temperature for %s; model enforces default", self.model)
            self.temperature: Optional[float] = None
        else:
            self.temperature = float(temperature) if temperature is not None else None
        self.client = self._init_client()
        if self.client is None:
            logging.info("Topic LLM unavailable; falling back to keyword heuristics.")

    def _init_client(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            logging.warning("openai package not available; topic LLM disabled")
            return None
        try:
            return OpenAI(api_key=api_key, timeout=self.timeout_s)
        except TypeError:  # Mock clients may not accept timeout
            return OpenAI(api_key=api_key)

    def classify(
        self,
        *,
        headline: Optional[str],
        content: Optional[str],
        limit: Optional[int] = None,
    ) -> List[str]:
        if not self.client:
            return []

        use_limit = max(1, limit or self.default_limit)
        parts = [part.strip() for part in (headline, content) if part and part.strip()]
        if not parts:
            return []
        excerpt = "\n\n".join(parts)
        if len(excerpt) > self.max_chars:
            excerpt = excerpt[: self.max_chars]

        taxonomy_text = "\n".join(f"{item['id']}: {item['description']}" for item in self.taxonomy)
        system_prompt = (
            "You are an NFL news topic classifier. Choose zero or more topic IDs from the provided taxonomy. "
            "Respond ONLY with JSON in the format {\"topics\": [\"topic_id\", ...]} using IDs exactly as listed."
        )
        user_prompt = (
            f"Topic taxonomy:\n{taxonomy_text}\n\n"
            f"Article headline: {headline or 'N/A'}\n\n"
            f"Article excerpt:\n{excerpt}\n\n"
            f"Select up to {use_limit} topics that best categorize the article. If no topics apply, return an empty list."
        )

        for attempt in range(self.max_retries):
            try:
                request_payload: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    self._token_param: 300,
                }
                if self.temperature is not None:
                    request_payload["temperature"] = self.temperature

                try:
                    response = self.client.chat.completions.create(
                        timeout=self.timeout_s,
                        **request_payload,
                    )
                except TypeError:
                    response = self.client.chat.completions.create(
                        **request_payload,
                    )
            except Exception as exc:  # pragma: no cover - network dependent
                logging.debug("Topic LLM request failed (attempt %s): %s", attempt + 1, exc)
                continue

            content_text = (response.choices[0].message.content or "").strip()
            try:
                data = self._extract_json_dict(content_text)
            except ValueError as parse_err:
                logging.debug("Topic LLM parse error: %s", parse_err)
                continue

            topics = self._normalize_topics(data, use_limit)
            if topics:
                return topics

        return []

    def _extract_json_dict(self, payload: str) -> Dict[str, Any]:
        text = payload.strip()
        if not text:
            raise ValueError("Empty LLM response")
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise ValueError("No JSON object found in LLM response")
            text = match.group(0)
        return json.loads(text)

    def _normalize_topics(self, data: Dict[str, Any], limit: int) -> List[str]:
        raw_topics: Any = data.get("topics", [])
        if isinstance(raw_topics, str):
            raw_topics = [raw_topics]
        if not isinstance(raw_topics, list):
            return []
        topics: List[str] = []
        seen: set[str] = set()
        for item in raw_topics:
            candidate = ""
            if isinstance(item, str):
                candidate = item
            elif isinstance(item, dict):
                candidate = (
                    item.get("id")
                    or item.get("topic")
                    or item.get("name")
                    or ""
                )
            candidate = candidate.strip().lower().replace(" ", "_")
            if not candidate or candidate not in self.allowed_ids or candidate in seen:
                continue
            topics.append(candidate)
            seen.add(candidate)
            if len(topics) >= limit:
                break
        return topics


def _fallback_topics(text: str, limit: int) -> List[str]:
    if not text:
        return []
    topics: List[str] = []
    seen: set[str] = set()
    for topic_id, pattern in FALLBACK_TOPIC_PATTERNS:
        if pattern.search(text) and topic_id not in seen:
            topics.append(topic_id)
            seen.add(topic_id)
            if len(topics) >= limit:
                break
    return topics


def _coerce_iso_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = value.strip()
    if not text:
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SummaryLLM:
    """Generate context summaries from legacy article content using OpenAI gpt models."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        max_retries: int = 2,
        timeout_s: float = 30.0,
        max_chars: int = 7000,
    ) -> None:
        self.model = model or os.getenv("BACKFILL_SUMMARY_MODEL") or os.getenv("URL_CONTEXT_MODEL") or "gpt-5-nano"
        self.max_retries = max(1, max_retries)
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self._is_gpt5 = self.model.lower().startswith("gpt-5")
        self._token_param = "max_completion_tokens" if self._is_gpt5 else "max_tokens"

        api_key = os.getenv("OPENAI_API_KEY")
        self.extractor = URLContextExtractor(
            openai_api_key=api_key,
            preferred_provider="openai",
            enable_caching=False,
        )
        self.client = self.extractor.openai_client
        if not self.client:
            logging.warning("Context summary LLM unavailable; falling back to heuristic summaries")

    def _prepare_content(self, content: str) -> str:
        text = content.strip()
        if not text:
            return ""
        collapsed = re.sub(r"\s+", " ", text)
        if len(collapsed) <= self.max_chars:
            return collapsed
        return collapsed[: self.max_chars] + "..."

    def _build_prompt(self, *, record: Dict[str, Any], content_excerpt: str) -> str:
        publication_date = record.get("publication_date") or "Unknown"
        prompt = f"""
You are an expert NFL news analyst. Read the provided legacy article content and produce a complete context summary that mirrors the output of the live pipeline.

Metadata:
- URL: {record.get('url')}
- Title: {record.get('title')}
- Publisher: {record.get('publisher')}
- Source Name: {record.get('source_name')}
- Publication Date: {publication_date}

Article Content:
"""
        prompt = textwrap.dedent(prompt).strip()
        content_block = content_excerpt if content_excerpt else "(Content unavailable)"
        requirements = """
REQUIREMENTS:
1. Base the summary strictly on the supplied content; do not invent details or assume outcomes not present in the text.
2. Capture the main actors (players, teams, coaches) and the core event (what happened, when, why it matters).
3. Produce embedding-friendly prose (no bullet lists) in 3-6 sentences.
4. Output canonical team and player names when possible.
5. Return structured JSON exactly in this shape:
{
  "summary": "Full summary text",
  "entities": {
    "players": ["Full Player Name", ...],
    "teams": ["Full Team Name", ...],
    "coaches": ["Coach Name", ...]
  },
  "key_topics": ["topic1", "topic2", ...],
  "story_category": "injury|trade|performance|news|analysis",
  "confidence": 0.8
}

Only return valid JSON, no explanation.
"""
        requirements = textwrap.dedent(requirements).strip()
        return f"{prompt}\n\n<<<ARTICLE_CONTENT_START>>>\n{content_block}\n<<<ARTICLE_CONTENT_END>>>\n\n{requirements}"

    def generate(self, *, article: Dict[str, Any], record: Dict[str, Any]) -> Optional[Any]:
        if not self.client:
            return None
        content = (article.get("Content") or "").strip()
        if not content:
            return None

        excerpt = self._prepare_content(content)
        if not excerpt:
            return None

        news_item = ProcessedNewsItem(
            url=record.get("url", f"legacy://article/{article.get('id')}") or f"legacy://article/{article.get('id')}",
            title=record.get("title") or (article.get("headline") or "Legacy Article"),
            publication_date=_coerce_iso_datetime(record.get("publication_date")),
            source_name=record.get("source_name") or "Legacy",
            publisher=record.get("publisher") or "Legacy",
            description=excerpt,
        )

        prompt = self._build_prompt(record=record, content_excerpt=excerpt)
        base_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert NFL analyst producing structured outputs for downstream pipelines."},
                {"role": "user", "content": prompt},
            ],
            self._token_param: 700,
            "timeout": self.timeout_s,
        }
        if not self._is_gpt5:
            base_payload["temperature"] = 0.1

        base_delay = 0.5
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**base_payload)
            except Exception as exc:  # pragma: no cover - network dependent
                if attempt == self.max_retries - 1:
                    logging.warning("Summary LLM request failed for %s: %s", news_item.url, exc)
                    return None
                backoff = base_delay * (2 ** attempt)
                logging.debug("Summary LLM retry %s for %s after error: %s", attempt + 1, news_item.url, exc)
                time_sleep = min(backoff, 5.0)
                try:
                    import time

                    time.sleep(time_sleep)
                except Exception:
                    pass
                continue

            if not response.choices or not response.choices[0].message.content:
                if attempt == self.max_retries - 1:
                    logging.warning("Summary LLM returned empty response for %s", news_item.url)
                    return None
                backoff = base_delay * (2 ** attempt)
                try:
                    import time

                    time.sleep(min(backoff, 5.0))
                except Exception:
                    pass
                continue

            summary = self.extractor._parse_llm_response(
                response.choices[0].message.content,
                self.model,
                news_item,
            )
            if summary:
                summary.generated_at = datetime.now(timezone.utc)
                return summary

        return None


class ContextSummaryGenerator:
    """Build context summary payloads for backfill rows using an LLM with fallbacks."""

    def __init__(
        self,
        *,
        summary_label: str,
        summary_llm_model: Optional[str],
        summary_confidence: float,
        topic_extractor: Optional[LLMTopicExtractor],
        topic_limit: int,
        llm_retries: int = 2,
        llm_timeout: float = 30.0,
    ) -> None:
        self.summary_label = summary_label
        self.summary_confidence = summary_confidence
        self.topic_extractor = topic_extractor
        self.topic_limit = topic_limit
        self.llm: Optional[SummaryLLM] = None
        try:
            self.llm = SummaryLLM(
                model=summary_llm_model,
                max_retries=llm_retries,
                timeout_s=llm_timeout,
            )
            if not self.llm.client:
                self.llm = None
        except Exception as exc:
            logging.warning("Unable to initialize summary LLM: %s", exc)
            self.llm = None

    @staticmethod
    def _fallback_summary_text(article: Dict[str, Any], record: Dict[str, Any]) -> str:
        content = (article.get("Content") or "").strip()
        if content:
            normalized = re.sub(r"\s+", " ", content)
            return normalized[:600] + ("..." if len(normalized) > 600 else "")
        headline = (article.get("headline") or record.get("title") or record.get("url") or "").strip()
        return headline or "Legacy article"

    def build_payload(
        self,
        *,
        article: Dict[str, Any],
        record: Dict[str, Any],
        news_url_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        summary_obj = None
        llm_used = False
        llm_failed = False

        if self.llm:
            summary_obj = self.llm.generate(article=article, record=record)
            if summary_obj:
                llm_used = not summary_obj.fallback_used
            else:
                llm_failed = True

        summary_text = ""
        llm_model = self.summary_label
        confidence = float(self.summary_confidence)
        entities: Optional[Dict[str, Any]] = None
        raw_topics: List[str] = []
        fallback_used = True

        if summary_obj and summary_obj.summary_text:
            summary_text = summary_obj.summary_text.strip()
            llm_model = summary_obj.llm_model or (self.llm.model if self.llm else self.summary_label)
            confidence = float(summary_obj.confidence_score or self.summary_confidence)
            entities = summary_obj.entities if summary_obj.entities else None
            raw_topics = list(summary_obj.key_topics or [])
            fallback_used = bool(summary_obj.fallback_used)

        if not summary_text:
            summary_text = self._fallback_summary_text(article, record)
            fallback_used = True

        canonical_topics = extract_key_topics(
            headline=article.get("headline"),
            content=article.get("Content"),
            topic_llm=self.topic_extractor,
            limit=self.topic_limit,
        )
        key_topics = canonical_topics or raw_topics

        generated_at = None
        if summary_obj and summary_obj.generated_at:
            generated_at = summary_obj.generated_at.astimezone(timezone.utc).isoformat()
        payload = {
            "news_url_id": news_url_id,
            "summary_text": summary_text,
            "entities": entities or None,
            "key_topics": key_topics,
            "llm_model": llm_model,
            "confidence_score": confidence,
            "fallback_used": bool(fallback_used),
            "generated_at": generated_at or record.get("created_at") or record.get("publication_date"),
            "created_at": record.get("created_at") or record.get("publication_date"),
            "updated_at": record.get("updated_at") or record.get("created_at") or record.get("publication_date"),
        }

        meta = {
            "llm_used": llm_used and not fallback_used,
            "llm_failed": llm_failed,
            "fallback_used": fallback_used,
        }
        return payload, meta


def extract_key_topics(
    *,
    headline: Optional[str],
    content: Optional[str],
    topic_llm: Optional[LLMTopicExtractor],
    limit: int,
) -> List[str]:
    text = " ".join(part for part in [headline or "", content or ""] if part).strip()
    if topic_llm:
        try:
            topics = topic_llm.classify(headline=headline, content=content, limit=limit)
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.warning("Topic LLM classification failed; using fallback. Error: %s", exc)
        else:
            if topics:
                return topics
    return _fallback_topics(text, limit)

# ------------------------------
# Helpers
# ------------------------------


def chunked(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


@dataclass
class Credentials:
    api_url: str
    api_key: str


class SupabaseRestClient:
    """Minimal REST wrapper with helpers tailored for this backfill."""

    def __init__(self, creds: Credentials) -> None:
        self._base_url = creds.api_url.rstrip("/") + "/rest/v1"
        self._session = requests.Session()
        self._headers = {
            "apikey": creds.api_key,
            "Authorization": f"Bearer {creds.api_key}",
        }

    # ------------------------------
    # Fetch helpers
    # ------------------------------

    def fetch_existing_news_urls(self) -> Dict[str, str]:
        existing: Dict[str, str] = {}
        offset = 0
        page_size = 1000
        while True:
            params = {"select": "id,url", "limit": page_size, "offset": offset}
            resp = self._session.get(
                f"{self._base_url}/news_urls",
                params=params,
                headers={**self._headers, "Prefer": "count=exact"},
                timeout=60,
            )
            resp.raise_for_status()
            rows = resp.json()
            for row in rows:
                url = (row.get("url") or "").strip()
                if url:
                    existing[url] = row["id"]
            if len(rows) < page_size:
                break
            offset += page_size
        logging.info("Loaded %s existing news_urls records", len(existing))
        return existing

    def fetch_existing_summaries(self) -> Dict[str, Dict[str, Any]]:
        existing: Dict[str, Dict[str, Any]] = {}
        offset = 0
        page_size = 1000
        while True:
            params = {
                "select": "news_url_id,llm_model,fallback_used",
                "limit": page_size,
                "offset": offset,
            }
            resp = self._session.get(
                f"{self._base_url}/context_summaries",
                params=params,
                headers=self._headers,
                timeout=60,
            )
            resp.raise_for_status()
            rows = resp.json()
            for row in rows:
                news_url_id = row.get("news_url_id")
                if news_url_id:
                    existing[str(news_url_id)] = {
                        "llm_model": row.get("llm_model"),
                        "fallback_used": bool(row.get("fallback_used")),
                    }
            if len(rows) < page_size:
                break
            offset += page_size
        logging.info("Loaded %s existing context summaries", len(existing))
        return existing

    def iterate_source_articles(
        self,
        *,
        batch_size: int,
        offset: int = 0,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        fetched = 0
        while True:
            if limit is not None:
                remaining = limit - fetched
                if remaining <= 0:
                    break
                fetch_size = min(batch_size, remaining)
            else:
                fetch_size = batch_size

            params = {
                "select": SOURCE_ARTICLE_COLUMNS,
                "limit": fetch_size,
                "offset": offset,
                "order": "id.asc",
                "contentType": "eq.news_article",
            }
            if start_date:
                params["publishedAt"] = f"gte.{start_date}"
            resp = self._session.get(
                f"{self._base_url}/SourceArticles",
                params=params,
                headers=self._headers,
                timeout=120,
            )
            resp.raise_for_status()
            rows: List[Dict[str, Any]] = resp.json()
            if not rows:
                break
            yield rows
            batch_count = len(rows)
            offset += batch_count
            fetched += batch_count
            if batch_count < fetch_size:
                break

    # ------------------------------
    # Insert helpers
    # ------------------------------

    def insert_news_urls(self, payload: List[Dict[str, Any]]) -> None:
        if not payload:
            return
        resp = self._session.post(
            f"{self._base_url}/news_urls",
            params={"on_conflict": "url"},
            headers={
                **self._headers,
                "Content-Type": "application/json",
                "Prefer": "resolution=ignore-duplicates,return=minimal",
            },
            data=json.dumps(payload),
            timeout=90,
        )
        resp.raise_for_status()

    def insert_context_summaries(self, payload: List[Dict[str, Any]]) -> None:
        if not payload:
            return
        for chunk in chunked(payload, 200):
            resp = self._session.post(
                f"{self._base_url}/context_summaries",
                params={"on_conflict": "news_url_id"},
                headers={
                    **self._headers,
                    "Content-Type": "application/json",
                    "Prefer": "resolution=ignore-duplicates,return=minimal",
                },
                data=json.dumps(list(chunk)),
                timeout=90,
            )
            try:
                resp.raise_for_status()
            except requests.HTTPError:
                logging.error(
                    "context_summaries insert failed (%s): %s",
                    resp.status_code,
                    resp.text[:500],
                )
                raise


# ------------------------------
# Transformation helpers
# ------------------------------

def load_credentials() -> Credentials:
    load_dotenv(".env")
    api_url = os.getenv("SUPABASE_URL")
    api_key = os.getenv("SUPABASE_KEY")
    if not api_url or not api_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
    return Credentials(api_url=api_url, api_key=api_key)


def parse_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return None


def parse_start_date(value: str) -> str:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("start date cannot be blank")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date format: {text}") from exc
    dt = dt.replace(tzinfo=dt.tzinfo or timezone.utc).astimezone(timezone.utc)
    return dt.isoformat()


def build_description(content: Optional[str], headline: Optional[str], max_length: int = 500) -> Optional[str]:
    text = (content or "").strip()
    if text:
        text = " ".join(text.split())
    if not text:
        text = (headline or "").strip()
    if not text:
        return None
    if len(text) > max_length:
        truncated = text[: max_length - 3]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        text = truncated + "..."
    return text


def clean_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned


def deterministic_news_url_id(url: str) -> str:
    return str(uuid.uuid5(NEWS_URL_NAMESPACE, url))


def transform_article(article: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = (article.get("url") or "").strip()
    if not url:
        return None, "missing_url"

    source_id = article.get("source")
    mapping = SOURCE_ID_MAP.get(source_id)
    if not mapping:
        return None, "unknown_source"

    title = (article.get("headline") or article.get("uniqueName") or url).strip()
    if not title:
        return None, "missing_title"

    publication_date = parse_timestamp(article.get("publishedAt")) or parse_timestamp(article.get("created_at"))
    if not publication_date:
        return None, "missing_publication_date"

    created_at = parse_timestamp(article.get("created_at"))
    updated_at = parse_timestamp(article.get("updated_at")) or created_at

    description = build_description(article.get("Content"), article.get("headline"))

    legacy_payload = clean_metadata(
        {
            "id": article.get("id"),
            "unique_name": article.get("uniqueName"),
            "href": article.get("href"),
            "source_numeric_id": source_id,
            "is_processed": article.get("isProcessed"),
            "content_type": article.get("contentType"),
            "author": article.get("Author"),
            "is_article_created": article.get("isArticleCreated"),
            "is_translated": article.get("isTranslated"),
            "duplication_of": article.get("duplication_of"),
            "cluster_id": article.get("cluster_id"),
            "created_at": created_at,
            "updated_at": updated_at,
            "content": article.get("Content"),
        }
    )

    raw_metadata = {"legacy_source_article": legacy_payload} if legacy_payload else None
    news_url_id = deterministic_news_url_id(url)

    record: Dict[str, Any] = {
        "id": news_url_id,
        "url": url,
        "title": title,
        "description": description,
        "publication_date": publication_date,
        "source_name": mapping["source_name"],
        "publisher": mapping["publisher"],
        "relevance_score": 0.0,
        "filter_method": DEFAULT_FILTER_METHOD,
        "filter_reasoning": DEFAULT_FILTER_REASON,
        "entities": None,
        "categories": [],
        "raw_metadata": raw_metadata,
        "created_at": created_at or publication_date,
        "updated_at": updated_at or created_at or publication_date,
    }
    return record, None


# ------------------------------
# CLI plumbing
# ------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill legacy SourceArticles into modern tables")
    parser.add_argument("--batch-size", type=int, default=200, help="Rows fetched per request")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset in SourceArticles")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process")
    parser.add_argument(
        "--start-date",
        type=parse_start_date,
        default=None,
        help="Earliest SourceArticles.publishedAt to include (ISO-8601)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Process without inserting")
    parser.add_argument("--skip-news-urls", action="store_true", help="Skip news_urls inserts")
    parser.add_argument("--skip-summaries", action="store_true", help="Skip context_summaries inserts")
    parser.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL, help="Model label stored with summaries")
    parser.add_argument("--summary-confidence", type=float, default=DEFAULT_CONFIDENCE, help="Confidence score for summaries")
    parser.add_argument("--summary-llm-model", default=None, help="OpenAI model used to generate context summaries (default gpt-5-nano)")
    parser.add_argument("--summary-llm-timeout", type=float, default=30.0, help="Timeout (seconds) for summary LLM requests")
    parser.add_argument("--summary-llm-retries", type=int, default=4, help="Retry attempts for summary LLM calls")
    parser.add_argument("--topic-limit", type=int, default=5, help="Maximum canonical topics to store per article")
    parser.add_argument("--topics-llm-model", default=None, help="Override OpenAI model for topic classification")
    parser.add_argument("--topics-llm-temperature", type=float, default=None, help="Sampling temperature for topic LLM (ignored for gpt-5 models)")
    parser.add_argument("--topics-llm-timeout", type=float, default=20.0, help="Timeout (seconds) for topic LLM requests")
    parser.add_argument("--topics-llm-retries", type=int, default=2, help="Retry attempts for topic LLM calls")
    parser.add_argument("--disable-topics-llm", action="store_true", help="Disable topic LLM (use keyword fallback only)")
    parser.add_argument(
        "--refresh-fallback-summaries",
        action="store_true",
        default=True,
        help="Regenerate summaries previously created via fallback/legacy path",
    )
    parser.add_argument(
        "--no-refresh-fallback-summaries",
        dest="refresh_fallback_summaries",
        action="store_false",
        help="Do not regenerate fallback summaries",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    return parser.parse_args(argv)


# ------------------------------
# Main processing loop
# ------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    try:
        creds = load_credentials()
    except RuntimeError as exc:  # pragma: no cover - runtime guard
        logging.error(str(exc))
        return 1

    client = SupabaseRestClient(creds)

    topic_extractor: Optional[LLMTopicExtractor] = None
    if not args.disable_topics_llm:
        topic_extractor = LLMTopicExtractor(
            model=args.topics_llm_model,
            default_limit=args.topic_limit,
            max_retries=args.topics_llm_retries,
            temperature=args.topics_llm_temperature,
            timeout_s=args.topics_llm_timeout,
        )

    summary_generator: Optional[ContextSummaryGenerator] = None
    if not args.skip_summaries:
        summary_generator = ContextSummaryGenerator(
            summary_label=args.summary_model,
            summary_llm_model=args.summary_llm_model,
            summary_confidence=args.summary_confidence,
            topic_extractor=topic_extractor,
            topic_limit=args.topic_limit,
            llm_retries=args.summary_llm_retries,
            llm_timeout=args.summary_llm_timeout,
        )

    try:
        news_url_map = client.fetch_existing_news_urls()
        existing_summary_meta = client.fetch_existing_summaries() if not args.skip_summaries else {}
        existing_summary_ids = set(existing_summary_meta.keys())
    except requests.HTTPError as exc:
        logging.error("Failed to load existing records: %s", exc)
        return 1

    expected_model = (args.summary_model or "").strip().lower()

    if not args.skip_summaries and args.refresh_fallback_summaries:
        refreshed = 0
        for summary_id, meta in existing_summary_meta.items():
            llm_model = (meta.get("llm_model") or "").strip().lower()
            fallback_used = bool(meta.get("fallback_used"))
            if fallback_used or (expected_model and llm_model == expected_model):
                existing_summary_ids.discard(summary_id)
                refreshed += 1
        if refreshed:
            logging.info("Marked %s existing fallback summaries for regeneration", refreshed)

    stats = {
        "processed": 0,
        "news_urls_prepared": 0,
        "news_urls_inserted": 0,
        "duplicates": 0,
        "skipped_missing_url": 0,
        "skipped_unknown_source": 0,
        "skipped_missing_title": 0,
        "skipped_missing_publication_date": 0,
        "summaries_prepared": 0,
        "summaries_inserted": 0,
        "summaries_existing": 0,
        "summaries_missing_text": 0,
        "summaries_llm_success": 0,
        "summaries_llm_failed": 0,
        "summaries_llm_fallback": 0,
    }

    pending_news: List[Dict[str, Any]] = []
    pending_summaries: List[Dict[str, Any]] = []

    sample_news: List[Dict[str, Any]] = []
    sample_summary: List[Dict[str, Any]] = []

    try:
        for rows in client.iterate_source_articles(
            batch_size=args.batch_size,
            offset=args.offset,
            limit=args.limit,
            start_date=args.start_date,
        ):
            logging.info("Fetched %s SourceArticles rows", len(rows))
            stats["processed"] += len(rows)

            prepared: List[Dict[str, Any]] = []

            for article in rows:
                record, error = transform_article(article)
                if error:
                    stats_key = {
                        "missing_url": "skipped_missing_url",
                        "unknown_source": "skipped_unknown_source",
                        "missing_title": "skipped_missing_title",
                        "missing_publication_date": "skipped_missing_publication_date",
                    }.get(error)
                    if stats_key:
                        stats[stats_key] += 1
                    logging.debug("Skipping article %s due to %s", article.get("id"), error)
                    continue

                url = record["url"]
                news_url_id = record["id"]
                existing_id = news_url_map.get(url)
                needs_news_url = False
                if existing_id:
                    news_url_id = existing_id
                    stats["duplicates"] += 1
                else:
                    needs_news_url = not args.skip_news_urls

                content = (article.get("Content") or "").strip()
                summary_text = content or (article.get("headline") or "").strip()
                needs_summary = (
                    not args.skip_summaries
                    and summary_text
                    and news_url_id not in existing_summary_ids
                )
                prepared.append(
                    {
                        "article": article,
                        "record": record,
                        "needs_news_url": needs_news_url,
                        "news_url_id": news_url_id,
                        "summary_text": summary_text,
                        "needs_summary": needs_summary,
                    }
                )

            for item in prepared:
                article = item["article"]
                record = item["record"]
                news_url_id = item["news_url_id"]

                if item["needs_news_url"]:
                    pending_news.append(record)
                    news_url_map[record["url"]] = news_url_id
                    stats["news_urls_prepared"] += 1
                    if len(sample_news) < 3:
                        sample_news.append(record.copy())

                if item["needs_summary"]:
                    if summary_generator:
                        summary_payload, meta = summary_generator.build_payload(
                            article=article,
                            record=record,
                            news_url_id=news_url_id,
                        )
                        if meta.get("llm_used"):
                            stats["summaries_llm_success"] += 1
                        if meta.get("llm_failed"):
                            stats["summaries_llm_failed"] += 1
                        if meta.get("fallback_used"):
                            stats["summaries_llm_fallback"] += 1
                    else:
                        summary_payload = {
                            "news_url_id": news_url_id,
                            "summary_text": item["summary_text"],
                            "entities": None,
                            "key_topics": extract_key_topics(
                                headline=article.get("headline"),
                                content=article.get("Content"),
                                topic_llm=topic_extractor,
                                limit=args.topic_limit,
                            ),
                            "llm_model": args.summary_model,
                            "confidence_score": float(args.summary_confidence),
                            "fallback_used": True,
                            "generated_at": record["created_at"],
                            "created_at": record["created_at"],
                            "updated_at": record["updated_at"],
                        }

                    pending_summaries.append(summary_payload)
                    existing_summary_ids.add(news_url_id)
                    stats["summaries_prepared"] += 1
                    if len(sample_summary) < 3:
                        sample_summary.append(summary_payload.copy())
                elif not args.skip_summaries and news_url_id in existing_summary_ids:
                    stats["summaries_existing"] += 1
                elif not args.skip_summaries and not item["summary_text"]:
                    stats["summaries_missing_text"] += 1

            if not args.dry_run:
                if pending_news and len(pending_news) >= args.batch_size:
                    client.insert_news_urls(pending_news)
                    stats["news_urls_inserted"] += len(pending_news)
                    pending_news.clear()
                if pending_summaries and len(pending_summaries) >= args.batch_size:
                    client.insert_context_summaries(pending_summaries)
                    stats["summaries_inserted"] += len(pending_summaries)
                    pending_summaries.clear()

        if args.dry_run:
            logging.info(
                "Dry-run complete. Prepared payload counts: news_urls=%s, summaries=%s",
                len(pending_news),
                len(pending_summaries),
            )
        else:
            if pending_news:
                client.insert_news_urls(pending_news)
                stats["news_urls_inserted"] += len(pending_news)
                pending_news.clear()
            if pending_summaries:
                client.insert_context_summaries(pending_summaries)
                stats["summaries_inserted"] += len(pending_summaries)
                pending_summaries.clear()

    except requests.HTTPError as exc:
        logging.exception("Supabase request failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return 1

    logging.info("Backfill completed: %s", stats)

    if args.dry_run:
        if sample_news:
            logging.debug("Sample news_urls payload: %s", json.dumps(sample_news, indent=2)[:1200])
        if sample_summary:
            logging.debug("Sample context_summaries payload: %s", json.dumps(sample_summary, indent=2)[:1200])

    return 0


if __name__ == "__main__":
    sys.exit(main())
