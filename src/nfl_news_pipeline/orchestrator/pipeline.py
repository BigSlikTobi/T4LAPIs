from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from ..config import ConfigManager
from ..models import FeedConfig, NewsItem, ProcessedNewsItem
from ..filters.relevance import AMBIGUOUS_HIGH, AMBIGUOUS_LOW
from ..filters.rule_based import RuleBasedFilter
from ..filters.llm import LLMFilter
from ..logging import AuditLogger
from ..storage import StorageManager
from ..processors.rss import RSSProcessor
from ..processors.sitemap import SitemapProcessor
from ..entities import EntitiesExtractor
try:
    # Prefer centralized builder that pulls from Supabase players/teams
    from src.core.data.entity_linking import build_entity_dictionary
except Exception:  # pragma: no cover
    build_entity_dictionary = None  # type: ignore[assignment]


@dataclass
class PipelineSummary:
    sources: int
    fetched_items: int
    filtered_in: int
    errors: int
    inserted: int
    updated: int
    store_errors: int
    duration_ms: int


class NFLNewsPipeline:
    def __init__(
        self,
        config_path: str,
        *,
        storage: StorageManager,
        audit: Optional[AuditLogger] = None,
    ) -> None:
        self.config_path = config_path
        self.storage = storage
        self.audit = audit

        # Lazily initialized components
        self.cm: Optional[ConfigManager] = None
        self.rss: Optional[RSSProcessor] = None
        self.sitemap: Optional[SitemapProcessor] = None

        # Build a comprehensive entity dictionary if possible (Supabase); fall back to lightweight
        entity_dict = None
        try:
            if os.environ.get("NEWS_PIPELINE_DISABLE_ENTITY_DICT", "").lower() not in {"1", "true", "yes"}:
                if build_entity_dictionary is not None:
                    # Best-effort; handle environments without DB access
                    entity_dict = build_entity_dictionary()
        except Exception:
            entity_dict = None

        # Entities extractor: uses dictionary when available, else alias/topic-only
        self.extractor = EntitiesExtractor(entity_dict=entity_dict)

    # -------- Public API --------
    def run(self) -> PipelineSummary:
        t0 = time.time()
        self.cm = ConfigManager(self.config_path)
        self.cm.load_config()
        defaults = self.cm.get_defaults()
        sources = self.cm.get_enabled_sources()

        self.rss = RSSProcessor(defaults)
        self.sitemap = SitemapProcessor(defaults)

        fetched_total = 0
        kept_total = 0
        errors_total = 0
        inserted_total = 0
        updated_total = 0
        store_errors_total = 0

        debug = os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}
        if debug:
            print(f"[pipeline] run: sources={len(sources)}")

        for feed in sources:
            if debug:
                print(f"[pipeline] source={feed.name} type={feed.type} -> start")
            try:
                f_count, k_count, ins, upd, serr = self._process_source(feed)
                fetched_total += f_count
                kept_total += k_count
                inserted_total += ins
                updated_total += upd
                store_errors_total += serr
                if debug:
                    print(
                        f"[pipeline] source={feed.name} -> fetched={f_count} kept={k_count} inserted={ins} updated={upd} store_errors={serr}"
                    )
            except Exception as e:  # pragma: no cover
                errors_total += 1
                if self.audit:
                    self.audit.log_error(context=f"process_source {feed.name}", exc=e)
                if debug:
                    print(f"[pipeline] source={feed.name} -> error: {e}")

        duration_ms = int((time.time() - t0) * 1000)
        if self.audit:
            self.audit.log_pipeline_summary(
                sources=len(sources),
                fetched_items=fetched_total,
                filtered_in=kept_total,
                errors=errors_total,
                duration_ms=duration_ms,
            )
        if debug:
            print(
                f"[pipeline] done: sources={len(sources)} fetched={fetched_total} kept={kept_total} errors={errors_total} time={duration_ms}ms"
            )

        return PipelineSummary(
            sources=len(sources),
            fetched_items=fetched_total,
            filtered_in=kept_total,
            errors=errors_total,
            inserted=inserted_total,
            updated=updated_total,
            store_errors=store_errors_total,
            duration_ms=duration_ms,
        )

    # -------- Internals --------
    def _process_source(self, feed: FeedConfig) -> Tuple[int, int, int, int, int]:
        now = datetime.now(timezone.utc)
        items: List[NewsItem] = []
        if self.audit:
            self.audit.log_fetch_start(feed.name)

        # Fetch
        t0 = time.time()
        if feed.type == "rss":
            assert self.rss is not None
            items = [it for it in self.rss.fetch_multiple([feed]) if it.source_name == feed.name]
        elif feed.type == "sitemap":
            assert self.sitemap is not None
            items = self.sitemap.fetch_sitemap(feed)
        else:
            items = []
        dt_ms = int((time.time() - t0) * 1000)

        if self.audit:
            self.audit.log_fetch_end(feed.name, items=len(items), duration_ms=dt_ms)
        if os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"[pipeline] source={feed.name} -> fetched {len(items)} items in {dt_ms}ms")

        # Watermark filtering (using storage.get_watermark directly with NewsItem)
        wm = self.storage.get_watermark(feed.name)
        if wm is not None:
            items = [i for i in items if (i.publication_date or now) > wm]
        if os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"[pipeline] source={feed.name} -> after watermark: {len(items)} candidates (wm={wm})")

        # Filter relevance
        kept: List[ProcessedNewsItem] = []
        if items:
            # 1) Rule-based pass for all items
            rule = RuleBasedFilter()
            rb_results = [rule.filter(it) for it in items]
            if os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
                pos = sum(1 for r in rb_results if r.is_relevant)
                print(f"[pipeline] source={feed.name} -> rule positives: {pos}/{len(items)}")

            # 2) Determine which items require LLM validation (ambiguous scores)
            to_validate_idx = [
                i for i, r in enumerate(rb_results)
                if (AMBIGUOUS_LOW <= (r.confidence_score or 0.0) <= AMBIGUOUS_HIGH)
            ]

            llm_results: Dict[int, Any] = {}
            if to_validate_idx:
                # Limits: maximum items and time budget per source
                try:
                    max_items = int(os.environ.get("NEWS_PIPELINE_LLM_MAX_ITEMS", "24"))
                except Exception:
                    max_items = 24
                try:
                    budget_s = float(os.environ.get("NEWS_PIPELINE_LLM_BUDGET_SECONDS", os.environ.get("OPENAI_TIMEOUT", "10")))
                except Exception:
                    budget_s = 10.0
                try:
                    workers = int(os.environ.get("NEWS_PIPELINE_LLM_WORKERS", "6"))
                except Exception:
                    workers = 6

                # Choose items to validate (keep order by recency if publication_date available)
                candidates = to_validate_idx
                if len(candidates) > max_items:
                    # Sort indices by publication_date desc if possible, else keep order
                    try:
                        candidates = sorted(
                            candidates,
                            key=lambda idx: (items[idx].publication_date or now),
                            reverse=True,
                        )[:max_items]
                    except Exception:
                        candidates = candidates[:max_items]

                llm = LLMFilter()
                start = time.time()
                if os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
                    print(
                        f"[pipeline] source={feed.name} -> LLM validate {len(candidates)} items (workers={workers}, budget={budget_s}s)"
                    )
                ex = ThreadPoolExecutor(max_workers=max(1, workers))
                try:
                    future_to_idx = {ex.submit(llm.filter, items[i]): i for i in candidates}
                    completed = set()
                    try:
                        for fut in as_completed(future_to_idx, timeout=budget_s):
                            completed.add(fut)
                            idx = future_to_idx[fut]
                            try:
                                llm_results[idx] = fut.result()
                            except Exception:
                                pass
                            if time.time() - start >= budget_s:
                                break
                    except FuturesTimeout:
                        if os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
                            print("[pipeline] LLM budget timeout; proceeding")
                    # Cancel any unfinished futures
                    for fut in future_to_idx.keys():
                        if fut not in completed and not fut.done():
                            fut.cancel()
                finally:
                    try:
                        ex.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass

            # 3) Decide final inclusion using LLM where available; rule otherwise
            for i, it in enumerate(items):
                res = llm_results.get(i, rb_results[i])
                if res.is_relevant:
                    # Categorize entities/topics
                    ents = self.extractor.extract(it)
                    entities_list: List[str] = []
                    # Combine players and teams into a flat list
                    if ents.players:
                        entities_list.extend(sorted(ents.players))
                    if ents.teams:
                        entities_list.extend(sorted(ents.teams))
                    categories_list: List[str] = sorted(ents.topics) if ents.topics else []

                    # Enrich raw_metadata with structured entity tags for storage layer
                    enriched_meta = dict(it.raw_metadata or {})
                    enriched_meta["entity_tags"] = {
                        "players": sorted(ents.players),
                        "teams": sorted(ents.teams),
                        "topics": sorted(ents.topics),
                    }

                    kept.append(
                        ProcessedNewsItem(
                            url=it.url,
                            title=it.title,
                            publication_date=it.publication_date,
                            source_name=it.source_name,
                            publisher=it.publisher,
                            description=it.description,
                            raw_metadata=enriched_meta,
                            relevance_score=res.confidence_score,
                            filter_method=getattr(res, "method", "rule"),
                            filter_reasoning=getattr(res, "reasoning", ""),
                            entities=entities_list,
                            categories=categories_list,
                        )
                    )

        if self.audit:
            self.audit.log_filter_summary(candidates=len(items), kept=len(kept))

        # Store with simple retry
        inserted = 0
        updated = 0
        store_errors = 0
        if kept:
            try:
                res = self._retry(lambda: self.storage.store_news_items(kept))
                inserted += res.inserted_count
                updated += res.updated_count
            except Exception as e:
                store_errors += 1
                if self.audit:
                    self.audit.log_error(context=f"store {feed.name}", exc=e)

            if self.audit:
                self.audit.log_store_summary(inserted=inserted, updated=updated, errors=store_errors)

            # Update watermark to latest processed publication_date
            try:
                latest = max((i.publication_date for i in kept if i.publication_date), default=None)
                if latest is not None:
                    self._retry(
                        lambda: self.storage.update_watermark(
                            feed.name,
                            last_processed_date=latest,
                            last_successful_run=now,
                            items_processed=len(kept),
                        )
                    )
            except Exception as e:  # pragma: no cover
                if self.audit:
                    self.audit.log_error(context=f"watermark {feed.name}", exc=e)

        return len(items), len(kept), inserted, updated, store_errors

    @staticmethod
    def _retry(fn, attempts: int = 2, backoff_ms: int = 100):
        last = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                last = e
                if i + 1 < attempts:
                    time.sleep(backoff_ms / 1000.0)
        raise last  # type: ignore[misc]
