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
from ..entities.llm_openai import OpenAIEntityExtractor
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

        # Entities extractor: uses dictionary when available; optionally enrich via LLM
        llm_client = None
        try:
            if os.environ.get("NEWS_PIPELINE_DISABLE_ENTITY_LLM", "").lower() not in {"1", "true", "yes"}:
                # Use OpenAI-based extractor (gpt-5-nano by default)
                llm_client = OpenAIEntityExtractor()
        except Exception:
            llm_client = None

        self.extractor = EntitiesExtractor(entity_dict=entity_dict, llm=llm_client)
        
        # Story grouping orchestrator (initialized lazily)
        self._story_grouping_orchestrator = None
        self._story_grouping_enabled = None

    # -------- Public API --------
    def run(self) -> PipelineSummary:
        t0 = time.time()
        self.cm = ConfigManager(self.config_path)
        self.cm.load_config()
        defaults = self.cm.get_defaults()
        sources = self.cm.get_enabled_sources()
        only = os.environ.get("NEWS_PIPELINE_ONLY_SOURCE")
        if only:
            sources = [s for s in sources if s.name == only]

        self.rss = RSSProcessor(defaults)
        self.sitemap = SitemapProcessor(defaults)

        fetched_total = 0
        kept_total = 0
        errors_total = 0
        inserted_total = 0
        updated_total = 0
        store_errors_total = 0
        # Metrics
        total_llm_validations = 0
        total_llm_cache_hits = 0
        total_llm_cache_misses = 0

        debug = os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}
        if debug:
            print(f"[pipeline] run: sources={len(sources)}")

        for feed in sources:
            if debug:
                print(f"[pipeline] source={feed.name} type={feed.type} -> start")
            try:
                f_count, k_count, ins, upd, serr, m = self._process_source(feed)
                fetched_total += f_count
                kept_total += k_count
                inserted_total += ins
                updated_total += upd
                store_errors_total += serr
                total_llm_validations += m.get("llm_validations", 0)
                total_llm_cache_hits += m.get("llm_cache_hits", 0)
                total_llm_cache_misses += m.get("llm_cache_misses", 0)
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
            # Emit metrics snapshot
            try:
                self.audit.log_event(
                    "metrics",
                    message="llm_cache",
                    data={
                        "validations": total_llm_validations,
                        "cache_hits": total_llm_cache_hits,
                        "cache_misses": total_llm_cache_misses,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception:
                pass
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
    def _process_source(self, feed: FeedConfig) -> Tuple[int, int, int, int, int, Dict[str, int]]:
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
        metrics: Dict[str, int] = {"llm_validations": 0, "llm_cache_hits": 0, "llm_cache_misses": 0}
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
                    # Warm-up: determine cache status by running .filter and checking method in result
                    future_to_idx = {ex.submit(llm.filter, items[i]): i for i in candidates}
                    completed = set()
                    try:
                        for fut in as_completed(future_to_idx, timeout=budget_s):
                            completed.add(fut)
                            idx = future_to_idx[fut]
                            try:
                                r = fut.result()
                                llm_results[idx] = r
                                metrics["llm_validations"] += 1
                                if getattr(r, "method", "") == "llm-cache":
                                    metrics["llm_cache_hits"] += 1
                                else:
                                    metrics["llm_cache_misses"] += 1
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
        ids_by_url: Dict[str, str] = {}
        if kept:
            try:
                res = self._retry(lambda: self.storage.store_news_items(kept))
                inserted += res.inserted_count
                updated += res.updated_count
                ids_by_url = res.ids_by_url
            except Exception as e:
                store_errors += 1
                if self.audit:
                    self.audit.log_error(context=f"store {feed.name}", exc=e)

            if self.audit:
                self.audit.log_store_summary(inserted=inserted, updated=updated, errors=store_errors)

            # Run story grouping if enabled and items were stored successfully
            if (inserted > 0 or updated > 0) and hasattr(self, '_should_run_story_grouping') and self._should_run_story_grouping():
                try:
                    self._run_story_grouping_for_items(kept, ids_by_url)
                except Exception as e:
                    if self.audit:
                        self.audit.log_error(context=f"story_grouping {feed.name}", exc=e)

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
        return len(items), len(kept), inserted, updated, store_errors, metrics

    def _should_run_story_grouping(self) -> bool:
        """Check if story grouping should be enabled for this run."""
        if self._story_grouping_enabled is None:
            try:
                # Check configuration
                defaults = self.cm.get_defaults() if self.cm else None
                if defaults and defaults.enable_story_grouping:
                    self._story_grouping_enabled = True
                else:
                    # Check environment variable override
                    env_enabled = os.environ.get("NEWS_PIPELINE_ENABLE_STORY_GROUPING", "").lower()
                    self._story_grouping_enabled = env_enabled in {"1", "true", "yes"}
            except Exception:
                self._story_grouping_enabled = False
        
        return self._story_grouping_enabled

    def _get_story_grouping_orchestrator(self):
        """Get or create the story grouping orchestrator."""
        if self._story_grouping_orchestrator is None:
            try:
                from ..orchestrator.story_grouping import (
                    StoryGroupingOrchestrator,
                    StoryGroupingSettings,
                )
                from ..group_manager import GroupManager
                from ..embedding import EmbeddingGenerator, EmbeddingErrorHandler
                from ..similarity import SimilarityCalculator
                from ..story_grouping import URLContextExtractor
                
                # Get configuration
                defaults = self.cm.get_defaults() if self.cm else None
                
                # Initialize settings from configuration
                settings = StoryGroupingSettings()
                if defaults:
                    settings.max_parallelism = defaults.story_grouping_max_parallelism
                    settings.max_stories_per_run = defaults.story_grouping_max_stories_per_run
                    settings.reprocess_existing = defaults.story_grouping_reprocess_existing
                
                settings.validate()
                
                # Create components
                context_extractor = URLContextExtractor()
                embedding_generator = EmbeddingGenerator()
                similarity_calculator = SimilarityCalculator()
                error_handler = EmbeddingErrorHandler()
                group_manager = GroupManager(self.storage)
                
                # Create orchestrator
                self._story_grouping_orchestrator = StoryGroupingOrchestrator(
                    context_extractor=context_extractor,
                    embedding_generator=embedding_generator,
                    group_manager=group_manager,
                    similarity_calculator=similarity_calculator,
                    error_handler=error_handler,
                    settings=settings,
                )
                
            except ImportError as e:
                if self.audit:
                    self.audit.log_error(context="story_grouping_init", exc=e)
                self._story_grouping_orchestrator = None
            except Exception as e:
                if self.audit:
                    self.audit.log_error(context="story_grouping_init", exc=e)
                self._story_grouping_orchestrator = None
        
        return self._story_grouping_orchestrator

    def _run_story_grouping_for_items(self, items: List[ProcessedNewsItem], ids_by_url: Dict[str, str]) -> None:
        """Run story grouping for the provided items."""
        orchestrator = self._get_story_grouping_orchestrator()
        if orchestrator is None:
            return
            
        # Filter to items that have IDs
        items_with_ids = [item for item in items if item.url in ids_by_url]
        if not items_with_ids:
            return
            
        debug = os.environ.get("NEWS_PIPELINE_DEBUG", "").lower() in {"1", "true", "yes"}
        if debug:
            print(f"[pipeline] running story grouping for {len(items_with_ids)} items")
            
        try:
            import asyncio
            
            # Run story grouping in async context
            result = asyncio.run(orchestrator.process_batch(items_with_ids, ids_by_url))
            
            if debug:
                print(f"[pipeline] story grouping completed: processed={result.metrics.processed_stories}, "
                      f"new_groups={result.metrics.new_groups_created}, "
                      f"updated_groups={result.metrics.existing_groups_updated}")
                      
            if self.audit:
                self.audit.log_event(
                    "story_grouping",
                    message="batch_completed",
                    data={
                        "total_stories": result.metrics.total_stories,
                        "processed_stories": result.metrics.processed_stories,
                        "skipped_stories": result.metrics.skipped_stories,
                        "new_groups_created": result.metrics.new_groups_created,
                        "existing_groups_updated": result.metrics.existing_groups_updated,
                        "processing_time_ms": result.metrics.total_processing_time_ms,
                    },
                )
                
        except Exception as e:
            if debug:
                print(f"[pipeline] story grouping error: {e}")
            if self.audit:
                self.audit.log_error(context="story_grouping_process", exc=e)

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
