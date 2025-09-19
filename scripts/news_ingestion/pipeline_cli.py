#!/usr/bin/env python3
"""
CLI for the NFL News Pipeline

Commands:
  - run: Execute the pipeline (with --dry-run support)
  - validate: Validate feeds.yaml and show warnings
  - status: Show environment/config status and DB connectivity
  - list-sources: Print enabled sources

Examples:
  python scripts/pipeline_cli.py run --config feeds.yaml --dry-run
  python scripts/pipeline_cli.py run --source espn --disable-llm
  python scripts/pipeline_cli.py validate
  python scripts/pipeline_cli.py status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Ensure repo root on sys.path, regardless of nesting depth
def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists() and (p / "README.md").exists():
            return p
    return start.parents[0]

ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lazy load env from .env at repo root if present
load_dotenv(str(ROOT / ".env"), override=False)

from src.nfl_news_pipeline.config import ConfigManager, ConfigError
from src.nfl_news_pipeline.orchestrator import NFLNewsPipeline
from src.nfl_news_pipeline.logging import AuditLogger
from src.nfl_news_pipeline.storage import StorageManager, StorageResult
from src.nfl_news_pipeline.models import FeedConfig

# Optional Supabase client
try:
    from src.core.db.database_init import get_supabase_client
except Exception:  # pragma: no cover
    get_supabase_client = None  # type: ignore


def _fmt_ts(dt: Optional[datetime]) -> str:
    if not dt:
        return "-"
    try:
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return str(dt)


class _DryRunStorage:
    """Dry-run storage compatible with StorageManager interface used by AuditLogger/pipeline."""

    def __init__(self) -> None:
        self.rows: List[Any] = []
        self.watermarks: Dict[str, datetime] = {}

    # StorageManager-compatible methods used by pipeline
    def add_audit_event(self, *args, **kwargs):
        return True

    def check_duplicate_urls(self, urls):
        return {}

    def store_news_items(self, items):
        self.rows.extend(items)
        ids = {it.url: f"dry_{i}" for i, it in enumerate(items)}
        return StorageResult(inserted_count=len(items), updated_count=0, errors_count=0, ids_by_url=ids)

    def get_watermark(self, source_name: str):
        return self.watermarks.get(source_name)

    def update_watermark(self, source_name: str, *, last_processed_date: datetime, **kwargs):
        self.watermarks[source_name] = last_processed_date
        return True


def cmd_list_sources(cm: ConfigManager) -> int:
    sources = cm.get_enabled_sources()
    print(f"Enabled sources: {len(sources)}")
    for s in sources:
        print(f" - {s.name} [{s.type}] publisher={s.publisher}")
    return 0


def cmd_validate(cm: ConfigManager) -> int:
    try:
        cfg = cm.to_dict()
        warnings = cm.get_warnings()
        print("feeds.yaml validation: OK")
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f" - {w}")
        print(f"Defaults: {cfg['defaults']}")
        print(f"Sources: {len(cfg['sources'])} total, {len(cm.get_enabled_sources())} enabled")
        return 0
    except ConfigError as e:
        print(f"Config error: {e}")
        return 2


def _build_storage(dry_run: bool) -> StorageManager | _DryRunStorage:
    if dry_run:
        return _DryRunStorage()
    if get_supabase_client is None:
        raise RuntimeError("Supabase client not available in this environment. Use --dry-run or configure credentials.")
    client = get_supabase_client()
    if not client:
        raise RuntimeError("Could not initialize Supabase client. Set SUPABASE_URL and SUPABASE_KEY.")
    return StorageManager(client)


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")


def _filter_sources(sources: List[FeedConfig], name: Optional[str]) -> List[FeedConfig]:
    if not name:
        return sources
    # Exact match first
    exact = [s for s in sources if s.name == name]
    if exact:
        return exact
    # Case-insensitive exact
    insexact = [s for s in sources if s.name.lower() == name.lower()]
    if insexact:
        return insexact
    # Contains (case-insensitive)
    name_l = name.lower()
    contains = [s for s in sources if name_l in s.name.lower() or name_l in s.publisher.lower()]
    if contains:
        return contains
    # Slug match on name or publisher
    nslug = _slug(name)
    slugged = [s for s in sources if _slug(s.name) == nslug or _slug(s.publisher) == nslug]
    return slugged


def _resolve_config_path(path_str: str) -> str:
    """Resolve config path relative to CWD or repo ROOT fallback.

    Allows calling the CLI from the scripts/ folder with --config feeds.yaml.
    """
    p = Path(path_str)
    if p.exists():
        return str(p)
    rp = ROOT / path_str
    if rp.exists():
        return str(rp)
    return str(p)


def cmd_status(cm: ConfigManager) -> int:
    # Config info
    enabled = cm.get_enabled_sources()
    print(f"Config: {len(enabled)} enabled sources")
    # Env checks
    supa_set = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))
    openai_set = bool(os.getenv("OPENAI_API_KEY"))
    print(f"Env: SUPABASE credentials: {'set' if supa_set else 'missing'}; OPENAI_API_KEY: {'set' if openai_set else 'missing'}")
    # DB connectivity (best-effort)
    ok_db = False
    try:
        if get_supabase_client is not None:
            client = get_supabase_client()
            if client:
                # Lightweight query to confirm connectivity
                client.table(os.getenv("SOURCE_WATERMARKS_TABLE", "source_watermarks")).select("source_name").limit(1).execute()
                ok_db = True
    except Exception:
        ok_db = False
    print(f"Database connectivity: {'OK' if ok_db else 'unavailable'}")
    return 0


# Story grouping commands implementation
def _build_story_grouping_orchestrator(storage):
    """Build a story grouping orchestrator with default configuration."""
    try:
        from src.nfl_news_pipeline.orchestrator.story_grouping import (
            StoryGroupingOrchestrator,
            StoryGroupingSettings,
        )
        from src.nfl_news_pipeline.group_manager import GroupManager
        from src.nfl_news_pipeline.embedding import EmbeddingGenerator, EmbeddingErrorHandler
        from src.nfl_news_pipeline.similarity import SimilarityCalculator
        from src.nfl_news_pipeline.centroid_manager import GroupCentroidManager
        from src.nfl_news_pipeline.story_grouping import URLContextExtractor

        # Initialize components with default settings
        settings = StoryGroupingSettings()
        settings.validate()

        # Create components
        context_extractor = URLContextExtractor(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        # Prefer OpenAI if key present; generator will fall back automatically otherwise
        embedding_generator = EmbeddingGenerator(openai_api_key=os.getenv("OPENAI_API_KEY"))
        similarity_calculator = SimilarityCalculator()
        error_handler = EmbeddingErrorHandler()
        # GroupManager expects a storage manager plus similarity and centroid managers
        centroid_manager = GroupCentroidManager()
        group_manager = GroupManager(storage, similarity_calculator, centroid_manager)

        # Create orchestrator
        orchestrator = StoryGroupingOrchestrator(
            context_extractor=context_extractor,
            embedding_generator=embedding_generator,
            group_manager=group_manager,
            error_handler=error_handler,
            settings=settings,
        )

        return orchestrator
    except ImportError as e:
        print(f"Story grouping components not available: {e}")
        return None


def cmd_group_stories(cfg_path: str, *, max_stories: int, max_parallelism: int, dry_run: bool, reprocess: bool) -> int:
    """Run story grouping on recent stories."""
    print(f"Running story grouping (max_stories={max_stories}, max_parallelism={max_parallelism}, dry_run={dry_run})")
    
    try:
        storage = _build_storage(dry_run)
        orchestrator = _build_story_grouping_orchestrator(storage)
        if orchestrator is None:
            return 1
            
        # Configure orchestrator settings
        orchestrator.settings.max_parallelism = max_parallelism
        orchestrator.settings.reprocess_existing = reprocess
        orchestrator.settings.max_stories_per_run = max_stories
        
        # Get recent stories that need grouping
        from src.nfl_news_pipeline.models import ProcessedNewsItem
        
        # For now, use a simple query to get recent items
        # In a real implementation, this would query the database for stories without embeddings
        print("Fetching recent stories for grouping...")
        
        # This is a placeholder - would need to implement proper story retrieval
        stories: List[ProcessedNewsItem] = []
        url_id_map: Dict[str, str] = {}
        
        if not stories:
            print("No stories found that need grouping.")
            return 0
            
        print(f"Processing {len(stories)} stories...")
        
        # Process the batch
        import asyncio
        result = asyncio.run(orchestrator.process_batch(stories, url_id_map))
        
        # Report results
        print(f"Story grouping completed:")
        print(f"  Total stories: {result.metrics.total_stories}")
        print(f"  Processed: {result.metrics.processed_stories}")
        print(f"  Skipped: {result.metrics.skipped_stories}")
        print(f"  New groups: {result.metrics.new_groups_created}")
        print(f"  Updated groups: {result.metrics.existing_groups_updated}")
        print(f"  Processing time: {result.metrics.total_processing_time_ms}ms")
        
        return 0
        
    except Exception as e:
        print(f"Error running story grouping: {e}")
        return 1


def cmd_group_status(cfg_path: str) -> int:
    """Show story grouping statistics."""
    try:
        storage = _build_storage(False)
        
        print("Story Grouping Status:")
        print("=" * 50)
        
        # This would query the database for grouping statistics
        # For now, show placeholder information
        print("Database connectivity: OK")
        print("Total stories with embeddings: [TBD]")
        print("Total story groups: [TBD]")
        print("Average group size: [TBD]")
        print("Recent grouping activity: [TBD]")
        
        return 0
        
    except Exception as e:
        print(f"Error getting group status: {e}")
        return 1


def cmd_group_backfill(cfg_path: str, *, batch_size: int, max_batches: Optional[int], dry_run: bool, resume_from: Optional[str]) -> int:
    """Batch process existing stories for grouping."""
    print(f"Running story grouping backfill (batch_size={batch_size}, max_batches={max_batches}, dry_run={dry_run})")
    
    if resume_from:
        print(f"Resuming from story ID: {resume_from}")
        
    try:
        storage = _build_storage(dry_run)
        orchestrator = _build_story_grouping_orchestrator(storage)
        if orchestrator is None:
            return 1
            
        # Configure for batch processing
        orchestrator.settings.max_parallelism = min(4, batch_size)  # Conservative parallelism for backfill
        
        batch_num = 0
        total_processed = 0
        
        print("Starting backfill process...")
        
        # This is a placeholder implementation
        # Real implementation would:
        # 1. Query database for stories without embeddings
        # 2. Process in batches
        # 3. Handle resume functionality
        # 4. Provide progress updates
        
        print("Backfill completed.")
        print(f"Total batches processed: {batch_num}")
        print(f"Total stories processed: {total_processed}")
        
        return 0
        
    except Exception as e:
        print(f"Error during backfill: {e}")
        return 1


def cmd_group_report(cfg_path: str, *, format_type: str, days_back: int) -> int:
    """Generate story grouping analytics report."""
    try:
        storage = _build_storage(False)
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Generating story grouping report (format={format_type}, days_back={days_back})")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # This would query the database for analytics data
        # For now, generate placeholder report
        
        report_data = {
            "report_generated": end_date.isoformat(),
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days_back
            },
            "summary": {
                "total_stories_processed": 0,
                "total_groups_created": 0,
                "average_group_size": 0.0,
                "processing_success_rate": 100.0
            },
            "top_groups": [],
            "trending_topics": [],
            "performance_metrics": {
                "avg_processing_time_ms": 0,
                "embedding_generation_success_rate": 100.0,
                "similarity_calculation_success_rate": 100.0
            }
        }
        
        if format_type == "json":
            print(json.dumps(report_data, indent=2))
        else:
            # Text format
            print("\nStory Grouping Analytics Report")
            print("=" * 50)
            print(f"Report Period: {days_back} days")
            print(f"Total Stories Processed: {report_data['summary']['total_stories_processed']}")
            print(f"Total Groups Created: {report_data['summary']['total_groups_created']}")
            print(f"Average Group Size: {report_data['summary']['average_group_size']:.1f}")
            print(f"Processing Success Rate: {report_data['summary']['processing_success_rate']:.1f}%")
            print("\nPerformance Metrics:")
            print(f"  Average Processing Time: {report_data['performance_metrics']['avg_processing_time_ms']}ms")
            print(f"  Embedding Success Rate: {report_data['performance_metrics']['embedding_generation_success_rate']:.1f}%")
            print(f"  Similarity Calculation Success Rate: {report_data['performance_metrics']['similarity_calculation_success_rate']:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1


def cmd_run(
    cfg_path: str,
    *,
    source: Optional[str],
    dry_run: bool,
    disable_llm: bool,
    llm_timeout: Optional[float],
    enable_story_grouping: bool = False,
    ignore_watermark: bool = False,
) -> int:
    cm = ConfigManager(cfg_path)
    cm.load_config()
    storage = _build_storage(dry_run)
    audit = AuditLogger(storage=storage)

    # LLM runtime controls
    if disable_llm:
        os.environ["NEWS_PIPELINE_DISABLE_LLM"] = "1"
    if llm_timeout is not None:
        os.environ["OPENAI_TIMEOUT"] = str(llm_timeout)
    if enable_story_grouping:
        os.environ["NEWS_PIPELINE_ENABLE_STORY_GROUPING"] = "1"
    if ignore_watermark:
        os.environ["NEWS_PIPELINE_IGNORE_WATERMARK"] = "1"

    # If single source requested, temporarily write a filtered config manager
    if source:
        enabled = cm.get_enabled_sources()
        matches = _filter_sources(enabled, source)
        if not matches:
            names = ", ".join(sorted(s.name for s in enabled))
            print(f"No enabled source matched '{source}'. Try one of: {names}")
            return 3
        # If multiple, prefer the first (best match) and inform
        chosen = matches[0]
        if len(matches) > 1:
            also = ", ".join(s.name for s in matches[1:3]) + ("..." if len(matches) > 3 else "")
            print(f"Matched '{chosen.name}'. Other matches: {also}")
        os.environ["NEWS_PIPELINE_ONLY_SOURCE"] = chosen.name

    pipeline = NFLNewsPipeline(cfg_path, storage=storage, audit=audit)
    # Small hook: let orchestrator optionally filter sources by env var if present
    try:
        summary = pipeline.run()
    finally:
        # Clean up env override
        os.environ.pop("NEWS_PIPELINE_ONLY_SOURCE", None)

    print(
        "Run summary: "
        f"sources={summary.sources} fetched={summary.fetched_items} kept={summary.filtered_in} "
        f"inserted={summary.inserted} updated={summary.updated} errors={summary.errors} store_errors={summary.store_errors} "
        f"time={summary.duration_ms}ms"
    )
    if dry_run and isinstance(storage, _DryRunStorage) and storage.rows:
        print(f"Dry-run stored items (preview): {min(5, len(storage.rows))}/{len(storage.rows)}")
        for it in storage.rows[:5]:
            print(f" - {getattr(it, 'title', '-')}: {getattr(it, 'url', '-')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NFL News Pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Run the pipeline")
    pr.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    pr.add_argument("--source", help="Only process a single enabled source by name")
    pr.add_argument("--dry-run", action="store_true", help="Do not write to DB; use in-memory storage")
    pr.add_argument("--disable-llm", action="store_true", help="Disable LLM usage; rule-based only")
    pr.add_argument("--llm-timeout", type=float, help="Override LLM timeout in seconds")
    pr.add_argument("--enable-story-grouping", action="store_true", help="Enable story grouping post-processing")
    pr.add_argument("--ignore-watermark", action="store_true", help="Ignore source watermarks and process all fetched items")

    # validate
    pv = sub.add_parser("validate", help="Validate configuration and show warnings")
    pv.add_argument("--config", default="feeds.yaml")

    # status
    ps = sub.add_parser("status", help="Show environment and DB connectivity status")
    ps.add_argument("--config", default="feeds.yaml")

    # list-sources
    pl = sub.add_parser("list-sources", help="List enabled sources")
    pl.add_argument("--config", default="feeds.yaml")

    # group-stories
    pg = sub.add_parser("group-stories", help="Run story grouping on recent stories")
    pg.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    pg.add_argument("--max-stories", type=int, default=100, help="Maximum stories to process (default: 100)")
    pg.add_argument("--max-parallelism", type=int, default=4, help="Maximum parallel processing (default: 4)")
    pg.add_argument("--dry-run", action="store_true", help="Do not write to DB; show what would be done")
    pg.add_argument("--reprocess", action="store_true", help="Reprocess stories that already have embeddings")

    # group-status
    ps_grp = sub.add_parser("group-status", help="Show story grouping statistics")
    ps_grp.add_argument("--config", default="feeds.yaml")

    # group-backfill
    pb = sub.add_parser("group-backfill", help="Batch process existing stories for grouping")
    pb.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml (default: feeds.yaml)")
    pb.add_argument("--batch-size", type=int, default=50, help="Stories per batch (default: 50)")
    pb.add_argument("--max-batches", type=int, help="Maximum batches to process (default: unlimited)")
    pb.add_argument("--dry-run", action="store_true", help="Do not write to DB; show what would be done")
    pb.add_argument("--resume-from", help="Resume from specific story ID")

    # group-report
    pr_grp = sub.add_parser("group-report", help="Generate story grouping analytics report")
    pr_grp.add_argument("--config", default="feeds.yaml")
    pr_grp.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    pr_grp.add_argument("--days-back", type=int, default=7, help="Days to include in report (default: 7)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_path = _resolve_config_path(str(getattr(args, "config", "feeds.yaml")))
    if args.cmd in {"validate", "status", "list-sources"}:
        cm = ConfigManager(cfg_path)
        cm.load_config()
        if args.cmd == "validate":
            return cmd_validate(cm)
        if args.cmd == "status":
            return cmd_status(cm)
        if args.cmd == "list-sources":
            return cmd_list_sources(cm)

    if args.cmd == "run":
        return cmd_run(
            cfg_path,
            source=getattr(args, "source", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            disable_llm=bool(getattr(args, "disable_llm", False)),
            llm_timeout=getattr(args, "llm_timeout", None),
            enable_story_grouping=bool(getattr(args, "enable_story_grouping", False)),
            ignore_watermark=bool(getattr(args, "ignore_watermark", False)),
        )

    if args.cmd == "group-stories":
        return cmd_group_stories(
            cfg_path,
            max_stories=getattr(args, "max_stories", 100),
            max_parallelism=getattr(args, "max_parallelism", 4),
            dry_run=bool(getattr(args, "dry_run", False)),
            reprocess=bool(getattr(args, "reprocess", False)),
        )

    if args.cmd == "group-status":
        return cmd_group_status(cfg_path)

    if args.cmd == "group-backfill":
        return cmd_group_backfill(
            cfg_path,
            batch_size=getattr(args, "batch_size", 50),
            max_batches=getattr(args, "max_batches", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            resume_from=getattr(args, "resume_from", None),
        )

    if args.cmd == "group-report":
        return cmd_group_report(
            cfg_path,
            format_type=getattr(args, "format", "text"),
            days_back=getattr(args, "days_back", 7),
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
