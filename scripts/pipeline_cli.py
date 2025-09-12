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
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
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


def cmd_run(cfg_path: str, *, source: Optional[str], dry_run: bool, disable_llm: bool, llm_timeout: Optional[float]) -> int:
    cm = ConfigManager(cfg_path)
    cm.load_config()
    storage = _build_storage(dry_run)
    audit = AuditLogger(storage=storage)

    # LLM runtime controls
    if disable_llm:
        os.environ["NEWS_PIPELINE_DISABLE_LLM"] = "1"
    if llm_timeout is not None:
        os.environ["OPENAI_TIMEOUT"] = str(llm_timeout)

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

    # validate
    pv = sub.add_parser("validate", help="Validate configuration and show warnings")
    pv.add_argument("--config", default="feeds.yaml")

    # status
    ps = sub.add_parser("status", help="Show environment and DB connectivity status")
    ps.add_argument("--config", default="feeds.yaml")

    # list-sources
    pl = sub.add_parser("list-sources", help="List enabled sources")
    pl.add_argument("--config", default="feeds.yaml")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_path = str(Path(args.config)) if hasattr(args, "config") else "feeds.yaml"
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
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
