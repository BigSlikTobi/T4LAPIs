"""
Pipeline CLI shim used by tests as `scripts.pipeline_cli`.

This module forwards calls to the real implementation in
`scripts/news_ingestion/pipeline_cli.py` while ensuring that names patched by
tests (e.g., `_build_storage`, `_build_story_grouping_orchestrator`,
`ConfigManager`, `NFLNewsPipeline`) are respected.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, List


# Ensure repo root is on sys.path so absolute imports like `scripts.*` resolve
def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists() and (p / "README.md").exists():
            return p
    # Fallback to the parent of this file's directory
    return start.parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_impl = importlib.import_module("scripts.news_ingestion.pipeline_cli")

# Expose patchable attributes to match test expectations
_build_storage = _impl._build_storage  # type: ignore[attr-defined]
_build_story_grouping_orchestrator = _impl._build_story_grouping_orchestrator  # type: ignore[attr-defined]
ConfigManager = _impl.ConfigManager
NFLNewsPipeline = _impl.NFLNewsPipeline

# Keep stable references to impl command functions to avoid accidental recursion
_IMPL_CMD_GROUP_STORIES = _impl.cmd_group_stories
_IMPL_CMD_GROUP_STATUS = _impl.cmd_group_status
_IMPL_CMD_GROUP_BACKFILL = _impl.cmd_group_backfill
_IMPL_CMD_GROUP_REPORT = _impl.cmd_group_report
_IMPL_CMD_RUN = _impl.cmd_run
_IMPL_BUILD_PARSER = _impl.build_parser


def _sync_patchables_into_impl() -> None:
    """Copy currently-bound shim attributes into the impl module.

    Allows tests to patch on this module path and have the impl use them.
    """
    setattr(_impl, "_build_storage", globals().get("_build_storage", _impl._build_storage))  # type: ignore[attr-defined]
    setattr(
        _impl,
        "_build_story_grouping_orchestrator",
        globals().get("_build_story_grouping_orchestrator", _impl._build_story_grouping_orchestrator),  # type: ignore[attr-defined]
    )
    setattr(_impl, "ConfigManager", globals().get("ConfigManager", _impl.ConfigManager))
    setattr(_impl, "NFLNewsPipeline", globals().get("NFLNewsPipeline", _impl.NFLNewsPipeline))


# Lightweight forwarders that keep patchables in sync
def build_parser():
    return _IMPL_BUILD_PARSER()


def cmd_run(
    cfg_path: str,
    *,
    source: Optional[str] = None,
    dry_run: bool = False,
    disable_llm: bool = False,
    llm_timeout: Optional[float] = None,
    enable_story_grouping: bool = False,
    ignore_watermark: bool = False,
) -> int:
    _sync_patchables_into_impl()
    return _IMPL_CMD_RUN(
        cfg_path,
        source=source,
        dry_run=dry_run,
        disable_llm=disable_llm,
        llm_timeout=llm_timeout,
        enable_story_grouping=enable_story_grouping,
        ignore_watermark=ignore_watermark,
    )


def cmd_group_stories(cfg_path: str, *, max_stories: int, max_parallelism: int, dry_run: bool, reprocess: bool) -> int:
    _sync_patchables_into_impl()
    return _IMPL_CMD_GROUP_STORIES(
        cfg_path,
        max_stories=max_stories,
        max_parallelism=max_parallelism,
        dry_run=dry_run,
        reprocess=reprocess,
    )


def cmd_group_status(cfg_path: str) -> int:
    _sync_patchables_into_impl()
    return _IMPL_CMD_GROUP_STATUS(cfg_path)


def cmd_group_backfill(
    cfg_path: str, *, batch_size: int, max_batches: Optional[int], dry_run: bool, resume_from: Optional[str]
) -> int:
    _sync_patchables_into_impl()
    return _IMPL_CMD_GROUP_BACKFILL(
        cfg_path,
        batch_size=batch_size,
        max_batches=max_batches,
        dry_run=dry_run,
        resume_from=resume_from,
    )


def cmd_group_report(cfg_path: str, *, format_type: str, days_back: int) -> int:
    _sync_patchables_into_impl()
    return _IMPL_CMD_GROUP_REPORT(cfg_path, format_type=format_type, days_back=days_back)


def main(argv: Optional[List[str]] = None) -> int:
    _sync_patchables_into_impl()
    # If tests patched command functions on this module, forward them to impl
    for _name in ("cmd_group_stories", "cmd_group_status", "cmd_group_backfill", "cmd_group_report"):
        if _name in globals():
            setattr(_impl, _name, globals()[_name])
    return _impl.main(argv)


if __name__ == "__main__":
    # Allow running this shim directly as a script
    _sync_patchables_into_impl()
    sys.exit(_impl.main(None))
