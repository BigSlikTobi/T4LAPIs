"""Shim exposing the full story pipeline CLI for tests and tooling."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_impl = importlib.import_module("scripts.news_ingestion.full_story_pipeline_cli")

run_full_pipeline = _impl.run_full_pipeline  # type: ignore[attr-defined]
build_parser = _impl.build_parser  # type: ignore[attr-defined]


def main(argv: Optional[List[str]] = None) -> int:
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
