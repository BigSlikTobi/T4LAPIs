"""Shim exposing the fetch-sources CLI."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_impl = importlib.import_module("scripts.news_ingestion.fetch_sources_cli")

fetch_sources = _impl.fetch_sources  # type: ignore[attr-defined]
run_fetch_cli = _impl.run_fetch_cli  # type: ignore[attr-defined]
build_parser = _impl.build_parser  # type: ignore[attr-defined]


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
