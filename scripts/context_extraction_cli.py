"""Shim exposing the context-extraction-only CLI."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_impl = importlib.import_module("scripts.news_ingestion.context_extraction_cli")

run_context_pipeline = _impl.run_context_pipeline  # type: ignore[attr-defined]
build_parser = _impl.build_parser  # type: ignore[attr-defined]
extract_context_batch = _impl.extract_context_batch  # type: ignore[attr-defined]
dump_context_from_supabase = _impl.dump_context_from_supabase  # type: ignore[attr-defined]


def main(argv: Optional[List[str]] = None) -> int:
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
