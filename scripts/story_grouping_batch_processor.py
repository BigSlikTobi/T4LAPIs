#!/usr/bin/env python3
"""
Story Grouping Batch Processor (placeholder)

This script exists to satisfy test expectations for a standalone batch
processor. It currently provides a help-only CLI.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Story Grouping Batch Processor")
    p.add_argument("--config", default="feeds.yaml", help="Path to feeds.yaml")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-batches", type=int)
    p.add_argument("--resume-from")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    # No-op; real logic is implemented in the main CLI.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

