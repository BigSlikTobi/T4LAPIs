#!/usr/bin/env python3
"""CLI script for loading NFL roster data."""

import sys
from pathlib import Path


def _repo_root() -> str:
    start = Path(__file__).resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() and (candidate / "README.md").exists():
            return str(candidate)
    return str(start.parents[0])


project_root = _repo_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.rosters import RostersDataLoader
from src.core.utils.cli import handle_cli_errors, print_results, setup_cli_logging, setup_cli_parser


@handle_cli_errors
def main() -> bool:
    """CLI interface for the rosters data loader."""
    parser = setup_cli_parser("Load NFL roster data into the database")
    parser.add_argument("season", type=int, help="NFL season year (e.g., 2025)")
    parser.add_argument(
        "--include-records",
        action="store_true",
        help="Include transformed records in dry-run output for inspection",
    )

    args = parser.parse_args()
    setup_cli_logging(args)

    print("🏈 NFL Rosters Data Loader")
    print(f"Loading roster data for season {args.season}")

    loader = RostersDataLoader()
    result = loader.load_data(
        season=args.season,
        dry_run=args.dry_run,
        clear_table=args.clear,
        include_records=args.include_records,
    )

    operation = f"roster data load for season {args.season}"
    print_results(result, operation, args.dry_run)

    version = result.get("version")
    skipped = result.get("skipped")

    if result.get("success"):
        if args.dry_run:
            if version is not None:
                print(f"Would apply version: {version}")
            if skipped:
                _print_skipped(skipped, prefix="Would skip")
        else:
            if version is not None:
                print(f"Applied version: {version}")
            if skipped:
                _print_skipped(skipped, prefix="Skipped records")

    return bool(result.get("success"))


def _print_skipped(skipped: list[dict], *, prefix: str) -> None:
    print(f"{prefix}: {len(skipped)}")
    preview = skipped[:5]
    for idx, entry in enumerate(preview, start=1):
        formatted = ", ".join(f"{key}={value}" for key, value in entry.items())
        print(f"  - [{idx}] {formatted}")
    remaining = len(skipped) - len(preview)
    if remaining > 0:
        print(f"  ... and {remaining} more")


if __name__ == "__main__":
    main()
