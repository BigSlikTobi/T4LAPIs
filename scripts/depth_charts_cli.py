#!/usr/bin/env python3
"""
CLI script for loading NFL Depth Charts data.
Supports loading by season with standard options (dry-run, clear, verbose).
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.depth_charts import DepthChartsDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the depth charts data loader."""
    parser = setup_cli_parser("Load NFL depth charts data into the database")
    parser.add_argument("years", nargs='+', type=int, help="NFL season years (e.g., 2023 2024)")
    parser.add_argument(
        "--team",
        action='append',
        help="Filter by team code (canonical, e.g., KC). Can be repeated for multiple teams."
    )
    parser.add_argument(
        "--latest-only",
        dest='latest_only',
        action='store_true',
        help="For each team, keep only the latest available depth chart snapshot (uses dt for 2025+)."
    )
    parser.add_argument(
        "--all-snapshots",
        dest='latest_only',
        action='store_false',
        help="Disable latest-only and return all snapshots for the season (can be large)."
    )
    parser.set_defaults(latest_only=True)
    parser.add_argument(
        "--as-of",
        help="Optional cutoff timestamp (e.g., 2025-09-15T00:00:00Z) to use with --latest-only."
    )

    args = parser.parse_args()
    setup_cli_logging(args)

    try:
        print("🏈 NFL Depth Charts Data Loader")
        print(f"Loading depth charts for seasons: {', '.join(map(str, args.years))}")

        # Create loader and run
        loader = DepthChartsDataLoader()
        # If exactly one team and dry-run, include full records to display
        include_records = bool(args.dry_run and args.team and len(args.team) == 1)

        result = loader.load_data(
            years=args.years,
            teams=args.team,
            latest_only=args.latest_only,
            as_of=args.as_of,
            dry_run=args.dry_run,
            clear_table=args.clear,
            include_records=include_records
        )

        # Print results using utility function
        scope = f" for teams {', '.join(args.team)}" if args.team else ""
        if args.latest_only:
            scope += " (latest-only)"
        operation = f"depth charts data load for seasons {', '.join(map(str, args.years))}{scope}"
        print_results(result, operation, args.dry_run)

        # If single-team dry run, print all records for review
        if args.dry_run and args.team and len(args.team) == 1 and result.get('records'):
            team = args.team[0]
            print(f"\nShowing all {len(result['records'])} depth chart rows for team {team}:")
            # Print concise rows
            for r in result['records']:
                print(f"{r.get('team')} | {r.get('pos_grp')} | {r.get('pos_abb')} | {r.get('pos_name')} | slot={r.get('pos_slot')} | rank={r.get('pos_rank')} | {r.get('player_name')} ({r.get('gsis_id')})")

        if result.get("success") and not args.dry_run:
            total_records = loader.get_record_count()
            print(f"Total depth chart records in database: {total_records:,}")

        return result.get("success", False)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    sys.exit(main())
