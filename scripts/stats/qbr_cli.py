#!/usr/bin/env python3
"""
CLI script for loading ESPN QBR data.
Supports week and player filters with sensible defaults.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.qbr import ESPNQBRDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    parser = setup_cli_parser("Load ESPN QBR data into the database")
    parser.add_argument("years", nargs='+', type=int, help="NFL season years (e.g., 2023 2024)")
    parser.add_argument("--week", type=int, help="Filter to a specific week. Defaults to latest week if omitted.")
    parser.add_argument("--player-id", dest="player_ids", action="append", help="Filter by player ID (repeat or comma-separated)")

    args = parser.parse_args()
    setup_cli_logging(args)

    try:
        print("🏈 ESPN QBR Data Loader")
        if args.player_ids:
            flat = []
            for v in args.player_ids:
                if v:
                    flat.extend([p.strip() for p in str(v).split(',') if p.strip()])
            player_param = flat if len(flat) > 1 else (flat[0] if flat else None)
            suffix = f" (players: {', '.join(flat)})"
        else:
            player_param = None
            suffix = ""

        loader = ESPNQBRDataLoader()
        result = loader.load_data(
            years=args.years,
            week=args.week,
            player_id=player_param,
            dry_run=args.dry_run,
            clear_table=args.clear
        )

        week_suffix = f" (week {args.week})" if args.week else " (latest week)"
        op = f"QBR data load for seasons {', '.join(map(str, args.years))}{week_suffix}{suffix}"
        print_results(result, op, args.dry_run)

        if result.get("success") and not args.dry_run:
            count = loader.get_record_count()
            print(f"Total QBR records in database: {count:,}")

        return result.get("success")
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    exit_code_or_bool = main()
    if isinstance(exit_code_or_bool, int):
        sys.exit(exit_code_or_bool)
    else:
        sys.exit(0 if exit_code_or_bool else 1)
