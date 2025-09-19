#!/usr/bin/env python3
"""
CLI script for loading Pro Football Reference (PFR) data.
Supports seasonal and weekly modes, with optional week and player filters.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.pfr import ProFootballReferenceDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    parser = setup_cli_parser("Load PFR data into the database")
    parser.add_argument("stat_type", choices=["pass", "rec", "rush"], help="PFR stat type")
    parser.add_argument("years", nargs='+', type=int, help="NFL season years (e.g., 2023 2024)")
    parser.add_argument("--weekly", action="store_true", help="Fetch weekly data instead of seasonal")
    parser.add_argument("--week", type=int, help="Filter to a specific week (weekly only). Defaults to latest week if omitted.")
    parser.add_argument("--player-id", dest="player_ids", action="append", help="Filter by GSIS or PFR player ID(s); GSIS will be mapped (repeat or comma-separated)")

    args = parser.parse_args()
    setup_cli_logging(args)

    try:
        print("🏈 PFR Data Loader")
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

        loader = ProFootballReferenceDataLoader()
        result = loader.load_data(
            stat_type=args.stat_type,
            years=args.years,
            weekly=args.weekly,
            week=args.week,
            player_id=player_param,
            dry_run=args.dry_run,
            clear_table=args.clear
        )

        mode = "weekly" if args.weekly else "seasonal"
        week_suffix = f" (week {args.week})" if args.weekly and args.week else (" (latest week)" if args.weekly else "")
        op = f"PFR {mode} {args.stat_type} data load for seasons {', '.join(map(str, args.years))}{week_suffix}{suffix}"
        print_results(result, op, args.dry_run)

        if result.get("success") and not args.dry_run:
            count = loader.get_record_count()
            print(f"Total PFR records in database: {count:,}")

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
