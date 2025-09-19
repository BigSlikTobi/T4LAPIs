#!/usr/bin/env python3
"""
CLI script for loading NFL Next Gen Stats data.
Supports loading different stat types (passing, rushing, receiving) by season.
"""

import sys
import os
from pathlib import Path

# Add project root to path (robust to nesting)
def _repo_root() -> str:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / 'src').exists() and (p / 'README.md').exists():
            return str(p)
    return str(start.parents[0])

project_root = _repo_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.ngs import NextGenStatsDataLoader
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


@handle_cli_errors
def main():
    """CLI interface for the Next Gen Stats data loader."""
    parser = setup_cli_parser("Load NFL Next Gen Stats data into the database")
    parser.add_argument("stat_type", choices=['passing', 'rushing', 'receiving', 'auto'], 
                       help="Type of NGS data to load")
    parser.add_argument("years", nargs='+', type=int, help="NFL season years (e.g., 2023 2024)")
    parser.add_argument(
        "--player-id",
        dest="player_ids",
        action="append",
        help="Filter by GSIS player ID (repeat for multiple or use comma-separated list)")
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    try:
        print("🏈 NFL Next Gen Stats Data Loader")
        if args.player_ids:
            flat_ids = []
            for v in args.player_ids:
                if v:
                    flat_ids.extend([p.strip() for p in str(v).split(',') if p.strip()])
            print(f"Loading {args.stat_type} NGS data for seasons: {', '.join(map(str, args.years))} (players: {', '.join(flat_ids)})")
        else:
            print(f"Loading {args.stat_type} NGS data for seasons: {', '.join(map(str, args.years))}")
        
        # Create loader and run
        loader = NextGenStatsDataLoader()
        # Normalize player_ids for loader
        player_id_param = None
        if args.player_ids:
            flat_ids = []
            for v in args.player_ids:
                if v:
                    flat_ids.extend([p.strip() for p in str(v).split(',') if p.strip()])
            player_id_param = flat_ids if len(flat_ids) > 1 else (flat_ids[0] if flat_ids else None)

        result = loader.load_data(
            stat_type=args.stat_type,
            years=args.years,
            player_id=player_id_param,
            dry_run=args.dry_run,
            clear_table=args.clear
        )
        
        # Print results using utility function
        suffix = ""
        if args.player_ids:
            suffix = f" (players: {', '.join(flat_ids)})"
        friendly_type = 'all (auto)' if args.stat_type == 'auto' else args.stat_type
        operation = f"Next Gen Stats {friendly_type} data load for seasons {', '.join(map(str, args.years))}{suffix}"
        print_results(result, operation, args.dry_run)
        
        if result["success"] and not args.dry_run:
            # Show current record count
            total_records = loader.get_record_count()
            print(f"Total NGS records in database: {total_records:,}")
        
        return result["success"]
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    exit_code_or_bool = main()
    # handle_cli_errors returns 0/1 for tests; pass through directly if int
    if isinstance(exit_code_or_bool, int):
        sys.exit(exit_code_or_bool)
    else:
        sys.exit(0 if exit_code_or_bool else 1)
