#!/usr/bin/env python3
"""
CLI tool for testing and demonstrating the entity dictionary builder.

This script allows you to build and inspect the entity dictionary that maps
player names and team names/abbreviations to their unique IDs.
"""

import sys
import argparse
import json
from typing import Dict, Any
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.data.entity_linking import build_entity_dictionary
from src.core.utils.cli import setup_cli_parser, setup_cli_logging, handle_cli_errors, print_results
from src.core.utils.logging import get_logger


@handle_cli_errors
def main():
    """CLI interface for the entity dictionary builder."""
    parser = setup_cli_parser("Build and inspect entity dictionary for NFL players and teams")
    parser.add_argument(
        "--output-file",
        help="Save dictionary to JSON file"
    )
    parser.add_argument(
        "--search",
        help="Search for specific entity in dictionary"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of sample entries to display (default: 10)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, not sample entries"
    )
    
    args = parser.parse_args()
    setup_cli_logging(args)
    
    logger = get_logger(__name__)
    
    try:
        print("🏈 NFL Entity Dictionary Builder")
        print("Building entity dictionary from players and teams tables...")
        print()
        
        # Build the entity dictionary
        entity_dict = build_entity_dictionary()
        
        if not entity_dict:
            print("❌ No entities found in dictionary")
            return False
        
        # Calculate statistics
        player_count = sum(1 for k, v in entity_dict.items() if v.startswith('00-'))
        team_count = len(entity_dict) - player_count
        
        print(f"✅ Successfully built entity dictionary")
        print(f"📊 Statistics:")
        print(f"   Total entities: {len(entity_dict)}")
        print(f"   Players: {player_count}")
        print(f"   Teams: {team_count}")
        print()
        
        # Search functionality
        if args.search:
            search_term = args.search.strip()
            matches = {k: v for k, v in entity_dict.items() if search_term.lower() in k.lower()}
            
            if matches:
                print(f"🔍 Search results for '{search_term}':")
                for name, entity_id in sorted(matches.items()):
                    entity_type = "Player" if entity_id.startswith('00-') else "Team"
                    print(f"   {name} → {entity_id} ({entity_type})")
            else:
                print(f"🔍 No matches found for '{search_term}'")
            print()
        
        # Display sample entries
        if not args.stats_only:
            print(f"📋 Sample entries (showing {min(args.sample_size, len(entity_dict))}):")
            
            # Show some players
            player_entries = {k: v for k, v in entity_dict.items() if v.startswith('00-')}
            if player_entries:
                print("   Players:")
                for i, (name, player_id) in enumerate(sorted(player_entries.items())):
                    if i >= args.sample_size // 2:
                        break
                    print(f"     {name} → {player_id}")
            
            # Show some teams
            team_entries = {k: v for k, v in entity_dict.items() if not v.startswith('00-')}
            if team_entries:
                print("   Teams:")
                for i, (name, team_abbr) in enumerate(sorted(team_entries.items())):
                    if i >= args.sample_size // 2:
                        break
                    print(f"     {name} → {team_abbr}")
            print()
        
        # Save to file if requested
        if args.output_file:
            try:
                with open(args.output_file, 'w') as f:
                    json.dump(entity_dict, f, indent=2, sort_keys=True)
                print(f"💾 Dictionary saved to {args.output_file}")
            except Exception as e:
                logger.error(f"Failed to save dictionary to file: {e}")
                print(f"❌ Failed to save to file: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Entity dictionary building failed: {e}")
        print(f"❌ Failed to build entity dictionary: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
