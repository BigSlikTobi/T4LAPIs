#!/usr/bin/env python3
"""
Simple script to load NFL teams data into Supabase database.
This demonstrates how to use the TeamsDataLoader class.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.data.loaders.teams import TeamsDataLoader
from src.core.utils.logging import setup_logging


def main():
    """Load teams data into the database."""
    # Setup logging
    setup_logging()
    
    try:
        print("🏈 Starting NFL Teams Data Load")
        
        # Create loader instance
        loader = TeamsDataLoader()
        
        # Check if teams already exist
        existing_count = loader.get_existing_teams_count()
        print(f"Found {existing_count} existing team records")
        
        if existing_count > 0:
            print("Teams already exist in database")
            print("Use --clear flag with the loader to replace existing data")
            return 0
        
        # Load teams data using new interface
        result = loader.load_data()
        
        if result["success"]:
            final_count = loader.get_record_count()
            print(f"✅ Successfully loaded {final_count} NFL teams into database")
            print(f"Fetched: {result['total_fetched']} records")
            print(f"Validated: {result['total_validated']} records")
            print(f"Inserted: {result['upsert_result']['affected_rows']} records")
        else:
            print(f"❌ Failed to load teams: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
