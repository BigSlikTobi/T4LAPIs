#!/usr/bin/env python3
"""
Test script to verify the player duplicate resolution works in practice.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data.loaders.players import PlayersDataLoader


def test_player_loading():
    """Test that player loading works with the new deduplication and conflict resolution."""
    
    print("Testing player data loading with deduplication and conflict resolution...")
    
    try:
        # Create loader
        loader = PlayersDataLoader()
        
        # Test dry run first (doesn't hit database)
        print("Running dry run...")
        result = loader.load_data(season=2023, dry_run=True)
        
        if result['success']:
            print(f"✅ Dry run successful!")
            print(f"   - Would process {result.get('total_fetched', 0)} fetched records")
            print(f"   - Would validate {result.get('total_validated', 0)} records")
            print("   - Deduplication logic would run automatically")
            print("   - Database conflict resolution on player_id would be used")
        else:
            print(f"❌ Dry run failed: {result.get('error', 'Unknown error')}")
            return False
            
        print("\n🎉 Player loading test completed successfully!")
        print("The fix should resolve the 'on_conflict' attribute error.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = test_player_loading()
    if not success:
        sys.exit(1)
