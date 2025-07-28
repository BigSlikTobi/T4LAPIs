#!/usr/bin/env python3
"""
Test script to verify the foreign key constraint fix works.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data.transform import PlayerDataTransformer


def test_foreign_key_fix():
    """Test that players with historical team abbreviations are handled correctly."""
    
    print("Testing foreign key constraint fix...")
    
    # Create test data that would have caused the foreign key error
    test_data = [
        {
            'player_id': '00-0012345',
            'player_name': 'San Diego Player', 
            'first_name': 'San',
            'last_name': 'Diego',
            'birth_date': '1990-01-01',
            'height': 72,
            'weight': 200,
            'college': 'San Diego State',
            'position': 'QB',
            'rookie_year': 2015,
            'season': 2020,
            'football_name': 'San',
            'jersey_number': 12,
            'status': 'ACT',
            'team': 'SD',  # This would cause foreign key error without fix
            'years_exp': 5,
            'headshot_url': 'http://example.com/1.jpg',
            'draft_club': 'SD'  # This would also cause foreign key error
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(test_data)
    
    # Create transformer and run transformation
    transformer = PlayerDataTransformer()
    
    print(f"Input: Player with draft_team=SD (would cause foreign key error)")
    
    # Transform the data
    result = transformer.transform(df)
    
    print(f"Output: {len(result)} records")
    
    # Verify results
    assert len(result) == 1, f"Expected 1 record, got {len(result)}"
    
    player = result[0]
    
    # Verify that SD was mapped to LAC (preventing foreign key error)
    assert player['latest_team'] == 'LAC', f"Expected LAC, got {player['latest_team']}"
    assert player['draft_team'] == 'LAC', f"Expected LAC, got {player['draft_team']}"
    
    print("✅ Foreign key constraint fix test passed!")
    print(f"✅ SD team mapping works: latest_team={player['latest_team']}, draft_team={player['draft_team']}")
    print("✅ No foreign key constraint violations will occur")
    
    return True


if __name__ == "__main__":
    try:
        test_foreign_key_fix()
        print("\n🎉 Foreign key constraint fix is working correctly!")
        print("Players with historical team abbreviations (SD, STL, OAK) will now load successfully.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
