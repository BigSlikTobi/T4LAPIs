#!/usr/bin/env python3
"""
Test script to verify player deduplication logic works correctly.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data.transform import PlayerDataTransformer


def test_player_deduplication():
    """Test that player deduplication keeps the most recent record for each player."""
    
    print("Testing player deduplication logic...")
    
    # Create test data with duplicate player_ids but different seasons
    test_data = [
        {
            'player_id': '00-0012345',
            'player_name': 'John Doe', 
            'first_name': 'John',
            'last_name': 'Doe',
            'birth_date': '1990-01-01',
            'height': 72,
            'weight': 200,
            'college': 'University A',
            'position': 'QB',
            'rookie_year': 2015,
            'season': 2022,  # Older season
            'football_name': 'John',
            'jersey_number': 12,
            'status': 'ACT',
            'team': 'KC',
            'years_exp': 7,
            'headshot_url': 'http://example.com/1.jpg'
        },
        {
            'player_id': '00-0012345',  # Same player
            'player_name': 'John Doe', 
            'first_name': 'John',
            'last_name': 'Doe',
            'birth_date': '1990-01-01',
            'height': 72,
            'weight': 205,  # Updated weight
            'college': 'University A',
            'position': 'QB',
            'rookie_year': 2015,
            'season': 2024,  # More recent season
            'football_name': 'John',
            'jersey_number': 12,
            'status': 'ACT',
            'team': 'DEN',  # New team
            'years_exp': 9,
            'headshot_url': 'http://example.com/1.jpg'
        },
        {
            'player_id': '00-0067890',  # Different player, no duplicates
            'player_name': 'Jane Smith', 
            'first_name': 'Jane',
            'last_name': 'Smith',
            'birth_date': '1995-05-15',
            'height': 68,
            'weight': 180,
            'college': 'University B',
            'position': 'WR',
            'rookie_year': 2020,
            'season': 2024,
            'football_name': 'Jane',
            'jersey_number': 88,
            'status': 'ACT',
            'team': 'SF',
            'years_exp': 4,
            'headshot_url': 'http://example.com/2.jpg'
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(test_data)
    
    # Create transformer and run transformation
    transformer = PlayerDataTransformer()
    
    print(f"Input: {len(df)} records")
    print(f"Player IDs in input: {df['player_id'].tolist()}")
    
    # Transform the data (this will call deduplication)
    result = transformer.transform(df)
    
    print(f"Output: {len(result)} records")
    print(f"Player IDs in output: {[r['player_id'] for r in result]}")
    
    # Verify results
    assert len(result) == 2, f"Expected 2 records, got {len(result)}"
    
    # Find the John Doe record
    john_record = None
    jane_record = None
    
    for record in result:
        if record['player_id'] == '00-0012345':
            john_record = record
        elif record['player_id'] == '00-0067890':
            jane_record = record
    
    assert john_record is not None, "John Doe record not found"
    assert jane_record is not None, "Jane Smith record not found"
    
    # Verify that we kept the more recent John Doe record (2024, not 2022)
    assert john_record['last_active_season'] == 2024, f"Expected season 2024, got {john_record['last_active_season']}"
    assert john_record['latest_team'] == 'DEN', f"Expected team DEN, got {john_record['latest_team']}"
    assert john_record['weight'] == 205, f"Expected weight 205, got {john_record['weight']}"
    
    print("✅ Deduplication test passed!")
    print(f"✅ Kept most recent record for duplicate player (season {john_record['last_active_season']})")
    print(f"✅ Kept single record for non-duplicate player")
    
    return True


if __name__ == "__main__":
    try:
        test_player_deduplication()
        print("\n🎉 All tests passed! Player deduplication is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
