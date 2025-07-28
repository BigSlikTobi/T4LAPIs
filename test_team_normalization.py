#!/usr/bin/env python3
"""
Test script to verify team abbreviation normalization works correctly.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data.transform import PlayerDataTransformer


def test_team_normalization():
    """Test that team abbreviation normalization works correctly."""
    
    print("Testing team abbreviation normalization...")
    
    # Create test data with historical team abbreviations
    test_data = [
        {
            'player_id': '00-0012345',
            'player_name': 'Test Player 1', 
            'first_name': 'Test',
            'last_name': 'Player1',
            'birth_date': '1990-01-01',
            'height': 72,
            'weight': 200,
            'college': 'University A',
            'position': 'QB',
            'rookie_year': 2015,
            'season': 2020,
            'football_name': 'Test',
            'jersey_number': 12,
            'status': 'ACT',
            'team': 'SD',  # San Diego Chargers - should be mapped to LAC
            'years_exp': 5,
            'headshot_url': 'http://example.com/1.jpg',
            'draft_club': 'SD'  # Should also be mapped
        },
        {
            'player_id': '00-0067890',
            'player_name': 'Test Player 2', 
            'first_name': 'Test',
            'last_name': 'Player2',
            'birth_date': '1995-05-15',
            'height': 68,
            'weight': 180,
            'college': 'University B',
            'position': 'WR',
            'rookie_year': 2018,
            'season': 2020,
            'football_name': 'Test',
            'jersey_number': 88,
            'status': 'ACT',
            'team': 'KC',  # Current team - should remain KC
            'years_exp': 2,
            'headshot_url': 'http://example.com/2.jpg',
            'draft_club': 'STL'  # St. Louis Rams - should be mapped to LAR
        },
        {
            'player_id': '00-0098765',
            'player_name': 'Test Player 3', 
            'first_name': 'Test',
            'last_name': 'Player3',
            'birth_date': '1992-03-10',
            'height': 70,
            'weight': 190,
            'college': 'University C',
            'position': 'RB',
            'rookie_year': 2016,
            'season': 2020,
            'football_name': 'Test',
            'jersey_number': 22,
            'status': 'ACT',
            'team': 'OAK',  # Oakland Raiders - should be mapped to LV
            'years_exp': 4,
            'headshot_url': 'http://example.com/3.jpg',
            'draft_club': 'INVALID'  # Invalid team - should be set to None
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(test_data)
    
    # Create transformer and run transformation
    transformer = PlayerDataTransformer()
    
    print(f"Input: {len(df)} records")
    
    # Transform the data
    result = transformer.transform(df)
    
    print(f"Output: {len(result)} records")
    
    # Verify results
    assert len(result) == 3, f"Expected 3 records, got {len(result)}"
    
    # Check team mappings
    player1 = next(r for r in result if r['player_id'] == '00-0012345')
    player2 = next(r for r in result if r['player_id'] == '00-0067890')
    player3 = next(r for r in result if r['player_id'] == '00-0098765')
    
    # Verify SD -> LAC mapping
    assert player1['latest_team'] == 'LAC', f"Expected LAC, got {player1['latest_team']}"
    assert player1['draft_team'] == 'LAC', f"Expected LAC, got {player1['draft_team']}"
    
    # Verify KC remains KC
    assert player2['latest_team'] == 'KC', f"Expected KC, got {player2['latest_team']}"
    # Verify STL -> LAR mapping
    assert player2['draft_team'] == 'LAR', f"Expected LAR, got {player2['draft_team']}"
    
    # Verify OAK -> LV mapping
    assert player3['latest_team'] == 'LV', f"Expected LV, got {player3['latest_team']}"
    # Verify invalid team -> None
    assert player3['draft_team'] is None, f"Expected None, got {player3['draft_team']}"
    
    print("✅ Team normalization test passed!")
    print(f"✅ SD -> LAC: {player1['latest_team']}")
    print(f"✅ STL -> LAR: {player2['draft_team']}")
    print(f"✅ OAK -> LV: {player3['latest_team']}")
    print(f"✅ INVALID -> None: {player3['draft_team']}")
    
    return True


if __name__ == "__main__":
    try:
        test_team_normalization()
        print("\n🎉 All tests passed! Team normalization is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
