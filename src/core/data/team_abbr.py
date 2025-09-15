"""Utilities for normalizing NFL team abbreviations to canonical forms.

This ensures consistency across loaders and avoids foreign key violations
caused by historical or alternative abbreviations from various data sources.
"""

from typing import Any, Optional
import pandas as pd


# Canonical abbreviations used in our database
CANONICAL = {
    'ARI',
    'ATL',
    'BAL',
    'BUF',
    'CAR',
    'CHI',
    'CIN',
    'CLE',
    'DAL',
    'DEN',
    'DET',
    'GB',
    'HOU',
    'IND',
    'JAX',
    'KC',
    'LAC',
        'LA',
    'LV',
    'MIA',
    'MIN',
    'NE',
    'NO',
    'NYG',
    'NYJ',
    'PHI',
    'PIT',
    'SEA',
    'SF',
    'TB',
    'TEN',
    'WAS'
}


# Mapping of historical/alternate codes to canonical ones
TEAM_ABBR_MAP = {
    # Los Angeles / St. Louis Rams
        'STL': 'LA', 'LAR': 'LA', 'LA': 'LA',
    # Raiders
    'OAK': 'LV', 'LVR': 'LV',
    # Chargers
    'SD': 'LAC', 'SDG': 'LAC',
    # Jaguars
    'JAC': 'JAX',
    # Washington
    'WSH': 'WAS',
    # Packers
    'GNB': 'GB',
    # Chiefs
    'KAN': 'KC',
    # 49ers
    'SFO': 'SF',
    # Buccaneers
    'TAM': 'TB',
    # Saints / Patriots older codes
    'NOR': 'NO', 'NWE': 'NE',
    # Cardinals historic (Phoenix)
    'PHX': 'ARI', 'PHO': 'ARI', 'ARZ': 'ARI',
    # Cleveland alt
    'CLV': 'CLE',
}


def normalize_team_abbr(team_abbr: Any) -> Optional[str]:
    """Normalize a team abbreviation to our canonical set.

    Returns None for unknown/invalid values to avoid FK issues.
    """
    if pd.isna(team_abbr) or not team_abbr:
        return None

    code = str(team_abbr).upper().strip()

    # First apply mapping if present
    if code in TEAM_ABBR_MAP:
        code = TEAM_ABBR_MAP[code]

    # If already canonical, accept
    if code in CANONICAL:
        return code

    # If not canonical but 2-3 chars, keep only if it looks like a known variant mapping
    # Otherwise, return None to avoid inserting invalid FKs
    return None
