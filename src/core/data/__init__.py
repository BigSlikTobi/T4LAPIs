"""
Data layer for NFL data fetching and transformation.
"""

from .fetch import (
    fetch_team_data,
    fetch_player_data,
    fetch_game_schedule_data,
    fetch_weekly_stats_data,
    fetch_seasonal_roster_data,
    fetch_weekly_roster_data,
    fetch_pbp_data,
    fetch_ngs_data,
    fetch_injury_data,
    fetch_combine_data,
    fetch_draft_data,
)

from .transform import (
    BaseDataTransformer,
    TeamDataTransformer,
    PlayerDataTransformer,
    GameDataTransformer,
)

from .loaders import (
    TeamsDataLoader,
    PlayersDataLoader,
    GamesDataLoader,
)

__all__ = [
    # Fetch functions
    'fetch_team_data',
    'fetch_player_data',
    'fetch_game_schedule_data',
    'fetch_weekly_stats_data',
    'fetch_seasonal_roster_data',
    'fetch_weekly_roster_data',
    'fetch_pbp_data',
    'fetch_ngs_data',
    'fetch_injury_data',
    'fetch_combine_data',
    'fetch_draft_data',
    # Transform classes
    'BaseDataTransformer',
    'TeamDataTransformer',
    'PlayerDataTransformer',
    'GameDataTransformer',
    # Loader classes
    'TeamsDataLoader',
    'PlayersDataLoader',
    'GamesDataLoader',
]
