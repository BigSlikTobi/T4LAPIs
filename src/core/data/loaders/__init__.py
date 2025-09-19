"""
Data loaders package for NFL data ingestion.
This package provides loaders for all types of NFL data sources.
"""

from .base import BaseDataLoader
from .teams import TeamsDataLoader
from .players import PlayersDataLoader
from .games import GamesDataLoader
from .player_weekly_stats import PlayerWeeklyStatsDataLoader

# Phase A: Core Analytics
from .pbp import PlayByPlayDataLoader
from .ngs import NextGenStatsDataLoader
from .snap_counts import SnapCountsDataLoader

# Phase B: Roster/Personnel Context
from .depth_charts import DepthChartsDataLoader
from .contracts import ContractsDataLoader
from .combine import CombineDataLoader
from .draft import DraftDataLoader, DraftValuesDataLoader

# Phase C: Reference + Advanced
from .pfr import ProFootballReferenceDataLoader
from .ftn import FootballStudyHallDataLoader
from .officials import OfficialsDataLoader
from .qbr import ESPNQBRDataLoader
from .lines import LinesDataLoader, WinTotalsDataLoader

__all__ = [
    'BaseDataLoader',
    'TeamsDataLoader',
    'PlayersDataLoader', 
    'GamesDataLoader',
    'PlayerWeeklyStatsDataLoader',
    # Phase A: Core Analytics
    'PlayByPlayDataLoader',
    'NextGenStatsDataLoader',
    'SnapCountsDataLoader',
    # Phase B: Roster/Personnel Context
    'DepthChartsDataLoader',
    'ContractsDataLoader',
    'CombineDataLoader',
    'DraftDataLoader',
    'DraftValuesDataLoader',
    # Phase C: Reference + Advanced
    'ProFootballReferenceDataLoader',
    'FootballStudyHallDataLoader',
    'OfficialsDataLoader',
    'ESPNQBRDataLoader',
    'LinesDataLoader',
    'WinTotalsDataLoader',
]
