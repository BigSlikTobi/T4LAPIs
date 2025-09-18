"""
Data fetching module for NFL data using nfl_data_py library.
This module provides functions to fetch raw data from various NFL data sources.
The goal is to separate data fetching from data transformation.
"""

import pandas as pd
import nfl_data_py as nfl
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def fetch_team_data() -> pd.DataFrame:
    """
    Fetch team data for the teams table.
    
    Returns:
        pd.DataFrame: Raw team description data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info("Fetching team data using nfl.import_team_desc()")
        team_df = nfl.import_team_desc()
        logger.info(f"Successfully fetched {len(team_df)} team records")
        return team_df
    except Exception as e:
        logger.error(f"Failed to fetch team data: {e}")
        raise


def fetch_player_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch player roster data for the players table.
    
    Args:
        years: List of years to fetch roster data for
        
    Returns:
        pd.DataFrame: Raw roster data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching seasonal roster data for years: {years}")
        roster_df = nfl.import_seasonal_rosters(years)
        logger.info(f"Successfully fetched {len(roster_df)} roster records for {len(years)} years")
        return roster_df
    except Exception as e:
        logger.error(f"Failed to fetch roster data for years {years}: {e}")
        raise


def fetch_game_schedule_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch game schedule data for the games table.
    
    Args:
        years: List of years to fetch schedule data for
        
    Returns:
        pd.DataFrame: Raw schedule data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching schedule data for years: {years}")
        schedule_df = nfl.import_schedules(years)
        logger.info(f"Successfully fetched {len(schedule_df)} schedule records for {len(years)} years")
        return schedule_df
    except Exception as e:
        logger.error(f"Failed to fetch schedule data for years {years}: {e}")
        raise


def fetch_seasonal_roster_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch seasonal roster data (alternative to regular rosters).
    
    Args:
        years: List of years to fetch seasonal roster data for
        
    Returns:
        pd.DataFrame: Raw seasonal roster data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching seasonal roster data for years: {years}")
        seasonal_roster_df = nfl.import_seasonal_rosters(years)
        logger.info(f"Successfully fetched {len(seasonal_roster_df)} seasonal roster records for {len(years)} years")
        return seasonal_roster_df
    except Exception as e:
        logger.error(f"Failed to fetch seasonal roster data for years {years}: {e}")
        raise


def fetch_weekly_roster_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch weekly roster data.
    
    Args:
        years: List of years to fetch weekly roster data for
        
    Returns:
        pd.DataFrame: Raw weekly roster data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching weekly roster data for years: {years}")
        weekly_roster_df = nfl.import_weekly_rosters(years)
        logger.info(f"Successfully fetched {len(weekly_roster_df)} weekly roster records for {len(years)} years")
        return weekly_roster_df
    except Exception as e:
        logger.error(f"Failed to fetch weekly roster data for years {years}: {e}")
        raise


def fetch_pbp_data(years: List[int], downsampling: bool = True) -> pd.DataFrame:
    """
    Fetch play-by-play data.
    
    Args:
        years: List of years to fetch play-by-play data for
        downsampling: Whether to downsample the data for performance
        
    Returns:
        pd.DataFrame: Raw play-by-play data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching play-by-play data for years: {years}, downsampling: {downsampling}")
        pbp_df = nfl.import_pbp_data(years, downsampling=downsampling)
        logger.info(f"Successfully fetched {len(pbp_df)} play-by-play records for {len(years)} years")
        return pbp_df
    except Exception as e:
        logger.error(f"Failed to fetch play-by-play data for years {years}: {e}")
        raise


def fetch_ngs_data(stat_type: str, years: List[int]) -> pd.DataFrame:
    """
    Fetch Next Gen Stats data.
    
    Args:
        stat_type: Type of NGS data ('passing', 'rushing', 'receiving')
        years: List of years to fetch NGS data for
        
    Returns:
        pd.DataFrame: Raw NGS data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching NGS {stat_type} data for years: {years}")
        ngs_df = nfl.import_ngs_data(stat_type, years)
        logger.info(f"Successfully fetched {len(ngs_df)} NGS {stat_type} records for {len(years)} years")
        return ngs_df
    except Exception as e:
        logger.error(f"Failed to fetch NGS {stat_type} data for years {years}: {e}")
        raise


def fetch_injury_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch injury data.
    
    Args:
        years: List of years to fetch injury data for
        
    Returns:
        pd.DataFrame: Raw injury data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching injury data for years: {years}")
        injury_df = nfl.import_injuries(years)
        logger.info(f"Successfully fetched {len(injury_df)} injury records for {len(years)} years")
        return injury_df
    except Exception as e:
        logger.error(f"Failed to fetch injury data for years {years}: {e}")
        raise


def fetch_roster_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch roster data specifically for the rosters table.
    
    Args:
        years: List of years to fetch roster data for
        
    Returns:
        pd.DataFrame: Raw roster data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching roster data for years: {years}")
        roster_df = nfl.import_seasonal_rosters(years)
        logger.info(f"Successfully fetched {len(roster_df)} roster records for {len(years)} years")
        return roster_df
    except Exception as e:
        logger.error(f"Failed to fetch roster data for years {years}: {e}")
        raise


def fetch_combine_data(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Fetch NFL Combine data.
    
    Args:
        years: List of years to fetch combine data for (if None, fetches all available)
        
    Returns:
        pd.DataFrame: Raw combine data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching combine data for years: {years if years else 'all available'}")
        combine_df = nfl.import_combine_data(years) if years else nfl.import_combine_data()
        logger.info(f"Successfully fetched {len(combine_df)} combine records")
        return combine_df
    except Exception as e:
        logger.error(f"Failed to fetch combine data for years {years}: {e}")
        raise


def fetch_draft_data(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Fetch NFL Draft data.
    
    Args:
        years: List of years to fetch draft data for (if None, fetches all available)
        
    Returns:
        pd.DataFrame: Raw draft data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching draft data for years: {years if years else 'all available'}")
        draft_df = nfl.import_draft_picks(years) if years else nfl.import_draft_picks()
        logger.info(f"Successfully fetched {len(draft_df)} draft records")
        return draft_df
    except Exception as e:
        logger.error(f"Failed to fetch draft data for years {years}: {e}")
        raise


def fetch_weekly_stats_data(years: List[int], weeks: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Fetch weekly player stats data for the player_weekly_stats table.
    
    Args:
        years: List of years to fetch weekly stats data for
        weeks: Optional list of weeks to filter by
        
    Returns:
        pd.DataFrame: Raw weekly stats data from nfl_data_py
        
    Raises:
        Exception: If data fetching fails
    """
    try:
        logger.info(f"Fetching weekly stats data for years: {years}, weeks: {weeks}")
        weekly_df = nfl.import_weekly_data(years=years)
        
        if weekly_df is None or weekly_df.empty:
            logger.warning("No weekly stats data retrieved")
            return pd.DataFrame()
        
        # Filter by weeks if specified
        if weeks is not None:
            initial_count = len(weekly_df)
            weekly_df = weekly_df[weekly_df['week'].isin(weeks)]
            logger.info(f"Filtered from {initial_count} to {len(weekly_df)} records for weeks {weeks}")
        
        logger.info(f"Successfully fetched {len(weekly_df)} weekly stats records")
        return weekly_df
    except Exception as e:
        logger.error(f"Failed to fetch weekly stats data for years {years}: {e}")
        raise
