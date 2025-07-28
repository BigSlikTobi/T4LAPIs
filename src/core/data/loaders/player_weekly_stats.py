"""Player weekly stats table data loader with upsert functionality."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseDataLoader
from ..fetch import fetch_weekly_stats_data
from ..transform import PlayerWeeklyStatsDataTransformer


class PlayerWeeklyStatsDataLoader(BaseDataLoader):
    """Loads player weekly stats data into the database with upsert functionality.
    
    Player weekly stats need upsert capability since stats might be updated
    as games progress or after final stats are confirmed.
    """
    
    def __init__(self):
        """Initialize the player weekly stats data loader."""
        super().__init__("player_weekly_stats")
    
    def fetch_raw_data(self, years: List[int] = None, weeks: List[int] = None, **kwargs) -> Any:
        """Fetch raw player weekly stats data.
        
        Args:
            years: List of years to load data for. Defaults to [2024].
            weeks: List of weeks to load data for. If None, loads all weeks.
            **kwargs: Additional arguments passed to fetch function.
            
        Returns:
            Raw data from the fetch function.
        """
        if years is None:
            years = [2024]
            
        self.logger.info(f"Fetching weekly stats data for years: {years}, weeks: {weeks}")
        return fetch_weekly_stats_data(years=years, weeks=weeks)
    
    @property
    def transformer_class(self):
        """Return the transformer class for player weekly stats data."""
        return PlayerWeeklyStatsDataTransformer
    
    def load_weekly_stats(self, years: List[int] = None, weeks: List[int] = None, 
                         batch_size: int = 100, **kwargs) -> Dict[str, Any]:
        """Load player weekly stats data for specified years and weeks.
        
        Args:
            years: List of years to load data for. Defaults to [2024].
            weeks: List of weeks to load data for. If None, loads all weeks.
            batch_size: Number of records to process in each batch.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary containing load results and statistics.
        """
        return self.load_data(
            years=years,
            weeks=weeks,
            batch_size=batch_size,
            **kwargs
        )
    
    def get_stats_count(self, week: int = None) -> int:
        """Get count of weekly stats records in the database.
        
        Args:
            week: Optional week filter.
            
        Returns:
            Number of weekly stats records matching the criteria.
        """
        return self.db_manager.get_record_count(self.table_name)

# Legacy compatibility - this can be removed in future versions
def main():
    """Simple test function for the loader."""
    import sys
    from src.core.utils.logging import setup_logging
    
    setup_logging()
    
    try:
        loader = PlayerWeeklyStatsDataLoader()
        
        # Load weekly stats for 2024, week 1 only for testing
        results = loader.load_weekly_stats(years=[2024], weeks=[1], batch_size=50)
        
        print(f"Load Results: {results}")
        
        # Get count
        count = loader.get_stats_count()
        print(f"Total weekly stats records: {count}")
        
        return 0 if results['success'] else 1
        
    except Exception as e:
        print(f"Error in main: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
