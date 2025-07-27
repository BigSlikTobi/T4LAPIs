"""Player weekly stats table data loader with upsert functionality."""

import logging
import sys
import os
from typing import Any, Dict, List, Optional

import pandas as pd

# Handle imports for both package usage and direct execution
if __name__ == "__main__":
    # When running directly, add the project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, project_root)
    from src.core.data.fetch import fetch_weekly_stats_data
    from src.core.data.transform import PlayerWeeklyStatsDataTransformer
    from src.core.db.database_init import get_supabase_client
else:
    # When imported as a package
    from ..fetch import fetch_weekly_stats_data
    from ..transform import PlayerWeeklyStatsDataTransformer
    from src.core.db.database_init import get_supabase_client


logger = logging.getLogger(__name__)


class PlayerWeeklyStatsDataLoader:
    """Loads player weekly stats data into the database with upsert functionality.
    
    Player weekly stats need upsert capability since stats might be updated
    as games progress or after final stats are confirmed.
    """
    
    def __init__(self):
        """Initialize the player weekly stats data loader."""
        self.table_name = "player_weekly_stats"
        self.supabase = get_supabase_client()
        
        if not self.supabase:
            raise RuntimeError("Could not initialize Supabase client")
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_weekly_stats(self, years: List[int] = None, weeks: List[int] = None, 
                         batch_size: int = 100) -> Dict[str, Any]:
        """Load player weekly stats data for specified years and weeks.
        
        Args:
            years: List of years to load data for. Defaults to [2024].
            weeks: List of weeks to load data for. If None, loads all weeks.
            batch_size: Number of records to process in each batch.
            
        Returns:
            Dictionary containing load results and statistics.
        """
        if years is None:
            years = [2024]
            
        self.logger.info(f"Starting weekly stats load for years: {years}, weeks: {weeks}")
        
        results = {
            'success': False,
            'total_records': 0,
            'loaded_records': 0,
            'failed_records': 0,
            'errors': []
        }
        
        try:
            # Initialize the transformer with Supabase client
            transformer = PlayerWeeklyStatsDataTransformer(self.supabase)
            
            # Fetch and transform data
            self.logger.info("Fetching weekly stats data...")
            df = transformer.fetch_data(years=years, weeks=weeks)
            
            if df.empty:
                self.logger.warning("No weekly stats data retrieved")
                results['success'] = True
                return results
            
            results['total_records'] = len(df)
            self.logger.info(f"Processing {len(df)} weekly stats records")
            
            # Transform data in batches
            transformed_records = []
            for _, record in df.iterrows():
                try:
                    transformed = transformer.transform_record(record.to_dict())
                    if transformed and transformer.validate_record(transformed):
                        transformed_records.append(transformed)
                    else:
                        results['failed_records'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing record: {e}")
                    results['failed_records'] += 1
                    results['errors'].append(str(e))
            
            self.logger.info(f"Successfully transformed {len(transformed_records)} records")
            
            # Load data in batches
            if transformed_records:
                loaded_count = self._load_batch(transformed_records, batch_size)
                results['loaded_records'] = loaded_count
                
            results['success'] = True
            self.logger.info(f"Weekly stats load completed. Loaded: {results['loaded_records']}, "
                           f"Failed: {results['failed_records']}")
            
        except Exception as e:
            error_msg = f"Error during weekly stats load: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            results['success'] = False
        
        return results
    
    def _load_batch(self, records: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Load a batch of weekly stats records using upsert."""
        total_loaded = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            try:
                # Use upsert to handle potential duplicates
                response = self.supabase.table(self.table_name).upsert(
                    batch, 
                    on_conflict="stat_id"
                ).execute()
                
                if response and hasattr(response, 'data'):
                    batch_count = len(response.data) if response.data else len(batch)
                    total_loaded += batch_count
                    self.logger.info(f"Loaded batch of {batch_count} weekly stats records "
                                   f"(Records {i+1}-{i+len(batch)})")
                else:
                    self.logger.warning(f"Unexpected response for batch {i//batch_size + 1}")
                    
            except Exception as e:
                self.logger.error(f"Error loading batch {i//batch_size + 1}: {e}")
                # Continue with next batch rather than failing completely
                continue
        
        return total_loaded
    
    def get_stats_count(self, week: int = None) -> int:
        """Get count of weekly stats records in the database.
        
        Args:
            week: Optional week filter.
            
        Returns:
            Number of weekly stats records matching the criteria.
        """
        try:
            query = self.supabase.table(self.table_name).select("*", count="exact")
            
            if week:
                # Since week is not a direct column, we need to join with games table
                # For now, just return total count if week is specified
                # In a production system, you'd want to join with the games table
                pass
                
            response = query.execute()
            
            if hasattr(response, 'count') and response.count is not None:
                return response.count
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Error getting weekly stats count: {e}")
            return 0
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of loaded weekly stats data.
        
        Returns:
            Dictionary containing validation results.
        """
        results = {
            'success': False,
            'total_stats': 0,
            'unique_players': 0,
            'unique_games': 0,
            'issues': []
        }
        
        try:
            # Get basic counts
            response = self.supabase.table(self.table_name).select(
                "stat_id,player_id,game_id", 
                count="exact"
            ).execute()
            
            if response.data:
                stats_data = response.data
                results['total_stats'] = len(stats_data)
                results['unique_players'] = len(set(stat['player_id'] for stat in stats_data))
                results['unique_games'] = len(set(stat['game_id'] for stat in stats_data))
                
                # Check for potential issues
                stat_ids = [stat['stat_id'] for stat in stats_data]
                if len(stat_ids) != len(set(stat_ids)):
                    results['issues'].append("Duplicate stat_id values found")
                
                # Check for missing required data
                for stat in stats_data:
                    if not stat.get('player_id'):
                        results['issues'].append("Records with missing player_id found")
                        break
                    if not stat.get('game_id'):
                        results['issues'].append("Records with missing game_id found")
                        break
            
            results['success'] = True
            self.logger.info(f"Data integrity validation completed")
            
        except Exception as e:
            error_msg = f"Error during data integrity validation: {e}"
            self.logger.error(error_msg)
            results['issues'].append(error_msg)
        
        return results


def main():
    """Main function for testing the loader."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        loader = PlayerWeeklyStatsDataLoader()
        
        # Load weekly stats for 2024, week 1 only for testing
        results = loader.load_weekly_stats(years=[2024], weeks=[1], batch_size=50)
        
        print(f"Load Results: {results}")
        
        # Check data integrity
        validation = loader.validate_data_integrity()
        print(f"Validation Results: {validation}")
        
        # Get count
        count = loader.get_stats_count()
        print(f"Total weekly stats records: {count}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
