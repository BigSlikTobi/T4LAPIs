"""Base data loader class providing common functionality for all NFL data loaders."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import pandas as pd

from ..transform import BaseDataTransformer
from ...utils.database import DatabaseManager
from ...utils.logging import get_logger


class BaseDataLoader(ABC):
    """Abstract base class for all NFL data loaders.
    
    Provides common functionality for fetching, transforming, and loading
    NFL data into the database with consistent error handling and logging.
    """
    
    def __init__(self, table_name: str):
        """Initialize the base data loader.
        
        Args:
            table_name: Name of the database table to manage
            
        Raises:
            RuntimeError: If database connection cannot be established
        """
        self.table_name = table_name
        self.db_manager = DatabaseManager(table_name)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Set by subclasses
        self._transformer_class: Optional[Type[BaseDataTransformer]] = None
    
    @property
    @abstractmethod
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the transformer class for this loader.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def fetch_raw_data(self, **kwargs) -> pd.DataFrame:
        """Fetch raw data from the NFL data source.
        
        Must be implemented by subclasses to define how to fetch their specific data type.
        
        Returns:
            Raw DataFrame from nfl_data_py or other source
        """
        pass
    
    def transform_data(self, raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform raw data using the appropriate transformer.
        
        Args:
            raw_data: Raw DataFrame from fetch_raw_data()
            
        Returns:
            List of validated and transformed records ready for database insertion
        """
        self.logger.info(f"Transforming {len(raw_data)} records")
        
        # Check if transformer requires db_manager (like PlayerWeeklyStatsDataTransformer)
        import inspect
        transformer_init = self.transformer_class.__init__
        init_params = inspect.signature(transformer_init).parameters
        
        if 'db_manager' in init_params:
            transformer = self.transformer_class(self.db_manager)
        else:
            transformer = self.transformer_class()
            
        transformed_records = transformer.transform(raw_data)
        
        self.logger.info(f"Successfully transformed {len(transformed_records)} records")
        return transformed_records
    
    def load_data(self, dry_run: bool = False, clear_table: bool = False,
                  include_records: bool = False, **fetch_kwargs) -> Dict[str, Any]:
        """Complete workflow to load data into the database.
        
        Args:
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            **fetch_kwargs: Arguments passed to fetch_raw_data()
            
        Returns:
            Dictionary with operation results
        """
        operation_name = f"{self.__class__.__name__} data load"
        self.logger.info(f"Starting {operation_name}")
        
        try:
            # Step 1: Fetch raw data
            self.logger.info("Fetching raw data...")
            raw_data = self.fetch_raw_data(**fetch_kwargs)
            
            if raw_data.empty:
                self.logger.warning("No raw data found")
                return {"success": False, "message": "No data found"}
            
            self.logger.info(f"Fetched {len(raw_data)} raw records")
            
            # Step 2: Transform data
            transformed_records = self.transform_data(raw_data)
            
            if not transformed_records:
                self.logger.error("No valid records after transformation")
                return {"success": False, "message": "No valid records after transformation"}
            
            # Step 3: Handle dry run
            if dry_run:
                return self._handle_dry_run(transformed_records, clear_table, include_records)
            
            # Step 4: Clear table if requested
            if clear_table:
                self.logger.info("Clearing existing data...")
                clear_result = self.db_manager.clear_table()
                if not clear_result:
                    return {"success": False, "message": "Failed to clear existing data"}
            
            # Step 5: Load data into database
            return self._load_to_database(raw_data, transformed_records, clear_table)
            
        except Exception as e:
            self.logger.error(f"Error during {operation_name}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _handle_dry_run(self, transformed_records: List[Dict[str, Any]], 
                       clear_table: bool,
                       include_records: bool = False) -> Dict[str, Any]:
        """Handle dry run mode.
        
        Args:
            transformed_records: Transformed records that would be loaded
            clear_table: Whether table would be cleared
            
        Returns:
            Dry run results dictionary
        """
        self.logger.info("DRY RUN - Would process the following operations:")
        self.logger.info(f"- Clear table: {clear_table}")
        self.logger.info(f"- Upsert {len(transformed_records)} records")
        
        # Show sample record
        if transformed_records:
            sample_record = transformed_records[0]
            self.logger.info(f"Sample record: {sample_record}")
        
        result = {
            "success": True,
            "dry_run": True,
            "would_clear": clear_table,
            "would_upsert": len(transformed_records),
            "sample_record": transformed_records[0] if transformed_records else None
        }

        if include_records:
            result["records"] = transformed_records

        return result
    
    def _load_to_database(self, raw_data: pd.DataFrame, 
                         transformed_records: List[Dict[str, Any]],
                         cleared_table: bool) -> Dict[str, Any]:
        """Load transformed records to database.
        
        Args:
            raw_data: Original raw data (for metrics)
            transformed_records: Transformed records to load
            cleared_table: Whether table was cleared
            
        Returns:
            Load operation results
        """
        self.logger.info(f"Loading {len(transformed_records)} records to database...")
        
        # Use upsert for players; teams use simple insert to align with tests
        if self.table_name == "players":
            # For players, use player_id as conflict resolution to implement overwrite strategy
            upsert_result = self.db_manager.upsert_records(transformed_records, on_conflict="player_id")
        elif self.table_name == "teams":
            # Teams are relatively static; use insert as tests expect
            upsert_result = self.db_manager.insert_records(transformed_records)
        else:
            upsert_result = self.db_manager.upsert_records(transformed_records)
        
        if not upsert_result["success"]:
            return {"success": False, "error": upsert_result["error"]}
        
        self.logger.info(f"{self.__class__.__name__} data load completed successfully")
        
        return {
            "success": True,
            "total_fetched": len(raw_data),
            "total_validated": len(transformed_records),
            "upsert_result": upsert_result,
            "cleared_table": cleared_table
        }
    
    def get_record_count(self, conditions: Optional[Dict[str, Any]] = None) -> int:
        """Get count of records in the table.
        
        Args:
            conditions: Optional filter conditions
            
        Returns:
            Number of records matching conditions
        """
        return self.db_manager.get_record_count(conditions)
    
    def clear_table(self) -> bool:
        """Clear all records from the table.
        
        Returns:
            True if successful, False otherwise
        """
        return self.db_manager.clear_table()
