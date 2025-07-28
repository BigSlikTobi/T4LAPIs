"""Database utility functions and classes."""

import logging
from typing import Optional, Dict, Any, List
from src.core.db.database_init import get_supabase_client

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Centralized database operations manager."""
    
    def __init__(self, table_name: str):
        """Initialize database manager for a specific table.
        
        Args:
            table_name: Name of the database table to manage
            
        Raises:
            RuntimeError: If Supabase client cannot be initialized
        """
        self.table_name = table_name
        self.supabase = get_supabase_client()
        
        if not self.supabase:
            raise RuntimeError("Could not initialize Supabase client")
        
        self.logger = logging.getLogger(f"{__name__}.{table_name}")
    
    def clear_table(self) -> bool:
        """Clear all records from the table.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Clearing all records from {self.table_name} table")
            
            # Use a condition that matches all records
            response = self.supabase.table(self.table_name).delete().neq('id', '').execute()
            
            if hasattr(response, 'error') and response.error:
                self.logger.error(f"Error clearing table: {response.error}")
                return False
                
            self.logger.info(f"Successfully cleared {self.table_name} table")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear {self.table_name} table: {e}")
            return False
    
    def get_record_count(self, conditions: Optional[Dict[str, Any]] = None) -> int:
        """Get count of records in the table.
        
        Args:
            conditions: Optional filter conditions
            
        Returns:
            Number of records
        """
        try:
            query = self.supabase.table(self.table_name).select("*", count="exact")
            
            # Apply conditions if provided
            if conditions:
                for column, value in conditions.items():
                    query = query.eq(column, value)
            
            response = query.execute()
            
            if hasattr(response, 'error') and response.error:
                self.logger.error(f"Error getting record count: {response.error}")
                return 0
                
            return response.count if hasattr(response, 'count') else 0
            
        except Exception as e:
            self.logger.error(f"Failed to get record count for {self.table_name}: {e}")
            return 0
    
    def insert_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert records into the table.
        
        Args:
            records: List of record dictionaries to insert
            
        Returns:
            Dictionary with operation results
        """
        try:
            self.logger.info(f"Inserting {len(records)} records into {self.table_name}")
            
            response = self.supabase.table(self.table_name).insert(records).execute()
            
            if hasattr(response, 'error') and response.error:
                self.logger.error(f"Error inserting records: {response.error}")
                return {"success": False, "error": str(response.error)}
            
            affected_rows = len(response.data) if hasattr(response, 'data') and response.data else 0
            self.logger.info(f"Successfully inserted {affected_rows} records")
            
            return {
                "success": True,
                "affected_rows": affected_rows,
                "data": response.data if hasattr(response, 'data') else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to insert records into {self.table_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def upsert_records(self, records: List[Dict[str, Any]], 
                      on_conflict: Optional[str] = None) -> Dict[str, Any]:
        """Upsert (insert or update) records into the table.
        
        Args:
            records: List of record dictionaries to upsert
            on_conflict: Column name to use for conflict resolution
            
        Returns:
            Dictionary with operation results
        """
        try:
            self.logger.info(f"Upserting {len(records)} records into {self.table_name}")
            
            # Build upsert parameters
            upsert_params = {'json': records}
            if on_conflict:
                upsert_params['on_conflict'] = on_conflict
                self.logger.info(f"Using conflict resolution on column: {on_conflict}")
            
            response = self.supabase.table(self.table_name).upsert(**upsert_params).execute()
            
            if hasattr(response, 'error') and response.error:
                self.logger.error(f"Error upserting records: {response.error}")
                return {"success": False, "error": str(response.error)}
            
            affected_rows = len(response.data) if hasattr(response, 'data') and response.data else 0
            self.logger.info(f"Successfully upserted {affected_rows} records")
            
            return {
                "success": True,
                "affected_rows": affected_rows,
                "data": response.data if hasattr(response, 'data') else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to upsert records into {self.table_name}: {e}")
            return {"success": False, "error": str(e)}
