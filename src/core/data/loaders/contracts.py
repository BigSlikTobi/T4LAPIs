"""
Contracts data loader for NFL financial analytics.
This module provides functionality to fetch NFL contract data and load it into the database.
"""

import pandas as pd
from typing import Type, List

from .base import BaseDataLoader
from ..transform import BaseDataTransformer
import nfl_data_py as nfl


class ContractsDataLoader(BaseDataLoader):
    """Loads NFL contracts data into the database.
    
    Contracts data provides contract values, guarantees, and salary cap information.
    """
    
    def __init__(self):
        """Initialize the contracts data loader."""
        super().__init__("contracts")
    
    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        """Return the ContractsDataTransformer class."""
        return ContractsDataTransformer
    
    def fetch_raw_data(self) -> pd.DataFrame:
        """Fetch raw contracts data from nfl_data_py.
        
        Returns:
            Raw contracts data DataFrame
        """
        self.logger.info("Fetching contracts data")
        try:
            contracts_df = nfl.import_contracts()
            self.logger.info(f"Successfully fetched {len(contracts_df)} contracts records")
            return contracts_df
        except Exception as e:
            self.logger.error(f"Failed to fetch contracts data: {e}")
            raise
    
    def load_contracts_data(self, dry_run: bool = False, clear_table: bool = False):
        """Legacy method for backward compatibility.
        
        Args:
            dry_run: If True, don't actually insert/update data
            clear_table: If True, clear existing data before loading
            
        Returns:
            Dictionary with operation results
        """
        return self.load_data(dry_run=dry_run, clear_table=clear_table)


class ContractsDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of contracts data.
    
    Transforms raw contracts data from nfl_data_py into the format expected
    by our contracts database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for contracts data."""
        return [
            'player_name', 'team', 'year_signed', 'years', 'value'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> dict:
        """Transform a single contracts record.
        
        Args:
            row: Single row from raw contracts DataFrame
            
        Returns:
            Transformed record ready for database insertion
        """
        # Essential identifiers
        record = {
            'player_name': str(row.get('player_name', '')) if pd.notna(row.get('player_name')) else None,
            'team': str(row.get('team', '')) if pd.notna(row.get('team')) else None,
            'year_signed': int(row.get('year_signed', 0)) if pd.notna(row.get('year_signed')) else None,
            'years': int(row.get('years', 0)) if pd.notna(row.get('years')) else None,
        }
        
        # Financial details
        record.update({
            'value': float(row.get('value', 0)) if pd.notna(row.get('value')) else None,
            'guaranteed': float(row.get('guaranteed', 0)) if pd.notna(row.get('guaranteed')) else None,
            'apy': float(row.get('apy', 0)) if pd.notna(row.get('apy')) else None,
            'inflated_value': float(row.get('inflated_value', 0)) if pd.notna(row.get('inflated_value')) else None,
            'inflated_guaranteed': float(row.get('inflated_guaranteed', 0)) if pd.notna(row.get('inflated_guaranteed')) else None,
            'inflated_apy': float(row.get('inflated_apy', 0)) if pd.notna(row.get('inflated_apy')) else None,
        })
        
        # Contract details
        record.update({
            'position': str(row.get('position', '')) if pd.notna(row.get('position')) else None,
            'age_when_signed': int(row.get('age_when_signed', 0)) if pd.notna(row.get('age_when_signed')) else None,
            'is_extension': bool(row.get('is_extension', False)) if pd.notna(row.get('is_extension')) else None,
            'is_restructure': bool(row.get('is_restructure', False)) if pd.notna(row.get('is_restructure')) else None,
        })
        
        # Salary cap information
        record.update({
            'percent_guaranteed': float(row.get('percent_guaranteed', 0)) if pd.notna(row.get('percent_guaranteed')) else None,
            'percent_cap_apy': float(row.get('percent_cap_apy', 0)) if pd.notna(row.get('percent_cap_apy')) else None,
            'percent_cap_total': float(row.get('percent_cap_total', 0)) if pd.notna(row.get('percent_cap_total')) else None,
        })
        
        # Additional details
        record.update({
            'dollars_per_year': float(row.get('dollars_per_year', 0)) if pd.notna(row.get('dollars_per_year')) else None,
            'contract_notes': str(row.get('contract_notes', '')) if pd.notna(row.get('contract_notes')) else None,
        })
        
        return record
    
    def _validate_record(self, record: dict) -> bool:
        """Validate a transformed contracts record.
        
        Args:
            record: Transformed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Must have essential identifiers
        if not record.get('player_name') or not record.get('team'):
            return False
        
        # Must have valid year signed
        year_signed = record.get('year_signed')
        if not year_signed or year_signed < 1990 or year_signed > 2030:
            return False
        
        # Contract length should be reasonable
        years = record.get('years')
        if years and (years < 1 or years > 15):
            return False
        
        # Contract value should be positive if present
        value = record.get('value')
        if value and value < 0:
            return False
        
        return True