#!/usr/bin/env python3
"""
Script to explore all available tables and columns in nfl_data_py library.
This will help understand the complete data structure available.
"""

import sys
import os
import logging
import pandas as pd
import traceback

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import nfl_data_py as nfl
except ImportError:
    print("Error: nfl_data_py not installed. Please install it first.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def explore_function(func_name, func_obj, test_years=None):
    """
    Explore a single nfl_data_py function to understand its structure.
    
    Args:
        func_name: Name of the function
        func_obj: The function object
        test_years: Years to test with (for functions that require years)
    
    Returns:
        dict: Information about the function and its data structure
    """
    logger.info(f"Exploring function: {func_name}")
    
    result = {
        'function_name': func_name,
        'status': 'unknown',
        'columns': [],
        'sample_data_shape': None,
        'error': None,
        'docstring': getattr(func_obj, '__doc__', 'No docstring available')
    }
    
    try:
        # Try to call the function with different parameter combinations
        df = None
        
        if func_name in ['import_team_desc', 'import_combine_data', 'import_draft_picks']:
            # Functions that don't require parameters or can be called without
            try:
                df = func_obj()
            except Exception as e1:
                try:
                    # Some functions might accept years as optional parameter
                    df = func_obj(test_years)
                except Exception as e2:
                    result['error'] = f"No params: {str(e1)}, With years: {str(e2)}"
        
        elif 'import_' in func_name:
            # Most import functions require years parameter
            if test_years:
                try:
                    df = func_obj(test_years)
                except Exception as e1:
                    try:
                        # Try with a single year
                        df = func_obj([test_years[0]])
                    except Exception as e2:
                        try:
                            # Try with different parameters for special functions
                            if func_name == 'import_ngs_data':
                                df = func_obj('passing', test_years)
                            elif func_name == 'import_pbp_data':
                                df = func_obj(test_years, downsampling=True)
                            else:
                                result['error'] = f"Years param: {str(e1)}, Single year: {str(e2)}"
                        except Exception as e3:
                            result['error'] = f"Years: {str(e1)}, Single: {str(e2)}, Special: {str(e3)}"
        
        if df is not None and isinstance(df, pd.DataFrame):
            result['status'] = 'success'
            result['columns'] = list(df.columns)
            result['sample_data_shape'] = df.shape
            logger.info(f"✅ {func_name}: {df.shape} - {len(df.columns)} columns")
        else:
            result['status'] = 'failed'
            if not result['error']:
                result['error'] = "No DataFrame returned"
            logger.warning(f"❌ {func_name}: {result['error']}")
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"💥 {func_name}: {str(e)}")
    
    return result


def main():
    """Main function to explore all nfl_data_py functions."""
    logger.info("Starting nfl_data_py exploration...")
    
    # Get all functions that start with 'import_'
    import_functions = [name for name in dir(nfl) if name.startswith('import_')]
    
    logger.info(f"Found {len(import_functions)} import functions")
    
    # Test years - use recent years that should have data
    test_years = [2023, 2024]
    
    results = []
    
    for func_name in sorted(import_functions):
        func_obj = getattr(nfl, func_name)
        if callable(func_obj):
            result = explore_function(func_name, func_obj, test_years)
            results.append(result)
    
    # Generate summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['failed', 'error']]
    
    logger.info("=" * 80)
    logger.info(f"EXPLORATION COMPLETE")
    logger.info(f"Total functions: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info("=" * 80)
    
    # Print successful functions summary
    print("\n" + "="*80)
    print("SUCCESSFUL FUNCTIONS AND THEIR COLUMNS")
    print("="*80)
    
    for result in successful:
        print(f"\n📊 {result['function_name']}")
        print(f"   Shape: {result['sample_data_shape']}")
        print(f"   Columns ({len(result['columns'])}): {', '.join(result['columns'][:10])}{'...' if len(result['columns']) > 10 else ''}")
    
    # Print failed functions
    if failed:
        print("\n" + "="*80)
        print("FAILED FUNCTIONS")
        print("="*80)
        
        for result in failed:
            print(f"\n❌ {result['function_name']}")
            print(f"   Error: {result['error']}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        
        # Save results for further processing
        print(f"\nExploration completed. Found {len(results)} functions.")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        traceback.print_exc()
        sys.exit(1)
