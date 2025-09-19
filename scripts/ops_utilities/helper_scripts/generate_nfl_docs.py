#!/usr/bin/env python3
"""
Detailed exploration script to create a comprehensive markdown file 
documenting all NFL data tables and columns.
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import json
from datetime import datetime

# Add the project root to the Python path
def _repo_root() -> str:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / 'src').exists() and (p / 'README.md').exists():
            return str(p)
    return str(start.parents[0])

sys.path.insert(0, _repo_root())

try:
    import nfl_data_py as nfl
except ImportError:
    print("Error: nfl_data_py not installed. Please install it first.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce logging noise
logger = logging.getLogger(__name__)


def get_detailed_function_info(func_name, func_obj, test_years=[2023]):
    """Get detailed information about a function including full column list and sample data."""
    
    result = {
        'function_name': func_name,
        'status': 'unknown',
        'columns': [],
        'column_types': {},
        'sample_data_shape': None,
        'error': None,
        'docstring': getattr(func_obj, '__doc__', 'No docstring available').strip() if getattr(func_obj, '__doc__', None) else 'No docstring available',
        'sample_data': {}
    }
    
    try:
        df = None
        
        # Handle different function signatures
        if func_name == 'import_team_desc':
            df = func_obj()
        elif func_name == 'import_combine_data':
            df = func_obj()
        elif func_name == 'import_draft_picks':
            df = func_obj()
        elif func_name == 'import_contracts':
            df = func_obj()  # Takes no arguments
        elif func_name == 'import_players':
            df = func_obj()  # Takes no arguments
        elif func_name == 'import_ids':
            df = func_obj()  # Takes no arguments
        elif func_name == 'import_ngs_data':
            # Try different stat types
            for stat_type in ['passing', 'rushing', 'receiving']:
                try:
                    df = func_obj(stat_type, test_years)
                    break
                except:
                    continue
        elif func_name in ['import_seasonal_pfr', 'import_weekly_pfr']:
            # Try different stat types
            for s_type in ['pass', 'rec', 'rush']:
                try:
                    df = func_obj(s_type, test_years)
                    break
                except:
                    continue
        else:
            # Most functions take years parameter
            try:
                df = func_obj(test_years)
            except:
                # Try with single year
                df = func_obj(test_years[0])
        
        if df is not None and isinstance(df, pd.DataFrame):
            result['status'] = 'success'
            result['columns'] = list(df.columns)
            result['sample_data_shape'] = df.shape
            
            # Get column types
            result['column_types'] = {col: str(df[col].dtype) for col in df.columns}
            
            # Get sample data (first row as dict, but clean it up)
            if len(df) > 0:
                sample_row = df.iloc[0].to_dict()
                # Convert any non-serializable types to strings
                result['sample_data'] = {k: str(v) if pd.isna(v) else v for k, v in sample_row.items()}
            
            print(f"✅ {func_name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
        else:
            result['status'] = 'failed'
            result['error'] = "No DataFrame returned"
            print(f"❌ {func_name}: No DataFrame returned")
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"💥 {func_name}: {str(e)}")
    
    return result


def generate_markdown_documentation(results):
    """Generate comprehensive markdown documentation."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# NFL Data Python Library - Complete Table and Column Reference

> **Generated on:** {timestamp}  
> **Library:** nfl_data_py  
> **Total Functions Explored:** {len(results)}

This document provides a comprehensive overview of all available data tables and their columns in the `nfl_data_py` library.

## Summary

| Status | Count | Functions |
|--------|-------|-----------|
| ✅ Successful | {len([r for r in results if r['status'] == 'success'])} | {', '.join([r['function_name'] for r in results if r['status'] == 'success'])[:100]}... |
| ❌ Failed | {len([r for r in results if r['status'] in ['failed', 'error']])} | {', '.join([r['function_name'] for r in results if r['status'] in ['failed', 'error']])} |

## Available Data Tables

"""
    
    # Group by status
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['failed', 'error']]
    
    # Document successful functions
    for i, result in enumerate(sorted(successful, key=lambda x: x['function_name']), 1):
        func_name = result['function_name']
        shape = result['sample_data_shape']
        columns = result['columns']
        
        md_content += f"""### {i}. {func_name}

**Data Shape:** {shape[0]:,} rows × {shape[1]} columns  
**Function Call:** `nfl.{func_name}()`

**Description:**  
{result['docstring'][:200]}{'...' if len(result['docstring']) > 200 else ''}

**Columns ({len(columns)}):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
"""
        
        # Add column details
        for j, col in enumerate(columns, 1):
            col_type = result['column_types'].get(col, 'unknown')
            sample_val = result['sample_data'].get(col, 'N/A')
            # Truncate long sample values
            if isinstance(sample_val, str) and len(str(sample_val)) > 50:
                sample_val = str(sample_val)[:47] + "..."
            elif sample_val is None or sample_val == 'nan':
                sample_val = 'NULL'
            
            md_content += f"| {j} | `{col}` | {col_type} | {sample_val} |\n"
        
        md_content += "\n---\n\n"
    
    # Document failed functions
    if failed:
        md_content += "## Functions Requiring Special Parameters\n\n"
        md_content += "The following functions require specific parameters or have special usage patterns:\n\n"
        
        for result in sorted(failed, key=lambda x: x['function_name']):
            func_name = result['function_name']
            error = result['error']
            docstring = result['docstring']
            
            md_content += f"""### {func_name}

**Status:** ❌ Requires Special Parameters  
**Error:** {error}  
**Function Call:** `nfl.{func_name}(...)`

**Description:**  
{docstring[:300]}{'...' if len(docstring) > 300 else ''}

**Usage Notes:**  
"""
            
            # Add specific usage notes for known functions
            if func_name == 'import_ngs_data':
                md_content += "- Requires `stat_type` parameter: 'passing', 'rushing', or 'receiving'\n"
                md_content += "- Example: `nfl.import_ngs_data('passing', [2023])`\n"
            elif func_name in ['import_seasonal_pfr', 'import_weekly_pfr']:
                md_content += "- Requires `s_type` parameter: 'pass', 'rec', or 'rush'\n"
                md_content += f"- Example: `nfl.{func_name}('pass', [2023])`\n"
            elif func_name in ['import_contracts', 'import_players']:
                md_content += "- Takes no parameters\n"
                md_content += f"- Example: `nfl.{func_name}()`\n"
            elif func_name == 'import_ids':
                md_content += "- May require specific parameter format\n"
                md_content += f"- Check function documentation: `help(nfl.{func_name})`\n"
            
            md_content += "\n---\n\n"
    
    # Add usage examples
    md_content += """## Usage Examples

### Basic Data Fetching
```python
import nfl_data_py as nfl
import pandas as pd

# Fetch team information
teams = nfl.import_team_desc()

# Fetch player rosters for specific years
rosters = nfl.import_seasonal_rosters([2023, 2024])

# Fetch game schedules
schedules = nfl.import_schedules([2023])

# Fetch weekly player stats
weekly_stats = nfl.import_weekly_data([2023])
```

### Advanced Data Fetching
```python
# Next Gen Stats (requires stat type)
passing_ngs = nfl.import_ngs_data('passing', [2023])
rushing_ngs = nfl.import_ngs_data('rushing', [2023])

# Play-by-play data
pbp = nfl.import_pbp_data([2023])

# Pro Football Reference data (requires stat type)
seasonal_passing = nfl.import_seasonal_pfr('pass', [2023])
weekly_receiving = nfl.import_weekly_pfr('rec', [2023])
```

## Integration with Your Project

These functions can be easily integrated into your existing data pipeline:

```python
from src.core.data.fetch import (
    fetch_team_data,
    fetch_player_data,
    fetch_game_schedule_data,
    fetch_weekly_stats_data
)

# Your existing fetch functions already wrap these:
teams = fetch_team_data()  # -> nfl.import_team_desc()
players = fetch_player_data([2023])  # -> nfl.import_seasonal_rosters([2023])
games = fetch_game_schedule_data([2023])  # -> nfl.import_schedules([2023])
stats = fetch_weekly_stats_data([2023])  # -> nfl.import_weekly_data([2023])
```

---

*This documentation was automatically generated by exploring the nfl_data_py library.*
"""
    
    return md_content


def main():
    """Main function to generate comprehensive documentation."""
    print("🔍 Exploring nfl_data_py library...")
    
    # Get all import functions
    import_functions = [name for name in dir(nfl) if name.startswith('import_')]
    print(f"Found {len(import_functions)} import functions\n")
    
    results = []
    
    for func_name in sorted(import_functions):
        func_obj = getattr(nfl, func_name)
        if callable(func_obj):
            result = get_detailed_function_info(func_name, func_obj)
            results.append(result)
    
    # Generate markdown documentation
    print(f"\n📝 Generating documentation...")
    md_content = generate_markdown_documentation(results)
    
    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), '..', 'docs', 'nfl_data_py_reference.md')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✅ Documentation saved to: {output_file}")
    print(f"📊 Summary: {len([r for r in results if r['status'] == 'success'])} successful, {len([r for r in results if r['status'] != 'success'])} failed")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
