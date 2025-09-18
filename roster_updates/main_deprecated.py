#!/usr/bin/env python3
"""
DEPRECATED: This script has been replaced by the unified core data loader system.

Use the new script instead:
    python scripts/load_rosters.py --years 2024 --version 1

The new system provides:
- Better error handling and validation
- Consistent CLI interface with other data loaders
- Integration with the core data pipeline
- Comprehensive logging and dry-run capabilities
- Uses nfl_data_py as the canonical data source
- No SSL monkey-patching or custom request handling needed

This script is kept for reference but will be removed in a future version.
"""

import sys
import os

def main():
    print("⚠️  DEPRECATED SCRIPT")
    print()
    print("This roster_updates/main.py script has been replaced by the unified core data loader system.")
    print()
    print("Please use the new script instead:")
    print("    python scripts/load_rosters.py --years 2024 --version 1")
    print()
    print("The new system provides:")
    print("- Better error handling and validation") 
    print("- Consistent CLI interface with other data loaders")
    print("- Integration with the core data pipeline")
    print("- Comprehensive logging and dry-run capabilities")
    print("- Uses nfl_data_py as the canonical data source")
    print("- No SSL monkey-patching or custom request handling needed")
    print()
    print("For help with the new script:")
    print("    python scripts/load_rosters.py --help")
    print()
    return 1

if __name__ == "__main__":
    sys.exit(main())