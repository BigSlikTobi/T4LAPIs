#!/usr/bin/env python3
"""
Debug script to check environment variable configuration.
Useful for troubleshooting Docker environment issues.
"""

import sys
import os
from pathlib import Path

# Add project root to path (robust to nesting)
def _repo_root() -> str:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / 'src').exists() and (p / 'README.md').exists():
            return str(p)
    return str(start.parents[0])

project_root = _repo_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.db.database_init import SupabaseConnection


def main():
    """Debug environment variables and Supabase connection."""
    print("🔍 Environment Variable Debug Tool")
    print("=" * 50)
    
    # Check environment variables directly
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    ci_mode = os.getenv("CI") == 'true' or os.getenv("GITHUB_ACTIONS") == 'true'
    
    print(f"SUPABASE_URL set: {'✅' if supabase_url else '❌'}")
    if supabase_url:
        print(f"SUPABASE_URL value: {supabase_url[:50]}..." if len(supabase_url) > 50 else f"SUPABASE_URL value: {supabase_url}")
        has_quotes = supabase_url.startswith(('"', "'")) or supabase_url.endswith(('"', "'"))
        print(f"SUPABASE_URL has quotes: {'✅' if has_quotes else '❌'}")
    
    print(f"SUPABASE_KEY set: {'✅' if supabase_key else '❌'}")
    if supabase_key:
        print(f"SUPABASE_KEY length: {len(supabase_key)} characters")
        has_quotes = supabase_key.startswith(('"', "'")) or supabase_key.endswith(('"', "'"))
        print(f"SUPABASE_KEY has quotes: {'✅' if has_quotes else '❌'}")
    
    print(f"CI mode detected: {'✅' if ci_mode else '❌'}")
    
    print("\n" + "=" * 50)
    print("Testing Supabase Connection...")
    
    try:
        # Test the connection
        connection = SupabaseConnection()
        debug_info = connection.debug_env_vars()
        
        print(f"Environment variables loaded by SupabaseConnection:")
        for key, value in debug_info.items():
            print(f"  {key}: {'✅' if value else '❌'}")
        
        if connection.is_connected():
            print("✅ Supabase client initialized successfully!")
        else:
            print("❌ Supabase client failed to initialize")
            
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
    
    print("\n" + "=" * 50)
    print("Debug complete")


if __name__ == "__main__":
    main()
