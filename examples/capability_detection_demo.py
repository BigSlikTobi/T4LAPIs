#!/usr/bin/env python3
"""
Demonstration script showing the improved story grouping capability detection.

This script demonstrates how the new capability-based gating logic works
with different storage implementations, replacing the old .client check.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_news_pipeline.storage.protocols import (
    StoryGroupingCapable, has_story_grouping_capability, get_grouping_client
)


class LegacySupabaseStorage:
    """Traditional storage with .client attribute (before this fix)."""
    
    def __init__(self, client=None):
        if client is None:
            self.client = Mock(name="supabase_client")
        else:
            self.client = client
        

class ModernCapableStorage(StoryGroupingCapable):
    """Modern storage implementing the StoryGroupingCapable protocol."""
    
    def __init__(self, client=None):
        self._client = client or Mock(name="database_client")
        
    def get_grouping_client(self):
        return self._client


class DuckTypedStorage:
    """Storage using duck typing - has method but doesn't inherit protocol."""
    
    def __init__(self, client=None):
        self._client = client or Mock(name="database_client")
        
    def get_grouping_client(self):
        return self._client


class IncapableStorage:
    """Storage without any grouping capabilities."""
    
    def __init__(self):
        pass


def main():
    print("Story Grouping Capability Detection Demo")
    print("=" * 50)
    
    # Test different storage types
    storages = [
        ("Legacy Supabase Storage", LegacySupabaseStorage()),
        ("Modern Protocol Storage", ModernCapableStorage()),
        ("Duck-Typed Storage", DuckTypedStorage()),
        ("Incapable Storage", IncapableStorage()),
        ("Legacy Storage (no client)", type('NoClientStorage', (), {'client': None})()),
    ]
    
    for name, storage in storages:
        print(f"\n{name}:")
        print(f"  Has capability: {has_story_grouping_capability(storage)}")
        
        try:
            client = get_grouping_client(storage)
            print(f"  Client available: Yes ({type(client).__name__})")
        except ValueError as e:
            print(f"  Client available: No ({e})")
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- Legacy storages with .client work (backward compatibility)")
    print("- Modern storages implementing protocol work (preferred)")
    print("- Duck-typed storages with get_grouping_client() work")
    print("- Storages without capabilities are properly detected and rejected")
    print("- Clear error messages help with debugging")


if __name__ == "__main__":
    main()