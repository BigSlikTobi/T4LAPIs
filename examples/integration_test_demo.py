#!/usr/bin/env python3
"""
Integration test showing old vs new behavior for story grouping gating.

This script demonstrates how the capability-based approach solves the
problem described in issue #52.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_news_pipeline.orchestrator.pipeline import NFLNewsPipeline
from nfl_news_pipeline.models import DefaultsConfig


class MockAuditLogger:
    def __init__(self):
        self.events = []
        
    def log_event(self, event_type: str, message: str, data: dict = None):
        self.events.append({'type': event_type, 'message': message, 'data': data})
        print(f"[AUDIT] {event_type}: {message}")


class OldStyleStorage:
    """Storage that would fail with old .client check."""
    
    def __init__(self):
        # No .client attribute - would fail old check
        self._database = Mock(name="alternative_db")
        
    def get_grouping_client(self):
        """New method that provides the database client."""
        return self._database
        
    # Required storage interface
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        result = Mock()
        result.inserted_count = len(items)
        result.updated_count = 0
        result.errors_count = 0
        result.ids_by_url = {item.url: f"id_{i}" for i, item in enumerate(items)}
        return result
        
    def get_watermark(self, source_name):
        return None
        
    def update_watermark(self, source_name, **kwargs):
        return True


class TraditionalSupabaseStorage:
    """Traditional storage with .client attribute."""
    
    def __init__(self):
        self.client = Mock(name="supabase_client")
        
    def get_grouping_client(self):
        return self.client
        
    # Required storage interface  
    def check_duplicate_urls(self, urls):
        return {}
        
    def store_news_items(self, items):
        result = Mock()
        result.inserted_count = len(items)
        result.updated_count = 0
        result.errors_count = 0
        result.ids_by_url = {item.url: f"id_{i}" for i, item in enumerate(items)}
        return result
        
    def get_watermark(self, source_name):
        return None
        
    def update_watermark(self, source_name, **kwargs):
        return True


def create_mock_config_manager(enable_grouping=True):
    """Create a mock config manager."""
    cm = Mock()
    defaults = DefaultsConfig()
    defaults.enable_story_grouping = enable_grouping
    cm.get_defaults.return_value = defaults
    return cm


def test_storage_type(name: str, storage, expect_enabled: bool):
    """Test story grouping enablement with a specific storage type."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Expected story grouping enabled: {expect_enabled}")
    print(f"{'='*60}")
    
    # Clear any existing env vars
    for key in ['NEWS_PIPELINE_ENABLE_STORY_GROUPING', 'NEWS_PIPELINE_DISABLE_STORY_GROUPING']:
        if key in os.environ:
            del os.environ[key]
    
    audit_logger = MockAuditLogger()
    
    try:
        pipeline = NFLNewsPipeline(
            config_path="fake_config.yaml",
            storage=storage,
            audit=audit_logger
        )
        pipeline.cm = create_mock_config_manager(enable_grouping=True)
        
        enabled = pipeline._should_run_story_grouping()
        
        print(f"Story grouping enabled: {enabled}")
        
        if enabled == expect_enabled:
            print("✅ PASS: Behavior matches expectation")
        else:
            print("❌ FAIL: Behavior does not match expectation")
            
        # Show audit events
        disable_events = [e for e in audit_logger.events if 'disabled' in e['message']]
        if disable_events:
            print("Audit events:")
            for event in disable_events:
                print(f"  - {event['type']}: {event['message']}")
                if event['data']:
                    print(f"    Data: {event['data']}")
        else:
            print("No disable events logged")
            
        return enabled == expect_enabled
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("Story Grouping Capability-Based Gating Integration Test")
    print("This demonstrates the fix for issue #52")
    
    # Test scenarios
    scenarios = [
        ("Traditional Supabase Storage", TraditionalSupabaseStorage(), True),
        ("Alternative Storage (no .client attr)", OldStyleStorage(), True),
    ]
    
    results = []
    for name, storage, expected in scenarios:
        result = test_storage_type(name, storage, expected)
        results.append((name, result))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The capability-based approach works correctly.")
        print("Both traditional Supabase storage and alternative storage implementations")
        print("are now properly supported for story grouping.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())