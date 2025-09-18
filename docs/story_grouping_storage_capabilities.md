# Story Grouping Storage Capability Requirements

## Overview

As of the fix for [issue #52](https://github.com/BigSlikTobi/T4LAPIs/issues/52), story grouping enablement uses a capability-based detection system instead of relying solely on the presence of a `.client` attribute.

## Storage Requirements

For story grouping to be enabled, storage implementations must provide a database client that can be used by `GroupStorageManager`. The system detects this capability through multiple methods:

### Method 1: Protocol Implementation (Recommended)

Implement the `StoryGroupingCapable` protocol:

```python
from nfl_news_pipeline.storage.protocols import StoryGroupingCapable

class MyStorage(StoryGroupingCapable):
    def __init__(self, database_client):
        self._db = database_client
    
    def get_grouping_client(self):
        """Return database client for grouping operations."""
        return self._db
```

### Method 2: Legacy Client Attribute (Backward Compatible)

Provide a `.client` attribute:

```python
class MyStorage:
    def __init__(self, supabase_client):
        self.client = supabase_client  # Will be detected automatically
```

### Method 3: Duck Typing

Provide a `get_grouping_client()` method:

```python
class MyStorage:
    def __init__(self, database_client):
        self._db = database_client
    
    def get_grouping_client(self):
        return self._db
```

## Capability Detection

The system uses `has_story_grouping_capability(storage)` to determine if a storage implementation supports story grouping. This function checks the methods above in order.

## Error Handling

When story grouping is disabled due to missing capabilities:

1. The orchestrator logs an audit event with type `story_grouping_disabled`
2. The event includes the storage type for debugging
3. Clear error messages explain what's missing

## Environment Variable Control

Environment variables continue to control story grouping:

- `NEWS_PIPELINE_DISABLE_STORY_GROUPING=1` - Forces story grouping off
- `NEWS_PIPELINE_ENABLE_STORY_GROUPING=1` - Forces story grouping on (if storage capable)

## Migration Guide

### Existing Supabase Storage

No changes required. Existing storage with `.client` attribute continues to work.

### Custom Storage Implementations

If you have custom storage implementations that should support story grouping:

1. **Recommended**: Implement `StoryGroupingCapable` protocol
2. **Alternative**: Add `get_grouping_client()` method
3. **Legacy**: Ensure `.client` attribute exists and is not None

### Testing

The system is thoroughly tested with various storage types. See `tests/test_story_grouping_capability_gating.py` for examples.

## Benefits

- **Flexible**: Supports multiple storage backend types
- **Clear**: Better error messages when capabilities are missing  
- **Compatible**: Maintains backward compatibility with existing code
- **Testable**: Easy to mock and test different capability scenarios