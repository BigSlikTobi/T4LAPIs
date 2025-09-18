"""Storage capability protocols for story grouping functionality."""

from __future__ import annotations

from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class StoryGroupingCapable(Protocol):
    """Protocol defining storage capabilities required for story grouping.
    
    This protocol defines the minimum interface that a storage implementation
    must provide to support story grouping functionality. It focuses on the
    ability to provide a database client rather than the direct implementation
    of grouping methods, as those are handled by specialized managers.
    """
    
    def get_grouping_client(self) -> Any:
        """Get a database client suitable for story grouping operations.
        
        Returns:
            A database client (typically Supabase client) that can be used
            to initialize GroupStorageManager and related components.
            
        Raises:
            NotImplementedError: If storage doesn't support story grouping
            Exception: If client creation fails
        """
        ...


@runtime_checkable
class SupabaseStorageCapable(Protocol):
    """Protocol for storage that provides Supabase client access.
    
    This protocol represents storage implementations that have a Supabase
    client available, which is the most common case for story grouping.
    """
    
    client: Any  # Supabase client
    
    def get_grouping_client(self) -> Any:
        """Get the Supabase client for grouping operations."""
        ...


def has_story_grouping_capability(storage: Any) -> bool:
    """Check if storage supports story grouping capabilities.
    
    This function implements capability detection for story grouping support.
    It checks for various ways a storage implementation might provide the
    required database client for grouping operations.
    
    Args:
        storage: Storage instance to check
        
    Returns:
        True if storage supports story grouping, False otherwise
    """
    # Method 1: Explicit protocol implementation
    if isinstance(storage, StoryGroupingCapable):
        try:
            client = storage.get_grouping_client()
            return client is not None
        except (NotImplementedError, AttributeError):
            pass
    
    # Method 2: Legacy Supabase client attribute (for backward compatibility)
    if hasattr(storage, 'client') and storage.client is not None:
        return True
    
    # Method 3: Direct grouping client method (duck typing)
    if hasattr(storage, 'get_grouping_client'):
        try:
            client = storage.get_grouping_client()
            return client is not None
        except (NotImplementedError, AttributeError, Exception):
            pass
    
    return False


def get_grouping_client(storage: Any) -> Any:
    """Extract the grouping client from a storage implementation.
    
    Args:
        storage: Storage instance to extract client from
        
    Returns:
        Database client suitable for grouping operations
        
    Raises:
        ValueError: If storage doesn't support grouping or client is unavailable
    """
    # Method 1: Explicit protocol implementation
    if isinstance(storage, StoryGroupingCapable):
        try:
            return storage.get_grouping_client()
        except NotImplementedError:
            pass
    
    # Method 2: Legacy Supabase client attribute
    if hasattr(storage, 'client') and storage.client is not None:
        return storage.client
    
    # Method 3: Direct grouping client method
    if hasattr(storage, 'get_grouping_client'):
        try:
            return storage.get_grouping_client()
        except (NotImplementedError, AttributeError):
            pass
    
    raise ValueError(
        "Storage does not support story grouping capabilities. "
        "Required: either 'client' attribute or 'get_grouping_client()' method."
    )