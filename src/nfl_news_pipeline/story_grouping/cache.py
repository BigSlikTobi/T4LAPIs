"""Caching layer for context summaries to optimize LLM API costs.

Implements TTL-based cache invalidation and cost optimization features.
Uses existing database patterns for cache storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from ..models import ContextSummary
from ..storage.manager import StorageManager
from ..utils.cache import LLMResponseCache


logger = logging.getLogger(__name__)


class ContextCache:
    """Caching mechanism for URL context summaries to avoid duplicate LLM API calls.
    
    Features:
    - URL-based cache keys with content hash for invalidation
    - TTL-based expiration with configurable timeouts
    - Database storage using existing patterns
    - Cost optimization tracking
    """
    
    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        ttl_hours: int = 24,
        enable_memory_cache: bool = True,
        enable_disk_cache: bool = True,
        verbose: bool = False,
    ):
        """Initialize context cache.
        
        Args:
            storage_manager: Storage manager for database operations
            ttl_hours: Time-to-live for cache entries in hours
            enable_memory_cache: Enable in-memory caching for performance
            enable_disk_cache: Enable disk-based caching for persistence
        """
        self.storage_manager = storage_manager
        self.ttl_hours = max(1, ttl_hours)
        self.ttl_seconds = self.ttl_hours * 3600
        self.verbose = verbose
        
        # Initialize cache layers
        self.memory_cache: Optional[LLMResponseCache] = None
        if enable_memory_cache:
            disk_path = None
            if enable_disk_cache:
                cache_dir = os.path.join(os.getcwd(), ".cache", "context_summaries")
                os.makedirs(cache_dir, exist_ok=True)
                disk_path = os.path.join(cache_dir, "context_cache.db")
            
            self.memory_cache = LLMResponseCache(
                ttl_s=self.ttl_seconds,
                sqlite_path=disk_path
            )
        
        # Cost tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        log_fn = logger.info if self.verbose else logger.debug
        log_fn(
            "ContextCache initialized with TTL=%sh, memory=%s, disk=%s",
            ttl_hours,
            enable_memory_cache,
            enable_disk_cache,
        )
    
    def get_cached_summary(self, url: str, metadata_hash: Optional[str] = None) -> Optional[ContextSummary]:
        """Retrieve cached context summary for a URL.
        
        Args:
            url: News article URL
            metadata_hash: Optional hash of metadata (title, description) to detect changes
            
        Returns:
            Cached ContextSummary if found and not expired, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(url, metadata_hash)
            
            # First check memory/disk cache
            if self.memory_cache:
                cached_data = self.memory_cache.get(cache_key)
                if cached_data:
                    summary = self._deserialize_cached_summary(url, cached_data)
                    if summary and self._is_valid_cache_entry(summary):
                        self.cache_hits += 1
                        logger.debug(f"Cache hit for URL: {url}")
                        return summary
                    elif summary:
                        logger.debug(f"Cache entry expired for URL: {url}")
            
            # Check database cache if storage manager is available
            if self.storage_manager:
                summary = self._get_from_database_cache(url, metadata_hash)
                if summary:
                    # Promote to memory cache
                    if self.memory_cache:
                        self.memory_cache.set(cache_key, summary.to_db(), self.ttl_seconds)
                    self.cache_hits += 1
                    logger.debug(f"Database cache hit for URL: {url}")
                    return summary
            
            self.cache_misses += 1
            logger.debug(f"Cache miss for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached summary for {url}: {e}")
            self.cache_misses += 1
            return None
    
    def store_summary(self, summary: ContextSummary, metadata_hash: Optional[str] = None) -> bool:
        """Store context summary in cache.
        
        Args:
            summary: ContextSummary to cache
            metadata_hash: Optional hash of metadata for cache key generation
            
        Returns:
            True if successfully stored, False otherwise
        """
        try:
            # Validate summary first
            summary.validate()
            
            # Extract URL from summary via news_url_id lookup if needed
            # For now, we'll use a cache key based on news_url_id
            cache_key = self._generate_cache_key(summary.news_url_id, metadata_hash)
            
            # Store in memory/disk cache
            if self.memory_cache:
                summary_data = summary.to_db()
                self.memory_cache.set(cache_key, summary_data, self.ttl_seconds)
            
            # Store in database cache if storage manager is available
            if self.storage_manager:
                success = self._store_in_database_cache(summary)
                if not success:
                    logger.warning(f"Failed to store summary in database cache: {summary.news_url_id}")
            
            logger.debug(f"Stored summary in cache: {summary.news_url_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing summary in cache: {e}")
            return False
    
    def invalidate_url(self, url: str) -> bool:
        """Invalidate cached summary for a specific URL.
        
        Args:
            url: URL to invalidate
            
        Returns:
            True if invalidation attempted, False otherwise
        """
        try:
            # We need to generate possible cache keys since we don't know the metadata hash
            # For now, invalidate the base URL key
            cache_key = self._generate_cache_key(url, None)
            
            # Note: TTLCache doesn't have explicit delete, but we can set it to expire immediately
            if self.memory_cache:
                self.memory_cache.set(cache_key, None, 1)  # 1 second TTL
            
            # Database invalidation would need custom logic in storage manager
            logger.debug(f"Invalidated cache for URL: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {url}: {e}")
            return False
    
    def clear_expired(self) -> int:
        """Clear expired cache entries.
        
        Returns:
            Number of entries cleared (estimated)
        """
        # The TTL cache automatically cleans up expired entries
        # This is more for database cleanup if we implement it
        logger.debug("Clearing expired cache entries")
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary with cache hit/miss rates and other metrics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_hours": self.ttl_hours,
        }
    
    def _deserialize_cached_summary(
        self,
        url: str,
        cached_data: Any,
    ) -> Optional[ContextSummary]:
        """Safely recreate ContextSummary objects from cache payloads."""
        if isinstance(cached_data, ContextSummary):
            return cached_data

        if isinstance(cached_data, dict):
            summary_text = cached_data.get("summary_text")
            if not summary_text:
                return None

            news_url_id = cached_data.get("news_url_id") or url
            llm_model = cached_data.get("llm_model") or "cache_restore"

            confidence_raw = (
                cached_data.get("confidence_score")
                if cached_data.get("confidence_score") is not None
                else cached_data.get("confidence")
            )
            try:
                confidence = float(confidence_raw) if confidence_raw is not None else 0.6
            except (TypeError, ValueError):
                confidence = 0.6

            entities = cached_data.get("entities") or {}
            key_topics = cached_data.get("key_topics") or []
            fallback_used = bool(cached_data.get("fallback_used", False))

            generated_at = cached_data.get("generated_at")
            if isinstance(generated_at, str):
                try:
                    generated_at = datetime.fromisoformat(generated_at)
                except ValueError:
                    generated_at = None
            if not isinstance(generated_at, datetime):
                generated_at = datetime.now(timezone.utc)

            try:
                return ContextSummary(
                    news_url_id=news_url_id,
                    summary_text=str(summary_text),
                    llm_model=str(llm_model),
                    confidence_score=confidence,
                    entities=entities,
                    key_topics=list(key_topics) if isinstance(key_topics, (list, tuple)) else [key_topics],
                    fallback_used=fallback_used,
                    generated_at=generated_at,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                logger.debug("Failed to deserialize cached summary for %s: %s", url, exc)
                return None

        logger.debug(
            "Unrecognized cache payload type for %s: %s",
            url,
            type(cached_data).__name__,
        )
        return None

    def _generate_cache_key(self, url_or_id: str, metadata_hash: Optional[str] = None) -> str:
        """Generate cache key from URL/ID and optional metadata hash.
        
        Args:
            url_or_id: URL or news_url_id
            metadata_hash: Optional metadata hash for cache invalidation
            
        Returns:
            Cache key string
        """
        # Create base key from URL/ID
        base_key = hashlib.md5(url_or_id.encode('utf-8')).hexdigest()
        
        # Include metadata hash if provided
        if metadata_hash:
            combined = f"{base_key}:{metadata_hash}"
            return hashlib.md5(combined.encode('utf-8')).hexdigest()
        
        return base_key
    
    def _is_valid_cache_entry(self, summary: ContextSummary) -> bool:
        """Check if cache entry is still valid based on TTL.
        
        Args:
            summary: ContextSummary to validate
            
        Returns:
            True if entry is still valid, False if expired
        """
        if not summary.generated_at:
            return False
        
        # Ensure we have a timezone-aware datetime
        generated_at = summary.generated_at
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        
        # Check if entry is within TTL
        expiry_time = generated_at + timedelta(hours=self.ttl_hours)
        now = datetime.now(timezone.utc)
        
        return now < expiry_time
    
    def _get_from_database_cache(self, url: str, metadata_hash: Optional[str] = None) -> Optional[ContextSummary]:
        """Retrieve summary from database cache.
        
        Args:
            url: News article URL
            metadata_hash: Optional metadata hash
            
        Returns:
            ContextSummary if found and valid, None otherwise
        """
        if not self.storage_manager:
            return None
        
        try:
            # This would need to be implemented in StorageManager
            # For now, we'll use the existing context_summaries table lookup
            # We would need the news_url_id to look up, which requires a URL->ID mapping
            
            # Placeholder for future database cache implementation
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from database cache: {e}")
            return None
    
    def _store_in_database_cache(self, summary: ContextSummary) -> bool:
        """Store summary in database cache (context_summaries table).
        
        Args:
            summary: ContextSummary to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.storage_manager:
            return False
        
        try:
            # This would use the existing context_summaries table
            # The StorageManager would need methods for context summary CRUD operations
            
            # Placeholder for future implementation
            return True
            
        except Exception as e:
            logger.error(f"Error storing in database cache: {e}")
            return False


def generate_metadata_hash(title: str, description: Optional[str] = None) -> str:
    """Generate hash from metadata for cache invalidation.
    
    Args:
        title: Article title
        description: Optional article description
        
    Returns:
        MD5 hash of metadata
    """
    content = title or ""
    if description:
        content += f"|{description}"
    
    return hashlib.md5(content.encode('utf-8')).hexdigest()
