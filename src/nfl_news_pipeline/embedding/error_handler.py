"""Error handling and retry logic for embedding operations."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models import ContextSummary, StoryEmbedding

logger = logging.getLogger(__name__)


class EmbeddingErrorType(Enum):
    """Types of embedding generation errors."""
    API_RATE_LIMIT = "api_rate_limit"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    API_CONNECTION_ERROR = "api_connection_error"
    API_AUTHENTICATION_ERROR = "api_authentication_error"
    MODEL_LOADING_ERROR = "model_loading_error"
    MEMORY_ERROR = "memory_error"
    VALIDATION_ERROR = "validation_error"
    DIMENSION_MISMATCH = "dimension_mismatch"
    STORAGE_ERROR = "storage_error"
    UNKNOWN_ERROR = "unknown_error"


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE_RETRY = "immediate_retry"
    NO_RETRY = "no_retry"


class EmbeddingErrorHandler:
    """Handle embedding generation errors with sophisticated retry logic.
    
    Provides error categorization, retry strategies, and fallback mechanisms
    to ensure robust embedding generation.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 300
    ):
        """Initialize the error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay between retries
            jitter: Whether to add random jitter to delays
            enable_circuit_breaker: Whether to enable circuit breaker pattern
            circuit_breaker_threshold: Number of failures to trigger circuit breaker
            circuit_breaker_timeout: Circuit breaker timeout in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Circuit breaker state
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self._failure_counts: Dict[str, int] = {}
        self._circuit_breaker_open_times: Dict[str, datetime] = {}
        
        # Error mapping to retry strategies
        self.error_retry_map = {
            EmbeddingErrorType.API_RATE_LIMIT: RetryStrategy.EXPONENTIAL_BACKOFF,
            EmbeddingErrorType.API_CONNECTION_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
            EmbeddingErrorType.MODEL_LOADING_ERROR: RetryStrategy.LINEAR_BACKOFF,
            EmbeddingErrorType.MEMORY_ERROR: RetryStrategy.LINEAR_BACKOFF,
            EmbeddingErrorType.DIMENSION_MISMATCH: RetryStrategy.IMMEDIATE_RETRY,
            EmbeddingErrorType.STORAGE_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
            EmbeddingErrorType.API_QUOTA_EXCEEDED: RetryStrategy.NO_RETRY,
            EmbeddingErrorType.API_AUTHENTICATION_ERROR: RetryStrategy.NO_RETRY,
            EmbeddingErrorType.VALIDATION_ERROR: RetryStrategy.NO_RETRY,
            EmbeddingErrorType.UNKNOWN_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
        }
        
        logger.info(f"EmbeddingErrorHandler initialized with max_retries={max_retries}")

    async def handle_embedding_error(
        self,
        error: Exception,
        context_summary: ContextSummary,
        news_url_id: str,
        embedding_generator,
        attempt: int = 0
    ) -> Optional[StoryEmbedding]:
        """Handle embedding generation error with appropriate retry logic.
        
        Args:
            error: The exception that occurred
            context_summary: The context summary being processed
            news_url_id: The news URL ID
            embedding_generator: The embedding generator instance
            attempt: Current attempt number (0-based)
            
        Returns:
            StoryEmbedding if retry succeeds, None if all retries exhausted
        """
        error_type = self._classify_error(error)
        retry_strategy = self.error_retry_map.get(error_type, RetryStrategy.NO_RETRY)
        
        logger.warning(
            f"Embedding error for {news_url_id} (attempt {attempt + 1}): "
            f"{error_type.value} - {str(error)}"
        )
        
        # Check circuit breaker
        if self.enable_circuit_breaker and self._is_circuit_breaker_open(error_type.value):
            logger.error(f"Circuit breaker open for {error_type.value}, skipping retry")
            return None
        
        # Check if we should retry
        if retry_strategy == RetryStrategy.NO_RETRY or attempt >= self.max_retries:
            self._record_failure(error_type.value)
            logger.error(f"No more retries for {news_url_id} after {attempt + 1} attempts")
            return None
        
        # Calculate delay and wait
        delay = self._calculate_delay(retry_strategy, attempt)
        if delay > 0:
            logger.info(f"Retrying {news_url_id} in {delay:.2f} seconds")
            await asyncio.sleep(delay)
        
        # Attempt retry
        try:
            result = await embedding_generator.generate_embedding(context_summary, news_url_id)
            self._record_success(error_type.value)
            logger.info(f"Retry successful for {news_url_id} on attempt {attempt + 2}")
            return result
        except Exception as retry_error:
            # Recursive retry
            return await self.handle_embedding_error(
                retry_error, context_summary, news_url_id, embedding_generator, attempt + 1
            )

    async def handle_batch_embedding_errors(
        self,
        errors: List[Tuple[Exception, ContextSummary, str]],
        embedding_generator,
        max_individual_retries: int = 2
    ) -> List[Optional[StoryEmbedding]]:
        """Handle errors from batch embedding generation.
        
        Args:
            errors: List of (error, context_summary, news_url_id) tuples
            embedding_generator: The embedding generator instance
            max_individual_retries: Maximum retries for individual items
            
        Returns:
            List of StoryEmbedding objects (None for failed items)
        """
        results = []
        
        for error, context_summary, news_url_id in errors:
            try:
                result = await self.handle_embedding_error(
                    error, context_summary, news_url_id, embedding_generator, 0
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to handle error for {news_url_id}: {e}")
                results.append(None)
                
        return results

    async def handle_storage_error(
        self,
        error: Exception,
        embedding: StoryEmbedding,
        storage_manager,
        attempt: int = 0
    ) -> bool:
        """Handle storage operation errors with retry logic.
        
        Args:
            error: The storage exception that occurred
            embedding: The embedding being stored
            storage_manager: The storage manager instance
            attempt: Current attempt number (0-based)
            
        Returns:
            True if retry succeeds, False if all retries exhausted
        """
        error_type = self._classify_storage_error(error)
        retry_strategy = self.error_retry_map.get(error_type, RetryStrategy.EXPONENTIAL_BACKOFF)
        
        logger.warning(
            f"Storage error for {embedding.news_url_id} (attempt {attempt + 1}): "
            f"{error_type.value} - {str(error)}"
        )
        
        # Check circuit breaker for storage operations
        if self.enable_circuit_breaker and self._is_circuit_breaker_open("storage"):
            logger.error("Circuit breaker open for storage operations, skipping retry")
            return False
        
        # Check if we should retry
        if retry_strategy == RetryStrategy.NO_RETRY or attempt >= self.max_retries:
            self._record_failure("storage")
            logger.error(f"No more storage retries for {embedding.news_url_id}")
            return False
        
        # Calculate delay and wait
        delay = self._calculate_delay(retry_strategy, attempt)
        if delay > 0:
            logger.info(f"Retrying storage for {embedding.news_url_id} in {delay:.2f} seconds")
            await asyncio.sleep(delay)
        
        # Attempt retry
        try:
            result = await storage_manager.store_embedding(embedding)
            if result:
                self._record_success("storage")
                logger.info(f"Storage retry successful for {embedding.news_url_id}")
                return True
            else:
                # Storage returned False - treat as failure
                return await self.handle_storage_error(
                    Exception("Storage operation returned False"),
                    embedding, storage_manager, attempt + 1
                )
        except Exception as retry_error:
            # Recursive retry
            return await self.handle_storage_error(
                retry_error, embedding, storage_manager, attempt + 1
            )

    def _classify_error(self, error: Exception) -> EmbeddingErrorType:
        """Classify an embedding generation error by type."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Check for specific error patterns
        if "rate limit" in error_str or "429" in error_str:
            return EmbeddingErrorType.API_RATE_LIMIT
        elif "quota" in error_str or "usage limit" in error_str:
            return EmbeddingErrorType.API_QUOTA_EXCEEDED
        elif "authentication" in error_str or "401" in error_str or "api key" in error_str:
            return EmbeddingErrorType.API_AUTHENTICATION_ERROR
        elif "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return EmbeddingErrorType.API_CONNECTION_ERROR
        elif "memory" in error_str or "oom" in error_str:
            return EmbeddingErrorType.MEMORY_ERROR
        elif "model" in error_str and ("load" in error_str or "download" in error_str):
            return EmbeddingErrorType.MODEL_LOADING_ERROR
        elif "dimension" in error_str or "shape" in error_str:
            return EmbeddingErrorType.DIMENSION_MISMATCH
        elif "validation" in error_str or "invalid" in error_str:
            return EmbeddingErrorType.VALIDATION_ERROR
        elif "valueerror" in error_type_name:
            return EmbeddingErrorType.VALIDATION_ERROR
        elif "memoryerror" in error_type_name:
            return EmbeddingErrorType.MEMORY_ERROR
        else:
            return EmbeddingErrorType.UNKNOWN_ERROR

    def _classify_storage_error(self, error: Exception) -> EmbeddingErrorType:
        """Classify a storage error by type."""
        error_str = str(error).lower()
        
        if "connection" in error_str or "network" in error_str or "timeout" in error_str:
            return EmbeddingErrorType.STORAGE_ERROR
        elif "validation" in error_str or "invalid" in error_str:
            return EmbeddingErrorType.VALIDATION_ERROR
        else:
            return EmbeddingErrorType.STORAGE_ERROR

    def _calculate_delay(self, strategy: RetryStrategy, attempt: int) -> float:
        """Calculate delay for retry based on strategy and attempt number."""
        if strategy == RetryStrategy.NO_RETRY or strategy == RetryStrategy.IMMEDIATE_RETRY:
            return 0.0
        
        if strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay

    def _is_circuit_breaker_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        if not self.enable_circuit_breaker:
            return False
        
        # Check if circuit breaker is currently open
        if operation in self._circuit_breaker_open_times:
            open_time = self._circuit_breaker_open_times[operation]
            if datetime.now(timezone.utc) - open_time < timedelta(seconds=self.circuit_breaker_timeout):
                return True
            else:
                # Circuit breaker timeout expired, reset
                del self._circuit_breaker_open_times[operation]
                self._failure_counts[operation] = 0
        
        # Check if we should open the circuit breaker
        failure_count = self._failure_counts.get(operation, 0)
        if failure_count >= self.circuit_breaker_threshold:
            self._circuit_breaker_open_times[operation] = datetime.now(timezone.utc)
            logger.warning(f"Circuit breaker opened for {operation} after {failure_count} failures")
            return True
        
        return False

    def _record_failure(self, operation: str) -> None:
        """Record a failure for circuit breaker tracking."""
        if not self.enable_circuit_breaker:
            return
        
        self._failure_counts[operation] = self._failure_counts.get(operation, 0) + 1
        logger.debug(f"Recorded failure for {operation}, count: {self._failure_counts[operation]}")

    def _record_success(self, operation: str) -> None:
        """Record a success, which resets failure count."""
        if not self.enable_circuit_breaker:
            return
        
        if operation in self._failure_counts:
            del self._failure_counts[operation]
        if operation in self._circuit_breaker_open_times:
            del self._circuit_breaker_open_times[operation]
        logger.debug(f"Recorded success for {operation}, reset failure count")

    async def create_fallback_embedding(
        self,
        context_summary: ContextSummary,
        news_url_id: str,
        fallback_strategy: str = "metadata_based"
    ) -> Optional[StoryEmbedding]:
        """Create a fallback embedding when all generation methods fail.
        
        Args:
            context_summary: The context summary
            news_url_id: The news URL ID
            fallback_strategy: Strategy for fallback ("metadata_based" or "zero_vector")
            
        Returns:
            Fallback StoryEmbedding or None if fallback also fails
        """
        try:
            if fallback_strategy == "zero_vector":
                # Create a zero vector as last resort
                from ..models import EMBEDDING_DIM
                zero_vector = [0.0] * EMBEDDING_DIM
                
                return StoryEmbedding(
                    news_url_id=news_url_id,
                    embedding_vector=zero_vector,
                    model_name="fallback_zero",
                    model_version="1.0",
                    summary_text=context_summary.summary_text,
                    confidence_score=0.0,
                    generated_at=datetime.now(timezone.utc)
                )
            
            elif fallback_strategy == "metadata_based":
                # Create simple hash-based vector from text
                import hashlib
                from ..models import EMBEDDING_DIM
                
                text_hash = hashlib.md5(context_summary.summary_text.encode()).hexdigest()
                
                # Convert hash to vector
                vector = []
                for i in range(0, min(len(text_hash), EMBEDDING_DIM), 2):
                    hex_pair = text_hash[i:i+2] if i+1 < len(text_hash) else text_hash[i:i+1] + "0"
                    vector.append(int(hex_pair, 16) / 255.0 - 0.5)  # Normalize to [-0.5, 0.5]
                
                # Pad or truncate to correct dimension
                while len(vector) < EMBEDDING_DIM:
                    vector.append(0.0)
                vector = vector[:EMBEDDING_DIM]
                
                return StoryEmbedding(
                    news_url_id=news_url_id,
                    embedding_vector=vector,
                    model_name="fallback_hash",
                    model_version="1.0",
                    summary_text=context_summary.summary_text,
                    confidence_score=0.1,
                    generated_at=datetime.now(timezone.utc)
                )
            
            else:
                logger.error(f"Unknown fallback strategy: {fallback_strategy}")
                return None
                
        except Exception as e:
            logger.error(f"Fallback embedding creation failed for {news_url_id}: {e}")
            return None

    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about errors and circuit breaker state.
        
        Returns:
            Dictionary with error handling statistics
        """
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "circuit_breaker_enabled": self.enable_circuit_breaker,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.circuit_breaker_timeout,
            "current_failure_counts": dict(self._failure_counts),
            "open_circuit_breakers": list(self._circuit_breaker_open_times.keys()),
            "error_retry_strategies": {k.value: v.value for k, v in self.error_retry_map.items()}
        }

    def reset_circuit_breakers(self) -> None:
        """Manually reset all circuit breakers."""
        self._failure_counts.clear()
        self._circuit_breaker_open_times.clear()
        logger.info("All circuit breakers have been reset")

    async def validate_embedding_pipeline(
        self,
        test_summary: ContextSummary,
        embedding_generator,
        storage_manager
    ) -> Dict[str, Any]:
        """Validate the embedding pipeline end-to-end.
        
        Args:
            test_summary: A test context summary
            embedding_generator: The embedding generator to test
            storage_manager: The storage manager to test
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "embedding_generation": False,
            "storage_operation": False,
            "retrieval_operation": False,
            "errors": [],
            "warnings": []
        }
        
        test_url_id = f"test_{int(time.time())}"
        
        try:
            # Test embedding generation
            embedding = await embedding_generator.generate_embedding(test_summary, test_url_id)
            if embedding:
                results["embedding_generation"] = True
                
                # Test storage
                storage_success = await storage_manager.store_embedding(embedding)
                if storage_success:
                    results["storage_operation"] = True
                    
                    # Test retrieval
                    retrieved = await storage_manager.get_embedding_by_url_id(test_url_id)
                    if retrieved:
                        results["retrieval_operation"] = True
                    else:
                        results["errors"].append("Failed to retrieve stored embedding")
                        
                    # Cleanup test data
                    try:
                        await storage_manager.delete_embedding(test_url_id)
                    except Exception as cleanup_error:
                        results["warnings"].append(f"Failed to cleanup test data: {cleanup_error}")
                else:
                    results["errors"].append("Failed to store embedding")
            else:
                results["errors"].append("Failed to generate embedding")
                
        except Exception as e:
            results["errors"].append(f"Pipeline validation failed: {e}")
            
        return results