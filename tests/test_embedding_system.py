"""Tests for embedding generation, storage, and error handling."""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.nfl_news_pipeline.models import ContextSummary, StoryEmbedding, EMBEDDING_DIM
from src.nfl_news_pipeline.embedding import EmbeddingGenerator, EmbeddingStorageManager, EmbeddingErrorHandler
from src.nfl_news_pipeline.embedding.error_handler import EmbeddingErrorType, RetryStrategy


@pytest.fixture
def sample_context_summary():
    """Create a sample context summary for testing."""
    return ContextSummary(
        news_url_id="test-url-123",
        summary_text="Chiefs quarterback trade rumors intensify as team looks for backup options",
        llm_model="gpt-4o-nano",
        confidence_score=0.9,
        entities={"teams": ["KC", "Chiefs"], "positions": ["quarterback"]},
        key_topics=["trade", "quarterback", "backup"],
        fallback_used=False,
        generated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_embedding():
    """Create a sample embedding for testing."""
    vector = np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()
    return StoryEmbedding(
        news_url_id="test-url-123",
        embedding_vector=vector,
        model_name="text-embedding-3-small",
        model_version="1.0",
        summary_text="Chiefs quarterback trade rumors intensify",
        confidence_score=0.9,
        generated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    mock_client = Mock()
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    return mock_client


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""

    def test_init_with_defaults(self):
        """Test EmbeddingGenerator initialization with default settings."""
        generator = EmbeddingGenerator()
        
        assert generator.openai_model == "text-embedding-3-small"
        assert generator.sentence_transformer_model == "all-MiniLM-L6-v2"
        assert generator.batch_size == 32
        assert generator.use_openai_primary is True
        assert generator.openai_client is None

    def test_init_with_api_key(self):
        """Test EmbeddingGenerator initialization with OpenAI API key."""
        with patch('openai.OpenAI') as mock_openai:
            generator = EmbeddingGenerator(openai_api_key="test-key")
            
            mock_openai.assert_called_once_with(api_key="test-key")
            assert generator.openai_client is not None

    def test_prepare_embedding_text(self, sample_context_summary):
        """Test text preparation for embedding generation."""
        generator = EmbeddingGenerator()
        
        text = generator._prepare_embedding_text(sample_context_summary)
        
        assert "Chiefs quarterback trade rumors" in text
        assert "Topics: trade, quarterback, backup" in text
        assert "teams: KC, Chiefs" in text
        assert "positions: quarterback" in text

    def test_normalize_vector_correct_dimension(self):
        """Test vector normalization with correct dimensions."""
        generator = EmbeddingGenerator()
        vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        
        normalized = generator._normalize_vector(vector)
        
        assert len(normalized) == EMBEDDING_DIM
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6  # Should be unit vector

    def test_normalize_vector_padding(self):
        """Test vector normalization with padding for small vectors."""
        generator = EmbeddingGenerator()
        small_vector = np.random.rand(384).astype(np.float32)  # Smaller than EMBEDDING_DIM
        
        normalized = generator._normalize_vector(small_vector)
        
        assert len(normalized) == EMBEDDING_DIM
        # First 384 elements should be normalized original, rest should be 0
        assert np.all(normalized[384:] == 0)

    def test_normalize_vector_truncation(self):
        """Test vector normalization with truncation for large vectors."""
        generator = EmbeddingGenerator()
        large_vector = np.random.rand(EMBEDDING_DIM + 100).astype(np.float32)
        
        normalized = generator._normalize_vector(large_vector)
        
        assert len(normalized) == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_generate_transformer_embedding(self, sample_context_summary):
        """Test embedding generation using sentence transformer."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            # Mock the sentence transformer
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_transformer.return_value = mock_model
            
            generator = EmbeddingGenerator(use_openai_primary=False)
            
            embedding = await generator.generate_embedding(sample_context_summary, "test-url-123")
            
            assert embedding is not None
            assert embedding.news_url_id == "test-url-123"
            assert len(embedding.embedding_vector) == EMBEDDING_DIM
            assert embedding.model_name == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_generate_openai_embedding_fallback(self, sample_context_summary):
        """Test fallback to sentence transformer when OpenAI fails."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            # Mock the sentence transformer
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_transformer.return_value = mock_model
            
            # Create generator with OpenAI but no API key (will fail)
            generator = EmbeddingGenerator(use_openai_primary=True)
            
            embedding = await generator.generate_embedding(sample_context_summary, "test-url-123")
            
            assert embedding is not None
            assert embedding.model_name == "all-MiniLM-L6-v2"  # Should use fallback

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, sample_context_summary):
        """Test batch embedding generation."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            # Mock the sentence transformer for batch processing
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
            mock_transformer.return_value = mock_model
            
            generator = EmbeddingGenerator(use_openai_primary=False, batch_size=2)
            
            summaries = [sample_context_summary, sample_context_summary]
            url_ids = ["test-url-1", "test-url-2"]
            
            embeddings = await generator.generate_embeddings_batch(summaries, url_ids)
            
            assert len(embeddings) == 2
            assert embeddings[0].news_url_id == "test-url-1"
            assert embeddings[1].news_url_id == "test-url-2"

    def test_get_model_info(self):
        """Test model information retrieval."""
        generator = EmbeddingGenerator()
        
        info = generator.get_model_info()
        
        assert "primary_method" in info
        assert "openai_model" in info
        assert "sentence_transformer_model" in info
        assert "embedding_dimension" in info
        assert info["embedding_dimension"] == EMBEDDING_DIM


class TestEmbeddingStorageManager:
    """Test cases for EmbeddingStorageManager."""

    def test_init(self, mock_supabase_client):
        """Test EmbeddingStorageManager initialization."""
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        assert storage.supabase == mock_supabase_client
        assert storage.table_name == "story_embeddings"

    @pytest.mark.asyncio
    async def test_store_embedding_new(self, mock_supabase_client, sample_embedding):
        """Test storing a new embedding."""
        # Mock successful insertion
        mock_response = Mock()
        mock_response.data = [{"id": "new-id"}]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.insert.return_value.execute.return_value = mock_response
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = Mock(data=[])
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.store_embedding(sample_embedding)
        
        assert result is True
        mock_table.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_embedding_update_existing(self, mock_supabase_client, sample_embedding):
        """Test updating an existing embedding."""
        # Mock existing embedding found
        existing_data = sample_embedding.to_db()
        mock_existing_response = Mock()
        mock_existing_response.data = [existing_data]
        
        # Mock successful update
        mock_update_response = Mock()
        mock_update_response.data = [existing_data]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_existing_response
        mock_table.update.return_value.eq.return_value.execute.return_value = mock_update_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.store_embedding(sample_embedding)
        
        assert result is True
        mock_table.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_by_url_id(self, mock_supabase_client, sample_embedding):
        """Test retrieving embedding by URL ID."""
        # Mock successful retrieval
        mock_response = Mock()
        mock_response.data = [sample_embedding.to_db()]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.get_embedding_by_url_id("test-url-123")
        
        assert result is not None
        assert result.news_url_id == "test-url-123"

    @pytest.mark.asyncio
    async def test_get_embedding_by_url_id_not_found(self, mock_supabase_client):
        """Test retrieving non-existent embedding."""
        # Mock no data found
        mock_response = Mock()
        mock_response.data = []
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.get_embedding_by_url_id("non-existent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_store_embeddings_batch(self, mock_supabase_client, sample_embedding):
        """Test batch storage of embeddings."""
        # Mock no existing embeddings
        mock_existing_response = Mock()
        mock_existing_response.data = []
        
        # Mock successful batch insert
        mock_insert_response = Mock()
        mock_insert_response.data = [{"id": "1"}, {"id": "2"}]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.in_.return_value.execute.return_value = mock_existing_response
        mock_table.insert.return_value.execute.return_value = mock_insert_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        # Create two embeddings
        embedding2 = StoryEmbedding(
            news_url_id="test-url-456",
            embedding_vector=sample_embedding.embedding_vector,
            model_name="test-model",
            model_version="1.0",
            summary_text="Test summary 2",
            confidence_score=0.8
        )
        
        successful, failed = await storage.store_embeddings_batch([sample_embedding, embedding2])
        
        assert successful == 2
        assert failed == 0

    @pytest.mark.asyncio
    async def test_get_embedding_stats(self, mock_supabase_client):
        """Test retrieving embedding statistics."""
        # Mock count response
        mock_count_response = Mock()
        mock_count_response.count = 100
        
        # Mock models response
        mock_models_response = Mock()
        mock_models_response.data = [
            {"model_name": "test-model", "model_version": "1.0"},
            {"model_name": "test-model", "model_version": "1.0"},
            {"model_name": "other-model", "model_version": "2.0"}
        ]
        
        # Mock recent count response
        mock_recent_response = Mock()
        mock_recent_response.count = 25
        
        mock_table = mock_supabase_client.table.return_value
        
        # Configure different mock responses for different calls
        def select_side_effect(*args, **kwargs):
            if "count" in kwargs and kwargs["count"] == "exact":
                if len(args) == 1:  # Total count call
                    return Mock(execute=lambda: mock_count_response)
                else:  # Recent count call
                    return Mock(gte=lambda x, y: Mock(execute=lambda: mock_recent_response))
            else:  # Models call
                return Mock(execute=lambda: mock_models_response)
        
        mock_table.select.side_effect = select_side_effect
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        stats = await storage.get_embedding_stats()
        
        assert "total_embeddings" in stats
        assert "model_distribution" in stats
        assert "recent_embeddings_24h" in stats

    @pytest.mark.asyncio
    async def test_verify_embedding_integrity(self, mock_supabase_client, sample_embedding):
        """Test embedding integrity verification."""
        # Mock successful retrieval
        mock_response = Mock()
        mock_response.data = [sample_embedding.to_db()]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.verify_embedding_integrity("test-url-123")
        
        assert result["exists"] is True
        assert result["valid"] is True
        assert "vector_length" in result
        assert "vector_norm" in result


class TestEmbeddingErrorHandler:
    """Test cases for EmbeddingErrorHandler."""

    def test_init_with_defaults(self):
        """Test EmbeddingErrorHandler initialization with defaults."""
        handler = EmbeddingErrorHandler()
        
        assert handler.max_retries == 3
        assert handler.base_delay == 1.0
        assert handler.max_delay == 60.0
        assert handler.jitter is True
        assert handler.enable_circuit_breaker is True

    def test_classify_error_rate_limit(self):
        """Test error classification for rate limiting."""
        handler = EmbeddingErrorHandler()
        
        error = Exception("Rate limit exceeded")
        error_type = handler._classify_error(error)
        
        assert error_type == EmbeddingErrorType.API_RATE_LIMIT

    def test_classify_error_authentication(self):
        """Test error classification for authentication errors."""
        handler = EmbeddingErrorHandler()
        
        error = Exception("Invalid API key")
        error_type = handler._classify_error(error)
        
        assert error_type == EmbeddingErrorType.API_AUTHENTICATION_ERROR

    def test_classify_error_memory(self):
        """Test error classification for memory errors."""
        handler = EmbeddingErrorHandler()
        
        error = MemoryError("Out of memory")
        error_type = handler._classify_error(error)
        
        assert error_type == EmbeddingErrorType.MEMORY_ERROR

    def test_calculate_delay_exponential(self):
        """Test delay calculation for exponential backoff."""
        handler = EmbeddingErrorHandler(base_delay=1.0, jitter=False)
        
        delay0 = handler._calculate_delay(RetryStrategy.EXPONENTIAL_BACKOFF, 0)
        delay1 = handler._calculate_delay(RetryStrategy.EXPONENTIAL_BACKOFF, 1)
        delay2 = handler._calculate_delay(RetryStrategy.EXPONENTIAL_BACKOFF, 2)
        
        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_calculate_delay_linear(self):
        """Test delay calculation for linear backoff."""
        handler = EmbeddingErrorHandler(base_delay=1.0, jitter=False)
        
        delay0 = handler._calculate_delay(RetryStrategy.LINEAR_BACKOFF, 0)
        delay1 = handler._calculate_delay(RetryStrategy.LINEAR_BACKOFF, 1)
        delay2 = handler._calculate_delay(RetryStrategy.LINEAR_BACKOFF, 2)
        
        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 3.0

    def test_calculate_delay_max_limit(self):
        """Test delay calculation respects maximum limit."""
        handler = EmbeddingErrorHandler(base_delay=1.0, max_delay=5.0, jitter=False)
        
        delay = handler._calculate_delay(RetryStrategy.EXPONENTIAL_BACKOFF, 10)
        
        assert delay <= 5.0

    def test_circuit_breaker_opening(self):
        """Test circuit breaker opening after threshold failures."""
        handler = EmbeddingErrorHandler(
            circuit_breaker_threshold=3,
            enable_circuit_breaker=True
        )
        
        # Record failures
        for _ in range(3):
            handler._record_failure("test_operation")
        
        # Circuit breaker should be open
        assert handler._is_circuit_breaker_open("test_operation") is True

    def test_circuit_breaker_reset_on_success(self):
        """Test circuit breaker reset on successful operation."""
        handler = EmbeddingErrorHandler(
            circuit_breaker_threshold=3,
            enable_circuit_breaker=True
        )
        
        # Record failures
        for _ in range(2):
            handler._record_failure("test_operation")
        
        # Record success - should reset
        handler._record_success("test_operation")
        
        # Circuit breaker should not be open
        assert handler._is_circuit_breaker_open("test_operation") is False

    @pytest.mark.asyncio
    async def test_create_fallback_embedding_zero_vector(self, sample_context_summary):
        """Test creating zero vector fallback embedding."""
        handler = EmbeddingErrorHandler()
        
        embedding = await handler.create_fallback_embedding(
            sample_context_summary, "test-url", "zero_vector"
        )
        
        assert embedding is not None
        assert embedding.model_name == "fallback_zero"
        assert all(x == 0.0 for x in embedding.embedding_vector)
        assert embedding.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_create_fallback_embedding_metadata_based(self, sample_context_summary):
        """Test creating metadata-based fallback embedding."""
        handler = EmbeddingErrorHandler()
        
        embedding = await handler.create_fallback_embedding(
            sample_context_summary, "test-url", "metadata_based"
        )
        
        assert embedding is not None
        assert embedding.model_name == "fallback_hash"
        assert len(embedding.embedding_vector) == EMBEDDING_DIM
        assert embedding.confidence_score == 0.1

    @pytest.mark.asyncio
    async def test_handle_embedding_error_no_retry(self, sample_context_summary):
        """Test error handling with no retry strategy."""
        handler = EmbeddingErrorHandler()
        mock_generator = Mock()
        
        # Authentication error should not retry
        auth_error = Exception("Invalid API key")
        
        result = await handler.handle_embedding_error(
            auth_error, sample_context_summary, "test-url", mock_generator, 0
        )
        
        assert result is None
        mock_generator.generate_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_embedding_error_with_retry_success(self, sample_context_summary, sample_embedding):
        """Test error handling with successful retry."""
        handler = EmbeddingErrorHandler(base_delay=0.01)  # Fast retry for testing
        mock_generator = AsyncMock()
        mock_generator.generate_embedding.return_value = sample_embedding
        
        # Rate limit error should retry
        rate_limit_error = Exception("Rate limit exceeded")
        
        result = await handler.handle_embedding_error(
            rate_limit_error, sample_context_summary, "test-url", mock_generator, 0
        )
        
        assert result == sample_embedding
        mock_generator.generate_embedding.assert_called_once()

    def test_get_error_stats(self):
        """Test error statistics retrieval."""
        handler = EmbeddingErrorHandler()
        
        stats = handler.get_error_stats()
        
        assert "max_retries" in stats
        assert "circuit_breaker_enabled" in stats
        assert "error_retry_strategies" in stats
        assert "current_failure_counts" in stats

    def test_reset_circuit_breakers(self):
        """Test manual circuit breaker reset."""
        handler = EmbeddingErrorHandler()
        
        # Record some failures
        handler._record_failure("test_op")
        handler._record_failure("test_op")
        
        # Reset
        handler.reset_circuit_breakers()
        
        assert len(handler._failure_counts) == 0
        assert len(handler._circuit_breaker_open_times) == 0

    @pytest.mark.asyncio
    async def test_validate_embedding_pipeline(self, sample_context_summary, sample_embedding):
        """Test end-to-end pipeline validation."""
        handler = EmbeddingErrorHandler()
        
        # Mock successful generator
        mock_generator = AsyncMock()
        mock_generator.generate_embedding.return_value = sample_embedding
        
        # Mock successful storage manager
        mock_storage = AsyncMock()
        mock_storage.store_embedding.return_value = True
        mock_storage.get_embedding_by_url_id.return_value = sample_embedding
        mock_storage.delete_embedding.return_value = True
        
        result = await handler.validate_embedding_pipeline(
            sample_context_summary, mock_generator, mock_storage
        )
        
        assert result["embedding_generation"] is True
        assert result["storage_operation"] is True
        assert result["retrieval_operation"] is True
        assert len(result["errors"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])