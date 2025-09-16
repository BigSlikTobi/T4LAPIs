"""
Comprehensive unit tests for embedding generation (Task 11.1).

These tests provide comprehensive coverage for the EmbeddingGenerator
with properly mocked dependencies to avoid external network calls.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from src.nfl_news_pipeline.models import ContextSummary, StoryEmbedding, EMBEDDING_DIM
from src.nfl_news_pipeline.embedding import EmbeddingGenerator, EmbeddingStorageManager, EmbeddingErrorHandler
from src.nfl_news_pipeline.embedding.error_handler import EmbeddingErrorType, RetryStrategy


class TestEmbeddingGeneratorComprehensive:
    """Comprehensive unit tests for EmbeddingGenerator with mocked dependencies."""
    
    @pytest.fixture
    def sample_context_summary(self):
        """Create a comprehensive context summary for testing."""
        return ContextSummary(
            news_url_id="test-url-12345",
            summary_text="Patrick Mahomes led the Kansas City Chiefs to a decisive 28-14 victory over the Las Vegas Raiders, throwing three touchdown passes and demonstrating exceptional quarterback leadership throughout the game.",
            llm_model="gpt-4o-nano",
            confidence_score=0.92,
            entities={
                "players": ["Patrick Mahomes"],
                "teams": ["Kansas City Chiefs", "Las Vegas Raiders"],
                "coaches": [],
                "positions": ["quarterback"]
            },
            key_topics=["victory", "touchdown", "quarterback", "leadership"],
            fallback_used=False,
            generated_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def mock_embedding_vector(self):
        """Create a mock embedding vector with correct dimensions."""
        # Create a normalized vector for consistent testing
        vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        return vector / np.linalg.norm(vector)  # Normalize
    
    def test_init_with_openai_primary_config(self):
        """Test initialization with OpenAI as primary embedding provider."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            generator = EmbeddingGenerator(
                openai_api_key="test-openai-key",
                openai_model="text-embedding-3-small",
                use_openai_primary=True,
                batch_size=16
            )
            
            assert generator.openai_client == mock_client
            assert generator.use_openai_primary is True
            assert generator.openai_model == "text-embedding-3-small"
            assert generator.batch_size == 16
            mock_openai_class.assert_called_once_with(api_key="test-openai-key")
    
    def test_init_with_sentence_transformer_primary(self):
        """Test initialization with SentenceTransformer as primary provider."""
        generator = EmbeddingGenerator(
            sentence_transformer_model="all-MiniLM-L6-v2",
            use_openai_primary=False,
            batch_size=32
        )
        
        assert generator.openai_client is None
        assert generator.use_openai_primary is False
        assert generator.sentence_transformer_model == "all-MiniLM-L6-v2"
        assert generator.batch_size == 32
        # Sentence transformer should be lazy-loaded
        assert generator._sentence_transformer is None
    
    @pytest.mark.asyncio
    async def test_generate_embedding_openai_success(self, sample_context_summary, mock_embedding_vector):
        """Test successful embedding generation with OpenAI."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock OpenAI embedding response
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = mock_embedding_vector.tolist()
            mock_client.embeddings.create.return_value = mock_response
            
            generator = EmbeddingGenerator(
                openai_api_key="test-key",
                use_openai_primary=True
            )
            
            result = await generator.generate_embedding(sample_context_summary, "test-url-12345")
            
            assert isinstance(result, StoryEmbedding)
            assert result.news_url_id == "test-url-12345"
            assert len(result.embedding_vector) == EMBEDDING_DIM
            assert result.model_name == "text-embedding-3-small"
            assert result.model_version == "1.0"
            assert result.summary_text == sample_context_summary.summary_text
            assert result.confidence_score == sample_context_summary.confidence_score
            
            # Verify OpenAI API call
            mock_client.embeddings.create.assert_called_once()
            call_args = mock_client.embeddings.create.call_args
            assert call_args[1]['model'] == "text-embedding-3-small"
            assert len(call_args[1]['input']) > 0
    
    @pytest.mark.asyncio
    async def test_generate_embedding_sentence_transformer_success(self, sample_context_summary, mock_embedding_vector):
        """Test successful embedding generation with SentenceTransformer."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            mock_model = Mock()
            mock_model.encode.return_value = mock_embedding_vector
            mock_transformer_class.return_value = mock_model
            
            generator = EmbeddingGenerator(
                use_openai_primary=False,
                sentence_transformer_model="all-MiniLM-L6-v2"
            )
            
            result = await generator.generate_embedding(sample_context_summary, "test-url-12345")
            
            assert isinstance(result, StoryEmbedding)
            assert result.news_url_id == "test-url-12345"
            assert len(result.embedding_vector) == EMBEDDING_DIM
            assert result.model_name == "all-MiniLM-L6-v2"
            assert result.confidence_score == sample_context_summary.confidence_score
            
            # Verify SentenceTransformer call
            mock_model.encode.assert_called_once()
            encoded_text = mock_model.encode.call_args[0][0]
            assert sample_context_summary.summary_text in encoded_text
    
    @pytest.mark.asyncio
    async def test_generate_embedding_openai_fallback_to_transformer(self, sample_context_summary, mock_embedding_vector):
        """Test fallback from OpenAI to SentenceTransformer when OpenAI fails."""
        with patch('openai.OpenAI') as mock_openai_class, \
             patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            
            # Setup failing OpenAI client
            mock_openai_client = Mock()
            mock_openai_class.return_value = mock_openai_client
            mock_openai_client.embeddings.create.side_effect = Exception("OpenAI API Error")
            
            # Setup working SentenceTransformer
            mock_transformer = Mock()
            mock_transformer.encode.return_value = mock_embedding_vector
            mock_transformer_class.return_value = mock_transformer
            
            generator = EmbeddingGenerator(
                openai_api_key="test-key",
                use_openai_primary=True
            )
            
            result = await generator.generate_embedding(sample_context_summary, "test-url-12345")
            
            assert isinstance(result, StoryEmbedding)
            assert result.model_name == "all-MiniLM-L6-v2"  # Should use fallback model
            assert len(result.embedding_vector) == EMBEDDING_DIM
    
    @pytest.mark.asyncio
    async def test_generate_embedding_all_methods_fail(self, sample_context_summary):
        """Test behavior when all embedding methods fail."""
        with patch('openai.OpenAI') as mock_openai_class, \
             patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            
            # Setup failing OpenAI client
            mock_openai_client = Mock()
            mock_openai_class.return_value = mock_openai_client
            mock_openai_client.embeddings.create.side_effect = Exception("OpenAI Error")
            
            # Setup failing SentenceTransformer
            mock_transformer_class.side_effect = Exception("Model loading failed")
            
            generator = EmbeddingGenerator(
                openai_api_key="test-key",
                use_openai_primary=True
            )
            
            with pytest.raises(ValueError, match="All embedding methods failed"):
                await generator.generate_embedding(sample_context_summary, "test-url-12345")
    
    def test_prepare_embedding_text_comprehensive(self, sample_context_summary):
        """Test embedding text preparation with various summary components."""
        generator = EmbeddingGenerator()
        
        # Test with full summary
        embedding_text = generator._prepare_embedding_text(sample_context_summary)
        
        assert sample_context_summary.summary_text in embedding_text
        assert "Patrick Mahomes" in embedding_text
        assert "Kansas City Chiefs" in embedding_text
        assert "Las Vegas Raiders" in embedding_text
        assert "victory" in embedding_text
        assert "quarterback" in embedding_text
        
        # Test with minimal summary
        minimal_summary = ContextSummary(
            news_url_id="test",
            summary_text="Short summary",
            llm_model="test",
            confidence_score=0.8
        )
        
        minimal_text = generator._prepare_embedding_text(minimal_summary)
        assert minimal_text == "Short summary"
        
        # Test with empty entities and topics
        empty_entities_summary = ContextSummary(
            news_url_id="test",
            summary_text="Summary text",
            llm_model="test",
            confidence_score=0.8,
            entities={},
            key_topics=[]
        )
        
        empty_text = generator._prepare_embedding_text(empty_entities_summary)
        assert empty_text == "Summary text"
    
    def test_normalize_vector_comprehensive(self, mock_embedding_vector):
        """Test vector normalization with various input types and edge cases."""
        generator = EmbeddingGenerator()
        
        # Test with numpy array
        normalized_np = generator._normalize_vector(mock_embedding_vector)
        assert isinstance(normalized_np, np.ndarray)
        assert abs(np.linalg.norm(normalized_np) - 1.0) < 1e-6  # Should be unit vector
        
        # Test with list
        vector_list = mock_embedding_vector.tolist()
        normalized_list = generator._normalize_vector(vector_list)
        assert isinstance(normalized_list, np.ndarray)
        assert abs(np.linalg.norm(normalized_list) - 1.0) < 1e-6
        
        # Test with zero vector (edge case)
        zero_vector = np.zeros(EMBEDDING_DIM)
        with pytest.raises(ValueError, match="Cannot normalize zero vector"):
            generator._normalize_vector(zero_vector)
        
        # Test with very small vector
        tiny_vector = np.full(EMBEDDING_DIM, 1e-10)
        normalized_tiny = generator._normalize_vector(tiny_vector)
        assert abs(np.linalg.norm(normalized_tiny) - 1.0) < 1e-6
        
        # Test with wrong dimensions
        wrong_dim_vector = np.random.rand(100)  # Wrong dimension
        with pytest.raises(ValueError, match="Vector must have dimension"):
            generator._normalize_vector(wrong_dim_vector)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_success(self, mock_embedding_vector):
        """Test successful batch embedding generation."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            mock_model = Mock()
            # Return batch of embeddings
            batch_embeddings = np.array([mock_embedding_vector, mock_embedding_vector * 0.8])
            mock_model.encode.return_value = batch_embeddings
            mock_transformer_class.return_value = mock_model
            
            generator = EmbeddingGenerator(
                use_openai_primary=False,
                batch_size=2
            )
            
            summaries = [
                ContextSummary(
                    news_url_id=f"test-url-{i}",
                    summary_text=f"Summary {i}",
                    llm_model="test",
                    confidence_score=0.8
                )
                for i in range(2)
            ]
            
            results = await generator.generate_embeddings_batch(summaries)
            
            assert len(results) == 2
            for i, result in enumerate(results):
                assert isinstance(result, StoryEmbedding)
                assert result.news_url_id == f"test-url-{i}"
                assert len(result.embedding_vector) == EMBEDDING_DIM
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_partial_failures(self, mock_embedding_vector):
        """Test batch processing with some individual failures."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            mock_model = Mock()
            # Simulate batch failure but individual successes
            mock_model.encode.side_effect = [
                Exception("Batch failed"),  # Batch call fails
                mock_embedding_vector,  # Individual call 1 succeeds
                mock_embedding_vector * 0.8,  # Individual call 2 succeeds
            ]
            mock_transformer_class.return_value = mock_model
            
            generator = EmbeddingGenerator(use_openai_primary=False)
            
            summaries = [
                ContextSummary(
                    news_url_id=f"test-url-{i}",
                    summary_text=f"Summary {i}",
                    llm_model="test", 
                    confidence_score=0.8
                )
                for i in range(2)
            ]
            
            results = await generator.generate_embeddings_batch(summaries)
            
            assert len(results) == 2
            for result in results:
                assert isinstance(result, StoryEmbedding)
    
    def test_get_model_info_comprehensive(self):
        """Test model information retrieval."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            generator = EmbeddingGenerator(
                openai_api_key="test-key",
                openai_model="text-embedding-3-large",
                sentence_transformer_model="all-mpnet-base-v2",
                use_openai_primary=True
            )
            
            info = generator.get_model_info()
            
            assert info["use_openai_primary"] is True
            assert info["openai_model"] == "text-embedding-3-large"
            assert info["sentence_transformer_model"] == "all-mpnet-base-v2"
            assert info["batch_size"] == 32  # default
            assert "openai_available" in info
            assert "sentence_transformer_available" in info
    
    @pytest.mark.asyncio
    async def test_embedding_text_length_limits(self):
        """Test handling of very long text inputs."""
        # Create a very long summary
        long_text = "This is a very long summary. " * 1000  # ~30k characters
        long_summary = ContextSummary(
            news_url_id="test-long",
            summary_text=long_text,
            llm_model="test",
            confidence_score=0.8
        )
        
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            mock_transformer_class.return_value = mock_model
            
            generator = EmbeddingGenerator(use_openai_primary=False)
            
            # Should handle long text without error
            result = await generator.generate_embedding(long_summary, "test-long")
            
            assert isinstance(result, StoryEmbedding)
            
            # Verify that text was potentially truncated for embedding
            encoded_text = mock_model.encode.call_args[0][0]
            # Most embedding models have token limits, so text might be truncated
            assert len(encoded_text) <= len(long_text)
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, mock_embedding_vector):
        """Test concurrent embedding generation for performance."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            mock_model = Mock()
            mock_model.encode.return_value = mock_embedding_vector
            mock_transformer_class.return_value = mock_model
            
            generator = EmbeddingGenerator(use_openai_primary=False)
            
            # Create multiple summaries
            summaries = [
                ContextSummary(
                    news_url_id=f"concurrent-test-{i}",
                    summary_text=f"Concurrent summary {i}",
                    llm_model="test",
                    confidence_score=0.8
                )
                for i in range(5)
            ]
            
            # Generate embeddings concurrently
            tasks = [
                generator.generate_embedding(summary, summary.news_url_id)
                for summary in summaries
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for i, result in enumerate(results):
                assert isinstance(result, StoryEmbedding)
                assert result.news_url_id == f"concurrent-test-{i}"


class TestEmbeddingStorageManagerComprehensive:
    """Comprehensive unit tests for EmbeddingStorageManager."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Create a mock Supabase client with proper structure."""
        mock_client = Mock()
        mock_table = Mock()
        mock_client.table.return_value = mock_table
        return mock_client
    
    @pytest.fixture
    def sample_embedding(self, mock_embedding_vector):
        """Create a sample embedding for testing."""
        return StoryEmbedding(
            news_url_id="test-storage-url",
            embedding_vector=mock_embedding_vector.tolist(),
            model_name="text-embedding-3-small",
            model_version="1.0",
            summary_text="Test summary for storage",
            confidence_score=0.85,
            generated_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def mock_embedding_vector(self):
        """Create a mock embedding vector."""
        return np.random.rand(EMBEDDING_DIM).astype(np.float32)
    
    def test_init_with_supabase_client(self, mock_supabase_client):
        """Test initialization with Supabase client."""
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        assert storage.supabase == mock_supabase_client
        assert storage.table_name == "story_embeddings"
    
    @pytest.mark.asyncio
    async def test_store_embedding_new_record(self, mock_supabase_client, sample_embedding):
        """Test storing a new embedding record."""
        # Mock no existing record
        mock_existing_response = Mock()
        mock_existing_response.data = []
        
        # Mock successful insert
        mock_insert_response = Mock()
        mock_insert_response.data = [sample_embedding.to_db()]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_existing_response
        mock_table.insert.return_value.execute.return_value = mock_insert_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.store_embedding(sample_embedding)
        
        assert result is True
        mock_table.insert.assert_called_once()
        inserted_data = mock_table.insert.call_args[0][0]
        assert inserted_data["news_url_id"] == sample_embedding.news_url_id
        assert len(inserted_data["embedding_vector"]) == EMBEDDING_DIM
    
    @pytest.mark.asyncio
    async def test_store_embedding_update_existing(self, mock_supabase_client, sample_embedding):
        """Test updating an existing embedding record."""
        # Mock existing record found
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
    async def test_store_embedding_database_error(self, mock_supabase_client, sample_embedding):
        """Test handling of database errors during storage."""
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.side_effect = Exception("Database error")
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.store_embedding(sample_embedding)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_embedding_by_url_id_found(self, mock_supabase_client, sample_embedding):
        """Test retrieving an existing embedding by URL ID."""
        mock_response = Mock()
        mock_response.data = [sample_embedding.to_db()]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.get_embedding_by_url_id("test-storage-url")
        
        assert isinstance(result, StoryEmbedding)
        assert result.news_url_id == "test-storage-url"
        assert len(result.embedding_vector) == EMBEDDING_DIM
    
    @pytest.mark.asyncio
    async def test_get_embedding_by_url_id_not_found(self, mock_supabase_client):
        """Test retrieving non-existent embedding."""
        mock_response = Mock()
        mock_response.data = []
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.get_embedding_by_url_id("non-existent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_embeddings_by_model_batch(self, mock_supabase_client, sample_embedding):
        """Test retrieving embeddings by model name in batch."""
        # Create multiple embeddings with same model
        embeddings_data = []
        for i in range(3):
            embedding_copy = StoryEmbedding(
                news_url_id=f"test-url-{i}",
                embedding_vector=sample_embedding.embedding_vector,
                model_name=sample_embedding.model_name,
                model_version=sample_embedding.model_version,
                summary_text=f"Summary {i}",
                confidence_score=0.8,
                generated_at=datetime.now(timezone.utc)
            )
            embeddings_data.append(embedding_copy.to_db())
        
        mock_response = Mock()
        mock_response.data = embeddings_data
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        results = await storage.get_embeddings_by_model("text-embedding-3-small", "1.0")
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, StoryEmbedding)
            assert result.news_url_id == f"test-url-{i}"
            assert result.model_name == "text-embedding-3-small"
    
    @pytest.mark.asyncio
    async def test_delete_embedding_success(self, mock_supabase_client):
        """Test successful embedding deletion."""
        mock_response = Mock()
        mock_response.data = [{"id": "deleted-id"}]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.delete.return_value.eq.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        result = await storage.delete_embedding("test-url-to-delete")
        
        assert result is True
        mock_table.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embedding_statistics(self, mock_supabase_client):
        """Test retrieving embedding statistics."""
        mock_response = Mock()
        mock_response.data = [
            {
                "model_name": "text-embedding-3-small",
                "count": 150,
                "avg_confidence": 0.87
            },
            {
                "model_name": "all-MiniLM-L6-v2", 
                "count": 75,
                "avg_confidence": 0.82
            }
        ]
        
        mock_table = mock_supabase_client.table.return_value
        mock_table.select.return_value.execute.return_value = mock_response
        
        storage = EmbeddingStorageManager(mock_supabase_client)
        
        stats = await storage.get_embedding_statistics()
        
        assert len(stats) == 2
        assert stats[0]["model_name"] == "text-embedding-3-small"
        assert stats[0]["count"] == 150
        assert stats[1]["avg_confidence"] == 0.82


class TestEmbeddingErrorHandlerComprehensive:
    """Comprehensive unit tests for EmbeddingErrorHandler."""
    
    @pytest.fixture
    def sample_context_summary(self):
        """Create a sample context summary."""
        return ContextSummary(
            news_url_id="test-error-handling",
            summary_text="Test summary for error handling",
            llm_model="test-model",
            confidence_score=0.8,
            entities={"teams": ["Test Team"]},
            key_topics=["test"],
            generated_at=datetime.now(timezone.utc)
        )
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        handler = EmbeddingErrorHandler(
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            backoff_multiplier=3.0
        )
        
        assert handler.max_retries == 5
        assert handler.base_delay == 2.0
        assert handler.max_delay == 60.0
        assert handler.backoff_multiplier == 3.0
    
    @pytest.mark.asyncio
    async def test_handle_embedding_error_with_retry_success(self, sample_context_summary):
        """Test error handling with successful retry."""
        handler = EmbeddingErrorHandler(max_retries=3)
        
        # Mock function that fails twice then succeeds
        mock_embedding_func = AsyncMock()
        success_embedding = StoryEmbedding(
            news_url_id=sample_context_summary.news_url_id,
            embedding_vector=[0.1] * EMBEDDING_DIM,
            model_name="test-model",
            model_version="1.0",
            summary_text=sample_context_summary.summary_text,
            confidence_score=0.8,
            generated_at=datetime.now(timezone.utc)
        )
        
        mock_embedding_func.side_effect = [
            Exception("Temporary error 1"),
            Exception("Temporary error 2"),
            success_embedding
        ]
        
        with patch('asyncio.sleep') as mock_sleep:  # Speed up test
            result = await handler.handle_embedding_error(
                mock_embedding_func,
                sample_context_summary,
                "test-url"
            )
        
        assert isinstance(result, StoryEmbedding)
        assert result.news_url_id == sample_context_summary.news_url_id
        assert mock_embedding_func.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries with delays
    
    @pytest.mark.asyncio
    async def test_handle_embedding_error_max_retries_exceeded(self, sample_context_summary):
        """Test error handling when max retries are exceeded."""
        handler = EmbeddingErrorHandler(max_retries=2)
        
        # Mock function that always fails
        mock_embedding_func = AsyncMock()
        mock_embedding_func.side_effect = Exception("Persistent error")
        
        with patch('asyncio.sleep'):  # Speed up test
            result = await handler.handle_embedding_error(
                mock_embedding_func,
                sample_context_summary,
                "test-url"
            )
        
        assert result is None
        assert mock_embedding_func.call_count == 3  # Initial + 2 retries
    
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
        assert embedding.summary_text == sample_context_summary.summary_text
        
        # Verify that the embedding is deterministic for same input
        embedding2 = await handler.create_fallback_embedding(
            sample_context_summary, "test-url", "metadata_based"
        )
        assert embedding.embedding_vector == embedding2.embedding_vector
    
    @pytest.mark.asyncio
    async def test_create_fallback_embedding_random(self, sample_context_summary):
        """Test creating random fallback embedding."""
        handler = EmbeddingErrorHandler()
        
        embedding = await handler.create_fallback_embedding(
            sample_context_summary, "test-url", "random"
        )
        
        assert embedding is not None
        assert embedding.model_name == "fallback_random"
        assert len(embedding.embedding_vector) == EMBEDDING_DIM
        assert embedding.confidence_score == 0.05
        
        # Verify that random embeddings are different
        embedding2 = await handler.create_fallback_embedding(
            sample_context_summary, "test-url-2", "random"
        )
        assert embedding.embedding_vector != embedding2.embedding_vector
    
    def test_classify_error_type(self):
        """Test error type classification."""
        handler = EmbeddingErrorHandler()
        
        # Test network errors
        network_errors = [
            Exception("Connection timeout"),
            Exception("Connection refused"),
            Exception("Network unreachable"),
            TimeoutError("Request timeout"),
        ]
        
        for error in network_errors:
            error_type = handler._classify_error_type(error)
            assert error_type == EmbeddingErrorType.NETWORK_ERROR
        
        # Test API errors
        api_errors = [
            Exception("API rate limit exceeded"),
            Exception("Invalid API key"),
            Exception("Quota exceeded"),
        ]
        
        for error in api_errors:
            error_type = handler._classify_error_type(error)
            assert error_type == EmbeddingErrorType.API_ERROR
        
        # Test model errors
        model_errors = [
            Exception("Model not found"),
            Exception("Model loading failed"),
            Exception("Out of memory"),
        ]
        
        for error in model_errors:
            error_type = handler._classify_error_type(error)
            assert error_type == EmbeddingErrorType.MODEL_ERROR
        
        # Test unknown errors
        unknown_error = Exception("Some unknown error")
        error_type = handler._classify_error_type(unknown_error)
        assert error_type == EmbeddingErrorType.UNKNOWN_ERROR
    
    def test_calculate_retry_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        handler = EmbeddingErrorHandler(
            base_delay=1.0,
            max_delay=30.0,
            backoff_multiplier=2.0
        )
        
        # Test exponential growth
        delay1 = handler._calculate_retry_delay(1)
        delay2 = handler._calculate_retry_delay(2)
        delay3 = handler._calculate_retry_delay(3)
        
        assert delay1 == 2.0  # base_delay * multiplier^1
        assert delay2 == 4.0  # base_delay * multiplier^2
        assert delay3 == 8.0  # base_delay * multiplier^3
        
        # Test max delay cap
        delay_large = handler._calculate_retry_delay(10)
        assert delay_large <= 30.0
    
    @pytest.mark.asyncio
    async def test_should_retry_based_on_error_type(self):
        """Test retry decision based on error type."""
        handler = EmbeddingErrorHandler()
        
        # Network errors should be retried
        assert handler._should_retry(EmbeddingErrorType.NETWORK_ERROR, 1) is True
        
        # API errors should be retried with limits
        assert handler._should_retry(EmbeddingErrorType.API_ERROR, 1) is True
        assert handler._should_retry(EmbeddingErrorType.API_ERROR, 5) is False  # Too many retries
        
        # Model errors should have limited retries
        assert handler._should_retry(EmbeddingErrorType.MODEL_ERROR, 1) is True
        assert handler._should_retry(EmbeddingErrorType.MODEL_ERROR, 3) is False
        
        # Unknown errors should be retried cautiously
        assert handler._should_retry(EmbeddingErrorType.UNKNOWN_ERROR, 1) is True
        assert handler._should_retry(EmbeddingErrorType.UNKNOWN_ERROR, 2) is False