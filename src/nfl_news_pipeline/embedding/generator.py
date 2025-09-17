"""Embedding generation for story similarity analysis."""

from __future__ import annotations

import logging
import asyncio
import inspect
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import openai
import sentence_transformers

from ..models import ContextSummary, StoryEmbedding, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate semantic embeddings from story summaries.
    
    Supports both OpenAI API embeddings (primary) and local sentence-transformers
    (fallback) to ensure reliable embedding generation.
    """

    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        use_openai_primary: bool = True
    ):
        """Initialize the embedding generator.
        
        Args:
            openai_api_key: OpenAI API key for primary embedding generation
            openai_model: OpenAI embedding model name  
            sentence_transformer_model: Local transformer model name for fallback
            batch_size: Maximum batch size for processing multiple summaries
            use_openai_primary: Whether to use OpenAI as primary (True) or sentence-transformers (False)
        """
        self.openai_model = openai_model
        self.sentence_transformer_model = sentence_transformer_model
        self.batch_size = batch_size
        self.use_openai_primary = use_openai_primary

        # Initialize OpenAI client if API key provided (use sync client for compatibility with tests)
        self.openai_client = None
        if openai_api_key:
            # Use OpenAI (sync) to match existing tests' expectations
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            
        # Initialize sentence transformer model (lazy loading)
        self._sentence_transformer: Optional[sentence_transformers.SentenceTransformer] = None
        self._sentence_transformer_cls = sentence_transformers.SentenceTransformer
        
        logger.info(
            f"EmbeddingGenerator initialized with primary model: "
            f"{'OpenAI ' + openai_model if use_openai_primary else 'SentenceTransformer ' + sentence_transformer_model}"
        )

    @property
    def sentence_transformer(self) -> sentence_transformers.SentenceTransformer:
        """Lazy load sentence transformer model to avoid startup delays."""
        if self._sentence_transformer is None:
            logger.info(f"Loading sentence transformer model: {self.sentence_transformer_model}")
            self._sentence_transformer = self._sentence_transformer_cls(self.sentence_transformer_model)
        return self._sentence_transformer

    async def generate_embedding(
        self, 
        context_summary: ContextSummary, 
        news_url_id: str
    ) -> StoryEmbedding:
        """Generate a single embedding from a context summary.
        
        Args:
            context_summary: The context summary to embed
            news_url_id: ID of the news URL this embedding belongs to
            
        Returns:
            StoryEmbedding with normalized vector and metadata
            
        Raises:
            ValueError: If embedding generation fails with both methods
        """
        embedding_text = self._prepare_embedding_text(context_summary)
        
        # Try primary method first
        vector = None
        model_name = None
        model_version = None
        
        if self.use_openai_primary and self.openai_client:
            try:
                vector, model_name, model_version = await self._generate_openai_embedding(embedding_text)
                logger.debug(f"Generated OpenAI embedding for news_url_id: {news_url_id}")
            except Exception as e:
                logger.warning(f"OpenAI embedding failed for {news_url_id}: {e}")
                
        # Try sentence transformer as fallback or primary
        if vector is None:
            try:
                vector, model_name, model_version = self._generate_transformer_embedding(embedding_text)
                logger.debug(f"Generated transformer embedding for news_url_id: {news_url_id}")
            except Exception as e:
                logger.error(f"Transformer embedding failed for {news_url_id}: {e}")
                raise ValueError(f"All embedding methods failed for {news_url_id}: {e}") from e
        
        # Normalize and validate vector
        normalized_vector = self._normalize_vector(vector)
        
        return StoryEmbedding(
            news_url_id=news_url_id,
            embedding_vector=normalized_vector.tolist(),
            model_name=model_name,
            model_version=model_version,
            summary_text=context_summary.summary_text,
            confidence_score=context_summary.confidence_score,
            generated_at=datetime.now(timezone.utc)
        )

    async def generate_embeddings_batch(
        self, 
        context_summaries: List[ContextSummary], 
        news_url_ids: Optional[List[str]] = None
    ) -> List[StoryEmbedding]:
        """Generate embeddings for multiple summaries efficiently.
        
        Args:
            context_summaries: List of context summaries to embed
            news_url_ids: Corresponding list of news URL IDs
            
        Returns:
            List of StoryEmbedding objects
            
        Raises:
            ValueError: If lengths don't match or batch processing fails
        """
        if news_url_ids is None:
            derived_ids: List[str] = []
            for index, summary in enumerate(context_summaries):
                summary_id = getattr(summary, "news_url_id", None)
                if not summary_id:
                    raise ValueError(
                        "news_url_ids not provided and context summary missing news_url_id"
                    )
                derived_ids.append(summary_id)
            news_url_ids = derived_ids

        if len(context_summaries) != len(news_url_ids):
            raise ValueError("context_summaries and news_url_ids must have same length")
            
        embeddings = []
        
        # Process in batches to manage memory and API limits
        for i in range(0, len(context_summaries), self.batch_size):
            batch_summaries = context_summaries[i:i + self.batch_size]
            batch_ids = news_url_ids[i:i + self.batch_size]
            
            batch_embeddings = await self._process_batch(batch_summaries, batch_ids)
            embeddings.extend(batch_embeddings)
            
            # Brief pause between batches to respect rate limits
            if i + self.batch_size < len(context_summaries):
                time.sleep(0.1)
                
        return embeddings

    async def _process_batch(
        self, 
        summaries: List[ContextSummary], 
        ids: List[str]
    ) -> List[StoryEmbedding]:
        """Process a single batch of summaries."""
        embeddings = []
        
        # Prepare embedding texts
        embedding_texts = [self._prepare_embedding_text(summary) for summary in summaries]
        
        # Try batch generation with primary method
        vectors = None
        model_name = None
        model_version = None
        
        if self.use_openai_primary and self.openai_client:
            try:
                vectors, model_name, model_version = await self._generate_openai_embeddings_batch(embedding_texts)
                logger.debug(f"Generated OpenAI batch embeddings for {len(ids)} summaries")
            except Exception as e:
                logger.warning(f"OpenAI batch embedding failed: {e}")
        
        # Fallback to sentence transformer batch
        if vectors is None:
            try:
                vectors, model_name, model_version = self._generate_transformer_embeddings_batch(embedding_texts)
                logger.debug(f"Generated transformer batch embeddings for {len(ids)} summaries")
            except Exception as e:
                logger.error(f"Transformer batch embedding failed: {e}")
                # Fall back to individual processing
                for summary, url_id in zip(summaries, ids):
                    try:
                        embedding = await self.generate_embedding(summary, url_id)
                        embeddings.append(embedding)
                    except Exception as individual_error:
                        logger.error(f"Individual embedding failed for {url_id}: {individual_error}")
                        # Skip this embedding rather than fail the entire batch
                        continue
                return embeddings
        
        # Create StoryEmbedding objects
        generation_time = datetime.now(timezone.utc)
        for i, (summary, url_id) in enumerate(zip(summaries, ids)):
            if i < len(vectors):
                normalized_vector = self._normalize_vector(vectors[i])
                embeddings.append(StoryEmbedding(
                    news_url_id=url_id,
                    embedding_vector=normalized_vector.tolist(),
                    model_name=model_name,
                    model_version=model_version,
                    summary_text=summary.summary_text,
                    confidence_score=summary.confidence_score,
                    generated_at=generation_time
                ))
            else:
                logger.warning(f"Missing vector for {url_id} in batch response")
                
        return embeddings

    async def _generate_openai_embedding(self, text: str) -> tuple[np.ndarray, str, str]:
        """Generate embedding using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        response = await self._openai_embeddings_create_async(
            input_payload=text
        )
        
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        model_version = self._extract_openai_model_version(response)
        
        return vector, self.openai_model, model_version

    async def _generate_openai_embeddings_batch(self, texts: List[str]) -> tuple[List[np.ndarray], str, str]:
        """Generate embeddings using OpenAI API in batch."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        response = await self._openai_embeddings_create_async(
            input_payload=texts
        )
        
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        model_version = self._extract_openai_model_version(response)
        
        return vectors, self.openai_model, model_version

    async def _openai_embeddings_create_async(self, input_payload: Union[str, List[str]]):
        """Call OpenAI embeddings.create in an async-friendly way for both sync and async clients.
        
        If the client is async, await directly; otherwise, offload the blocking call to a thread.
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        create_fn = self.openai_client.embeddings.create
        kwargs = {
            "model": self.openai_model,
            "input": input_payload,
            "encoding_format": "float",
        }

        if inspect.iscoroutinefunction(create_fn):
            return await create_fn(**kwargs)
        # Sync client: run in thread to avoid blocking
        return await asyncio.to_thread(create_fn, **kwargs)

    def _generate_transformer_embedding(self, text: str) -> tuple[np.ndarray, str, str]:
        """Generate embedding using sentence transformer."""
        vector = self.sentence_transformer.encode(text, convert_to_numpy=True)
        vector = np.asarray(vector, dtype=np.float32)

        # Get model version/info
        model_info = getattr(self.sentence_transformer, '_modules', {})
        model_version = str(hash(str(model_info)))[:8]  # Simple version hash

        return vector, self.sentence_transformer_model, model_version

    def _generate_transformer_embeddings_batch(self, texts: List[str]) -> tuple[List[np.ndarray], str, str]:
        """Generate embeddings using sentence transformer in batch."""
        vectors = self.sentence_transformer.encode(texts, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype=np.float32)

        # Get model version/info
        model_info = getattr(self.sentence_transformer, '_modules', {})
        model_version = str(hash(str(model_info)))[:8]  # Simple version hash

        # Convert to list of individual arrays
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)
        vector_list = [np.asarray(vectors[i], dtype=np.float32) for i in range(len(vectors))]

        return vector_list, self.sentence_transformer_model, model_version

    def _prepare_embedding_text(self, summary: ContextSummary) -> str:
        """Prepare text for embedding generation by combining summary components."""
        parts = [summary.summary_text]
        
        # Add key topics if available
        if summary.key_topics:
            topics_text = " Topics: " + ", ".join(summary.key_topics)
            parts.append(topics_text)
            
        # Add entities if available
        if summary.entities:
            if isinstance(summary.entities, dict):
                entities_parts = []
                for entity_type, entities_list in summary.entities.items():
                    if isinstance(entities_list, list):
                        entities_parts.append(f"{entity_type}: {', '.join(entities_list)}")
                if entities_parts:
                    parts.append(" Entities: " + "; ".join(entities_parts))
            
        return " ".join(parts)

    def _normalize_vector(self, vector: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Normalize embedding vector for consistent similarity calculations."""
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim > 1:
            vector = vector.flatten()

        target_dim = EMBEDDING_DIM
        current_dim = vector.shape[0]

        if current_dim == 0:
            raise ValueError("Cannot normalize empty vector")

        if current_dim < target_dim:
            if current_dim < max(1, target_dim // 4):
                raise ValueError(
                    f"Vector must have dimension {target_dim}, received {current_dim}"
                )
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:current_dim] = vector
            vector = padded
        elif current_dim > target_dim:
            vector = vector[:target_dim]

        norm = np.linalg.norm(vector)
        if norm <= 0.0:
            raise ValueError("Cannot normalize zero vector")

        return (vector / norm).astype(np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current embedding models."""
        info = {
            "primary_method": "openai" if self.use_openai_primary else "sentence_transformer",
            "use_openai_primary": self.use_openai_primary,
            "openai_model": self.openai_model,
            "sentence_transformer_model": self.sentence_transformer_model,
            "openai_available": self.openai_client is not None,
            "sentence_transformer_available": self._sentence_transformer is not None,
            "embedding_dimension": EMBEDDING_DIM,
            "batch_size": self.batch_size,
        }

        if hasattr(self, '_sentence_transformer') and self._sentence_transformer:
            info["sentence_transformer_loaded"] = True
            info["sentence_transformer_max_seq_length"] = getattr(
                self._sentence_transformer, 'max_seq_length', 'unknown'
            )
        else:
            info["sentence_transformer_loaded"] = False
            
        return info

    def _extract_openai_model_version(self, response: Any) -> str:
        """Derive model version from OpenAI response mocks safely."""
        candidates: Sequence[Any] = (
            getattr(response, "model", None),
            getattr(response, "model_version", None),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return "1.0"
