"""Storage and retrieval operations for story embeddings."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from supabase import Client

from ..models import StoryEmbedding
from ..storage.manager import StorageManager

logger = logging.getLogger(__name__)


class EmbeddingStorageManager:
    """Manage storage and retrieval of story embeddings in Supabase.
    
    Extends existing storage patterns to handle embedding-specific operations
    including version tracking and efficient batch operations.
    """

    def __init__(self, supabase_client: Client):
        """Initialize with Supabase client.
        
        Args:
            supabase_client: Authenticated Supabase client
        """
        self.supabase = supabase_client
        self.table_name = "story_embeddings"
        
        logger.info("EmbeddingStorageManager initialized")

    async def store_embedding(self, embedding: StoryEmbedding) -> bool:
        """Store a single story embedding in the database.
        
        Args:
            embedding: The StoryEmbedding to store
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If embedding validation fails
            Exception: For database operation errors
        """
        try:
            # Validate embedding before storage
            embedding.validate()
            
            # Check if embedding already exists for this news_url_id
            existing = await self.get_embedding_by_url_id(embedding.news_url_id)
            if existing:
                logger.info(f"Updating existing embedding for news_url_id: {embedding.news_url_id}")
                return await self.update_embedding(embedding)
            
            # Prepare data for insertion
            insert_data = embedding.to_db()
            
            # Remove id if it's None to let database generate it
            if insert_data.get("id") is None:
                insert_data.pop("id", None)
            
            # Insert into database
            response = self.supabase.table(self.table_name).insert(insert_data).execute()
            
            if response.data:
                logger.debug(f"Successfully stored embedding for news_url_id: {embedding.news_url_id}")
                return True
            else:
                logger.error(f"No data returned when storing embedding for {embedding.news_url_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store embedding for {embedding.news_url_id}: {e}")
            raise

    async def store_embeddings_batch(self, embeddings: List[StoryEmbedding]) -> Tuple[int, int]:
        """Store multiple embeddings efficiently using batch operations.
        
        Args:
            embeddings: List of StoryEmbedding objects to store
            
        Returns:
            Tuple of (successful_count, failed_count)
            
        Raises:
            ValueError: If embeddings list is empty
        """
        if not embeddings:
            raise ValueError("embeddings list cannot be empty")
            
        successful_count = 0
        failed_count = 0
        
        # Prepare batch data
        insert_data = []
        update_data = []
        
        # Check which embeddings already exist
        url_ids = [emb.news_url_id for emb in embeddings]
        existing_embeddings = await self.get_embeddings_by_url_ids(url_ids)
        existing_url_ids = {emb.news_url_id for emb in existing_embeddings}
        
        for embedding in embeddings:
            try:
                embedding.validate()
                db_data = embedding.to_db()
                
                # Remove id if None to let database generate it
                if db_data.get("id") is None:
                    db_data.pop("id", None)
                
                if embedding.news_url_id in existing_url_ids:
                    update_data.append(db_data)
                else:
                    insert_data.append(db_data)
                    
            except Exception as e:
                logger.error(f"Validation failed for embedding {embedding.news_url_id}: {e}")
                failed_count += 1
        
        # Perform batch insert for new embeddings
        if insert_data:
            try:
                response = self.supabase.table(self.table_name).insert(insert_data).execute()
                if response.data:
                    successful_count += len(response.data)
                    logger.info(f"Batch inserted {len(response.data)} embeddings")
                else:
                    failed_count += len(insert_data)
                    logger.error("Batch insert returned no data")
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                failed_count += len(insert_data)
        
        # Update existing embeddings individually (Supabase doesn't support batch upsert easily)
        for update_item in update_data:
            try:
                response = self.supabase.table(self.table_name).update(update_item).eq(
                    "news_url_id", update_item["news_url_id"]
                ).execute()
                
                if response.data:
                    successful_count += 1
                else:
                    failed_count += 1
                    logger.error(f"Update failed for {update_item['news_url_id']}")
                    
            except Exception as e:
                logger.error(f"Update failed for {update_item['news_url_id']}: {e}")
                failed_count += 1
        
        logger.info(f"Batch operation completed: {successful_count} successful, {failed_count} failed")
        return successful_count, failed_count

    async def get_embedding_by_url_id(self, news_url_id: str) -> Optional[StoryEmbedding]:
        """Retrieve embedding for a specific news URL.
        
        Args:
            news_url_id: The news URL ID to look up
            
        Returns:
            StoryEmbedding if found, None otherwise
        """
        try:
            response = self.supabase.table(self.table_name).select("*").eq(
                "news_url_id", news_url_id
            ).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                return StoryEmbedding.from_db(response.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding for {news_url_id}: {e}")
            raise

    async def get_embeddings_by_url_ids(self, news_url_ids: List[str]) -> List[StoryEmbedding]:
        """Retrieve embeddings for multiple news URLs efficiently.
        
        Args:
            news_url_ids: List of news URL IDs to look up
            
        Returns:
            List of StoryEmbedding objects found
        """
        if not news_url_ids:
            return []
            
        try:
            response = self.supabase.table(self.table_name).select("*").in_(
                "news_url_id", news_url_ids
            ).execute()
            
            embeddings = []
            if response.data:
                for row in response.data:
                    try:
                        embeddings.append(StoryEmbedding.from_db(row))
                    except Exception as e:
                        logger.error(f"Failed to parse embedding from DB row: {e}")
                        continue
                        
            logger.debug(f"Retrieved {len(embeddings)} embeddings for {len(news_url_ids)} requested URLs")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings for multiple URLs: {e}")
            raise

    async def get_embeddings_by_model(
        self, 
        model_name: str, 
        model_version: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StoryEmbedding]:
        """Retrieve embeddings generated by a specific model version.
        
        Args:
            model_name: Name of the embedding model
            model_version: Specific version (optional)
            limit: Maximum number of results (optional)
            
        Returns:
            List of StoryEmbedding objects matching criteria
        """
        try:
            query = self.supabase.table(self.table_name).select("*").eq("model_name", model_name)
            
            if model_version:
                query = query.eq("model_version", model_version)
                
            if limit:
                query = query.limit(limit)
                
            response = query.order("created_at", desc=True).execute()
            
            embeddings = []
            if response.data:
                for row in response.data:
                    try:
                        embeddings.append(StoryEmbedding.from_db(row))
                    except Exception as e:
                        logger.error(f"Failed to parse embedding from DB row: {e}")
                        continue
                        
            logger.debug(f"Retrieved {len(embeddings)} embeddings for model {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings by model {model_name}: {e}")
            raise

    async def update_embedding(self, embedding: StoryEmbedding) -> bool:
        """Update an existing embedding.
        
        Args:
            embedding: The updated StoryEmbedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding.validate()
            update_data = embedding.to_db()
            
            # Remove id from update data - we'll use news_url_id as the key
            update_data.pop("id", None)
            
            response = self.supabase.table(self.table_name).update(update_data).eq(
                "news_url_id", embedding.news_url_id
            ).execute()
            
            if response.data:
                logger.debug(f"Successfully updated embedding for news_url_id: {embedding.news_url_id}")
                return True
            else:
                logger.error(f"No data returned when updating embedding for {embedding.news_url_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update embedding for {embedding.news_url_id}: {e}")
            raise

    async def delete_embedding(self, news_url_id: str) -> bool:
        """Delete an embedding by news URL ID.
        
        Args:
            news_url_id: The news URL ID of the embedding to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.supabase.table(self.table_name).delete().eq(
                "news_url_id", news_url_id
            ).execute()
            
            if response.data:
                logger.info(f"Successfully deleted embedding for news_url_id: {news_url_id}")
                return True
            else:
                logger.warning(f"No embedding found to delete for news_url_id: {news_url_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete embedding for {news_url_id}: {e}")
            raise

    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings.
        
        Returns:
            Dictionary with embedding statistics
        """
        try:
            # Total count
            count_response = self.supabase.table(self.table_name).select("id", count="exact").execute()
            total_count = count_response.count if hasattr(count_response, 'count') else 0
            
            # Model distribution
            models_response = self.supabase.table(self.table_name).select(
                "model_name, model_version"
            ).execute()
            
            model_counts = {}
            if models_response.data:
                for row in models_response.data:
                    model_key = f"{row['model_name']}:{row['model_version']}"
                    model_counts[model_key] = model_counts.get(model_key, 0) + 1
            
            # Recent embeddings count (last 24 hours)
            from datetime import datetime, timedelta
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            recent_response = self.supabase.table(self.table_name).select(
                "id", count="exact"
            ).gte("created_at", yesterday).execute()
            recent_count = recent_response.count if hasattr(recent_response, 'count') else 0
            
            stats = {
                "total_embeddings": total_count,
                "recent_embeddings_24h": recent_count,
                "model_distribution": model_counts,
                "table_name": self.table_name
            }
            
            logger.debug(f"Retrieved embedding stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding stats: {e}")
            raise

    async def cleanup_old_embeddings(
        self, 
        days_old: int = 30, 
        keep_latest_per_url: bool = True
    ) -> int:
        """Clean up old embedding versions while preserving latest.
        
        Args:
            days_old: Delete embeddings older than this many days
            keep_latest_per_url: Whether to keep the latest embedding per URL
            
        Returns:
            Number of embeddings deleted
        """
        try:
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
            
            if keep_latest_per_url:
                # Complex query to keep latest per news_url_id
                # For now, just log and return 0 - this would need more sophisticated logic
                logger.info(f"Cleanup with keep_latest_per_url=True not yet implemented")
                return 0
            else:
                # Simple deletion of old embeddings
                response = self.supabase.table(self.table_name).delete().lt(
                    "created_at", cutoff_date
                ).execute()
                
                deleted_count = len(response.data) if response.data else 0
                logger.info(f"Cleaned up {deleted_count} old embeddings")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old embeddings: {e}")
            raise

    async def verify_embedding_integrity(self, news_url_id: str) -> Dict[str, Any]:
        """Verify the integrity of a stored embedding.
        
        Args:
            news_url_id: The news URL ID to verify
            
        Returns:
            Dictionary with verification results
        """
        try:
            embedding = await self.get_embedding_by_url_id(news_url_id)
            if not embedding:
                return {"exists": False, "valid": False, "error": "Embedding not found"}
            
            try:
                embedding.validate()
                
                # Additional checks
                vector_length = len(embedding.embedding_vector)
                vector_norm = sum(x*x for x in embedding.embedding_vector) ** 0.5
                
                return {
                    "exists": True,
                    "valid": True,
                    "vector_length": vector_length,
                    "vector_norm": vector_norm,
                    "model_name": embedding.model_name,
                    "model_version": embedding.model_version,
                    "confidence_score": embedding.confidence_score
                }
                
            except Exception as validation_error:
                return {
                    "exists": True,
                    "valid": False,
                    "error": str(validation_error)
                }
                
        except Exception as e:
            logger.error(f"Failed to verify embedding integrity for {news_url_id}: {e}")
            return {"exists": False, "valid": False, "error": str(e)}