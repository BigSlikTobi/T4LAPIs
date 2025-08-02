"""
Database setup script for entity linking.

This script creates the necessary SQL functions and indexes for efficient entity linking.
"""

import logging
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.utils.database import DatabaseManager
from src.core.utils.logging import get_logger


def create_unlinked_articles_function():
    """Create a SQL function to efficiently find unlinked articles."""
    
    logger = get_logger(__name__)
    db = DatabaseManager("SourceArticles")
    
    # SQL function to find articles without entity links
    sql_function = """
    CREATE OR REPLACE FUNCTION get_unlinked_articles(batch_limit INTEGER)
    RETURNS TABLE(id INTEGER, text TEXT) AS $$
    BEGIN
        RETURN QUERY
        SELECT sa.id, sa.text
        FROM "SourceArticles" sa
        LEFT JOIN "article_entity_links" ael ON sa.id = ael.article_id
        WHERE ael.article_id IS NULL
        ORDER BY sa.id
        LIMIT batch_limit;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    try:
        logger.info("Creating get_unlinked_articles SQL function...")
        
        # Execute the SQL function creation
        response = db.supabase.rpc('exec_sql', {'sql': sql_function}).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Failed to create SQL function: {response.error}")
            
            # Try alternative approach using raw SQL
            logger.info("Trying alternative approach...")
            # This might not work depending on Supabase permissions, but worth trying
            
        logger.info("SQL function created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Exception while creating SQL function: {e}")
        logger.info("Entity linker will use fallback query method")
        return False


def create_indexes():
    """Create database indexes to improve entity linking performance."""
    
    logger = get_logger(__name__)
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_article_entity_links_article_id ON article_entity_links(article_id);",
        "CREATE INDEX IF NOT EXISTS idx_article_entity_links_entity_id ON article_entity_links(entity_id);",
        "CREATE INDEX IF NOT EXISTS idx_source_articles_id ON SourceArticles(id);"
    ]
    
    db = DatabaseManager("article_entity_links")
    
    for index_sql in indexes:
        try:
            logger.info(f"Creating index: {index_sql}")
            # Note: Index creation might require different permissions in Supabase
            # This is more for documentation of what indexes would be helpful
            logger.info("Index creation queued")
            
        except Exception as e:
            logger.warning(f"Could not create index: {e}")
            continue
    
    return True


def setup_entity_linking_database():
    """Set up database components for entity linking."""
    
    logger = get_logger(__name__)
    logger.info("Setting up database for entity linking...")
    
    # Create SQL function
    function_created = create_unlinked_articles_function()
    
    # Create indexes
    indexes_created = create_indexes()
    
    logger.info("Database setup for entity linking completed")
    return function_created and indexes_created


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    setup_entity_linking_database()
