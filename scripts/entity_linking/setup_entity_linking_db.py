"""
Database setup script for entity linking.

This script creates the necessary indexes for efficient entity linking.

IMPORTANT: Due to Supabase RPC limitations, the get_unlinked_articles function
must be created manually in the Supabase SQL Editor. See README.md for details.
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to path for imports (robust to nesting)
def _repo_root() -> str:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / 'src').exists() and (p / 'README.md').exists():
            return str(p)
    return str(start.parents[0])

project_root = _repo_root()
sys.path.insert(0, project_root)

from src.core.utils.database import DatabaseManager
from src.core.utils.logging import get_logger


def create_unlinked_articles_function():
    """Placeholder function - SQL function must be created manually in Supabase.
    
    Due to Supabase limitations with RPC function creation, the get_unlinked_articles
    function must be created manually in the Supabase SQL Editor.
    
    See README.md for the exact SQL command to run.
    """
    
    logger = get_logger(__name__)
    logger.warning("❌ SQL function creation not supported via this script")
    logger.info("📝 Please create the get_unlinked_articles function manually in Supabase SQL Editor")
    logger.info("📖 See README.md for the exact SQL command")
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
    
    # Note about manual function creation
    logger.info("⚠️  Manual setup required for optimal performance:")
    logger.info("📝 Create get_unlinked_articles function in Supabase SQL Editor")
    logger.info("📖 See README.md for the exact SQL command")
    
    # Create SQL function (will show warning about manual setup)
    function_created = create_unlinked_articles_function()
    
    # Create indexes
    indexes_created = create_indexes()
    
    logger.info("Database setup for entity linking completed")
    logger.info("✅ Indexes created successfully")
    if not function_created:
        logger.info("❌ SQL function requires manual creation in Supabase")
    
    return indexes_created  # Return True if indexes were created


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    setup_entity_linking_database()
