#!/usr/bin/env python3
"""
Story Grouping Batch Processing Utility

This script provides batch processing capabilities for story grouping,
specifically designed for backfilling existing stories that haven't been
processed for similarity grouping yet.

Usage:
    python scripts/story_grouping_batch_processor.py --help
    python scripts/story_grouping_batch_processor.py --dry-run --batch-size 25
    python scripts/story_grouping_batch_processor.py --resume-from story_id_123
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path (robust to nesting)
def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists() and (p / "README.md").exists():
            return p
    return start.parents[0]

ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StoryGroupingBatchProcessor:
    """Handles batch processing of stories for similarity grouping."""
    
    def __init__(self, storage_manager, dry_run: bool = False):
        self.storage = storage_manager
        self.dry_run = dry_run
        self.orchestrator = None
        
    async def initialize_orchestrator(self):
        """Initialize the story grouping orchestrator."""
        try:
            from src.nfl_news_pipeline.orchestrator.story_grouping import (
                StoryGroupingOrchestrator,
                StoryGroupingSettings,
            )
            from src.nfl_news_pipeline.group_manager import GroupManager
            from src.nfl_news_pipeline.embedding import EmbeddingGenerator, EmbeddingErrorHandler
            from src.nfl_news_pipeline.similarity import SimilarityCalculator
            from src.nfl_news_pipeline.story_grouping import URLContextExtractor
            
            # Conservative settings for batch processing
            settings = StoryGroupingSettings(
                max_parallelism=2,  # Lower parallelism for batch processing
                max_candidates=5,
                candidate_similarity_floor=0.35,
                prioritize_recent=False,  # Process in order for backfill
                reprocess_existing=False,
            )
            settings.validate()
            
            # Create components
            context_extractor = URLContextExtractor()
            embedding_generator = EmbeddingGenerator()
            similarity_calculator = SimilarityCalculator()
            error_handler = EmbeddingErrorHandler()
            group_manager = GroupManager(self.storage)
            
            # Create orchestrator
            self.orchestrator = StoryGroupingOrchestrator(
                context_extractor=context_extractor,
                embedding_generator=embedding_generator,
                group_manager=group_manager,
                similarity_calculator=similarity_calculator,
                error_handler=error_handler,
                settings=settings,
            )
            
            logger.info("Story grouping orchestrator initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import story grouping components: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def get_unprocessed_stories(self, batch_size: int, resume_from: Optional[str] = None) -> Tuple[List, Dict[str, str]]:
        """Get a batch of stories that haven't been processed for grouping yet."""
        # This is a placeholder implementation
        # In a real implementation, this would:
        # 1. Query the database for news items that don't have corresponding story_embeddings
        # 2. Handle the resume_from parameter to skip already processed items
        # 3. Return ProcessedNewsItem objects and their URL-to-ID mapping
        
        logger.info(f"Fetching batch of {batch_size} unprocessed stories")
        if resume_from:
            logger.info(f"Resuming from story ID: {resume_from}")
            
        # Placeholder return
        stories = []
        url_id_map = {}
        
        return stories, url_id_map
    
    async def process_batch(self, stories: List, url_id_map: Dict[str, str]) -> Dict:
        """Process a batch of stories for grouping."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")
            
        if self.dry_run:
            logger.info(f"DRY RUN: Would process {len(stories)} stories")
            # Return mock results for dry run
            return {
                "total_stories": len(stories),
                "processed_stories": len(stories),
                "skipped_stories": 0,
                "new_groups_created": max(1, len(stories) // 3),
                "existing_groups_updated": max(1, len(stories) // 4),
                "processing_time_ms": 1000,
                "dry_run": True
            }
        
        # Process the batch
        result = await self.orchestrator.process_batch(stories, url_id_map)
        
        return {
            "total_stories": result.metrics.total_stories,
            "processed_stories": result.metrics.processed_stories,
            "skipped_stories": result.metrics.skipped_stories,
            "new_groups_created": result.metrics.new_groups_created,
            "existing_groups_updated": result.metrics.existing_groups_updated,
            "processing_time_ms": result.metrics.total_processing_time_ms,
            "dry_run": False
        }
    
    async def run_backfill(self, 
                          batch_size: int = 50, 
                          max_batches: Optional[int] = None,
                          resume_from: Optional[str] = None,
                          progress_file: Optional[str] = None) -> Dict:
        """Run the complete backfill process."""
        start_time = time.time()
        total_processed = 0
        total_batches = 0
        total_new_groups = 0
        total_updated_groups = 0
        
        logger.info(f"Starting story grouping backfill (batch_size={batch_size}, max_batches={max_batches})")
        
        # Initialize orchestrator
        if not await self.initialize_orchestrator():
            return {"error": "Failed to initialize orchestrator"}
        
        current_resume_point = resume_from
        
        while True:
            # Check batch limit
            if max_batches and total_batches >= max_batches:
                logger.info(f"Reached maximum batch limit ({max_batches})")
                break
            
            # Get next batch
            try:
                stories, url_id_map = await self.get_unprocessed_stories(batch_size, current_resume_point)
            except Exception as e:
                logger.error(f"Failed to fetch stories: {e}")
                break
            
            # Check if we're done
            if not stories:
                logger.info("No more unprocessed stories found")
                break
            
            logger.info(f"Processing batch {total_batches + 1} with {len(stories)} stories")
            
            # Process batch
            try:
                batch_result = await self.process_batch(stories, url_id_map)
                
                total_processed += batch_result["processed_stories"]
                total_new_groups += batch_result["new_groups_created"]
                total_updated_groups += batch_result["existing_groups_updated"]
                total_batches += 1
                
                logger.info(f"Batch {total_batches} completed: "
                           f"processed={batch_result['processed_stories']}, "
                           f"new_groups={batch_result['new_groups_created']}, "
                           f"updated_groups={batch_result['existing_groups_updated']}")
                
                # Update resume point for next iteration
                # In a real implementation, this would be the ID of the last processed story
                current_resume_point = None
                
                # Save progress if requested
                if progress_file:
                    await self.save_progress(progress_file, {
                        "batches_completed": total_batches,
                        "stories_processed": total_processed,
                        "last_resume_point": current_resume_point,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Failed to process batch {total_batches + 1}: {e}")
                break
            
            # Brief pause between batches to avoid overwhelming the system
            await asyncio.sleep(1)
        
        duration = time.time() - start_time
        
        summary = {
            "total_batches": total_batches,
            "total_stories_processed": total_processed,
            "total_new_groups": total_new_groups,
            "total_updated_groups": total_updated_groups,
            "duration_seconds": duration,
            "average_batch_time": duration / max(1, total_batches),
            "stories_per_second": total_processed / max(1, duration)
        }
        
        logger.info(f"Backfill completed: {json.dumps(summary, indent=2)}")
        return summary
    
    async def save_progress(self, progress_file: str, progress_data: Dict):
        """Save progress to a file for resume capability."""
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            logger.debug(f"Progress saved to {progress_file}")
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")


def build_storage_manager(dry_run: bool):
    """Build a storage manager for the batch processor."""
    if dry_run:
        # Use a mock storage manager for dry runs
        class MockStorageManager:
            pass
        return MockStorageManager()
    
    # In a real implementation, this would build the actual StorageManager
    # with database connectivity
    try:
        from src.core.db.database_init import get_supabase_client
        from src.nfl_news_pipeline.storage import StorageManager
        
        client = get_supabase_client()
        if not client:
            raise RuntimeError("Could not initialize Supabase client")
        return StorageManager(client)
        
    except Exception as e:
        logger.error(f"Failed to build storage manager: {e}")
        return None


async def main():
    """Main entry point for the batch processor."""
    parser = argparse.ArgumentParser(description="Story Grouping Batch Processor")
    parser.add_argument("--batch-size", type=int, default=50, 
                       help="Number of stories per batch (default: 50)")
    parser.add_argument("--max-batches", type=int, 
                       help="Maximum number of batches to process (default: unlimited)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be processed without making changes")
    parser.add_argument("--resume-from", 
                       help="Resume processing from specific story ID")
    parser.add_argument("--progress-file", 
                       help="File to save progress for resume capability")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build storage manager
    storage = build_storage_manager(args.dry_run)
    if storage is None and not args.dry_run:
        logger.error("Failed to initialize storage manager")
        return 1
    
    # Create and run batch processor
    processor = StoryGroupingBatchProcessor(storage, dry_run=args.dry_run)
    
    try:
        summary = await processor.run_backfill(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            resume_from=args.resume_from,
            progress_file=args.progress_file
        )
        
        if "error" in summary:
            logger.error(f"Backfill failed: {summary['error']}")
            return 1
        
        print("\n" + "="*50)
        print("BACKFILL SUMMARY")
        print("="*50)
        print(f"Total Batches: {summary['total_batches']}")
        print(f"Total Stories Processed: {summary['total_stories_processed']}")
        print(f"New Groups Created: {summary['total_new_groups']}")
        print(f"Existing Groups Updated: {summary['total_updated_groups']}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Average Batch Time: {summary['average_batch_time']:.1f} seconds")
        print(f"Processing Rate: {summary['stories_per_second']:.1f} stories/second")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Backfill failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
