#!/usr/bin/env python3
"""
Script to load player weekly stats data into the database.
Usage: python scripts/load_player_weekly_stats.py [--years 2024] [--weeks 1,2,3] [--batch-size 100]
"""

import argparse
import logging
import sys
import os
from typing import List

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.insert(0, src_dir)

from core.data.loaders.player_weekly_stats import PlayerWeeklyStatsDataLoader


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load player weekly stats data into the database"
    )
    
    parser.add_argument(
        "--years",
        type=str,
        default="2024",
        help="Comma-separated list of years to load (default: 2024)"
    )
    
    parser.add_argument(
        "--weeks",
        type=str,
        default=None,
        help="Comma-separated list of weeks to load (default: all weeks)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records to process in each batch (default: 100)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data, don't load new data"
    )
    
    return parser.parse_args()


def parse_list_arg(arg_string: str) -> List[int]:
    """Parse a comma-separated string into a list of integers."""
    if not arg_string:
        return None
    return [int(x.strip()) for x in arg_string.split(",")]


def main():
    """Main function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting player weekly stats data loading script")
    
    try:
        # Parse arguments
        years = parse_list_arg(args.years)
        weeks = parse_list_arg(args.weeks) if args.weeks else None
        
        logger.info(f"Target years: {years}")
        logger.info(f"Target weeks: {weeks}")
        logger.info(f"Batch size: {args.batch_size}")
        
        # Initialize loader
        loader = PlayerWeeklyStatsDataLoader()
        
        if args.validate_only:
            # Only validate existing data
            logger.info("Validation mode - checking existing data integrity")
            validation_results = loader.validate_data_integrity()
            logger.info(f"Validation results: {validation_results}")
        else:
            # Load new data
            results = loader.load_weekly_stats(
                years=years,
                weeks=weeks,
                batch_size=args.batch_size
            )
            
            # Print results summary
            logger.info("=== LOAD RESULTS SUMMARY ===")
            logger.info(f"Success: {results['success']}")
            logger.info(f"Total records processed: {results['total_records']}")
            logger.info(f"Successfully loaded: {results['loaded_records']}")
            logger.info(f"Failed records: {results['failed_records']}")
            
            if results['errors']:
                logger.warning(f"Errors encountered: {len(results['errors'])}")
                for error in results['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")
                if len(results['errors']) > 5:
                    logger.warning(f"  ... and {len(results['errors']) - 5} more errors")
            
            # Validate data after loading
            if results['success'] and results['loaded_records'] > 0:
                logger.info("Validating loaded data...")
                validation_results = loader.validate_data_integrity()
                logger.info(f"Post-load validation: {validation_results}")
            
            # Show final counts
            logger.info("=== FINAL DATABASE COUNTS ===")
            count = loader.get_stats_count()
            logger.info(f"Total weekly stats records: {count}")
        
        logger.info("Player weekly stats loading script completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
