"""Logging utility functions and configurations."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", 
                 format_string: Optional[str] = None,
                 include_timestamp: bool = True) -> None:
    """Set up centralized logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        include_timestamp: Whether to include timestamp in logs
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    # Configure basic logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to appropriate levels
    setup_library_logging()


def setup_library_logging() -> None:
    """Configure logging for third-party libraries."""
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('supabase').setLevel(logging.WARNING)
    
    # NFL data library can be verbose
    logging.getLogger('nfl_data_py').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with standardized configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
