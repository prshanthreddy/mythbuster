"""Logging configuration for MythBuster AI."""

import logging
import sys
from pathlib import Path
from config import config

def setup_logging() -> logging.Logger:
    """Set up application logging."""
    
    # Create logger
    logger = logging.getLogger("mythbuster")
    logger.setLevel(getattr(logging, config.log_level))
    
    # Avoid duplicate logs
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    log_path = Path(config.log_file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logging()