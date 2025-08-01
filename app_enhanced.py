#!/usr/bin/env python3
"""
🕵️ MythBuster AI - Enhanced Version

An intelligent myth-busting application that combines vector search,
web research, and AI analysis to fact-check claims and beliefs.

Author: Enhanced by AI Assistant
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from logger import logger
from ui import MythBusterUI

def main():
    """Main application entry point."""
    try:
        logger.info("🚀 Starting MythBuster AI Enhanced...")
        logger.info(f"Configuration loaded: Vector store exists: {config.vector_store_exists}")
        logger.info(f"Image generation available: {config.has_image_generation}")
        
        # Create and launch UI
        app = MythBusterUI()
        app.launch()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()