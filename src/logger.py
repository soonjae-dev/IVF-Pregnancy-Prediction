"""
===============================================================================
Logging Configuration (src/logger.py)
===============================================================================
This module configures the standard logging for the pipeline.
It outputs logs to both the console and a local log file with timestamps.
"""

import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = "logs"):
    """
    Set up the project logger to write to console and a log file.
    """
    # Automatically create a directory to save logs
    os.makedirs(log_dir, exist_ok=True)
    
    # Use the execution timestamp as the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("IVF_Pipeline")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate outputs from multiple logger calls
    if not logger.handlers:
        # 1. Console output handler
        c_handler = logging.StreamHandler()
        # 2. File output handler
        f_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # Specify log format: [Time] [Level] Message
        log_format = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger

# Create a global instance to be imported by other modules
logger = setup_logger()
