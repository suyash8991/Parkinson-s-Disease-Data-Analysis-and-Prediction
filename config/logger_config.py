# logger_config.py

import logging
import os
from datetime import datetime
import traceback as tb

class Logger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Logger._initialized:
            # Create logs directory if it doesn't exist
            logs_dir = 'logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            # Create log filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = os.path.join(logs_dir, f'parkinson_analysis_{timestamp}.log')

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filename),
                    logging.StreamHandler()
                ]
            )

            self.logger = logging.getLogger('ParkinsonsAnalysis')
            Logger._initialized = True

    def get_logger(self):
        return self.logger

def log_decorator(func):
    """Decorator to add logging to any function"""
    def wrapper(*args, **kwargs):
        logger = Logger().get_logger()
        try:
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}\n{tb.format_exc()}")
            raise
    return wrapper