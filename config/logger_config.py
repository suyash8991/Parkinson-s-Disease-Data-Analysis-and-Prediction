# logger_config.py

import logging
import os
from datetime import datetime
import traceback as tb
import optuna

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

            # Create log filename with timestamp for main logs
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # log_filename = os.path.join(logs_dir, f'parkinson_analysis_{timestamp}.log')
            log_filename = os.path.join(logs_dir, f'parkinson_analysis.log')

            # Configure main logging
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

            # Setup Optuna logging in a separate file
            self._setup_optuna_logging(logs_dir, timestamp)

    def _setup_optuna_logging(self, logs_dir, timestamp):
        """Setup a separate logger for Optuna that doesn't output to console"""
        optuna_log_filename = os.path.join(logs_dir, f'optuna_logs.log')
        
        # Create a file handler for Optuna logs
        optuna_file_handler = logging.FileHandler(optuna_log_filename)
        optuna_file_handler.setLevel(logging.INFO)
        optuna_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Get Optuna logger and configure it
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        optuna_logger.handlers.clear()
        
        # Add only file handler
        optuna_logger.addHandler(optuna_file_handler)
        
        # Prevent propagation to root logger
        optuna_logger.propagate = False
        
        # Disable default Optuna handler and enable our custom logging
        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce verbosity
    
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
