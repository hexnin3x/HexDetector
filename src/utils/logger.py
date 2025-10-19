"""
Logger Module for HexDetector

Provides comprehensive logging functionality for tracking execution,
debugging, and monitoring the HexDetector pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Try to import settings, fallback to defaults if not available
try:
    from ..config import settings
    LOG_LEVEL = getattr(logging, settings.LOG_LEVEL.upper())
    LOG_FORMAT = settings.LOG_FORMAT
    LOG_FILE = settings.LOG_FILE
    CONSOLE_LOG = settings.CONSOLE_LOG
    FILE_LOG = settings.FILE_LOG
except:
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = Path('logs/hexdetector.log')
    CONSOLE_LOG = True
    FILE_LOG = True


class Logger:
    """
    Advanced logging class for HexDetector with support for
    console and file logging, custom formatting, and error tracking.
    """
    
    _instances = {}
    
    def __new__(cls, name='HexDetector'):
        """Singleton pattern to reuse logger instances"""
        if name not in cls._instances:
            instance = super(Logger, cls).__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(self, name='HexDetector'):
        """
        Initialize the logger with console and file handlers.
        
        Parameters:
        name (str): Name of the logger
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        self.formatter = logging.Formatter(
            LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler
        if CONSOLE_LOG:
            self._add_console_handler()
        
        # Add file handler
        if FILE_LOG:
            self._add_file_handler()
        
        self._initialized = True
    
    def _add_console_handler(self):
        """Add console handler for logging to stdout"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add file handler for logging to file"""
        # Create logs directory if it doesn't exist
        log_dir = Path(LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
        except:
            file_handler = logging.FileHandler(LOG_FILE)
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def log_info(self, message):
        """
        Log an informational message.
        
        Parameters:
        message (str): The message to log
        """
        self.logger.info(message)
    
    def log_debug(self, message):
        """
        Log a debug message.
        
        Parameters:
        message (str): The message to log
        """
        self.logger.debug(message)
    
    def log_warning(self, message):
        """
        Log a warning message.
        
        Parameters:
        message (str): The message to log
        """
        self.logger.warning(message)
    
    def log_error(self, message, exception=None):
        """
        Log an error message with optional exception details.
        
        Parameters:
        message (str): The error message to log
        exception (Exception): Optional exception object
        """
        self.logger.error(message)
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
            self.logger.debug(traceback.format_exc())
    
    def log_critical(self, message):
        """
        Log a critical message.
        
        Parameters:
        message (str): The message to log
        """
        self.logger.critical(message)
    
    def log_exception(self, message):
        """
        Log an exception with full stack trace.
        
        Parameters:
        message (str): The message to log
        """
        self.logger.exception(message)
    
    def log_step(self, step_name, step_number=None):
        """
        Log the start of a processing step.
        
        Parameters:
        step_name (str): Name of the step
        step_number (int): Optional step number
        """
        separator = "=" * 60
        if step_number:
            message = f"\n{separator}\nSTEP {step_number}: {step_name}\n{separator}"
        else:
            message = f"\n{separator}\n{step_name}\n{separator}"
        self.logger.info(message)
    
    def log_progress(self, current, total, prefix='Progress'):
        """
        Log progress for long-running operations.
        
        Parameters:
        current (int): Current progress value
        total (int): Total value
        prefix (str): Prefix for the progress message
        """
        percentage = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"{prefix}: {current}/{total} ({percentage:.2f}%)")
    
    def log_metrics(self, metrics_dict):
        """
        Log a dictionary of metrics in a formatted way.
        
        Parameters:
        metrics_dict (dict): Dictionary of metric names and values
        """
        self.logger.info("Metrics:")
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_time(self, operation_name, duration_seconds):
        """
        Log the duration of an operation.
        
        Parameters:
        operation_name (str): Name of the operation
        duration_seconds (float): Duration in seconds
        """
        if duration_seconds < 60:
            time_str = f"{duration_seconds:.2f} seconds"
        elif duration_seconds < 3600:
            minutes = duration_seconds / 60
            time_str = f"{minutes:.2f} minutes"
        else:
            hours = duration_seconds / 3600
            time_str = f"{hours:.2f} hours"
        
        self.logger.info(f"{operation_name} completed in {time_str}")
    
    def log_separator(self, char='-', length=60):
        """
        Log a separator line.
        
        Parameters:
        char (str): Character to use for the separator
        length (int): Length of the separator
        """
        self.logger.info(char * length)


# Convenience functions for quick logging
def get_logger(name='HexDetector'):
    """Get or create a logger instance"""
    return Logger(name)


def log_info(message, logger_name='HexDetector'):
    """Quick info logging"""
    Logger(logger_name).log_info(message)


def log_error(message, exception=None, logger_name='HexDetector'):
    """Quick error logging"""
    Logger(logger_name).log_error(message, exception)


def log_warning(message, logger_name='HexDetector'):
    """Quick warning logging"""
    Logger(logger_name).log_warning(message)