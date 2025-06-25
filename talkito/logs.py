"""Centralized logging configuration for talkito - provides DRY logging setup across all modules."""
import logging
import sys
from pathlib import Path
from typing import Optional

# Global state for logging configuration
_log_enabled = False
_log_file = None
_is_configured = False
_original_stderr = None
_stderr_file = None

def setup_logging(log_file_path: Optional[str] = None, mode: str = 'w') -> None:
    """Set up centralized logging configuration with optional file output."""
    global _log_enabled, _log_file, _is_configured
    
    if _is_configured:
        return
    
    # Enable logging if log file path is provided
    if log_file_path:
        _log_enabled = True
        _log_file = Path(log_file_path)
    
    if not _log_enabled:
        return
    
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers on the root logger to prevent console output
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Configure file handler
    file_handler = logging.FileHandler(str(_log_file), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # Create a custom formatter that includes milliseconds
    class MillisecondFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            from datetime import datetime
            ct = datetime.fromtimestamp(record.created)
            if datefmt:
                s = ct.strftime(datefmt)[:-3]  # Remove last 3 digits to get milliseconds
            else:
                s = ct.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            return s
    
    formatter = MillisecondFormatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)
    
    # Suppress noisy AWS boto3 logs
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Redirect stderr to separate .err file if requested
    # _redirect_stderr(log_file_path)
    
    _is_configured = True
    
    # Log initial setup message
    logger = get_logger(__name__)
    logger.info("Logging initialized")

def _redirect_stderr(log_file_path: str) -> None:
    """Redirect stderr to separate .err file to capture system errors."""
    global _original_stderr, _stderr_file
    
    if log_file_path and _original_stderr is None:
        try:
            # Create separate .err file for stderr
            err_file_path = log_file_path.replace('.log', '.err')
            if not err_file_path.endswith('.err'):
                err_file_path += '.err'
            
            # Open err file in append mode for stderr
            _stderr_file = open(err_file_path, 'a')
            # Save original stderr
            _original_stderr = sys.stderr
            # Redirect stderr
            sys.stderr = _stderr_file
            logger = get_logger(__name__)
            logger.info(f"Redirected stderr to {err_file_path}")
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to redirect stderr: {e}")

def restore_stderr() -> None:
    """Restore original stderr."""
    global _original_stderr, _stderr_file
    
    if _original_stderr is not None:
        sys.stderr = _original_stderr
        _original_stderr = None
        
    if _stderr_file is not None:
        _stderr_file.close()
        _stderr_file = None

def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module name."""
    return logging.getLogger(name)

def log_message(level: str, message: str, logger_name: Optional[str] = None) -> None:
    """Log a message with custom level handling (supports BUFFER and FILTER levels)."""
    if not _log_enabled:
        return
        
    logger = get_logger(logger_name or __name__)
    
    # Map custom levels to standard ones
    if level == "BUFFER":
        logger.info(f"[BUFFER] {message}")
    elif level == "FILTER":
        logger.debug(f"[FILTER] {message}")
    elif level.upper() == "DEBUG":
        logger.debug(message)
    else:
        getattr(logger, level.lower(), logger.info)(message)

def is_logging_enabled() -> bool:
    """Check if logging is enabled."""
    return _log_enabled

def log_debug(message_func, logger_name: Optional[str] = None) -> None:
    """Log a debug message using lazy evaluation - message_func is only called if logging is enabled."""
    if not _log_enabled:
        return
    log_message("DEBUG", message_func(), logger_name)

def get_log_file() -> Optional[Path]:
    """Get the current log file path."""
    return _log_file