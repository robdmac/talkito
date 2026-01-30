"""Centralized logging configuration for talkito - provides DRY logging setup across all modules."""
import logging
import os
import sys
import builtins
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Suppress AWS credential messages globally (even before logging is configured)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)

# Global state for logging configuration
_log_enabled = False
_log_file = None
_is_configured = False
_original_stderr = None
_stderr_file = None
_original_print = None

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
    logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Suppress websocket and AssemblyAI debug messages
    logging.getLogger('websocket').setLevel(logging.ERROR)
    logging.getLogger('websockets').setLevel(logging.ERROR)
    logging.getLogger('websocket.client').setLevel(logging.ERROR)
    logging.getLogger('websockets.client').setLevel(logging.ERROR)
    logging.getLogger('websockets.protocol').setLevel(logging.ERROR)
    logging.getLogger('assemblyai').setLevel(logging.INFO)
    logging.getLogger('assemblyai.streaming').setLevel(logging.INFO)
    logging.getLogger('assemblyai.websocket').setLevel(logging.ERROR)
    
    # Suppress noisy uvicorn/FastMCP logs
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.error').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('fastmcp').setLevel(logging.WARNING)
    logging.getLogger('mcp').setLevel(logging.WARNING)
    logging.getLogger('starlette').setLevel(logging.WARNING)
    logging.getLogger('anyio').setLevel(logging.WARNING)
    
    # Suppress Google Speech Recognition related logs
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger('grpc').setLevel(logging.ERROR)
    logging.getLogger('grpc._channel').setLevel(logging.ERROR)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('google.auth').setLevel(logging.WARNING)
    logging.getLogger('google.cloud').setLevel(logging.WARNING)
    
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

    if level in ["ERROR", "CRITICAL"]:
        if is_orcabot_mode():
            emit_orcabot_notice(message, level=level.lower())
        else:
            print(message)

    if not _log_enabled:
        return

    logger = get_logger(logger_name or __name__)
    
    # Get caller information
    import inspect
    frame = inspect.currentframe()
    try:
        # Go up the call stack to find the actual caller
        caller_frame = frame.f_back
        
        # If the caller is a wrapper function (like in update.py), go one more level up
        if caller_frame and caller_frame.f_code.co_name == 'log_message':
            caller_frame = caller_frame.f_back
        
        if caller_frame:
            filename = caller_frame.f_code.co_filename
            line_number = caller_frame.f_lineno
            # Get just the filename without the full path
            filename = filename.split('/')[-1]
            caller_info = f"{filename}:{line_number}"
            
            # Format message with caller info
            formatted_message = f"[{caller_info}] {message}"
        else:
            # Fallback if we can't get caller info
            formatted_message = message
        
        # Map custom levels to standard ones
        if level == "BUFFER":
            logger.info(f"[BUFFER] {formatted_message}")
        elif level == "FILTER":
            logger.debug(f"[FILTER] {formatted_message}")
        elif level.upper() == "DEBUG":
            logger.debug(formatted_message)
        else:
            getattr(logger, level.lower(), logger.info)(formatted_message)
    finally:
        del frame

def is_logging_enabled() -> bool:
    """Check if logging is enabled."""
    return _log_enabled

def get_log_file() -> Optional[Path]:
    """Get the current log file path."""
    return _log_file

def is_orcabot_mode() -> bool:
    """Check whether TalkiTo is running in Orcabot mode."""
    if os.environ.get("TALKITO_ORCABOT_ENABLED") == "1":
        return True
    if os.environ.get("TALKITO_ORCABOT_MODE") == "1":
        return True
    return bool(os.environ.get("ORCABOT_SESSION_ID") and os.environ.get("ORCABOT_PTY_ID"))

def _emit_orcabot_status(payload: dict) -> bool:
    session_id = os.environ.get("ORCABOT_SESSION_ID")
    pty_id = os.environ.get("ORCABOT_PTY_ID")
    if not session_id or not pty_id:
        return False

    url = f"http://localhost:8081/sessions/{session_id}/ptys/{pty_id}/status"
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False

def emit_orcabot_notice(message: str, level: str = "info", category: Optional[str] = None) -> bool:
    """Send a structured notice to Orcabot instead of printing to console."""
    if not is_orcabot_mode():
        return False
    payload = {
        "action": "notice",
        "level": level,
        "message": message,
    }
    if category:
        payload["category"] = category
    return _emit_orcabot_status(payload)

def install_orcabot_print_hook() -> None:
    """Route stdout/stderr print calls to Orcabot in Orcabot mode."""
    global _original_print
    if _original_print is not None:
        return

    if not is_orcabot_mode():
        return

    _original_print = builtins.print

    def _orcabot_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        file = kwargs.get("file", sys.stdout)
        if file not in (sys.stdout, sys.stderr):
            return _original_print(*args, **kwargs)
        message = sep.join(str(a) for a in args)
        if end and end != "\n":
            message += end
        level = "error" if file is sys.stderr else "info"
        emit_orcabot_notice(message, level=level)

    builtins.print = _orcabot_print

def restore_print_hook() -> None:
    """Restore the original print function if it was overridden."""
    global _original_print
    if _original_print is None:
        return
    builtins.print = _original_print
    _original_print = None
