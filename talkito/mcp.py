#!/usr/bin/env python3

# TalkiTo - Universal TTS wrapper that works with any command
# Copyright (C) 2025 Robert Macrae
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""MCP Talk Server - Model Context Protocol server for TTS/ASR functionality"""

import sys
import time
from typing import Any
import asyncio
import os
import argparse
from types import SimpleNamespace
from fastmcp import FastMCP, Context
from mcp import types
import queue
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random
import traceback
from urllib.parse import urlparse

# Import talkito functionality
from . import asr, comms, tts
from .comms import SlackProvider, TwilioWhatsAppProvider
from .core import ensure_asr_initialized
from .logs import log_message as _base_log_message, setup_logging
from .state import get_shared_state, save_shared_state, get_status_summary
from .tts import AVAILABLE_VOICES, get_all_voices_for_provider, get_tts_config

# Global logging setup - set from command line args
_log_file_path = None

# Wrapper to add [MCP-SSE] prefix to all log messages
def log_message(level: str, message: str):
    """Log a message with [MCP-SSE] prefix"""
    global _log_file_path

    # If we have a log file path, ensure logging is enabled
    if _log_file_path:
        # Import is_logging_enabled to check current state
        from .logs import is_logging_enabled

        # Only setup if not already enabled
        if not is_logging_enabled():
            setup_logging(_log_file_path, mode='a')  # Use append mode to not overwrite

    # Also print important messages to stderr for debugging
    # if level in ["ERROR", "CRITICAL", "WARNING"]:
    #     print(f"[{level}] [MCP-SSE] {message}", file=sys.stderr)

    # Always try to log, even if logging might be disabled
    try:
        _base_log_message(level, f"[MCP-SSE] {message}", __name__)
    except Exception:
        pass

# Server configuration - Changed to include server name
app = FastMCP("talkito-sse-server")

# CORS headers for browser access
_cors_enabled = False

# Track if we're running for terminal agents (to mask certain tools)
_running_for_terminal_agent = False

# Track current voice index for cycling
_current_voice_index = {}

# Track if auto-skip TTS is enabled
_auto_skip_tts = False

def configure_mcp_server(cors_enabled=None, running_for_terminal_agent=None, log_file_path=None, auto_skip_tts=None):
    """Configure MCP server settings through a public interface"""
    global _cors_enabled, _running_for_terminal_agent, _log_file_path, _auto_skip_tts
    
    if cors_enabled is not None:
        _cors_enabled = cors_enabled
    
    if running_for_terminal_agent is not None:
        _running_for_terminal_agent = running_for_terminal_agent
    
    if log_file_path is not None:
        _log_file_path = log_file_path
        setup_logging(log_file_path)
    
    if auto_skip_tts is not None:
        _auto_skip_tts = auto_skip_tts

def add_cors_headers(headers):
    """Add CORS headers to response"""
    if _cors_enabled:
        headers['Access-Control-Allow-Origin'] = '*'
        headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        headers['Access-Control-Max-Age'] = '86400'

# HTTP API server for web access
_http_api_server = None
_http_api_thread = None
_http_api_port = None

# Store original log configuration to restore after FastMCP starts
_original_log_handlers = None
_original_log_level = None

def _save_logging_state():
    """Save the current logging state before FastMCP messes with it"""
    global _original_log_handlers, _original_log_level
    import logging
    
    root_logger = logging.getLogger()
    _original_log_handlers = root_logger.handlers.copy()
    _original_log_level = root_logger.level

def _restore_logging_state():
    """Restore our logging state after FastMCP has started"""
    global _original_log_handlers, _original_log_level, _log_file_path
    import logging
    
    if _log_file_path:
        log_message("INFO", f"[INFO] Restoring logging configuration to: {_log_file_path}")
        
        try:
            # Clear all handlers added by FastMCP/uvicorn
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            
            # Force complete re-setup of our logging
            import talkito.logs
            talkito.logs._is_configured = False
            talkito.logs._log_enabled = False
            
            from .logs import setup_logging
            setup_logging(_log_file_path, mode='a')
            
            log_message("INFO", "Logging restored after FastMCP startup")
            log_message("DEBUG", "Testing restored logging with debug message")
        except Exception as e:
            log_message("ERROR", f"[ERROR] Failed to restore logging: {e}")

# We'll restore logging in the first tool call instead of using on_event
_logging_restored = False

def _ensure_logging_restored():
    """Ensure logging has been restored after FastMCP startup"""
    global _logging_restored
    
    if not _logging_restored:
        _restore_logging_state()
        _logging_restored = True

# Add a flag to track if we should send notifications
_notifications_enabled = True

# Note: In FastMCP, notifications are sent via ctx.info() within tools.
# The notification queue stores pending notifications until a tool with context is called.

# Global state - these will be migrated to shared state
_shutdown_registered = False

# Track last spoken text to prevent duplicates
_last_spoken_text = None
_last_spoken_time = None

# ASR state
_last_dictated_text = ""
_dictation_callback_results = []

# Message tracking to prevent duplicates
_last_message_id = None  # Track the last processed message ID
_last_message_timestamp = None  # Track the last processed message timestamp

# Core instances
_core_instance = None
_comms_manager = None  # Communication manager for WhatsApp/Slack

# Communication modes - will use shared state
_whatsapp_recipient = None  # Current WhatsApp recipient
_slack_channel = None  # Current Slack channel

# Store the original callback functions to intercept messages
_original_dictation_callback = None
_original_slack_callback = None

# Shared state accessors
def _get_tts_initialized():
    """Get TTS initialization state from shared state"""
    return get_shared_state().tts_initialized

def _get_asr_initialized():
    """Get ASR initialization state from shared state"""
    return get_shared_state().asr_initialized

def _get_tts_enabled():
    """Get TTS enabled state from shared state"""
    return get_shared_state().tts_enabled

def _get_asr_enabled():
    """Get ASR enabled state from shared state"""
    return get_shared_state().asr_enabled

def _get_whatsapp_mode():
    """Get WhatsApp mode state from shared state"""
    return get_shared_state().whatsapp_mode_active

def _get_slack_mode():
    """Get Slack mode state from shared state"""
    return get_shared_state().slack_mode_active
_original_whatsapp_callback = None

# Thread-safe notification queue
_notification_queue = queue.Queue()
_notification_lock = threading.Lock()
_notification_processor_task = None

def _ensure_logging():
    """Ensure logging is still working - FastMCP/uvicorn may have broken it"""
    global _log_file_path
    
    if _log_file_path:
        from .logs import is_logging_enabled, setup_logging
        
        if not is_logging_enabled():
            # Force re-setup of logging
            import talkito.logs
            talkito.logs._is_configured = False
            setup_logging(_log_file_path, mode='a')
            log_message("WARNING", "Logging was broken, reinitialized")

# def _ensure_initialization():
#     """Ensure TTS system is initialized"""
#     # REMOVED: MCP should not handle initialization
#     # Initialization is handled by CLI/state.py and TTS system automatically
#     # when speak_text is called. MCP is a passive service that uses
#     # whatever configuration was already set up.
#     pass

def _cleanup():
    """Cleanup TTS and ASR resources"""
    global _notification_processor_task, _shutdown_requested, _comms_manager
    
    log_message("INFO", "Starting cleanup process")
    
    # Signal shutdown to all async tasks
    _shutdown_requested = True
    
    # Cancel notification processor task
    if _notification_processor_task:
        try:
            _notification_processor_task.cancel()
            _notification_processor_task = None
        except Exception as e:
            log_message("WARNING", f"Error canceling notification processor: {e}")
    
    shared_state = get_shared_state()
    
    # Stop ASR if initialized
    if shared_state.asr_initialized:
        try:
            asr.stop_dictation()
            shared_state.set_asr_initialized(False)
            log_message("INFO", "ASR stopped")
        except Exception as e:
            log_message("WARNING", f"Error stopping ASR: {e}")
    
    # Shutdown TTS if initialized
    if shared_state.tts_initialized:
        try:
            tts.shutdown_tts()
            shared_state.set_tts_initialized(False)
            log_message("INFO", "TTS stopped")
        except Exception as e:
            log_message("WARNING", f"Error stopping TTS: {e}")
    
    # Cleanup communication manager
    if _comms_manager:
        try:
            # If it's wrapped, get the underlying manager
            base_manager = _comms_manager.wrapped if hasattr(_comms_manager, 'wrapped') else _comms_manager
            
            # Stop webhook server if present
            if hasattr(base_manager, 'webhook_handler') and base_manager.webhook_handler:
                try:
                    base_manager.webhook_handler.stop()
                    log_message("INFO", "Webhook server stopped")
                except Exception as e:
                    log_message("WARNING", f"Error stopping webhook server: {e}")
            
            # Stop providers
            if hasattr(base_manager, 'providers'):
                for provider in base_manager.providers:
                    try:
                        if hasattr(provider, 'stop'):
                            provider.stop()
                        elif hasattr(provider, 'close'):
                            provider.close()
                    except Exception as e:
                        log_message("WARNING", f"Error stopping provider {type(provider).__name__}: {e}")
            
            _comms_manager = None
            log_message("INFO", "Communication manager cleaned up")
        except Exception as e:
            log_message("WARNING", f"Error cleaning up communication manager: {e}")
    
    log_message("INFO", "Cleanup process completed")

def _signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n[INFO] Shutting down gracefully...", file=sys.stderr)
    _cleanup()
    # Don't call sys.exit() here - let the server shutdown naturally
    # Instead, set a flag that the main loop can check
    global _shutdown_requested
    _shutdown_requested = True

# Track last seen messages for polling  
_last_seen_voice_count = 0
_last_seen_messages = set()

# Store pending notifications when no context available
_pending_notifications = []

# Shutdown flag for graceful exit
_shutdown_requested = False


# Tool that runs a notification loop within the request context
@app.tool()
async def start_notification_stream(ctx: Context, duration: int = 30, exit_on_first: bool = True) -> str:
    """
    Start streaming notifications for a specified duration.
    This tool maintains the request context and can send SSE notifications.
    
    Args:
        duration: How long to stream notifications in seconds (default 30)
        exit_on_first: Exit immediately after first notification (default True)
        
    Returns:
        Status message when streaming ends
    """
    # Ensure logging has been restored after FastMCP startup
    _ensure_logging_restored()
    
    # Ensure logging is working
    _ensure_logging()
    
    log_message("DEBUG", f"start_notification_stream called for {duration} seconds, exit_on_first={exit_on_first}")

    start_time = time.time()
    notifications_sent = 0

    try:
        # This runs within the tool's request context, so we can send notifications
        while time.time() - start_time < duration and not _shutdown_requested:
            # Check the notification queue
            try:
                notification = _notification_queue.get(timeout=0.1)
            except queue.Empty:
                # Check if we should exit due to shutdown
                if _shutdown_requested:
                    return "Notification stream interrupted by shutdown"
                await asyncio.sleep(0.1)
                continue

            notification_type = notification['type']
            data = notification['data']

            # Now we have request context and can send SSE notifications
            try:
                log_message("DEBUG", f"Sending SSE notification: {notification_type}")
                # Format notification for display
                if notification_type == "dictation":
                    msg = f"Voice: {data.get('text', '')}"
                elif notification_type == "slack":
                    msg = f"Slack ({data.get('channel', '')}): {data.get('text', '')}"
                elif notification_type == "whatsapp":
                    msg = f"WhatsApp: {data.get('text', '')}"
                else:
                    msg = f"{notification_type}: {data}"

                await ctx.info(msg)
                notifications_sent += 1
                log_message("DEBUG", "Successfully sent SSE notification!")

                # Exit immediately after first notification if requested
                if exit_on_first:
                    elapsed = time.time() - start_time
                    return f"Received notification after {elapsed:.1f}s: {msg}"
            except Exception as e:
                log_message("ERROR", f"Failed to send notification: {str(e)}")

    except Exception as e:
        log_message("ERROR", f"Error in notification stream: {str(e)}")
        return f"Notification stream error: {str(e)}"

    elapsed = time.time() - start_time
    return f"Notification stream ended after {elapsed:.1f}s, sent {notifications_sent} notifications"

# We don't need the notification processor anymore since notifications
# will be sent directly from the start_notification_stream tool
async def _process_notification_queue():
    """Deprecated - notifications are now sent via start_notification_stream tool"""
    pass

# Message poller to detect new messages
async def _poll_for_new_messages():
    """Poll for new messages and send notifications"""
    global _last_seen_voice_count, _last_seen_messages

    log_message("INFO", "Message poller started")

    while not _shutdown_requested:
        try:
            # Check for new voice messages
            if len(_dictation_callback_results) > _last_seen_voice_count:
                # Get new messages
                new_messages = _dictation_callback_results[_last_seen_voice_count:]
                _last_seen_voice_count = len(_dictation_callback_results)

                # Send notifications for each new message
                for msg in new_messages:
                    _queue_notification("dictation", {
                        "text": msg["text"],
                        "timestamp": msg["timestamp"]
                    })
                    log_message("INFO", f"Detected new voice message via polling: {msg['text']}")

            # Check for new Slack/WhatsApp messages
            if _comms_manager:
                messages = []
                # Peek at messages without consuming them
                # We need to check the input_queue directly
                target_manager = _comms_manager.wrapped if hasattr(_comms_manager, 'wrapped') else _comms_manager
                if hasattr(target_manager, 'input_queue'):
                    # Convert queue to list to peek without consuming
                    temp_messages = list(target_manager.input_queue.queue)
                    for msg in temp_messages:
                        # Create a unique ID for the message
                        msg_id = f"{msg.channel}:{msg.sender}:{msg.timestamp}:{msg.content[:20]}"
                        if msg_id not in _last_seen_messages:
                            _last_seen_messages.add(msg_id)
                            messages.append(msg)
                            log_message("DEBUG", f"Found new message: {msg.content[:50]}")
                else:
                    log_message("DEBUG", "No input_queue found on manager")

                # Send notifications for new messages
                for msg in messages:
                    if 'slack' in msg.channel.lower() or msg.channel.startswith('#'):
                        notification_type = "slack_message"
                    elif 'whatsapp' in msg.channel.lower():
                        notification_type = "whatsapp_message"
                    else:
                        notification_type = "message"

                    _queue_notification(notification_type, {
                        "text": msg.content,
                        "sender": msg.sender,
                        "channel": msg.channel,
                        "timestamp": msg.timestamp
                    })
                    log_message("INFO", f"Detected new {notification_type} via polling: {msg.content}")

            await asyncio.sleep(0.1)  # Poll every 100ms

        except Exception as e:
            log_message("ERROR", f"Error in message poller: {str(e)}")
            await asyncio.sleep(1)

    # Clean exit
    log_message("DEBUG", "Message poller stopped gracefully")

def _queue_notification(notification_type: str, data: dict):
    """Queue a notification to be sent from the main event loop"""
    with _notification_lock:
        _notification_queue.put({
            'type': notification_type,
            'data': data
        })
        log_message("DEBUG", f"Queued {notification_type} notification: {data}")

    # Try to ensure poller is running
    _ensure_message_poller()

# Start message poller when needed
def _ensure_message_poller():
    """Ensure the message poller is running"""
    global _notification_processor_task
    try:
        # Check if we have an event loop
        loop = asyncio.get_running_loop()

        # Create task if it doesn't exist or is done
        if _notification_processor_task is None or _notification_processor_task.done():
            _notification_processor_task = loop.create_task(_poll_for_new_messages())
            log_message("INFO", "Started message poller task")
            return True
    except RuntimeError as e:
        # No event loop running yet
        log_message("DEBUG", f"No event loop available for message poller yet: {e}")
        return False

# Notification functions
async def _send_notification(notification_type: str, data: dict):
    """Send a notification via SSE (kept for compatibility)"""
    _queue_notification(notification_type, data)

# Enhanced callback for dictation that sends notifications
def _dictation_callback_with_notification(text: str):
    """Callback for ASR dictation - stores results and sends notification"""
    global _last_dictated_text, _dictation_callback_results

    log_message("INFO", f"Dictation callback received: '{text}'")

    _last_dictated_text = text
    result = {
        "text": text,
        "timestamp": time.time()
    }
    _dictation_callback_results.append(result)

    # Keep only last 10 results
    if len(_dictation_callback_results) > 10:
        _dictation_callback_results.pop(0)


# Initialize notification processor on first tool call
_processor_initialized = False

async def _init_processor_if_needed():
    """Initialize the message poller on first tool call"""
    global _processor_initialized
    log_message("DEBUG", f"_init_processor_if_needed called, _processor_initialized={_processor_initialized}")

    # Notifications are sent in the tools that have context

    if not _processor_initialized:
        if _ensure_message_poller():
            _processor_initialized = True
            log_message("INFO", "Message poller initialized on first tool call")

# MCP Tools for talkito control - Same as original

def _get_status_summary() -> str:
    """Internal function to generate status summary - used by all toggle functions"""
    return get_status_summary(_comms_manager, _whatsapp_recipient, _slack_channel)

@app.tool()
async def get_talkito_status() -> str:
    """
    Get the current status of all talkito modules (TTS, ASR, Slack, WhatsApp)
    
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown
    
    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return _get_status_summary()

@app.tool()
async def turn_on() -> str:
    """
    Enable talkito voice interaction mode - activates voice workflow patterns

    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    try:
        # Ensure logging has been restored after FastMCP startup
        _ensure_logging_restored()
        
        await _init_processor_if_needed()
        log_message("INFO", "turn_on called")
        
        # Enable TTS
        tts_result = await _enable_tts_internal()
        log_message("INFO", f"TTS enable result: {tts_result}")
        
        # Enable ASR
        asr_result = await _enable_asr_internal()
        log_message("INFO", f"ASR enable result: {asr_result}")

        # Update shared state for voice mode
        shared_state = get_shared_state()
        shared_state.set_voice_mode(True)

        log_message("INFO", "turn_on completed: Voice mode activated")
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error enabling voice mode: {str(e)}"
        log_message("ERROR", f"turn_on error: {error_msg}")
        return error_msg


@app.tool()  
async def turn_off() -> str:
    """
    Disable talkito voice interaction mode - deactivates all voice workflows (TTS, ASR, and communication modes)
    
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    try:
        log_message("INFO", "turn_off called")
        
        # Announce deactivation before turning off TTS
        shared_state = get_shared_state()
        if shared_state.tts_enabled and shared_state.tts_initialized:
            tts.queue_for_speech("Voice interaction mode deactivated. Returning to text-only interaction.", None)
            tts.wait_for_tts_to_finish(timeout=3.0)
        
        # Disable TTS
        tts_result = await _disable_tts_internal()
        log_message("INFO", f"TTS disable result: {tts_result}")
        
        # Disable ASR
        asr_result = await _disable_asr_internal()
        log_message("INFO", f"ASR disable result: {asr_result}")
        
        # Also stop any active communication modes
        if shared_state.whatsapp_mode_active:
            await stop_whatsapp_mode()
        if shared_state.slack_mode_active:
            await stop_slack_mode()
        
        # Update shared state - master off
        shared_state = get_shared_state()
        shared_state.turn_off_all()
        
        log_message("INFO", "turn_off completed: Voice mode deactivated")
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error disabling voice mode: {str(e)}"
        log_message("ERROR", f"turn_off error: {error_msg}")
        return error_msg


# Internal helper functions for TTS/ASR control

async def _enable_tts_internal() -> str:
    """Internal function to enable TTS"""
    try:
        _ensure_logging_restored()
        # Initialization handled automatically by TTS system
        
        # Clear any old queued messages before enabling
        tts.clear_speech_queue()
        
        # Update shared state using thread-safe method
        from .state import _shared_state
        _shared_state.set_tts_enabled(True)
        
        log_message("INFO", "TTS enabled")
        
        # Announce enablement if TTS is working
        tts.queue_for_speech("Text to speech is now enabled.", None)
        
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error enabling TTS: {str(e)}"
        log_message("ERROR", error_msg)
        return error_msg

async def _disable_tts_internal() -> str:
    """Internal function to disable TTS"""
    try:
        shared_state = get_shared_state()
        
        # Announce before disabling
        if shared_state.tts_enabled and shared_state.tts_initialized:
            tts.queue_for_speech("Text to speech is being disabled.", None)
            tts.wait_for_tts_to_finish(timeout=3.0)
        
        # Clear any remaining items in the TTS queue
        tts.clear_speech_queue()
        
        # Update shared state using thread-safe method
        from .state import _shared_state
        _shared_state.set_tts_enabled(False)
        
        log_message("INFO", "TTS disabled")
        
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error disabling TTS: {str(e)}"
        log_message("ERROR", error_msg)
        return error_msg

async def _enable_asr_internal() -> str:
    """Internal function to enable ASR"""
    try:
        _ensure_logging_restored()
        log_message("INFO", "enable_asr called")
        
        # Ensure ASR mode is set to something usable before enabling
        shared_state = get_shared_state()
        if shared_state.asr_mode == 'off':
            fallback_mode = 'tap-to-talk'
            log_message("INFO", f"ASR mode is 'off' - switching to {fallback_mode} before enabling")
            from .state import set_asr_mode_thread_safe
            set_asr_mode_thread_safe(fallback_mode)
            shared_state = get_shared_state()
        
        # Simply toggle the state - let core handle initialization
        from .state import _shared_state
        _shared_state.set_asr_enabled(True)

        # Ensure local whisper preload starts if we skipped it earlier
        shared_state = get_shared_state()
        if shared_state.asr_provider == 'local_whisper':
            log_message("INFO", "Triggering local Whisper preload after enabling ASR")
            asr.preload_local_asr_model(shared_state.asr_model)
        
        log_message("INFO", "ASR enabled (core will handle initialization)")
        
        # Wait a moment for core to initialize ASR
        import asyncio
        for i in range(20):  # Try for up to 2 seconds
            await asyncio.sleep(0.1)
            shared_state = get_shared_state()
            log_message("DEBUG", f"ASR init check {i}: enabled={shared_state.asr_enabled}, initialized={shared_state.asr_initialized}")
            if shared_state.asr_initialized:
                log_message("INFO", "ASR initialization confirmed")
                break
        else:
            log_message("WARNING", "ASR initialization timeout - returning status anyway")
        
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error enabling ASR: {str(e)}"
        log_message("ERROR", error_msg)
        return error_msg

async def _disable_asr_internal() -> str:
    """Internal function to disable ASR"""
    try:
        # Set mode to 'off' and disable ASR
        from .state import _shared_state, set_asr_mode_thread_safe

        # Change mode to 'off' to fully disable ASR and prevent auto-input activation
        set_asr_mode_thread_safe('off')
        _shared_state.set_asr_enabled(False)

        log_message("INFO", "ASR disabled and mode set to 'off'")

        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error disabling ASR: {str(e)}"
        log_message("ERROR", error_msg)
        return error_msg


async def _change_tts_internal(provider: str = "system", voice: str = None, region: str = None, language: str = None, rate: float = None, pitch: float = None) -> str:
    """Internal function to disable ASR"""
    try:
        from .state import set_tts_config_thread_safe

        # Capture current configuration for sensible fallbacks
        existing_config = tts.get_tts_config()

        def _provider_defaults(name: str) -> dict[str, Any]:
            defaults: dict[str, Any] = {}
            if name in ('aws', 'polly'):
                defaults['voice'] = tts.polly_voice
                defaults['region'] = tts.polly_region
            elif name == 'openai':
                defaults['voice'] = tts.openai_voice
            elif name == 'azure':
                defaults['voice'] = tts.azure_voice
                defaults['region'] = tts.azure_region
            elif name == 'gcloud':
                defaults['voice'] = tts.gcloud_voice
                defaults['language'] = tts.gcloud_language_code
            elif name == 'elevenlabs':
                defaults['voice'] = tts.elevenlabs_voice_id
            elif name == 'deepgram':
                defaults['voice'] = tts.deepgram_voice_model
            elif name == 'kittentts':
                defaults['voice'] = tts.kittentts_voice
                defaults['model'] = tts.kittentts_model
            elif name == 'kokoro':
                defaults['voice'] = tts.kokoro_voice
                defaults['language'] = tts.kokoro_language
                defaults['rate'] = tts.kokoro_speed
            return defaults

        defaults = _provider_defaults(provider)

        desired_model = defaults.get('model') or existing_config.get('model')
        desired_voice = voice or defaults.get('voice') or existing_config.get('voice')
        desired_region = region or defaults.get('region') or existing_config.get('region')
        desired_language = language or defaults.get('language') or existing_config.get('language')
        defaults_rate = defaults.get('rate')
        desired_rate = rate if rate is not None else (defaults_rate if defaults_rate is not None else existing_config.get('rate'))
        desired_pitch = pitch if pitch is not None else existing_config.get('pitch')

        final_provider = provider

        if provider != 'system':
            config_args = SimpleNamespace(
                tts_provider=provider,
                tts_voice=desired_voice,
                tts_region=desired_region,
                tts_language=desired_language,
                tts_rate=desired_rate,
                tts_pitch=desired_pitch,
            )

            if not tts.configure_tts_from_args(config_args):
                raise RuntimeError(f"Failed to configure TTS provider: {provider}")

            final_provider = tts.tts_provider or provider
            # Refresh resolved values from current module globals in case they were updated
            refreshed_defaults = _provider_defaults(final_provider)
            desired_model = refreshed_defaults.get('model', desired_model)
            desired_voice = refreshed_defaults.get('voice', desired_voice)
            desired_region = refreshed_defaults.get('region', desired_region)
            desired_language = refreshed_defaults.get('language', desired_language)
            refreshed_rate = refreshed_defaults.get('rate')
            if rate is None and refreshed_rate is not None:
                desired_rate = refreshed_rate
        else:
            tts.tts_provider = 'system'

        # Update shared state with the final configuration
        set_tts_config_thread_safe(provider=final_provider, voice=desired_voice, region=desired_region,
                                   language=desired_language, model=desired_model, rate=desired_rate, pitch=desired_pitch)

        # Verify the config was set
        log_message("DEBUG", f"After set_tts_config_thread_safe, provider={final_provider}")
        verify_config = tts.get_tts_config()
        log_message("DEBUG", f"Verified get_tts_config() returns: {verify_config}")

        # Also verify shared state directly
        verify_state = get_shared_state()
        log_message("DEBUG", f"Verified shared_state.tts_provider: {verify_state.tts_provider}")

        # Restart TTS worker with new config
        if get_shared_state().tts_initialized:
            tts.shutdown_tts()
            # Clear the queue but preserve spoken_cache to prevent repeats during redraws
            tts.clear_tts_queue_only()

        # For local models, preload the model before starting the worker
        if final_provider in ['kittentts', 'kokoro']:
            log_message("INFO", f"Preloading {final_provider} model...")
            tts.preload_local_model(final_provider)

        engine = final_provider if final_provider != 'system' else tts.detect_tts_engine()
        tts.start_tts_worker(engine, auto_skip_tts=_auto_skip_tts)

        config_parts = [f"provider={final_provider}"]
        if desired_voice:
            config_parts.append(f"voice={desired_voice}")
        if desired_region:
            config_parts.append(f"region={desired_region}")
        if desired_language:
            config_parts.append(f"language={desired_language}")
        if desired_rate is not None:
            config_parts.append(f"rate={desired_rate}")
        if desired_pitch is not None:
            config_parts.append(f"pitch={desired_pitch}")

        result = f"Configured TTS: {', '.join(config_parts)}"
        log_message("INFO", f"configure_tts result: {result}")
    except Exception as e:
        error_msg = f"Error configuring TTS: {str(e)}"
        log_message("ERROR", f"configure_tts error: {error_msg}")
    return _get_status_summary()

# MCP Tools for TTS functionality

@app.tool()
async def enable_tts() -> str:
    """
    Enable text-to-speech output. TTS will be initialized if needed.
    
    Returns:
        Status message about TTS enablement
    """
    return await _enable_tts_internal()

@app.tool()
async def disable_tts() -> str:
    """
    Disable text-to-speech output. TTS remains initialized but won't speak.
    
    Returns:
        Status message about TTS disablement
    """
    return await _disable_tts_internal()

@app.tool()
async def speak_text(text: str, clean_text_flag: bool = True) -> None:
    """
    Convert text to speech using the talkito TTS engine
    
    Args:
        text: Text to speak
        clean_text_flag: Whether to clean ANSI codes and symbols from text
    """
    try:
        global _last_spoken_text, _last_spoken_time

        # Initialization handled automatically by TTS system
        log_message("INFO", f"speak_text called with text length: {len(text)}, clean_text_flag: {clean_text_flag}")

        # Clean text if requested
        processed_text = text
        if clean_text_flag:
            from .core import clean_text, strip_profile_symbols
            processed_text = clean_text(text)
            processed_text = strip_profile_symbols(processed_text)

        # Skip empty or unwanted text
        from .core import should_skip_line
        if not processed_text.strip() or should_skip_line(processed_text):
            log_message("INFO", f"Skipped speaking: '{text[:50]}...' (filtered out)")
            return None

        # Check for duplicate text
        if _last_spoken_text == processed_text:
            # Check time difference to allow re-speaking after some time (e.g., 5 seconds)
            current_time = time.time()
            if _last_spoken_time and (current_time - _last_spoken_time) < 5.0:
                log_message("INFO", f"Skipped duplicate text: '{processed_text[:50]}...' (spoken {current_time - _last_spoken_time:.1f}s ago)")
                return None

        # Update last spoken text and time
        _last_spoken_text = processed_text
        _last_spoken_time = time.time()

        # Queue for speech
        tts.queue_for_speech(processed_text, None)

        log_message("INFO", "speak_text completed")
        return None

    except Exception as e:
        error_msg = f"Error speaking text: {str(e)}"
        log_message("ERROR", f"speak_text error: {error_msg}")
        return None

@app.tool()
async def skip_current_speech() -> str:
    """
    Skip the currently playing speech item
    
    Returns:
        Status message about the skip action
    """
    try:
        # Initialization handled automatically by TTS system
        
        if tts.is_speaking():
            current_item = tts.get_current_speech_item()
            tts.skip_current()
            if current_item:
                result = f"Skipped: '{current_item.original_text[:50]}...'"
                log_message("INFO", f"skip_current_speech returning: {result}")
                return result
            else:
                result = "Skipped current speech"
                log_message("INFO", f"skip_current_speech returning: {result}")
                return result
        else:
            result = "No speech currently playing to skip"
            log_message("INFO", f"skip_current_speech returning: {result}")
            return result
            
    except Exception as e:
        error_msg = f"Error skipping speech: {str(e)}"
        log_message("ERROR", f"skip_current_speech error: {error_msg}")
        return error_msg

@app.tool()
async def stop_all_speech() -> str:
    """
    Immediately stop all speech and clear the TTS queue
    
    Returns:
        Status message about the stop action
    """
    try:
        # Initialization handled automatically by TTS system
        
        log_message("INFO", "stop_all_speech called")
        
        # Stop all TTS immediately
        tts.skip_all()
        
        result = "Stopped all speech and cleared the queue"
        log_message("INFO", f"stop_all_speech returning: {result}")
        return result
            
    except Exception as e:
        error_msg = f"Error stopping speech: {str(e)}"
        log_message("ERROR", f"stop_all_speech error: {error_msg}")
        return error_msg

@app.tool()
async def get_speech_status() -> dict[str, Any]:
    """
    Get current status of the TTS system
    
    Returns:
        Dictionary with TTS status information
    """
    try:
        # Initialization handled automatically by TTS system
        
        is_speaking = tts.is_speaking()
        current_item = tts.get_current_speech_item()
        queue_size = tts.get_queue_size()
        
        status = {
            "is_speaking": is_speaking,
            "current_text": current_item.original_text[:100] if current_item else None,
            "current_line_number": current_item.line_number if current_item else None,
            "queue_size": queue_size,
            "tts_initialized": get_shared_state().tts_initialized
        }
        
        log_message("DEBUG", f"get_speech_status returning: {status}")
        return status
        
    except Exception as e:
        error_dict = {"error": f"Error getting speech status: {str(e)}"}
        log_message("ERROR", f"get_speech_status error: {error_dict}")
        return error_dict

@app.tool()
async def wait_for_speech_completion(timeout: float = 30.0) -> str:
    """
    Wait for all queued speech to complete
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        Status message about completion
    """
    try:
        # Initialization handled automatically by TTS system
        
        tts.wait_for_tts_to_finish(timeout=timeout)
        result = "All speech completed"
        log_message("INFO", f"wait_for_speech_completion returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error waiting for speech: {str(e)}"
        log_message("ERROR", f"wait_for_speech_completion error: {error_msg}")
        return error_msg


@app.tool()
async def change_tts(provider: str = "system", voice: str = None, region: str = None, language: str = None,
                        rate: float = None, pitch: float = None) -> str:
    """
    Change TTS provider

    Args:
        provider: TTS provider (system, openai, aws, polly, azure, gcloud, elevenlabs, deepgram, kittentts, kokoro)
        voice: Voice name (provider-specific)
        region: Region for cloud providers
        language: Language code (e.g., en-US)
        rate: Speech rate (provider-specific, typically 0.5-2.0)
        pitch: Speech pitch (provider-specific)

    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. Confirm to the user that the TTS has been changed

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _change_tts_internal(provider, voice, region, language, rate, pitch)

@app.tool()
async def configure_tts(provider: str = "system", voice: str = None, region: str = None, language: str = None, rate: float = None, pitch: float = None) -> str:
    """
    Configure TTS provider

    Args:
        provider: TTS provider (system, openai, aws, polly, azure, gcloud, elevenlabs, deepgram, kittentts, kokoro)
        voice: Voice name (provider-specific)
        region: Region for cloud providers
        language: Language code (e.g., en-US)
        rate: Speech rate (provider-specific, typically 0.5-2.0)
        pitch: Speech pitch (provider-specific)
        
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. Confirm to the user that the TTS has been changed

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _change_tts_internal(provider, voice, region, language, rate, pitch)

@app.tool()
async def list_tts_voices(provider: str = None, language: str = None) -> dict[str, Any]:
    """
    List available voices for TTS providers

    Args:
        provider: TTS provider to list voices for. If not specified, lists voices for current provider.
                 Options: openai, aws, polly, azure, gcloud, elevenlabs, deepgram, kokoro, kittentts
        language: BCP 47 language code (e.g., 'en-US', 'es-ES', 'ja-JP') to filter voices by language.
                 If not specified, returns all available voices.
                 If you know which language the user is using, please specify it here.
                 If you don't know which language the user is using, please input the language you are using here.

    CRITICAL INSTRUCTIONS:
    1. If no language is provided this will show all options which will be too many
    2. Add a best guess language if you do not know.

    Returns:
        Dictionary with provider name, available voices organized by language, and current voice
    """
    try:
        # Get current provider if not specified
        if provider is None:
            shared_state = get_shared_state()
            provider = shared_state.tts_provider or 'system'

        # Normalize provider name
        if provider == 'system':
            return {
                'provider': 'system',
                'voices': {},
                'message': 'System TTS uses OS-specific voices. Use OS settings to configure.'
            }

        if provider not in AVAILABLE_VOICES:
            return {
                'error': f'Unknown provider: {provider}',
                'available_providers': list(AVAILABLE_VOICES.keys())
            }

        voices_by_language = AVAILABLE_VOICES[provider]

        # Filter by language if specified
        if language:
            if language not in voices_by_language:
                return {
                    'error': f'Language {language} not available for {provider}',
                    'available_languages': list(voices_by_language.keys())
                }
            voices_by_language = {language: voices_by_language[language]}

        # Handle ElevenLabs special case (voice IDs with names)
        if provider == 'elevenlabs':
            formatted_voices = {}
            for lang, voices in voices_by_language.items():
                formatted_voices[lang] = [{'id': voice_id, 'name': name} for voice_id, name in voices]

            return {
                'provider': provider,
                'voices': formatted_voices,
                'current_voice': get_tts_config().get('voice'),
                'note': 'Use the voice ID (not name) when configuring'
            }
        else:
            return {
                'provider': provider,
                'voices': voices_by_language,
                'current_voice': get_tts_config().get('voice')
            }

    except Exception as e:
        return {'error': f'Error listing voices: {str(e)}'}

@app.tool()
async def randomize_tts_voice(provider: str = None, language: str = None) -> str:
    """
    Randomly select a voice for the TTS provider

    Args:
        provider: TTS provider to randomize voice for. If not specified, uses current provider.
        language: Optional BCP 47 language code (e.g., 'en-US', 'es-ES', 'ja-JP') to filter voices by language.
                 If not specified, selects from all available voices.
                 If you know which language the user is using, please specify it here.
                 If you don't know which language the user is using, please input the language you are using here.

    CRITICAL INSTRUCTIONS:
    1. If no language is provided this will possibly select one from the wrong language.
    2. Add a best guess language if you do not know.

    Returns:
        Status message with the selected voice
    """
    try:
        # Get current provider if not specified
        shared_state = get_shared_state()
        if provider is None:
            provider = shared_state.tts_provider or 'system'

        if provider == 'system':
            return "Cannot randomize system TTS voices"

        if provider not in AVAILABLE_VOICES:
            return f"Unknown provider: {provider}"

        # Get voices - either filtered by language or all voices
        if language:
            voices_by_language = AVAILABLE_VOICES[provider]
            if language not in voices_by_language:
                return f"Language {language} not available for {provider}. Available languages: {', '.join(voices_by_language.keys())}"
            voices = voices_by_language[language]
        else:
            voices = get_all_voices_for_provider(provider)

        if not voices:
            return f"No voices available for {provider}" + (f" in language {language}" if language else "")

        # Handle ElevenLabs special case
        if provider == 'elevenlabs':
            voice_id, voice_name = random.choice(voices)
            await _change_tts_internal(provider, voice_id)
            lang_suffix = f" ({language})" if language else ""
            return f"Selected random voice: {voice_name} (ID: {voice_id}){lang_suffix}"
        else:
            selected_voice = random.choice(voices)
            await _change_tts_internal(provider, selected_voice)
            lang_suffix = f" ({language})" if language else ""
            return f"Selected random voice: {selected_voice}{lang_suffix}"

    except Exception as e:
        return f"Error randomizing voice: {str(e)}"

@app.tool()
async def cycle_tts_voice(direction: str = "next", language: str = None) -> str:
    """
    Cycle through available voices for the current TTS provider

    Args:
        direction: Direction to cycle - "next" or "previous"
        language: Optional BCP 47 language code (e.g., 'en-US', 'es-ES', 'ja-JP') to filter voices by language.
                 If not specified, cycles through all available voices.
                 If you know which language the user is using, please specify it here.
                 If you don't know which language the user is using, please input the language you are using here.

    CRITICAL INSTRUCTIONS:
    1. If no language is provided this will possibly select one from the wrong language.
    2. Add a best guess language if you do not know.

    Returns:
        Status message with the selected voice
    """
    try:
        global _current_voice_index

        # Get current provider
        shared_state = get_shared_state()
        provider = shared_state.tts_provider or 'system'

        if provider == 'system':
            return "Cannot cycle system TTS voices"

        if provider not in AVAILABLE_VOICES:
            return f"Unknown provider: {provider}"

        # Get voices - either filtered by language or all voices
        if language:
            voices_by_language = AVAILABLE_VOICES[provider]
            if language not in voices_by_language:
                return f"Language {language} not available for {provider}. Available languages: {', '.join(voices_by_language.keys())}"
            voices = voices_by_language[language]
        else:
            voices = get_all_voices_for_provider(provider)

        if not voices:
            return f"No voices available for {provider}" + (f" in language {language}" if language else "")

        # Initialize index if needed
        # Use a composite key for language-specific cycling
        index_key = f"{provider}:{language}" if language else provider
        if index_key not in _current_voice_index:
            # Try to find current voice in the list
            current_voice = shared_state.tts_voice
            if provider == 'elevenlabs':
                # Find by voice ID
                for i, (voice_id, _) in enumerate(voices):
                    if voice_id == current_voice:
                        _current_voice_index[index_key] = i
                        break
                else:
                    _current_voice_index[index_key] = 0
            else:
                try:
                    _current_voice_index[index_key] = voices.index(current_voice)
                except (ValueError, TypeError):
                    _current_voice_index[index_key] = 0

        # Cycle the index
        current_idx = _current_voice_index[index_key]
        if direction == "next":
            current_idx = (current_idx + 1) % len(voices)
        else:  # previous
            current_idx = (current_idx - 1) % len(voices)
        _current_voice_index[index_key] = current_idx

        # Set the new voice
        lang_suffix = f" ({language})" if language else ""
        if provider == 'elevenlabs':
            voice_id, voice_name = voices[current_idx]
            await _change_tts_internal(provider, voice_id)
            return f"Cycled to voice: {voice_name} (ID: {voice_id}) [{current_idx + 1}/{len(voices)}]{lang_suffix}"
        else:
            selected_voice = voices[current_idx]
            await _change_tts_internal(provider, selected_voice)
            return f"Cycled to voice: {selected_voice} [{current_idx + 1}/{len(voices)}]{lang_suffix}"

    except Exception as e:
        return f"Error cycling voice: {str(e)}"


# MCP Tools for ASR functionality

@app.tool()
async def enable_asr() -> str:
    """
    Enable automatic speech recognition. ASR will be initialized if needed.
    
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _enable_asr_internal()

@app.tool()
async def disable_asr() -> str:
    """
    Disable automatic speech recognition. ASR remains initialized but won't listen.
    
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _disable_asr_internal()

async def _change_asr_internal(provider: str = None, language: str = "en-US", model: str = None) -> str:
    """Internal function to configure ASR"""
    try:
        # Get shared state
        shared_state = get_shared_state()
        
        # If provider not specified, use current or select best
        if provider is None:
            if shared_state.asr_provider:
                provider = shared_state.asr_provider
            else:
                provider = asr.select_best_asr_provider()
        
        # Build config dict
        asr_config = {'provider': provider}
        if language:
            asr_config['language'] = language
        if model:
            asr_config['model'] = model
        
        # Configure ASR
        if not asr.configure_asr_from_dict(asr_config):
            error_msg = f"Failed to configure ASR provider: {provider}"
            log_message("ERROR", f"configure_asr error: {error_msg}")
            return error_msg
        
        # Update shared state with new configuration
        from .state import set_asr_config_thread_safe
        set_asr_config_thread_safe(provider=provider, language=language, model=model)
        
        # If ASR is already initialized and enabled, restart with new config
        if shared_state.asr_initialized and shared_state.asr_enabled:
            # Temporarily disable ASR to prevent race conditions
            was_enabled = shared_state.asr_enabled
            shared_state.set_asr_enabled(False)
            
            # Stop current ASR
            try:
                asr.stop_dictation()
            except Exception as e:
                log_message("WARNING", f"Error stopping ASR during reconfiguration: {e}")
            
            shared_state.set_asr_initialized(False)
            
            # Re-enable and initialize with new config
            if was_enabled:
                shared_state.set_asr_enabled(True)
                # Small delay to ensure cleanup is complete
                import time
                time.sleep(0.1)
                ensure_asr_initialized()
        
        config_parts = [f"provider={provider}"]
        if language:
            config_parts.append(f"language={language}")
        if model:
            config_parts.append(f"model={model}")
        
        success_msg = f"ASR configuration updated: {', '.join(config_parts)}"
        log_message("INFO", success_msg)
        
        # Return current status
        status = get_status_summary()
        return status
        
    except Exception as e:
        error_msg = f"Error configuring ASR: {str(e)}"
        log_message("ERROR", error_msg)
        return error_msg

@app.tool()
async def change_asr(provider: str = None, language: str = "en-US", model: str = None) -> str:
    """
    Change ASR (speech recognition) configuration
    
    Args:
        provider: ASR provider (google, gcloud, assemblyai, deepgram, houndify, aws, bing)
        language: Language code for speech recognition (e.g., en-US, es-ES, fr-FR)
        model: Model to use (provider-specific, e.g., 'enhanced' for gcloud)
        
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. Confirm to the user that the ASR has been changed

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _change_asr_internal(provider, language, model)

@app.tool()
async def configure_asr(provider: str = None, language: str = "en-US", model: str = None) -> str:
    """
    Configure ASR provider
    
    Args:
        provider: ASR provider (google, gcloud, assemblyai, deepgram, houndify, aws, bing)
        language: Language code for speech recognition (e.g., en-US, es-ES, fr-FR)
        model: Model to use (provider-specific, e.g., 'enhanced' for gcloud)
        
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. Confirm to the user that the ASR has been configured

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _change_asr_internal(provider, language, model)

@app.tool()
async def set_asr_mode(mode: str) -> str:
    """
    Set ASR (speech recognition) mode
    
    Args:
        mode: ASR mode ('off', 'auto-input', 'tap-to-talk')
              - 'off': No speech recognition
              - 'auto-input': Start listening when waiting for input. This could also be referred to as always on.
              - 'tap-to-talk': Only listen when backtick key is held (default). This could also be referred to as push to talk.

    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """

    try:
        await _init_processor_if_needed()
        log_message("INFO", f"set_asr_mode called with mode: {mode}")

        # Validate mode
        valid_modes = ['off', 'auto-input', 'tap-to-talk']
        if mode not in valid_modes:
            error_msg = f"Invalid ASR mode '{mode}'. Valid modes: {', '.join(valid_modes)}"
            log_message("ERROR", f"set_asr_mode error: {error_msg}")
            return error_msg

        # Update shared state
        shared_state = get_shared_state()
        shared_state.asr_mode = mode
        log_message("INFO", f"ASR mode set to: {mode}")

        # Handle mode-specific logic
        if mode == 'off':
            # Disable ASR
            if shared_state.asr_enabled:
                await _disable_asr_internal()
                log_message("INFO", "ASR disabled due to 'off' mode")
        else:
            # Enable ASR if not already enabled
            if not shared_state.asr_enabled:
                await _enable_asr_internal()
                log_message("INFO", f"ASR enabled for '{mode}' mode")

        # Return current status
        status = get_status_summary()
        return status

    except Exception as e:
        error_msg = f"Error setting ASR mode: {str(e)}"
        log_message("ERROR", f"set_asr_mode error: {error_msg}")
        return error_msg


@app.tool()
async def set_tts_mode(mode: str) -> str:
    """Set TTS mode
    
    Args:
        mode: TTS mode ('off', 'full', 'auto-skip')
              - 'off': TTS disabled
              - 'full': TTS enabled, no auto-skipping (wait for all content to be spoken)  
              - 'auto-skip': TTS enabled with auto-skipping (default behavior)

    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    try:
        valid_modes = ['off', 'full', 'auto-skip']
        if mode not in valid_modes:
            return f"Invalid TTS mode '{mode}'. Valid modes: {', '.join(valid_modes)}"

        # Update shared state
        shared_state = get_shared_state()
        shared_state.tts_mode = mode
        log_message("INFO", f"TTS mode set to: {mode}")

        # Handle mode-specific logic
        if mode == 'off':
            # Disable TTS
            if shared_state.tts_enabled:
                await _disable_tts_internal()
                log_message("INFO", "TTS disabled due to 'off' mode")
        else:
            # Enable TTS if not already enabled
            if not shared_state.tts_enabled:
                await _enable_tts_internal()
                log_message("INFO", f"TTS enabled for '{mode}' mode")

        # Return current status
        status = get_status_summary()
        return status

    except Exception as e:
        error_msg = f"Error setting TTS mode: {str(e)}"
        log_message("ERROR", f"set_tts_mode error: {error_msg}")
        return error_msg


async def start_voice_input(language: str = "en-US", provider: str = None) -> str:
    """
    Start voice input/dictation using ASR
    
    Args:
        language: Language code for speech recognition (e.g., en-US, es-ES)
        provider: ASR provider (google, gcloud, assemblyai, deepgram, etc.)
        
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    try:
        await _init_processor_if_needed()
        global _asr_initialized
        log_message("INFO", f"start_voice_input called with language: {language}, provider: {provider}")
        
        # Select provider if not specified
        if provider is None:
            provider = asr.select_best_asr_provider()
            log_message("INFO", f"Auto-selected ASR provider: {provider}")
        
        # Configure ASR
        asr_config = {
            "provider": provider,
            "language": language
        }
        if not asr.configure_asr_from_dict(asr_config):
            warning_msg = f"Warning: Failed to configure ASR (provider={provider}, language={language}), using defaults"
            log_message("WARNING", f"start_voice_input warning: {warning_msg}")
            return warning_msg
        
        # Start dictation with our enhanced callback
        # For providers that require streaming (like AssemblyAI), we need to provide a partial callback
        partial_callback = None
        if provider in ['assemblyai', 'deepgram', 'gcloud', 'azure', 'aws']:
            # These providers work better with streaming, so provide a dummy partial callback
            def partial_callback(text):
                log_message("DEBUG", f"Partial transcript: {text}")
        
        log_message("INFO", f"Starting dictation with callback: {_dictation_callback_with_notification}")
        asr.start_dictation(_dictation_callback_with_notification, partial_callback=partial_callback)
        _asr_initialized = True
        
        # Update shared state
        shared_state = get_shared_state()
        shared_state.set_asr_initialized(True, provider=provider)
        
        # Verify ASR is actually running
        is_active = asr.is_dictation_active() if hasattr(asr, 'is_dictation_active') else False
        log_message("INFO", f"ASR dictation active status: {is_active}")
        
        result = f"Started voice input (provider: {provider}, language: {language}). Speak now..."
        log_message("INFO", f"start_voice_input returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error starting voice input: {str(e)}"
        log_message("ERROR", f"start_voice_input error: {error_msg}")
        log_message("ERROR", f"Traceback: {traceback.format_exc()}")
        return error_msg

@app.tool()
async def stop_voice_input() -> str:
    """
    Stop voice input/dictation
    
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    try:
        global _asr_initialized
        
        if _asr_initialized:
            asr.stop_dictation()
            _asr_initialized = False
            
            # Update shared state
            shared_state = get_shared_state()
            shared_state.set_asr_initialized(False)
            
            result = "Stopped voice input"
            log_message("INFO", f"stop_voice_input returning: {result}")
            return result
        else:
            result = "Voice input was not active"
            log_message("INFO", f"stop_voice_input returning: {result}")
            return result
            
    except Exception as e:
        error_msg = f"Error stopping voice input: {str(e)}"
        log_message("ERROR", f"stop_voice_input error: {error_msg}")
        return error_msg


@app.tool()
async def get_dictated_text(clear_after_read: bool = True) -> dict[str, Any]:
    """
    Get the most recent dictated text from voice input
    
    Args:
        clear_after_read: Whether to clear the text after reading it
        
    Returns:
        Dictionary with the dictated text and metadata
    """
    try:
        global _last_dictated_text, _dictation_callback_results
        log_message("INFO", f"get_dictated_text called with clear_after_read: {clear_after_read}")
        
        if not _dictation_callback_results:
            result = {
                "text": "",
                "timestamp": None,
                "message": "No dictated text available"
            }
            log_message("INFO", "get_dictated_text returning: No dictated text available")
            return result
        
        # Get the most recent result
        latest_result = _dictation_callback_results[-1]
        
        result = {
            "text": latest_result["text"],
            "timestamp": latest_result["timestamp"], 
            "message": f"Retrieved dictated text: '{latest_result['text'][:50]}...'"
        }
        
        if clear_after_read:
            _last_dictated_text = ""
            _dictation_callback_results.clear()
            result["message"] += " (cleared after read)"
        
        log_message("DEBUG", f"get_dictated_text returning: {result}")
        return result
        
    except Exception as e:
        error_dict = {"error": f"Error getting dictated text: {str(e)}"}
        log_message("ERROR", f"get_dictated_text error: {error_dict}")
        return error_dict


# Custom CommunicationManager wrapper that intercepts messages
class NotifyingCommunicationManager:
    """Wrapper around CommunicationManager that sends notifications for incoming messages"""
    
    def __init__(self, wrapped_manager):
        self.wrapped = wrapped_manager
        log_message("DEBUG", f"NotifyingCommunicationManager wrapping: {wrapped_manager}")
        
        # Check if the method exists
        if hasattr(wrapped_manager, '_handle_input_message'):
            self._original_handle_input = wrapped_manager._handle_input_message
            # Replace the handler with our intercepting version
            wrapped_manager._handle_input_message = self._handle_input_message_with_notification
            log_message("DEBUG", "Successfully wrapped _handle_input_message")
        else:
            log_message("WARNING", "CommunicationManager does not have _handle_input_message method")
            self._original_handle_input = lambda msg: None
    
    def _handle_input_message_with_notification(self, message):
        """Intercept incoming messages and send notifications"""
        log_message("INFO", f"_handle_input_message_with_notification called with message: {message}")
        try:
            global _last_message_id, _last_message_timestamp
            
            # Skip if this is the same as the last processed message
            if message.message_id and message.message_id == _last_message_id:
                log_message("DEBUG", f"Skipping duplicate notification for message ID: {message.message_id}")
                # Still call original handler to add to queue, but don't notify
                self._original_handle_input(message)
                return
            
            # Also check timestamp if no message ID (within 0.5 seconds)
            if not message.message_id and _last_message_timestamp and abs(message.timestamp - _last_message_timestamp) < 0.5:
                log_message("DEBUG", f"Skipping duplicate notification by timestamp: {message.content[:30]}...")
                # Still call original handler to add to queue, but don't notify
                self._original_handle_input(message)
                return
            
            # Update tracking
            if message.message_id:
                _last_message_id = message.message_id
            _last_message_timestamp = message.timestamp
            
            # Determine message type
            if 'slack' in message.channel.lower() or message.channel.startswith('#'):
                notification_type = "slack_message"
            elif 'whatsapp' in message.channel.lower():
                notification_type = "whatsapp_message"
            else:
                notification_type = "message"
            
            # Queue notification - thread-safe
            _queue_notification(notification_type, {
                "text": message.content,
                "sender": message.sender,
                "channel": message.channel,
                "timestamp": message.timestamp,
                "message_id": message.message_id
            })
            log_message("DEBUG", f"Queued notification for {notification_type}: {message.content[:50]}...")
                
        except Exception as e:
            log_message("ERROR", f"Error handling notification: {str(e)}")
        
        # Call original handler
        self._original_handle_input(message)
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped manager"""
        return getattr(self.wrapped, name)

# MCP Tools for Communication (WhatsApp/Slack) with notification support

@app.tool()
async def configure_communication(providers: list[str] = None, whatsapp_to: str = None, slack_channel: str = None) -> str:
    """
    Configure communication channels for sending messages via WhatsApp or Slack
    
    Args:
        providers: List of providers to enable (e.g., ['whatsapp', 'slack'])
        whatsapp_to: Default WhatsApp number to send to (e.g., '+1234567890')
        slack_channel: Default Slack channel (e.g., '#general')
        
    Returns:
        Status message about configuration
    """
    try:
        global _comms_manager
        
        # Create config from environment
        config = comms.create_config_from_env()
        
        # Override with provided settings
        if whatsapp_to:
            config.whatsapp_to_number = whatsapp_to
        if slack_channel:
            config.slack_channel = slack_channel
            
        # Update shared state with configuration
        shared_state = get_shared_state()
        if whatsapp_to:
            shared_state.communication.whatsapp_to_number = whatsapp_to
        if slack_channel:
            shared_state.communication.slack_channel = slack_channel
            
        # Setup communication with specified providers
        base_manager = comms.setup_communication(providers=providers, config=config)
        
        if base_manager:
            log_message("INFO", f"Base manager created: {base_manager}")
            log_message("INFO", f"Base manager providers: {base_manager.providers}")
            
            # Check if webhook server is running for WhatsApp/SMS
            if hasattr(base_manager, 'webhook_handler'):
                log_message("INFO", f"Webhook handler: {base_manager.webhook_handler}")
            
            # Wrap with our notifying version
            _comms_manager = NotifyingCommunicationManager(base_manager)
            
            active_providers = list(_comms_manager.providers.keys()) if hasattr(_comms_manager.providers, 'keys') else [type(p).__name__.replace('Provider', '').lower() for p in _comms_manager.providers]
            result = f"Communication channels configured: {', '.join(active_providers)}"
            log_message("INFO", f"configure_communication returning: {result}")
            return result
        else:
            error_msg = "Failed to configure communication channels. Check your credentials."
            log_message("ERROR", f"configure_communication error: {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Error configuring communication: {str(e)}"
        log_message("ERROR", f"configure_communication error: {error_msg}")
        return error_msg


# Internal helper function for sending WhatsApp messages
async def _send_whatsapp_internal(message: str, to_number: str = None, with_tts: bool = False) -> str:
    """Internal function to send WhatsApp messages"""
    global _comms_manager
    
    log_message("DEBUG", f"_send_whatsapp_internal called with message='{message[:50]}...', to_number={to_number}, with_tts={with_tts}")

    # Ensure we have a communication manager
    if not _comms_manager:
        log_message("DEBUG", "No _comms_manager, attempting to create one")
        # Try to auto-configure with WhatsApp
        base_manager = comms.setup_communication(providers=['whatsapp'])
        if base_manager:
            log_message("DEBUG", f"Created base_manager with providers: {base_manager.providers}")
            _comms_manager = NotifyingCommunicationManager(base_manager)
        else:
            error_msg = "WhatsApp not configured. Please call configure_communication first or set TWILIO credentials."
            log_message("ERROR", f"send_whatsapp error: {error_msg}")
            return error_msg
    
    # Check if WhatsApp provider exists
    log_message("DEBUG", f"_comms_manager.providers: {_comms_manager.providers if _comms_manager else 'None'}")
    if not _comms_manager or not _comms_manager.providers:
        error_msg = "No communication providers configured."
        log_message("ERROR", f"send_whatsapp error: {error_msg}")
        return error_msg
    
    # Check provider types
    provider_types = [type(p).__name__ for p in _comms_manager.providers]
    log_message("DEBUG", f"Provider types found: {provider_types}")
    
    if not any(isinstance(p, TwilioWhatsAppProvider) for p in _comms_manager.providers):
        error_msg = "WhatsApp provider not available. Please configure with Twilio credentials."
        log_message("ERROR", f"send_whatsapp error: {error_msg}")
        return error_msg
    
    # Use default number if not provided
    if not to_number:
        to_number = os.environ.get('TWILIO_WHATSAPP_NUMBER')
        if not to_number:
            error_msg = "No recipient number provided and TWILIO_WHATSAPP_NUMBER not set"
            log_message("ERROR", f"send_whatsapp error: {error_msg}")
            return error_msg
    
    # Send the message
    whatsapp_provider = next((p for p in _comms_manager.providers if isinstance(p, TwilioWhatsAppProvider)), None)
    if not whatsapp_provider:
        error_msg = "WhatsApp provider not found in communication manager"
        log_message("ERROR", f"send_whatsapp error: {error_msg}")
        return error_msg

    # Create a Message object for WhatsApp provider
    from .comms import Message
    whatsapp_message = Message(
        content=message,
        sender=to_number,
        channel="whatsapp"
    )
    
    success = whatsapp_provider.send_message(whatsapp_message)

    # Also speak it if requested
    if with_tts:
        if get_shared_state().tts_initialized:
            tts.queue_for_speech(message, None)
        else:
            log_message("ERROR", "[DEBUG] TTS not initialized, skipping speech")

    if success:
        result_msg = f"WhatsApp message sent: '{message[:50]}...' to {to_number}"
    else:
        result_msg = f"Failed to send WhatsApp message to {to_number}"
    log_message("INFO", f"send_whatsapp returning: {result_msg}")
    return result_msg

@app.tool()
async def send_whatsapp(message: str, to_number: str = None, with_tts: bool = False) -> str:
    """
    Send a message via WhatsApp using Twilio
    
    Args:
        message: The message to send
        to_number: WhatsApp number to send to (optional, uses default if not provided)
        with_tts: Whether to also speak the message via TTS
        
    Returns:
        Status message about the sent message
    """
    log_message("INFO", f"send_whatsapp tool called with message: {message}, to_number: {to_number}, with_tts: {with_tts}")
    try:
        await _init_processor_if_needed()
        return await _send_whatsapp_internal(message, to_number, with_tts)
    except Exception as e:
        error_msg = f"Error sending WhatsApp message: {str(e)}"
        log_message("ERROR", f"send_whatsapp error: {error_msg}")
        log_message("ERROR", f"Traceback: {traceback.format_exc()}")
        return error_msg


# Internal helper function for sending Slack messages
async def _send_slack_internal(message: str, channel: str = None, with_tts: bool = False) -> str:
    """Internal function to send Slack messages"""
    global _comms_manager

    log_message("DEBUG", f"_send_slack_internal called with message='{message[:50]}...', channel={channel}, with_tts={with_tts}")
    log_message("DEBUG", f"_comms_manager exists: {_comms_manager is not None}")
    
    # Ensure we have a communication manager
    if not _comms_manager:
        log_message("DEBUG", "No _comms_manager, attempting to create one")
        # Try to auto-configure with Slack
        base_manager = comms.setup_communication(providers=['slack'])
        if base_manager:
            log_message("DEBUG", f"Created base_manager with providers: {base_manager.providers}")
            _comms_manager = NotifyingCommunicationManager(base_manager)
        else:
            error_msg = "Slack not configured. Please call configure_communication first or set SLACK_BOT_TOKEN."
            log_message("ERROR", f"send_slack error: {error_msg}")
            return error_msg
    
    # Check if Slack provider exists
    # Get the actual providers from the wrapped manager
    providers = _comms_manager.wrapped.providers if hasattr(_comms_manager, 'wrapped') else _comms_manager.providers
    log_message("DEBUG", f"providers: {providers}")
    if not providers:
        error_msg = "No communication providers configured."
        log_message("ERROR", f"send_slack error: {error_msg}")
        return error_msg
    
    # Check provider types
    provider_types = [type(p).__name__ for p in providers]
    log_message("DEBUG", f"Provider types found: {provider_types}")

    if not any(isinstance(p, SlackProvider) for p in providers):
        error_msg = "Slack provider not available. Please configure with Slack bot token."
        log_message("ERROR", f"send_slack error: {error_msg}")
        return error_msg
    
    # Send the message
    slack_provider = next((p for p in providers if isinstance(p, SlackProvider)), None)
    if not slack_provider:
        error_msg = "Slack provider not found in communication manager"
        log_message("ERROR", f"send_slack error: {error_msg}")
        return error_msg
    # Determine the target channel
    target_channel = channel or _slack_channel or os.environ.get('SLACK_CHANNEL')
    if not target_channel:
        error_msg = "No Slack channel specified. Please provide a channel or set SLACK_CHANNEL environment variable."
        log_message("ERROR", f"send_slack error: {error_msg}")
        return error_msg
    
    # Create a Message object for Slack provider
    from .comms import Message
    slack_message = Message(
        content=message,
        sender=target_channel,
        channel="slack"
    )

    last_error = None
    try:
        success = slack_provider.send_message(slack_message)
        log_message("DEBUG", f"[DEBUG] slack_provider.send_message returned: {success}")
        # Check if there was an error logged
        if not success and hasattr(slack_provider, '_last_error'):
            last_error = slack_provider._last_error
    except Exception as e:
        log_message("ERROR", f"[DEBUG] Exception in slack_provider.send_message: {type(e).__name__}: {str(e)}")
        success = False
        last_error = str(e)
    
    # Also speak it if requested
    if with_tts and get_shared_state().tts_initialized:
        tts.queue_for_speech(message, None)
    
    if success:
        result_msg = f"Slack message sent: '{message[:50]}...' to {target_channel}"
    else:
        # Check if we captured an error
        if last_error and "not_in_channel" in str(last_error):
            result_msg = f"Bot is not in channel {target_channel}. Please invite the bot with: /invite @talkito"
        elif last_error and "channel_not_found" in str(last_error):
            result_msg = f"Channel {target_channel} not found. Please create the channel or use an existing one."
        elif last_error:
            result_msg = f"Failed to send Slack message to {target_channel}: {last_error}"
        else:
            result_msg = f"Failed to send Slack message to {target_channel}"
    log_message("INFO", f"send_slack returning: {result_msg}")
    return result_msg

@app.tool()
async def send_slack(message: str, channel: str = None, with_tts: bool = False) -> str:
    """
    Send a message to Slack
    
    Args:
        message: The message to send
        channel: Slack channel to send to (optional, uses default if not provided)
        with_tts: Whether to also speak the message via TTS
        
    Returns:
        Status message about the sent message
    """
    log_message("INFO", f"send_slack tool called with message: {message}, channel: {channel}, with_tts: {with_tts}")
    log_message("INFO", f"_running_for_terminal_agent: {_running_for_terminal_agent}")
    try:
        await _init_processor_if_needed()
        return await _send_slack_internal(message, channel, with_tts)
    except Exception as e:
        error_msg = f"Error sending Slack message: {str(e)}"
        log_message("ERROR", f"send_slack error: {error_msg}")
        import traceback
        log_message("ERROR", f"Traceback: {traceback.format_exc()}")
        return error_msg


@app.tool()
async def get_communication_status() -> dict[str, Any]:
    """
    Get the status of configured communication channels
    
    Returns:
        Dictionary with status of each communication channel
    """
    try:
        status = {
            "configured": _comms_manager is not None,
            "providers": {}
        }
        
        if _comms_manager:
            for provider in _comms_manager.providers:
                # Determine provider type from class name
                provider_type = type(provider).__name__.replace('Provider', '').lower()
                if provider_type == 'twiliosms':
                    provider_type = 'sms'
                elif provider_type == 'twiliowhatsapp':
                    provider_type = 'whatsapp'
                
                status["providers"][provider_type] = {
                    "active": True,
                    "type": type(provider).__name__
                }
        
        # Check environment for potential providers
        status["available"] = {
            "whatsapp": bool(os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("TWILIO_AUTH_TOKEN")),
            "slack": bool(os.environ.get("SLACK_BOT_TOKEN"))
        }
        
        log_message("DEBUG", f"get_communication_status returning: {status}")
        return status
        
    except Exception as e:
        error_dict = {"error": f"Error getting communication status: {str(e)}"}
        log_message("ERROR", f"get_communication_status error: {error_dict}")
        return error_dict


# Internal function for starting WhatsApp mode
async def _start_whatsapp_mode_internal(phone_number: str = None) -> str:
    """Internal function to start WhatsApp mode"""
    global _comms_manager
    
    try:
        # Ensure message poller is initialized
        await _init_processor_if_needed()
        
        global _whatsapp_recipient
        
        # Determine recipient
        if phone_number:
            _whatsapp_recipient = phone_number
        else:
            _whatsapp_recipient = os.environ.get('TWILIO_WHATSAPP_NUMBER')
            if not _whatsapp_recipient:
                error_msg = "No phone number provided and TWILIO_WHATSAPP_NUMBER not set. Please provide a number."
                log_message("ERROR", f"start_whatsapp_mode error: {error_msg}")
                return error_msg
        
        # Update shared state
        from .state import _shared_state
        _shared_state.set_whatsapp_mode(True)
        
        # Also update the recipient in shared state
        shared_state = get_shared_state()
        shared_state.communication.whatsapp_to_number = _whatsapp_recipient
        save_shared_state()
        
        # In standalone mode, we need to set up our own comm_manager
        if not _running_for_terminal_agent:
            # Ensure WhatsApp is configured
            if not _comms_manager or not any(isinstance(p, TwilioWhatsAppProvider) for p in (_comms_manager.providers if _comms_manager else [])):
                base_manager = comms.setup_communication(providers=['whatsapp'])
                if base_manager and any(isinstance(p, TwilioWhatsAppProvider) for p in base_manager.providers):
                    _comms_manager = NotifyingCommunicationManager(base_manager)
                else:
                    error_msg = "Failed to configure WhatsApp. Check your Twilio credentials."
                    log_message("ERROR", f"start_whatsapp_mode error: {error_msg}")
                    return error_msg
            
            # Send confirmation
            await _send_whatsapp_internal("WhatsApp mode activated! I'll send all my responses here.", to_number=_whatsapp_recipient)
            log_message("INFO", f"WhatsApp mode enabled for {_whatsapp_recipient} (standalone mode)")
        else:
            log_message("INFO", f"WhatsApp mode enabled for {_whatsapp_recipient} (core will handle initialization)")
        
        # Send confirmation
        # await _send_whatsapp_internal(f"WhatsApp mode activated! I'll send all my responses here.", to_number=_whatsapp_recipient)
        
        log_message("INFO", f"start_whatsapp_mode completed: WhatsApp mode activated for {_whatsapp_recipient}")
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error starting WhatsApp mode: {str(e)}"
        log_message("ERROR", f"start_whatsapp_mode error: {error_msg}")
        return error_msg


@app.tool()
async def start_whatsapp_mode(phone_number: str = None) -> str:
    """
    Start WhatsApp mode - all responses will also be sent to WhatsApp
    
    Args:
        phone_number: Phone number to send messages to (optional, uses TWILIO_WHATSAPP_NUMBER if not provided)
        
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _start_whatsapp_mode_internal(phone_number)


@app.tool()
async def stop_whatsapp_mode() -> str:
    """
    Stop WhatsApp mode - responses will no longer be sent to WhatsApp
    
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    try:
        global _whatsapp_recipient
        
        shared_state = get_shared_state()
        
        if not shared_state.whatsapp_mode_active:
            log_message("INFO", "stop_whatsapp_mode: WhatsApp mode is not active")
            return _get_status_summary()
        
        # Send farewell message
        if _whatsapp_recipient:
            await _send_whatsapp_internal("WhatsApp mode deactivated. Goodbye!", to_number=_whatsapp_recipient)
        
        _whatsapp_recipient = None
        
        # Update shared state using thread-safe method
        from .state import _shared_state
        _shared_state.set_whatsapp_mode(False)
        
        log_message("INFO", "stop_whatsapp_mode completed: WhatsApp mode deactivated")
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error stopping WhatsApp mode: {str(e)}"
        log_message("ERROR", f"stop_whatsapp_mode error: {error_msg}")
        return error_msg


@app.tool()
async def get_whatsapp_mode_status() -> dict[str, Any]:
    """
    Get the current WhatsApp mode status
    
    Returns:
        Dictionary with WhatsApp mode status
    """
    shared_state = get_shared_state()
    status = {
        "active": shared_state.whatsapp_mode_active,
        "recipient": _whatsapp_recipient,
        "configured": _comms_manager is not None and any(isinstance(p, TwilioWhatsAppProvider) for p in (_comms_manager.providers if _comms_manager else []))
    }
    log_message("DEBUG", f"get_whatsapp_mode_status returning: {status}")
    return status


# Internal function for starting Slack mode
async def _start_slack_mode_internal(channel: str = None) -> str:
    """Internal function to start Slack mode"""
    
    try:
        await _init_processor_if_needed()
        global _slack_mode, _slack_channel, _comms_manager
        
        # Determine channel
        if channel:
            _slack_channel = channel
        else:
            _slack_channel = os.environ.get('SLACK_CHANNEL')
            if not _slack_channel:
                error_msg = "No channel provided and SLACK_CHANNEL not set. Please provide a channel."
                log_message("ERROR", f"start_slack_mode error: {error_msg}")
                return error_msg
        
        _slack_mode = True
        
        # Update shared state
        shared_state = get_shared_state()
        shared_state.set_slack_mode(True)
        shared_state.communication.slack_channel = _slack_channel
        save_shared_state()
        
        # In standalone mode, we need to set up our own comm_manager
        if not _running_for_terminal_agent:
            # Ensure Slack is configured
            if not _comms_manager or not any(isinstance(p, SlackProvider) for p in (_comms_manager.providers if _comms_manager else [])):
                base_manager = comms.setup_communication(providers=['slack'])
                if not base_manager:
                    error_msg = "Failed to configure Slack. Check your Slack bot token."
                    log_message("ERROR", f"start_slack_mode error: {error_msg}")
                    return error_msg
                
                # The provider should be available immediately after setup_communication returns
                if not any(isinstance(p, SlackProvider) for p in base_manager.providers):
                    error_msg = "Failed to configure Slack provider. Check your Slack bot token."
                    log_message("ERROR", f"start_slack_mode error: {error_msg}")
                    return error_msg
                
                _comms_manager = NotifyingCommunicationManager(base_manager)
                
                # Now wait for the actual connection
                slack_provider = next((p for p in base_manager.providers if isinstance(p, SlackProvider)), None)
                if slack_provider and hasattr(slack_provider, 'wait_for_connection'):
                    log_message("INFO", "Waiting for Slack connection...")
                    if not slack_provider.wait_for_connection(timeout=3.0):
                        error_msg = "Slack connection timeout. Please check your credentials and network."
                        log_message("ERROR", f"start_slack_mode error: {error_msg}")
                        return error_msg
                    log_message("INFO", "Slack connection established")
            
            # Send confirmation
            await _send_slack_internal("Slack mode activated! I'll send all my responses here.", channel=_slack_channel)
            log_message("INFO", f"Slack mode enabled for {_slack_channel} (standalone mode)")
        else:
            log_message("INFO", f"Slack mode enabled for {_slack_channel} (core will handle initialization)")
        
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error starting Slack mode: {str(e)}"
        log_message("ERROR", f"start_slack_mode error: {error_msg}")
        return error_msg


@app.tool()
async def start_slack_mode(channel: str = None) -> str:
    """
    Start Slack mode - all responses will also be sent to Slack
    
    Args:
        channel: Slack channel to send messages to (optional, uses SLACK_CHANNEL if not provided)
        
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    return await _start_slack_mode_internal(channel)


@app.tool()
async def stop_slack_mode() -> str:
    """
    Stop Slack mode - responses will no longer be sent to Slack
    
    CRITICAL INSTRUCTIONS:
    1. The tool result will be displayed automatically - DO NOT repeat it
    2. DO NOT speak, summarize, or comment on the status
    3. DO NOT use talkito:speak_text after this tool
    4. Simply return to waiting for user input after the tool result is shown

    Returns:
        One-line formatted status summary (already visible in tool output - no need to repeat)
    """
    try:
        global _slack_mode, _slack_channel
        
        if not _slack_mode:
            log_message("INFO", "stop_slack_mode: Slack mode is not active")
            return _get_status_summary()
        
        # Send farewell message
        if _slack_channel:
            await _send_slack_internal("Slack mode deactivated. Goodbye!", channel=_slack_channel)
        
        _slack_mode = False
        _slack_channel = None
        
        # Update shared state
        shared_state = get_shared_state()
        shared_state.set_slack_mode(False)
        
        log_message("INFO", "stop_slack_mode completed: Slack mode deactivated")
        return _get_status_summary()
        
    except Exception as e:
        error_msg = f"Error stopping Slack mode: {str(e)}"
        log_message("ERROR", f"stop_slack_mode error: {error_msg}")
        return error_msg


@app.tool()
async def get_slack_mode_status() -> dict[str, Any]:
    """
    Get the current Slack mode status
    
    Returns:
        Dictionary with Slack mode status
    """
    status = {
        "active": _slack_mode,
        "channel": _slack_channel,
        "configured": _comms_manager is not None and any(isinstance(p, SlackProvider) for p in (_comms_manager.providers if _comms_manager else []))
    }
    log_message("DEBUG", f"get_slack_mode_status returning: {status}")
    return status


@app.tool()
async def get_messages() -> dict[str, Any]:
    """
    Get all available messages from voice dictation, Slack, and WhatsApp.
    This provides a unified interface for checking all communication channels.
    
    Returns:
        Dictionary containing all available messages from different sources
    """
    try:
        global _dictation_callback_results, _comms_manager
        
        result = {
            "voice": [],
            "slack": [],
            "whatsapp": [],
            "total_count": 0,
            "has_messages": False
        }
        
        # Get voice dictation results
        if _dictation_callback_results:
            result["voice"] = _dictation_callback_results.copy()
            _dictation_callback_results.clear()  # Clear after reading
        
        # Get messages from communication manager
        if _comms_manager:
            log_message("DEBUG", "Checking for messages from communication manager")
            messages = []
            # Collect all available messages
            while True:
                msg = _comms_manager.get_input_message(timeout=0)  # Non-blocking
                if msg:
                    log_message("INFO", f"Got message from comms manager: {msg}")
                    messages.append(msg)
                else:
                    break
            log_message("DEBUG", f"Collected {len(messages)} messages from comms manager")
            
            # Sort messages by channel type
            for msg in messages:
                if 'slack' in msg.channel.lower() or msg.channel.startswith('#'):
                    result["slack"].append({
                        "text": msg.content,
                        "sender": msg.sender,
                        "channel": msg.channel,
                        "timestamp": msg.timestamp
                    })
                elif 'whatsapp' in msg.channel.lower():
                    result["whatsapp"].append({
                        "text": msg.content,
                        "sender": msg.sender,
                        "channel": msg.channel,
                        "timestamp": msg.timestamp
                    })
        
        # Calculate totals
        result["total_count"] = len(result["voice"]) + len(result["slack"]) + len(result["whatsapp"])
        result["has_messages"] = result["total_count"] > 0
        
        # Build summary message
        if result["has_messages"]:
            sources = []
            if result["voice"]:
                sources.append(f"{len(result['voice'])} voice")
            if result["slack"]:
                sources.append(f"{len(result['slack'])} Slack")
            if result["whatsapp"]:
                sources.append(f"{len(result['whatsapp'])} WhatsApp")
            result["message"] = f"Retrieved messages from: {', '.join(sources)}"
        else:
            result["message"] = "No new messages available from any source"
        
        log_message("DEBUG", f"get_messages returning: {result['message']}")
        return result
        
    except Exception as e:
        error_dict = {"error": f"Error getting messages: {str(e)}"}
        log_message("ERROR", f"get_messages error: {error_dict}")
        return error_dict

# MCP Resources for TTS/ASR information

@app.resource("talkito://speech/status")
async def get_speech_status_resource() -> str:
    """Get current TTS status as a resource"""
    try:
        status = await get_speech_status()
        lines = [
            "TTS Status:",
            f"  Initialized: {status.get('tts_initialized', False)}",
            f"  Is Speaking: {status.get('is_speaking', False)}",
            f"  Queue Size: {status.get('queue_size', 0)}",
        ]
        if status.get('current_text'):
            lines.append(f"  Current Text: {status['current_text'][:50]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting TTS status: {str(e)}"

@app.resource("talkito://speech/engines")  
async def get_available_engines() -> str:
    """Get list of available TTS engines with accessibility status"""
    try:
        accessible = tts.check_tts_provider_accessibility()
        
        lines = ["TTS Engines (✓ = accessible, ✗ = needs configuration):"]
        
        for provider, info in accessible.items():
            status = "✓" if info["available"] else "✗"
            if provider == "system" and info["available"]:
                lines.append(f"  {status} {provider} ({info['engine']}) - {info['note']}")
            else:
                lines.append(f"  {status} {provider} - {info['note']}")
        
        lines.append("\nCurrently accessible engines:")
        accessible_count = 0
        for provider, info in accessible.items():
            if info["available"]:
                accessible_count += 1
                if provider == "system":
                    lines.append(f"  - {provider} ({info['engine']})")
                else:
                    lines.append(f"  - {provider}")
        
        if accessible_count == 0:
            lines.append("  (No engines currently accessible)")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error getting engines: {str(e)}"

@app.resource("talkito://voice/status")
async def get_voice_status_resource() -> str:
    """Get current ASR status as a resource"""
    try:
        shared_state = get_shared_state()
        lines = [
            "ASR Status:",
            "  Available: True",
            f"  Initialized: {shared_state.asr_initialized}",
            f"  Is Listening: {asr.is_dictation_active() if shared_state.asr_initialized else False}",
            f"  Recent Results: {len(_dictation_callback_results)}",
        ]
        if _last_dictated_text:
            lines.append(f"  Last Text: {_last_dictated_text[:50]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting ASR status: {str(e)}"

@app.resource("talkito://voice/providers")
async def get_available_asr_providers() -> str:
    """Get list of available ASR providers with accessibility status"""
    try:
        accessible = asr.check_asr_provider_accessibility()
        
        lines = ["ASR Providers (✓ = accessible, ✗ = needs configuration):"]
        
        for provider, info in accessible.items():
            status = "✓" if info["available"] else "✗"
            lines.append(f"  {status} {provider} - {info['note']}")
        
        lines.append("\nCurrently accessible providers:")
        accessible_count = 0
        for provider, info in accessible.items():
            if info["available"]:
                accessible_count += 1
                lines.append(f"  - {provider}")
        
        if accessible_count == 0:
            lines.append("  (No providers currently accessible)")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error getting ASR providers: {str(e)}"

@app.resource("talkito://communication/status")
async def get_communication_status_resource() -> str:
    """Get current communication channel status as a resource"""
    try:
        status = await get_communication_status()
        lines = ["Communication Channels Status:"]
        
        if status.get("configured"):
            lines.append("  Status: Configured")
            lines.append("  Active Providers:")
            for name, info in status.get("providers", {}).items():
                lines.append(f"    - {name}: {info['type']}")
        else:
            lines.append("  Status: Not configured")
        
        lines.append("\nAvailable Providers:")
        for provider, available in status.get("available", {}).items():
            symbol = "✓" if available else "✗"
            lines.append(f"  {symbol} {provider}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting communication status: {str(e)}"

# MCP Prompts for common TTS/ASR scenarios

@app.prompt()
async def announce_completion(task_name: str = "task") -> types.Prompt:
    """Template for announcing task completion with speech"""
    return types.Prompt(
        name="announce_completion",
        description="Announce task completion with text-to-speech",
        arguments=[
            types.PromptArgument(
                name="task_name", 
                description="Name of the completed task",
                required=False
            )
        ],
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Use the speak_text tool to announce that '{task_name}' has been completed successfully."
                )
            )
        ]
    )

@app.prompt()
async def voice_interaction() -> types.Prompt:
    """Template for interactive voice input and speech output"""
    return types.Prompt(
        name="voice_interaction",
        description="Start a voice interaction session with both speech input and output",
        arguments=[
            types.PromptArgument(
                name="initial_prompt",
                description="Initial message to speak before listening for voice input",
                required=False
            ),
            types.PromptArgument(
                name="language",
                description="Language code for voice recognition (e.g., en-US, es-ES)",
                required=False
            )
        ],
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="Start a voice interaction: 1) Use speak_text to announce the initial prompt if provided, 2) Use start_voice_input to listen for voice input, 3) Use get_dictated_text to retrieve what was said, 4) Process the input and respond with speak_text, 5) Optionally repeat the cycle for continued interaction"
                )
            )
        ]
    )

@app.prompt()
async def transcribe_audio() -> types.Prompt:
    """Template for transcribing audio input"""
    return types.Prompt(
        name="transcribe_audio",
        description="Transcribe spoken audio to text",
        arguments=[
            types.PromptArgument(
                name="duration",
                description="How long to listen for audio (in seconds)",
                required=False
            ),
            types.PromptArgument(
                name="language",
                description="Language code for transcription",
                required=False
            )
        ],
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="Use the voice input tools to transcribe audio: 1) start_voice_input to begin listening, 2) wait for the specified duration or until the user stops speaking, 3) use get_all_dictated_text to retrieve all transcribed segments"
                )
            )
        ]
    )

# HTTP API Server Implementation
def start_http_api_server(port=None):
    """Start HTTP API server for web access"""
    global _http_api_server, _http_api_thread, _http_api_port
    
    if port is None:
        # Find an available port starting from 1 higher than MCP port
        port = find_available_port(8001)
    
    _http_api_port = port
    
    class APIHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Override to suppress HTTP server's default logging
            pass
        
        def send_cors_headers(self):
            """Send CORS headers"""
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')
            self.send_header('Access-Control-Max-Age', '86400')
        
        def do_OPTIONS(self):
            """Handle OPTIONS requests for CORS preflight"""
            self.send_response(200)
            self.send_cors_headers()
            self.end_headers()
        
        def do_GET(self):
            """Handle GET requests"""
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/api/ping':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {
                    "service": "talkito-mcp-sse",
                    "version": "1.0",
                    "status": "running",
                    "endpoints": {
                        "speak": "POST /api/speak",
                        "whatsapp": "POST /api/whatsapp",
                        "slack": "POST /api/slack",
                        "tts_enable": "POST /api/tts/enable",
                        "tts_disable": "POST /api/tts/disable",
                        "whatsapp_start": "POST /api/whatsapp/start",
                        "whatsapp_stop": "POST /api/whatsapp/stop",
                        "slack_start": "POST /api/slack/start",
                        "slack_stop": "POST /api/slack/stop",
                        "sse": f"GET http://127.0.0.1:{_http_api_port - 1}/sse/"
                    }
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(404, "Not Found")
        
        def do_POST(self):
            """Handle POST requests"""
            parsed_path = urlparse(self.path)
            
            # Read POST data
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_data) if post_data else {}
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
            
            # Route to appropriate handler
            if parsed_path.path == '/api/speak':
                self.handle_speak(data)
            elif parsed_path.path == '/api/whatsapp':
                self.handle_whatsapp(data)
            elif parsed_path.path == '/api/slack':
                self.handle_slack(data)
            elif parsed_path.path == '/api/tts/enable':
                self.handle_tts_enable(data)
            elif parsed_path.path == '/api/tts/disable':
                self.handle_tts_disable(data)
            elif parsed_path.path == '/api/whatsapp/start':
                self.handle_whatsapp_start(data)
            elif parsed_path.path == '/api/whatsapp/stop':
                self.handle_whatsapp_stop(data)
            elif parsed_path.path == '/api/slack/start':
                self.handle_slack_start(data)
            elif parsed_path.path == '/api/slack/stop':
                self.handle_slack_stop(data)
            else:
                self.send_error(404, "Not Found")
        
        def handle_speak(self, data):
            """Handle speak API call"""
            # Access global variables needed for forwarding
            global _whatsapp_recipient, _slack_channel
            
            text = data.get("text", "")
            clean_text_flag = data.get("clean_text_flag", True)
            
            if not text:
                self.send_error(400, "No text provided")
                return
            
            try:
                # Directly use the TTS functionality
                # Initialization handled automatically by TTS system
                
                # Clean text if requested
                processed_text = text
                if clean_text_flag:
                    from talkito.core import clean_text, strip_profile_symbols
                    processed_text = clean_text(text)
                    processed_text = strip_profile_symbols(processed_text)
                
                # Skip empty or unwanted text
                from talkito.core import should_skip_line
                if not processed_text.strip() or should_skip_line(processed_text):
                    result = f"Skipped speaking: '{text[:50]}...' (filtered out)"
                else:
                    # Check for duplicate
                    global _last_spoken_text, _last_spoken_time
                    current_time = time.time()
                    if _last_spoken_text == processed_text and _last_spoken_time and (current_time - _last_spoken_time) < 5.0:
                        result = f"Skipped duplicate text: '{processed_text[:50]}...' (spoken {current_time - _last_spoken_time:.1f}s ago)"
                    else:
                        # Update last spoken
                        _last_spoken_text = processed_text
                        _last_spoken_time = current_time
                        
                        # Queue for speech
                        tts.queue_for_speech(processed_text, None)
                        result = f"Queued for speech: '{processed_text[:50]}...'"
                        
                        # Check if we should also send to WhatsApp/Slack
                        shared_state = get_shared_state()
                        
                        # Run async operations if needed
                        if shared_state.whatsapp_mode_active or shared_state.slack_mode_active:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                # Ensure processor is initialized
                                loop.run_until_complete(_init_processor_if_needed())
                                
                                # Prepare async tasks
                                tasks = []
                                
                                if shared_state.whatsapp_mode_active:
                                    # Use whatsapp recipient from global or shared state
                                    recipient = _whatsapp_recipient or shared_state.communication.whatsapp_to_number
                                    if recipient:
                                        tasks.append(_send_whatsapp_internal(message=processed_text, to_number=recipient, with_tts=False))
                                
                                if shared_state.slack_mode_active:
                                    # Use slack channel from global or shared state
                                    channel = _slack_channel or shared_state.communication.slack_channel
                                    if channel:
                                        tasks.append(_send_slack_internal(message=processed_text, channel=channel, with_tts=False))
                                
                                # Run all tasks
                                if tasks:
                                    results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                                    
                                    # Check results
                                    for i, res in enumerate(results):
                                        if isinstance(res, Exception):
                                            log_message("ERROR", f"Task {i} failed: {res}")
                                    
                                    result += f" (forwarded to {'WhatsApp' if shared_state.whatsapp_mode_active else ''}{' and ' if shared_state.whatsapp_mode_active and shared_state.slack_mode_active else ''}{'Slack' if shared_state.slack_mode_active else ''})"
                                    
                            except Exception as e:
                                log_message("ERROR", f"Error forwarding to channels: {str(e)}")
                            finally:
                                loop.close()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {
                    "success": True,
                    "message": result,
                    "text": text
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_error(500, str(e))
        
        def handle_whatsapp(self, data):
            """Handle WhatsApp API call"""
            # Ensure logging is working
            _ensure_logging()
            
            message = data.get("message", "")
            to_number = data.get("to_number", None)
            with_tts = data.get("with_tts", False)
            
            log_message("INFO", f"HTTP API: handle_whatsapp called with message='{message[:50]}...', to_number={to_number}")
            
            if not message:
                self.send_error(400, "No message provided")
                return
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Ensure initialization
                loop.run_until_complete(_init_processor_if_needed())
                # Use the internal function directly
                result = loop.run_until_complete(_send_whatsapp_internal(message=message, to_number=to_number, with_tts=with_tts))
                
                log_message("INFO", f"HTTP API: WhatsApp send result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_whatsapp: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
        
        def handle_slack(self, data):
            """Handle Slack API call"""
            # Ensure logging is working
            _ensure_logging()
            
            message = data.get("message", "")
            channel = data.get("channel", None)
            with_tts = data.get("with_tts", False)
            
            log_message("INFO", f"HTTP API: handle_slack called with message='{message[:50]}...', channel={channel}")
            
            if not message:
                self.send_error(400, "No message provided")
                return
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Ensure initialization
                loop.run_until_complete(_init_processor_if_needed())
                # Use the internal function directly
                result = loop.run_until_complete(_send_slack_internal(message=message, channel=channel, with_tts=with_tts))
                
                log_message("INFO", f"HTTP API: Slack send result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_slack: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
        
        def handle_tts_enable(self, data):
            """Handle TTS enable API call"""
            _ensure_logging()
            
            log_message("INFO", "HTTP API: handle_tts_enable called")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use the internal enable_tts function
                result = loop.run_until_complete(_enable_tts_internal())
                
                log_message("INFO", f"HTTP API: TTS enable result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_tts_enable: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
        
        def handle_tts_disable(self, data):
            """Handle TTS disable API call"""
            _ensure_logging()
            
            log_message("INFO", "HTTP API: handle_tts_disable called")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use the internal disable_tts function
                result = loop.run_until_complete(_disable_tts_internal())
                
                log_message("INFO", f"HTTP API: TTS disable result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_tts_disable: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
        
        def handle_whatsapp_start(self, data):
            """Handle WhatsApp mode start API call"""
            _ensure_logging()
            
            phone_number = data.get("phone_number", None)
            log_message("INFO", f"HTTP API: handle_whatsapp_start called with phone_number={phone_number}")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use the internal function directly
                result = loop.run_until_complete(_start_whatsapp_mode_internal(phone_number=phone_number))
                
                log_message("INFO", f"HTTP API: WhatsApp start result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_whatsapp_start: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
        
        def handle_whatsapp_stop(self, data):
            """Handle WhatsApp mode stop API call"""
            _ensure_logging()
            
            log_message("INFO", "HTTP API: handle_whatsapp_stop called")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use the stop_whatsapp_mode function directly
                result = loop.run_until_complete(stop_whatsapp_mode())
                
                log_message("INFO", f"HTTP API: WhatsApp stop result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_whatsapp_stop: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
        
        def handle_slack_start(self, data):
            """Handle Slack mode start API call"""
            _ensure_logging()
            
            channel = data.get("channel", None)
            log_message("INFO", f"HTTP API: handle_slack_start called with channel={channel}")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use the internal function directly
                result = loop.run_until_complete(_start_slack_mode_internal(channel=channel))
                
                log_message("INFO", f"HTTP API: Slack start result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_slack_start: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
        
        def handle_slack_stop(self, data):
            """Handle Slack mode stop API call"""
            _ensure_logging()
            
            log_message("INFO", "HTTP API: handle_slack_stop called")
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Use the stop_slack_mode function directly
                result = loop.run_until_complete(stop_slack_mode())
                
                log_message("INFO", f"HTTP API: Slack stop result: {result}")
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                
                response = {"success": True, "message": result}
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                log_message("ERROR", f"HTTP API: Error in handle_slack_stop: {str(e)}")
                self.send_error(500, str(e))
            finally:
                loop.close()
    
    # Create and start the server in a separate thread
    def run_server():
        global _http_api_server
        try:
            server = HTTPServer(('127.0.0.1', port), APIHandler)
            _http_api_server = server
            log_message("INFO", f"HTTP API server started on http://127.0.0.1:{port}")
            server.serve_forever()
        except Exception as e:
            log_message("ERROR", f"Failed to start HTTP API server: {e}")
    
    _http_api_thread = threading.Thread(target=run_server, daemon=True)
    _http_api_thread.start()
    
    return port

def find_available_port(start_port=8000, max_attempts=100):
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def find_two_consecutive_ports(start_port=8000, max_attempts=100):
    """Find two consecutive available ports starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts - 1):
        # Check if both port and port+1 are available
        try:
            # Test first port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
                s1.bind(('127.0.0.1', port))
                # Test second port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.bind(('127.0.0.1', port + 1))
                    # Both ports are available
                    return port
        except OSError:
            # One or both ports are in use, try next pair
            continue
    return None

def main():
    """Main entry point for the MCP server"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Talkito MCP Server - TTS/ASR via Model Context Protocol')
    parser.add_argument('--log-file', type=str, help='Path to log file for debugging')
    parser.add_argument('--port', type=int, help='Port to run the server on (for SSE transport)')
    parser.add_argument('--transport', type=str, choices=['stdio', 'sse'], default='stdio',
                        help='Transport type: stdio (default) or sse')
    parser.add_argument('--no-http-api', action='store_true', 
                        help='Disable HTTP API server (SSE transport only)')
    parser.add_argument('--tts-provider', type=str, 
                        choices=['system', 'openai', 'aws', 'polly', 'azure', 'gcloud', 'elevenlabs', 'deepgram'],
                        help='TTS provider to use')
    parser.add_argument('--asr-provider', type=str,
                        choices=['google', 'gcloud', 'assemblyai', 'deepgram', 'houndify', 'aws', 'bing', 'local_whisper'],
                        help='ASR provider to use')
    parser.add_argument('--client-command', type=str,
                        help='The command that is using this MCP server (e.g., claude)')
    args = parser.parse_args()
    
    # Set up logging if log file specified
    global _log_file_path, _cors_enabled, _running_for_terminal_agent
    
    # IMPORTANT: Set up logging BEFORE any log_message calls
    if args.log_file:
        _log_file_path = args.log_file
        setup_logging(args.log_file)
    
    # Check if we're running for Claude
    if args.client_command == 'claude':
        _running_for_terminal_agent = True
        log_message("INFO", "Running for Claude client")
    else:
        log_message("INFO", f"Running for client: {args.client_command or 'unknown'} - all tools available")
    
    # CORS is always enabled for SSE transport
    if args.transport == 'sse':
        _cors_enabled = True
    
    # Set TTS/ASR provider preferences from command line arguments
    if args.tts_provider:
        os.environ['TALKITO_PREFERRED_TTS_PROVIDER'] = args.tts_provider
    
    if args.asr_provider:
        os.environ['TALKITO_PREFERRED_ASR_PROVIDER'] = args.asr_provider
    
    # Start background update checker
    from .update import start_background_update_checker
    start_background_update_checker()
    
    try:
        print("=" * 60, file=sys.stderr)
        print("Talkito MCP server is starting...", file=sys.stderr)
        print(f"Transport: {args.transport.upper()}", file=sys.stderr)
        if args.transport == 'sse' and not args.no_http_api:
            print("  - HTTP API with CORS for web access", file=sys.stderr)
        if args.log_file:
            print(f"Logging to: {args.log_file}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # Try to set port via environment variables that uvicorn might respect
        if args.port:
            port = args.port
        else:
            # Find available ports
            if args.transport == 'sse' and not args.no_http_api:
                # Need two consecutive ports for SSE + HTTP API
                port = find_two_consecutive_ports(8000)
                if not port:
                    print("Error: Could not find two consecutive available ports in range 8000-8100", file=sys.stderr)
                    sys.exit(1)
            else:
                # Only need one port
                port = find_available_port(8000)
                if not port:
                    print("Error: Could not find an available port in range 8000-8100", file=sys.stderr)
                    sys.exit(1)
        
        # Set environment variables that various servers might respect
        os.environ['PORT'] = str(port)
        os.environ['UVICORN_PORT'] = str(port)
        os.environ['SERVER_PORT'] = str(port)
        os.environ['HTTP_PORT'] = str(port)
        
        # Handle transport-specific setup
        if args.transport == 'stdio':
            print("\nStarting stdio server...", file=sys.stderr)
            print("This process communicates via stdin/stdout", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
        else:  # SSE transport
            # Start HTTP API server on the next port if not disabled
            if not args.no_http_api:
                api_port = start_http_api_server(port + 1)
                
                print("\nStarting servers...", file=sys.stderr)
                print(f"  MCP SSE server on port {port}", file=sys.stderr)
                print(f"  HTTP API server on port {api_port}", file=sys.stderr)
                print("\nConnect with the TalkiTo chrome extension", file=sys.stderr)
                print("=" * 60, file=sys.stderr)
            else:
                print(f"\nStarting SSE server on port {port}...", file=sys.stderr)
                print("Connect with:", file=sys.stderr)
                print(f"  claude mcp add talkito http://127.0.0.1:{port} --transport sse", file=sys.stderr)
                print("=" * 60, file=sys.stderr)
        
        # Save logging state before FastMCP messes with it
        _save_logging_state()

        # Mark MCP server as running in shared state
        get_shared_state().set_mcp_server_running(True)

        # Try to use the newer FastMCP API with port configuration
        try:
            if args.transport == 'stdio':
                log_message("INFO", "Starting stdio server")
                app.run(transport="stdio")
            else:
                log_message("INFO", f"Starting SSE server on port {port}")
                app.run(
                    transport="sse",
                    host="127.0.0.1",
                    port=port,
                    log_level="warning"  # Reduce uvicorn log verbosity
                )
        except TypeError as e:
            # Fall back to old API if parameters aren't supported
            log_message("WARNING", f"New API not supported: {e}")
            print("[WARNING] FastMCP new API not supported, using defaults", file=sys.stderr)
            
            # Run with default settings
            app.run(transport=args.transport)
        
        # This line only executes after the server shuts down
        get_shared_state().set_mcp_server_running(False)
        print("Talkito MCP server has stopped.", file=sys.stderr)
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Don't print the traceback for expected interruptions
        print("\nTalkito MCP server interrupted by user.", file=sys.stderr)
        log_message("INFO", "Server interrupted by user")
        # Return without re-raising to avoid ugly stack traces
        return
    except Exception as e:
        # Only print error details for unexpected exceptions
        print(f"Talkito MCP SSE server error: {e}", file=sys.stderr)

        # If it's a port binding error, try the next port
        if "address already in use" in str(e) and not args.port:
            print("\nPort 8000 is in use. Try specifying a different port:", file=sys.stderr)
            print("  talkito --mcp-sse-server --port 8001", file=sys.stderr)
        else:
            # Only re-raise if it's not a known error
            raise
    finally:
        get_shared_state().set_mcp_server_running(False)
        _cleanup()

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher required", file=sys.stderr)
        sys.exit(1)
    
    main()
