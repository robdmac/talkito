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

"""
MCP Talk Server - Model Context Protocol server for Talk TTS/ASR functionality
Provides TTS and ASR capabilities to AI applications through MCP protocol
"""

import sys
import time
from typing import Any
import signal
import asyncio
import atexit
import os
import argparse

# Import MCP SDK
try:
    from mcp import types
    from mcp.server import FastMCP
except ImportError:
    print("Error: MCP SDK not found. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import talkito functionality
from . import tts
from . import comms
from .comms import SlackProvider, TwilioWhatsAppProvider, TwilioSMSProvider
from .core import TalkitoCore
from .logs import log_message as _base_log_message, setup_logging

# Check if ASR is available
try:
    from . import asr
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False

# Wrapper to add [MCP] prefix to all log messages
def log_message(level: str, message: str):
    """Log a message with [MCP] prefix"""
    _base_log_message(level, f"[MCP] {message}", __name__)

# Server configuration
app = FastMCP("talkito-tts-server")

# Global state
_tts_initialized = False
_asr_initialized = False
_shutdown_registered = False
_last_dictated_text = ""
_dictation_callback_results = []
_core_instance = None
_comms_manager = None  # Communication manager for WhatsApp/Slack
_whatsapp_mode = False  # Flag for WhatsApp mode
_whatsapp_recipient = None  # Current WhatsApp recipient
_slack_mode = False  # Flag for Slack mode
_slack_channel = None  # Current Slack channel

def _ensure_initialization():
    """Ensure TTS system is initialized"""
    global _tts_initialized, _shutdown_registered, _core_instance, _comms_manager
    
    if not _tts_initialized:
        # Create core instance
        _core_instance = TalkitoCore(verbosity_level=0)
        
        # Log initialization
        log_message("INFO", "Initializing TTS system")
        
        # Select the best available TTS provider
        best_provider = tts.select_best_tts_provider()
        
        if best_provider == 'system':
            # Use system TTS engine
            engine = tts.detect_tts_engine()
            if engine == "none":
                raise RuntimeError("No TTS engine found. Please install espeak, festival, flite (Linux) or use macOS")
            tts.start_tts_worker(engine, auto_skip_tts=False)
        else:
            # Configure and use the selected provider
            tts_config = {'provider': best_provider}
            if not tts.configure_tts_from_dict(tts_config):
                # Fall back to system if configuration fails
                log_message("WARNING", f"Failed to configure {best_provider}, falling back to system")
                engine = tts.detect_tts_engine()
                if engine == "none":
                    raise RuntimeError("No TTS engine found")
                tts.start_tts_worker(engine, auto_skip_tts=False)
            else:
                tts.start_tts_worker(best_provider, auto_skip_tts=False)
        
        _tts_initialized = True
        log_message("INFO", f"TTS initialized with provider: {tts.tts_provider}")
        
        # Register cleanup on exit
        if not _shutdown_registered:
            atexit.register(_cleanup)
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            _shutdown_registered = True

def _cleanup():
    """Cleanup TTS and ASR resources"""
    global _tts_initialized, _asr_initialized
    
    if _asr_initialized and ASR_AVAILABLE:
        try:
            asr.stop_dictation()
            _asr_initialized = False
        except:
            pass
    
    if _tts_initialized:
        try:
            tts.shutdown_tts()
            _tts_initialized = False
        except:
            pass

def _signal_handler(signum, frame):
    """Handle shutdown signals"""
    _cleanup()
    sys.exit(0)

# MCP Tools for talkito control

@app.tool()
async def turn_on() -> str:
    """
    Enable talkito voice interaction mode - activates voice workflow patterns
    
    Returns:
        Instructions for voice mode activation
    """
    try:
        _ensure_initialization()
        log_message("INFO", "turn_on called")
        
        # Start with a voice announcement
        tts.queue_for_speech("Voice interaction mode activated. I will now speak my responses and listen for your voice input.", None)
        
        result = """✅ Voice interaction mode ACTIVATED!

I'm now following the voice interaction patterns from TALKITO.md:
- Speaking key conclusions after each response
- Automatically listening for your voice input
- Continuous hands-free conversation mode

Say 'stop voice mode' or call turn_off to deactivate."""
        log_message("INFO", f"turn_on returning: Voice mode activated")
        return result
        
    except Exception as e:
        error_msg = f"Error enabling voice mode: {str(e)}"
        log_message("ERROR", f"turn_on error: {error_msg}")
        return error_msg


@app.tool()  
async def turn_off() -> str:
    """
    Disable talkito voice interaction mode - deactivates voice workflow
    
    Returns:
        Confirmation of deactivation
    """
    try:
        _ensure_initialization()
        
        # Announce deactivation
        tts.queue_for_speech("Voice interaction mode deactivated. Returning to text-only interaction.", None)
        
        # Stop any active voice input
        if ASR_AVAILABLE and _asr_initialized:
            try:
                asr.stop_dictation()
            except:
                pass
        
        result = """Voice interaction mode is now DISABLED.

Returning to normal text-only interaction:
- No automatic speech output
- No automatic voice input
- Standard Claude interaction restored

You can re-enable voice mode at any time with the turn_on tool."""
        log_message("INFO", f"turn_off returning: Voice mode deactivated")
        return result
        
    except Exception as e:
        error_msg = f"Error disabling voice mode: {str(e)}"
        log_message("ERROR", f"turn_off error: {error_msg}")
        return error_msg


# MCP Tools for TTS functionality

@app.tool()
async def speak_text(text: str, clean_text_flag: bool = True) -> str:
    """
    Convert text to speech using the talkito TTS engine
    
    Args:
        text: Text to speak
        clean_text_flag: Whether to clean ANSI codes and symbols from text
    
    Returns:
        Status message about the speech request
    """
    try:
        _ensure_initialization()
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
            result = f"Skipped speaking: '{text[:50]}...' (filtered out)"
            log_message("INFO", f"speak_text returning: {result}")
            return result
        
        # Queue for speech
        tts.queue_for_speech(processed_text, None)
        
        result = f"Queued for speech: '{processed_text[:50]}...'"
        log_message("INFO", f"speak_text returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error speaking text: {str(e)}"
        log_message("ERROR", f"speak_text error: {error_msg}")
        return error_msg

@app.tool()
async def skip_current_speech() -> str:
    """
    Skip the currently playing speech item
    
    Returns:
        Status message about the skip action
    """
    try:
        _ensure_initialization()
        
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
async def get_speech_status() -> dict[str, Any]:
    """
    Get current status of the TTS system
    
    Returns:
        Dictionary with TTS status information
    """
    try:
        _ensure_initialization()
        
        is_speaking = tts.is_speaking()
        current_item = tts.get_current_speech_item()
        queue_size = tts.get_queue_size()
        
        status = {
            "is_speaking": is_speaking,
            "current_text": current_item.original_text[:100] if current_item else None,
            "current_line_number": current_item.line_number if current_item else None,
            "queue_size": queue_size,
            "tts_initialized": _tts_initialized
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
        _ensure_initialization()
        
        tts.wait_for_tts_to_finish(timeout=timeout)
        result = "All speech completed"
        log_message("INFO", f"wait_for_speech_completion returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error waiting for speech: {str(e)}"
        log_message("ERROR", f"wait_for_speech_completion error: {error_msg}")
        return error_msg

@app.tool()
async def configure_tts(provider: str = "system", voice: str = None, region: str = None, language: str = None, rate: float = None, pitch: float = None) -> str:
    """
    Configure TTS provider and voice settings
    
    Args:
        provider: TTS provider (system, openai, aws, polly, azure, gcloud, elevenlabs)
        voice: Voice name (provider-specific)
        region: Region for cloud providers
        language: Language code (e.g., en-US)
        rate: Speech rate (provider-specific, typically 0.5-2.0)
        pitch: Speech pitch (provider-specific)
        
    Returns:
        Status message about configuration
    """
    try:
        # Build config dict
        tts_config = {'provider': provider}
        if voice:
            tts_config['voice'] = voice
        if region:
            tts_config['region'] = region  
        if language:
            tts_config['language'] = language
        if rate is not None:
            tts_config['rate'] = rate
        if pitch is not None:
            tts_config['pitch'] = pitch
        
        # Configure TTS
        if provider != 'system':
            if not tts.configure_tts_from_dict(tts_config):
                error_msg = f"Failed to configure TTS provider: {provider}"
                log_message("ERROR", f"configure_tts error: {error_msg}")
                return error_msg
        
        # Restart TTS worker with new config
        if _tts_initialized:
            tts.shutdown_tts()
        
        engine = provider if provider != 'system' else tts.detect_tts_engine()
        tts.start_tts_worker(engine, auto_skip_tts=False)
        
        config_parts = [f"provider={provider}"]
        if voice: config_parts.append(f"voice={voice}")
        if region: config_parts.append(f"region={region}")
        if language: config_parts.append(f"language={language}")
        if rate is not None: config_parts.append(f"rate={rate}")
        if pitch is not None: config_parts.append(f"pitch={pitch}")
        
        result = f"Configured TTS: {', '.join(config_parts)}"
        log_message("INFO", f"configure_tts returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error configuring TTS: {str(e)}"
        log_message("ERROR", f"configure_tts error: {error_msg}")
        return error_msg

# MCP Tools for ASR functionality

def _dictation_callback(text: str):
    """Callback for ASR dictation - stores results for MCP retrieval"""
    global _last_dictated_text, _dictation_callback_results
    
    log_message("INFO", f"Dictation callback received: '{text}'")
    
    _last_dictated_text = text
    _dictation_callback_results.append({
        "text": text,
        "timestamp": time.time()
    })
    
    # Keep only last 10 results
    if len(_dictation_callback_results) > 10:
        _dictation_callback_results.pop(0)

@app.tool()
async def start_voice_input(language: str = "en-US", provider: str = None) -> str:
    """
    Start voice input/dictation using ASR
    
    Args:
        language: Language code for speech recognition (e.g., en-US, es-ES)
        provider: ASR provider (google, gcloud, assemblyai, deepgram, etc.)
        
    Returns:
        Status message about starting voice input
    """
    try:
        global _asr_initialized
        log_message("INFO", f"start_voice_input called with language: {language}, provider: {provider}")
        
        if not ASR_AVAILABLE:
            error_msg = "Error: ASR not available. Install with: pip install talkito[asr]"
            log_message("ERROR", f"start_voice_input error: {error_msg}")
            return error_msg
        
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
        
        # Start dictation with our callback
        # For providers that require streaming (like AssemblyAI), we need to provide a partial callback
        partial_callback = None
        if provider in ['assemblyai', 'deepgram', 'gcloud', 'azure', 'aws']:
            # These providers work better with streaming, so provide a dummy partial callback
            partial_callback = lambda text: log_message("DEBUG", f"Partial transcript: {text}")
        
        asr.start_dictation(_dictation_callback, partial_callback=partial_callback)
        _asr_initialized = True
        
        result = f"Started voice input (provider: {provider}, language: {language}). Speak now..."
        log_message("INFO", f"start_voice_input returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error starting voice input: {str(e)}"
        log_message("ERROR", f"start_voice_input error: {error_msg}")
        return error_msg

@app.tool()
async def stop_voice_input() -> str:
    """
    Stop voice input/dictation
    
    Returns:
        Status message about stopping voice input
    """
    try:
        global _asr_initialized
        
        if not ASR_AVAILABLE:
            result = "ASR not available"
            log_message("INFO", f"stop_voice_input returning: {result}")
            return result
        
        if _asr_initialized:
            asr.stop_dictation()
            _asr_initialized = False
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
async def get_voice_input_status() -> dict[str, Any]:
    """
    Get status of voice input system
    
    Returns:
        Dictionary with ASR status information
    """
    try:
        status = {
            "asr_available": ASR_AVAILABLE,
            "is_listening": asr.is_dictation_active() if ASR_AVAILABLE and _asr_initialized else False,
            "asr_initialized": _asr_initialized,
            "last_text": _last_dictated_text,
            "recent_results_count": len(_dictation_callback_results)
        }
        
        log_message("DEBUG", f"get_voice_input_status returning: {status}")
        return status
        
    except Exception as e:
        error_dict = {"error": f"Error getting voice input status: {str(e)}"}
        log_message("ERROR", f"get_voice_input_status error: {error_dict}")
        return error_dict

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
            log_message("INFO", f"get_dictated_text returning: No dictated text available")
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


# MCP Tools for Communication (WhatsApp/Slack)

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
            
        # Setup communication with specified providers
        _comms_manager = comms.setup_communication(providers=providers, config=config)
        
        if _comms_manager:
            active_providers = list(_comms_manager.providers.keys())
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
    try:
        global _comms_manager
        
        # Ensure we have a communication manager
        if not _comms_manager:
            # Try to auto-configure with WhatsApp
            _comms_manager = comms.setup_communication(providers=['whatsapp'])
            if not _comms_manager:
                error_msg = "WhatsApp not configured. Please call configure_communication first or set TWILIO credentials."
                log_message("ERROR", f"send_whatsapp error: {error_msg}")
                return error_msg
        
        # Check if WhatsApp provider exists
        if not _comms_manager or not _comms_manager.providers:
            error_msg = "No communication providers configured."
            log_message("ERROR", f"send_whatsapp error: {error_msg}")
            return error_msg
        
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
        result = whatsapp_provider.send_message(message, to_number)
        
        # Also speak it if requested
        if with_tts and _tts_initialized:
            tts.queue_for_speech(message, None)
        
        result_msg = f"WhatsApp message sent: '{message[:50]}...' to {result.get('to', to_number)}"
        log_message("INFO", f"send_whatsapp returning: {result_msg}")
        return result_msg
        
    except Exception as e:
        error_msg = f"Error sending WhatsApp message: {str(e)}"
        log_message("ERROR", f"send_whatsapp error: {error_msg}")
        return error_msg


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
    try:
        global _comms_manager
        
        # Ensure we have a communication manager
        if not _comms_manager:
            # Try to auto-configure with Slack
            _comms_manager = comms.setup_communication(providers=['slack'])
            if not _comms_manager:
                error_msg = "Slack not configured. Please call configure_communication first or set SLACK_BOT_TOKEN."
                log_message("ERROR", f"send_slack error: {error_msg}")
                return error_msg
        
        # Check if Slack provider exists
        if not _comms_manager or not _comms_manager.providers:
            error_msg = "No communication providers configured."
            log_message("ERROR", f"send_slack error: {error_msg}")
            return error_msg
        
        if not any(isinstance(p, SlackProvider) for p in _comms_manager.providers):
            error_msg = "Slack provider not available. Please configure with Slack bot token."
            log_message("ERROR", f"send_slack error: {error_msg}")
            return error_msg
        
        # Send the message
        slack_provider = next((p for p in _comms_manager.providers if isinstance(p, SlackProvider)), None)
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
        success = slack_provider.send_message(slack_message)
        
        # Also speak it if requested
        if with_tts and _tts_initialized:
            tts.queue_for_speech(message, None)
        
        if success:
            result_msg = f"Slack message sent: '{message[:50]}...' to {target_channel}"
        else:
            result_msg = f"Failed to send Slack message to {target_channel}"
        log_message("INFO", f"send_slack returning: {result_msg}")
        return result_msg
        
    except Exception as e:
        error_msg = f"Error sending Slack message: {str(e)}"
        log_message("ERROR", f"send_slack error: {error_msg}")
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


@app.tool()
async def start_whatsapp_mode(phone_number: str = None) -> str:
    """
    Start WhatsApp mode - all responses will also be sent to WhatsApp
    
    Args:
        phone_number: Phone number to send messages to (optional, uses TWILIO_WHATSAPP_NUMBER if not provided)
        
    Returns:
        Status message about WhatsApp mode activation
    """
    try:
        global _whatsapp_mode, _whatsapp_recipient, _comms_manager
        
        # Determine recipient
        if phone_number:
            _whatsapp_recipient = phone_number
        else:
            _whatsapp_recipient = os.environ.get('TWILIO_WHATSAPP_NUMBER')
            if not _whatsapp_recipient:
                error_msg = "No phone number provided and TWILIO_WHATSAPP_NUMBER not set. Please provide a number."
                log_message("ERROR", f"start_whatsapp_mode error: {error_msg}")
                return error_msg
        
        # Ensure WhatsApp is configured
        if not _comms_manager or not any(isinstance(p, TwilioWhatsAppProvider) for p in (_comms_manager.providers if _comms_manager else [])):
            _comms_manager = comms.setup_communication(providers=['whatsapp'])
            if not _comms_manager or not any(isinstance(p, TwilioWhatsAppProvider) for p in (_comms_manager.providers if _comms_manager else [])):
                error_msg = "Failed to configure WhatsApp. Check your Twilio credentials."
                log_message("ERROR", f"start_whatsapp_mode error: {error_msg}")
                return error_msg
        
        _whatsapp_mode = True
        
        # Send confirmation
        await send_whatsapp(f"WhatsApp mode activated! I'll send all my responses here.", _whatsapp_recipient)
        
        result = f"WhatsApp mode activated! Sending messages to {_whatsapp_recipient}"
        log_message("INFO", f"start_whatsapp_mode returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error starting WhatsApp mode: {str(e)}"
        log_message("ERROR", f"start_whatsapp_mode error: {error_msg}")
        return error_msg


@app.tool()
async def stop_whatsapp_mode() -> str:
    """
    Stop WhatsApp mode - responses will no longer be sent to WhatsApp
    
    Returns:
        Status message about WhatsApp mode deactivation
    """
    try:
        global _whatsapp_mode, _whatsapp_recipient
        
        if not _whatsapp_mode:
            result = "WhatsApp mode is not active"
            log_message("INFO", f"stop_whatsapp_mode returning: {result}")
            return result
        
        # Send farewell message
        if _whatsapp_recipient:
            await send_whatsapp("WhatsApp mode deactivated. Goodbye!", _whatsapp_recipient)
        
        _whatsapp_mode = False
        _whatsapp_recipient = None
        
        result = "WhatsApp mode deactivated"
        log_message("INFO", f"stop_whatsapp_mode returning: {result}")
        return result
        
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
    status = {
        "active": _whatsapp_mode,
        "recipient": _whatsapp_recipient,
        "configured": _comms_manager is not None and any(isinstance(p, TwilioWhatsAppProvider) for p in (_comms_manager.providers if _comms_manager else []))
    }
    log_message("DEBUG", f"get_whatsapp_mode_status returning: {status}")
    return status


@app.tool()
async def start_slack_mode(channel: str = None) -> str:
    """
    Start Slack mode - all responses will also be sent to Slack
    
    Args:
        channel: Slack channel to send messages to (optional, uses SLACK_CHANNEL if not provided)
        
    Returns:
        Status message about Slack mode activation
    """
    try:
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
        
        # Ensure Slack is configured
        if not _comms_manager or not any(isinstance(p, SlackProvider) for p in (_comms_manager.providers if _comms_manager else [])):
            _comms_manager = comms.setup_communication(providers=['slack'])
            if not _comms_manager:
                error_msg = "Failed to configure Slack. Check your Slack bot token."
                log_message("ERROR", f"start_slack_mode error: {error_msg}")
                return error_msg
            
            # The provider should be available immediately after setup_communication returns
            if not any(isinstance(p, SlackProvider) for p in _comms_manager.providers):
                error_msg = "Failed to configure Slack provider. Check your Slack bot token."
                log_message("ERROR", f"start_slack_mode error: {error_msg}")
                return error_msg
            
            # Now wait for the actual connection
            slack_provider = next((p for p in _comms_manager.providers if isinstance(p, SlackProvider)), None)
            if slack_provider and hasattr(slack_provider, 'wait_for_connection'):
                log_message("INFO", "Waiting for Slack connection...")
                if not slack_provider.wait_for_connection(timeout=3.0):
                    error_msg = "Slack connection timeout. Please check your credentials and network."
                    log_message("ERROR", f"start_slack_mode error: {error_msg}")
                    return error_msg
                log_message("INFO", "Slack connection established")
        
        _slack_mode = True
        
        # Send confirmation
        await send_slack(f"Slack mode activated! I'll send all my responses here.", _slack_channel)
        
        result = f"Slack mode activated! Sending messages to {_slack_channel}"
        log_message("INFO", f"start_slack_mode returning: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error starting Slack mode: {str(e)}"
        log_message("ERROR", f"start_slack_mode error: {error_msg}")
        return error_msg


@app.tool()
async def stop_slack_mode() -> str:
    """
    Stop Slack mode - responses will no longer be sent to Slack
    
    Returns:
        Status message about Slack mode deactivation
    """
    try:
        global _slack_mode, _slack_channel
        
        if not _slack_mode:
            result = "Slack mode is not active"
            log_message("INFO", f"stop_slack_mode returning: {result}")
            return result
        
        # Send farewell message
        if _slack_channel:
            await send_slack("Slack mode deactivated. Goodbye!", _slack_channel)
        
        _slack_mode = False
        _slack_channel = None
        
        result = "Slack mode deactivated"
        log_message("INFO", f"stop_slack_mode returning: {result}")
        return result
        
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
            messages = []
            # Collect all available messages
            while True:
                msg = _comms_manager.get_input_message(timeout=0)  # Non-blocking
                if msg:
                    messages.append(msg)
                else:
                    break
            
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
        status = await get_voice_input_status()
        lines = [
            "ASR Status:",
            f"  Available: {status.get('asr_available', False)}",
            f"  Initialized: {status.get('asr_initialized', False)}",
            f"  Is Listening: {status.get('is_listening', False)}",
            f"  Recent Results: {status.get('recent_results_count', 0)}",
        ]
        if status.get('last_text'):
            lines.append(f"  Last Text: {status['last_text'][:50]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting ASR status: {str(e)}"

@app.resource("talkito://voice/providers")
async def get_available_asr_providers() -> str:
    """Get list of available ASR providers with accessibility status"""
    try:
        if not ASR_AVAILABLE:
            return "ASR not available - install with: pip install talkito[asr]"
        
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
async def read_aloud() -> types.Prompt:
    """Template for reading text content aloud"""
    return types.Prompt(
        name="read_aloud",
        description="Read provided text content using text-to-speech",
        arguments=[
            types.PromptArgument(
                name="content",
                description="Text content to read aloud", 
                required=True
            )
        ],
        messages=[
            types.PromptMessage(
                role="user", 
                content=types.TextContent(
                    type="text",
                    text="Use the speak_text tool to read the provided content aloud, cleaning any formatting as needed."
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

def main():
    """Main entry point for the MCP server"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Talkito MCP Server - TTS/ASR via Model Context Protocol')
    parser.add_argument('--log-file', type=str, help='Path to log file for debugging')
    args = parser.parse_args()
    
    # Set up logging if log file specified
    if args.log_file:
        setup_logging(args.log_file)
        log_message("INFO", f"MCP server starting with log file: {args.log_file}")
    
    try:
        # Important: For stdio transport, we must use stderr for logging
        # stdout is reserved for the MCP protocol communication
        print("=" * 60, file=sys.stderr)
        print("Talkito MCP server is starting...", file=sys.stderr)
        print("Transport: stdio (communicating via stdin/stdout)", file=sys.stderr)
        if args.log_file:
            print(f"Logging to: {args.log_file}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # The server is ready immediately when run() executes
        # This is a blocking call that handles all MCP communication
        app.run(transport='stdio')
        
        # This line only executes after the server shuts down
        print("Talkito MCP server has stopped.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nTalkito MCP server interrupted by user.", file=sys.stderr)
    except Exception as e:
        print(f"Talkito MCP server error: {e}", file=sys.stderr)
        raise
    finally:
        _cleanup()

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher required", file=sys.stderr)
        sys.exit(1)
    
    main()