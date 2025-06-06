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
import atexit
import logging

# Import MCP SDK
try:
    from mcp import types
    from mcp.server import FastMCP
except ImportError:
    print("Error: MCP SDK not found. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import talkito functionality
from . import tts
from . import asr
from .core import TalkitoCore

# Check if ASR is available
try:
    from . import asr
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False

# Server configuration
app = FastMCP("talkito-tts-server")

# Global state
_tts_initialized = False
_asr_initialized = False
_shutdown_registered = False
_last_dictated_text = ""
_dictation_callback_results = []
_core_instance = None
_logger = None

def _ensure_initialization():
    """Ensure TTS system is initialized"""
    global _tts_initialized, _shutdown_registered, _core_instance, _logger
    
    if not _tts_initialized:
        # Create core instance
        _core_instance = TalkitoCore(verbosity_level=0)
        _logger = _core_instance.logger
        
        # Detect and start TTS engine
        engine = tts.detect_tts_engine()
        if engine == "none":
            raise RuntimeError("No TTS engine found. Please install espeak, festival, flite (Linux) or use macOS")
        
        # Start TTS worker with auto-skip disabled for MCP usage
        tts.start_tts_worker(engine, auto_skip_tts=False)
        _tts_initialized = True
        
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
        
        # Clean text if requested
        processed_text = text
        if clean_text_flag and _core_instance:
            processed_text = _core_instance.clean_text(text)
            processed_text = _core_instance.strip_profile_symbols(processed_text)
        
        # Skip empty or unwanted text
        if not processed_text.strip() or (_core_instance and _core_instance.should_skip_line(processed_text)):
            return f"Skipped speaking: '{text[:50]}...' (filtered out)"
        
        # Queue for speech
        tts.queue_for_speech(processed_text, None)
        
        return f"Queued for speech: '{processed_text[:50]}...'"
        
    except Exception as e:
        return f"Error speaking text: {str(e)}"

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
                return f"Skipped: '{current_item.original_text[:50]}...'"
            else:
                return "Skipped current speech"
        else:
            return "No speech currently playing to skip"
            
    except Exception as e:
        return f"Error skipping speech: {str(e)}"

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
        
        return status
        
    except Exception as e:
        return {"error": f"Error getting speech status: {str(e)}"}

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
        return "All speech completed"
        
    except Exception as e:
        return f"Error waiting for speech: {str(e)}"

@app.tool()
async def configure_tts(provider: str = "system", voice: str = None, region: str = None, language: str = None, rate: float = None, pitch: float = None) -> str:
    """
    Configure TTS provider and voice settings
    
    Args:
        provider: TTS provider (system, openai, polly, azure, gcloud, elevenlabs)
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
                return f"Failed to configure TTS provider: {provider}"
        
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
        
        return f"Configured TTS: {', '.join(config_parts)}"
        
    except Exception as e:
        return f"Error configuring TTS: {str(e)}"

# MCP Tools for ASR functionality

def _dictation_callback(text: str):
    """Callback for ASR dictation - stores results for MCP retrieval"""
    global _last_dictated_text, _dictation_callback_results
    
    _last_dictated_text = text
    _dictation_callback_results.append({
        "text": text,
        "timestamp": time.time()
    })
    
    # Keep only last 10 results
    if len(_dictation_callback_results) > 10:
        _dictation_callback_results.pop(0)

@app.tool()
async def start_voice_input(language: str = "en-US", provider: str = "google") -> str:
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
        
        if not ASR_AVAILABLE:
            return "Error: ASR not available. Install with: pip install talkito[asr]"
        
        # Configure ASR
        asr_config = {
            "provider": provider,
            "language": language
        }
        if not asr.configure_asr_from_dict(asr_config):
            return f"Warning: Failed to configure ASR (provider={provider}, language={language}), using defaults"
        
        # Start dictation with our callback
        asr.start_dictation(_dictation_callback)
        _asr_initialized = True
        
        return f"Started voice input (provider: {provider}, language: {language}). Speak now..."
        
    except Exception as e:
        return f"Error starting voice input: {str(e)}"

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
            return "ASR not available"
        
        if _asr_initialized:
            asr.stop_dictation()
            _asr_initialized = False
            return "Stopped voice input"
        else:
            return "Voice input was not active"
            
    except Exception as e:
        return f"Error stopping voice input: {str(e)}"

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
            "is_listening": asr.is_listening() if ASR_AVAILABLE and _asr_initialized else False,
            "asr_initialized": _asr_initialized,
            "last_text": _last_dictated_text,
            "recent_results_count": len(_dictation_callback_results)
        }
        
        return status
        
    except Exception as e:
        return {"error": f"Error getting voice input status: {str(e)}"}

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
        
        if not _dictation_callback_results:
            return {
                "text": "",
                "timestamp": None,
                "message": "No dictated text available"
            }
        
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
        
        return result
        
    except Exception as e:
        return {"error": f"Error getting dictated text: {str(e)}"}

@app.tool()
async def get_all_dictated_text(clear_after_read: bool = True) -> dict[str, Any]:
    """
    Get all recent dictated text from voice input
    
    Args:
        clear_after_read: Whether to clear the history after reading
        
    Returns:
        Dictionary with all dictated text entries
    """
    try:
        global _dictation_callback_results
        
        if not _dictation_callback_results:
            return {
                "entries": [],
                "count": 0,
                "message": "No dictated text available"
            }
        
        result = {
            "entries": _dictation_callback_results.copy(),
            "count": len(_dictation_callback_results),
            "message": f"Retrieved {len(_dictation_callback_results)} dictated text entries"
        }
        
        if clear_after_read:
            _dictation_callback_results.clear()
            result["message"] += " (cleared after read)"
        
        return result
        
    except Exception as e:
        return {"error": f"Error getting dictated text history: {str(e)}"}

# MCP Resources for TTS/ASR information

@app.resource("talk://speech/status")
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

@app.resource("talk://speech/engines")  
async def get_available_engines() -> str:
    """Get list of available TTS engines"""
    try:
        # Use talkito's detection logic
        detected_engine = tts.detect_tts_engine()
        
        engines = []
        if detected_engine != "none":
            engines.append(f"system ({detected_engine})")
        
        # Add cloud providers
        cloud_engines = ["openai", "polly", "azure", "gcloud", "elevenlabs"]
        engines.extend(cloud_engines)
        
        return "Available TTS engines:\n" + "\n".join(f"  - {engine}" for engine in engines)
        
    except Exception as e:
        return f"Error getting engines: {str(e)}"

@app.resource("talk://voice/status")
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

@app.resource("talk://voice/providers")
async def get_available_asr_providers() -> str:
    """Get list of available ASR providers"""
    try:
        if not ASR_AVAILABLE:
            return "ASR not available - install with: pip install talkito[asr]"
        
        providers = [
            "google (free, no API key required)",
            "gcloud (Google Cloud Speech-to-Text)",
            "assemblyai (AssemblyAI)",
            "deepgram (Deepgram)",
            "houndify (Houndify)",
            "aws (AWS Transcribe)",
            "bing (Microsoft Bing Speech)"
        ]
        
        return "Available ASR providers:\n" + "\n".join(f"  - {provider}" for provider in providers)
        
    except Exception as e:
        return f"Error getting ASR providers: {str(e)}"

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
    try:
        # Run the MCP server using stdio transport
        app.run(transport='stdio')
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher required", file=sys.stderr)
        sys.exit(1)
    
    main()