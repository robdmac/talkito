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
tts.py - Text-to-Speech engine and text processing utilities
Handles TTS engine detection, mathematical symbol conversion, and speech synthesis
Supports both system TTS engines and external API providers (OpenAI, etc.)
"""

import subprocess
import shutil
import re
import time
import threading
import queue
import logging
import os
import argparse
import tempfile
from collections import deque
from typing import Optional, List, Tuple, Deque, Dict, Any
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime

# Import centralized logging utilities
try:
    from .logs import log_message as _base_log_message
except ImportError:
    # Fallback for standalone execution
    def _base_log_message(level: str, message: str, logger_name: str = None):
        print(f"[{level}] {message}")

# Try to load .env files if available
try:
    from dotenv import load_dotenv
    # Load .env first (takes precedence)
    load_dotenv()
    # Also load .talkito.env (won't override existing vars from .env)
    load_dotenv('.talkito.env')
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Configuration constants
MIN_SPEAK_LENGTH = 1  # Minimum characters before speaking
CACHE_SIZE = 1000  # Cache size for similarity checking
CACHE_TIMEOUT = 1800  # Seconds before a cached item can be spoken again
SIMILARITY_THRESHOLD = 0.85  # How similar text must be to be considered a repeat
DEBOUNCE_TIME = 0.5  # Seconds to wait before speaking rapidly changing text

# Global configuration
auto_skip_tts_enabled = False  # Whether to auto-skip long text
disable_tts = False  # Whether to disable TTS completely (for testing)
tts_provider = "system"  # Current TTS provider (system, openai, polly, azure, gcloud, elevenlabs, deepgram, etc.)
openai_voice = os.environ.get('OPENAI_VOICE', 'alloy')  # Default OpenAI voice
polly_voice = os.environ.get('AWS_POLLY_VOICE', 'Joanna')  # Default AWS Polly voice
polly_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')  # Default AWS region for Polly
azure_voice = os.environ.get('AZURE_VOICE', 'en-US-AriaNeural')  # Default Azure voice
azure_region = os.environ.get('AZURE_REGION', 'eastus')  # Default Azure region
gcloud_voice = os.environ.get('GCLOUD_VOICE', 'en-US-Journey-F')  # Default Google Cloud voice
gcloud_language_code = os.environ.get('GCLOUD_LANGUAGE_CODE', 'en-US')  # Default Google Cloud language code
elevenlabs_voice_id = os.environ.get('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')  # Default ElevenLabs voice (Rachel)
elevenlabs_model_id = os.environ.get('ELEVENLABS_MODEL_ID', 'eleven_monolingual_v1')  # Default ElevenLabs model
deepgram_voice_model = os.environ.get('DEEPGRAM_VOICE_MODEL', 'aura-asteria-en')  # Default Deepgram model


# Provider registry for easier management
TTS_PROVIDERS = {
    'openai': {
        'func': None,  # Will be set to speak_with_openai after function definition
        'env_var': 'OPENAI_API_KEY',
        'voice_var': 'openai_voice',
        'display_name': 'OpenAI',
        'install': 'pip install openai',
        'config_keys': ['voice']
    },
    'aws': {
        'func': None,  # Will be set to speak_with_polly after function definition
        'env_var': None,  # AWS uses multiple env vars or config files
        'env_name': 'AWS credentials',
        'voice_var': 'polly_voice',
        'region_var': 'polly_region',
        'display_name': 'AWS Polly',
        'install': 'pip install boto3',
        'config_keys': ['voice', 'region']
    },
    'azure': {
        'func': None,  # Will be set to speak_with_azure after function definition
        'env_var': 'AZURE_SPEECH_KEY',
        'voice_var': 'azure_voice',
        'region_var': 'azure_region',
        'display_name': 'Microsoft Azure',
        'install': 'pip install azure-cognitiveservices-speech',
        'config_keys': ['voice', 'region']
    },
    'gcloud': {
        'func': None,  # Will be set to speak_with_gcloud after function definition
        'env_var': 'GOOGLE_APPLICATION_CREDENTIALS',
        'voice_var': 'gcloud_voice',
        'language_var': 'gcloud_language_code',
        'display_name': 'Google Cloud',
        'install': 'pip install google-cloud-texttospeech',
        'config_keys': ['voice', 'language']
    },
    'elevenlabs': {
        'func': None,  # Will be set to speak_with_elevenlabs after function definition
        'env_var': 'ELEVENLABS_API_KEY',
        'voice_var': 'elevenlabs_voice_id',
        'model_var': 'elevenlabs_model_id',
        'display_name': 'ElevenLabs',
        'install': 'pip install requests',
        'config_keys': ['voice', 'model']
    },
    'deepgram': {
        'func': None,  # Will be set to speak_with_deepgram after function definition
        'env_var': 'DEEPGRAM_API_KEY',
        'model_var': 'deepgram_voice_model',
        'display_name': 'Deepgram',
        'install': 'pip install deepgram-sdk',
        'config_keys': ['model']
    }
}


@dataclass
class SpeechItem:
    """Represents an item in the speech queue with metadata"""
    text: str
    original_text: str
    line_number: Optional[int] = None
    timestamp: Optional[datetime] = None
    source: str = "output"  # "output", "error", etc.
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PlaybackControl:
    """Controls for TTS playback"""
    def __init__(self):
        self.current_index = 0
        self.is_paused = False
        self.skip_current = False
        self.skip_all = False
        self.current_process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        
    def pause(self):
        """Pause current playback"""
        with self.lock:
            self.is_paused = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
                    
    def resume(self):
        """Resume playback"""
        with self.lock:
            self.is_paused = False
            
    def skip_current_item(self):
        """Skip the currently playing item"""
        with self.lock:
            self.skip_current = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
                    
    def skip_all_items(self):
        """Skip all remaining items in queue"""
        with self.lock:
            self.skip_all = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
                    
    def reset_skip_flags(self):
        """Reset skip flags"""
        with self.lock:
            self.skip_current = False
            self.skip_all = False


# Mathematical symbol replacements mapping
SYMBOL_REPLACEMENTS = {
    '+': ' plus ',
    '-': ' minus ',
    '×': ' times ',
    '*': ' times ',
    '÷': ' divided by ',
    '/': ' divided by ',
    '=': ' equals ',
    '≠': ' not equals ',
    '<': ' less than ',
    '>': ' greater than ',
    '≤': ' less than or equal to ',
    '≥': ' greater than or equal to ',
    '±': ' plus or minus ',
    '%': ' percent ',
    '^': ' to the power of ',
    '√': ' square root of ',
    '∞': ' infinity ',
    'π': ' pi ',
    '∑': ' sum ',
    '∏': ' product ',
    '∫': ' integral ',
    '∂': ' partial derivative ',
    '∇': ' gradient ',
    '∈': ' is in ',
    '∉': ' is not in ',
    '⊂': ' is a subset of ',
    '⊃': ' is a superset of ',
    '∪': ' union ',
    '∩': ' intersection ',
    '∅': ' empty set ',
    '≈': ' approximately equals ',
    '≡': ' is identical to ',
    '∝': ' is proportional to ',
    '∠': ' angle ',
    '⊥': ' is perpendicular to ',
    '∥': ' is parallel to ',
    '°': ' degrees ',
    '✻': '',  # Remove decorative bullet
}


# Global state for TTS functionality
tts_queue: queue.Queue = queue.Queue()
spoken_cache: Deque[Tuple[str, float]] = deque(maxlen=CACHE_SIZE)
tts_worker_thread: Optional[threading.Thread] = None
shutdown_event = threading.Event()
last_queued_text = ""
last_queue_time = 0
# Playback control
playback_control = PlaybackControl()
speech_history: List[SpeechItem] = []
current_speech_item: Optional[SpeechItem] = None

# Track when speech actually finishes (including audio playback)
last_speech_end_time = 0.0
SPEECH_BUFFER_TIME = 0.1  # Seconds to wait after TTS process ends for audio to finish

# Track the highest line number that has been spoken
highest_spoken_line_number = -1


# Wrapper for module-specific logging
def log_message(level: str, message: str):
    """Log a message using centralized logging"""
    _base_log_message(level, message, __name__)


def check_tts_provider_accessibility() -> Dict[str, Dict[str, Any]]:
    """Check which TTS providers are accessible based on API keys and environment"""
    accessible = {}
    
    # System TTS
    detected_engine = detect_tts_engine()
    if detected_engine != "none":
        accessible["system"] = {
            "available": True,
            "engine": detected_engine,
            "note": "System TTS engine detected"
        }
    else:
        accessible["system"] = {
            "available": False,
            "note": "No system TTS engine found"
        }
    
    # OpenAI
    accessible["openai"] = {
        "available": bool(os.environ.get("OPENAI_API_KEY")),
        "note": "Requires OPENAI_API_KEY environment variable"
    }
    
    # Amazon Polly - Check using boto3's credential chain
    polly_available = False
    polly_note = "Requires AWS credentials (env vars, ~/.aws/credentials, or IAM role)"
    try:
        import boto3
        # Try to create a session to check if credentials are available
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is not None:
            polly_available = True
            polly_note = "AWS credentials detected"
    except ImportError:
        polly_note = "Requires boto3 package (pip install boto3)"
    except Exception:
        pass
    
    accessible["aws"] = {
        "available": polly_available,
        "note": polly_note
    }
    # Keep 'polly' for backward compatibility
    accessible["polly"] = accessible["aws"]
    
    # Azure
    accessible["azure"] = {
        "available": bool(os.environ.get("AZURE_SPEECH_KEY") and os.environ.get("AZURE_SPEECH_REGION")),
        "note": "Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION"
    }
    
    # Google Cloud
    accessible["gcloud"] = {
        "available": bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCLOUD_SERVICE_ACCOUNT_JSON")),
        "note": "Requires GOOGLE_APPLICATION_CREDENTIALS or GCLOUD_SERVICE_ACCOUNT_JSON"
    }
    
    # ElevenLabs
    accessible["elevenlabs"] = {
        "available": bool(os.environ.get("ELEVENLABS_API_KEY")),
        "note": "Requires ELEVENLABS_API_KEY environment variable"
    }
    
    return accessible


def detect_tts_engine() -> str:
    """Detect available TTS engine on the system"""
    engines = [
        ("say", "say"),        # macOS
        ("espeak", "espeak"),  # Linux
        ("festival", "festival"),  # Linux
        ("flite", "flite")     # Linux
    ]

    for cmd, name in engines:
        if shutil.which(cmd):
            return name
    return "none"


def _create_temp_audio_file(suffix: str = ".mp3") -> str:
    """Create a temporary audio file and return its path"""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        return tmp_file.name


def _play_audio_file(audio_path: str, use_process_control: bool = True) -> bool:
    """Play an audio file using available system players"""
    process = None
    
    # Detect and create player process
    if shutil.which("afplay"):  # macOS
        process = subprocess.Popen(["afplay", audio_path])
    elif shutil.which("mpg123"):
        process = subprocess.Popen(["mpg123", "-q", audio_path])
    elif shutil.which("play"):  # SoX
        process = subprocess.Popen(["play", "-q", audio_path])
    else:
        log_message("ERROR", "No audio player found (afplay, mpg123, or play)")
        return False
    
    if not process:
        return False
    
    # Store process for interruption support if requested
    if use_process_control:
        with playback_control.lock:
            playback_control.current_process = process
    
    # Wait for completion or interruption
    try:
        process.wait()
        return process.returncode == 0
    except:
        return False
    finally:
        if use_process_control:
            with playback_control.lock:
                playback_control.current_process = None


def _cleanup_temp_file(file_path: str):
    """Safely remove a temporary file"""
    if os.path.exists(file_path):
        os.unlink(file_path)


def _handle_import_error(library_name: str, install_command: str) -> bool:
    """Handle import errors for TTS provider libraries"""
    log_message("ERROR", f"{library_name} library not installed. Run: {install_command}")
    return False


def _handle_provider_error(provider_name: str, error: Exception) -> bool:
    """Handle general provider errors"""
    log_message("ERROR", f"{provider_name} TTS failed: {error}")
    return False


def synthesize_and_play(synthesize_func, text: str, use_process_control: bool = True) -> bool:
    """Common pattern for synthesizing and playing audio from any provider"""
    try:
        # Get audio data from the provider
        audio_data = synthesize_func(text)
        if not audio_data:
            return False
            
        # Create temporary file for audio
        tmp_path = _create_temp_audio_file(".mp3")
        
        try:
            # Write audio content to file
            with open(tmp_path, 'wb') as tmp_file:
                tmp_file.write(audio_data)
            
            # Play the audio file
            return _play_audio_file(tmp_path, use_process_control=use_process_control)
        finally:
            _cleanup_temp_file(tmp_path)
    except Exception as e:
        log_message("ERROR", f"Audio synthesis/playback failed: {e}")
        return False


def validate_provider_config(provider: str) -> bool:
    """Unified configuration validation for any provider"""
    provider_info = TTS_PROVIDERS.get(provider)
    if not provider_info:
        return True  # System provider, no validation needed
    
    # Check environment variable if required
    env_var = provider_info.get('env_var')
    if env_var and not os.environ.get(env_var):
        print(f"Error: {env_var} environment variable not set")
        print(f"Please set it with: export {env_var}='your-api-key'")
        return False
    
    # Special case for AWS Polly - check AWS credentials
    if provider in ['aws', 'polly']:
        try:
            import boto3
            # Try to create a client to verify credentials
            test_client = boto3.client('polly', region_name=polly_region)
            test_client.describe_voices(LanguageCode='en-US')
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            return False
        except Exception as e:
            print(f"Error: AWS credentials not configured or invalid: {e}")
            print("Please configure AWS credentials (e.g., AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)")
            return False
    
    return True


def _synthesize_openai(text: str) -> Optional[bytes]:
    """Synthesize speech using OpenAI TTS API"""
    import openai
    import io
    
    # Generate speech using OpenAI
    response = openai.audio.speech.create(
        model="tts-1",
        voice=openai_voice,
        input=text
    )
    
    # Convert streaming response to bytes
    audio_data = io.BytesIO()
    for chunk in response.iter_bytes():
        audio_data.write(chunk)
    
    return audio_data.getvalue()


def speak_with_openai(text: str) -> bool:
    """Speak text using OpenAI TTS API"""
    try:
        import openai
        # Check for API key is now done by validate_provider_config
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            log_message("ERROR", "OPENAI_API_KEY environment variable not set")
            return False
        
        return synthesize_and_play(_synthesize_openai, text, use_process_control=False)
    except ImportError:
        return _handle_import_error("OpenAI", "pip install openai")
    except Exception as e:
        return _handle_provider_error("OpenAI", e)


def _synthesize_polly(text: str) -> Optional[bytes]:
    """Synthesize speech using AWS Polly TTS API"""
    import boto3
    
    # Create Polly client
    polly_client = boto3.client('polly', region_name=polly_region)
    
    # Generate speech using AWS Polly
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId=polly_voice,
        Engine='neural'  # Use neural voice for better quality
    )
    
    # Return audio stream as bytes
    return response['AudioStream'].read()


def speak_with_polly(text: str) -> bool:
    """Speak text using AWS Polly TTS API"""
    try:
        import boto3
        return synthesize_and_play(_synthesize_polly, text, use_process_control=False)
    except ImportError:
        return _handle_import_error("boto3", "pip install boto3")
    except Exception as e:
        return _handle_provider_error("AWS Polly", e)


def _synthesize_azure(text: str) -> Optional[bytes]:
    """Synthesize speech using Microsoft Azure TTS API"""
    import azure.cognitiveservices.speech as speechsdk
    
    # Check for API key and region
    speech_key = os.environ.get('AZURE_SPEECH_KEY')
    if not speech_key:
        raise ValueError("AZURE_SPEECH_KEY environment variable not set")
    
    # Create speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = azure_voice
    
    # Set output format to mp3
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )
    
    # Create synthesizer without audio output (we'll get the audio data)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    
    # Synthesize the text
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Return the audio data
        return bytes(result.audio_data)
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        error_msg = f"Azure TTS synthesis canceled: {cancellation_details.reason}"
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_msg += f" - {cancellation_details.error_details}"
        raise Exception(error_msg)
    else:
        raise Exception(f"Azure TTS synthesis failed with reason: {result.reason}")


def speak_with_azure(text: str) -> bool:
    """Speak text using Microsoft Azure TTS API"""
    try:
        import azure.cognitiveservices.speech as speechsdk
        return synthesize_and_play(_synthesize_azure, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("Azure Speech SDK", "pip install azure-cognitiveservices-speech")
    except Exception as e:
        return _handle_provider_error("Microsoft Azure", e)


def _synthesize_gcloud(text: str) -> Optional[bytes]:
    """Synthesize speech using Google Cloud Text-to-Speech API"""
    from google.cloud import texttospeech
    
    # Create a client
    client = texttospeech.TextToSpeechClient()
    
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request - handle both simple voice names and full voice specs
    if gcloud_voice and '-' in gcloud_voice:
        # Full voice name like "en-US-Journey-F"
        voice = texttospeech.VoiceSelectionParams(
            language_code=gcloud_language_code,
            name=gcloud_voice
        )
    else:
        # Simple voice name - let Google pick the best match
        voice = texttospeech.VoiceSelectionParams(
            language_code=gcloud_language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
    
    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,  # Normal speed
        pitch=0.0,  # Normal pitch
        volume_gain_db=0.0  # Normal volume
    )
    
    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    return response.audio_content


def speak_with_gcloud(text: str) -> bool:
    """Speak text using Google Cloud Text-to-Speech API"""
    try:
        from google.cloud import texttospeech
        return synthesize_and_play(_synthesize_gcloud, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("Google Cloud Text-to-Speech", "pip install google-cloud-texttospeech")
    except Exception as e:
        return _handle_provider_error("Google Cloud", e)


def _synthesize_elevenlabs(text: str) -> Optional[bytes]:
    """Synthesize speech using ElevenLabs TTS API"""
    import requests
    
    # Get API key
    api_key = os.environ.get('ELEVENLABS_API_KEY')
    
    # ElevenLabs API endpoint
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": elevenlabs_model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    # Make the request
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code != 200:
        log_message("ERROR", f"ElevenLabs API returned status {response.status_code}: {response.text}")
        print(f"ElevenLabs API Error: {response.text}")
        return None
    
    return response.content


def speak_with_elevenlabs(text: str) -> bool:
    """Speak text using ElevenLabs TTS API"""
    try:
        import requests
        
        # Check for API key
        api_key = os.environ.get('ELEVENLABS_API_KEY')
        if not api_key:
            log_message("ERROR", "ELEVENLABS_API_KEY environment variable not set")
            return False
        
        return synthesize_and_play(_synthesize_elevenlabs, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("requests", "pip install requests")
    except Exception as e:
        return _handle_provider_error("ElevenLabs", e)


def _synthesize_deepgram(text: str) -> Optional[bytes]:
    """Synthesize speech using Deepgram TTS API"""
    from deepgram import DeepgramClient, SpeakOptions
    import tempfile
    import os
    
    # Get API key
    api_key = os.environ.get('DEEPGRAM_API_KEY')
    
    # Create Deepgram client
    deepgram = DeepgramClient(api_key)
    
    # Configure options
    options = SpeakOptions(
        model=deepgram_voice_model,
        encoding="mp3"
    )
    
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    
    try:
        # Use the save method as shown in the reference
        response = deepgram.speak.rest.v("1").save(tmp_filename, {"text": text}, options)
        
        # Read the audio file
        with open(tmp_filename, 'rb') as f:
            audio_data = f.read()
        
        return audio_data
    except Exception as e:
        log_message("ERROR", f"Deepgram TTS error: {e}")
        return None
    finally:
        # Clean up temp file
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def speak_with_deepgram(text: str) -> bool:
    """Speak text using Deepgram TTS API"""
    try:
        from deepgram import DeepgramClient
        
        # Check for API key
        api_key = os.environ.get('DEEPGRAM_API_KEY')
        if not api_key:
            log_message("ERROR", "DEEPGRAM_API_KEY environment variable not set")
            return False
        
        return synthesize_and_play(_synthesize_deepgram, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("deepgram-sdk", "pip install deepgram-sdk")
    except Exception as e:
        return _handle_provider_error("Deepgram", e)


# Initialize provider functions in the registry
TTS_PROVIDERS['openai']['func'] = speak_with_openai
TTS_PROVIDERS['aws']['func'] = speak_with_polly
TTS_PROVIDERS['azure']['func'] = speak_with_azure
TTS_PROVIDERS['gcloud']['func'] = speak_with_gcloud
TTS_PROVIDERS['elevenlabs']['func'] = speak_with_elevenlabs
TTS_PROVIDERS['deepgram']['func'] = speak_with_deepgram

# Add 'polly' as an alias for 'aws' for backward compatibility
TTS_PROVIDERS['polly'] = TTS_PROVIDERS['aws']


def context_aware_symbol_replacement(text: str) -> str:
    """Replace symbols based on their context"""
    # Extract just the filename from file paths before processing
    # Match common file path patterns and replace with just the filename
    def extract_filename(match):
        full_path = match.group(0)
        # Get just the filename (last component after /)
        filename = full_path.split('/')[-1]
        return filename
    
    # Replace absolute and relative file paths with just the filename
    # This comprehensive regex handles various path formats:
    # - /absolute/path/to/file.ext -> file.ext
    # - ./relative/path/file.ext -> file.ext
    # - ~/home/path/file.ext -> file.ext
    # - relative/path/file.ext -> file.ext
    text = re.sub(r'(?:/?(?:[\w.-]+/)+)([\w.-]+)', extract_filename, text)
    
    # This regex matches filenames with extensions (e.g., talk.py -> talk dot py)
    text = re.sub(r'(\w+)\.(\w+)', r'\1 dot \2', text)
    
    # Remove bullet points (- or + at start of line)
    text = re.sub(r'^[-+]\s+', '', text, flags=re.MULTILINE)
    
    # Handle hashtag symbol
    text = re.sub(r'#', ' hashtag ', text)
    
    # Handle double dash (--) when it precedes a word (e.g., --option)
    text = re.sub(r'--(\w)', r' dash dash \1', text)
    
    # Handle dates (preserve the pattern for later processing)
    text = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1 to \2 to \3', text)
    
    # Replace negative numbers (-5 -> minus 5)
    text = re.sub(r'(?<![a-zA-Z0-9])-(\d+)', r' minus \1', text)
    
    # Replace subtraction with spaces around it (5 - 3 -> 5 minus 3)
    text = re.sub(r'(\d+)\s+-\s+(\d+)', r'\1 minus \2', text)
    
    # Replace dashes surrounded by spaces with dash (word - word -> word dash word)
    text = re.sub(r'(\w+)\s+-\s+', r'\1 dash ', text)
    
    # Replace ranges (10-20 -> 10 to 20) - must come after spaced subtraction
    text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
    
    # Replace plus in math contexts
    text = re.sub(r'\+', ' plus ', text)
    
    # Division: number / number with spaces (e.g., "10 / 2")
    text = re.sub(r'(\d+)\s+/\s+(\d+)', r'\1 divided by \2', text)
    
    # URLs: http://example.com or https://example.com - MUST come before other slash handling
    text = re.sub(r'(https?:)//([^\s]+)', r'\1 slash slash \2', text)
    
    # Commands that start with slash (e.g., /install-github-app, /help)
    text = re.sub(r'(?<![.\w])/([a-zA-Z][\w-]+)', r'slash \1', text)
    
    # Remaining "Either/or" pattern - only for simple word/word without dots or extensions
    text = re.sub(r'(?<![.\w])(\w+)/(\w+)(?![.\w])', r'\1 or \2', text)

    # Word and acronym replacements
    text = re.sub(r'cwd', ' current working directory ', text)
    text = re.sub(r'Usage:', 'Use as follows:', text)
    text = re.sub(r'Todos', 'To do\'s', text)
    
    # Replace other mathematical symbols unconditionally
    skip_symbols = {'+', '-', '/'}  # Already handled above
    for symbol, replacement in SYMBOL_REPLACEMENTS.items():
        if symbol not in skip_symbols:
            text = text.replace(symbol, replacement)
    
    return text


def add_sentence_ending(text: str) -> str:
    """Add a period to text if it doesn't already end with punctuation"""
    if not text:
        return text
    
    # Check if text already ends with sentence-ending punctuation
    text = text.rstrip()  # Remove trailing whitespace
    if text and text[-1] not in '.!?:;':
        return text + '.'
    return text


def clean_punctuation_sequences(text: str) -> str:
    """Clean up awkward punctuation sequences like '?.' or '!.'"""
    # Replace punctuation followed by period+space with just the punctuation+space
    text = re.sub(r'([!?:;])\.\s', r'\1 ', text)
    # Replace punctuation followed by period at end with just the punctuation
    text = re.sub(r'([!?:;])\.$', r'\1', text)
    # Only clean up multiple periods with spaces between (like ". ." or ". . .")
    # This preserves intentional ellipsis like "..." or ".."
    text = re.sub(r'\.(\s+\.)+', '.', text)
    return text


def extract_speakable_text(text: str) -> (str, str):
    """Extract only speakable text and convert mathematical symbols"""
    # Convert function names: underscores to spaces, drop ()
    text = re.sub(r'(\w+)_(\w+)(?:_(\w+))*\(\)', lambda m: ' '.join(m.group(0).replace('_', ' ').replace('()', '').split()), text)
    text = re.sub(r'(\w+)_(\w+)(?:_(\w+))*', lambda m: ' '.join(m.group(0).split('_')), text)
    
    # Remove bracketed comments at the end
    text = re.sub(r' *\([^)]*\)$', '', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Code

    # Handle bullet point lists by converting to numbered items
    # Check if the text contains bullet points (☐, •, -, *, etc.)
    bullet_patterns = [r'☐', r'•', r'▪', r'▫', r'■', r'□', r'◦', r'‣', r'⁃']
    
    # Split text into lines and process each
    lines = text.split('. ')
    processed_spoken_lines = []
    processed_written_lines = []
    bullet_counter = 0
    spoken_text = ""
    written_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        written_line = line
            
        # Check if this line starts with a bullet
        has_bullet = False
        for pattern in bullet_patterns:
            if line.startswith(pattern):
                has_bullet = True
                # Remove the bullet and clean up
                line = line.replace(pattern, '', 1).strip()
                break

        line = context_aware_symbol_replacement(line)

        line = re.sub(r'[^a-zA-Z0-9 .,!?:\'-]', '', line)  # Added colon to allowed chars
        line = re.sub(r' +', ' ', line)
        line = line.strip()
        if line:
            if has_bullet:
                bullet_counter += 1
                processed_spoken_lines.append(f"{bullet_counter}: {line}")
                processed_written_lines.append(f"{bullet_counter}: {written_line}")
            else:
                # Keep non-bullet lines as-is
                processed_spoken_lines.append(line)
                processed_written_lines.append(written_line)
    
    if processed_spoken_lines:
        spoken_text = '. '.join(processed_spoken_lines) + '.'

    if processed_written_lines:
        written_text = '. '.join(processed_written_lines) + '.'
    
    # Skip text that contains no letters (just punctuation, numbers, spaces)
    if not re.search(r'[a-zA-Z]', text):
        return "", ""
    
    return spoken_text, written_text


def is_similar_to_recent(text: str) -> bool:
    """Check if text is similar to recently spoken text using fuzzy matching"""
    current_time = time.time()

    # Clean up expired cache entries
    while spoken_cache and (current_time - spoken_cache[0][1]) >= CACHE_TIMEOUT:
        spoken_cache.popleft()

    # Check for exact matches first
    if any(cached_text == text for cached_text, _ in spoken_cache):
        return True

    # Check for similarity
    for cached_text, _ in spoken_cache:
        similarity = SequenceMatcher(None, text.lower(), cached_text.lower()).ratio()
        if similarity >= SIMILARITY_THRESHOLD:
            log_message("INFO", f"Text '{text}' is {similarity:.2%} similar to '{cached_text}'")
            return True

    return False


def speak_text(text: str, engine: str) -> bool:
    """Speak text using appropriate engine. Returns True if completed, False if interrupted."""
    if disable_tts:
        return True

    # Use provider registry for cleaner routing
    provider_info = TTS_PROVIDERS.get(tts_provider)
    if provider_info and provider_info.get('func'):
        return provider_info['func'](text)

    return speak_with_default(text, engine)


def speak_with_default(text, engine):
    # Handle system TTS engines
    try:
        commands = {
            "say": ["say", text],
            "espeak": ["espeak", text],
            "festival": ["festival", "--tts"],
            "flite": ["flite", "-voice", "slt"]
        }

        cmd = commands.get(engine, [])
        if not cmd:
            return True

        kwargs = {"stderr": subprocess.DEVNULL}
        if engine in ["festival", "flite"]:
            kwargs["input"] = text.encode()
            kwargs["stdin"] = subprocess.PIPE
        
        # Store process in playback control
        process = subprocess.Popen(cmd, **kwargs)
        with playback_control.lock:
            playback_control.current_process = process
        
        # Wait for completion or interruption
        try:
            if engine in ["festival", "flite"] and "input" in kwargs:
                process.communicate(input=kwargs["input"])
            else:
                process.wait()
            return process.returncode == 0
        except:
            return False
        finally:
            with playback_control.lock:
                playback_control.current_process = None
                
    except Exception as e:
        log_message("ERROR", f"TTS failed: {e}")
        return False


def tts_worker(engine: str):
    """TTS worker thread - reads from queue and speaks"""
    global current_speech_item, highest_spoken_line_number
    log_message("INFO", f"Started TTS worker with engine: {engine}")

    while not shutdown_event.is_set():
        try:
            # Check if we should skip all
            if playback_control.skip_all:
                # Clear the queue
                while not tts_queue.empty():
                    try:
                        tts_queue.get_nowait()
                    except:
                        break
                playback_control.reset_skip_flags()
                continue
                
            # Wait while paused
            while playback_control.is_paused and not shutdown_event.is_set():
                time.sleep(0.1)
                
            # Get next item from queue
            item = tts_queue.get(timeout=0.1)

            if item == "__SHUTDOWN__":
                break
                
            # Handle both old string format and new SpeechItem format
            if isinstance(item, str):
                speech_item = SpeechItem(text=item, original_text=item)
            else:
                speech_item = item
                
            # Check if this line number is older than what we've already spoken
            if speech_item.line_number is not None and speech_item.line_number <= highest_spoken_line_number:
                log_message("INFO", f"Skipping line {speech_item.line_number} (already spoken up to line {highest_spoken_line_number}): '{speech_item.text[:50]}...'")
                continue
                
            current_speech_item = speech_item

            # Tell ASR to ignore input while we're speaking to prevent feedback
            try:
                from . import asr
                if hasattr(asr, 'set_ignore_input'):
                    asr.set_ignore_input(True)
                    log_message("DEBUG", "Set ASR to ignore input during TTS")
            except Exception as e:
                log_message("DEBUG", f"Could not set ASR ignore flag: {e}")

            # Include line number in log if available
            if speech_item.line_number is not None:
                log_message("INFO", f"Speaking via {engine} (line {speech_item.line_number}): '{speech_item.text}'")
            else:
                log_message("INFO", f"Speaking via {engine}: '{speech_item.text}'")

            speak_text(speech_item.text, engine)
            
            # Update highest spoken line number if applicable
            if speech_item.line_number is not None and speech_item.line_number > highest_spoken_line_number:
                highest_spoken_line_number = speech_item.line_number
                log_message("DEBUG", f"Updated highest_spoken_line_number to {highest_spoken_line_number}")
            
            # Check if we should skip current
            if playback_control.skip_current:
                playback_control.reset_skip_flags()
                
            # Add to history
            speech_history.append(speech_item)
            
            # Clear current item and track end time
            current_speech_item = None
            global last_speech_end_time
            last_speech_end_time = time.time()
            
            # Resume ASR input after speaking
            try:
                from . import asr
                if hasattr(asr, 'set_ignore_input'):
                    # Small delay to ensure TTS audio has fully finished
                    time.sleep(0.5)
                    asr.set_ignore_input(False)
                    log_message("DEBUG", "Resumed ASR after speaking")
            except Exception as e:
                log_message("DEBUG", f"Could not resume ASR: {e}")

        except queue.Empty:
            continue
        except Exception as e:
            log_message("ERROR", f"TTS worker error: {e}")


def queue_for_speech(text: str, line_number: Optional[int] = None, source: str = "output") -> str:
    """Queue text for speaking with debouncing"""
    global last_queued_text, last_queue_time

    # Log the original text before any filtering
    log_message("INFO", f"queue_for_speech received: '{text}'")

    speakable_text, written_text = extract_speakable_text(text)

    if not speakable_text or len(speakable_text) < MIN_SPEAK_LENGTH:
        log_message("INFO", f"Text too short: '{text}' -> '{speakable_text}'")
        return ""
    
    # Add sentence ending if needed
    speakable_text = add_sentence_ending(speakable_text)
    
    # Clean up any awkward punctuation sequences
    speakable_text = clean_punctuation_sequences(speakable_text)

    # If filtered text is longer than 20 characters and auto-skip is enabled, clear the queue to jump to this message
    if auto_skip_tts_enabled and len(speakable_text) > 20:
        log_message("INFO", f"Long text detected ({len(speakable_text)} chars), clearing queue to jump to latest")
        # Clear the queue
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except:
                break
        # Also skip current item if playing
        playback_control.skip_current_item()

    # Debounce rapidly changing text
    current_time = time.time()
    if (last_queued_text and
            SequenceMatcher(None, speakable_text.lower(), last_queued_text.lower()).ratio() > 0.7 and
            current_time - last_queue_time < DEBOUNCE_TIME):
        log_message("INFO", f"Debouncing similar text: '{speakable_text}'")
        last_queued_text = speakable_text
        last_queue_time = current_time
        return ""

    if is_similar_to_recent(speakable_text):
        log_message("INFO", f"Recently spoken: '{speakable_text}'")
        return ""

    # Add to cache and queue
    spoken_cache.append((speakable_text, current_time))
    last_queued_text = speakable_text
    last_queue_time = current_time

    # Create speech item
    speech_item = SpeechItem(
        text=speakable_text,
        original_text=text,
        line_number=line_number,
        source=source
    )

    try:
        tts_queue.put_nowait(speech_item)
    except queue.Full:
        log_message("WARNING", "TTS queue full, skipping text")

    return written_text


def start_tts_worker(engine: str, auto_skip_tts: bool = False) -> threading.Thread:
    """Start the TTS worker thread"""
    global tts_worker_thread, auto_skip_tts_enabled
    auto_skip_tts_enabled = auto_skip_tts
    tts_worker_thread = threading.Thread(target=tts_worker, args=(engine,), daemon=True)
    tts_worker_thread.start()
    return tts_worker_thread


def reset_tts_cache():
    """Reset TTS caches - useful for testing"""
    global spoken_cache, last_queued_text, last_queue_time, highest_spoken_line_number
    
    spoken_cache.clear()
    last_queued_text = ""
    last_queue_time = 0
    highest_spoken_line_number = -1
    
    # Also clear the queue
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except:
            break
    
    log_message("INFO", "TTS cache reset")


def wait_for_tts_to_finish(timeout: Optional[float] = None) -> bool:
    """Wait for all queued TTS to finish playing. Returns True if finished, False if timed out."""
    start_time = time.time()
    
    # Wait for queue to empty
    while not tts_queue.empty():
        if timeout and (time.time() - start_time) > timeout:
            return False
        time.sleep(0.1)
    
    # Wait for current item to finish speaking
    while current_speech_item is not None:
        if timeout and (time.time() - start_time) > timeout:
            return False
        time.sleep(0.1)
    
    # Wait a bit more for audio buffer to play out
    time.sleep(SPEECH_BUFFER_TIME)
    
    log_message("INFO", "All TTS finished")
    return True


def shutdown_tts():
    """Shutdown the TTS system gracefully"""
    shutdown_event.set()
    tts_queue.put("__SHUTDOWN__")
    if tts_worker_thread:
        tts_worker_thread.join(timeout=0.5)


# Playback control functions
def pause_playback():
    """Pause TTS playback"""
    playback_control.pause()
    log_message("INFO", "Playback paused")


def resume_playback():
    """Resume TTS playback"""
    playback_control.resume()
    log_message("INFO", "Playback resumed")


def skip_current():
    """Skip the currently playing item"""
    playback_control.skip_current_item()
    log_message("INFO", "Skipping current item")


def skip_all():
    """Skip all remaining items in the queue"""
    playback_control.skip_all_items()
    # Clear the queue
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except:
            break
    log_message("INFO", "Skipped all items")


def navigate_to_previous():
    """Navigate to previous item in history"""
    if len(speech_history) > 1:
        # Skip current and get previous
        playback_control.skip_current_item()
        prev_item = speech_history[-2]
        # Re-queue the previous item
        tts_queue.put_nowait(prev_item)
        log_message("INFO", f"Navigating to previous item: {prev_item.text[:50]}...")
        return True
    return False


def navigate_to_next():
    """Skip to next item in queue"""
    playback_control.skip_current_item()
    log_message("INFO", "Navigating to next item")
    return True


def get_current_speech_item() -> Optional[SpeechItem]:
    """Get the currently speaking item"""
    return current_speech_item


def get_speech_history() -> List[SpeechItem]:
    """Get the speech history"""
    return speech_history.copy()


def get_queue_size() -> int:
    """Get the current size of the speech queue"""
    return tts_queue.qsize()


def is_paused() -> bool:
    """Check if playback is paused"""
    return playback_control.is_paused


def is_speaking() -> bool:
    """Check if TTS is currently speaking or recently finished"""
    # Check if actively speaking
    if current_speech_item is not None:
        return True
    
    # Check if we recently finished speaking (audio might still be playing)
    time_since_speech = time.time() - last_speech_end_time
    return time_since_speech < SPEECH_BUFFER_TIME


def get_highest_spoken_line_number() -> int:
    """Get the highest line number that has been spoken"""
    return highest_spoken_line_number


def parse_arguments():
    """Parse command-line arguments for TTS provider selection"""
    parser = argparse.ArgumentParser(description='Text-to-Speech with multiple provider support')
    parser.add_argument('--tts-provider', type=str, default=None,
                       choices=['system', 'openai', 'aws', 'polly', 'azure', 'gcloud', 'elevenlabs', 'deepgram'],
                       help='TTS provider to use (default: auto-select best available)')
    parser.add_argument('--voice', type=str, default=None,
                       help='Voice to use (provider-specific)')
    parser.add_argument('--language', type=str, default='en-US',
                       help='Language code (mainly for Google Cloud TTS, default: en-US)')
    parser.add_argument('--region', type=str, default=None,
                       help='Region for cloud providers (AWS Polly/Azure)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Enable debug logging to file (e.g., ~/.talkito_tts.log)')
    parser.add_argument('text', nargs='*', help='Text to speak (multiple words allowed)')
    return parser.parse_args()


def configure_tts_provider(args):
    """Configure TTS provider based on command-line arguments"""
    global tts_provider, openai_voice, polly_voice, polly_region, azure_voice, azure_region, gcloud_voice, gcloud_language_code, elevenlabs_voice_id
    
    # Auto-select provider if not specified
    provider = args.tts_provider
    if provider is None:
        provider = select_best_tts_provider()
        print(f"Auto-selected TTS provider: {provider}")
    
    tts_provider = provider
    
    # Validate provider configuration
    if not validate_provider_config(tts_provider):
        return False
    
    # Get provider info
    provider_info = TTS_PROVIDERS.get(tts_provider)
    if not provider_info:
        # System provider
        log_message("INFO", "Using system default TTS")
        return True
    
    # Handle provider-specific configuration
    if tts_provider == 'openai':
        if args.voice:
            openai_voice = args.voice
        log_message("INFO", f"Using OpenAI TTS with voice: {openai_voice}")
        
    elif tts_provider in ['aws', 'polly']:
        polly_region = args.region or 'us-east-1'
        if args.voice:
            polly_voice = args.voice
        log_message("INFO", f"Using AWS Polly TTS with voice: {polly_voice} in region: {polly_region}")
        
    elif tts_provider == 'azure':
        # Additional Azure-specific validation
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            return False
            
        azure_region = args.region or 'eastus'
        if args.voice:
            azure_voice = args.voice
        log_message("INFO", f"Using Microsoft Azure TTS with voice: {azure_voice} in region: {azure_region}")
        
    elif tts_provider == 'gcloud':
        # Additional Google Cloud validation
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            print("Note: On macOS, grpcio may require a clean reinstall:")
            print("      pip uninstall grpcio && pip install grpcio --force-reinstall --no-cache-dir")
            return False
        except Exception as e:
            print(f"Error: Google Cloud credentials not configured or invalid: {e}")
            print("Please check your GOOGLE_APPLICATION_CREDENTIALS file")
            return False
            
        gcloud_language_code = args.language
        if args.voice:
            gcloud_voice = args.voice
        log_message("INFO", f"Using Google Cloud TTS with voice: {gcloud_voice} in language: {gcloud_language_code}")
        
    elif tts_provider == 'elevenlabs':
        if args.voice:
            elevenlabs_voice_id = args.voice
        log_message("INFO", f"Using ElevenLabs TTS with voice ID: {elevenlabs_voice_id}")
    
    return True


def select_best_tts_provider() -> str:
    """Select the best available TTS provider based on accessibility and preferences.
    
    Order of preference:
    1. TALKITO_PREFERRED_TTS_PROVIDER from environment (if accessible)
    2. First accessible non-system provider (alphabetically)
    3. System provider as fallback
    """
    preferred = os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER')
    accessible = check_tts_provider_accessibility()
    
    # Check if preferred provider is accessible
    if preferred and preferred in accessible and accessible[preferred]['available']:
        log_message("INFO", f"Using preferred TTS provider: {preferred}")
        return preferred
    
    # Get all accessible providers except system
    available_providers = [
        provider for provider, info in sorted(accessible.items())
        if info['available'] and provider != 'system'
    ]
    
    # Use first available non-system provider
    if available_providers:
        provider = available_providers[0]
        log_message("INFO", f"Selected TTS provider: {provider} (first available)")
        return provider
    
    # Fall back to system if available
    if accessible.get('system', {}).get('available'):
        log_message("INFO", "Falling back to system TTS provider")
        return 'system'
    
    # No providers available
    log_message("WARNING", "No TTS providers available")
    return 'system'


def configure_tts_from_dict(config: dict) -> bool:
    """Configure TTS provider from a configuration dictionary (for use by talkito.py)"""
    global tts_provider, openai_voice, polly_voice, polly_region, azure_voice, azure_region, gcloud_voice, gcloud_language_code, elevenlabs_voice_id, elevenlabs_model_id, deepgram_voice_model
    
    provider = config.get('provider', 'system')
    tts_provider = provider
    
    # Validate provider configuration
    if not validate_provider_config(provider):
        return False
    
    # Get provider info
    provider_info = TTS_PROVIDERS.get(provider)
    if not provider_info:
        # System provider
        log_message("INFO", "Using system default TTS")
        return True
    
    # Handle provider-specific configuration
    if provider == 'openai':
        if config.get('voice'):
            openai_voice = config['voice']
        log_message("INFO", f"Using OpenAI TTS with voice: {openai_voice}")
        
    elif provider in ['aws', 'polly']:
        # Additional AWS validation
        try:
            import boto3
            region = config.get('region', 'us-east-1')
            test_client = boto3.client('polly', region_name=region)
            test_client.describe_voices(LanguageCode='en-US')
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            return False
        except Exception as e:
            print(f"Error: AWS credentials not configured or invalid: {e}")
            print("Please configure AWS credentials (e.g., AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)")
            return False
            
        polly_region = config.get('region', 'us-east-1')
        if config.get('voice'):
            polly_voice = config['voice']
        log_message("INFO", f"Using AWS Polly TTS with voice: {polly_voice} in region: {polly_region}")
        
    elif provider == 'azure':
        # Additional Azure validation
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            return False
            
        azure_region = config.get('region', 'eastus')
        if config.get('voice'):
            azure_voice = config['voice']
        log_message("INFO", f"Using Microsoft Azure TTS with voice: {azure_voice} in region: {azure_region}")
        
    elif provider == 'gcloud':
        # Additional Google Cloud validation
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            print("Note: On macOS, grpcio may require a clean reinstall:")
            print("      pip uninstall grpcio && pip install grpcio --force-reinstall --no-cache-dir")
            return False
        except Exception as e:
            print(f"Error: Google Cloud credentials not configured or invalid: {e}")
            print("Please check your GOOGLE_APPLICATION_CREDENTIALS file")
            return False
            
        gcloud_language_code = config.get('language', 'en-US')
        if config.get('voice'):
            gcloud_voice = config['voice']
        log_message("INFO", f"Using Google Cloud TTS with voice: {gcloud_voice} in language: {gcloud_language_code}")
        
    elif provider == 'elevenlabs':
        if config.get('voice'):
            elevenlabs_voice_id = config['voice']
        if config.get('model'):
            elevenlabs_model_id = config['model']
        log_message("INFO", f"Using ElevenLabs TTS with voice ID: {elevenlabs_voice_id}")
    
    elif provider == 'deepgram':
        if config.get('model'):
            deepgram_voice_model = config['model']
        log_message("INFO", f"Using Deepgram TTS with model: {deepgram_voice_model}")
    
    return True


# Main entry point for testing
if __name__ == "__main__":
    import sys
    
    # Parse arguments
    args = parse_arguments()
    
    # Configure TTS provider
    if not configure_tts_provider(args):
        exit(1)
    
    # Set up logging - if log-file is specified
    # Set up logging using centralized logging from logs module
    if args.log_file:
        try:
            from .logs import setup_logging
        except ImportError:
            # Running standalone - try direct import
            from logs import setup_logging
        setup_logging(args.log_file, mode='w')
        print(f"Logging to: {args.log_file}")
    
    # Determine text to speak
    text_to_speak = None
    
    if args.text:
        # Join multiple words from command line
        text_to_speak = ' '.join(args.text)
    elif not sys.stdin.isatty():
        # Read from stdin if piped
        text_to_speak = sys.stdin.read().strip()
    else:
        # Interactive mode - prompt for input
        print("Interactive TTS mode. Type text and press Enter to speak.")
        print("Press Ctrl+D (EOF) or Ctrl+C to exit.")
        
        # Detect TTS engine based on provider
        if tts_provider == 'system':
            engine = detect_tts_engine()
            if engine == "none":
                print("Error: No TTS engine found on this system")
                exit(1)
        else:
            engine = 'cloud'  # Use cloud engine for external providers
        
        # Start the TTS worker for interactive mode
        start_tts_worker(engine)
        
        try:
            while True:
                try:
                    line = input("> ")
                    if line.strip():  # Only speak non-empty lines
                        queue_for_speech(line)
                        # Give a moment for speech to start
                        time.sleep(0.1)
                except EOFError:
                    print("\nExiting interactive mode.")
                    break
        except KeyboardInterrupt:
            print("\nCancelled.")
        finally:
            # Wait for any remaining speech to complete
            wait_for_tts_to_finish()
            shutdown_tts()
            exit(0)
    
    # Speak the text if we have any
    if text_to_speak:
        # Detect TTS engine based on provider
        if tts_provider == 'system':
            engine = detect_tts_engine()
            if engine == "none":
                print("Error: No TTS engine found on this system")
                exit(1)
        else:
            engine = 'cloud'  # Use cloud engine for external providers
        
        # Start the TTS worker
        start_tts_worker(engine)
        
        # Queue the text for speech
        queue_for_speech(text_to_speak)
        
        # Wait for speech to complete
        wait_for_tts_to_finish()
        
        # Shutdown the TTS worker
        shutdown_tts()
        
        log_message("INFO", "Speech completed")
    else:
        print("No text provided to speak.")