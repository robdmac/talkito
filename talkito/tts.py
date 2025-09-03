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

"""Text-to-Speech engine and text processing utilities with TTS engine detection, symbol conversion, and speech synthesis support."""

import subprocess
import shutil
import re
import time
import threading
import queue
import os
import argparse
import tempfile
import sys
import random
import signal
from collections import deque
from typing import Optional, List, Tuple, Deque, Dict, Any, Callable
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

# Import centralized logging utilities
try:
    from .logs import log_message as _base_log_message
except ImportError:
    # Fallback for standalone execution
    def _base_log_message(level: str, message: str, logger_name: str = None):
        print(f"[{level}] {message}")

# Import shared state
try:
    from .state import get_shared_state
    SHARED_STATE_AVAILABLE = True
except ImportError:
    SHARED_STATE_AVAILABLE = False

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
CACHE_TIMEOUT = 18000  # Seconds before a cached item can be spoken again
SIMILARITY_THRESHOLD = 0.85  # How similar text must be to be considered a repeat
DEBOUNCE_TIME = 0.5  # Seconds to wait before speaking rapidly changing text
SKIP_INTERJECTIONS = ["oh", "hmm", "um", "right", "okay"]  # Interjections to add when auto-skipping

# Global configuration
auto_skip_tts_enabled = False  # Whether to auto-skip long text
disable_tts = False  # Whether to disable TTS completely (for testing)
tts_provider = "system"  # Current TTS provider (system, openai, polly, azure, gcloud, elevenlabs, deepgram, etc.)
_kittentts_warning_shown = False  # Track if KittenTTS warning has been shown
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
kittentts_model = os.environ.get('KITTENTTS_MODEL', 'KittenML/kitten-tts-nano-0.2')  # Default KittenTTS model
kittentts_voice = os.environ.get('KITTENTTS_VOICE', 'expr-voice-3-f')  # Default KittenTTS voice
kokoro_language = os.environ.get('KOKORO_LANGUAGE', 'a')  # Default Kokoro language (American English)
kokoro_voice = os.environ.get('KOKORO_VOICE', 'af_heart')  # Default Kokoro voice
kokoro_speed = os.environ.get('KOKORO_SPEED', '1.0')  # Default Kokoro speed


def get_tts_config():
    """Get TTS configuration from shared state or module globals."""
    if SHARED_STATE_AVAILABLE:
        try:
            state = get_shared_state()
            config = {
                'provider': state.tts_provider or tts_provider,
                'voice': None,
                'region': None,
                'language': None,
                'rate': state.tts_rate,
                'pitch': state.tts_pitch
            }
            
            # Map provider-specific voice settings
            if state.tts_provider == 'openai':
                config['voice'] = state.tts_voice or openai_voice
            elif state.tts_provider in ['aws', 'polly']:
                config['voice'] = state.tts_voice or polly_voice
                config['region'] = state.tts_region or polly_region
            elif state.tts_provider == 'azure':
                config['voice'] = state.tts_voice or azure_voice
                config['region'] = state.tts_region or azure_region
            elif state.tts_provider == 'gcloud':
                config['voice'] = state.tts_voice or gcloud_voice
                config['language'] = state.tts_language or gcloud_language_code
            elif state.tts_provider == 'elevenlabs':
                config['voice'] = state.tts_voice or elevenlabs_voice_id
            elif state.tts_provider == 'deepgram':
                config['voice'] = state.tts_voice or deepgram_voice_model
            elif state.tts_provider == 'kittentts':
                config['voice'] = state.tts_voice or kittentts_voice
                config['model'] = state.tts_model or kittentts_model
            elif state.tts_provider == 'kokoro':
                config['voice'] = state.tts_voice or kokoro_voice
                config['language'] = state.tts_language or kokoro_language
                config['speed'] = float(state.tts_rate or kokoro_speed)
            
            return config
        except Exception:
            pass
    
    # Fallback to module globals
    return {
        'provider': tts_provider,
        'voice': None,
        'region': None,
        'language': None,
        'rate': None,
        'pitch': None
    }


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
    },
    'kittentts': {
        'func': None,  # Will be set to speak_with_kittentts after function definition
        'env_var': None,  # KittenTTS doesn't need an API key
        'model_var': 'kittentts_model',
        'voice_var': 'kittentts_voice',
        'display_name': 'KittenTTS',
        'install': 'pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl soundfile',
        'config_keys': ['model', 'voice']
    },
    'kokoro': {
        'func': None,  # Will be set to speak_with_kokoro after function definition
        'env_var': None,  # Kokoro doesn't need an API key
        'language_var': 'kokoro_language',
        'voice_var': 'kokoro_voice',
        'speed_var': 'kokoro_speed',
        'display_name': 'KokoroTTS',
        'install': 'pip install kokoro>=0.9.4 soundfile',
        'config_keys': ['language', 'voice', 'speed']
    }
}


@dataclass
class SpeechItem:
    """Item in speech queue with text, timestamp, and metadata."""
    text: str
    original_text: str
    line_number: Optional[int] = None
    timestamp: Optional[datetime] = None
    start_time: Optional[float] = None  # Time when speech actually starts playing
    source: str = "output"  # "output", "error", etc.
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PlaybackControl:
    """Controls TTS playback state and process management."""
    def __init__(self):
        self.current_index = 0
        self.is_paused = False
        self.skip_current = False
        self.skip_all = False
        self.current_process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        
    def pause(self):
        """Pause current TTS playback."""
        with self.lock:
            self.is_paused = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
                    
    def resume(self):
        """Resume paused TTS playback."""
        with self.lock:
            self.is_paused = False
            
    def skip_current_item(self):
        """Skip the currently playing TTS item."""
        with self.lock:
            self.skip_current = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
                    
    def skip_all_items(self):
        """Skip all remaining items in TTS queue."""
        with self.lock:
            self.skip_all = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
                    
    def reset_skip_flags(self):
        """Reset all skip flags to false."""
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

# Thread safety locks
_state_lock = threading.RLock()  # Reentrant lock for nested access
_cache_lock = threading.Lock()  # Separate lock for cache operations


# Wrapper for module-specific logging
def log_message(level: str, message: str):
    """Log message with module name using centralized logger."""
    _base_log_message(level, message, __name__)


def check_tts_provider_accessibility(requested_provider: str = None) -> Dict[str, Dict[str, Any]]:
    """Check TTS provider accessibility based on API keys and environment."""
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
    
    # Deepgram
    accessible["deepgram"] = {
        "available": bool(os.environ.get("DEEPGRAM_API_KEY")),
        "note": "Requires DEEPGRAM_API_KEY environment variable"
    }
    
    # KittenTTS - Check if dependencies are installed
    kittentts_available = False
    kittentts_note = "Ultra-lightweight TTS that runs without GPU (no API key required)"
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
            from kittentts import KittenTTS
            import soundfile as sf
        # Check if the model is cached
        from .models import check_model_cached, with_download_progress
        model_name = kittentts_model  # Use the actual configured model name
        if not check_model_cached('kittentts', model_name):
            # Only prompt for download if this provider was specifically requested
            if requested_provider == 'kittentts' or os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER') == 'kittentts':
                # Ask user for consent and download if approved
                try:
                    def create_model():
                        return KittenTTS(model_name)
                    
                    decorated_func = with_download_progress('kittentts', model_name, create_model)
                    decorated_func()  # This will ask for consent and download
                    kittentts_available = True
                except RuntimeError as e:
                    if "Download cancelled" in str(e):
                        kittentts_note = f"KittenTTS model '{model_name}' download declined by user"
                    else:
                        kittentts_note = f"KittenTTS model download failed: {e}"
                    kittentts_available = False
                except Exception as e:
                    kittentts_note = f"KittenTTS model download failed: {e}"
                    kittentts_available = False
            else:
                # Model not cached and not specifically requested - mark as unavailable  
                kittentts_note = f"KittenTTS model '{model_name}' not cached. Will be downloaded on first use."
                kittentts_available = False
        else:
            kittentts_available = True
    except ImportError:
        kittentts_note = "Requires KittenTTS package (pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl soundfile)"
    
    accessible["kittentts"] = {
        "available": kittentts_available,
        "note": kittentts_note
    }
    
    
    # KokoroTTS - Check if dependencies are installed
    kokoro_available = False
    kokoro_note = "High-quality 82M parameter TTS model (no API key required)"
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
            from kokoro import KPipeline
            import soundfile as sf
        # Check if the model is cached
        from .models import check_model_cached, with_download_progress
        if not check_model_cached('kokoro', 'default'):
            # Only prompt for download if this provider was specifically requested
            if requested_provider == 'kokoro' or os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER') == 'kokoro':
                # Ask user for consent and download if approved
                try:
                    repo_id = 'hexgrad/Kokoro-82M'
                    def create_pipeline():
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
                            return KPipeline(lang_code='en-us', repo_id=repo_id)
                    
                    decorated_func = with_download_progress('kokoro', 'default', create_pipeline)
                    decorated_func()  # This will ask for consent and download
                    kokoro_available = True
                except RuntimeError as e:
                    if "Download cancelled" in str(e):
                        kokoro_note = "KokoroTTS model download declined by user"
                    else:
                        kokoro_note = f"KokoroTTS model download failed: {e}"
                    kokoro_available = False
                except Exception as e:
                    kokoro_note = f"KokoroTTS model download failed: {e}"
                    kokoro_available = False
            else:
                # Model not cached and not specifically requested - mark as unavailable
                kokoro_note = "KokoroTTS model not cached. Will be downloaded on first use."
                kokoro_available = False
        else:
            kokoro_available = True
    except ImportError:
        kokoro_note = "Requires KokoroTTS package (pip install kokoro>=0.9.4 soundfile)"
    
    accessible["kokoro"] = {
        "available": kokoro_available,
        "note": kokoro_note
    }
    
    
    return accessible


def detect_tts_engine() -> str:
    """Detect system TTS engine (say, espeak, festival, flite)."""
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
    """Create temporary audio file with given suffix."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        return tmp_file.name


def _play_audio_file(audio_path: str, use_process_control: bool = True) -> bool:
    """Play audio file using system player (afplay, mpg123, play)."""
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
        # Poll process with timeout to check for interruptions
        while process.poll() is None:
            # Check if we should stop
            if shutdown_event.is_set() or (use_process_control and (playback_control.skip_current or playback_control.skip_all)):
                process.terminate()
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        return process.returncode == 0
    except:
        return False
    finally:
        if use_process_control:
            with playback_control.lock:
                playback_control.current_process = None


def _cleanup_temp_file(file_path: str):
    """Remove temporary file if it exists."""
    if os.path.exists(file_path):
        os.unlink(file_path)


def _handle_import_error(library_name: str, install_command: str) -> bool:
    """Log import error and return False."""
    log_message("ERROR", f"{library_name} library not installed. Run: {install_command}")
    return False


def _handle_provider_error(provider_name: str, error: Exception) -> bool:
    """Log provider error and return False."""
    log_message("ERROR", f"{provider_name} TTS failed: {error}")
    return False


def synthesize_and_play(synthesize_func, text: str, use_process_control: bool = True) -> bool:
    """Synthesize audio via provider function and play it."""
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
    """Validate provider configuration and API keys."""
    provider_info = TTS_PROVIDERS.get(provider)
    if not provider_info:
        return True  # System provider, no validation needed
    
    # Check environment variable if required
    env_var = provider_info.get('env_var')
    if env_var and not os.environ.get(env_var):
        print(f"Error: {env_var} environment variable not set")
        print(f"Please set it with: export {env_var}='your-api-key'")
        return False
    
    # Special case for KittenTTS - check if package is installed
    if provider == 'kittentts':
        global _kittentts_warning_shown
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
                from kittentts import KittenTTS
                import soundfile as sf
        except ImportError:
            # Only print the installation message once
            if not _kittentts_warning_shown:
                print(f"Error: KittenTTS dependencies not installed")
                print(f"Please install with:")
                print(f"  pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl")
                print(f"  pip install soundfile")
                _kittentts_warning_shown = True
            log_message("DEBUG", "KittenTTS dependencies not installed")
            return False
    
    # Special case for KokoroTTS - check if package is installed
    if provider == 'kokoro':
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
                from kokoro import KPipeline
                import soundfile as sf
        except ImportError:
            print(f"Error: KokoroTTS dependencies not installed")
            print(f"Please install with:")
            print(f"  pip install kokoro>=0.9.4 soundfile")
            log_message("DEBUG", "KokoroTTS dependencies not installed")
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


class TTSProvider(ABC):
    """Base class for all TTS providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize provider with optional configuration dict."""
        self.config = config or {}
        self.provider_name = self.__class__.__name__.replace('Provider', '')
    
    @abstractmethod
    def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize speech from text, return audio bytes or None."""
        pass
    
    @abstractmethod
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate provider config, return (is_valid, error_message)."""
        pass
    
    def speak(self, text: str, use_process_control: bool = True) -> bool:
        """Synthesize and play audio, return True if successful."""
        return synthesize_and_play(self.synthesize, text, use_process_control)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get config value from instance, shared state, or default."""
        # First check instance config
        if key in self.config:
            return self.config[key]
        
        # Then check shared state
        from .state import get_shared_state
        shared_state = get_shared_state()
        if hasattr(shared_state, 'tts') and hasattr(shared_state.tts, key):
            return getattr(shared_state.tts, key)
        
        # Finally return default
        return default


def _synthesize_openai(text: str) -> Optional[bytes]:
    """Synthesize speech using OpenAI TTS API."""
    import openai
    import io
    
    # Get configuration from shared state
    config = get_tts_config()
    voice = config.get('voice') or openai_voice
    
    # Generate speech using OpenAI
    response = openai.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    
    # Convert streaming response to bytes
    audio_data = io.BytesIO()
    for chunk in response.iter_bytes():
        audio_data.write(chunk)
    
    return audio_data.getvalue()


def speak_with_openai(text: str) -> bool:
    """Speak text using OpenAI TTS API."""
    try:
        import openai
        # Check for API key is now done by validate_provider_config
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            log_message("ERROR", "OPENAI_API_KEY environment variable not set")
            return False
        
        return synthesize_and_play(_synthesize_openai, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("OpenAI", "pip install openai")
    except Exception as e:
        return _handle_provider_error("OpenAI", e)


def _synthesize_polly(text: str) -> Optional[bytes]:
    """Synthesize speech using AWS Polly TTS API."""
    import boto3
    
    # Get configuration from shared state
    config = get_tts_config()
    voice = config.get('voice') or polly_voice
    region = config.get('region') or polly_region
    
    # Create Polly client
    polly_client = boto3.client('polly', region_name=region)
    
    # Generate speech using AWS Polly
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId=voice,
        Engine='neural'  # Use neural voice for better quality
    )
    
    # Return audio stream as bytes
    return response['AudioStream'].read()


def speak_with_polly(text: str) -> bool:
    """Speak text using AWS Polly TTS API."""
    try:
        import boto3
        return synthesize_and_play(_synthesize_polly, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("boto3", "pip install boto3")
    except Exception as e:
        return _handle_provider_error("AWS Polly", e)


def _synthesize_azure(text: str) -> Optional[bytes]:
    """Synthesize speech using Microsoft Azure TTS API."""
    import azure.cognitiveservices.speech as speechsdk
    
    # Get configuration from shared state
    config = get_tts_config()
    voice = config.get('voice') or azure_voice
    region = config.get('region') or azure_region
    
    # Check for API key and region
    speech_key = os.environ.get('AZURE_SPEECH_KEY')
    if not speech_key:
        raise ValueError("AZURE_SPEECH_KEY environment variable not set")
    
    # Create speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    
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
    """Speak text using Microsoft Azure TTS API."""
    try:
        import azure.cognitiveservices.speech as speechsdk
        return synthesize_and_play(_synthesize_azure, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("Azure Speech SDK", "pip install azure-cognitiveservices-speech")
    except Exception as e:
        return _handle_provider_error("Microsoft Azure", e)


def _synthesize_gcloud(text: str) -> Optional[bytes]:
    """Synthesize speech using Google Cloud Text-to-Speech API."""
    from google.cloud import texttospeech
    
    # Get configuration from shared state
    config = get_tts_config()
    voice_name = config.get('voice') or gcloud_voice
    language = config.get('language') or gcloud_language_code
    
    # Create a client
    client = texttospeech.TextToSpeechClient()
    
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request - handle both simple voice names and full voice specs
    if voice_name and '-' in voice_name:
        # Full voice name like "en-US-Journey-F"
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name=voice_name
        )
    else:
        # Simple voice name - let Google pick the best match
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
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
    """Speak text using Google Cloud Text-to-Speech API."""
    try:
        from google.cloud import texttospeech
        return synthesize_and_play(_synthesize_gcloud, text, use_process_control=True)
    except ImportError:
        return _handle_import_error("Google Cloud Text-to-Speech", "pip install google-cloud-texttospeech")
    except Exception as e:
        return _handle_provider_error("Google Cloud", e)


def _synthesize_elevenlabs(text: str) -> Optional[bytes]:
    """Synthesize speech using ElevenLabs TTS API."""
    import requests
    
    # Get configuration from shared state
    config = get_tts_config()
    voice_id = config.get('voice') or elevenlabs_voice_id
    
    # Get API key
    api_key = os.environ.get('ELEVENLABS_API_KEY')
    
    # ElevenLabs API endpoint
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
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
    """Speak text using ElevenLabs TTS API."""
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
    """Synthesize speech using Deepgram TTS API."""
    from deepgram import DeepgramClient, SpeakOptions
    import tempfile
    import os
    
    # Get configuration from shared state
    config = get_tts_config()
    model = config.get('voice') or deepgram_voice_model
    
    # Get API key
    api_key = os.environ.get('DEEPGRAM_API_KEY')
    
    # Create Deepgram client
    deepgram = DeepgramClient(api_key)
    
    # Configure options
    options = SpeakOptions(
        model=model,
        encoding="mp3"
    )
    
    # Initialize tmp_filename before try block to ensure it's always defined
    tmp_filename = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
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
        # Clean up temp file - tmp_filename is guaranteed to be defined
        if tmp_filename and os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def speak_with_deepgram(text: str) -> bool:
    """Speak text using Deepgram TTS API."""
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


def _synthesize_kittentts(text: str) -> Optional[bytes]:
    """Synthesize speech using KittenTTS."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
            from kittentts import KittenTTS
            import soundfile as sf
        import tempfile
        import os
        
        # Get configuration from shared state
        config = get_tts_config()
        model_name = config.get('model') or kittentts_model
        voice = config.get('voice') or kittentts_voice
        
        # Create KittenTTS model (should already be cached from availability check)
        log_message("DEBUG", f"Using KittenTTS model: {model_name}")
        m = KittenTTS(model_name)
        
        # Generate audio with the specified voice
        audio = m.generate(text, voice=voice)
        
        # Save to temporary WAV file first (soundfile doesn't support MP3 writing directly)
        # KittenTTS returns audio at 24000 Hz sample rate
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            sf.write(tmp_wav.name, audio, 24000)
            tmp_wav_path = tmp_wav.name
        
        try:
            # Convert WAV to MP3 using ffmpeg if available, otherwise use WAV directly
            if shutil.which('ffmpeg'):
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                    tmp_mp3_path = tmp_mp3.name
                
                # Convert to MP3
                subprocess.run(
                    ['ffmpeg', '-i', tmp_wav_path, '-acodec', 'mp3', '-ab', '128k', tmp_mp3_path, '-y'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                
                # Read MP3 data
                with open(tmp_mp3_path, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up MP3 file
                os.unlink(tmp_mp3_path)
            else:
                # If ffmpeg not available, use WAV directly
                with open(tmp_wav_path, 'rb') as f:
                    audio_data = f.read()
            
            return audio_data
        finally:
            # Always clean up WAV file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
    except Exception as e:
        log_message("ERROR", f"KittenTTS synthesis error: {e}")
        return None


def speak_with_kittentts(text: str) -> bool:
    """Speak text using KittenTTS."""
    try:
        return synthesize_and_play(_synthesize_kittentts, text, use_process_control=True)
    except Exception as e:
        return _handle_provider_error("KittenTTS", e)


def _synthesize_kokoro(text: str) -> Optional[bytes]:
    """Synthesize speech using KokoroTTS."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            from kokoro import KPipeline
            import soundfile as sf
        import tempfile
        import os
        
        # Get configuration from shared state
        config = get_tts_config()
        language = config.get('language') or kokoro_language
        voice = config.get('voice') or kokoro_voice
        speed = float(config.get('speed') or kokoro_speed)
        
        # Create Kokoro pipeline (should already be cached from availability check)
        repo_id = 'hexgrad/Kokoro-82M'
        log_message("DEBUG", f"Using KokoroTTS model")
        
        # Suppress torch warnings during pipeline creation and usage
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            pipeline = KPipeline(lang_code=language, repo_id=repo_id)
            
            # Generate audio with the specified voice and speed
            # Kokoro returns a generator, we need to process all chunks
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=speed)):
                audio_chunks.append(audio)
        
        # Concatenate all audio chunks
        import numpy as np
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
        else:
            log_message("ERROR", "KokoroTTS generated no audio")
            return None
        
        # Save to temporary WAV file first (Kokoro outputs at 24000 Hz)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            sf.write(tmp_wav.name, full_audio, 24000)
            tmp_wav_path = tmp_wav.name
        
        try:
            # Convert WAV to MP3 using ffmpeg if available, otherwise use WAV directly
            if shutil.which('ffmpeg'):
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                    tmp_mp3_path = tmp_mp3.name
                
                # Convert to MP3
                subprocess.run(
                    ['ffmpeg', '-i', tmp_wav_path, '-acodec', 'mp3', '-ab', '128k', tmp_mp3_path, '-y'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                
                # Read MP3 data
                with open(tmp_mp3_path, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up MP3 file
                os.unlink(tmp_mp3_path)
            else:
                # If ffmpeg not available, use WAV directly
                with open(tmp_wav_path, 'rb') as f:
                    audio_data = f.read()
            
            return audio_data
        finally:
            # Always clean up WAV file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
    except Exception as e:
        log_message("ERROR", f"KokoroTTS synthesis error: {e}")
        return None


def speak_with_kokoro(text: str) -> bool:
    """Speak text using KokoroTTS."""
    try:
        return synthesize_and_play(_synthesize_kokoro, text, use_process_control=True)
    except Exception as e:
        return _handle_provider_error("KokoroTTS", e)


# Initialize provider functions in the registry
TTS_PROVIDERS['openai']['func'] = speak_with_openai
TTS_PROVIDERS['aws']['func'] = speak_with_polly
TTS_PROVIDERS['azure']['func'] = speak_with_azure
TTS_PROVIDERS['gcloud']['func'] = speak_with_gcloud
TTS_PROVIDERS['elevenlabs']['func'] = speak_with_elevenlabs
TTS_PROVIDERS['deepgram']['func'] = speak_with_deepgram
TTS_PROVIDERS['kittentts']['func'] = speak_with_kittentts
TTS_PROVIDERS['kokoro']['func'] = speak_with_kokoro

# Add 'polly' as an alias for 'aws' for backward compatibility
TTS_PROVIDERS['polly'] = TTS_PROVIDERS['aws']


def context_aware_symbol_replacement(text: str) -> str:
    """Replace mathematical and special symbols based on context."""
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
    """Add period if text doesn't end with punctuation."""
    if not text:
        return text
    
    # Check if text already ends with sentence-ending punctuation
    text = text.rstrip()  # Remove trailing whitespace
    if text and text[-1] not in '.!?:;':
        return text + '.'
    return text


def clean_punctuation_sequences(text: str) -> str:
    """Clean awkward punctuation sequences like '?.' or '!'."""
    # Replace punctuation followed by period+space with just the punctuation+space
    text = re.sub(r'([!?:;])\.\s', r'\1 ', text)
    # Replace punctuation followed by period at end with just the punctuation
    text = re.sub(r'([!?:;])\.$', r'\1', text)
    # Only clean up multiple periods with spaces between (like ". ." or ". . .")
    # This preserves intentional ellipsis like "..." or ".."
    text = re.sub(r'\.(\s+\.)+', '.', text)
    return text


def extract_speakable_text(text: str) -> (str, str):
    """Extract speakable text and convert mathematical symbols."""
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
    """Check if text is similar to recently spoken text."""
    current_time = time.time()

    with _cache_lock:
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
    """Speak text using appropriate engine, return completion status."""
    if disable_tts:
        return True

    # Use provider registry for cleaner routing
    provider_info = TTS_PROVIDERS.get(tts_provider)
    if provider_info and provider_info.get('func'):
        return provider_info['func'](text)

    return speak_with_default(text, engine)


def save_tts_audio(text: str, filename: str, provider: str = None) -> bool:
    """Save TTS audio to a file instead of playing it.
    
    Args:
        text: Text to synthesize
        filename: Output file path (should end with .wav)
        provider: TTS provider to use (defaults to current tts_provider)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if provider is None:
        provider = tts_provider
    
    # Get synthesis function for the provider
    synthesis_funcs = {
        'kokoro': _synthesize_kokoro,
        'kittentts': _synthesize_kittentts,
        'openai': _synthesize_openai,
        'elevenlabs': _synthesize_elevenlabs,
        'aws': _synthesize_polly,  # AWS uses polly
        'polly': _synthesize_polly,
        'azure': _synthesize_azure,
        'gcloud': _synthesize_gcloud,
        'deepgram': _synthesize_deepgram
    }
    
    synthesis_func = synthesis_funcs.get(provider)
    if not synthesis_func:
        log_message("ERROR", f"Provider {provider} not supported for audio saving")
        return False
    
    try:
        # Generate audio bytes
        log_message("INFO", f"Generating TTS audio with {provider}: {text}")
        audio_bytes = synthesis_func(text)
        
        if not audio_bytes:
            log_message("ERROR", f"Failed to generate audio with {provider}")
            return False
        
        # Save to file
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        
        log_message("INFO", f"Saved TTS audio to: {filename}")
        return True
        
    except Exception as e:
        log_message("ERROR", f"Failed to save TTS audio: {e}")
        return False


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
        # On macOS, create a new session and process group so we can kill the entire group
        if sys.platform == 'darwin':
            # Use os.setsid() to create a new session (and process group)
            kwargs['preexec_fn'] = os.setsid
        
        process = subprocess.Popen(cmd, **kwargs)
        log_message("DEBUG", f"Started {engine} process with PID: {process.pid}")
        with playback_control.lock:
            playback_control.current_process = process
        
        # Wait for completion or interruption
        try:
            if engine in ["festival", "flite"] and "input" in kwargs:
                # For stdin-based engines, we can't easily poll, so just communicate
                stdout, stderr = process.communicate(input=kwargs["input"])
            else:
                # Poll process with timeout to check for interruptions
                while process.poll() is None:
                    # Check if we should stop
                    if shutdown_event.is_set() or playback_control.skip_current or playback_control.skip_all:
                        process.terminate()
                        try:
                            process.wait(timeout=0.1)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        return False
                    time.sleep(0.01)  # Small sleep to avoid busy waiting
            
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
    """Worker thread that reads from queue and speaks text."""
    global current_speech_item, last_speech_end_time
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

            # Tell ASR to ignore input while we're speaking to prevent feedback
            try:
                from . import asr
                if hasattr(asr, 'set_ignore_input'):
                    asr.set_ignore_input(True)
                    log_message("DEBUG", "Set ASR to ignore input during TTS")
            except Exception as e:
                log_message("DEBUG", f"Could not set ASR ignore flag: {e}")

            # Set current speech item with thread safety and record start time
            with _state_lock:
                speech_item.start_time = time.time()
                current_speech_item = speech_item
            
            # Include line number in log if available
            if speech_item.line_number is not None:
                log_message("INFO", f"Speaking via {engine} (line {speech_item.line_number}): '{speech_item.text}'")
            else:
                log_message("INFO", f"Speaking via {engine}: '{speech_item.text}'")

            speak_text(speech_item.text, engine)

            # Check if we should skip current or if shutdown was requested
            if playback_control.skip_current or shutdown_event.is_set():
                playback_control.reset_skip_flags()
                if shutdown_event.is_set():
                    break
            
            # Clear current item and track end time
            with _state_lock:
                current_speech_item = None
                last_speech_end_time = time.time()
            
            # Resume ASR input after speaking (but only for appropriate modes)
            try:
                from . import asr
                from .state import get_shared_state
                
                if hasattr(asr, 'set_ignore_input'):
                    # Small delay to ensure TTS audio has fully finished
                    time.sleep(0.5)
                    
                    # Get current ASR mode from shared state
                    shared_state = get_shared_state()
                    asr_mode = getattr(shared_state, 'asr_mode', 'auto-input')
                    
                    # Only resume ASR automatically for auto-input mode
                    # For tap-to-talk mode, ASR should only be active when key is pressed
                    if asr_mode in ['auto-input']:
                        asr.set_ignore_input(False)
                        log_message("DEBUG", "Resumed ASR after speaking")
                    else:
                        log_message("DEBUG", f"Not resuming ASR after speaking (mode: {asr_mode})")
            except Exception as e:
                log_message("DEBUG", f"Could not resume ASR: {e}")

        except queue.Empty:
            continue
        except Exception as e:
            log_message("ERROR", f"TTS worker error: {e}")


def queue_for_speech(text: str, line_number: Optional[int] = None, source: str = "output", exception_match: bool = False) -> str:
    """Queue text for TTS with debouncing and filtering."""
    global highest_spoken_line_number, last_queued_text, last_queue_time

    # Check shared state if available
    if SHARED_STATE_AVAILABLE:
        shared_state = get_shared_state()
        if not shared_state.tts_enabled:
            log_message("DEBUG", "TTS disabled in shared state, not queueing speech")
            return ""

    # Log the original text before any filtering
    log_message("INFO", f"queue_for_speech received: '{text}' (exception_match={exception_match})")

    speakable_text, written_text = extract_speakable_text(text)

    if not speakable_text or len(speakable_text) < MIN_SPEAK_LENGTH:
        log_message("INFO", f"Text too short: '{text}' -> '{speakable_text}'")
        return ""
    
    # Add sentence ending if needed
    speakable_text = add_sentence_ending(speakable_text)
    
    # Clean up any awkward punctuation sequences
    speakable_text = clean_punctuation_sequences(speakable_text)

    # Get current time for various time-based checks
    current_time = time.time()
    
    # Always check for exact duplicates of the last spoken text
    with _state_lock:
        if last_queued_text and speakable_text == last_queued_text:
            log_message("INFO", f"Skipping exact duplicate of last spoken text: '{speakable_text}'")
            return ""
    
    # Check if we're in tool use mode (between PreToolUse and PostToolUse hooks)
    in_tool_use = False
    if SHARED_STATE_AVAILABLE:
        shared_state = get_shared_state()
        in_tool_use = shared_state.get_in_tool_use()
    
    # Skip similarity checks if we're in tool use mode (prompting the user)
    if not in_tool_use:
        # Debounce rapidly changing text
        with _state_lock:
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
    else:
        log_message("INFO", f"Bypassing similarity checks for exception pattern match: '{speakable_text}'")

    # If filtered text is longer than 20 characters and auto-skip is enabled, clear the queue to jump to this message
    if auto_skip_tts_enabled and len(speakable_text) > 20:
        log_message("INFO", f"Long text detected ({len(speakable_text)} chars), clearing queue to jump to latest")

        # Check if something is currently playing and how long it's been playing
        is_currently_playing = playback_control.current_process is not None
        playing_long_enough = False
        
        if is_currently_playing and current_speech_item and current_speech_item.start_time:
            time_playing = time.time() - current_speech_item.start_time
            playing_long_enough = time_playing >= 1.0  # Minimum 1 second
            log_message("DEBUG", f"Current item has been playing for {time_playing:.2f} seconds")

        # Clear the queue
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except:
                break

        # Only skip current item if something is actually playing
        if is_currently_playing:
            playback_control.skip_current_item()

            # Add an interjection only if the current item has been playing long enough
            if playing_long_enough:
                interjection = random.choice(SKIP_INTERJECTIONS)
                # Prepend the interjection to the text for TTS only
                speakable_text = f"{interjection}, {speakable_text}"
                log_message("INFO", f"Added interjection '{interjection}' for smoother transition")
            else:
                log_message("DEBUG", "Skipping interjection - current item hasn't played long enough")

    with _state_lock:
        last_queued_text = speakable_text
        last_queue_time = current_time

        if line_number is not None and line_number <= highest_spoken_line_number:
            log_message("INFO",
                        f"Skipping line {line_number} (already spoken up to line {highest_spoken_line_number}): '{speakable_text[:50]}...'")
            return ""

        if line_number is not None and line_number > highest_spoken_line_number:
            highest_spoken_line_number = line_number
            log_message("DEBUG", f"Updated highest_spoken_line_number to {highest_spoken_line_number}")

    # Use cache lock for cache operations
    with _cache_lock:
        spoken_cache.append((speakable_text, current_time))

    with _state_lock:
        # Create speech item
        speech_item = SpeechItem(
            text=speakable_text,
            original_text=text,
            line_number=line_number,
            source=source
        )

        try:
            tts_queue.put_nowait(speech_item)
            # Thread-safe append to history
            speech_history.append(speech_item)
        except queue.Full:
            log_message("WARNING", "TTS queue full, skipping text")

    return written_text


def start_tts_worker(engine: str, auto_skip_tts: bool = False) -> threading.Thread:
    """Start TTS worker thread with given engine."""
    global tts_worker_thread, auto_skip_tts_enabled
    auto_skip_tts_enabled = auto_skip_tts
    log_message("INFO", f"TTS worker started with auto_skip_tts={auto_skip_tts}")
    # Clear the shutdown event in case it was set from a previous shutdown
    shutdown_event.clear()
    tts_worker_thread = threading.Thread(target=tts_worker, args=(engine,), daemon=True)
    tts_worker_thread.start()
    return tts_worker_thread


def reset_tts_cache():
    """Reset all TTS caches and queue."""
    global last_queued_text, last_queue_time, highest_spoken_line_number
    
    with _cache_lock:
        spoken_cache.clear()
        speech_history.clear()
    
    with _state_lock:
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


def clear_speech_queue():
    """Clear all pending items from TTS queue."""
    items_cleared = 0
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            items_cleared += 1
        except:
            break
    
    if items_cleared > 0:
        log_message("INFO", f"Cleared {items_cleared} items from TTS queue")


def wait_for_tts_to_finish(timeout: Optional[float] = None) -> bool:
    """Wait for TTS queue to finish, return True if completed."""
    start_time = time.time()
    
    # Wait for queue to empty
    while not tts_queue.empty():
        if timeout and (time.time() - start_time) > timeout:
            return False
        time.sleep(0.1)
    
    # Wait for current item to finish speaking
    while True:
        with _state_lock:
            if current_speech_item is None:
                break
        if timeout and (time.time() - start_time) > timeout:
            return False
        time.sleep(0.1)
    
    # Wait a bit more for audio buffer to play out
    time.sleep(SPEECH_BUFFER_TIME)
    
    log_message("INFO", "All TTS finished")
    return True


def shutdown_tts():
    """Shutdown TTS system gracefully."""
    global tts_worker_thread
    
    # Signal shutdown
    shutdown_event.set()
    
    # Send shutdown signal to queue
    try:
        tts_queue.put("__SHUTDOWN__", timeout=0.1)
    except queue.Full:
        pass
    
    # Wait for worker thread
    with _state_lock:
        thread = tts_worker_thread
    
    if thread and thread.is_alive():
        thread.join(timeout=0.5)


def stop_tts_immediately():
    """Stop all TTS immediately for interrupt handling."""
    log_message("INFO", "stop_tts_immediately called - killing all audio")
    # Signal shutdown first to stop the worker thread
    shutdown_event.set()
    
    # Kill any current TTS process
    playback_control.skip_all_items()
    
    # Clear the queue completely
    try:
        while True:
            tts_queue.get_nowait()
    except queue.Empty:
        pass
    
    # Force terminate any subprocess that might be running
    with playback_control.lock:
        if playback_control.current_process:
            try:
                # For macOS 'say' command, we might need to kill the entire process group
                pid = playback_control.current_process.pid
                log_message("INFO", f"Attempting to kill TTS process PID: {pid}")
                if sys.platform == 'darwin':  # macOS
                    # Get process group ID first before any termination attempts
                    # to avoid race condition where process dies between operations
                    pgid = None
                    try:
                        pgid = os.getpgid(pid)
                    except ProcessLookupError:
                        log_message("INFO", "Process already terminated")
                        playback_control.current_process = None
                    except Exception as e:
                        log_message("WARNING", f"Could not get process group: {e}")
                    
                    # Now try to kill the process group if we got the pgid
                    if pgid is not None:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                            log_message("INFO", f"Successfully killed process group {pgid}")
                        except ProcessLookupError:
                            log_message("INFO", "Process group already terminated")
                        except Exception as e:
                            log_message("ERROR", f"Failed to kill process group: {e}")
                            # Fallback to regular kill
                            try:
                                playback_control.current_process.kill()
                                log_message("INFO", "Used fallback kill method")
                            except:
                                pass
                    else:
                        # No pgid, try regular kill
                        try:
                            playback_control.current_process.kill()
                            log_message("INFO", "Killed process directly (no pgid)")
                        except:
                            pass
                else:
                    # For other platforms, use terminate then kill
                    playback_control.current_process.terminate()
                    try:
                        playback_control.current_process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        playback_control.current_process.kill()
            except Exception as e:
                log_message("ERROR", f"Error stopping TTS process: {e}")
            finally:
                playback_control.current_process = None


# Playback control functions
def pause_playback():
    """Pause TTS playback."""
    playback_control.pause()
    log_message("INFO", "Playback paused")


def resume_playback():
    """Resume TTS playback."""
    playback_control.resume()
    log_message("INFO", "Playback resumed")


def skip_current():
    """Skip currently playing TTS item."""
    playback_control.skip_current_item()
    log_message("INFO", "Skipping current item")


def skip_all():
    """Skip all remaining items in queue."""
    playback_control.skip_all_items()
    # Clear the queue
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except:
            break
    log_message("INFO", "Skipped all items")


def navigate_to_previous():
    """Navigate to previous item in speech history."""
    with _state_lock:
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
    """Skip to next item in TTS queue."""
    playback_control.skip_current_item()
    log_message("INFO", "Navigating to next item")
    return True


def get_current_speech_item() -> Optional[SpeechItem]:
    """Get currently speaking SpeechItem."""
    return current_speech_item


def get_speech_history() -> List[SpeechItem]:
    """Get copy of speech history list."""
    with _state_lock:
        return speech_history.copy()


def get_queue_size() -> int:
    """Get current TTS queue size."""
    return tts_queue.qsize()


def is_paused() -> bool:
    """Check if TTS playback is paused."""
    return playback_control.is_paused


def is_speaking() -> bool:
    """Check if TTS is speaking or recently finished."""
    # Check if actively speaking
    if current_speech_item is not None:
        return True
    
    # Check if TTS process is still running
    with playback_control.lock:
        if playback_control.current_process is not None:
            if playback_control.current_process.poll() is None:
                # Process is still running
                return True
    
    # Check if we recently finished speaking (audio might still be playing)
    time_since_speech = time.time() - last_speech_end_time
    return time_since_speech < SPEECH_BUFFER_TIME


def get_highest_spoken_line_number() -> int:
    """Get highest line number spoken so far."""
    return highest_spoken_line_number


def parse_arguments():
    """Parse TTS provider command-line arguments."""
    parser = argparse.ArgumentParser(description='Text-to-Speech with multiple provider support')
    parser.add_argument('--tts-provider', type=str, default=None,
                       choices=['system', 'openai', 'aws', 'polly', 'azure', 'gcloud', 'elevenlabs', 'deepgram', 'kittentts', 'kokoro'],
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
    """Configure TTS provider from command-line args."""
    global tts_provider, openai_voice, polly_voice, polly_region, azure_voice, azure_region, gcloud_voice, gcloud_language_code, elevenlabs_voice_id, kittentts_model, kittentts_voice, kokoro_language, kokoro_voice, kokoro_speed
    
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
    
    elif tts_provider == 'deepgram':
        if args.voice:
            deepgram_voice_model = args.voice
        log_message("INFO", f"Using Deepgram TTS with model: {deepgram_voice_model}")
    
    elif tts_provider == 'kittentts':
        # Additional KittenTTS validation
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
                from kittentts import KittenTTS
                import soundfile as sf
        except ImportError:
            print(f"Error: KittenTTS dependencies not installed")
            print(f"Please install with:")
            print(f"  pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl")
            print(f"  pip install soundfile")
            return False
        
        if args.voice:
            kittentts_voice = args.voice
        log_message("INFO", f"Using KittenTTS with model: {kittentts_model} and voice: {kittentts_voice}")
    
    elif tts_provider == 'kokoro':
        # Additional KokoroTTS validation
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
                from kokoro import KPipeline
                import soundfile as sf
        except ImportError:
            print(f"Error: KokoroTTS dependencies not installed")
            print(f"Please install with:")
            print(f"  pip install kokoro>=0.9.4 soundfile")
            return False
        
        kokoro_language = args.language if hasattr(args, 'language') else 'a'
        if args.voice:
            kokoro_voice = args.voice
        log_message("INFO", f"Using KokoroTTS with language: {kokoro_language} and voice: {kokoro_voice}")
    
    return True


def select_best_tts_provider() -> str:
    """Select best available TTS provider by preference order."""
    # Check shared state first
    state_provider = None
    if SHARED_STATE_AVAILABLE:
        try:
            state = get_shared_state()
            state_provider = state.tts_provider
        except:
            pass
    
    preferred = state_provider or os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER')
    accessible = check_tts_provider_accessibility(requested_provider=preferred)
    
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
    """Configure TTS provider from config dictionary."""
    global tts_provider, openai_voice, polly_voice, polly_region, azure_voice, azure_region, gcloud_voice, gcloud_language_code, elevenlabs_voice_id, elevenlabs_model_id, deepgram_voice_model, kittentts_model, kittentts_voice
    
    provider = config.get('provider', 'system')
    original_provider = provider
    tts_provider = provider
    
    # Validate provider configuration
    if not validate_provider_config(provider):
        log_message("WARNING", f"TTS provider {provider} validation failed")
        # Fall back to best available provider
        fallback_provider = select_best_tts_provider()
        if fallback_provider != provider:
            log_message("INFO", f"Falling back to TTS provider: {fallback_provider}")
            provider = fallback_provider
            tts_provider = provider
            config = dict(config)  # Make a copy
            config['provider'] = provider
            # Validate fallback provider
            if not validate_provider_config(provider):
                log_message("ERROR", f"Fallback TTS provider {fallback_provider} also failed")
                return False
        else:
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
    
    elif provider == 'kittentts':
        # Additional KittenTTS validation
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
                from kittentts import KittenTTS
                import soundfile as sf
        except ImportError:
            print(f"Error: KittenTTS dependencies not installed")
            print(f"Please install with:")
            print(f"  pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl")
            print(f"  pip install soundfile")
            return False
        
        if config.get('voice'):
            kittentts_voice = config['voice']
        if config.get('model'):
            kittentts_model = config['model']
        log_message("INFO", f"Using KittenTTS with model: {kittentts_model} and voice: {kittentts_voice}")
    
    # Update shared state with the actually working provider
    from .state import get_shared_state
    shared_state = get_shared_state()
    shared_state.set_tts_config(provider=provider)
    
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