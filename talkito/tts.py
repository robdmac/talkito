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

import argparse
import io
import queue
import os
import random
import re
import requests
import shutil
import signal
import soundfile as sf
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import deque
from typing import Optional, List, Tuple, Deque, Dict, Any
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

# Import centralized logging utilities
try:
    from .logs import log_message as _base_log_message
except ImportError:
    # Fallback for standalone execution
    def _base_log_message(level: str, message: str, logger_name: str = None):
        print(f"[{level}] {message}")

def patch_phonemizer_espeak_api():
    """
    Make official phonemizer behave like the fork for loaders that call
    EspeakWrapper.set_data_path / set_library.
    """
    import os
    from importlib import import_module

    # Import the wrapper class
    wrapper = import_module('phonemizer.backend.espeak.wrapper')
    EspeakWrapper = wrapper.EspeakWrapper

    # Provide missing methods as no-ops that set env vars + class attributes
    if not hasattr(EspeakWrapper, 'set_data_path'):
        def _set_data_path(path):
            if path:
                os.environ['ESPEAKNG_DATA_PATH'] = path
                os.environ.setdefault('ESPEAK_DATA_PATH', path)
                try:
                    # some builds read a class attribute
                    setattr(EspeakWrapper, 'data_path', path)
                except Exception:
                    pass
        EspeakWrapper.set_data_path = staticmethod(_set_data_path)

    if not hasattr(EspeakWrapper, 'set_library'):
        def _set_library(path):
            if path:
                os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = path
                try:
                    setattr(EspeakWrapper, 'library_path', path)
                except Exception:
                    pass
        EspeakWrapper.set_library = staticmethod(_set_library)

    # Optionally set them now using espeakng-loader (if present)
    try:
        import espeakng_loader
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception:
        # fall back to whatever env vars the user has set
        pass

# Call this BEFORE importing KittenTTS/Kokoro/etc.
patch_phonemizer_espeak_api()

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
kittentts_model = os.environ.get('KITTENTTS_MODEL', 'kitten-tts-nano-0.2')  # Default KittenTTS model
kittentts_voice = os.environ.get('KITTENTTS_VOICE', 'expr-voice-3-f')  # Default KittenTTS voice
kokoro_language = os.environ.get('KOKORO_LANGUAGE', 'a')  # Default Kokoro language (American English)
kokoro_voice = os.environ.get('KOKORO_VOICE', 'af_heart')  # Default Kokoro voice
kokoro_speed = os.environ.get('KOKORO_SPEED', '1.0')  # Default Kokoro speed

# Local model caching for offline TTS providers (kokoro/kittentts)
_local_model_cache = None
_local_model_provider = None  # Track which provider is cached ('kokoro' or 'kittentts')
_local_model_loading = False
_local_model_error = None
_local_model_cache_lock = threading.Lock()


def _create_model_instance(provider: str):
    """Create model instance for the specified provider."""
    if provider == 'kokoro':
        # Check if spaCy model is available
        from .models import check_spacy_model_consent
        if not check_spacy_model_consent('kokoro'):
            raise RuntimeError("spaCy language model required for kokoro but download was declined")
        
        with suppress_ai_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            from kokoro import KPipeline
        
        # Model creation - consent was already obtained in main thread
        repo_id = 'hexgrad/Kokoro-82M'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            return KPipeline(lang_code='en-us', repo_id=repo_id)
        
    elif provider == 'kittentts':
        with suppress_ai_warnings():
            from kittentts import KittenTTS
        
        # Model creation - consent was already obtained in main thread
        model_name = kittentts_model
        log_message("DEBUG", f"KittenTTS(model_name) with model_name = {model_name}")
        return KittenTTS(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _load_model_background(provider: str):
    """Load model in background thread."""
    global _local_model_cache, _local_model_provider, _local_model_loading, _local_model_error
    
    try:
        log_message("INFO", f"Background loading {provider} model...")
        model = _create_model_instance(provider)
        
        with _local_model_cache_lock:
            _local_model_cache = model
            _local_model_provider = provider
            _local_model_loading = False
            _local_model_error = None
            
        log_message("INFO", f"{provider} model loaded successfully in background")
        
    except Exception as e:
        with _local_model_cache_lock:
            _local_model_loading = False
            _local_model_error = f"{provider} model loading failed: {e}"
        log_message("ERROR", f"Background {provider} model loading failed: {e}")


def preload_local_model(provider: str):
    """Start background preloading of local model for specified provider."""
    global _local_model_loading
    
    if provider not in ['kokoro', 'kittentts']:
        log_message("WARNING", f"Unknown provider for preloading: {provider}")
        return
    
    with _local_model_cache_lock:
        # Skip if already loading or loaded the same provider
        if _local_model_loading:
            log_message("DEBUG", f"Model already loading, skipping preload for {provider}")
            return
        
        if _local_model_provider == provider and _local_model_cache is not None:
            log_message("DEBUG", f"Model already cached for {provider}, skipping preload")
            return
        
        # Check if model needs consent BEFORE starting background thread
        # This must happen in main thread where input() works
        from .models import check_model_cached, ask_user_consent
        
        model_download_started = False
        if provider == 'kokoro':
            model_name = 'default'  # Kokoro uses 'default' model name
            if not check_model_cached('kokoro', model_name):
                if not ask_user_consent('kokoro', model_name):
                    log_message("INFO", f"User declined download for {provider} model '{model_name}'")
                    # Fall back to next best available provider
                    fallback_provider = select_best_tts_provider(excluded_providers={'kokoro'})
                    print(f"Download declined. Falling back to {fallback_provider} TTS provider.")
                    
                    # Update shared state with fallback provider
                    try:
                        if SHARED_STATE_AVAILABLE:
                            from .state import get_shared_state
                            shared_state = get_shared_state()
                            shared_state.set_tts_config(provider=fallback_provider)
                            log_message("INFO", f"[TTS_PRELOAD] Updated shared state to use fallback provider: {fallback_provider}")
                    except Exception as e:
                        log_message("WARNING", f"[TTS_PRELOAD] Could not update shared state with fallback: {e}")
                    
                    return
                model_download_started = True
        elif provider == 'kittentts':
            model_name = kittentts_model  # Use current kittentts model setting  
            if not check_model_cached('kittentts', model_name):
                if not ask_user_consent('kittentts', model_name):
                    log_message("INFO", f"User declined download for {provider} model '{model_name}'")
                    # Fall back to next best available provider
                    fallback_provider = select_best_tts_provider(excluded_providers={'kittentts'})
                    print(f"Download declined. Falling back to {fallback_provider} TTS provider.")
                    
                    # Update shared state with fallback provider
                    try:
                        if SHARED_STATE_AVAILABLE:
                            from .state import get_shared_state
                            shared_state = get_shared_state()
                            shared_state.set_tts_config(provider=fallback_provider)
                            log_message("INFO", f"[TTS_PRELOAD] Updated shared state to use fallback provider: {fallback_provider}")
                    except Exception as e:
                        log_message("WARNING", f"[TTS_PRELOAD] Could not update shared state with fallback: {e}")
                    
                    return
                model_download_started = True
        
        # Start background loading (consent already obtained)
        _local_model_loading = True
        _local_model_error = None
    
    thread = threading.Thread(target=_load_model_background, args=(provider,), daemon=True)
    thread.start()
    log_message("INFO", f"Started background loading of {provider} model")
    
    # Show confirmation message if download was started
    if model_download_started:
        print(f"Downloading {provider} model in background. TTS will start automatically when ready.")


def get_cached_local_model(provider: str, timeout: float = 10.0):
    """Get cached local model, waiting for background loading if needed."""
    start_time = time.time()
    need_to_preload = False

    while True:
        with _local_model_cache_lock:
            # Check for errors
            if _local_model_error and not _local_model_loading:
                print(f"Error loading {provider} model: {_local_model_error}")
                return None
            
            # Check if we have the right model cached
            if _local_model_provider == provider and _local_model_cache is not None:
                return _local_model_cache
            
            # Check if not loading and not cached - this means no one started loading
            if not _local_model_loading:
                print(f"{provider} model not loaded and no background loading in progress. Starting now...")
                need_to_preload = True

        if need_to_preload:
            preload_local_model(provider)
        
        # Check timeout
        if time.time() - start_time > timeout:
            print(f"Timeout waiting for {provider} model to load after {timeout} seconds")
            return None
        
        time.sleep(0.1)


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


# Provider registry for metadata (install instructions, env vars, etc.)
TTS_PROVIDERS = {
    'openai': {
        'env_var': 'OPENAI_API_KEY',
        'voice_var': 'openai_voice',
        'display_name': 'OpenAI',
        'install': 'pip install openai',
        'config_keys': ['voice']
    },
    'aws': {
        'env_var': None,  # AWS uses multiple env vars or config files
        'env_name': 'AWS credentials',
        'voice_var': 'polly_voice',
        'region_var': 'polly_region',
        'display_name': 'AWS Polly',
        'install': 'pip install boto3',
        'config_keys': ['voice', 'region']
    },
    'azure': {
        'env_var': 'AZURE_SPEECH_KEY',
        'voice_var': 'azure_voice',
        'region_var': 'azure_region',
        'display_name': 'Microsoft Azure',
        'install': 'pip install azure-cognitiveservices-speech',
        'config_keys': ['voice', 'region']
    },
    'gcloud': {
        'env_var': 'GOOGLE_APPLICATION_CREDENTIALS',
        'voice_var': 'gcloud_voice',
        'language_var': 'gcloud_language_code',
        'display_name': 'Google Cloud',
        'install': 'pip install google-cloud-texttospeech',
        'config_keys': ['voice', 'language']
    },
    'elevenlabs': {
        'env_var': 'ELEVENLABS_API_KEY',
        'voice_var': 'elevenlabs_voice_id',
        'model_var': 'elevenlabs_model_id',
        'display_name': 'ElevenLabs',
        'install': 'pip install requests',
        'config_keys': ['voice', 'model']
    },
    'deepgram': {
        'env_var': 'DEEPGRAM_API_KEY',
        'model_var': 'deepgram_voice_model',
        'display_name': 'Deepgram',
        'install': 'pip install deepgram-sdk',
        'config_keys': ['model']
    },
    'kittentts': {
        'env_var': None,  # KittenTTS doesn't need an API key
        'model_var': 'kittentts_model',
        'voice_var': 'kittentts_voice',
        'display_name': 'KittenTTS',
        'install': 'pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl soundfile',
        'config_keys': ['model', 'voice']
    },
    'kokoro': {
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
                except Exception:
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
                except Exception:
                    pass
                    
    def skip_all_items(self):
        """Skip all remaining items in TTS queue."""
        with self.lock:
            self.skip_all = True
            if self.current_process:
                try:
                    self.current_process.terminate()
                except Exception:
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


@contextmanager
def suppress_ai_warnings():
    """Context manager to suppress common AI package warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="click")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        yield


# Wrapper for module-specific logging
def log_message(level: str, message: str):
    """Log message with module name using centralized logger."""
    _base_log_message(level, message, __name__)


def check_tts_provider_accessibility(requested_provider: str = None) -> Dict[str, Dict[str, Any]]:
    """Check TTS provider accessibility based on API keys and environment."""
    log_message("INFO", f"check_tts_provider_accessibility called with requested_provider={requested_provider}")
    
    accessible = {}
    
    # System TTS
    detected_engine = detect_tts_engine()
    log_message("INFO", f"System TTS detection completed - detected: {detected_engine}")
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
    
    # For accessibility check, assume kittentts is available if requested
    # Actual validation happens during model loading with user consent
    if requested_provider == 'kittentts':
        kittentts_available = True
        kittentts_note = "KittenTTS package (validation deferred to model loading)"
    else:
        # Only do expensive import check if this provider is NOT specifically requested
        try:
            # Just check if kittentts package is importable - actual model loading happens later
            with suppress_ai_warnings():
                import kittentts  # noqa: F401 # Just check package availability
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
    
    # For accessibility check, assume kokoro is available if requested
    # Actual validation happens during model loading with user consent
    if requested_provider == 'kokoro':
        kokoro_available = True
        kokoro_note = "KokoroTTS package (validation deferred to model loading)"
        log_message("INFO", "KokoroTTS availability assumed for requested provider")
    else:
        # Only do expensive import check if this provider is NOT specifically requested
        try:
            # This import is still expensive but we skip it for the common case
            with suppress_ai_warnings():
                import kokoro  # noqa: F401 # Just check package availability
            kokoro_available = True
        except Exception as e:
            # Catch all exceptions, not just ImportError (e.g., AttributeError from EspeakWrapper)
            kokoro_note = f"Kokoro package error: {str(e)}" if not isinstance(e, ImportError) else "Requires KokoroTTS package (pip install kokoro>=0.9.4 soundfile)"
        log_message("INFO", f"KokoroTTS availability check completed - available: {kokoro_available}")
    
    accessible["kokoro"] = {
        "available": kokoro_available,
        "note": kokoro_note
    }

    log_message("INFO", f"check_tts_provider_accessibility completed - providers checked: {list(accessible.keys())}")
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
    """Play audio file using an available system player.
    Prefers players that support the file's format (wav/mp3), with robust fallbacks.
    """
    path = Path(audio_path)
    if not path.exists():
        log_message("ERROR", f"Audio file not found: {audio_path}")
        return False

    ext = path.suffix.lower()
    is_wav = ext == ".wav"
    is_mp3 = ext == ".mp3"

    # Build candidate player commands in priority order for each format
    candidates = []

    if is_wav:
        # macOS / Linux players that handle WAV well
        # afplay (macOS), sox 'play', ffplay, PulseAudio/ALSA, VLC CLI
        candidates.extend([
            (["afplay", audio_path], "afplay"),
            (["play", "-q", audio_path], "play"),
            (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path], "ffplay"),
            (["paplay", audio_path], "paplay"),
            (["aplay", audio_path], "aplay"),
            (["cvlc", "--play-and-exit", "--intf", "dummy", audio_path], "cvlc"),
        ])

        # Windows (WAV only): PowerShell SoundPlayer
        if sys.platform.startswith("win"):
            candidates.append((
                ["powershell", "-NoProfile", "-Command",
                 f'[Console]::OutputEncoding=[Text.UTF8]; '
                 f'$p=New-Object System.Media.SoundPlayer "{audio_path}"; '
                 f'$p.PlaySync();'],
                "powershell-wav"
            ))

    elif is_mp3:
        # MP3: prefer afplay (macOS), mpg123 (fast), sox 'play', ffplay, vlc CLI
        candidates.extend([
            (["afplay", audio_path], "afplay"),
            (["mpg123", "-q", audio_path], "mpg123"),
            (["play", "-q", audio_path], "play"),
            (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path], "ffplay"),
            (["cvlc", "--play-and-exit", "--intf", "dummy", audio_path], "cvlc"),
        ])
    else:
        # Unknown extension: try broadly capable players
        candidates.extend([
            (["afplay", audio_path], "afplay"),
            (["play", "-q", audio_path], "play"),
            (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path], "ffplay"),
            (["mpg123", "-q", audio_path], "mpg123"),
            (["paplay", audio_path], "paplay"),
            (["aplay", audio_path], "aplay"),
            (["cvlc", "--play-and-exit", "--intf", "dummy", audio_path], "cvlc"),
        ])

    # Pick the first available player from the candidate list
    chosen_cmd = None
    for cmd, binary in candidates:
        # Special case: "powershell-wav" isn't a binary to which/which
        if binary == "powershell-wav":
            chosen_cmd = cmd
            break
        if shutil.which(binary):
            chosen_cmd = cmd
            break

    if not chosen_cmd:
        # Helpful, format-aware error
        need = "a WAV-capable player (afplay, play/sox, ffplay, paplay/aplay, or VLC)"
        if is_mp3:
            need = "an MP3-capable player (afplay, mpg123, play/sox, ffplay, or VLC)"
        log_message("ERROR", f"No suitable audio player found for {ext or 'unknown format'}. Install {need}.")
        return False

    # Launch player
    try:
        process = subprocess.Popen(
            chosen_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # ensures we can terminate the whole group
        )
    except Exception as e:
        log_message("ERROR", f"Failed to start audio player {chosen_cmd[0]}: {e}")
        return False

    if use_process_control:
        with playback_control.lock:
            playback_control.current_process = process

    try:
        # Poll + allow cooperative interruption
        while process.poll() is None:
            if shutdown_event.is_set() or (use_process_control and (playback_control.skip_current or playback_control.skip_all)):
                try:
                    process.terminate()
                    process.wait(timeout=0.25)
                except Exception:
                    process.kill()
                return False
            time.sleep(0.01)
        return process.returncode == 0
    except Exception:
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

def _write_temp_audio(audio_bytes: bytes, ext: str) -> str:
    path = _create_temp_audio_file(ext)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path

def synthesize_and_play(synthesize_func, text: str, use_process_control: bool = True) -> bool:
    """Synthesize audio via provider function and play it."""
    try:
        result = synthesize_func(text)
        if not result or not isinstance(result, tuple) or len(result) != 2:
            log_message("ERROR", f"Synthesizer returned unexpected result: {result!r}")
            return False
        audio_bytes, ext = result
        if not audio_bytes:
            log_message("ERROR", "No audio returned from synthesizer")
            return False
        tmp_path = _write_temp_audio(audio_bytes, ext)
        try:
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
    
    # Skip validation for local providers - they'll be validated during actual model loading
    # This avoids expensive import operations during startup
    if provider in ['kittentts', 'kokoro']:
        log_message("DEBUG", f"Skipping validation for local provider {provider} - will validate during model loading")
        return True
    
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
    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        """Synthesize speech from text, return (audio bytes, format) or None."""
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


class OpenAIProvider(TTSProvider):
    """OpenAI TTS provider implementation."""
    
    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        import openai
        try:
            voice = self.get_config_value('voice', openai_voice)
            response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
            audio_data = io.BytesIO()
            for chunk in response.iter_bytes():
                audio_data.write(chunk)
            return audio_data.getvalue(), ".mp3"
        except Exception as e:
            log_message("ERROR", f"OpenAI TTS synthesis error: {e}")
            return None


class AWSPollyProvider(TTSProvider):
    """AWS Polly TTS provider implementation."""
    
    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        import boto3

        try:
            voice = self.get_config_value('voice', polly_voice)
            region = self.get_config_value('region', polly_region)

            polly_client = boto3.client('polly', region_name=region)

            response = polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice,
                Engine='neural'
            )

            return response['AudioStream'].read(), ".mp3"
        except Exception as e:
            log_message("ERROR", f"AWS Polly TTS synthesis error: {e}")
            return None

class AzureProvider(TTSProvider):
    """Azure TTS provider implementation."""
    
    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        import azure.cognitiveservices.speech as speechsdk
        
        voice = self.get_config_value('voice', azure_voice)
        region = self.get_config_value('region', azure_region)
        
        speech_key = os.environ.get('AZURE_SPEECH_KEY')
        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable not set")
        
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
        speech_config.speech_synthesis_voice_name = voice
        
        # Set output format to mp3
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        
        # Create synthesizer without audio output (we'll get the audio data directly)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data, ".mp3"
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_msg = f"Azure TTS synthesis canceled: {cancellation_details.reason}"
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                error_msg += f" - {cancellation_details.error_details}"
            raise Exception(error_msg)
        
        return None


class GoogleCloudProvider(TTSProvider):
    """Google Cloud TTS provider implementation."""
    
    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        from google.cloud import texttospeech

        try:
            voice = self.get_config_value('voice', gcloud_voice)
            language_code = self.get_config_value('language_code', gcloud_language_code)

            client = texttospeech.TextToSpeechClient()

            synthesis_input = texttospeech.SynthesisInput(text=text)

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )

            return response.audio_content, ".mp3"
        except Exception as e:
            log_message("ERROR", f"Google TTS synthesis error: {e}")
            return None


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS provider implementation."""
    
    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        voice_id = self.get_config_value('voice_id', elevenlabs_voice_id)
        api_key = os.environ.get('ELEVENLABS_API_KEY')
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
            'xi-api-key': api_key
        }
        
        data = {
            'text': text,
            'model_id': 'eleven_monolingual_v1',
            'voice_settings': {
                'stability': 0.5,
                'similarity_boost': 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content, ".mp3"
        
        return None


class DeepgramProvider(TTSProvider):
    """Deepgram TTS provider implementation."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        model = self.get_config_value('model', deepgram_voice_model)  # <-- use 'model'
        api_key = os.environ.get('DEEPGRAM_API_KEY')
        if not api_key or not model:
            log_message("ERROR", "Deepgram missing API key or model")
            return None

        url = f"https://api.deepgram.com/v1/speak?model={model}"
        headers = {
            'Authorization': f'Token {api_key}',
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
        }
        response = requests.post(url, json={'text': text}, headers=headers)
        if response.ok:
            return response.content, ".mp3"
        log_message("ERROR", f"Deepgram error {response.status_code}: {response.text}")
        return None

class KittenTTSProvider(TTSProvider):
    """KittenTTS provider implementation."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            m = get_cached_local_model('kittentts', timeout=10.0)
            if m is None:
                raise RuntimeError("KittenTTS model unavailable")
            audio = m.generate(text, voice=self.get_config_value('voice', kittentts_voice))
            buf = io.BytesIO()
            sf.write(buf, audio, 24000, format='WAV')
            return buf.getvalue(), ".wav"
        except Exception as e:
            log_message("ERROR", f"KittenTTS synthesis error: {e}")
            return None

class KokoroTTSProvider(TTSProvider):
    """KokoroTTS provider implementation."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            # Get configuration from shared state
            config = get_tts_config()
            voice = config.get('voice') or kokoro_voice
            speed = float(config.get('speed') or kokoro_speed)
            pipeline = get_cached_local_model('kokoro', timeout=10.0)

            if pipeline is None:
                log_message("ERROR", "Failed to get cached Kokoro model")
                return None

            log_message("DEBUG", "Using cached KokoroTTS model")

            # Generate audio with the specified voice and speed
            # Kokoro returns a generator, we need to process all chunks
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=speed)):
                audio_chunks.append(audio)

            # Concatenate all audio chunks
            import numpy as np
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                buf = io.BytesIO()
                sf.write(buf, full_audio, 24000, format='WAV')
                return buf.getvalue(), ".wav"
            else:
                log_message("ERROR", "KokoroTTS generated no audio")

        except Exception as e:
            log_message("ERROR", f"KokoroTTS synthesis error: {e}")
            return None

# Provider class registry
PROVIDER_CLASSES = {
    'openai': OpenAIProvider,
    'aws': AWSPollyProvider,
    'polly': AWSPollyProvider,  # Backward compatibility
    'azure': AzureProvider,
    'gcloud': GoogleCloudProvider,
    'elevenlabs': ElevenLabsProvider,
    'deepgram': DeepgramProvider,
    'kittentts': KittenTTSProvider,
    'kokoro': KokoroTTSProvider,
}


def create_tts_provider(provider_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[TTSProvider]:
    """Factory function to create TTS provider instances."""
    provider_class = PROVIDER_CLASSES.get(provider_name.lower())
    if provider_class:
        return provider_class(config)
    return None


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

    # Use provider classes directly
    provider = create_tts_provider(tts_provider)
    if provider:
        try:
            return provider.speak(text)
        except Exception as e:
            log_message("ERROR", f"TTS provider {tts_provider} failed: {e}")
            return False

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
    
    # Use provider factory to get the provider instance
    tts_provider_instance = create_tts_provider(provider)
    if not tts_provider_instance:
        log_message("ERROR", f"Provider {provider} not supported for audio saving")
        return False
    
    try:
        # Generate audio bytes
        log_message("INFO", f"Generating TTS audio with {provider}: {text}")
        audio_bytes, ext = tts_provider_instance.synthesize(text)
        
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
        except Exception:
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
                    except queue.Empty:
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
        is_currently_speaking = is_speaking()
        playing_long_enough = False
        
        log_message("DEBUG", f"Interjection check: is_currently_playing={is_currently_playing}, is_actually_speaking={is_currently_speaking}, current_speech_item={current_speech_item is not None}")
        
        if is_currently_playing and current_speech_item and current_speech_item.start_time:
            time_playing = time.time() - current_speech_item.start_time
            playing_long_enough = time_playing >= 1.0  # Minimum 1 second
            log_message("DEBUG", f"Current item has been playing for {time_playing:.2f} seconds")

        # Clear the queue
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except queue.Empty:
                break

        # Only skip current item if something is actually speaking (use is_speaking() for accuracy)
        if is_currently_speaking:
            playback_control.skip_current_item()
            log_message("DEBUG", "Skipped current item - was actually speaking")

            # Add an interjection only if the current item has been playing long enough
            if playing_long_enough:
                interjection = random.choice(SKIP_INTERJECTIONS)
                # Prepend the interjection to the text for TTS only
                speakable_text = f"{interjection}, {speakable_text}"
                log_message("INFO", f"Added interjection '{interjection}' for smoother transition")
            else:
                log_message("DEBUG", "Skipping interjection - current item hasn't played long enough")
        elif is_currently_playing:
            log_message("WARNING", "Process exists but not actually speaking - race condition detected!")
        else:
            log_message("DEBUG", "Nothing currently playing - no skip needed")

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
        except queue.Empty:
            break
    
    log_message("INFO", "TTS cache reset")


def clear_speech_queue():
    """Clear all pending items from TTS queue."""
    items_cleared = 0
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            items_cleared += 1
        except queue.Empty:
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
                            except Exception:
                                pass
                    else:
                        # No pgid, try regular kill
                        try:
                            playback_control.current_process.kill()
                            log_message("INFO", "Killed process directly (no pgid)")
                        except Exception:
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
        except queue.Empty:
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
    global tts_provider, openai_voice, polly_voice, polly_region, azure_voice, azure_region, gcloud_voice, gcloud_language_code, elevenlabs_voice_id, deepgram_voice_model, kittentts_model, kittentts_voice, kokoro_language, kokoro_voice, kokoro_speed
    
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
            import azure.cognitiveservices.speech as speechsdk  # noqa: F401
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
            texttospeech.TextToSpeechClient()  # Test instantiation to verify availability
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
            with suppress_ai_warnings():
                from kittentts import KittenTTS  # noqa: F401
        except ImportError:
            print("Error: KittenTTS dependencies not installed")
            print("Please install with:")
            print("  pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl")
            print("  pip install soundfile")
            return False
        
        if args.voice:
            kittentts_voice = args.voice
        log_message("INFO", f"Using KittenTTS with model: {kittentts_model} and voice: {kittentts_voice}")
    
    elif tts_provider == 'kokoro':
        # Skip expensive kokoro import - will validate during actual model loading
        log_message("DEBUG", "Skipping kokoro validation in configure_tts_provider - will validate during model loading")
        
        kokoro_language = args.language if hasattr(args, 'language') else 'a'
        if args.voice:
            kokoro_voice = args.voice
        log_message("INFO", f"Using KokoroTTS with language: {kokoro_language} and voice: {kokoro_voice}")
    
    return True


def select_best_tts_provider(excluded_providers=None) -> str:
    """Select best available TTS provider by preference order with thorough validation."""
    log_message("INFO", "select_best_tts_provider called")
    excluded_providers = excluded_providers or set()
    
    # Check shared state first
    state_provider = None
    if SHARED_STATE_AVAILABLE:
        try:
            state = get_shared_state()
            state_provider = state.tts_provider
        except Exception:
            pass

    preferred = state_provider or os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER')

    accessible = check_tts_provider_accessibility(requested_provider=preferred)

    # Check if preferred provider is accessible, properly configured, and not excluded
    if preferred and preferred in accessible and accessible[preferred]['available'] and preferred not in excluded_providers:
        if validate_provider_config(preferred):
            log_message("INFO", f"Using preferred TTS provider: {preferred}")
            return preferred
        else:
            log_message("WARNING", f"Preferred TTS provider {preferred} failed validation, searching for alternatives")
    
    # Get all accessible providers except system and excluded providers
    available_providers = [
        provider for provider, info in sorted(accessible.items())
        if info['available'] and provider != 'system' and provider not in excluded_providers
    ]
    
    # Validate each provider thoroughly before selecting
    for provider in available_providers:
        if validate_provider_config(provider):
            log_message("INFO", f"Selected TTS provider: {provider} (first validated)")
            return provider
        else:
            log_message("WARNING", f"TTS provider {provider} failed validation, trying next")
    
    # Fall back to system if available and not excluded
    if 'system' not in excluded_providers and accessible.get('system', {}).get('available'):
        if validate_provider_config('system'):
            log_message("INFO", "Falling back to system TTS provider")
            return 'system'
        else:
            log_message("WARNING", "System TTS provider failed validation")
    
    # Last resort: return system anyway (it should always work)
    log_message("WARNING", "No fully validated TTS providers available, defaulting to system")
    return 'system'


def configure_tts_from_dict(config: dict) -> bool:
    """Configure TTS provider from config dictionary."""
    global tts_provider, openai_voice, polly_voice, polly_region, azure_voice, azure_region, gcloud_voice, gcloud_language_code, elevenlabs_voice_id, elevenlabs_model_id, deepgram_voice_model, kittentts_model, kittentts_voice
    
    provider = config.get('provider', 'system')
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
            import azure.cognitiveservices.speech as speechsdk  # noqa: F401
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
            from google.cloud import texttospeech  # noqa: F401
            texttospeech.TextToSpeechClient()  # Test instantiation to verify availability
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
            with suppress_ai_warnings():
                from kittentts import KittenTTS  # noqa: F401
        except ImportError:
            print("Error: KittenTTS dependencies not installed")
            print("Please install with:")
            print("  pip install kittentts")
            print("  pip install soundfile")
            return False
        
        if config.get('voice'):
            kittentts_voice = config['voice']
        if config.get('model'):
            kittentts_model = config['model']
        log_message("INFO", f"Using KittenTTS with model: {kittentts_model} and voice: {kittentts_voice}")
        
        # Background preloading started earlier in initialization
        
    elif provider == 'kokoro':
        # Skip expensive kokoro import - will validate during actual model loading
        log_message("DEBUG", "Skipping kokoro validation in configure_tts_from_dict - will validate during model loading")
        
        # Update global config variables
        global kokoro_voice, kokoro_language, kokoro_speed
        if config.get('voice'):
            kokoro_voice = config['voice']
        if config.get('language'):
            kokoro_language = config['language']
        if config.get('speed'):
            kokoro_speed = str(config['speed'])
        log_message("INFO", f"Using KokoroTTS with language: {kokoro_language} and voice: {kokoro_voice}")
        
        # Background preloading started earlier in initialization
    
    # Update shared state with the actually working provider
    from .state import get_shared_state
    shared_state = get_shared_state()
    shared_state.set_tts_config(provider=provider)
    
    return True


# Main entry point for testing
if __name__ == "__main__":

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