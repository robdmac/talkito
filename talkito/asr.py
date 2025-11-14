#!/usr/bin/env python3

# Talkito - Universal TTS wrapper that works with any command
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

"""Automatic Speech Recognition module demonstrating a more DRY and maintainable approach to supporting multiple ASR providers."""

# ruff: noqa: E402

# Suppress pkg_resources deprecation warnings from Google Cloud SDK dependencies
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.rpc')

# Suppress absl and gRPC warnings
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging if used

import atexit
import threading
import queue
import argparse
import numpy as np
import platform
import math
import re
import time
import tempfile
from typing import Optional, Callable, Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
from contextlib import contextmanager


# Keep global references to prevent garbage collection of callbacks
_alsa_error_handler = None
_jack_error_handler = None
_jack_info_handler = None

# Suppress ALSA/JACK warnings by installing custom error handlers
def _suppress_alsa_jack_warnings():
    """Suppress ALSA and JACK library warnings by monkey-patching error handlers."""
    global _alsa_error_handler, _jack_error_handler, _jack_info_handler

    try:
        import ctypes

        # === ALSA Suppression ===
        # Define the error handler function type for ALSA
        ALSA_ERROR_HANDLER = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,  # filename
            ctypes.c_int,      # line
            ctypes.c_char_p,  # function
            ctypes.c_int,      # err
            ctypes.c_char_p   # fmt
        )

        # Create a no-op error handler for ALSA
        def alsa_error_handler(filename, line, function, err, fmt):
            pass  # Silently ignore all ALSA errors

        # Store as global to prevent garbage collection (critical!)
        _alsa_error_handler = ALSA_ERROR_HANDLER(alsa_error_handler)

        # Try to load ALSA library and set custom error handler
        try:
            asound = ctypes.cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(_alsa_error_handler)
        except (OSError, AttributeError):
            # ALSA library not found or function not available - that's fine
            pass

        # === JACK Suppression ===
        # Define the error/info handler function type for JACK
        JACK_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_char_p)

        # Create no-op handlers for JACK
        def jack_error_handler(msg):
            pass  # Silently ignore JACK errors

        def jack_info_handler(msg):
            pass  # Silently ignore JACK info

        # Store as globals to prevent garbage collection
        _jack_error_handler = JACK_HANDLER(jack_error_handler)
        _jack_info_handler = JACK_HANDLER(jack_info_handler)

        # Try to load JACK library and set custom handlers
        try:
            jack = ctypes.cdll.LoadLibrary('libjack.so.0')
            jack.jack_set_error_function(_jack_error_handler)
            jack.jack_set_info_function(_jack_info_handler)
        except (OSError, AttributeError):
            # JACK library not found or functions not available - that's fine
            pass

    except Exception:
        # If anything fails, just continue without suppression
        pass

# Install ALSA and JACK error handlers before importing audio libraries
_suppress_alsa_jack_warnings()

# Import audio libraries
import speech_recognition as sr
import pyaudio

from .logs import log_message as _base_log_message
from .state import load_dotenv

load_dotenv()
load_dotenv('.talkito.env')

SQUARE_BRACKETS_CLEANER = re.compile(r'\[.*?\]')

# Wrapper to add [ASR] prefix to all log messages
def log_message(level: str, message: str):
    """Log a message with [ASR] prefix"""
    _base_log_message(level, f"[ASR] {message}", __name__)


@contextmanager
def temp_audio_file(audio_data: bytes, suffix: str = '.wav'):
    """Context manager for temporary audio file handling"""
    tmp_file = None
    tmp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp_file.name  # Get the path immediately after creation
        try:
            tmp_file.write(audio_data)
            tmp_file.flush()  # Ensure data is written
        finally:
            # Always close the file, even if write fails
            if tmp_file:
                try:
                    tmp_file.close()
                except Exception as e:
                    log_message("WARNING", f"Failed to close temp file: {e}")
        yield tmp_path
    finally:
        # Clean up the file if it was created
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                log_message("WARNING", f"Failed to delete temp file {tmp_path}: {e}")


# Unified VAD configuration
# Default minimum silence duration before ending utterance (in milliseconds)
DEFAULT_MIN_SILENCE_MS = int(os.environ.get('TALKITO_ASR_MIN_SILENCE_MS', '1500'))

@dataclass
class ASRConfig:
    """Configuration for an ASR provider"""
    provider: str
    language: str = os.environ.get('ASR_LANGUAGE', 'en-US')
    region: Optional[str] = None
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_key: Optional[str] = None
    user_id: Optional[str] = None
    model: Optional[str] = None


# Consolidated credential checking
CREDENTIAL_ENV_VARS = {
    'gcloud': 'GOOGLE_APPLICATION_CREDENTIALS',
    'assemblyai': 'ASSEMBLYAI_API_KEY',
    'deepgram': 'DEEPGRAM_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'azure': 'AZURE_SPEECH_KEY',
    'whisper': 'OPENAI_API_KEY',
}


def check_api_credentials(provider: str, config: ASRConfig) -> Tuple[bool, Optional[str]]:
    """Unified credential checking for all providers"""
    # Providers that don't need credentials
    if provider in ['google_free', 'local_whisper']:
        return True, None
    
    # Special case: Whisper local mode
    if provider == 'whisper':
        api_key = config.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            # Local mode is OK without API key
            return True, None
        return True, None
    
    # Special case: Houndify needs two credentials
    if provider == 'houndify':
        client_id = config.client_id or os.environ.get('HOUNDIFY_CLIENT_ID')
        client_key = config.client_key or os.environ.get('HOUNDIFY_CLIENT_KEY')
        if not client_id or not client_key:
            return False, "Houndify credentials not provided. Set HOUNDIFY_CLIENT_ID and HOUNDIFY_CLIENT_KEY."
        return True, None
    
    # Special case: AWS uses boto3's credential chain
    if provider == 'aws':
        try:
            import boto3
            # Try to create a client to verify credentials
            client = boto3.client('transcribe', region_name=config.region or 'us-east-1')
            client.list_transcription_jobs(MaxResults=1)
            return True, None
        except ImportError:
            return True, None  # Will be caught in import check
        except Exception:
            return False, "AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
    
    # Special case: Google Cloud
    if provider == 'gcloud':
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            return False, "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
        return True, None
    
    # Special case: Azure needs key and region
    if provider == 'azure':
        key = config.api_key or os.environ.get('AZURE_SPEECH_KEY')
        region = config.region or os.environ.get('AZURE_SPEECH_REGION')
        if not key:
            return False, "Azure Speech key not provided. Set AZURE_SPEECH_KEY environment variable."
        if not region:
            return False, "Azure Speech region not provided. Set AZURE_SPEECH_REGION environment variable."
        return True, None
    
    # Standard API key check for remaining providers
    env_var = CREDENTIAL_ENV_VARS.get(provider)
    if env_var and env_var != 'GOOGLE_APPLICATION_CREDENTIALS':
        key = config.api_key or os.environ.get(env_var)
        if not key:
            return False, f"{provider} API key not provided. Set {env_var} environment variable."
    
    return True, None


# Consolidated import checking
REQUIRED_IMPORTS = {
    'gcloud': ('google.cloud.speech', 'pip install google-cloud-speech'),
    'assemblyai': ('assemblyai', 'pip install assemblyai'),
    'houndify': ('houndify', 'pip install houndify'),
    'aws': ('boto3', 'pip install boto3 amazon-transcribe'),
    'azure': ('azure.cognitiveservices.speech', 'pip install azure-cognitiveservices-speech'),
    'deepgram': ('deepgram', 'pip install deepgram-sdk'),
    'whisper': ('whisper', 'pip install openai-whisper'),
    'local_whisper': ('faster_whisper', 'pip install faster-whisper'),
}


# Cache for provider validation results to avoid repeated prompts
_provider_validation_cache = {}

def clear_provider_cache():
    """Clear the provider validation cache"""
    global _provider_validation_cache
    _provider_validation_cache = {}

def check_provider_imports(provider: str, requested_provider: str = None) -> Tuple[bool, Optional[str]]:
    """Unified import checking for all providers"""
    
    # Check cache to avoid repeated prompts for the same provider
    # For offline providers like local_whisper, cache by provider name only to avoid repeated download prompts
    cache_key = provider if provider in ['local_whisper'] else f"{provider}_{requested_provider or 'none'}"
    if cache_key in _provider_validation_cache:
        result = _provider_validation_cache[cache_key]
        return result
    
    def cache_and_return(success: bool, message: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Cache the result and return it"""
        result = (success, message)
        _provider_validation_cache[cache_key] = result
        return result
    # Providers that use built-in speech_recognition
    if provider in ['google_free']:
        return cache_and_return(True, None)
    
    # Special case: local_whisper offline provider (try pywhispercpp first on Apple Silicon, then faster-whisper)
    if provider == 'local_whisper':
        # Try pywhispercpp first on Apple Silicon
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            try:
                from pywhispercpp.model import Model as PyWhisperModel  # noqa: F401
                return cache_and_return(True, "PyWhisperCpp CoreML available")
            except ImportError:
                pass  # Fall through to faster-whisper
        
        # Fall back to faster-whisper
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            # Check if the model is cached
            from .models import check_model_cached, with_download_progress
            # Use the actual model that will be used (check environment variable)
            model_name = os.environ.get('WHISPER_MODEL', 'small')
            if not check_model_cached('local_whisper', model_name):
                # Only prompt for download if this provider was specifically requested
                if (requested_provider == 'local_whisper' or provider == requested_provider or 
                    os.environ.get('TALKITO_PREFERRED_ASR_PROVIDER') == 'local_whisper'):
                    # Ask user for consent and download if approved
                    try:
                        def create_model():
                            # Use int8 for better compatibility, especially on macOS and CPU-only environments
                            default_compute_type = 'int8' if platform.system() == 'Darwin' else 'float16'
                            compute_type = os.environ.get('WHISPER_COMPUTE_TYPE', default_compute_type)
                            return WhisperModel(model_name, device='cpu', compute_type=compute_type)
                        
                        decorated_func = with_download_progress('local_whisper', model_name, create_model)
                        decorated_func()  # This will ask for consent and download
                        return cache_and_return(True, None)
                    except RuntimeError as e:
                        if "Download cancelled" in str(e):
                            return cache_and_return(False, f"faster-whisper model '{model_name}' download declined by user")
                        return cache_and_return(False, f"faster-whisper model download failed: {e}")
                    except Exception as e:
                        return cache_and_return(False, f"faster-whisper model download failed: {e}")
                else:
                    # Model not cached and not specifically requested - mark as unavailable
                    return cache_and_return(False, f"faster-whisper model '{model_name}' not cached. Will be downloaded on first use.")
            return cache_and_return(True, None)
        except ImportError:
            # Neither pywhispercpp nor faster-whisper available
            if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                return cache_and_return(False, "Local Whisper requires pywhispercpp (or faster-whisper which would be CPU limited): WHISPER_COREML=1 pip install pywhispercpp")
            else:
                return cache_and_return(False, "faster-whisper library not installed. Run: pip install faster-whisper")
        except Exception as e:
            return cache_and_return(False, f"faster-whisper library error: {e}")
    
    # Special case: Whisper can use either local or API mode
    if provider == 'whisper':
        # Try local whisper first
        try:
            import whisper
            whisper.available_models()
            return cache_and_return(True, None)
        except ImportError:
            # Try OpenAI API
            try:
                import openai  # noqa: F401
                return cache_and_return(True, None)
            except ImportError:
                return cache_and_return(False, "Neither local Whisper nor OpenAI library installed. Run: pip install openai-whisper or pip install openai")
        except Exception as e:
            # Whisper is installed but has issues
            return cache_and_return(False, f"Whisper library error: {e}")
    
    # Special case: Google Cloud - verify client creation works
    if provider == 'gcloud':
        try:
            from google.cloud import speech  # noqa: F401
            speech.SpeechClient()  # Test instantiation to verify availability
            return cache_and_return(True, None)
        except ImportError:
            return cache_and_return(False, "Google Cloud Speech library not installed. Run: pip install google-cloud-speech")
        except Exception as e:
            return cache_and_return(False, f"Google Cloud credentials invalid: {e}")
    
    # Standard import check for remaining providers
    module_info = REQUIRED_IMPORTS.get(provider)
    if not module_info:
        return cache_and_return(True, None)
    
    module_name, install_cmd = module_info
    try:
        importlib.import_module(module_name.split('.')[0])
        return cache_and_return(True, None)
    except ImportError:
        return cache_and_return(False, f"{module_name} library not installed. Run: {install_cmd}")


class AudioCaptureThread:
    """Reusable audio capture thread for all providers"""
    
    def __init__(self, microphone, queue_obj, is_active_check, stop_events, provider_name=""):
        self.microphone = microphone
        self.queue = queue_obj
        self.is_active_check = is_active_check
        self.stop_events = stop_events if isinstance(stop_events, list) else [stop_events]
        self.provider_name = provider_name
        self.thread = None
        self.local_stop = threading.Event()
    
    def start(self):
        """Start the audio capture thread"""
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self
    
    def stop(self, timeout=2.0):
        """Stop the audio capture thread"""
        self.local_stop.set()
        self.queue.put(None)  # Poison pill
        if self.thread:
            self.thread.join(timeout=timeout)
        log_message("DEBUG", f"Audio capture thread stopped for {self.provider_name}")
    
    def _capture_loop(self):
        """Main audio capture loop"""
        with self.microphone as source:
            log_message("INFO", f"Started audio capture for {self.provider_name}")
            
            # Add small delay for AWS to ensure proper initialization
            if self.provider_name == "AWS Transcribe":
                time.sleep(0.1)
            
            all_stop_events = self.stop_events + [self.local_stop]
            
            # Use smaller chunks for AWS to avoid buffer issues
            chunk_size = 512 if self.provider_name == "AWS Transcribe" else 1024
            
            while self.is_active_check() and not any(event.is_set() for event in all_stop_events):
                try:
                    # Add extra safety check for stream availability
                    if not hasattr(source, 'stream') or source.stream is None:
                        log_message("ERROR", "Audio stream not available")
                        break
                    
                    # Try with exception_on_overflow for newer PyAudio versions
                    try:
                        audio_data = source.stream.read(chunk_size, exception_on_overflow=False)
                    except TypeError:
                        # Fallback for older PyAudio versions
                        audio_data = source.stream.read(chunk_size)
                    except OSError as e:
                        # Handle buffer overflow/underflow
                        log_message("WARNING", f"Audio buffer error: {e}, continuing...")
                        continue
                        
                    if audio_data:
                        # Check if we should ignore input (e.g., during TTS playback)
                        with _ignore_input_lock:
                            should_ignore = _ignore_input
                        
                        if not should_ignore:
                            self.queue.put(audio_data)
                        else:
                            # Silently discard audio while ignoring input
                            pass
                except Exception as e:
                    if self.is_active_check():
                        log_message("ERROR", f"Error capturing audio: {e}")
                    break

            log_message("INFO", f"Audio capture thread ending for {self.provider_name}")


class StreamingResponseHandler:
    """Base class for handling streaming responses"""
    
    def __init__(self, engine, provider_name):
        self.engine = engine
        self.provider_name = provider_name
        self.current_partial = ""
    
    def handle_partial(self, text: str):
        """Handle partial transcript"""
        # Don't process if engine is stopped
        if not self.engine.is_active or self.engine.stop_event.is_set():
            return
            
        if text != self.current_partial:
            self.current_partial = text
            self.engine.recognition_active = True
            if self.engine.partial_callback:
                self.engine.partial_callback(text)
                log_message("DEBUG", f"{self.provider_name} partial: {text}")
    
    def handle_final(self, text: str):
        """Handle final transcript"""
        # Don't process if engine is stopped
        if not self.engine.is_active or self.engine.stop_event.is_set():
            return
            
        if text and self.engine.text_callback:
            processed = self.engine._process_dictation_text(text)
            self.engine.text_callback(processed)
            log_message("INFO", f"{self.provider_name} final: {text}")
            self.current_partial = ""
            self.engine.last_recognition_time = time.time()
            self.engine.recognition_active = True


def common_stream_loop(provider_name: str, engine, microphone, process_audio_fn):
    """Common streaming loop used by most providers"""
    audio_queue = queue.Queue()
    audio_capture = AudioCaptureThread(
        microphone, audio_queue,
        lambda: engine.is_active,
        [engine.stop_event],
        provider_name
    ).start()
    
    try:
        handler = StreamingResponseHandler(engine, provider_name)
        process_audio_fn(audio_queue, handler)
    finally:
        audio_capture.stop()


class ASRProvider(ABC):
    """Abstract base class for ASR providers"""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.name = config.provider
        
    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming mode"""
        return True  # Most providers support streaming
        
    @abstractmethod
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        """Check if credentials are configured. Returns (success, error_message)"""
        pass
    
    @abstractmethod
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        """Check if required libraries are installed. Returns (success, error_message)"""
        pass
    
    @abstractmethod
    def recognize(self, audio) -> str:
        """Recognize speech from audio data"""
        pass
    
    @abstractmethod
    def stream(self, engine, microphone) -> None:
        """Stream audio for real-time recognition"""
        pass
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate provider is ready to use"""
        # Check imports first
        success, error = self.check_imports()
        if not success:
            return False, error
            
        # Then check credentials
        success, error = self.check_credentials()
        if not success:
            return False, error
            
        return True, None


class GoogleFreeProvider(ASRProvider):
    """Free Google Speech Recognition provider"""
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('google_free', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('google_free')
    
    def recognize(self, audio) -> str:
        recognizer = sr.Recognizer()
        return recognizer.recognize_google(
            audio, 
            language=self.config.language,
            show_all=False
        )
    
    def stream(self, engine, microphone) -> None:
        """Stream using listen_in_background"""
        recognizer = sr.Recognizer()
        
        def callback(recognizer, audio):
            try:
                # Check if we should ignore input (e.g., during TTS playback or tap-to-talk not active)
                with _ignore_input_lock:
                    should_ignore = _ignore_input
                
                if should_ignore:
                    log_message("DEBUG", "[GOOGLE] Discarding audio segment due to ignore_input flag")
                    return
                
                result = recognizer.recognize_google(
                    audio, 
                    language=self.config.language,
                    show_all=True
                )
                
                if result and 'alternative' in result:
                    best = result['alternative'][0]
                    transcript = best.get('transcript', '')
                    
                    if transcript:
                        processed = engine._process_dictation_text(transcript)
                        if engine.text_callback:
                            engine.text_callback(processed)
                        log_message("INFO", f"Google transcript: {transcript}")
                        engine.last_recognition_time = time.time()
                        engine.recognition_active = True
                            
            except sr.UnknownValueError:
                log_message("DEBUG", "Google could not understand audio")
            except sr.RequestError as e:
                log_message("ERROR", f"Google recognition error: {e}")
        
        # Start background listening
        stop_listening = recognizer.listen_in_background(
            microphone,
            callback,
            phrase_time_limit=engine.phrase_time_limit
        )
        
        # Keep thread alive
        while engine.is_active and not engine.stop_event.is_set():
            time.sleep(0.1)
            
        # Stop the background listener properly
        try:
            stop_listening(wait_for_stop=True)
            log_message("INFO", "Stopped Google background listener")
        except Exception as e:
            log_message("ERROR", f"Error stopping Google background listener: {e}")


class GoogleCloudProvider(ASRProvider):
    """Google Cloud Speech-to-Text provider"""
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('gcloud', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('gcloud')
    
    def recognize(self, audio) -> str:
        raise NotImplementedError("Batch recognition not needed for Google Cloud")
    
    def stream(self, engine, microphone) -> None:
        """Stream using Google Cloud streaming API"""
        from google.cloud import speech
        
        def process_audio(audio_queue, handler):
            client = speech.SpeechClient()

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.config.language,
                enable_automatic_punctuation=True,
                model="latest_long",
                use_enhanced=True,
                max_alternatives=1
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False  # Process multiple utterances
            )
            
            # Main loop that restarts the stream when needed
            while engine.is_active and not engine.stop_event.is_set():
                try:
                    log_message("INFO", "Starting Google Cloud streaming recognition")
                    
                    # Create audio generator
                    def audio_generator():
                        """Continuously yield audio chunks"""
                        # Keep track of how long we've been streaming
                        start_time = time.time()
                        
                        while engine.is_active and not engine.stop_event.is_set():
                            try:
                                # Check if we're approaching the 5-minute limit
                                if time.time() - start_time > 290:  # 4:50 minutes
                                    log_message("INFO", "Approaching 5-minute limit, ending stream")
                                    break
                                
                                # Get audio chunk
                                chunk = audio_queue.get(timeout=0.1)
                                if chunk is None:
                                    break
                                    
                                yield speech.StreamingRecognizeRequest(audio_content=chunk)
                                
                            except queue.Empty:
                                # Continue waiting for audio
                                continue
                            except Exception as e:
                                log_message("ERROR", f"Error in audio generator: {e}")
                                break
                    
                    # Create the request stream
                    requests = audio_generator()
                    
                    # Get responses
                    responses = client.streaming_recognize(
                        streaming_config,
                        requests,
                        timeout=300.0
                    )
                    
                    # Process responses directly (like the sample code)
                    for response in responses:
                        # Check if we should stop processing
                        if not engine.is_active or engine.stop_event.is_set():
                            log_message("DEBUG", "Stopping response processing - engine stopped")
                            break
                            
                        if not response.results:
                            continue
                        
                        # Process the first result (streaming only cares about first)
                        result = response.results[0]
                        if not result.alternatives:
                            continue
                        
                        transcript = result.alternatives[0].transcript
                        
                        if result.is_final:
                            handler.handle_final(transcript)
                            log_message("DEBUG", f"Google Cloud final: {transcript}")
                        else:
                            handler.handle_partial(transcript)
                            log_message("DEBUG", f"Google Cloud partial: {transcript}")
                    
                    # If we get here, the stream ended normally
                    log_message("INFO", "Google Cloud stream ended, restarting...")
                    time.sleep(0.1)  # Brief pause before restarting
                    
                except Exception as e:
                    error_msg = str(e)
                    if "Maximum allowed stream duration" in error_msg or "Exceeded maximum allowed stream duration" in error_msg:
                        log_message("INFO", "Google Cloud stream hit time limit, restarting...")
                        time.sleep(0.1)
                        continue
                    elif "OUT_OF_RANGE" in error_msg:
                        log_message("INFO", "Google Cloud stream timed out, restarting...")
                        time.sleep(0.1)
                        continue
                    elif engine.is_active:
                        log_message("ERROR", f"Google Cloud streaming error: {e}")
                        print(f"\nGoogle Cloud Error: {e}")
                        # Don't exit on DeadlineExceeded errors
                        if "DeadlineExceeded" not in error_msg:
                            break
                        else:
                            time.sleep(0.1)
                            continue
                    else:
                        break
        
        common_stream_loop("Google Cloud", engine, microphone, process_audio)


class AssemblyAIProvider(ASRProvider):
    """AssemblyAI Speech Recognition provider"""
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('assemblyai', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('assemblyai')
    
    def recognize(self, audio) -> str:
        raise NotImplementedError("Batch recognition not needed for Assmbly AI")
    
    def stream(self, engine, microphone) -> None:
        """Stream using AssemblyAI v3 streaming API"""
        from assemblyai.streaming.v3 import (
            StreamingClient,
            StreamingClientOptions,
            StreamingError,
            StreamingEvents,
            StreamingParameters,
            TurnEvent,
            BeginEvent,
            TerminationEvent,
        )
        
        def process_audio(audio_queue, handler):
            # Configure API key
            api_key = self.config.api_key or os.environ.get('ASSEMBLYAI_API_KEY')
            error_reported = False
            client = None
            
            # Create event handlers
            def on_begin(client_ref: StreamingClient, event: BeginEvent):
                log_message("INFO", f"AssemblyAI session started: {event.id}")
            
            def on_turn(client_ref: StreamingClient, event: TurnEvent):
                if event.transcript:
                    log_message("DEBUG", f"AssemblyAI turn - transcript: '{event.transcript}', end_of_turn: {event.end_of_turn}")
                    
                    # TurnEvent represents a complete utterance/turn
                    if event.end_of_turn:
                        # This is a final transcript for this turn
                        handler.handle_final(event.transcript)
                    elif engine.partial_callback:
                        # This is a partial transcript
                        handler.handle_partial(event.transcript)
            
            def on_terminated(client_ref: StreamingClient, event: TerminationEvent):
                log_message("INFO", f"AssemblyAI session terminated: {event.audio_duration_seconds} seconds processed")
            
            def on_error(client_ref: StreamingClient, error: StreamingError):
                nonlocal error_reported
                log_message("ERROR", f"AssemblyAI error: {error}")
                if not error_reported:
                    error_reported = True
                    print(f"\nAssemblyAI Error: {error}")
                    # Signal the engine to stop
                    engine.is_active = False
                    engine.stop_event.set()
            
            try:
                # Create client
                client = StreamingClient(
                    StreamingClientOptions(
                        api_key=api_key,
                        api_host="streaming.assemblyai.com",
                    )
                )
                
                # Register event handlers
                client.on(StreamingEvents.Begin, on_begin)
                client.on(StreamingEvents.Turn, on_turn)
                client.on(StreamingEvents.Termination, on_terminated)
                client.on(StreamingEvents.Error, on_error)
                
                # Connect with parameters
                # Configure voice activity detection parameters
                params = StreamingParameters(
                    sample_rate=16000,
                    format_turns=False,  # We handle formatting ourselves
                )
                
                # Add configurable VAD parameters for less aggressive end-of-turn detection
                # Use unified setting for minimum silence
                params.end_of_turn_confidence_threshold = float(os.environ.get('ASSEMBLYAI_END_OF_TURN_CONFIDENCE', '0.5'))
                params.min_end_of_turn_silence_when_confident = DEFAULT_MIN_SILENCE_MS
                
                if os.environ.get('ASSEMBLYAI_MAX_SILENCE_MS'):
                    params.max_turn_silence = int(os.environ.get('ASSEMBLYAI_MAX_SILENCE_MS'))
                else:
                    params.max_turn_silence = 5000  # Higher = less aggressive (default: 2400ms)
                
                log_message("INFO", f"AssemblyAI VAD settings: confidence={params.end_of_turn_confidence_threshold}, "
                           f"min_silence={params.min_end_of_turn_silence_when_confident}ms (unified), "
                           f"max_silence={params.max_turn_silence}ms")
                
                client.connect(params)
                
                log_message("INFO", "AssemblyAI v3 connection established")
                
                # Stream audio
                while engine.is_active and not engine.stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        # Send raw audio bytes to the client
                        client.stream(chunk)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        if engine.is_active:
                            log_message("ERROR", f"Error streaming to AssemblyAI: {e}")
                        break
                
            except Exception as e:
                log_message("ERROR", f"AssemblyAI v3 streaming error: {e}")
                print(f"\nAssemblyAI Error: {e}")
                engine.is_active = False
                engine.stop_event.set()
            finally:
                if client:
                    try:
                        client.disconnect(terminate=True)
                        log_message("INFO", "AssemblyAI v3 disconnected")
                    except Exception:
                        pass
        
        common_stream_loop("AssemblyAI", engine, microphone, process_audio)


class HoundifyProvider(ASRProvider):
    """Houndify Speech Recognition provider"""
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('houndify', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('houndify')
    
    def recognize(self, audio) -> str:
        raise NotImplementedError("Batch recognition not needed for Houndify")
    
    def stream(self, engine, microphone) -> None:
        """Stream using Houndify streaming API"""
        import houndify
        
        client_id = self.config.client_id or os.environ.get('HOUNDIFY_CLIENT_ID')
        client_key = self.config.client_key or os.environ.get('HOUNDIFY_CLIENT_KEY')
        user_id = self.config.user_id or "default_user"
        
        # Response handler
        handler = StreamingResponseHandler(engine, "Houndify")
        
        class StreamingListener(houndify.HoundListener):
            def onPartialTranscript(self, transcript):
                log_message("DEBUG", f"Houndify onPartialTranscript: {transcript}")
                if transcript:
                    handler.handle_partial(transcript)

            def onFinalResponse(self, response):
                # Extract final transcript from response
                log_message("DEBUG", f"Houndify onFinalResponse: {response}")
                try:
                    if 'AllResults' in response and response['AllResults']:
                        for result in response['AllResults']:
                            if 'RawTranscription' in result:
                                transcript = result['RawTranscription']
                                if transcript:
                                    handler.handle_final(transcript)
                                    log_message("INFO", f"Houndify final transcript: {transcript}")
                                    return
                    if 'Disambiguation' in response and 'ChoiceData' in response['Disambiguation']:
                        if response['Disambiguation']['ChoiceData']:
                            transcript = response['Disambiguation']['ChoiceData'][0].get('Transcription', None)
                            if transcript:
                                handler.handle_final(transcript)
                                log_message("INFO", f"Houndify final transcript: {transcript}")
                                return
                except Exception as e:
                    log_message("ERROR", f"Error parsing Houndify response: {e}")
                    log_message("DEBUG", f"Response was: {response}")
            
            def onError(self, err):
                log_message("ERROR", f"Houndify error: {err}")
        
        # Create a persistent client
        client = None
        listener = None

        # Main streaming loop
        try:
            # Initialize client once
            client = houndify.StreamingHoundClient(
                clientID=client_id,
                clientKey=client_key,
                userID=user_id,
                enableVAD=True,
                useSpeex=False
            )
            # Houndify requires 16kHz sample rate
            client.setSampleRate(16000)
            # Set location (optional but recommended)
            client.setLocation(37.388309, -121.973968)
            listener = StreamingListener()
            client.start(listener)

            with microphone as source:
                bytes_sent = 0
                chunks_sent = 0
                
                while engine.is_active and not engine.stop_event.is_set():
                    try:
                        # Check if we should ignore input (e.g., during TTS playback or tap-to-talk not active)
                        with _ignore_input_lock:
                            should_ignore = _ignore_input
                        
                        # Use smaller buffer size like the reference (256 frames = 512 bytes for 16-bit audio)
                        try:
                            audio_data = source.stream.read(512, exception_on_overflow=False)
                        except TypeError:
                            # Fallback for older PyAudio versions
                            audio_data = source.stream.read(512)
                        
                        if should_ignore:
                            log_message("DEBUG", "[HOUNDIFY] Discarding audio segment due to ignore_input flag")
                            continue  # Skip processing but keep reading to prevent buffer overflow
                        
                        bytes_sent += len(audio_data)
                        chunks_sent += 1
                        
                        if chunks_sent % 50 == 0:  # Log progress
                            log_message("DEBUG", f"Houndify streaming: {chunks_sent} chunks, {bytes_sent} bytes")
                        
                        filled = client.fill(audio_data)
                        if filled:
                            # Client indicated it's done with this utterance
                            log_message("INFO", f"Houndify utterance complete after {bytes_sent} bytes")
                            # Finish current session and start a new one
                            client.finish()
                            
                            # Create new client for next utterance
                            client = houndify.StreamingHoundClient(
                                clientID=client_id,
                                clientKey=client_key,
                                userID=user_id,
                                enableVAD=True,
                                useSpeex=False
                            )
                            # Houndify requires 16kHz sample rate
                            client.setSampleRate(16000)
                            # Set location (optional but recommended)
                            client.setLocation(37.388309, -121.973968)
                            listener = StreamingListener()
                            client.start(listener)
                            
                            # Reset counters
                            bytes_sent = 0
                            chunks_sent = 0
                            
                    except Exception as e:
                        if engine.is_active:
                            log_message("ERROR", f"Error streaming to Houndify: {e}")
                        break
        except Exception as e:
            log_message("ERROR", f"Houndify streaming error: {e}")
        finally:
            # Clean up
            if client:
                try:
                    client.finish()
                except Exception:
                    pass


class AWSTranscribeProvider(ASRProvider):
    """Amazon Transcribe provider"""
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('aws', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('aws')
    
    def recognize(self, audio) -> str:
        raise NotImplementedError("Batch recognition not implemented for AWS Transcribe")
    
    def stream(self, engine, microphone) -> None:
        """Stream using AWS Transcribe"""
        # Try new SDK first
        try:
            from amazon_transcribe.client import TranscribeStreamingClient  # noqa: F401
            log_message("INFO", "Using amazon-transcribe SDK for streaming")
            self._stream_with_sdk(engine, microphone)
        except ImportError as e:
            # Fall back to websocket - but it's not implemented
            log_message("ERROR", "amazon-transcribe SDK not available and websocket fallback not implemented")
            raise ImportError(f"amazon-transcribe SDK required: {e}")
    
    def _stream_with_sdk(self, engine, microphone):
        """Stream using the amazon-transcribe SDK"""
        from amazon_transcribe.client import TranscribeStreamingClient
        import asyncio
        
        def process_audio(audio_queue, handler):
            
            async def mic_stream():
                while engine.is_active:
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        yield chunk
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        log_message("ERROR", f"Error in mic_stream: {e}")
                        break
            
            async def run_streaming():
                log_message("INFO", f"Creating AWS Transcribe client for region: {self.config.region or 'us-east-1'}")
                client = TranscribeStreamingClient(region=self.config.region or 'us-east-1')
                
                log_message("INFO", "Starting stream transcription...")
                stream = await client.start_stream_transcription(
                    language_code=self.config.language,
                    media_sample_rate_hz=16000,
                    media_encoding="pcm",
                    enable_partial_results_stabilization=True,
                    partial_results_stability="high"
                )
                log_message("INFO", "Stream transcription started")
                
                # Process events directly
                async def handle_stream():
                    log_message("INFO", "Starting to handle stream events...")
                    event_count = 0
                    async for event in stream.output_stream:
                        event_count += 1
                        log_message("DEBUG", f"Received event #{event_count}: {type(event).__name__}")
                        
                        # The event IS the transcript event directly
                        try:
                            # Check if event has transcript attribute
                            if not hasattr(event, 'transcript') or event.transcript is None:
                                log_message("DEBUG", f"Event has no transcript: {type(event).__name__}")
                                continue
                            
                            # Check if transcript has results
                            if not hasattr(event.transcript, 'results') or not event.transcript.results:
                                log_message("DEBUG", "Transcript has no results")
                                continue
                                
                            results = event.transcript.results
                            log_message("DEBUG", f"Transcript event with {len(results)} results")
                            
                            for result in results:
                                if not result or not hasattr(result, 'alternatives') or not result.alternatives:
                                    continue
                                
                                # Check if alternatives list is not empty
                                if len(result.alternatives) == 0:
                                    continue
                                    
                                transcript = result.alternatives[0].transcript
                                
                                if hasattr(result, 'is_partial') and result.is_partial:
                                    log_message("DEBUG", f"Partial transcript: {transcript}")
                                    handler.handle_partial(transcript)
                                else:
                                    log_message("INFO", f"Final transcript: {transcript}")
                                    handler.handle_final(transcript)
                        except AttributeError as e:
                            log_message("DEBUG", f"Event has no transcript: {e}")
                            continue
                
                async def send_audio(stream, audio_generator):
                    chunk_count = 0
                    async for chunk in audio_generator:
                        chunk_count += 1
                        if chunk_count % 100 == 0:  # Log every 100 chunks
                            log_message("DEBUG", f"Sent {chunk_count} audio chunks to AWS")
                        await stream.input_stream.send_audio_event(audio_chunk=chunk)
                    log_message("INFO", f"Audio stream ended. Total chunks sent: {chunk_count}")
                    await stream.input_stream.end_stream()
                
                await asyncio.gather(
                    handle_stream(),
                    send_audio(stream, mic_stream())
                )
            
            # Run in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_streaming())
        
        common_stream_loop("AWS Transcribe", engine, microphone, process_audio)
    

class AzureSpeechProvider(ASRProvider):
    """Azure Speech Services provider"""
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('azure', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('azure')

    def recognize(self, audio) -> str:
        raise NotImplementedError("Batch recognition not needed for Azure Speech")
    
    def stream(self, engine, microphone) -> None:
        """Stream using Azure Speech continuous recognition"""
        import azure.cognitiveservices.speech as speechsdk
        
        def process_audio(audio_queue, handler):
            key = self.config.api_key or os.environ.get('AZURE_SPEECH_KEY')
            region = self.config.region or os.environ.get('AZURE_SPEECH_REGION')
            
            # Configure speech config
            speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
            speech_config.speech_recognition_language = self.config.language
            
            # Enable dictation mode for better continuous speech
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000"
            )
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1000"
            )
            
            # Create push stream and audio config
            stream = speechsdk.audio.PushAudioInputStream()
            recognizer = None
            try:
                audio_config = speechsdk.audio.AudioConfig(stream=stream)
                
                # Create recognizer
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config,
                    audio_config=audio_config
                )
                
                # Set up callbacks
                recognizer.recognizing.connect(lambda evt: handler.handle_partial(evt.result.text))
                recognizer.recognized.connect(lambda evt: handler.handle_final(evt.result.text) if evt.result.text else None)
                recognizer.session_started.connect(lambda evt: log_message('INFO', 'SESSION STARTED: {}'.format(evt)))

                def canceled_callback(evt):
                    log_message("ERROR", f"Azure canceled: {format(evt)}")
                    if hasattr(evt, 'result') and hasattr(evt.result, 'cancellation_details'):
                        details = evt.result.cancellation_details
                        if hasattr(details, 'error_details'):
                            print(f'Azure Error: {details.error_details}')
                        else:
                            print(f'Azure Error: {details.reason}')

                recognizer.canceled.connect(canceled_callback)

                def session_stopped(evt):
                    log_message("INFO", f"Azure session stopped: {evt.session_id}")
                    # Don't automatically stop the engine - Azure may restart the session
                    engine.is_active = False
                    engine.stop_event.set()

                recognizer.session_stopped.connect(session_stopped)
                
                # Start continuous recognition
                recognizer.start_continuous_recognition()
                log_message("INFO", "Azure continuous recognition started")
                
                # Stream audio
                while engine.is_active and not engine.stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        stream.write(chunk)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        if engine.is_active:
                            log_message("ERROR", f"Error streaming to Azure: {e}")
                        break
            finally:
                # Always clean up resources
                if recognizer:
                    try:
                        recognizer.stop_continuous_recognition()
                    except Exception as e:
                        log_message("WARNING", f"Error stopping recognizer: {e}")
                
                try:
                    stream.close()
                except Exception as e:
                    log_message("WARNING", f"Error closing stream: {e}")
        
        common_stream_loop("Azure", engine, microphone, process_audio)


class DeepgramProvider(ASRProvider):
    """Deepgram Speech Recognition provider"""
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('deepgram', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('deepgram')
    
    def recognize(self, audio) -> str:
        raise NotImplementedError("Batch recognition not needed for Deepgram")
    
    def stream(self, engine, microphone) -> None:
        """Stream using Deepgram SDK"""
        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents  # noqa: F401
        except ImportError:
            log_message("ERROR", "Deepgram SDK not found. Please install: pip install deepgram-sdk")
            raise ImportError("Deepgram SDK required: pip install deepgram-sdk")
        
        def process_audio(audio_queue, handler):
            api_key = self.config.api_key or os.environ.get('DEEPGRAM_API_KEY')
            
            try:
                # Create Deepgram client
                deepgram = DeepgramClient(api_key)
                
                # Create websocket connection - use live instead of websocket
                dg_connection = deepgram.listen.websocket.v("1")
                
                # Set up event handlers
                def on_message(self, result, **kwargs):
                    if result.channel and result.channel.alternatives:
                        sentence = result.channel.alternatives[0].transcript
                        if sentence:
                            # Check if this is final or interim
                            if hasattr(result, 'is_final') and result.is_final:
                                handler.handle_final(sentence)
                            elif engine.partial_callback:
                                handler.handle_partial(sentence)

                            log_message("DEBUG", f"Deepgram transcript - text: '{sentence}', is_final: {getattr(result, 'is_final', False)}")

                def on_error(self, error, **kwargs):
                    log_message("ERROR", f"Deepgram error: {error}")
                    if "401" in str(error) or "Unauthorized" in str(error):
                        print("\nDeepgram API key error. Please check your DEEPGRAM_API_KEY.")
                    engine.is_active = False
                    engine.stop_event.set()

                def on_close(self, close, **kwargs):
                    log_message("INFO", "Deepgram connection closed")

                # Register event handlers
                dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
                dg_connection.on(LiveTranscriptionEvents.Error, on_error)
                dg_connection.on(LiveTranscriptionEvents.Close, on_close)
                
                # Configure options
                # Use unified setting for utterance end delay
                utterance_end_ms = DEFAULT_MIN_SILENCE_MS
                
                options = LiveOptions(
                    model="nova-3",
                    language=self.config.language,
                    punctuate = True,
                    smart_format = True,
                    interim_results = True,
                    utterance_end_ms = utterance_end_ms,
                    vad_events = True,
                    encoding = "linear16",
                    sample_rate = 16000
                )
                
                log_message("INFO", f"Deepgram VAD settings: utterance_end_ms={utterance_end_ms}ms (unified)")

                # Start connection
                if not dg_connection.start(options):
                    log_message("ERROR", "Failed to start Deepgram connection")
                    print("\nError: Failed to start Deepgram connection")
                    engine.stop_event.set()
                    return
                
                log_message("INFO", "Deepgram connection established")
                
                # Stream audio
                while engine.is_active and not engine.stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        dg_connection.send(chunk)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        if engine.is_active:
                            log_message("ERROR", f"Error streaming to Deepgram: {e}")
                        break
                
                # Finish the stream
                dg_connection.finish()
                log_message("INFO", "Deepgram stream finished")
                
            except Exception as e:
                log_message("ERROR", f"Deepgram streaming error: {e}")
                print(f"\nDeepgram Error: {e}")
                engine.is_active = False
                engine.stop_event.set()
        
        common_stream_loop("Deepgram", engine, microphone, process_audio)


class WhisperProvider(ASRProvider):
    """OpenAI Whisper provider - supports both local and API modes"""
    
    @property
    def supports_streaming(self) -> bool:
        """Whisper doesn't support streaming"""
        return False
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('whisper', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('whisper')

    
    def recognize(self, audio) -> str:
        """Recognize using OpenAI Whisper API"""
        import openai
        import httpx
        
        # Configure API
        api_key = self.config.api_key or os.environ.get('OPENAI_API_KEY')
        # Disable HTTP logging from OpenAI client
        client = openai.OpenAI(
            api_key=api_key,
            http_client=httpx.Client(
                event_hooks={"request": [], "response": []}
            )
        )
        
        # Get audio data and use temp file
        wav_data = audio.get_wav_data()
        
        with temp_audio_file(wav_data) as tmp_path:
            # Transcribe using API
            with open(tmp_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model='whisper-1',
                    file=audio_file,
                    language=self.config.language.split('-')[0]  # 'en-US' -> 'en'
                )
            
            return transcript.text.strip()
    
    def stream(self, engine, microphone) -> None:
        """Whisper doesn't support streaming - will use phrase-based mode"""
        raise NotImplementedError("Whisper doesn't support streaming")


class FasterWhisperProvider(ASRProvider):
    """faster-whisper provider - offline local ASR using chunked processing"""

    @property
    def supports_streaming(self) -> bool:
        """Supports chunked real-time streaming with faster-whisper"""
        return True
    
    def check_credentials(self) -> Tuple[bool, Optional[str]]:
        return check_api_credentials('local_whisper', self.config)
    
    def check_imports(self) -> Tuple[bool, Optional[str]]:
        return check_provider_imports('local_whisper')
    
    def recognize(self, audio) -> str:
        """Recognize using faster-whisper offline model"""
        from faster_whisper import WhisperModel  # noqa: F401

        # Get configuration - model defaults to small
        model_name = self.config.model or os.environ.get('WHISPER_MODEL', 'small')
        device = os.environ.get('WHISPER_DEVICE', 'cpu')
        compute_type = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')
        
        log_message("DEBUG", f"Using faster-whisper model: {model_name}, device: {device}, compute_type: {compute_type}")
        
        # Initialize model (should already be cached from availability check)
        log_message("DEBUG", f"Using faster-whisper model: {model_name}")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        
        # Get audio data as numpy array
        wav_data = audio.get_wav_data()
        
        # Convert WAV data to numpy array
        # The audio from speech_recognition is in 16-bit PCM format at 16kHz
        with temp_audio_file(wav_data) as tmp_path:
            # Load audio using faster-whisper's method
            # faster-whisper expects audio as float32 array
            try:
                # Try to read with soundfile first
                import soundfile as sf
                audio_array, sample_rate = sf.read(tmp_path, dtype='float32')
            except ImportError:
                log_message("DEBUG", "soundfile not available, using numpy fallback")
                # Fallback: convert from bytes directly
                # WAV data from speech_recognition is 16-bit PCM at 16kHz
                audio_array = np.frombuffer(wav_data[44:], dtype=np.int16).astype(np.float32) / 32768.0
                # sample_rate = 16000  # Known constant, no need to assign
            except Exception:
                # Fallback: convert from bytes directly
                # WAV data from speech_recognition is 16-bit PCM at 16kHz
                audio_array = np.frombuffer(wav_data[44:], dtype=np.int16).astype(np.float32) / 32768.0
                # sample_rate = 16000  # Known constant, no need to assign
            
            # Ensure mono audio
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Convert language format: 'en-US' -> 'en'  
            language = self.config.language.split('-')[0] if self.config.language else 'en'
            
            # Transcribe
            segments, info = model.transcribe(
                audio_array,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=None,
                word_timestamps=False
            )
            
            # Collect all text from segments
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text)
            
            result = ' '.join(transcript_parts).strip()
            log_message("INFO", f"faster-whisper transcribed: '{result}' (language: {info.language}, probability: {info.language_probability:.2f})")
            
            return result
    
    def stream(self, engine, microphone) -> None:
        """Stream audio using faster-whisper with chunked processing and separate transcription thread"""
        import queue
        global _tap_to_talk_outstanding_frames
        
        # Get configuration
        model_name = self.config.model or os.environ.get('WHISPER_MODEL', 'small')
        device = os.environ.get('WHISPER_DEVICE', 'cpu')
        compute_type = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')
        
        log_message("INFO", f"[LOCAL_WHISPER] Initializing model: {model_name}, device: {device}, compute_type: {compute_type}")
        
        # Get optimal backend configuration
        backend_info = _get_local_whisper_backend_info()
        use_pywhispercpp = backend_info["use_pywhispercpp"]

        try:
            # Get the cached model with timeout
            model = get_cached_local_whisper_model(timeout_seconds=60.0)
            if model is None:
                raise ValueError("Failed to load local whisper model within timeout")

            if use_pywhispercpp:
                coreml_model_name =  model_name
                log_message("INFO", f"[LOCAL_WHISPER] Using pre-loaded PyWhisperCpp CoreML model: {coreml_model_name}")
            else:
                log_message("INFO", f"[LOCAL_WHISPER] Using pre-loaded faster-whisper model: {model_name}")
            
            # Audio configuration optimized for real-time processing
            CHUNK = 2048    # Larger PyAudio buffer reduces context switching overhead
            FORMAT = pyaudio.paFloat32
            CHANNELS = 1
            RATE = 16000
            
            # Optimize transcription window duration based on model size for faster response
            TRANSCRIPTION_WINDOW_DURATION = 1.1
                
            # Calculate how many PyAudio frames to accumulate before transcribing
            FRAMES_TO_ACCUMULATE = int(RATE * TRANSCRIPTION_WINDOW_DURATION / CHUNK)
            
            log_message("INFO", f"[LOCAL_WHISPER] Audio config - pyaudio_frame_size: {CHUNK} samples (~{CHUNK/RATE*1000:.0f}ms), transcription_window: {TRANSCRIPTION_WINDOW_DURATION}s, frames_to_accumulate: {FRAMES_TO_ACCUMULATE}")

            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Open stream with optimized buffer size
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            audio_queue = queue.Queue(maxsize=10)

            def transcription_worker():
                """Worker thread that handles transcription without blocking audio capture"""
                global _tap_to_talk_outstanding_frames
                
                # Log model reference to monitor if it's shared
                log_message("DEBUG", f"[LOCAL_WHISPER] Starting transcription worker thread (model id: {id(model)})")
                log_message("DEBUG", "[LOCAL_WHISPER] Starting transcription worker thread")
                while True:
                    try:
                        # Get audio segment from queue (non-blocking)
                        audio_segment_data = audio_queue.get_nowait()
                        
                        # Check for shutdown signal
                        if audio_segment_data is None:
                            log_message("DEBUG", "[LOCAL_WHISPER] Transcription worker shutting down")
                            break
                            
                        audio_np, energy_level, frames_in_segment = audio_segment_data
                        
                        # Check if we should ignore input (e.g., during TTS) before processing
                        # But don't ignore if we still have outstanding chunks to process
                        with _ignore_input_lock:
                            should_ignore = _ignore_input
                        
                        with _tap_to_talk_counter_lock:
                            has_outstanding_frames = _tap_to_talk_outstanding_frames > 0
                        
                        if should_ignore and not has_outstanding_frames:
                            log_message("DEBUG", f"[LOCAL_WHISPER] Discarding audio segment due to ignore_input flag: energy={energy_level:.4f}")
                            continue
                        
                        log_message("DEBUG", f"[LOCAL_WHISPER] Processing audio segment in worker thread: energy={energy_level:.4f}")
                        
                        # Transcribe using the appropriate backend
                        try:
                            start_time = time.time()
                            
                            # Check if we're using PyWhisperCpp or faster-whisper
                            if hasattr(model, '__class__') and 'pywhispercpp' in str(type(model)):
                                # PyWhisperCpp CoreML backend - direct transcription
                                raw_result = model.transcribe(audio_np, print_progress=False, print_realtime=False, print_timestamps=False, print_special=False)
                                transcription_time = time.time() - start_time
                                log_message("DEBUG", f"[LOCAL_WHISPER] PyWhisperCpp transcription took {transcription_time:.3f}s for {len(audio_np)/RATE:.3f}s audio (real-time factor: {transcription_time/(len(audio_np)/RATE):.2f}x)")
                                log_message("DEBUG", f"[LOCAL_WHISPER] PyWhisperCpp raw result type: {type(raw_result)}, content: '{raw_result}'")
                                
                                # Handle both string and list results from PyWhisperCpp
                                if isinstance(raw_result, list):
                                    # If it's a list of segments, extract text
                                    text_result = ""
                                    segment_count = len(raw_result)
                                    for segment in raw_result:
                                        if hasattr(segment, 'text'):
                                            text_result += segment.text
                                        elif isinstance(segment, str):
                                            text_result += segment
                                        else:
                                            text_result += str(segment)
                                elif isinstance(raw_result, str):
                                    # If it's already a string, use it directly
                                    text_result = raw_result
                                    segment_count = 1 if text_result.strip() else 0
                                else:
                                    # Fallback: convert to string
                                    text_result = str(raw_result)
                                    segment_count = 1 if text_result.strip() else 0

                                # Remove sound annotations such as [MUSIC] or [CLOCK TICKING]
                                text_result = SQUARE_BRACKETS_CLEANER.sub('', text_result).strip()
                                
                                log_message("DEBUG", f"[LOCAL_WHISPER] PyWhisperCpp final result: text_length={len(text_result)}, segments={segment_count}, content='{text_result}'")
                                
                            else:
                                # faster-whisper backend with optimized settings
                                segments_generator, info = model.transcribe(
                                    audio_np,
                                    beam_size=1,                       # Minimum beam size for fastest decoding
                                    best_of=1,                         # No multiple passes
                                    temperature=0.0,                   # Deterministic output, no sampling
                                    language="en",                     # Force English to skip language detection  
                                    condition_on_previous_text=False,  # Don't use context for speed
                                    vad_filter=False,                  # Disable VAD for tap-to-talk mode
                                    compression_ratio_threshold=2.4,   # Skip overly repetitive text
                                    log_prob_threshold=-1.0,          # Skip low confidence text
                                    no_speech_threshold=0.6,          # Skip segments with no speech
                                    initial_prompt=None,              # No context prompt
                                    suppress_blank=True,              # Don't output blank tokens
                                    suppress_tokens=[-1],             # Suppress special tokens
                                    without_timestamps=True           # Skip timestamp calculation for speed
                                )
                                
                                # Force generator evaluation to get actual transcription time
                                segments = list(segments_generator)
                                transcription_time = time.time() - start_time
                                log_message("DEBUG", f"[LOCAL_WHISPER] faster-whisper transcription took {transcription_time:.3f}s for {len(audio_np)/RATE:.3f}s audio (real-time factor: {transcription_time/(len(audio_np)/RATE):.2f}x)")
                                log_message("DEBUG", f"[LOCAL_WHISPER] Model response: language={info.language}, language_probability={info.language_probability:.2f}")
                                
                                # Extract text from segments
                                text_result = ""
                                segment_count = 0
                                for segment in segments:
                                    segment_count += 1
                                    log_message("DEBUG", f"[LOCAL_WHISPER] Segment {segment_count}: '{segment.text}' (start={segment.start:.2f}s, end={segment.end:.2f}s)")
                                    text_result += segment.text
                                
                                log_message("DEBUG", f"[LOCAL_WHISPER] faster-whisper total segments: {segment_count}, combined_text: '{text_result}'")
                            
                            if text_result.strip():
                                log_message("INFO", f"[LOCAL_WHISPER] Transcribed: '{text_result.strip()}'")
                                
                                # Check if we should ignore this transcription (tap-to-talk session may have ended)
                                # But don't ignore if we still have outstanding chunks to process
                                waiting_for_lock_start_time = time.time()
                                with _ignore_input_lock:
                                    should_ignore = _ignore_input
                                
                                with _tap_to_talk_counter_lock:
                                    has_outstanding_frames = _tap_to_talk_outstanding_frames > 0
                                waiting_for_lock_duration = time.time() - waiting_for_lock_start_time
                                log_message("DEBUG", f"[LOCAL_WHISPER] Waiting for lock took: {waiting_for_lock_duration:.3f}s")
                                
                                if should_ignore and not has_outstanding_frames:
                                    log_message("DEBUG", f"[LOCAL_WHISPER] Discarding transcription due to ignore_input flag: '{text_result.strip()}'")
                                else:
                                    # Process and callback
                                    processing_start_time = time.time()
                                    processed = engine._process_dictation_text(text_result.strip())
                                    processing_duration = time.time() - processing_start_time
                                    log_message("DEBUG", f"[LOCAL_WHISPER] Processing took {processing_duration:.3f}s")
                                    if processed and engine.text_callback:
                                        callback_start_time = time.time()
                                        engine.text_callback(processed)
                                        callback_duration = time.time() - callback_start_time
                                        log_message("DEBUG", f"[LOCAL_WHISPER] Callback execution took {callback_duration:.3f}s")
                            else:
                                log_message("DEBUG", f"[LOCAL_WHISPER] Model returned empty text (segments={segment_count})")
                                
                        except Exception as model_error:
                            log_message("ERROR", f"[LOCAL_WHISPER] Model transcription failed: {model_error}")
                        
                        # Mark task done
                        audio_queue.task_done()
                        
                        # Decrement outstanding frames counter for tap-to-talk tracking
                        with _tap_to_talk_counter_lock:
                            if _tap_to_talk_outstanding_frames > 0:
                                # Decrement by the number of frames that were processed in this segment
                                frames_to_decrement = min(frames_in_segment, _tap_to_talk_outstanding_frames)
                                _tap_to_talk_outstanding_frames -= frames_to_decrement
                                log_message("DEBUG", f"[LOCAL_WHISPER] Tap-to-talk outstanding frames (-{frames_to_decrement}): {_tap_to_talk_outstanding_frames} (decremented by {frames_to_decrement})")
                        
                    except queue.Empty:
                        # No audio available - check if engine is still active
                        if not engine.is_active or engine.stop_event.is_set():
                            log_message("DEBUG", "[LOCAL_WHISPER] Transcription worker detected shutdown")
                            break
                        # Brief sleep to prevent busy-waiting
                        time.sleep(0.001)  # 1ms
                    except Exception as e:
                        log_message("ERROR", f"[LOCAL_WHISPER] Transcription worker error: {e}")
                        
                log_message("DEBUG", "[LOCAL_WHISPER] Transcription worker thread ended")
            
            # Start transcription worker thread
            transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
            transcription_thread.start()
            
            # Warm up the model with a short test audio chunk to avoid first-call delays
            # log_message("INFO", "[LOCAL_WHISPER] Warming up model with test audio...")
            # test_audio = np.zeros(int(RATE * 1.2), dtype=np.float32)  # 0.5 seconds of silence
            # try:
            #     if use_pywhispercpp:
            #         # PyWhisperCpp warm-up
            #         model.transcribe(test_audio, print_progress=False, print_realtime=False, print_timestamps=False, print_special=False)
            #     else:
            #         # faster-whisper warm-up
            #         segments, _ = model.transcribe(test_audio, language="en", task="transcribe")
            #         list(segments)  # Force evaluation to ensure model is loaded
            #     log_message("INFO", "[LOCAL_WHISPER] Model warm-up completed")
            # except Exception as e:
            #     log_message("WARNING", f"[LOCAL_WHISPER] Model warm-up failed (non-critical): {e}")
            
            log_message("INFO", "[LOCAL_WHISPER] Starting segmented audio capture with separate transcription thread")
            
            audio_buffer = []
            frame_count = 0
            
            # Pre-fill audio buffer before marking as ready for tap-to-talk
            log_message("DEBUG", "[LOCAL_WHISPER] Pre-filling audio buffer...")
            warmup_frames = 0
            while warmup_frames < 3 and engine.is_active:  # Collect 3 frames (~384ms) to prime the buffer
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    warmup_frames += 1
                    log_message("DEBUG", f"[LOCAL_WHISPER] Pre-filled frame {warmup_frames}/3")
                except Exception as e:
                    log_message("WARNING", f"[LOCAL_WHISPER] Buffer pre-fill warning: {e}")
                    break
            log_message("INFO", "[LOCAL_WHISPER] Audio stream fully warmed up and ready")
            
            # Track ignore_input state to detect tap-to-talk key release
            with _ignore_input_lock:
                previous_ignore_input = _ignore_input

            process_remaining_frames = False
            
            # Streaming loop with chunked processing
            while engine.is_active and not engine.stop_event.is_set():
                try:
                    # Check if we should ignore input (e.g., during TTS)
                    with _ignore_input_lock:
                        current_ignore_input = _ignore_input
                    
                    # Detect transition from pressed (False) to released (True)
                    transitioned_to_released = not previous_ignore_input and current_ignore_input
                    if transitioned_to_released:
                        # Check if we have outstanding frames or partial buffer
                        with _tap_to_talk_counter_lock:
                            outstanding_frames = _tap_to_talk_outstanding_frames
                        
                        if outstanding_frames > 0 or (audio_buffer and frame_count > 0):
                            process_remaining_frames = True
                            log_message("DEBUG", f"[LOCAL_WHISPER] Key released with {frame_count} buffered frames and {outstanding_frames} outstanding frames - will process them")
                        else:
                            log_message("DEBUG", "[LOCAL_WHISPER] Key released with no remaining frames to process")
                    previous_ignore_input = current_ignore_input
                    
                    if current_ignore_input and not process_remaining_frames:
                        # Skip audio processing but keep reading to prevent buffer overflow
                        stream.read(CHUNK, exception_on_overflow=False)
                        audio_buffer = []  # Clear buffer when ignoring
                        frame_count = 0
                        time.sleep(0.01)
                        continue
                    
                    # Read audio data
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.float32)
                    
                    # Debug raw audio data occasionally
                    if frame_count == 0:  # Log first frame details
                        log_message("DEBUG", f"[LOCAL_WHISPER] Raw audio: bytes={len(data)}, samples={len(audio_array)}, dtype={audio_array.dtype}")
                        log_message("DEBUG", f"[LOCAL_WHISPER] Sample values: min={np.min(audio_array):.6f}, max={np.max(audio_array):.6f}, mean={np.mean(audio_array):.6f}")
                    
                    audio_buffer.extend(audio_array)
                    frame_count += 1
                    
                    # Check if we're in tap-to-talk mode and if tap-to-talk is currently active
                    from .state import get_shared_state
                    shared_state = get_shared_state()
                    asr_mode = getattr(shared_state, 'asr_mode', 'auto-input')
                    
                    if asr_mode == 'tap-to-talk':
                        # In tap-to-talk mode, active when ignore_input is False
                        with _ignore_input_lock:
                            is_tap_to_talk_active = not _ignore_input
                    else:
                        # In continuous/auto-input modes, never use tap-to-talk logic
                        is_tap_to_talk_active = False
                    
                    # Increment outstanding frames counter for each PyAudio frame during active tap-to-talk
                    if is_tap_to_talk_active and not process_remaining_frames:
                        with _tap_to_talk_counter_lock:
                            _tap_to_talk_outstanding_frames += 1
                            log_message("DEBUG", f"[LOCAL_WHISPER] Tap-to-talk outstanding frames (+1): {_tap_to_talk_outstanding_frames}")
                    
                    # Process when we have enough audio accumulated for transcription window
                    if transitioned_to_released and frame_count < FRAMES_TO_ACCUMULATE:
                        # Tap-to-talk key released - pad partial frames to minimum processing size
                        if audio_buffer:
                            log_message("INFO", f"[LOCAL_WHISPER] Tap-to-talk released - processing partial segment ({frame_count}/{FRAMES_TO_ACCUMULATE} frames)")
                            
                            # Pad to the configured transcription window duration for consistency with other segments  
                            current_samples = len(audio_buffer)
                            min_samples = FRAMES_TO_ACCUMULATE * CHUNK
                            if current_samples < min_samples:
                                padding_needed = min_samples - current_samples
                                audio_buffer.extend([0.0] * padding_needed)
                                log_message("DEBUG", f"[LOCAL_WHISPER] Padded {current_samples} samples to {len(audio_buffer)} samples ({len(audio_buffer)/RATE:.1f}s)")
                            
                            # Force processing by setting frame_count to threshold
                            frame_count = FRAMES_TO_ACCUMULATE
                    
                    if frame_count >= FRAMES_TO_ACCUMULATE:
                        # Convert to numpy array and normalize
                        audio_np = np.array(audio_buffer, dtype=np.float32)
                        
                        # Calculate energy level for debugging
                        energy_level = np.sqrt(np.mean(audio_np ** 2))  # RMS energy
                        max_amplitude = np.max(np.abs(audio_np))  # Peak amplitude
                        
                        # Debug audio format
                        log_message("DEBUG", f"[LOCAL_WHISPER] Audio format: dtype={audio_np.dtype}, shape={audio_np.shape}, min={np.min(audio_np):.4f}, max={np.max(audio_np):.4f}")
                        log_message("DEBUG", f"[LOCAL_WHISPER] Audio segment: energy={energy_level:.4f}, peak={max_amplitude:.4f}, samples={len(audio_np)}")
                        
                        # Whisper expects 16kHz mono float32 with values roughly in [-1, 1] range
                        # Let's verify and potentially normalize
                        if np.max(np.abs(audio_np)) > 1.0:
                            log_message("WARNING", "[LOCAL_WHISPER] Audio clipping detected, normalizing...")
                            audio_np = audio_np / np.max(np.abs(audio_np))
                        
                        # Check if this is tap-to-talk mode and microphone is active
                        # In tap-to-talk, transcribe everything since user is intentionally speaking  
                        with _ignore_input_lock:
                            current_ignore_state = _ignore_input
                        log_message("DEBUG", f"[LOCAL_WHISPER] ignore_input={current_ignore_state}, is_tap_to_talk_active={is_tap_to_talk_active}, energy={energy_level:.4f}")
                        
                        # Convert speech_recognition energy threshold (for 16-bit audio) to RMS energy threshold (for float32 audio)
                        # speech_recognition uses energy = sum(audio²) for 16-bit integers (-32768 to 32767)
                        # We use RMS energy = sqrt(mean(audio²)) for float32 (-1.0 to 1.0) 
                        # Rough conversion: RMS threshold ≈ sqrt(sr_threshold / (32768² * samples_per_frame))
                        sr_energy_threshold = engine.recognizer.energy_threshold
                        samples_per_transcription_window = int(RATE * TRANSCRIPTION_WINDOW_DURATION)  
                        rms_energy_threshold = math.sqrt(sr_energy_threshold / (32768.0 ** 2 * samples_per_transcription_window))
                        
                        if is_tap_to_talk_active or process_remaining_frames or energy_level > rms_energy_threshold:
                            # Ensure audio segment meets minimum size requirements for whisper
                            min_samples_required = FRAMES_TO_ACCUMULATE * CHUNK
                            if len(audio_np) < min_samples_required:
                                padding_needed = min_samples_required - len(audio_np)
                                audio_np = np.concatenate([audio_np, np.zeros(padding_needed, dtype=np.float32)])
                                log_message("DEBUG", f"[LOCAL_WHISPER] Padded audio from {len(audio_np)-padding_needed} to {len(audio_np)} samples ({len(audio_np)/RATE:.1f}s)")
                            
                            if is_tap_to_talk_active:
                                log_message("DEBUG", f"[LOCAL_WHISPER] Queueing audio segment (tap-to-talk active): shape={audio_np.shape}, energy={energy_level:.4f}")
                            elif process_remaining_frames:
                                log_message("DEBUG", f"[LOCAL_WHISPER] Queueing audio segment (remaining frames): shape={audio_np.shape}, energy={energy_level:.4f}")
                            else:
                                log_message("DEBUG", f"[LOCAL_WHISPER] Queueing audio segment (energy threshold): shape={audio_np.shape}, energy={energy_level:.4f}")
                            try:
                                # Queue audio for transcription (non-blocking) - include frame count for outstanding frames tracking
                                audio_queue.put_nowait((audio_np, energy_level, frame_count))
                                
                                # Note: Outstanding frames are tracked per individual PyAudio frame, not per segment
                            except queue.Full:
                                log_message("WARNING", "[LOCAL_WHISPER] Transcription queue full, dropping audio segment")
                        else:
                            log_message("DEBUG", f"[LOCAL_WHISPER] Skipping transcription - energy too low ({energy_level:.4f})")
                        
                        # Reset buffer for next segment
                        audio_buffer = []
                        frame_count = 0
                        
                        # Clear the remaining frames flag after processing
                        if process_remaining_frames:
                            process_remaining_frames = False
                            log_message("DEBUG", "[LOCAL_WHISPER] Finished processing remaining frames")
                    
                except Exception as e:
                    if engine.is_active:
                        log_message("ERROR", f"[LOCAL_WHISPER] Chunk processing error: {e}")
                    time.sleep(0.01)
                    
        except Exception as e:
            log_message("ERROR", f"[LOCAL_WHISPER] Streaming error: {e}")
        finally:
            try:
                # Signal transcription worker to stop
                audio_queue.put_nowait(None)  # Shutdown signal
                
                # Wait for transcription thread to finish (with timeout)
                if transcription_thread.is_alive():
                    transcription_thread.join(timeout=2.0)
                    if transcription_thread.is_alive():
                        log_message("WARNING", "[LOCAL_WHISPER] Transcription thread did not shut down gracefully")
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                log_message("INFO", "[LOCAL_WHISPER] Stopped streaming audio capture and transcription thread")
            except Exception:
                pass


def check_asr_provider_accessibility() -> Dict[str, Dict[str, Any]]:
    """Check which ASR providers are accessible based on API keys and environment"""
    accessible = {}
    
    # Google (free)
    accessible["google"] = {
        "available": True,
        "note": "Free, no API key required"
    }
    
    # Google Cloud
    accessible["gcloud"] = {
        "available": bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCLOUD_SERVICE_ACCOUNT_JSON")),
        "note": "Requires GOOGLE_APPLICATION_CREDENTIALS or GCLOUD_SERVICE_ACCOUNT_JSON"
    }
    
    # AssemblyAI
    accessible["assemblyai"] = {
        "available": bool(os.environ.get("ASSEMBLYAI_API_KEY")),
        "note": "Requires ASSEMBLYAI_API_KEY environment variable"
    }
    
    # Deepgram
    accessible["deepgram"] = {
        "available": bool(os.environ.get("DEEPGRAM_API_KEY")),
        "note": "Requires DEEPGRAM_API_KEY environment variable"
    }
    
    # Houndify
    accessible["houndify"] = {
        "available": bool(os.environ.get("HOUNDIFY_CLIENT_ID") and os.environ.get("HOUNDIFY_CLIENT_KEY")),
        "note": "Requires HOUNDIFY_CLIENT_ID and HOUNDIFY_CLIENT_KEY"
    }
    
    # AWS Transcribe - Check using boto3's credential chain
    aws_available = False
    aws_note = "Requires AWS credentials (env vars, ~/.aws/credentials, or IAM role)"
    try:
        import boto3
        # Try to create a session to check if credentials are available
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is not None:
            aws_available = True
            aws_note = "AWS credentials detected"
    except ImportError:
        aws_note = "Requires boto3 package (pip install boto3)"
    except Exception:
        pass
    
    accessible["aws"] = {
        "available": aws_available,
        "note": aws_note
    }
    
    # Microsoft Bing
    accessible["bing"] = {
        "available": bool(os.environ.get("BING_KEY")),
        "note": "Requires BING_KEY environment variable"
    }
    
    # local_whisper - Smart backend selection (PyWhisperCpp CoreML on Apple Silicon, faster-whisper elsewhere)
    whisper_backend_info = _get_local_whisper_backend_info()

    # Check if model is cached
    whisper_note = whisper_backend_info["note"]
    if whisper_backend_info["available"]:
        from .models import check_model_cached
        model_name = os.environ.get('WHISPER_MODEL', 'small')
        is_cached = check_model_cached('local_whisper', model_name)
        if is_cached:
            cache_status = " [cached]"
        else:
            cache_status = " [needs download]"

        if whisper_note:
            whisper_note += cache_status
        else:
            whisper_note = f"Model: {model_name}{cache_status}"

    accessible["local_whisper"] = {
        "available": whisper_backend_info["available"],
        "note": whisper_note
    }
    
    return accessible


def _get_local_whisper_backend_info():
    """Centralized function to determine optimal local whisper backend and availability."""
    import platform
    
    # Detect Apple Silicon specifically
    is_apple_silicon = (
        platform.system() == 'Darwin' and 
        platform.machine() == 'arm64'
    )
    
    # Check PyWhisperCpp CoreML availability on Apple Silicon
    has_pywhispercpp = False
    if is_apple_silicon:
        try:
            from pywhispercpp.model import Model  # noqa: F401
            has_pywhispercpp = True
        except ImportError:
            pass
    
    # Check faster-whisper availability (fallback)
    has_faster_whisper = False
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        has_faster_whisper = True
    except ImportError:
        pass
    
    # Determine best backend and create info
    if is_apple_silicon and has_pywhispercpp:
        return {
            "available": True,
            "note": None,
            "use_pywhispercpp": True,
            "is_apple_silicon": is_apple_silicon
        }
    elif has_faster_whisper:
        note = None
        if is_apple_silicon and not has_pywhispercpp:
            note = "On Apple Silicon consider pywhispercpp for CoreML support: WHISPER_COREML=1 pip install pywhispercpp"

        return {
            "available": True,
            "note": note,
            "use_pywhispercpp": False,
            "is_apple_silicon": is_apple_silicon
        }
    else:
        if is_apple_silicon:
            note = "Local Whisper requires pywhispercpp (or faster-whisper which would be CPU limited): WHISPER_COREML=1 pip install pywhispercpp"
        else:
            note = "Local Whisper requires faster-whisper: pip install faster-whisper)"
            
        return {
            "available": False,
            "note": note,
            "use_pywhispercpp": False,
            "is_apple_silicon": is_apple_silicon
        }


def select_best_asr_provider(preferred=None, excluded_providers=None) -> str:
    """Select the best available ASR provider with thorough validation of credentials and configuration."""
    accessible = check_asr_provider_accessibility()
    excluded_providers = excluded_providers or set()

    # Check if preferred provider is accessible, properly configured, and not excluded
    if preferred and preferred in accessible and preferred not in excluded_providers:
        if not accessible[preferred]['available'] and accessible[preferred]["note"]:
            print(accessible[preferred]["note"])
        if accessible[preferred]['available']:
            # Actually validate the provider including imports and configuration
            imports_ok, import_error = check_provider_imports(preferred, requested_provider=preferred)
            if imports_ok:
                log_message("INFO", f"Using preferred ASR provider: {preferred}")
                return preferred
            else:
                log_message("WARNING", f"ASR provider {preferred} validation failed: {import_error}")
                # Continue to fallback logic
    
    # Get all accessible providers except google (free) and excluded providers
    available_providers = [
        provider for provider, info in sorted(accessible.items())
        if info['available'] and provider != 'google' and provider not in excluded_providers
    ]
    
    # Validate each provider thoroughly before selecting
    for provider in available_providers:
        imports_ok, import_error = check_provider_imports(provider, requested_provider=provider)
        if imports_ok:
            log_message("INFO", f"Selected ASR provider: {provider} (first validated)")
            return provider
        else:
            log_message("WARNING", f"ASR provider {provider} failed validation: {import_error}, trying next")
    
    # Try local_whisper as ultimate fallback before google (if available and not excluded)
    if 'local_whisper' not in excluded_providers and accessible.get('local_whisper', {}).get('available'):
        imports_ok, import_error = check_provider_imports('local_whisper', requested_provider='local_whisper')
        if imports_ok:
            log_message("INFO", "Falling back to local_whisper ASR provider")
            return 'local_whisper'
        else:
            log_message("WARNING", f"local_whisper ASR provider failed validation: {import_error}")

    # Fall back to google if available and not excluded
    if 'google' not in excluded_providers and accessible.get('google', {}).get('available'):
        imports_ok, import_error = check_provider_imports('google', requested_provider='google')
        if imports_ok:
            log_message("INFO", "Falling back to Google free ASR provider")
            return 'google'
        else:
            log_message("WARNING", f"Google ASR provider failed validation: {import_error}")

    # Fall back to google_free if google is excluded but google_free isn't
    if 'google_free' not in excluded_providers:
        imports_ok, import_error = check_provider_imports('google_free', requested_provider='google_free')
        if imports_ok:
            log_message("INFO", "Falling back to Google free ASR provider (google_free)")
            return 'google_free'
        else:
            log_message("WARNING", f"Google free ASR provider failed validation: {import_error}")

    # Last resort: return google_free anyway (it should always work)
    log_message("WARNING", "No fully validated ASR providers available, defaulting to google_free")
    return 'google_free'


# Provider registry
PROVIDERS = {
    'assemblyai': AssemblyAIProvider,
    'aws': AWSTranscribeProvider,
    'azure': AzureSpeechProvider,
    'deepgram': DeepgramProvider,
    'local_whisper': FasterWhisperProvider,
    'gcloud': GoogleCloudProvider,
    'google': GoogleCloudProvider,
    'google_free': GoogleFreeProvider,
    'houndify': HoundifyProvider,
    'microsoft': AzureSpeechProvider,
    'openai': WhisperProvider,
    'whisper': WhisperProvider,
}


class DictationEngine:
    """Simplified dictation engine using provider abstraction"""
    
    def __init__(self, text_callback: Callable[[str], None], 
                 energy_threshold: int = 4000,
                 pause_threshold: float = None,
                 phrase_time_limit: Optional[float] = 5.0,
                 partial_callback: Optional[Callable[[str], None]] = None,
                 provider_config: Optional[ASRConfig] = None):
        
        self.recognizer = sr.Recognizer()
        # Use smaller chunk size for AWS to avoid buffer issues
        chunk_size = 512 if (provider_config and provider_config.provider == 'aws') else 1024
        self.microphone = sr.Microphone(sample_rate=16000, chunk_size=chunk_size)
        self.text_callback = text_callback
        self.partial_callback = partial_callback
        
        # Thread-safe state management
        self._state_lock = threading.RLock()  # Use RLock for nested access
        self._is_active = False
        self._is_stopped = False  # Track if already stopped
        
        self.audio_queue = queue.Queue()
        self._stop_lock = threading.Lock()  # Prevent race condition in stop()
        
        # Configure recognizer
        self.recognizer.energy_threshold = energy_threshold
        # Use unified setting, converting from ms to seconds for speech_recognition
        if pause_threshold is None:
            pause_threshold = DEFAULT_MIN_SILENCE_MS / 1000.0  # Convert to seconds
        self.recognizer.pause_threshold = pause_threshold
        self.phrase_time_limit = phrase_time_limit
        
        # Thread management
        self.listener_thread = None
        self.processor_thread = None
        self.stop_event = threading.Event()
        
        # Recognition state tracking
        self.last_recognition_time = 0
        self.recognition_active = False
        
        # Pause/resume support
        self.is_paused = False
        self.pause_lock = threading.Lock()
        
        # Provider configuration
        self.provider_config = provider_config or ASRConfig(provider='google_free')
        self.provider = None
        self._init_provider()
    
    @property
    def is_active(self):
        """Thread-safe getter for is_active"""
        with self._state_lock:
            return self._is_active
    
    @is_active.setter
    def is_active(self, value):
        """Thread-safe setter for is_active"""
        with self._state_lock:
            self._is_active = value
    
    @property
    def is_stopped(self):
        """Thread-safe getter for is_stopped"""
        with self._state_lock:
            return self._is_stopped
    
    @is_stopped.setter
    def is_stopped(self, value):
        """Thread-safe setter for is_stopped"""
        with self._state_lock:
            self._is_stopped = value
        
    def _init_provider(self):
        """Initialize the ASR provider"""
        provider_class = PROVIDERS.get(self.provider_config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.provider_config.provider}")
        
        self.provider = provider_class(self.provider_config)
        
        # Validate provider
        success, error = self.provider.validate()
        if not success:
            raise Exception(f"Provider validation failed: {error}")
        
        log_message("INFO", f"Initialized {self.provider_config.provider} provider")
    
    def calibrate(self, duration: float = 1.0):
        """Calibrate for ambient noise"""
        try:
            with self.microphone as source:
                log_message("INFO", f"Calibrating for ambient noise... ({duration}s)")
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                log_message("INFO", f"Calibration complete. Energy threshold: {self.recognizer.energy_threshold}")
        except Exception as e:
            log_message("ERROR", f"Failed to calibrate: {e}")
    
    def start(self):
        """Start dictation"""
        if self.is_active:
            return
        
        self.is_active = True
        self.stop_event.clear()
        
        # Calibrate before starting
        self.calibrate()
        
        # Choose streaming or phrase-based mode
        if self.partial_callback and self.provider.supports_streaming:
            # Use streaming mode
            log_message("INFO", f"Using streaming mode for {self.provider_config.provider}")
            self.listener_thread = threading.Thread(
                target=self._streaming_worker, 
                daemon=True
            )
            self.listener_thread.start()
        else:
            # Use phrase-based mode
            log_message("INFO", f"Using phrase-based mode for {self.provider_config.provider} (partial_callback={self.partial_callback is not None}, supports_streaming={self.provider.supports_streaming})")
            self.listener_thread = threading.Thread(
                target=self._phrase_worker,
                daemon=True
            )
            self.listener_thread.start()
            
            self.processor_thread = threading.Thread(
                target=self._process_worker,
                daemon=True
            )
            self.processor_thread.start()
        
        log_message("INFO", f"Started dictation with {self.provider_config.provider}")
    
    def stop(self):
        """Stop dictation"""
        with self._stop_lock:  # Prevent concurrent stop() calls
            if self.is_stopped:
                log_message("DEBUG", "Dictation engine already fully stopped, skipping")
                return
                
            if not self.is_active:
                log_message("DEBUG", "Dictation engine already inactive, skipping")
                self.is_stopped = True  # Mark as stopped even if inactive
                return
            
            log_message("INFO", "Stopping dictation engine...")
            self.is_active = False
            self.stop_event.set()
            
            # Wait for threads outside the lock to avoid deadlock
            listener_thread = self.listener_thread
            processor_thread = self.processor_thread
            self.listener_thread = None
            self.processor_thread = None
        
        # Join threads outside the lock
        if listener_thread:
            listener_thread.join(timeout=1.0)
        if processor_thread:
            processor_thread.join(timeout=1.0)
        
        # Note: We don't manually close the microphone stream here because
        # it's managed by the context manager in AudioCaptureThread and
        # _phrase_worker. Manually closing it here causes a double-free error.
        
        with self._stop_lock:
            self.is_stopped = True
        log_message("INFO", "Stopped dictation engine")
    
    def _streaming_worker(self):
        """Worker for streaming providers"""
        try:
            log_message("INFO", f"Starting streaming worker for {self.provider_config.provider}")
            self.provider.stream(self, self.microphone)
        except Exception as e:
            log_message("ERROR", f"Streaming error: {e}")
            import traceback
            log_message("ERROR", f"Traceback: {traceback.format_exc()}")
    
    def _phrase_worker(self):
        """Worker for phrase-based recognition"""
        with self.microphone as source:
            while self.is_active and not self.stop_event.is_set():
                try:
                    # Check ignore status before capture
                    with _ignore_input_lock:
                        ignore_status = _ignore_input

                    log_message("DEBUG", f"[ASR PHRASE] About to listen, ignore_input={ignore_status}")
                    audio = self.recognizer.listen(
                        source,
                        timeout=0.5,
                        phrase_time_limit=self.phrase_time_limit
                    )

                    # Check ignore status after capture (might have changed during listen)
                    with _ignore_input_lock:
                        ignore_status_after = _ignore_input

                    if ignore_status_after:
                        log_message("DEBUG", "[ASR PHRASE] Audio captured but ignoring due to ignore_input=True - discarding")
                        continue

                    self.audio_queue.put(audio)
                    log_message("DEBUG", f"[ASR PHRASE] Captured and queued audio phrase (queue size: {self.audio_queue.qsize()})")
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    if self.is_active:
                        log_message("ERROR", f"Error in phrase worker: {e}")
                    break
    
    def _process_worker(self):
        """Process audio from queue"""
        while self.is_active and not self.stop_event.is_set():
            try:
                log_message("DEBUG", f"[ASR PROCESS] Waiting for audio from queue (current size: {self.audio_queue.qsize()})")
                audio = self.audio_queue.get(timeout=0.5)
                log_message("DEBUG", f"[ASR PROCESS] Got audio from queue (remaining size: {self.audio_queue.qsize()})")
                
                # Skip processing if paused
                with self.pause_lock:
                    if self.is_paused:
                        log_message("DEBUG", "[ASR PROCESS] Skipping audio processing - ASR is paused")
                        continue
                
                # Skip processing if ignoring input (e.g., during TTS)
                with _ignore_input_lock:
                    if _ignore_input:
                        log_message("DEBUG", "[ASR PROCESS] Skipping audio processing - ASR is ignoring input")
                        continue
                
                # Set recognition active
                self.recognition_active = True
                log_message("DEBUG", "[ASR PROCESS] Starting recognition...")
                
                # Recognize using provider
                try:
                    text = self.provider.recognize(audio)
                    log_message("DEBUG", f"[ASR PROCESS] Recognition completed, text: '{text}'")
                    
                    # Process and callback
                    processed = self._process_dictation_text(text)
                    if processed and self.text_callback:
                        log_message("DEBUG", f"[ASR PROCESS] Calling text callback with: '{processed}'")
                        self.text_callback(processed)
                    
                    log_message("INFO", f"[ASR] Recognized: '{text}'")
                    self.last_recognition_time = time.time()
                    
                except sr.UnknownValueError:
                    log_message("DEBUG", "Could not understand audio")
                    self.recognition_active = False
                except Exception as e:
                    log_message("ERROR", f"Recognition error: {e}")
                    self.recognition_active = False
                    
            except queue.Empty:
                continue
    
    def _process_dictation_text(self, text: str) -> str:
        """Process recognized text for better dictation experience"""
        replacements = [
            (" period", "."), (" comma", ","), (" question mark", "?"),
            (" exclamation mark", "!"), (" exclamation point", "!"),
            (" new line", "\n"), (" new paragraph", "\n\n")
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
    
    def on_final_transcript(self, text: str):
        """Handle final transcript from streaming providers"""
        with self.pause_lock:
            if self.is_paused:
                log_message("DEBUG", f"Ignoring final transcript - ASR is paused: '{text}'")
                return
        
        processed = self._process_dictation_text(text)
        if processed and self.text_callback:
            self.text_callback(processed)
        log_message("INFO", f"Recognized (streaming): '{text}'")
        self.last_recognition_time = time.time()
    
    def on_partial_transcript(self, text: str):
        """Handle partial transcript from streaming providers"""
        with self.pause_lock:
            if self.is_paused:
                log_message("DEBUG", f"Ignoring partial transcript - ASR is paused: '{text}'")
                return
        
        if self.partial_callback:
            self.partial_callback(text)
        self.last_recognition_time = time.time()
    
    def pause(self):
        """Pause recognition without stopping threads"""
        with self.pause_lock:
            if not self.is_paused:
                self.is_paused = True
                log_message("INFO", "ASR paused")
                # Clear the audio queue to prevent processing stale audio
                try:
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
    
    def resume(self):
        """Resume recognition"""
        with self.pause_lock:
            if self.is_paused:
                self.is_paused = False
                log_message("INFO", "ASR resumed")
    
    def is_recognizing(self) -> bool:
        """Check if actively recognizing speech"""
        # Don't consider as recognizing if paused
        with self.pause_lock:
            if self.is_paused:
                return False

        result = self.recognition_active and (time.time() - self.last_recognition_time < 2.0)
        # log_message("DEBUG", f"is_recognizing: recognition_active={self.recognition_active}, time_since_last={time.time() - self.last_recognition_time:.1f}s, result={result}")
        return result

    def process_audio_file(self, audio_file_path: str, delay: float = 0.1):
        """This function exists just for end to end testing. It allowes delayed processing of an audio file in leu of a mic input"""
        import threading

        def _process_file():
            """Internal function to process the file after delay"""
            log_message("INFO", f"Processing audio file: {audio_file_path}")

            try:
                with sr.AudioFile(audio_file_path) as source:
                    # Read the entire audio file
                    audio = self.recognizer.record(source)
                    log_message("DEBUG", f"Audio file loaded, processing with {self.provider_config.provider}")

                    # Process using the provider's recognize method
                    text = self.provider.recognize(audio)

                    if text:
                        log_message("INFO", f"Transcribed from file: {text}")
                        # Call the text callback with the transcribed text
                        if self.text_callback:
                            self.text_callback(text)
                    else:
                        log_message("WARNING", "No transcription result from audio file")

            except Exception as e:
                log_message("ERROR", f"Error processing audio file: {e}")
                import traceback
                log_message("ERROR", f"Traceback: {traceback.format_exc()}")

        # Schedule processing after delay using Timer (non-blocking)
        log_message("INFO", f"Scheduled audio file processing in {delay}s: {audio_file_path}")
        timer = threading.Timer(delay, _process_file)
        timer.daemon = True
        timer.start()


# Simplified configuration functions
def create_config_from_args(args) -> ASRConfig:
    """Create ASR config from command line arguments"""
    # Use provider selection if not specified
    provider = args.asr_provider
    if provider is None:
        provider = select_best_asr_provider()
        print(f"Auto-selected ASR provider: {provider}")
    
    config = ASRConfig(
        provider=provider,
        language=args.language
    )
    
    # Provider-specific settings
    if provider == 'assemblyai':
        config.api_key = args.assemblyai_api_key
    elif provider == 'houndify':
        config.client_id = args.houndify_client_id
        config.client_key = args.houndify_client_key
        config.user_id = args.houndify_user_id
    elif provider == 'aws':
        config.region = args.aws_region
    elif provider == 'whisper':
        config.api_key = args.openai_api_key
        config.model = args.whisper_model
    elif provider == 'azure':
        config.api_key = args.azure_speech_key
        config.region = args.azure_speech_region
    elif provider == 'deepgram':
        config.api_key = args.deepgram_api_key
        config.model = args.deepgram_model
    elif provider == 'local_whisper':
        config.model = args.whisper_model
    
    return config


def create_config_from_dict(data: dict) -> ASRConfig:
    """Create ASR config from dictionary"""
    return ASRConfig(
        provider=data.get('provider', 'google_free'),
        language=data.get('language', 'en-US'),
        region=data.get('region') or data.get('azure_speech_region'),
        api_key=data.get('api_key') or data.get('assemblyai_api_key') or data.get('openai_api_key') or data.get('azure_speech_key') or data.get('deepgram_api_key'),
        client_id=data.get('client_id') or data.get('houndify_client_id'),
        client_key=data.get('client_key') or data.get('houndify_client_key'),
        user_id=data.get('user_id') or data.get('houndify_user_id'),
        model=data.get('model') or data.get('whisper_model') or data.get('deepgram_model')
    )


# Global engine management (simplified)
_engine: Optional[DictationEngine] = None
_ignore_input: bool = False  # Flag to temporarily ignore ASR input
_ignore_input_lock = threading.Lock()  # Thread safety for _ignore_input
_tap_to_talk_outstanding_frames: int = 0  # Counter for outstanding audio frames during tap-to-talk
_tap_to_talk_counter_lock = threading.Lock()  # Thread safety for frames counter


def start_dictation(text_callback, partial_callback=None, config: Optional[ASRConfig] = None):
    """Start global dictation engine"""
    global _engine
    
    if _engine is None:
        _engine = DictationEngine(
            text_callback, 
            partial_callback=partial_callback,
            provider_config=config
        )
    else:
        # Update callbacks
        _engine.text_callback = text_callback
        _engine.partial_callback = partial_callback
    
    if not _engine.is_active:
        _engine.start()
    
    return _engine


def stop_dictation():
    """Stop global dictation engine"""
    global _engine
    if _engine:
        try:
            _engine.stop()
        except Exception as e:
            log_message("ERROR", f"Error stopping engine: {e}")
        finally:
            _engine = None


def pause_dictation():
    """Pause global dictation engine"""
    if _engine:
        _engine.pause()


def resume_dictation():
    """Resume global dictation engine"""
    if _engine:
        _engine.resume()


def is_recognizing() -> bool:
    """Check if actively recognizing speech"""
    result = _engine is not None and _engine.is_recognizing()
    # log_message("DEBUG", f"Module is_recognizing: engine exists={_engine is not None}, result={result}")
    return result


# Backward compatibility functions for Talkito
def configure_asr_from_args(args) -> bool:
    """Backward compatibility function for talkito.py"""
    config = ASRConfig(
        provider=args.asr_provider,
        language=args.asr_language if args.asr_language else 'en-US',
        model=args.asr_model,
    )
    try:
        # Validate the configuration
        provider_class = PROVIDERS.get(config.provider)
        if not provider_class:
            log_message("WARNING", f"Unknown ASR provider: {config.provider}, falling back to best available")
            # Fall back to best available provider
            fallback_provider = select_best_asr_provider()
            config.provider = fallback_provider
            provider_class = PROVIDERS.get(fallback_provider)
            if not provider_class:
                return False
        
        provider = provider_class(config)
        success, error = provider.validate()
        if not success:
            log_message("WARNING", f"ASR provider {config.provider} validation failed: {error}")
            print(error)
            # Fall back to best available provider
            fallback_provider = select_best_asr_provider()
            if fallback_provider != config.provider:
                log_message("INFO", f"Falling back to ASR provider: {fallback_provider}")
                config.provider = fallback_provider
                provider_class = PROVIDERS.get(fallback_provider)
                if not provider_class:
                    return False
                provider = provider_class(config)
                success, error = provider.validate()
                if not success:
                    log_message("ERROR", f"Fallback ASR provider {fallback_provider} also failed: {error}")
                    return False
            else:
                return False
            
        # Store config for later use
        global _stored_config
        _stored_config = config
        
        # Update shared state with the actually working provider
        from .state import get_shared_state
        shared_state = get_shared_state()
        shared_state.set_asr_config(provider=config.provider, language=config.language, model=config.model)
        
        return True
    except Exception as e:
        log_message("ERROR", f"Error configuring ASR: {e}")
        print(f"Error configuring ASR: {e}")
        return False

# Global stored config for backward compatibility
_stored_config: Optional[ASRConfig] = None

# Global storage for preloaded local whisper model
_local_whisper_model = None

# Local whisper model loading state tracking (similar to TTS local model caching)
_local_whisper_model_loading = False
_local_whisper_model_error = None  
_local_whisper_model_cache_lock = threading.Lock()

# Monkey patch PyWhisperCpp Model to prevent destructor errors during shutdown
try:
    import pywhispercpp.model
    original_del = getattr(pywhispercpp.model.Model, '__del__', None)
    
    def safe_del(self):
        try:
            if original_del:
                original_del(self)
        except (TypeError, AttributeError):
            # Silently ignore destructor errors during shutdown when C++ cleanup functions become None
            pass
    
    pywhispercpp.model.Model.__del__ = safe_del
except ImportError:
    # PyWhisperCpp not installed, skip monkey patch
    pass

# Override the original start_dictation to handle backward compatibility
_original_start_dictation = start_dictation

def start_dictation(text_callback, partial_callback=None, config: Optional[ASRConfig] = None):
    """Backward compatible start_dictation"""
    # If no config provided and we have a stored config from configure_asr_from_dict
    if config is None and _stored_config is not None:
        config = _stored_config
    
    return _original_start_dictation(text_callback, partial_callback, config)


def set_ignore_input(ignore: bool):
    """Set whether to temporarily ignore ASR input"""
    global _ignore_input, _engine
    with _ignore_input_lock:
        _ignore_input = ignore
    
    # Clear the audio queue when starting to ignore input to prevent processing old audio
    if ignore and _engine and hasattr(_engine, 'audio_queue'):
        try:
            queue_size_before = _engine.audio_queue.qsize()
            cleared_count = 0
            # Drain the audio queue
            while not _engine.audio_queue.empty():
                try:
                    _engine.audio_queue.get_nowait()
                    cleared_count += 1
                except queue.Empty:
                    break
            log_message("DEBUG", f"[ASR IGNORE] Cleared {cleared_count} items from audio queue (was {queue_size_before}, now {_engine.audio_queue.qsize()})")
        except Exception as e:
            log_message("DEBUG", f"[ASR IGNORE] Could not clear ASR audio queue: {e}")
    
    log_message("DEBUG", f"ASR ignore_input set to: {ignore}")


def is_ignoring_input() -> bool:
    """Check if ASR is currently ignoring input"""
    with _ignore_input_lock:
        return _ignore_input


def is_dictation_active() -> bool:
    """Check if dictation engine is active"""
    return _engine is not None and _engine.is_active


def get_tap_to_talk_outstanding_frames() -> int:
    """Get count of outstanding audio frames during tap-to-talk session"""
    global _tap_to_talk_outstanding_frames
    with _tap_to_talk_counter_lock:
        return _tap_to_talk_outstanding_frames


def main():
    """Example usage"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr-provider', default=None,
                       choices=list(PROVIDERS.keys()),
                       help='ASR provider to use (default: auto-select best available)')
    parser.add_argument('--language', default='en-US')
    parser.add_argument('--assemblyai-api-key')
    parser.add_argument('--houndify-client-id')
    parser.add_argument('--houndify-client-key')
    parser.add_argument('--houndify-user-id', default='default_user')
    parser.add_argument('--aws-region', default='us-east-1')
    parser.add_argument('--openai-api-key', help='OpenAI API key for Whisper API mode')
    parser.add_argument('--whisper-model', default='base',
                       help='Whisper model name (local: base/small/medium/large, API: whisper-1)')
    parser.add_argument('--azure-speech-key', help='Azure Speech Services subscription key')
    parser.add_argument('--azure-speech-region', help='Azure Speech Services region (e.g., eastus)')
    parser.add_argument('--deepgram-api-key', help='Deepgram API key')
    parser.add_argument('--deepgram-model', default='nova-2',
                       help='Deepgram model name (e.g., nova-2, nova, base)')
    parser.add_argument('--whisper-model', default='small',
                       help='whisper model name for streaming ASR (e.g., medium, small, base, tiny)')
    # Legacy compatibility
    parser.add_argument('--faster-whisper-model', dest='whisper_model', 
                       help='(deprecated) use --whisper-model instead')

    # Logging options
    parser.add_argument('--log-file', type=str, default=None,
                       help='Enable debug logging to file (e.g., ~/.talk_asr.log)')
    
    args = parser.parse_args()
    
    # Set up logging using centralized logging
    if args.log_file:
        from .logs import setup_logging
        setup_logging(args.log_file)
        print(f"Logging to: {args.log_file}")
    
    # Create config
    config = create_config_from_args(args)
    
    # Print callbacks
    def print_text(text):
        # Clear any partial transcript line first
        print("\r" + " " * 80 + "\r", end='')  # Clear the line
        print(f">>> {text}")
    
    def print_partial(text):
        # Clear the line and print partial in brackets
        print(f"\r[{text}]" + " " * (80 - len(text) - 2), end='', flush=True)
    
    try:
        # Start dictation
        print(f"Starting {config.provider} dictation...")
        print("Speak into your microphone. Press Ctrl+C to stop.")
        print("-" * 50)
        engine = start_dictation(print_text, partial_callback=print_partial, config=config)
        
        # Run until interrupted or engine stops
        while engine and engine.is_active:
            time.sleep(0.1)
        
        # If we exited due to engine stopping (not KeyboardInterrupt)
        if engine and not engine.is_active:
            print("\nStopping due to error...")
            stop_dictation()
            
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_dictation()


def get_cached_local_whisper_model(timeout_seconds: float = 10.0):
    """Get the cached local whisper model, waiting up to timeout_seconds for it to load"""
    global _local_whisper_model, _local_whisper_model_loading, _local_whisper_model_error
    
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        with _local_whisper_model_cache_lock:
            # Check if we have an error
            if _local_whisper_model_error:
                log_message("ERROR", f"[LOCAL_WHISPER_CACHE] Model loading failed: {_local_whisper_model_error}")
                return None
            
            # Check if model is ready
            if _local_whisper_model is not None:
                log_message("DEBUG", "[LOCAL_WHISPER_CACHE] Model retrieved from cache")
                return _local_whisper_model
            
            # If not loading and no model, return error - don't start new load
            if not _local_whisper_model_loading:
                log_message("ERROR", "[LOCAL_WHISPER_CACHE] No model available and none loading - preload_pywhisper_model() should be called first")
                return None
        
        # Brief sleep before checking again
        time.sleep(0.1)
    
    # Timeout reached
    log_message("WARNING", f"[LOCAL_WHISPER_CACHE] Timeout after {timeout_seconds}s waiting for model")
    return None


def preload_local_asr_model(model_name: str = None):
    """Start background preloading of local ASR model (following TTS pattern)."""
    global _local_whisper_model_loading
    import threading
    
    # Use default model if none specified
    if not model_name:
        model_name = os.environ.get('WHISPER_MODEL', 'small')
    
    # Get backend information
    backend_info = _get_local_whisper_backend_info()
    if not backend_info["available"]:
        log_message("INFO", f"[ASR_PRELOAD] Local whisper not available: {backend_info.get('note', 'Unknown reason')}")
        return
    
    use_pywhispercpp = backend_info["use_pywhispercpp"]
    
    with _local_whisper_model_cache_lock:
        # Skip if already loading or loaded
        if _local_whisper_model_loading:
            log_message("DEBUG", f"[ASR_PRELOAD] Model already loading, skipping preload for {model_name}")
            return
        
        if _local_whisper_model is not None:
            log_message("DEBUG", f"[ASR_PRELOAD] Model already cached, skipping preload for {model_name}")
            return
        
        # Check if model needs consent BEFORE starting background thread
        # This must happen in main thread where input() works (following TTS pattern)
        from .models import check_model_cached, ask_user_consent
        
        model_download_started = False
        if use_pywhispercpp:
            coreml_model_name = model_name
            provider_name = 'pywhispercpp'
            if not check_model_cached('pywhispercpp', coreml_model_name):
                if not ask_user_consent('PyWhisperCpp', coreml_model_name):
                    log_message("INFO", f"[ASR_PRELOAD] User declined download for PyWhisperCpp model '{coreml_model_name}'")
                    # Fall back to next best available provider
                    fallback_provider = select_best_asr_provider(excluded_providers={'local_whisper'})
                    print(f"Download declined. Falling back to {fallback_provider} ASR provider.")
                    
                    # Update shared state with fallback provider
                    try:
                        from .state import get_shared_state
                        shared_state = get_shared_state()
                        shared_state.set_asr_config(provider=fallback_provider, language='en-US', model=None)
                        log_message("INFO", f"[ASR_PRELOAD] Updated shared state to use fallback provider: {fallback_provider}")
                    except Exception as e:
                        log_message("WARNING", f"[ASR_PRELOAD] Could not update shared state with fallback: {e}")
                    
                    return
                model_download_started = True
        else:
            provider_name = 'local_whisper'
            if not check_model_cached('local_whisper', model_name):
                if not ask_user_consent('faster-whisper', model_name):
                    log_message("INFO", f"[ASR_PRELOAD] User declined download for faster-whisper model '{model_name}'")
                    # Fall back to next best available provider  
                    fallback_provider = select_best_asr_provider(excluded_providers={'local_whisper'})
                    print(f"Download declined. Falling back to {fallback_provider} ASR provider.")
                    
                    # Update shared state with fallback provider
                    try:
                        from .state import get_shared_state
                        shared_state = get_shared_state()
                        shared_state.set_asr_config(provider=fallback_provider, language='en-US', model=None)
                        log_message("INFO", f"[ASR_PRELOAD] Updated shared state to use fallback provider: {fallback_provider}")
                    except Exception as e:
                        log_message("WARNING", f"[ASR_PRELOAD] Could not update shared state with fallback: {e}")
                    
                    return
                model_download_started = True
        
        # Start background loading (consent already obtained)
        _local_whisper_model_loading = True
        _local_whisper_model_error = None
    
    # Start background loading thread
    thread = threading.Thread(target=_load_asr_model_background, args=(model_name,), daemon=True)
    thread.start()
    log_message("INFO", f"[ASR_PRELOAD] Started background loading of {provider_name} model: {model_name}")
    
    # Show confirmation message if download was started
    if model_download_started:
        print(f"Downloading {provider_name} model '{model_name}' in background. ASR will start automatically when ready.")


def _load_asr_model_background(model_name: str):
    """Load ASR model in background thread (following TTS pattern)."""
    global _local_whisper_model, _local_whisper_model_loading, _local_whisper_model_error
    
    try:
        log_message("INFO", f"[ASR_PRELOAD] Background loading model: {model_name}")
        
        # Get backend information
        backend_info = _get_local_whisper_backend_info()
        use_pywhispercpp = backend_info["use_pywhispercpp"]
        
        if use_pywhispercpp:
            # Load PyWhisperCpp model
            from pywhispercpp.model import Model as PyWhisperModel

            # Redirect stderr to suppress ggml_metal_free messages during model creation
            with open(os.devnull, 'w') as devnull:
                old_stderr = os.dup(2)
                os.dup2(devnull.fileno(), 2)
                try:
                    model = PyWhisperModel(
                        model_name,
                        n_threads=4,
                        print_progress=False,
                        print_realtime=False,
                        print_timestamps=False,
                        print_special=False,
                    )
                finally:
                    os.dup2(old_stderr, 2)
                    os.close(old_stderr)
                    
            # Register cleanup handler to suppress Metal deallocation messages on exit
            def cleanup_model():
                global _local_whisper_model
                if _local_whisper_model:
                    # Redirect stderr to suppress ggml_metal_free messages
                    with open(os.devnull, 'w') as devnull:
                        old_stderr = os.dup(2)
                        os.dup2(devnull.fileno(), 2)
                        try:
                            _local_whisper_model = None
                        finally:
                            os.dup2(old_stderr, 2)
                            os.close(old_stderr)
            atexit.register(cleanup_model)
        else:
            # Load faster-whisper model
            from faster_whisper import WhisperModel
            
            device = os.environ.get('WHISPER_DEVICE', 'cpu')
            compute_type = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')
            
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        
        with _local_whisper_model_cache_lock:
            _local_whisper_model = model
            _local_whisper_model_loading = False
            _local_whisper_model_error = None
            
        log_message("INFO", f"[ASR_PRELOAD] Model loaded successfully in background: {model_name}")
        
    except Exception as e:
        with _local_whisper_model_cache_lock:
            _local_whisper_model_loading = False
            _local_whisper_model_error = f"Model loading failed: {e}"
        log_message("ERROR", f"[ASR_PRELOAD] Background model loading failed: {e}")

if __name__ == "__main__":
    main()
