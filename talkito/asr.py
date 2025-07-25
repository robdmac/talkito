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

"""
asr.py - Automatic Speech Recognition module
Demonstrates a more DRY and maintainable approach to supporting multiple ASR providers
"""

# Suppress pkg_resources deprecation warnings from Google Cloud SDK dependencies
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google.rpc')

# Suppress absl and gRPC warnings
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging if used

import speech_recognition as sr
import threading
import queue
import argparse
import time
import tempfile
from typing import Optional, Callable, Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
from contextlib import contextmanager
from pathlib import Path

from .logs import log_message as _base_log_message

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
    if provider == 'google_free':
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
}


def check_provider_imports(provider: str) -> Tuple[bool, Optional[str]]:
    """Unified import checking for all providers"""
    # Providers that use built-in speech_recognition
    if provider in ['google_free']:
        return True, None
    
    # Special case: Whisper can use either local or API mode
    if provider == 'whisper':
        # Try local whisper first
        try:
            import whisper
            whisper.available_models()
            return True, None
        except ImportError:
            # Try OpenAI API
            try:
                import openai
                return True, None
            except ImportError:
                return False, "Neither local Whisper nor OpenAI library installed. Run: pip install openai-whisper or pip install openai"
        except Exception as e:
            # Whisper is installed but has issues
            return False, f"Whisper library error: {e}"
    
    # Special case: Google Cloud - verify client creation works
    if provider == 'gcloud':
        try:
            from google.cloud import speech
            client = speech.SpeechClient()
            return True, None
        except ImportError:
            return False, "Google Cloud Speech library not installed. Run: pip install google-cloud-speech"
        except Exception as e:
            return False, f"Google Cloud credentials invalid: {e}"
    
    # Standard import check for remaining providers
    module_info = REQUIRED_IMPORTS.get(provider)
    if not module_info:
        return True, None
    
    module_name, install_cmd = module_info
    try:
        importlib.import_module(module_name.split('.')[0])
        return True, None
    except ImportError:
        return False, f"{module_name} library not installed. Run: {install_cmd}"


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
                    except:
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
                        # Use smaller buffer size like the reference (256 frames = 512 bytes for 16-bit audio)
                        try:
                            audio_data = source.stream.read(512, exception_on_overflow=False)
                        except TypeError:
                            # Fallback for older PyAudio versions
                            audio_data = source.stream.read(512)
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
                except:
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
            from amazon_transcribe.client import TranscribeStreamingClient
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
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
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
                        print(f"\nDeepgram API key error. Please check your DEEPGRAM_API_KEY.")
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
    
    return accessible


def select_best_asr_provider() -> str:
    """Select the best available ASR provider based on accessibility and preferences.
    
    Order of preference:
    1. TALKITO_PREFERRED_ASR_PROVIDER from environment (if accessible)
    2. First accessible non-google provider (alphabetically)
    3. Google free as fallback
    """
    preferred = os.environ.get('TALKITO_PREFERRED_ASR_PROVIDER')
    accessible = check_asr_provider_accessibility()
    
    # Check if preferred provider is accessible
    if preferred and preferred in accessible and accessible[preferred]['available']:
        log_message("INFO", f"Using preferred ASR provider: {preferred}")
        return preferred
    
    # Get all accessible providers except google (free)
    available_providers = [
        provider for provider, info in sorted(accessible.items())
        if info['available'] and provider != 'google'
    ]
    
    # Use first available non-google provider
    if available_providers:
        provider = available_providers[0]
        log_message("INFO", f"Selected ASR provider: {provider} (first available)")
        return provider
    
    # Fall back to google if available
    if accessible.get('google', {}).get('available'):
        log_message("INFO", "Falling back to Google free ASR provider")
        return 'google'
    
    # No providers available
    log_message("WARNING", "No ASR providers available, defaulting to google")
    return 'google'


# Provider registry
PROVIDERS = {
    'assemblyai': AssemblyAIProvider,
    'aws': AWSTranscribeProvider,
    'azure': AzureSpeechProvider,
    'deepgram': DeepgramProvider,
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
                    audio = self.recognizer.listen(
                        source,
                        timeout=0.5,
                        phrase_time_limit=self.phrase_time_limit
                    )
                    self.audio_queue.put(audio)
                    log_message("DEBUG", "Captured audio phrase")
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
                audio = self.audio_queue.get(timeout=0.5)
                
                # Skip processing if paused
                with self.pause_lock:
                    if self.is_paused:
                        log_message("DEBUG", "Skipping audio processing - ASR is paused")
                        continue
                
                # Set recognition active
                self.recognition_active = True
                
                # Recognize using provider
                try:
                    text = self.provider.recognize(audio)
                    
                    # Process and callback
                    processed = self._process_dictation_text(text)
                    if processed and self.text_callback:
                        self.text_callback(processed)
                    
                    log_message("INFO", f"Recognized: '{text}'")
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
        log_message("DEBUG", f"is_recognizing: recognition_active={self.recognition_active}, time_since_last={time.time() - self.last_recognition_time:.1f}s, result={result}")
        return result


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
    # Don't clear stored config - it should persist for the session


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
    log_message("DEBUG", f"Module is_recognizing: engine exists={_engine is not None}, result={result}")
    return result


# Backward compatibility functions for Talkito
def configure_asr_from_dict(config_dict: dict) -> bool:
    """Backward compatibility function for talkito.py"""
    try:
        config = create_config_from_dict(config_dict)
        # Validate the configuration
        provider_class = PROVIDERS.get(config.provider)
        if not provider_class:
            return False
        
        provider = provider_class(config)
        success, error = provider.validate()
        if not success:
            print(error)
            return False
            
        # Store config for later use
        global _stored_config
        _stored_config = config
        return True
    except Exception as e:
        print(f"Error configuring ASR: {e}")
        return False


# Global stored config for backward compatibility
_stored_config: Optional[ASRConfig] = None


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
    global _ignore_input
    with _ignore_input_lock:
        _ignore_input = ignore
    log_message("DEBUG", f"ASR ignore_input set to: {ignore}")


def is_ignoring_input() -> bool:
    """Check if ASR is currently ignoring input"""
    with _ignore_input_lock:
        return _ignore_input


def is_dictation_active() -> bool:
    """Check if dictation engine is active"""
    return _engine is not None and _engine.is_active


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


if __name__ == "__main__":
    main()