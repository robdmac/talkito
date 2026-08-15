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
import atexit
import base64
import contextlib
import importlib.util
import io
import json
import queue
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from collections import deque
from typing import Optional, List, Tuple, Deque, Dict, Any, Callable
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

from .state import get_shared_state

# Import centralized logging utilities
try:
    from .logs import log_message as _base_log_message
    from .logs import emit_orcabot_notice, is_orcabot_mode
except ImportError:
    # Fallback for standalone execution
    def _base_log_message(level: str, message: str, logger_name: str = None):
        print(f"[{level}] {message}")
    def emit_orcabot_notice(message: str, level: str = "info", category: str = None):
        return False
    def is_orcabot_mode() -> bool:
        return False

# def patch_hf_hub_download():


def patch_phonemizer_espeak_api():
    """
    Make official phonemizer behave like the fork for loaders that call
    EspeakWrapper.set_data_path / set_library.
    """
    import os
    from importlib import import_module

    try:
        # Import the wrapper class
        wrapper = import_module('phonemizer.backend.espeak.wrapper')
        EspeakWrapper = wrapper.EspeakWrapper
    except ImportError:
        # phonemizer not installed - skip patching (only needed for KittenTTS/Kokoro)
        return

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


@dataclass
class SpeechItem:
    """Item in speech queue with text, timestamp, and metadata."""
    text: str
    original_text: str
    line_number: Optional[int] = None
    timestamp: Optional[datetime] = None
    start_time: Optional[float] = None  # Time when speech actually starts playing
    source: str = "output"  # "output", "error", etc.
    is_exception: bool = False  # True for exception patterns (questions, etc.) that should not be auto-skipped

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# Configuration constants
MIN_SPEAK_LENGTH = 2  # Minimum characters before speaking; 4 silently ate "ok", "yes", "no"
CACHE_SIZE = 10000  # Cache size for similarity checking
SIMILARITY_THRESHOLD = 0.85  # How similar text must be to be considered a repeat
DEBOUNCE_TIME = 0.5  # Seconds to wait before speaking rapidly changing text
SKIP_INTERJECTIONS = ["oh", "hmm", "um", "right", "okay"]  # Interjections to add when auto-skipping

# Compiled regex patterns for performance
RE_FILENAME_PATH = re.compile(r'(?:/?(?:[\w.-]+/)+)([\w.-]+)')
RE_FILE_EXTENSION = re.compile(r'(\w+)\.(\w+)')
RE_LIST_MARKERS = re.compile(r'^[-+]\s+', re.MULTILINE)
RE_MULTIPLE_ASTERISKS = re.compile(r'\*\*+')
RE_MULTIPLE_HASHES = re.compile(r'##+')
RE_MULTIPLE_QUESTIONS = re.compile(r'\?\?+')
RE_SINGLE_HASH = re.compile(r'#')
RE_DASH_WORD = re.compile(r'--(\w)')
RE_DATE_FORMAT = re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})')
RE_NEGATIVE_NUMBER = re.compile(r'(?<![a-zA-Z0-9])-(\d+)')
RE_NUMBER_MINUS_NUMBER = re.compile(r'(\d+)\s+-\s+(\d+)')
RE_WORD_DASH = re.compile(r'(\w+)\s+-\s+')
RE_NUMBER_RANGE = re.compile(r'(\d+)-(\d+)')
RE_PLUS_SIGN = re.compile(r'\+')
RE_DIVISION = re.compile(r'(\d+)\s+/\s+(\d+)')
RE_HTTP_URL_ALL = re.compile(r'https?://\S+')
RE_HTTP_URL = re.compile(r'(https?:)//([^\s]+)')
RE_SUSPICIOUS_QUERY = re.compile(
    r'\b(?:code|client_id|redirect_uri|response_type|code_challenge|code_challenge_method|state|scope)=[^\s]+',
    re.IGNORECASE
)
RE_PERCENT_ENCODE = re.compile(r'%[0-9A-Fa-f]{2}')
RE_SLASH_PATH = re.compile(r'(?<![.\w])/([a-zA-Z][\w-]+)')
RE_SLASH_OR = re.compile(r'(?<![.\w])(\w+)/(\w+)(?![.\w])')
RE_QUOTE_PREFIX = re.compile(r'^> ')
RE_CWD = re.compile(r'cwd')
RE_USAGE = re.compile(r'Usage:')
RE_TODOS = re.compile(r'Todos')
RE_ANGLE_BRACKETS = re.compile(r'<(\w+)>')
RE_PUNCT_PERIOD_SPACE = re.compile(r'([!?:;])\.\s')
RE_PUNCT_PERIOD_END = re.compile(r'([!?:;])\.$')
RE_MULTIPLE_PERIODS = re.compile(r'\.(\s+\.)+')
RE_GREATER_UNDERSCORE = re.compile(r'>_')
RE_NEWLINES = re.compile(r'\n+')
RE_FUNCTION_CALL = re.compile(r'(\w+)_(\w+)(?:_(\w+))*\(\)')
RE_UNDERSCORE_WORDS = re.compile(r'(\w+)_(\w+)(?:_(\w+))*')
RE_TRAILING_PARENS = re.compile(r' *\([^)]*\)$')
RE_MARKDOWN_BOLD = re.compile(r'\*\*([^*]+)\*\*')
RE_MARKDOWN_ITALIC = re.compile(r'\*([^*]+)\*')
RE_MARKDOWN_CODE = re.compile(r'`([^`]+)`')
RE_NON_SPEECH_CHARS = re.compile(r'[^a-zA-Z0-9 .,!?:\'-•☐▪▫■□◦‣⁃-]')
RE_MULTIPLE_SPACES = re.compile(r' +')
RE_HAS_LETTERS = re.compile(r'[a-zA-Z]')

# Global configuration
auto_skip_tts_enabled = False  # Whether to auto-skip long text
disable_tts = False  # Whether to disable TTS completely (for testing)
tts_provider = "system"  # Current TTS provider (system, openai, polly, azure, gcloud, elevenlabs, deepgram, etc.)
use_orcabot_playback = False  # Use Orcabot audio playback via HTTP
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
kittentts_model = os.environ.get('KITTENTTS_MODEL', 'KittenML/kitten-tts-mini-0.8')  # Default KittenTTS model
kittentts_voice = os.environ.get('KITTENTTS_VOICE', 'Jasper')  # Default KittenTTS voice
kokoro_language = os.environ.get('KOKORO_LANGUAGE', 'a')  # Default Kokoro language (American English)
kokoro_voice = os.environ.get('KOKORO_VOICE', 'af_heart')  # Default Kokoro voice
kokoro_speed = os.environ.get('KOKORO_SPEED', '1.0')  # Default Kokoro speed
# Only the quantized 2E backbone is shipped: it is the one NeuTTS configuration that runs faster
# than real time (RTF ~0.9 vs ~2.4 for the fp32 models) and the only one that supports streaming.
# The nano and fp32 variants live on the benchmark/tts-model-comparison branch.
neutts2e_model = os.environ.get('NEUTTS2E_MODEL', 'neuphonic/neutts-2e-q4-gguf')  # Default NeuTTS 2E backbone
neutts2e_voice = os.environ.get('NEUTTS2E_VOICE', 'emily')  # Default NeuTTS 2E speaker
neutts2e_emotion = os.environ.get('NEUTTS2E_EMOTION', 'neutral')  # Default NeuTTS 2E emotion
# The decoder-only ONNX codec avoids ~3GB of torch codec and semantic encoder that inference never runs
neutts_codec = os.environ.get('NEUTTS_CODEC', 'neuphonic/neucodec-onnx-decoder-int8')
neutts_device = os.environ.get('NEUTTS_DEVICE', 'cpu')  # Device for the NeuTTS codec


def _default_neutts_backbone_device() -> str:
    """Prefer Metal on Apple Silicon, where llama.cpp measures ~1.16x faster than CPU.

    Only applies when the device was left unset. An explicit NEUTTS_DEVICE is a deliberate choice
    and keeps its previous meaning of pinning the whole model, so setting it to cpu still opts out.
    """
    if 'NEUTTS_DEVICE' in os.environ:
        return neutts_device
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        return 'metal'
    return neutts_device


# The ONNX codec refuses any device but CPU, so a single setting would force the whole model onto
# CPU. The GGUF backbone is the bulk of synthesis time and llama.cpp runs it on Metal, so the two
# are configured separately.
neutts_backbone_device = os.environ.get('NEUTTS_BACKBONE_DEVICE', _default_neutts_backbone_device())
NEUTTS_SAMPLE_RATE = 24000  # Both NeuTTS models emit 24 kHz audio
NEUTTS_VARIANT_SEP = '|'  # Separates backbone from codec inside a neutts cache variant
# Piper voices are self-contained ONNX files rather than a repo, so the voice name is the model
piper_voice = os.environ.get('PIPER_VOICE', 'en_US-lessac-medium')  # Default Piper voice
piper_voice_dir = os.environ.get('PIPER_VOICE_DIR', str(Path.home() / '.cache' / 'talkito' / 'piper'))

# Local model caching for offline TTS providers (kokoro/kittentts)
KOKORO_REPO_ID = 'hexgrad/Kokoro-82M'
MODEL_LOAD_TIMEOUT = 30.0  # Seconds to wait for a model that is already present on disk
MODEL_DOWNLOAD_GRACE = 0.5  # Seconds to wait on an in-flight first-run download before giving up
MODEL_RETRY_COOLDOWN = 30.0  # Seconds before a failed model load is retried
MAX_CACHED_MODELS = 3  # Bound memory when several providers or languages are used in one session


@dataclass
class _ModelEntry:
    """Cache slot for one locally loaded TTS model."""
    model: Any = None
    loading: bool = False
    downloading: bool = False
    error: Optional[str] = None
    error_time: float = 0.0
    reported: bool = False
    last_used: float = 0.0


# Keyed by (provider, variant) so switching provider or language does not evict the other model
_local_models: Dict[Tuple[str, str], _ModelEntry] = {}
_local_model_cache_lock = threading.Lock()


def _release_local_models() -> None:
    """Close cached models at exit, while the interpreter can still run their teardown.

    llama.cpp releases its Metal residency sets in Llama.close(). Left to __del__ during interpreter
    shutdown, that runs after module globals have been torn down, so it raises part-way through, the
    release never completes, and ggml-metal's own static destructor then aborts the process on an
    assertion that the sets are empty. On CPU the same failure is only a printed traceback; on Metal
    it kills the process with SIGABRT after the work is already done.

    Closing here breaks that chain, because imports are still live at this point.
    """
    with _local_model_cache_lock:
        entries = [entry for entry in _local_models.values() if entry.model is not None]
        _local_models.clear()

    for entry in entries:
        model, entry.model = entry.model, None
        # NeuTTS holds the llama.cpp handle as .backbone; other providers may nest it elsewhere
        for name in ('backbone', 'model', 'llm', 'codec'):
            close = getattr(getattr(model, name, None), 'close', None)
            if callable(close):
                try:
                    close()
                except Exception as e:
                    log_message("DEBUG", f"Ignoring error closing {name}: {e}")
        del model


atexit.register(_release_local_models)
_delayed_items_lock = threading.Lock()
_delayed_timer = None
_delayed_speech_item: Optional[SpeechItem] = None

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
        # The 0.8.x line is published only as a GitHub release; PyPI still stops at the old 0.1 line
        'install': 'pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl soundfile phonemizer',
        'config_keys': ['model', 'voice']
    },
    'kokoro': {
        'env_var': None,  # Kokoro doesn't need an API key
        'language_var': 'kokoro_language',
        'voice_var': 'kokoro_voice',
        'speed_var': 'kokoro_speed',
        'display_name': 'KokoroTTS',
        'install': 'pip install \'kokoro>=0.9.4\' soundfile phonemizer',
        'config_keys': ['language', 'voice', 'speed']
    },
    'neutts2e': {
        'env_var': None,  # NeuTTS 2E doesn't need an API key
        'model_var': 'neutts2e_model',
        'voice_var': 'neutts2e_voice',
        'display_name': 'NeuTTS 2E',
        'install': 'pip install neutts llama-cpp-python soundfile',
        'config_keys': ['model', 'voice', 'emotion']
    },
    'piper': {
        'env_var': None,  # Piper doesn't need an API key
        'voice_var': 'piper_voice',
        'display_name': 'Piper',
        'install': 'pip install piper-tts',
        'config_keys': ['voice']
    }
}

# Local providers whose models are loaded in-process and cached by (provider, variant)
LOCAL_MODEL_PROVIDERS = ('kokoro', 'kittentts', 'neutts2e', 'piper')

# Available voices for each TTS provider (organized by language using BCP 47 codes)
AVAILABLE_VOICES = {
    'openai': {
        'en-US': ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse']
    },
    'aws': {
        'en-US': ['Joanna', 'Matthew', 'Ivy', 'Kendra', 'Kimberly', 'Salli', 'Joey', 'Justin', 'Kevin', 'Ruth', 'Stephen', 'Gregory', 'Danielle'],
        'en-GB': ['Amy', 'Brian', 'Emma', 'Arthur'],
        'en-AU': ['Nicole', 'Russell', 'Olivia'],
        'en-IN': ['Raveena', 'Kajal'],
        'en-NZ': ['Aria'],
        'en-ZA': ['Ayanda'],
        'es-ES': ['Lucia', 'Sergio'],
        'es-MX': ['Mia', 'Andres'],
        'es-US': ['Lupe', 'Pedro'],
        'fr-FR': ['Celine', 'Mathieu', 'Lea'],
        'fr-CA': ['Chantal', 'Gabrielle', 'Liam'],
        'de-DE': ['Marlene', 'Hans', 'Vicki', 'Daniel'],
        'it-IT': ['Carla', 'Giorgio', 'Bianca'],
        'pt-BR': ['Vitoria', 'Camila', 'Ricardo', 'Thiago'],
        'pt-PT': ['Ines'],
        'ja-JP': ['Mizuki', 'Takumi', 'Kazuha', 'Tomoko'],
        'ko-KR': ['Seoyeon'],
        'zh-CN': ['Zhiyu'],
        'ar-AE': ['Hala', 'Zayd'],
        'hi-IN': ['Aditi', 'Kajal'],
        'pl-PL': ['Ola'],
        'ru-RU': ['Tatyana', 'Maxim'],
        'sv-SE': ['Astrid'],
        'tr-TR': ['Filiz'],
        'nl-NL': ['Lotte', 'Laura'],
        'nb-NO': ['Liv', 'Ida'],
        'da-DK': ['Naja', 'Mads'],
        'ro-RO': ['Carmen']
    },
    'polly': {  # Alias for aws
        'en-US': ['Joanna', 'Matthew', 'Ivy', 'Kendra', 'Kimberly', 'Salli', 'Joey', 'Justin', 'Kevin'],
        'en-GB': ['Amy', 'Brian', 'Emma'],
        'en-AU': ['Nicole', 'Russell'],
        'en-IN': ['Raveena']
    },
    'azure': {
        'en-US': ['AriaNeural', 'GuyNeural', 'JennyNeural', 'AmberNeural', 'AshleyNeural', 'BrandonNeural', 'ChristopherNeural', 'CoraNeural', 'DavisNeural', 'ElizabethNeural', 'EricNeural', 'JacobNeural', 'JaneNeural', 'JasonNeural', 'MichelleNeural', 'MonicaNeural', 'NancyNeural', 'RogerNeural', 'SaraNeural', 'SteffanNeural', 'TonyNeural'],
        'en-GB': ['LibbyNeural', 'RyanNeural', 'SoniaNeural'],
        'en-AU': ['NatashaNeural', 'WilliamNeural'],
        'en-CA': ['ClaraNeural', 'LiamNeural'],
        'en-IN': ['NeerjaNeural', 'PrabhatNeural'],
        'es-ES': ['ElviraNeural', 'AlvaroNeural'],
        'es-MX': ['DaliaNeural', 'JorgeNeural'],
        'fr-FR': ['DeniseNeural', 'HenriNeural'],
        'fr-CA': ['SylvieNeural', 'JeanNeural', 'AntoineNeural'],
        'de-DE': ['KatjaNeural', 'ConradNeural'],
        'it-IT': ['ElsaNeural', 'IsabellaNeural', 'DiegoNeural'],
        'pt-BR': ['FranciscaNeural', 'AntonioNeural'],
        'pt-PT': ['RaquelNeural', 'DuarteNeural'],
        'ja-JP': ['NanamiNeural', 'KeitaNeural'],
        'ko-KR': ['SunHiNeural', 'InJoonNeural'],
        'zh-CN': ['XiaoxiaoNeural', 'YunxiNeural', 'YunjianNeural', 'XiaoyiNeural'],
        'zh-HK': ['HiuMaanNeural', 'WanLungNeural'],
        'zh-TW': ['HsiaoChenNeural', 'YunJheNeural']
    },
    'gcloud': {
        'en-US': ['Standard-A', 'Standard-B', 'Standard-C', 'Standard-D', 'Standard-E', 'Standard-F', 'Standard-G', 'Standard-H', 'Standard-I', 'Standard-J', 'Journey-D', 'Journey-F', 'News-K', 'News-L', 'News-M', 'News-N', 'Polyglot-1', 'Studio-M', 'Studio-O', 'Wavenet-A', 'Wavenet-B', 'Wavenet-C', 'Wavenet-D', 'Wavenet-E', 'Wavenet-F'],
        'en-GB': ['Standard-A', 'Standard-B', 'Standard-C', 'Standard-D', 'Standard-F', 'Wavenet-A', 'Wavenet-B', 'Wavenet-C', 'Wavenet-D', 'Wavenet-F'],
        'en-AU': ['Standard-A', 'Standard-B', 'Standard-C', 'Standard-D', 'Wavenet-A', 'Wavenet-B', 'Wavenet-C', 'Wavenet-D'],
        'en-IN': ['Standard-A', 'Standard-B', 'Standard-C', 'Standard-D', 'Wavenet-A', 'Wavenet-B', 'Wavenet-C', 'Wavenet-D']
    },
    'elevenlabs': {
        'en-US': [  # Voice ID, Name
            ('21m00Tcm4TlvDq8ikWAM', 'Rachel'),
            ('AZnzlk1XvdvUeBnXmlld', 'Domi'),
            ('EXAVITQu4vr4xnSDxMaL', 'Bella'),
            ('ErXwobaYiN019PkySvjV', 'Antoni'),
            ('MF3mGyEYCl7XYWbV9V6O', 'Elli'),
            ('TxGEqnHWrfWFTfGW9XjX', 'Josh'),
            ('VR6AewLTigWG4xSOukaG', 'Arnold'),
            ('pNInz6obpgDQGcFmaJgB', 'Adam'),
            ('yoZ06aMxZJJ28mfd3POQ', 'Sam'),
        ]
    },
    'deepgram': {
        'en-US': [
            # Aura 1 voices
            'aura-asteria-en', 'aura-luna-en', 'aura-stella-en', 'aura-athena-en', 'aura-hera-en', 'aura-orion-en', 'aura-arcas-en', 'aura-perseus-en', 'aura-angus-en', 'aura-orpheus-en', 'aura-helios-en', 'aura-zeus-en',
            # Aura 2 voices
            'aura-2-amalthea-en', 'aura-2-andromeda-en', 'aura-2-apollo-en', 'aura-2-arcas-en', 'aura-2-aries-en', 'aura-2-asteria-en', 'aura-2-athena-en', 'aura-2-atlas-en', 'aura-2-aurora-en', 'aura-2-callista-en', 'aura-2-cora-en', 'aura-2-cordelia-en', 'aura-2-delia-en', 'aura-2-draco-en', 'aura-2-electra-en', 'aura-2-harmonia-en', 'aura-2-helena-en', 'aura-2-hera-en', 'aura-2-hermes-en', 'aura-2-hyperion-en', 'aura-2-iris-en', 'aura-2-janus-en', 'aura-2-juno-en', 'aura-2-jupiter-en', 'aura-2-luna-en', 'aura-2-mars-en', 'aura-2-minerva-en', 'aura-2-neptune-en', 'aura-2-odysseus-en', 'aura-2-ophelia-en', 'aura-2-orion-en', 'aura-2-orpheus-en', 'aura-2-pandora-en', 'aura-2-phoebe-en', 'aura-2-pluto-en', 'aura-2-saturn-en', 'aura-2-selene-en', 'aura-2-thalia-en', 'aura-2-theia-en', 'aura-2-vesta-en', 'aura-2-zeus-en'
        ],
        'es-ES': [
            'aura-2-sirio-es', 'aura-2-nestor-es', 'aura-2-carina-es', 'aura-2-celeste-es', 'aura-2-alvaro-es', 'aura-2-diana-es', 'aura-2-aquila-es', 'aura-2-selena-es', 'aura-2-estrella-es', 'aura-2-javier-es'
        ]
    },
    'kittentts': {
        # Voice names for the 0.8 model line; the older 0.1/0.2 models used expr-voice-N-{m,f}
        'en-US': ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
    },
    'kokoro': {
        'en-US': ['af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
        'en-GB': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
        'ja-JP': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
        'zh-CN': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'],
        'es-ES': ['ef_dora', 'em_alex', 'em_santa'],
        'fr-FR': ['ff_siwis'],
        'hi-IN': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
        'it-IT': ['if_sara', 'im_nicola'],
        'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa']
    },
    'neutts2e': {
        'en-US': ['emily', 'paul', 'sophie', 'steven']
    },
    'piper': {
        # Piper ships a separate model file per voice; 'medium' is the quality tier that matches
        # kokoro and kittentts on size. Full list: python -m piper.download_voices --help
        'en-US': ['en_US-lessac-medium', 'en_US-amy-medium', 'en_US-ryan-medium', 'en_US-ryan-high',
                  'en_US-hfc_female-medium', 'en_US-hfc_male-medium', 'en_US-joe-medium',
                  'en_US-kristin-medium', 'en_US-libritts_r-medium', 'en_US-ljspeech-medium'],
        'en-GB': ['en_GB-alan-medium', 'en_GB-alba-medium', 'en_GB-cori-medium',
                  'en_GB-jenny_dioco-medium', 'en_GB-northern_english_male-medium']
    },
    'system': {}  # System voices depend on the OS
}

# Emotions supported by the NeuTTS 2E backbone
NEUTTS2E_EMOTIONS = ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised')


def disable_tts_completely(reason: str = None, args: Any = None) -> None:
    """Disable TTS at all levels: module globals, args, and shared state."""
    global disable_tts, tts_provider

    disable_tts = True
    tts_provider = None

    # Also set args if provided
    if args is not None and hasattr(args, 'disable_tts'):
        args.disable_tts = True

    # Update shared state
    try:
        shared_state = get_shared_state()
        shared_state.set_tts_enabled(False)
        shared_state.set_tts_config(provider=None)
        if reason:
            log_message("INFO", f"TTS disabled: {reason}")
    except Exception as e:
        log_message("WARNING", f"Could not update shared state to disable TTS: {e}")


def _normalize_kokoro_lang(lang_code: str) -> str:
    """Map a configured Kokoro language onto a code KPipeline accepts, falling back to American English."""
    lang_code = str(lang_code or 'a').lower()
    try:
        from kokoro.pipeline import ALIASES, LANG_CODES
    except ImportError:
        return lang_code
    normalized = ALIASES.get(lang_code, lang_code)
    if normalized not in LANG_CODES:
        log_message("WARNING", f"Unknown Kokoro language {lang_code!r}, falling back to 'a' (American English)")
        return 'a'
    return normalized


def _resolve_piper_voice() -> str:
    """Return the Piper voice to use, accepting names outside the curated menu shortlist.

    Piper publishes well over a hundred voices and AVAILABLE_VOICES lists fifteen of them, so that
    entry is a menu convenience and cannot act as an allowlist. Every caller goes through here:
    preload, configuration and synthesis resolving this separately is how a model gets fetched for
    one voice and then spoken with another.
    """
    state = get_shared_state()
    if state.tts_provider == 'piper' and state.tts_voice:
        return str(state.tts_voice)
    return piper_voice


def _model_variant(provider: str) -> str:
    """Return the cache variant for a provider: language for kokoro, model name for kittentts."""
    if provider == 'kokoro':
        config = get_tts_config()
        return _normalize_kokoro_lang(config.get('language') or kokoro_language)
    if provider == 'kittentts':
        return kittentts_model
    if provider == 'piper':
        return _resolve_piper_voice()
    if provider == 'neutts2e':
        # The codec is part of the identity: the same backbone built on a different codec is a
        # different model, so it must not be served from the same cache slot
        config = get_tts_config()
        backbone = str(config.get('model') or neutts2e_model)
        return f"{backbone}{NEUTTS_VARIANT_SEP}{neutts_codec}"
    return ''


def _split_neutts_variant(variant: str) -> Tuple[str, str]:
    """Split a neutts cache variant back into its backbone repo and codec repo."""
    if not variant:
        return neutts2e_model, neutts_codec
    backbone, _, codec = variant.partition(NEUTTS_VARIANT_SEP)
    return backbone or neutts2e_model, codec or neutts_codec


class _ThreadScopedCapture:
    """Divert one thread's stdout writes, leaving every other thread's alone.

    contextlib.redirect_stdout swaps sys.stdout for the whole process, and model loading runs in a
    background thread while the wrapped program owns the terminal - so a plain redirect would
    swallow the program's output for as long as the load takes.
    """

    def __init__(self, real, thread_id, sink):
        self._real, self._thread_id, self._sink = real, thread_id, sink

    def write(self, data):
        if threading.get_ident() == self._thread_id:
            self._sink.write(data)
            return len(data)
        return self._real.write(data)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


_stdout_capture_lock = threading.Lock()


@contextlib.contextmanager
def _quiet_model_load(label: str):
    """Send a model's loading chatter to the log instead of the terminal.

    neutts and the perth watermarker announce themselves with bare print() calls, which land in
    whatever the wrapped program is drawing - most visibly inside a TUI's input line. The text is
    kept, at DEBUG, rather than discarded.
    """
    sink = io.StringIO()
    with _stdout_capture_lock:
        real = sys.stdout
        sys.stdout = _ThreadScopedCapture(real, threading.get_ident(), sink)
    try:
        yield
    finally:
        with _stdout_capture_lock:
            sys.stdout = real
        for line in sink.getvalue().splitlines():
            if line.strip():
                log_message("DEBUG", f"{label}: {line.strip()}")


def _create_model_instance(provider: str, variant: str = ''):
    """Create model instance for the specified provider."""
    if provider == 'kokoro':
        # First check if kokoro module is installed
        try:
            with suppress_ai_warnings():
                import kokoro  # noqa: F401
        except ImportError:
            install_cmd = TTS_PROVIDERS['kokoro']['install']
            raise ImportError(
                f"Kokoro TTS module is not installed. "
                f"Install it with: {install_cmd}"
            )

        # Check if spaCy model is available
        from .models import check_spacy_model_consent
        if not check_spacy_model_consent('kokoro'):
            raise RuntimeError("spaCy language model required for kokoro but download was declined")

        with suppress_ai_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            from kokoro import KPipeline

        # Model creation - consent was already obtained in main thread
        lang_code = _normalize_kokoro_lang(variant or kokoro_language)
        log_message("DEBUG", f"About to create KPipeline(lang_code='{lang_code}', repo_id='{KOKORO_REPO_ID}')")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")
            pipeline = KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO_ID)
        log_message("DEBUG", "KPipeline created successfully")
        return pipeline

    elif provider == 'kittentts':
        # First check if kittentts module is installed
        try:
            with suppress_ai_warnings():
                import kittentts  # noqa: F401
        except ImportError:
            install_cmd = TTS_PROVIDERS['kittentts']['install']
            raise ImportError(
                f"KittenTTS module is not installed. "
                f"Install it with: {install_cmd}"
            )

        with suppress_ai_warnings():
            from kittentts import KittenTTS

        # Model creation - consent was already obtained in main thread
        model_name = variant or kittentts_model
        log_message("DEBUG", f"KittenTTS(model_name) with model_name = {model_name}")
        return KittenTTS(model_name)

    elif provider == 'neutts2e':
        try:
            with suppress_ai_warnings():
                import neutts  # noqa: F401
        except ImportError:
            install_cmd = TTS_PROVIDERS[provider]['install']
            raise ImportError(
                f"NeuTTS module is not installed. "
                f"Install it with: {install_cmd}"
            )

        with suppress_ai_warnings():
            from neutts import NeuTTS2E

        backbone_repo, codec_repo = _split_neutts_variant(variant)
        resolved_backbone = _resolve_neutts_backbone(backbone_repo)
        log_message("DEBUG", f"Creating NeuTTS2E(backbone_repo='{resolved_backbone}', codec_repo='{codec_repo}')")

        # A rejected codec/device pairing and a missing file are both certain to fail and free to
        # test, so rule them out before loading a backbone. Faults that only appear once the codec
        # is constructed - a missing onnxruntime, a corrupt model, a failed session - cost far too
        # much to exclude here, and are diagnosed below only if something actually goes wrong.
        resolved_codec = _resolve_neutts_codec(codec_repo)
        codec_device_error = _neutts_codec_device_error(resolved_codec, neutts_device)
        if codec_device_error:
            raise RuntimeError(codec_device_error)
        if resolved_codec.endswith('.onnx') and not os.path.isfile(resolved_codec):
            raise RuntimeError(f"NeuTTS ONNX codec is missing at {resolved_codec}")

        # 2E is a BPE model, so it loads no phonemizer and needs no language argument
        def _build(backbone_device):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torch")
                return NeuTTS2E(
                    backbone_repo=resolved_backbone,
                    backbone_device=backbone_device,
                    codec_repo=resolved_codec,
                    codec_device=neutts_device,
                )

        try:
            return _build(neutts_backbone_device)
        except Exception as e:
            # Metal is the default where the hardware suggests it, but llama.cpp may be built
            # without it and the GPU may be unavailable under some sandboxes. Losing speech
            # entirely is worse than losing the speed-up, so fall back rather than propagate.
            # Only initialization is covered; a Metal fault during synthesis still surfaces.
            if neutts_backbone_device == 'cpu':
                raise
            # NeuTTS2E loads the backbone before the codec, so this failure may belong to either.
            # The codec does not vary with the backbone device: if it is the one at fault, a retry
            # fails identically, so establish which half broke before spending a second backbone.
            # Checking here rather than up front keeps the working path free of the cost.
            try:
                codec_verified = _preflight_neutts_codec(resolved_codec, neutts_device)
            except Exception as codec_error:
                log_message("ERROR",
                            f"NeuTTS codec at {resolved_codec} does not load ({codec_error}); "
                            f"the backbone device is not the cause, so not retrying on CPU")
                raise
            if codec_verified:
                log_message("WARNING",
                            f"NeuTTS backbone failed on '{neutts_backbone_device}' ({e}); "
                            f"the codec loads, so retrying the backbone on CPU")
            else:
                # Say only what has been established. A torch codec cannot be checked cheaply, so
                # the cause is unknown and the retry is a hope rather than a diagnosis.
                log_message("WARNING",
                            f"NeuTTS construction failed on '{neutts_backbone_device}' ({e}); "
                            f"the codec could not be checked independently, so the cause is "
                            f"undetermined. Retrying on CPU")
            return _build('cpu')
    elif provider == 'piper':
        try:
            from piper import PiperVoice
        except ImportError:
            install_cmd = TTS_PROVIDERS['piper']['install']
            raise ImportError(
                f"Piper TTS module is not installed. "
                f"Install it with: {install_cmd}"
            )

        voice_file = _ensure_piper_voice(variant or piper_voice)
        log_message("DEBUG", f"Creating PiperVoice.load('{voice_file}')")
        return PiperVoice.load(voice_file)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def _shared_voice_for(provider: str) -> Optional[str]:
    """Return the shared-state voice only when it belongs to the given provider."""
    config = get_tts_config()
    if config.get('provider') != provider:
        return None
    return config.get('voice')


def _piper_voice_complete(onnx_path: Path) -> bool:
    """Report whether both halves of a Piper voice are present and non-empty.

    An interrupted download leaves a zero-byte file behind, which exists but cannot be loaded, so
    presence alone would mark a broken voice as ready and skip the fetch that would repair it. The
    json is parsed as well, since a truncated one is the same failure wearing a different hat.
    """
    config_path = onnx_path.with_suffix('.onnx.json')
    try:
        if not (onnx_path.is_file() and onnx_path.stat().st_size > 0):
            return False
        if not (config_path.is_file() and config_path.stat().st_size > 0):
            return False
        with open(config_path) as handle:
            json.load(handle)
    except (OSError, ValueError):
        return False
    return True


def piper_voice_path(voice: str) -> Path:
    """Return the on-disk path of a Piper voice, downloaded or not."""
    return Path(piper_voice_dir) / f"{voice}.onnx"


def _ensure_piper_voice(voice: str) -> str:
    """Download a Piper voice if it is not already complete on disk, and return its path.

    A voice is the pair of files, and the runtime needs both. Treating the weights alone as
    complete leaves an interrupted download unrepairable: the fetch is skipped and the load then
    fails every time, so both are required here and after downloading.
    """
    path = piper_voice_path(voice)
    config_path = path.with_suffix('.onnx.json')
    if _piper_voice_complete(path):
        return str(path)

    # Anything already on disk failed the completeness test above, so it has to be replaced rather
    # than kept: the downloader skips any existing non-empty file, which would leave a truncated
    # json in place and fail the load again on every attempt.
    partial = path.exists() or config_path.exists()
    if partial:
        log_message("WARNING",
                    f"Piper voice '{voice}' is incomplete on disk; re-fetching it")

    # Consent was already gathered by preload_local_model; this is the fetch itself
    from piper.download_voices import download_voice

    path.parent.mkdir(parents=True, exist_ok=True)
    log_message("INFO", f"Downloading Piper voice '{voice}' into {path.parent}")
    download_voice(voice, path.parent, force_redownload=partial)
    if not _piper_voice_complete(path):
        raise RuntimeError(f"Piper voice '{voice}' is incomplete after download at {path}")
    return str(path)



def _pcm_safe(audio):
    """Scale audio back under full scale before a PCM_16 write, which otherwise hard-clips.

    Neural vocoders can overshoot [-1, 1] on short utterances: NeuTTS 2E exceeded full scale on 5 of
    8 runs of "hi", and libsndfile clamps rather than wraps, pinning a third of the samples at the
    rail. Returns the input untouched when it already fits, so this only engages on the overshoot.
    """
    import numpy as np
    audio = np.asarray(audio)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 1.0:
        return audio
    log_message("DEBUG", f"Normalising TTS audio from peak {peak:.3f} to avoid PCM_16 clipping")
    return audio / peak

def _resolve_neutts_codec(codec_repo: str) -> str:
    """Resolve an ONNX codec repo id to its local model.onnx path.

    The neucodec ONNX repos ship no config.json, so NeuCodecOnnxDecoder.from_pretrained() fails on
    them; neutts accepts a direct .onnx file path instead, which is what we hand it.
    """
    if 'onnx' not in codec_repo or codec_repo.endswith('.onnx'):
        return codec_repo

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=codec_repo, filename='model.onnx')
        log_message("DEBUG", f"Resolved ONNX codec {codec_repo} to {path}")
        return path
    except Exception as e:
        log_message("WARNING", f"Could not resolve ONNX codec {codec_repo}, passing through: {e}")
        return codec_repo


def _neutts_codec_device_error(resolved_codec: str, codec_device: str) -> Optional[str]:
    """Report the codec/device combinations NeuTTS refuses, without loading anything.

    NeuTTS raises for an ONNX decoder on any device but CPU. Reproducing that rule is what makes
    the check meaningful: loading the decoder on its own always succeeds regardless of the device
    that will be asked for, so without this a bad codec_device looks like a healthy codec.
    """
    if 'onnx' in resolved_codec and codec_device != 'cpu':
        return (f"NeuTTS runs ONNX decoders on CPU only, but codec_device is '{codec_device}'. "
                f"Set NEUTTS_DEVICE=cpu, or choose a torch codec via NEUTTS_CODEC.")
    return None


def _preflight_neutts_codec(resolved_codec: str, codec_device: str) -> bool:
    """Load the codec alone, to establish whether it rather than the backbone is at fault.

    Returns True when the codec was verified and False when it could not be checked independently,
    so the caller can avoid asserting a cause it has not established. Raises when the codec is
    definitely broken.
    """
    error = _neutts_codec_device_error(resolved_codec, codec_device)
    if error:
        raise RuntimeError(error)

    if 'onnx' not in resolved_codec:
        # The torch codecs pull a multi-gigabyte semantic encoder at construction, which costs far
        # more to load twice than the retry it would save. talkito never selects one by default.
        return False

    try:
        from neucodec import NeuCodecOnnxDecoder

        if resolved_codec.endswith('.onnx'):
            if not os.path.isfile(resolved_codec):
                raise FileNotFoundError(resolved_codec)
            NeuCodecOnnxDecoder(resolved_codec)
        else:
            NeuCodecOnnxDecoder.from_pretrained(resolved_codec)
    except Exception as e:
        raise RuntimeError(f"NeuTTS codec failed to load from {resolved_codec}: {e}") from e
    return True


def _resolve_neutts_backbone(backbone_repo: str) -> str:
    """Resolve a GGUF backbone repo id to its local .gguf file.

    neutts loads GGUF repos via Llama.from_pretrained, which queries the Hub even when the file is
    already cached, so it fails offline. Handing it a file path takes the local-file branch instead.
    """
    if not backbone_repo.lower().endswith('gguf') or os.path.isfile(backbone_repo):
        return backbone_repo

    try:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(repo_id=backbone_repo, allow_patterns=['*.gguf'])
        candidates = sorted(Path(snapshot).glob('*.gguf'))
        if candidates:
            log_message("DEBUG", f"Resolved GGUF backbone {backbone_repo} to {candidates[0]}")
            return str(candidates[0])
    except Exception as e:
        log_message("WARNING", f"Could not resolve GGUF backbone {backbone_repo}, passing through: {e}")
    return backbone_repo


def _evict_stale_models_locked() -> None:
    """Drop least recently used models once the cache exceeds MAX_CACHED_MODELS; caller holds the lock."""
    loaded = [(key, entry) for key, entry in _local_models.items() if entry.model is not None]
    while len(loaded) > MAX_CACHED_MODELS:
        key, _ = min(loaded, key=lambda item: item[1].last_used)
        log_message("INFO", f"Evicting cached model {key} to bound memory")
        _local_models.pop(key, None)
        loaded = [(k, e) for k, e in loaded if k != key]


def _any_model_loading() -> bool:
    """True while any local model is loading, used to hold off the speech worker."""
    with _local_model_cache_lock:
        return any(entry.loading for entry in _local_models.values())


def _record_model_error(key: Tuple[str, str], message: str) -> None:
    """Store a load failure against a cache slot so it is retried after a cooldown rather than forever."""
    with _local_model_cache_lock:
        entry = _local_models.setdefault(key, _ModelEntry())
        entry.loading = False
        entry.downloading = False
        entry.error = message
        entry.error_time = time.monotonic()


def _load_model_background(provider: str, variant: str):
    """Load model in background thread."""
    key = (provider, variant)

    try:
        log_message("INFO", f"Background loading {provider} model...")
        with _quiet_model_load(provider):
            model = _create_model_instance(provider, variant)

        with _local_model_cache_lock:
            entry = _local_models.setdefault(key, _ModelEntry())
            entry.model = model
            entry.loading = False
            entry.downloading = False
            entry.error = None
            entry.reported = False
            entry.last_used = time.monotonic()
            _evict_stale_models_locked()

        log_message("INFO", f"{provider} model loaded successfully in background")

    except Exception as e:
        _record_model_error(key, f"{provider} model loading failed: {e}")
        log_message("ERROR", f"Background {provider} model loading failed: {e}")
        log_message("ERROR", f"Traceback: {traceback.format_exc()}")


def _model_needs_download(provider: str, variant: str) -> bool:
    """Return True when the provider's model still has to be fetched from the network."""
    from .models import check_model_cached

    if provider == 'kokoro':
        return not check_model_cached('kokoro', KOKORO_REPO_ID)
    if provider == 'neutts2e':
        backbone_repo, codec_repo = _split_neutts_variant(variant)
        return not check_model_cached(provider, backbone_repo, codec_repo)
    if provider == 'piper':
        return not check_model_cached(provider, variant or piper_voice)
    return not check_model_cached(provider, variant or kittentts_model)


def _fall_back_from_declined_download(provider: str) -> None:
    """Switch to the next usable TTS provider after the user declines a model download."""
    fallback_provider = select_best_tts_provider(excluded_providers={provider})
    if fallback_provider is None:
        print("Download declined and no fallback TTS provider available. TTS will be disabled.")
        log_message("WARNING", f"No fallback TTS provider available after user declined {provider} download")
        disable_tts_completely(f"user declined {provider} download and no fallback available")
        return

    print(f"Download declined. Falling back to {fallback_provider} TTS provider.")

    # Update shared state with fallback provider
    try:
        shared_state = get_shared_state()
        shared_state.set_tts_config(provider=fallback_provider)
        log_message("INFO", f"[TTS_PRELOAD] Updated shared state to use fallback provider: {fallback_provider}")
    except Exception as e:
        log_message("WARNING", f"[TTS_PRELOAD] Could not update shared state with fallback: {e}")


def preload_local_model(provider: str, variant: Optional[str] = None):
    """Start background preloading of local model for specified provider."""
    if provider not in LOCAL_MODEL_PROVIDERS:
        log_message("WARNING", f"Unknown provider for preloading: {provider}")
        return

    if variant is None:
        variant = _model_variant(provider)
    key = (provider, variant)

    with _local_model_cache_lock:
        entry = _local_models.get(key)
        if entry is not None:
            if entry.loading:
                log_message("DEBUG", f"Model already loading, skipping preload for {provider}")
                return
            if entry.model is not None:
                log_message("DEBUG", f"Model already cached for {provider}, skipping preload")
                return

    # Consent is gathered outside the lock so a prompt never blocks other speech threads.
    # input() only works on the main thread, so a worker that needs a download records an
    # error instead of silently competing for stdin with the wrapped program.
    needs_download = _model_needs_download(provider, variant)
    if needs_download:
        if threading.current_thread() is not threading.main_thread():
            message = (f"{provider} model needs downloading but consent cannot be requested from thread "
                       f"'{threading.current_thread().name}'; preload it at startup or set "
                       f"TALKITO_AUTO_APPROVE_DOWNLOADS=1")
            log_message("WARNING", message)
            _record_model_error(key, message)
            return

        from .models import ask_user_consent
        model_name = variant or provider
        if not ask_user_consent(provider, model_name):
            log_message("INFO", f"User declined download for {provider} model '{model_name}'")
            _record_model_error(key, f"user declined download of {provider} model '{model_name}'")
            _fall_back_from_declined_download(provider)
            return

    # Start background loading (consent already obtained)
    with _local_model_cache_lock:
        entry = _local_models.setdefault(key, _ModelEntry())
        if entry.loading or entry.model is not None:
            return
        entry.loading = True
        entry.downloading = needs_download
        entry.error = None
        entry.reported = False

    thread = threading.Thread(target=_load_model_background, args=(provider, variant), daemon=True)
    thread.start()
    log_message("INFO", f"Started background loading of {provider} model")

    # Show confirmation message if download was started
    if needs_download:
        print(f"Downloading {provider} model in background. TTS will start automatically when ready.")


def get_cached_local_model(provider: str, timeout: Optional[float] = None, variant: Optional[str] = None):
    """Get cached local model, waiting for background loading if needed."""
    if variant is None:
        variant = _model_variant(provider)
    key = (provider, variant)
    if timeout is None:
        timeout = MODEL_LOAD_TIMEOUT

    start_time = time.monotonic()

    while True:
        need_to_preload = False

        with _local_model_cache_lock:
            entry = _local_models.get(key)

            if entry is not None:
                # Model is ready
                if entry.model is not None:
                    entry.last_used = time.monotonic()
                    return entry.model

                # A previous failure is honoured for a cooldown, then retried rather than being permanent
                if entry.error and not entry.loading:
                    if time.monotonic() - entry.error_time < MODEL_RETRY_COOLDOWN:
                        if not entry.reported:
                            entry.reported = True
                            print(f"Error loading {provider} model: {entry.error}")
                        else:
                            log_message("DEBUG", f"{provider} model still in error cooldown: {entry.error}")
                        return None
                    log_message("INFO", f"Retrying {provider} model load after previous failure")
                    entry.error = None
                    entry.reported = False

                # A first-run download outlasts any sensible per-utterance wait, so drop this
                # utterance immediately instead of stalling the speech queue for every phrase
                if entry.loading and entry.downloading:
                    if time.monotonic() - start_time > MODEL_DOWNLOAD_GRACE:
                        log_message("INFO", f"{provider} model is still downloading, skipping this utterance")
                        return None

            if entry is None or not (entry.loading or entry.model is not None):
                log_message("DEBUG", f"{provider} model not loaded and no background loading in progress, starting now")
                need_to_preload = True

        if need_to_preload:
            preload_local_model(provider, variant)

        # Check timeout
        if time.monotonic() - start_time > timeout:
            log_message("WARNING", f"Timeout waiting for {provider} model to load after {timeout} seconds")
            return None

        time.sleep(0.05)


def get_all_voices_for_provider(provider: str) -> list:
    """Get flattened list of all voices for a provider across all languages."""
    if provider not in AVAILABLE_VOICES:
        return []

    provider_voices = AVAILABLE_VOICES[provider]
    if not provider_voices:
        return []

    # Flatten voices from all languages
    all_voices = []
    for lang_code, voices in provider_voices.items():
        all_voices.extend(voices)
    return all_voices

def get_state_voice_if_valid() -> Optional[str]:
    """Check if voice is valid for the given provider."""
    state = get_shared_state()
    provider = state.tts_provider or tts_provider
    if provider not in AVAILABLE_VOICES:
        return None

    voices = get_all_voices_for_provider(provider)

    # ElevenLabs voices are tuples (id, name)
    if provider == 'elevenlabs':
        valid_ids = [voice_id for voice_id, _ in voices]
        if state.tts_voice in valid_ids:
            return state.tts_voice
    else:
        if state.tts_voice in voices:
            return state.tts_voice
    return None

def get_tts_config():
    """Get TTS configuration from shared state or module globals."""
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
            config['voice'] = get_state_voice_if_valid() or openai_voice
        elif state.tts_provider in ['aws', 'polly']:
            config['voice'] = get_state_voice_if_valid() or polly_voice
            config['region'] = state.tts_region or polly_region
        elif state.tts_provider == 'azure':
            config['voice'] = get_state_voice_if_valid() or azure_voice
            config['region'] = state.tts_region or azure_region
        elif state.tts_provider == 'gcloud':
            config['voice'] = get_state_voice_if_valid() or gcloud_voice
            config['language'] = state.tts_language or gcloud_language_code
        elif state.tts_provider == 'elevenlabs':
            config['voice'] = get_state_voice_if_valid() or elevenlabs_voice_id
        elif state.tts_provider == 'deepgram':
            config['voice'] = get_state_voice_if_valid() or deepgram_voice_model
        elif state.tts_provider == 'kittentts':
            config['voice'] = get_state_voice_if_valid() or kittentts_voice
            config['model'] = state.tts_model or kittentts_model
        elif state.tts_provider == 'kokoro':
            config['voice'] = get_state_voice_if_valid() or kokoro_voice
            config['language'] = state.tts_language or kokoro_language
            config['speed'] = float(state.tts_rate or kokoro_speed)
        elif state.tts_provider == 'piper':
            # Not get_state_voice_if_valid(): that filters against the menu shortlist, which would
            # hand synthesis the default while preload used the requested voice
            config['voice'] = _resolve_piper_voice()

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

def _mark_playback_finished(speech_item: Optional[SpeechItem]) -> None:
    """Update playback bookkeeping when audio stops."""
    global current_speech_item, last_speech_end_time

    with _state_lock:
        if speech_item is not None and current_speech_item is speech_item:
            current_speech_item = None
        last_speech_end_time = time.time()


class AudioPlaybackThread(threading.Thread):
    """Thread for non-blocking audio playback."""
    def __init__(self, audio_path: str, playback_control: 'PlaybackControl', speech_item: Optional[SpeechItem] = None):
        super().__init__(daemon=True)
        self.audio_path = audio_path
        self.playback_control = playback_control
        self.process: Optional[subprocess.Popen] = None
        self.stopped = threading.Event()
        self.speech_item = speech_item

    def run(self):
        """Run the audio playback in this thread."""
        try:
            success = self._play_audio()
            # Notify playback control that this thread is done
            with self.playback_control.lock:
                if self.playback_control.current_playback_thread == self:
                    self.playback_control.current_playback_thread = None
                    self.playback_control.current_process = None
            return success
        except Exception as e:
            log_message("ERROR", f"Audio playback thread error: {e}")
            return False
        finally:
            _mark_playback_finished(self.speech_item)

    def stop(self):
        """Stop the audio playback."""
        self.stopped.set()
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.25)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

    def _play_audio(self) -> bool:
        """Play audio file using an available system player (blocking version)."""
        path = Path(self.audio_path)
        if not path.exists():
            log_message("ERROR", f"Audio file not found: {self.audio_path}")
            return False

        ext = path.suffix.lower()
        is_wav = ext == ".wav"
        is_mp3 = ext == ".mp3"

        # Build candidate player commands in priority order for each format
        candidates = []

        if is_wav:
            candidates.extend([
                (["afplay", self.audio_path], "afplay"),
                (["play", "-q", self.audio_path], "play"),
                (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.audio_path], "ffplay"),
                (["paplay", self.audio_path], "paplay"),
                (["aplay", self.audio_path], "aplay"),
                (["cvlc", "--play-and-exit", "--intf", "dummy", self.audio_path], "cvlc"),
            ])

            # Windows (WAV only): PowerShell SoundPlayer
            if sys.platform.startswith("win"):
                candidates.append((
                    ["powershell", "-NoProfile", "-Command",
                     f'[Console]::OutputEncoding=[Text.UTF8]; '
                     f'$p=New-Object System.Media.SoundPlayer "{self.audio_path}"; '
                     f'$p.PlaySync();'],
                    "powershell-wav"
                ))

        elif is_mp3:
            candidates.extend([
                (["afplay", self.audio_path], "afplay"),
                (["mpg123", "-q", self.audio_path], "mpg123"),
                (["play", "-q", self.audio_path], "play"),
                (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.audio_path], "ffplay"),
                (["cvlc", "--play-and-exit", "--intf", "dummy", self.audio_path], "cvlc"),
            ])
        else:
            candidates.extend([
                (["afplay", self.audio_path], "afplay"),
                (["play", "-q", self.audio_path], "play"),
                (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.audio_path], "ffplay"),
                (["mpg123", "-q", self.audio_path], "mpg123"),
                (["paplay", self.audio_path], "paplay"),
                (["aplay", self.audio_path], "aplay"),
                (["cvlc", "--play-and-exit", "--intf", "dummy", self.audio_path], "cvlc"),
            ])

        # Pick the first available player
        chosen_cmd = None
        for cmd, binary in candidates:
            if binary == "powershell-wav":
                chosen_cmd = cmd
                break
            if shutil.which(binary):
                chosen_cmd = cmd
                break

        if not chosen_cmd:
            need = "a WAV-capable player (afplay, play/sox, ffplay, paplay/aplay, or VLC)"
            if is_mp3:
                need = "an MP3-capable player (afplay, mpg123, play/sox, ffplay, or VLC)"
            log_message("ERROR", f"No suitable audio player found for {ext or 'unknown format'}. Install {need}.")
            return False

        # Launch player
        try:
            self.process = subprocess.Popen(
                chosen_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except Exception as e:
            log_message("ERROR", f"Failed to start audio player {chosen_cmd[0]}: {e}")
            return False

        # Store process in playback control for skip handling
        with self.playback_control.lock:
            self.playback_control.current_process = self.process
            log_message("DEBUG", f"Starting audio player process {self.process.pid}")

        try:
            # Poll + allow cooperative interruption
            while self.process.poll() is None:
                if self.stopped.is_set() or shutdown_event.is_set():
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=0.25)
                    except Exception:
                        self.process.kill()
                    # Clean up current_process reference
                    with self.playback_control.lock:
                        if self.playback_control.current_process == self.process:
                            self.playback_control.current_process = None
                    return False

                # Check playback control skip flags
                with self.playback_control.lock:
                    if self.playback_control.skip_current or self.playback_control.skip_all:
                        try:
                            self.process.terminate()
                            self.process.wait(timeout=0.25)
                        except Exception:
                            self.process.kill()
                        # Clean up current_process reference
                        if self.playback_control.current_process == self.process:
                            self.playback_control.current_process = None
                        return False

                time.sleep(0.01)

            # Process finished normally - clean up current_process reference
            success = self.process.returncode == 0
            with self.playback_control.lock:
                if self.playback_control.current_process == self.process:
                    self.playback_control.current_process = None
            return success
        except Exception:
            # Clean up current_process reference on exception
            with self.playback_control.lock:
                if self.playback_control.current_process == self.process:
                    self.playback_control.current_process = None
            return False

class PlaybackControl:
    """Controls TTS playback state and process management."""
    def __init__(self):
        self.current_index = 0
        self.is_paused = False
        self.skip_current = False
        self.skip_all = False
        self.current_process: Optional[subprocess.Popen] = None
        self.current_playback_thread: Optional[AudioPlaybackThread] = None
        self.lock = threading.Lock()

    def pause(self):
        """Pause current TTS playback."""
        with self.lock:
            self.is_paused = True
            if use_orcabot_playback:
                _stop_orcabot_audio()
            if self.current_playback_thread:
                self.current_playback_thread.stop()
            elif self.current_process:
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
            if use_orcabot_playback:
                _stop_orcabot_audio()
            if self.current_playback_thread:
                self.current_playback_thread.stop()
            elif self.current_process:
                try:
                    self.current_process.terminate()
                except Exception:
                    pass

    def skip_all_items(self):
        """Skip all remaining items in TTS queue."""
        with self.lock:
            self.skip_all = True
            if use_orcabot_playback:
                _stop_orcabot_audio()
            if self.current_playback_thread:
                self.current_playback_thread.stop()
            elif self.current_process:
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
last_received_text = ""
last_queue_time = 0
# Playback control
playback_control = PlaybackControl()
speech_history: List[SpeechItem] = []
current_speech_item: Optional[SpeechItem] = None

# Track when speech actually finishes (including audio playback)
last_speech_end_time = 0.0
SPEECH_BUFFER_TIME = 0.01  # Seconds to wait after TTS process ends for audio to finish

# Track the highest line number that has been spoken
highest_spoken_line_number = -1

# Global bullet point counter for TTS queue numbering
global_bullet_counter = 0

# Bullet point patterns for detection
BULLET_PATTERNS = ['☐', '•', '▪', '▫', '■', '□', '◦', '‣', '⁃', '-']

def apply_bullet_numbering(text: str) -> str:
    """Apply bullet point numbering using global counter, reset counter for non-bullet text."""
    global global_bullet_counter

    if not text or not text.strip():
        return text

    # Check if text starts with any bullet pattern
    text_stripped = text.strip()
    starts_with_bullet = False
    bullet_used = None

    for bullet in BULLET_PATTERNS:
        if text_stripped.startswith(bullet):
            starts_with_bullet = True
            bullet_used = bullet
            break

    if starts_with_bullet:
        # Increment counter and replace bullet with number
        global_bullet_counter += 1
        # Remove the bullet and replace with number
        remaining_text = text_stripped[len(bullet_used):].strip()
        numbered_text = f"{global_bullet_counter}: {remaining_text}"
        # Preserve original spacing/formatting by replacing the stripped portion
        return text.replace(text_stripped, numbered_text)
    else:
        # Reset counter for non-bullet text
        global_bullet_counter = 0
        return text

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
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
        yield


# Wrapper for module-specific logging
def log_message(level: str, message: str):
    """Log message with module name using centralized logger."""
    _base_log_message(level, message, __name__)


def check_tts_provider_accessibility(requested_provider: str = None) -> Dict[str, Dict[str, Any]]:
    """Check TTS provider accessibility based on API keys and environment."""
    log_message("INFO", f"check_tts_provider_accessibility called with requested_provider={requested_provider}")
    
    accessible = {}
    if use_orcabot_playback:
        log_message("INFO", "Orcabot playback enabled - skipping local TTS provider checks")
    
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
    if use_orcabot_playback:
        kittentts_note = "Disabled when Orcabot playback is enabled"
    elif requested_provider == 'kittentts':
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
            install_cmd = TTS_PROVIDERS['kittentts']['install']
            kittentts_note = f"Requires KittenTTS package ({install_cmd})"
    
    # Check if model is cached
    if kittentts_available:
        from .models import check_model_cached
        is_cached = check_model_cached('kittentts', kittentts_model)
        if is_cached:
            kittentts_note += " [cached]"
        else:
            kittentts_note += " [needs download]"

    accessible["kittentts"] = {
        "available": kittentts_available,
        "note": kittentts_note
    }


    # KokoroTTS - Check if dependencies are installed
    kokoro_available = False
    kokoro_note = "High-quality 82M parameter TTS model (no API key required)"
    
    # For accessibility check, assume kokoro is available if requested
    # Actual validation happens during model loading with user consent
    if use_orcabot_playback:
        kokoro_note = "Disabled when Orcabot playback is enabled"
    elif requested_provider == 'kokoro':
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
            if isinstance(e, ImportError):
                install_cmd = TTS_PROVIDERS['kokoro']['install']
                kokoro_note = f"Requires KokoroTTS package ({install_cmd})"
            else:
                kokoro_note = f"Kokoro package error: {str(e)}"
        log_message("INFO", f"KokoroTTS availability check completed - available: {kokoro_available}")

    # Check if model is cached
    if kokoro_available:
        from .models import check_model_cached
        is_cached = check_model_cached('kokoro', 'hexgrad/Kokoro-82M')
        if is_cached:
            kokoro_note += " [cached]"
        else:
            kokoro_note += " [needs download]"

    accessible["kokoro"] = {
        "available": kokoro_available,
        "note": kokoro_note
    }

    # NeuTTS (nano backbone) and NeuTTS 2E - one package, two backbones
    for neutts_provider, description in (
        ("neutts2e", "Emotional TTS with built-in speakers (no API key required)"),
    ):
        neutts_available = False
        neutts_note = description

        if use_orcabot_playback:
            neutts_note = "Disabled when Orcabot playback is enabled"
        elif requested_provider == neutts_provider:
            neutts_available = True
            neutts_note = "NeuTTS package (validation deferred to model loading)"
        else:
            # Only check that the package exists; importing neutts costs ~5s and ~466MB because it
            # eagerly pulls neucodec's torch stack, which is far too expensive for a probe
            if importlib.util.find_spec('neutts') is not None:
                neutts_available = True
            else:
                install_cmd = TTS_PROVIDERS[neutts_provider]['install']
                neutts_note = f"Requires NeuTTS package ({install_cmd})"

        # Check if model is cached
        if neutts_available:
            from .models import check_model_cached
            backbone = neutts2e_model
            if check_model_cached(neutts_provider, backbone, neutts_codec):
                neutts_note += " [cached]"
            else:
                neutts_note += " [needs download]"

        accessible[neutts_provider] = {
            "available": neutts_available,
            "note": neutts_note
        }

    # Piper - a small ONNX voice per language, so availability is just the package plus the file
    piper_available = False
    piper_note = "Fast offline VITS voices (no API key required)"

    if use_orcabot_playback:
        piper_note = "Disabled when Orcabot playback is enabled"
    elif importlib.util.find_spec('piper') is not None:
        piper_available = True
    else:
        piper_note = f"Requires Piper package ({TTS_PROVIDERS['piper']['install']})"

    if piper_available:
        from .models import check_model_cached
        if check_model_cached('piper', str(_shared_voice_for('piper') or piper_voice)):
            piper_note += " [cached]"
        else:
            piper_note += " [needs download]"

    accessible["piper"] = {
        "available": piper_available,
        "note": piper_note
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


def _orcabot_env_ready() -> bool:
    if requests is None:
        log_message("ERROR", "Orcabot playback requested but requests is not available")
        return False
    session_id = os.environ.get("ORCABOT_SESSION_ID")
    pty_id = os.environ.get("ORCABOT_PTY_ID")
    if not session_id or not pty_id:
        log_message("WARNING", "Orcabot playback requested but ORCABOT_SESSION_ID/ORCABOT_PTY_ID not set")
    return bool(session_id and pty_id)


def _emit_orcabot_audio(audio_path: Optional[str] = None,
                        audio_data: Optional[bytes] = None,
                        format: str = "mp3",
                        action: str = "play") -> bool:
    if requests is None:
        log_message("ERROR", "Orcabot playback requested but requests is not available")
        return False
    session_id = os.environ.get("ORCABOT_SESSION_ID")
    pty_id = os.environ.get("ORCABOT_PTY_ID")
    if not session_id or not pty_id:
        log_message("WARNING", "Orcabot playback requested but ORCABOT_SESSION_ID/ORCABOT_PTY_ID not set")
        return False

    body = {"action": action}
    if action == "play":
        if audio_path:
            body["path"] = audio_path
        elif audio_data:
            body["data"] = base64.b64encode(audio_data).decode()
            body["format"] = format

    try:
        resp = requests.post(
            f"http://localhost:8081/sessions/{session_id}/ptys/{pty_id}/audio",
            json=body,
            timeout=5
        )
        if resp.status_code != 204:
            log_message("ERROR", f"Orcabot playback failed: status={resp.status_code} body={resp.text[:200]}")
        return resp.status_code == 204
    except Exception as e:
        log_message("ERROR", f"Failed to emit audio via Orcabot: {e}")
        return False


def _emit_orcabot_status(payload: Dict[str, Any]) -> bool:
    if requests is None:
        log_message("ERROR", "Orcabot status requested but requests is not available")
        return False
    session_id = os.environ.get("ORCABOT_SESSION_ID")
    pty_id = os.environ.get("ORCABOT_PTY_ID")
    if not session_id or not pty_id:
        log_message("WARNING", "Orcabot status requested but ORCABOT_SESSION_ID/ORCABOT_PTY_ID not set")
        return False

    try:
        resp = requests.post(
            f"http://localhost:8081/sessions/{session_id}/ptys/{pty_id}/status",
            json=payload,
            timeout=5
        )
        if resp.status_code not in (200, 204):
            log_message("ERROR", f"Orcabot status post failed: status={resp.status_code} body={resp.text[:200]}")
        return resp.status_code in (200, 204)
    except Exception as e:
        log_message("ERROR", f"Failed to emit status via Orcabot: {e}")
        return False


def emit_orcabot_tts_status(enabled: bool,
                            initialized: bool,
                            mode: Optional[str],
                            provider: Optional[str],
                            voice: Optional[str]) -> bool:
    payload = {
        "action": "tts_status",
        "enabled": bool(enabled),
        "initialized": bool(initialized),
        "mode": mode or "unknown",
        "provider": provider or "unknown",
        "voice": voice or "unknown",
    }
    return _emit_orcabot_status(payload)


def _stop_orcabot_audio() -> bool:
    return _emit_orcabot_audio(action="stop")


def _play_audio_file_orcabot(audio_path: str, speech_item: Optional[SpeechItem]) -> bool:
    if not _orcabot_env_ready():
        return False

    path = Path(audio_path)
    if not path.exists():
        log_message("ERROR", f"Audio file not found: {audio_path}")
        return False

    ext = path.suffix.lower().lstrip('.')
    fmt = ext if ext in {"mp3", "wav", "ogg"} else "mp3"

    try:
        audio_bytes = path.read_bytes()
    except Exception as e:
        log_message("ERROR", f"Failed to read audio file for Orcabot: {e}")
        return False

    try:
        log_message("INFO", f"Orcabot playback: sending audio bytes format={fmt} size={len(audio_bytes)}")
        return _emit_orcabot_audio(audio_data=audio_bytes, format=fmt)
    finally:
        _mark_playback_finished(speech_item)


def _play_audio_file(audio_path: str, use_process_control: bool = True) -> bool:
    """Play audio file using non-blocking threaded playback.
    Returns immediately after starting the playback thread.
    """
    log_message("DEBUG", "_play_audio_file")
    with _state_lock:
        active_item = current_speech_item
    if use_orcabot_playback:
        return _play_audio_file_orcabot(audio_path, active_item)
    if not use_process_control:
        # For backward compatibility, if process control is disabled,
        # fall back to the old blocking behavior
        return _play_audio_file_blocking(audio_path, use_process_control, active_item)

    # Create and start playback thread
    playback_thread = AudioPlaybackThread(audio_path, playback_control, active_item)

    # Register the thread with playback control
    with playback_control.lock:
        # Stop any existing playback thread
        if playback_control.current_playback_thread:
            playback_control.current_playback_thread.stop()
        playback_control.current_playback_thread = playback_thread

    # Start the thread (non-blocking)
    playback_thread.start()

    # Return True immediately - the TTS worker can continue processing
    return True


def _play_audio_file_blocking(audio_path: str, use_process_control: bool = True, speech_item: Optional[SpeechItem] = None) -> bool:
    """Play audio file using an available system player (blocking version).
    This is the original blocking implementation, kept for fallback.
    """
    if speech_item is None:
        with _state_lock:
            speech_item = current_speech_item
    if use_orcabot_playback:
        return _play_audio_file_orcabot(audio_path, speech_item)

    try:
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
                log_message("DEBUG", "audio process started via original method")

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
                log_message("DEBUG", "audio process ended")
                playback_control.current_process = None
        _mark_playback_finished(speech_item)


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

def synthesize_and_play(synthesize_func, text: str, use_process_control: bool = True, needs_skip: bool = False) -> bool:
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

        if needs_skip:
            # Capture old thread before signaling skip
            with playback_control.lock:
                old_thread = playback_control.current_playback_thread

            playback_control.skip_current_item()

            # Wait for old playback to finish before starting new one preventing the skip flagging the new playback.
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=0.1)

            # Reset only skip_current (preserve skip_all if user requested it)
            with playback_control.lock:
                playback_control.skip_current = False

        if use_process_control:
            # For threaded playback, the AudioPlaybackThread will handle cleanup
            return _play_audio_file(tmp_path, use_process_control=use_process_control)
        else:
            # For blocking playback, clean up immediately after
            try:
                return _play_audio_file(tmp_path, use_process_control=use_process_control)
            finally:
                _cleanup_temp_file(tmp_path)

    except Exception as e:
        log_message("ERROR", f"Audio synthesis/playback failed: {e}")
        return False


def validate_provider_config(provider: str, silent: bool = False) -> bool:
    """Validate provider configuration and API keys.

    Args:
        provider: Provider name to validate
        silent: If True, suppress error messages (for auto-searching)
    """
    provider_info = TTS_PROVIDERS.get(provider)
    if not provider_info:
        return True  # System provider, no validation needed

    # Special case for Azure - requires both key and region
    if provider == 'azure':
        speech_key = os.environ.get('AZURE_SPEECH_KEY')
        speech_region = os.environ.get('AZURE_SPEECH_REGION')
        if not speech_key or not speech_region:
            if not silent:
                print("Error: Azure TTS requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables")
            return False
        return True

    # Check environment variable if required
    env_var = provider_info.get('env_var')
    if env_var and not os.environ.get(env_var):
        if not silent:
            print(f"Error: {env_var} environment variable not set")
            print(f"Please set it with: export {env_var}='your-api-key'")
        return False

    # For local providers, validate that the module is installed (but don't load the model yet)
    # This is lightweight - just checks if the module exists
    if provider in LOCAL_MODEL_PROVIDERS:
        try:
            with suppress_ai_warnings():
                if provider == 'kokoro':
                    import kokoro  # noqa: F401
                elif provider == 'kittentts':
                    import kittentts  # noqa: F401
                elif provider == 'neutts2e':
                    # find_spec avoids neutts' ~5s / ~466MB eager import of the neucodec torch stack
                    if importlib.util.find_spec('neutts') is None:
                        raise ImportError("neutts is not installed")
                elif provider == 'piper':
                    if importlib.util.find_spec('piper') is None:
                        raise ImportError("piper is not installed")
            log_message("DEBUG", f"Local provider {provider} module is installed")
            return True
        except ImportError:
            if not silent:
                install_cmd = TTS_PROVIDERS[provider]['install']
                print(f"Error: {provider} module is not installed.")
                print(f"Install it with: {install_cmd}")
            return False

    # Special case for AWS Polly - check AWS credentials
    if provider in ['aws', 'polly']:
        try:
            import boto3
            # Try to create a client to verify credentials
            test_client = boto3.client('polly', region_name=polly_region)
            test_client.describe_voices(LanguageCode='en-US')
        except ImportError:
            if not silent:
                print(f"Error: {provider_info['install']} required")
            return False
        except Exception as e:
            if not silent:
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
    
    def speak(self, text: str, use_process_control: bool = True, needs_skip: bool = False) -> bool:
        """Synthesize and play audio, return True if successful."""
        return synthesize_and_play(self.synthesize, text, use_process_control, needs_skip)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get config value from instance, shared state, or default."""
        # First check instance config
        if key in self.config:
            return self.config[key]
        
        # Then check shared state
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

        # Use broker URL if available, otherwise direct API
        base_url = os.environ.get('ELEVENLABS_BASE_URL', 'https://api.elevenlabs.io')
        url = f"{base_url}/v1/text-to-speech/{voice_id}"

        # Only include API key header if not going through broker
        headers = {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
        }
        if not base_url.startswith('http://localhost'):
            api_key = os.environ.get('ELEVENLABS_API_KEY')
            headers['xi-api-key'] = api_key
        
        data = {
            'text': text,
            'model_id': 'eleven_monolingual_v1',
            'voice_settings': {
                'stability': 0.5,
                'similarity_boost': 0.5
            }
        }

        try:
            req = Request(url,
                         data=json.dumps(data).encode('utf-8'),
                         headers={**headers, 'Content-Type': 'application/json'})
            with urlopen(req) as response:
                return response.read(), ".mp3"
        except HTTPError:
            return None


class DeepgramProvider(TTSProvider):
    """Deepgram TTS provider implementation."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        model = self.get_config_value('model', deepgram_voice_model)  # <-- use 'model'
        if not model:
            log_message("ERROR", "Deepgram missing model")
            return None

        # Use broker URL if available, otherwise direct API
        base_url = os.environ.get('DEEPGRAM_BASE_URL', 'https://api.deepgram.com')
        url = f"{base_url}/v1/speak?model={model}"

        headers = {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
        }
        # Only add auth header if not going through broker
        if not base_url.startswith('http://localhost'):
            api_key = os.environ.get('DEEPGRAM_API_KEY')
            if not api_key:
                log_message("ERROR", "Deepgram missing API key")
                return None
            headers['Authorization'] = f'Token {api_key}'
        try:
            req = Request(url,
                         data=json.dumps({'text': text}).encode('utf-8'),
                         headers=headers)
            with urlopen(req) as response:
                return response.read(), ".mp3"
        except HTTPError as e:
            log_message("ERROR", f"Deepgram error {e.code}: {e.read().decode('utf-8', errors='ignore')}")
            return None

class KittenTTSProvider(TTSProvider):
    """KittenTTS provider implementation."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            import soundfile as sf

            # Validate text to avoid ONNX BERT model errors
            if not text or not text.strip():
                log_message("WARNING", "KittenTTS: Empty text provided, skipping synthesis")
                return None

            # KittenTTS BERT model has issues with very short text (< 3 chars)
            if len(text.strip()) < 3:
                log_message("WARNING", f"KittenTTS: Text too short ('{text}'), padding to avoid BERT errors")
                text = text.strip() + "..."

            model_name = self.config.get('model') or kittentts_model
            m = get_cached_local_model('kittentts', variant=model_name)
            if m is None:
                raise RuntimeError("KittenTTS model unavailable")
            audio = m.generate(maybe_expand_numbers(text),
                               voice=self.get_config_value('voice', kittentts_voice))
            buf = io.BytesIO()
            sf.write(buf, _pcm_safe(audio), 24000, format='WAV')
            return buf.getvalue(), ".wav"
        except Exception as e:
            log_message("ERROR", f"KittenTTS synthesis error: {e}")
            return None

class KokoroTTSProvider(TTSProvider):
    """KokoroTTS provider implementation."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            # Instance config wins, then shared state, then module defaults
            config = get_tts_config()
            voice = self.config.get('voice') or config.get('voice') or kokoro_voice
            speed = float(self.config.get('speed') or config.get('speed') or kokoro_speed)
            language = self.config.get('language') or config.get('language') or kokoro_language
            pipeline = get_cached_local_model('kokoro', variant=_normalize_kokoro_lang(language))

            if pipeline is None:
                log_message("ERROR", "Failed to get cached Kokoro model")
                return None

            log_message("DEBUG", "Using cached KokoroTTS model")
            log_message("DEBUG", f"{text=} {voice=} {speed=}")

            # Generate audio with the specified voice and speed
            # Kokoro returns a generator, we need to process all chunks
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(
                    pipeline(maybe_expand_numbers(text), voice=voice, speed=speed)):
                log_message("DEBUG", f"Appending audio chunk {i}")
                audio_chunks.append(audio)

            # Concatenate all audio chunks
            import numpy as np
            if audio_chunks:
                import soundfile as sf
                full_audio = np.concatenate(audio_chunks)
                buf = io.BytesIO()
                sf.write(buf, _pcm_safe(full_audio), 24000, format='WAV')
                return buf.getvalue(), ".wav"
            else:
                log_message("ERROR", "KokoroTTS generated no audio")

        except Exception as e:
            log_message("ERROR", f"KokoroTTS synthesis error: {e}")
            return None

SPEECH_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
               "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
               "seventeen", "eighteen", "nineteen"]
SPEECH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
RE_SPEECH_NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)*')
RE_SPEECH_TIME = re.compile(r'\b(\d{1,2}):([0-5]\d)\b')
RE_SPEECH_ACRONYM = re.compile(r'\b[A-Z][A-Z0-9]{1,4}\b')

# Acronyms that are pronounced as words, so spelling them out would make them worse
ACRONYMS_SPOKEN_AS_WORDS = {
    'JSON', 'YAML', 'REST', 'SOAP', 'ASCII', 'CRUD', 'JPEG', 'RAID', 'NASA', 'SCUBA',
    'LAN', 'WAN', 'GIF', 'SIM', 'AJAX', 'ARIA', 'ONNX', 'CUDA', 'MIME', 'BIOS',
}

# All-caps words common in terminal output that are ordinary words, not acronyms
UPPERCASE_WORDS_NOT_ACRONYMS = {
    'ERROR', 'WARN', 'INFO', 'DEBUG', 'FATAL', 'TRACE', 'NOTE', 'TODO', 'FIXME', 'HACK',
    'PASS', 'FAIL', 'SKIP', 'OK', 'YES', 'NO', 'TRUE', 'FALSE', 'NULL', 'NONE', 'AND',
    'OR', 'NOT', 'IF', 'ELSE', 'FOR', 'NEW', 'ADD', 'GET', 'PUT', 'SET', 'RUN', 'END',
}


def _int_to_speech_words(n: int) -> str:
    """Spell out a non-negative integer; values of a million or more are read digit by digit."""
    if n < 20:
        return SPEECH_ONES[n]
    if n < 100:
        return (SPEECH_TENS[n // 10] + (" " + SPEECH_ONES[n % 10] if n % 10 else "")).strip()
    if n < 1000:
        rest = n % 100
        return (SPEECH_ONES[n // 100] + " hundred"
                + (" " + _int_to_speech_words(rest) if rest else "")).strip()
    if n < 1_000_000:
        rest = n % 1000
        return (_int_to_speech_words(n // 1000) + " thousand"
                + (" " + _int_to_speech_words(rest) if rest else "")).strip()
    return " ".join(SPEECH_ONES[int(d)] for d in str(n))


def _number_to_speech_words(token: str) -> str:
    """Convert one numeric token (possibly comma-grouped, decimal, or dotted) into words."""
    token = token.replace(',', '')
    if '.' not in token:
        return _int_to_speech_words(int(token)) if token.isdigit() else token

    parts = token.split('.')
    words = [_int_to_speech_words(int(parts[0]))] if parts[0].isdigit() else [parts[0]]
    for chunk in parts[1:]:
        words.append("point")
        # A short segment reads as a number (version-like); a long tail reads digit by digit
        if chunk.isdigit():
            words.append(_int_to_speech_words(int(chunk)) if len(chunk) <= 2
                         else " ".join(SPEECH_ONES[int(d)] for d in chunk))
    return " ".join(words)


def expand_acronyms_for_speech(text: str) -> str:
    """Space out letter-by-letter acronyms so BPE tokenizers do not read them as invented words.

    eSpeak spells unknown uppercase runs out automatically; BPE models split them into subwords
    and guess, turning 'SDK' into 'send god'. Words that happen to be uppercase and acronyms that
    are pronounced as words are left alone.

    Spelling the letters phonetically instead ("ess dee kay") measured no better than spacing them,
    so the simpler form stands.
    """
    def replace(match: 're.Match') -> str:
        token = match.group(0)
        if token in ACRONYMS_SPOKEN_AS_WORDS or token in UPPERCASE_WORDS_NOT_ACRONYMS:
            return token
        return " ".join(token)

    return RE_SPEECH_ACRONYM.sub(replace, text)


def normalize_for_bpe_speech(text: str) -> str:
    """Apply the text normalisation that phoneme-input models get from eSpeak for free."""
    return expand_acronyms_for_speech(expand_numbers_for_speech(text))


# Phoneme front ends expand digits themselves, but not always well; measured against the benchmark
# corpus this helps. Acronyms are deliberately excluded - spelling those out makes phoneme models
# worse, so only the number pass is applied here.
expand_numbers_for_phoneme_models = os.environ.get(
    'TALKITO_EXPAND_NUMBERS', '1').lower() in ('1', 'true', 'yes')


# Acronym expansion is off for phoneme models by default: eSpeak already spells unknown uppercase
# runs, and expanding them again measured worse. Kept as a toggle so the benchmark can retest it.
expand_acronyms_for_phoneme_models = os.environ.get(
    'TALKITO_EXPAND_ACRONYMS', '0').lower() in ('1', 'true', 'yes')


def maybe_expand_numbers(text: str) -> str:
    """Expand digits, and optionally acronyms, for a phoneme-input provider."""
    if expand_numbers_for_phoneme_models:
        text = expand_numbers_for_speech(text)
    if expand_acronyms_for_phoneme_models:
        text = expand_acronyms_for_speech(text)
    return text


def expand_numbers_for_speech(text: str) -> str:
    """Spell digits out as words for models that tokenize raw text instead of phonemes.

    Phoneme-input models (kokoro, kittentts, NeuTTS nano) get this from eSpeak automatically, but
    BPE-input models such as NeuTTS 2E receive the raw string and mis-speak bare digits.
    """
    # Clock times first, so the colon does not survive into the generic number pass
    def _speak_time(match: 're.Match') -> str:
        hour, minute = _int_to_speech_words(int(match.group(1))), match.group(2)
        if minute == '00':
            return f"{hour} o'clock"
        if minute[0] == '0':
            return f"{hour} oh {_int_to_speech_words(int(minute))}"
        return f"{hour} {_int_to_speech_words(int(minute))}"

    text = RE_SPEECH_TIME.sub(_speak_time, text)

    def _speak_number(match: 're.Match') -> str:
        words = _number_to_speech_words(match.group(0))
        # Keep 'v3' from becoming 'vthree' once the digits turn into words
        start = match.start()
        separator = " " if start and text[start - 1].isalpha() else ""
        return separator + words

    return RE_SPEECH_NUMBER.sub(_speak_number, text)


class NeuTTS2EProvider(TTSProvider):
    """NeuTTS 2E provider implementation, using its built-in speakers and emotions."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            import soundfile as sf

            if not text or not text.strip():
                log_message("WARNING", "NeuTTS2E: Empty text provided, skipping synthesis")
                return None

            config = get_tts_config()
            backbone = self.config.get('model') or neutts2e_model
            speaker = str(self.config.get('voice') or config.get('voice') or neutts2e_voice).lower()
            emotion = str(self.config.get('emotion') or neutts2e_emotion).lower()

            if speaker not in AVAILABLE_VOICES['neutts2e']['en-US']:
                log_message("WARNING", f"Unknown NeuTTS2E speaker {speaker!r}, falling back to 'emily'")
                speaker = 'emily'
            if emotion not in NEUTTS2E_EMOTIONS:
                log_message("WARNING", f"Unknown NeuTTS2E emotion {emotion!r}, falling back to 'neutral'")
                emotion = 'neutral'

            model = get_cached_local_model('neutts2e', variant=backbone)
            if model is None:
                raise RuntimeError("NeuTTS2E model unavailable")

            # 2E is a BPE-input model with no phonemizer, so digits and acronyms are spelled here
            spoken = normalize_for_bpe_speech(text)
            log_message("DEBUG", f"{spoken=} {speaker=} {emotion=}")
            # infer() prints the sampling seed, so it needs the same treatment as loading: this
            # runs per utterance, in the worker thread, while the wrapped program owns the terminal
            with _quiet_model_load('neutts2e'):
                audio = model.infer(spoken, speaker=speaker, emotion=emotion)

            buf = io.BytesIO()
            sf.write(buf, _pcm_safe(audio), NEUTTS_SAMPLE_RATE, format='WAV')
            return buf.getvalue(), ".wav"
        except Exception as e:
            log_message("ERROR", f"NeuTTS2E synthesis error: {e}")
            return None


class PiperTTSProvider(TTSProvider):
    """Piper provider implementation, a VITS model driven by eSpeak phonemes."""

    def synthesize(self, text: str) -> Optional[Tuple[bytes, str]]:
        try:
            import wave

            if not text or not text.strip():
                log_message("WARNING", "Piper: Empty text provided, skipping synthesis")
                return None

            voice = str(self.config.get('voice') or _resolve_piper_voice())

            model = get_cached_local_model('piper', variant=voice)
            if model is None:
                raise RuntimeError("Piper model unavailable")

            # Piper phonemizes through eSpeak, which expands digits itself
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav_file:
                model.synthesize_wav(text, wav_file)
            return buf.getvalue(), ".wav"
        except Exception as e:
            log_message("ERROR", f"Piper synthesis error: {e}")
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
    'neutts2e': NeuTTS2EProvider,
    'piper': PiperTTSProvider,
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
    text = RE_FILENAME_PATH.sub(extract_filename, text)

    # This regex matches filenames with extensions (e.g., talk.py -> talk dot py)
    text = RE_FILE_EXTENSION.sub(r'\1 dot \2', text)

    # Remove bullet points (- or + at start of line)
    text = RE_LIST_MARKERS.sub('', text)

    # Handle hashtag/asterix symbol
    text = RE_MULTIPLE_ASTERISKS.sub('', text)
    text = RE_MULTIPLE_HASHES.sub('hashtags', text)
    text = RE_MULTIPLE_QUESTIONS.sub('questionmarks', text)
    text = RE_SINGLE_HASH.sub(' hashtag ', text)

    # Handle double dash (--) when it precedes a word (e.g., --option)
    text = RE_DASH_WORD.sub(r' dash dash \1', text)

    # Handle dates (preserve the pattern for later processing)
    text = RE_DATE_FORMAT.sub(r'\1 to \2 to \3', text)

    # Replace negative numbers (-5 -> minus 5)
    text = RE_NEGATIVE_NUMBER.sub(r' minus \1', text)

    # Replace subtraction with spaces around it (5 - 3 -> 5 minus 3)
    text = RE_NUMBER_MINUS_NUMBER.sub(r'\1 minus \2', text)

    # Replace dashes surrounded by spaces with dash (word - word -> word dash word)
    text = RE_WORD_DASH.sub(r'\1 dash ', text)

    # Replace ranges (10-20 -> 10 to 20) - must come after spaced subtraction
    text = RE_NUMBER_RANGE.sub(r'\1 to \2', text)

    # Replace plus in math contexts
    text = RE_PLUS_SIGN.sub(' plus ', text)

    # Division: number / number with spaces (e.g., "10 / 2")
    text = RE_DIVISION.sub(r'\1 divided by \2', text)

    # URLs: drop all http/https URLs before slash handling
    text = RE_HTTP_URL_ALL.sub('', text)

    # Commands that start with slash (e.g., /install-github-app, /help)
    text = RE_SLASH_PATH.sub(r'slash \1', text)

    # Remaining "Either/or" pattern - only for simple word/word without dots or extensions
    text = RE_SLASH_OR.sub(r'\1 or \2', text)

    # Starting with any '>'
    text = RE_QUOTE_PREFIX.sub('', text)

    # Word and acronym replacements
    text = RE_CWD.sub(' current working directory ', text)
    text = RE_USAGE.sub('Use as follows:', text)
    text = RE_TODOS.sub('To do\'s', text)

    # Handle placeholder patterns like <from>, <to> by removing brackets
    text = RE_ANGLE_BRACKETS.sub(r'\1', text)

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
    text = RE_PUNCT_PERIOD_SPACE.sub(r'\1 ', text)
    # Replace punctuation followed by period at end with just the punctuation
    text = RE_PUNCT_PERIOD_END.sub(r'\1', text)
    # Only clean up multiple periods with spaces between (like ". ." or ". . .")
    # This preserves intentional ellipsis like "..." or ".."
    text = RE_MULTIPLE_PERIODS.sub('.', text)
    return text


def extract_speakable_text(text: str) -> str:
    """Extract speakable text and convert mathematical symbols."""

    # Drop OAuth/login URLs or query fragments entirely
    if RE_HTTP_URL_ALL.search(text) or RE_SUSPICIOUS_QUERY.search(text) or RE_PERCENT_ENCODE.search(text):
        return ""

    text = RE_GREATER_UNDERSCORE.sub('', text)

    # Replace newlines with periods for better speech flow
    text = RE_NEWLINES.sub('. ', text)

    # Convert function names: underscores to spaces, drop ()
    text = RE_FUNCTION_CALL.sub(lambda m: ' '.join(m.group(0).replace('_', ' ').replace('()', '').split()), text)
    text = RE_UNDERSCORE_WORDS.sub(lambda m: ' '.join(m.group(0).split('_')), text)

    # Remove bracketed comments at the end
    text = RE_TRAILING_PARENS.sub('', text)

    # Remove markdown formatting
    text = RE_MARKDOWN_BOLD.sub(r'\1', text)  # Bold
    text = RE_MARKDOWN_ITALIC.sub(r'\1', text)  # Italic
    text = RE_MARKDOWN_CODE.sub(r'\1', text)  # Code

    # Apply context-aware symbol replacement and clean up text
    text = context_aware_symbol_replacement(text)
    text = RE_NON_SPEECH_CHARS.sub('', text)  # Keep bullet points in allowed chars
    text = RE_MULTIPLE_SPACES.sub(' ', text)
    text = text.strip()

    spoken_text = text

    # Skip text that contains no letters (just punctuation, numbers, spaces)
    if not RE_HAS_LETTERS.search(text):
        return ""

    return spoken_text


def is_similar_to_recent(text: str) -> bool:
    """Check if text is similar to recently spoken text."""

    with _cache_lock:
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


def speak_text(text: str, engine: str, needs_skip: bool = False) -> bool:
    """Speak text using appropriate engine, return completion status."""
    if disable_tts:
        return True

    # Get TTS provider from shared state or fallback to global
    config = get_tts_config()
    current_provider = config.get('provider') or tts_provider or 'system'

    # Use provider classes directly
    provider = create_tts_provider(current_provider)
    if provider:
        try:
            result = provider.speak(text, use_process_control=True, needs_skip=needs_skip)
            return result
        except Exception as e:
            log_message("ERROR", f"TTS provider {current_provider} failed: {e}")
            log_message("ERROR", f"Traceback: {traceback.format_exc()}")
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
        # No number expansion here: system engines already normalise digits well, and expanding
        # first measured no better (numeric WER 5% -> 4%, identical pass counts)
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
        log_message("DEBUG", f"Started {engine} audio process with PID: {process.pid}")
        with playback_control.lock:
            playback_control.current_process = process
        
        # Wait for completion or interruption
        try:
            if engine in ["festival", "flite"] and "input" in kwargs:
                # For stdin-based engines, we can't easily poll, so just communicate
                stdout, stderr = process.communicate(input=kwargs["input"])
            else:
                log_message("DEBUG", "Poll process with timeout to check for interruptions")
                # Poll process with timeout to check for interruptions
                while process.poll() is None:
                    # Check if we should stop
                    if shutdown_event.is_set() or playback_control.skip_current or playback_control.skip_all:
                        log_message("DEBUG", ";looks like we should stop")
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
                log_message("DEBUG", "audio process is now none")
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
            while (playback_control.is_paused or _any_model_loading() or tts_queue.empty()) and not shutdown_event.is_set():
                time.sleep(0.1)

            text_to_speak = ""
            speech_item: Optional[SpeechItem] = None
            breakout = False
            while not tts_queue.empty():
                # Get next item from queue
                item = tts_queue.get()

                if item == "__SHUTDOWN__":
                    breakout = True
                    break

                # Handle both old string format and new SpeechItem format
                if isinstance(item, str):
                    speech_item = SpeechItem(text=item, original_text=item)
                else:
                    speech_item = item

                if speech_item:
                    if text_to_speak:
                        text_to_speak += ". "
                    text_to_speak += speech_item.text
            if breakout:
                break

            if speech_item is None:
                continue

            # Tell ASR to ignore input while we're speaking to prevent feedback
            try:
                from . import asr
                if hasattr(asr, 'set_ignore_input'):
                    asr.set_ignore_input(True)
                    log_message("DEBUG", "Set ASR to ignore input during TTS")
            except Exception as e:
                log_message("DEBUG", f"Could not set ASR ignore flag: {e}")

            # Check for auto-skip before starting audio generation
            needs_skip = False


            model_loading = _any_model_loading()
            log_message("DEBUG",
                        f"Auto-skip check: {auto_skip_tts_enabled=} {model_loading=} {tts_queue.empty()=}, current_speech_item={current_speech_item is not None}")

            if auto_skip_tts_enabled and not model_loading:
                # Check if something is currently playing and how long it's been playing
                is_currently_speaking = is_speaking()
                is_currently_playing = playback_control.current_process is not None
                playing_long_enough = False
                time_playing = None
                if is_currently_playing and current_speech_item and current_speech_item.start_time:
                    time_playing = time.time() - current_speech_item.start_time
                    playing_long_enough = time_playing >= 1.0  # Minimum 1 second
                    log_message("DEBUG", f"Current item has been playing for {time_playing:.2f} seconds")

                log_message("DEBUG", f"{is_currently_speaking=} {is_currently_playing=} {playing_long_enough=} {time_playing=}")

                # Only skip current item if something is actually speaking, a process exists, and neither item is an exception
                current_is_exception = bool(current_speech_item and current_speech_item.is_exception)

                if current_is_exception:
                    log_message("INFO", "Current speech item is an exception; skipping auto-skip")

                if is_currently_speaking and is_currently_playing and not current_is_exception:
                    needs_skip = True
                    log_message("INFO", f"Auto-skipping current audio for new text ({len(text_to_speak)} chars)")

                    # Add an interjection only if the current item has been playing long enough
                    if playing_long_enough:
                        interjection = random.choice(SKIP_INTERJECTIONS)
                        # Prepend the interjection to the text for TTS only
                        text_to_speak = f"{interjection}, {text_to_speak}"
                        log_message("INFO", f"Added interjection '{interjection}' for smoother transition")
                    else:
                        log_message("DEBUG", "Skipping interjection - current item hasn't played long enough")
                elif is_currently_playing:
                    log_message("WARNING", "Process exists but not actually speaking - race condition detected!")
                else:
                    log_message("DEBUG", "Nothing currently playing - no auto-skip needed")

            # Set current speech item with thread safety and record start time
            with _state_lock:
                speech_item.start_time = time.time()
                current_speech_item = speech_item

            playback_started = False
            if text_to_speak.strip():
                # Include line number in log if available
                if speech_item.line_number is not None:
                    log_message("INFO", f"Speaking via {engine} (line {speech_item.line_number}) qsize {tts_queue.qsize()}: '{text_to_speak}'")
                else:
                    log_message("INFO", f"Speaking via {engine}: '{text_to_speak}'")

                playback_started = speak_text(text_to_speak, engine, needs_skip)
                if disable_tts:
                    playback_started = False  # Ensure cleanup runs when TTS disabled (no actual playback)

            # Check if we should skip current or if shutdown was requested
            if playback_control.skip_current or shutdown_event.is_set():
                playback_control.reset_skip_flags()
                if shutdown_event.is_set():
                    break

            if not playback_started:
                with _state_lock:
                    if speech_item and current_speech_item is speech_item:
                        current_speech_item = None
                    last_speech_end_time = time.time()
            
            # Resume ASR input after speaking (but only for appropriate modes)
            try:
                from . import asr

                if hasattr(asr, 'set_ignore_input'):
                    # Small delay to ensure TTS audio has fully finished
                    time.sleep(0.2)
                    
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
            log_message("ERROR", f"Traceback: {traceback.format_exc()}")

def _retry_delayed_speech(speech_item: SpeechItem, callback: Optional[Callable[[str], None]] = None, forced: bool = False):
    global _delayed_timer, _delayed_speech_item, last_queued_text
    log_message("DEBUG", f"retry_delayed_speech [{speech_item.text}] {forced=}")
    with _delayed_items_lock:
        with _state_lock:
            if _delayed_timer:
                last_queued_text = ""
                _delayed_timer = None
                _delayed_speech_item = None
            try:
                # Skip bullet numbering - just filter bullets out in text processing
                # numbered_text = apply_bullet_numbering(speech_item.text)
                # speech_item.text = numbered_text

                with _cache_lock:
                    spoken_cache.append((speech_item.text, time.time()))
                tts_queue.put_nowait(speech_item)
                speech_history.append(speech_item)
                # Call the callback now that we're actually going to speak it
                if callback:
                    callback(speech_item.original_text)
            except queue.Full:
                log_message("WARNING", "TTS queue full, skipping text")

def _text_is_novel(new_text: str, old_text: str) -> bool:
    if not old_text:
        return True
    with _state_lock:
        if new_text == old_text:
            return False

        if old_text in new_text:
            return False

        if SequenceMatcher(None, new_text, old_text).ratio() > 0.7:
            return False

        return True


def queue_for_speech(original_text: str, line_number: Optional[int] = None, source: str = "output", exception_match: bool = False, writes_partial_output: bool = False, callback: Optional[Callable[[str], None]] = None, constituent_parts: Optional[List[str]] = None) -> str:
    """Queue text for TTS with debouncing and filtering."""
    global highest_spoken_line_number, last_queued_text, last_queue_time, _delayed_timer, _delayed_speech_item

    # Check shared state if available
    shared_state = get_shared_state()
    if not shared_state.tts_enabled:
        log_message("DEBUG", "TTS disabled in shared state, not queueing speech")
        return ""

    # Log the original text before any filtering
    log_message("INFO", f"queue_for_speech received: '{original_text}' (exception_match={exception_match}, has_constituent_parts={constituent_parts is not None})")

    speakable_text = extract_speakable_text(original_text)

    if not speakable_text or len(speakable_text) < MIN_SPEAK_LENGTH:
        log_message("INFO", f"Text too short: '{original_text}' -> '{speakable_text}'")
        return ""
    
    # Clean up any awkward punctuation sequences
    speakable_text = clean_punctuation_sequences(speakable_text)

    # Get current time for various time-based checks
    current_time = time.time()
    delay = False
    if writes_partial_output:
        log_message("INFO", f"_text_is_novel {speakable_text.lower()} {last_queued_text.lower()} = {_text_is_novel(speakable_text.lower(), last_queued_text.lower())} and _delayed_speech_item {_delayed_speech_item is not None}")
        text_is_novel = _text_is_novel(speakable_text.lower()[:-1], last_queued_text.lower()[:-1])
        if text_is_novel:
            if is_similar_to_recent(speakable_text):
                log_message("DEBUG", "On further inspection text is similar to something already spoken")
                text_is_novel = False

        if text_is_novel and _delayed_speech_item:
            log_message("INFO", f"The new text is novel {text_is_novel} so we early execute what was being kept back")
            with _delayed_items_lock:
                if _delayed_timer:
                    _delayed_timer.cancel()
            _retry_delayed_speech(_delayed_speech_item, callback, forced = True)

        if is_similar_to_recent(speakable_text):
            log_message("INFO", f"Recently spoken: '{speakable_text}'")
            return ""

        with _delayed_items_lock:
            if _delayed_timer:
                log_message("DEBUG", "Clearing out a delayed item as a partial of the new text")
                last_queued_text = ""  # If we've cancelled pending text then we don't want to debounce due to similarity
                _delayed_timer.cancel()
                _delayed_timer = None
                _delayed_speech_item = None

        with _state_lock:
            delay = True
            log_message("INFO", f"Will delay incomplete sentence: '{original_text}' by {2000}ms")

    else:
        # Always check for exact duplicates of the last spoken text
        with _state_lock:
            time_since_last = current_time - last_queue_time
            if last_queued_text and speakable_text == last_queued_text and (not exception_match or time_since_last < 5.0):
                log_message("INFO", f"Skipping exact duplicate of last spoken text: '{speakable_text}' (time_since_last={time_since_last:.1f}s, exception_match={exception_match})")
                return ""

        # Check if we're in tool use mode (between PreToolUse and PostToolUse hooks)
        shared_state = get_shared_state()
        in_tool_use = shared_state.get_in_tool_use()

        # Skip similarity checks if we're in tool use mode (prompting the user)
        if not (in_tool_use and "Do you " in speakable_text):
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

            # Check if half or more of constituent parts are already in cache
            if constituent_parts and len(constituent_parts) > 1:
                matches = 0
                for part in constituent_parts:
                    if part and part.strip():
                        if is_similar_to_recent(part):
                            matches += 1
                            log_message("DEBUG", f"Constituent part matches cache: '{part[:50]}...'")

                if matches >= len(constituent_parts) // 2:
                    log_message("INFO", f"Skipping text - {matches}/{len(constituent_parts)} constituent parts already in cache")
                    return ""
        else:
            log_message("INFO", f"Bypassing similarity checks for exception pattern match: '{speakable_text}'")

    # Auto-skip logic now handled in tts_worker for seamless switching
    # This avoids gaps between skipping current audio and generating new audio

    with _state_lock:
        last_queued_text = speakable_text
        last_queue_time = current_time

        if line_number is not None and line_number <= highest_spoken_line_number:
            log_message("INFO",
                        f"Skipping line {line_number} (already spoken up to line {highest_spoken_line_number}): '{speakable_text[:50]}...'")
            return ""

        if not delay and line_number is not None and line_number > highest_spoken_line_number:
            highest_spoken_line_number = line_number
            log_message("DEBUG", f"Updated highest_spoken_line_number to {highest_spoken_line_number}")

    with _state_lock:
        # Create speech item
        speech_item = SpeechItem(
            text=speakable_text,
            original_text=original_text,
            line_number=line_number,
            source=source,
            is_exception=exception_match
        )

        try:
            if delay:
                with _delayed_items_lock:
                    _delayed_timer = threading.Timer(2.0, _retry_delayed_speech, args=(speech_item,))
                    _delayed_speech_item = speech_item
                    _delayed_timer.start()
                log_message("INFO", f"Delaying incomplete sentence for {2000}ms: '{speakable_text}'")
                # TODO: We need a callback instead of a return for queue_output so core.py can also use the speakable text when appropriate
                return "--ignore--"
            else:
                # Skip bullet numbering - just filter bullets out in text processing
                # numbered_text = apply_bullet_numbering(speech_item.text)
                # speech_item.text = numbered_text

                # Use cache lock for cache operations
                with _cache_lock:
                    spoken_cache.append((speech_item.text, current_time))
                    # Also add constituent parts to cache to prevent future duplicates
                    if constituent_parts:
                        for part in constituent_parts:
                            if part and part.strip():  # Only add non-empty parts
                                spoken_cache.append((part, current_time))
                tts_queue.put_nowait(speech_item)
                # Thread-safe append to history
                speech_history.append(speech_item)
        except queue.Full:
            log_message("WARNING", "TTS queue full, skipping text")

    return speakable_text


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


def clear_tts_queue_only():
    """Clear TTS queue and state without clearing spoken_cache.

    Use this when changing TTS providers to avoid re-speaking content
    that was already spoken with the previous provider.
    """
    global last_queued_text, last_queue_time

    # Clear the queue to remove pending items
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break

    # Reset state variables but NOT spoken_cache
    with _state_lock:
        last_queued_text = ""
        last_queue_time = 0
        # Note: We intentionally don't reset highest_spoken_line_number
        # or clear spoken_cache to avoid repeating content

    log_message("INFO", "TTS queue cleared (spoken_cache preserved)")


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
    
    # Keep waiting while there is any work outstanding
    while True:
        queue_empty = tts_queue.empty()
        with _state_lock:
            active_item = current_speech_item
        with playback_control.lock:
            process_active = playback_control.current_process is not None
            thread_active = playback_control.current_playback_thread is not None

        if queue_empty and active_item is None and not process_active and not thread_active:
            break

        if timeout and (time.time() - start_time) > timeout:
            return False

        time.sleep(0.05)
    
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
    if use_orcabot_playback:
        _stop_orcabot_audio()
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
                log_message("DEBUG", "audio process finished")
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

            # Skip bullet numbering - just filter bullets out in text processing
            # numbered_text = apply_bullet_numbering(prev_item.text)
            # prev_item.text = numbered_text

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

    # Check if TTS process is still running
    with playback_control.lock:
        if playback_control.current_process is not None:
            if playback_control.current_process.poll() is None:
                # Process is still running
                log_message("INFO", "TTS playback is still running")
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
                       choices=['system', 'openai', 'aws', 'polly', 'azure', 'gcloud', 'elevenlabs',
                                'deepgram', 'kittentts', 'kokoro', 'neutts2e', 'piper'],
                       help='TTS provider to use (default: auto-select best available)')
    parser.add_argument('--voice', type=str, default=None,
                       help='Voice to use (provider-specific)')
    parser.add_argument('--language', type=str, default='en-US',
                       help='Language code (mainly for Google Cloud TTS, default: en-US)')
    parser.add_argument('--region', type=str, default=None,
                       help='Region for cloud providers (AWS Polly/Azure)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Enable debug logging to file (e.g., ~/.talkito_tts.log)')
    parser.add_argument('--orcabot', action='store_true',
                       help='Play audio via Orcabot HTTP playback instead of local audio player')
    parser.add_argument('text', nargs='*', help='Text to speak (multiple words allowed)')
    return parser.parse_args()


def select_best_tts_provider(excluded_providers=None) -> str | None:
    """Select best available TTS provider by preference order with thorough validation."""
    log_message("INFO", "select_best_tts_provider called")
    excluded_providers = excluded_providers or set()
    
    # Check shared state first
    state = get_shared_state()
    state_provider = state.tts_provider

    preferred = state_provider or os.environ.get('TALKITO_PREFERRED_TTS_PROVIDER')

    accessible = check_tts_provider_accessibility(requested_provider=preferred)

    # Check if preferred provider is accessible, properly configured, and not excluded
    if preferred and preferred in accessible and accessible[preferred]['available'] and preferred not in excluded_providers:
        if validate_provider_config(preferred):
            log_message("INFO", f"Using preferred TTS provider: {preferred}")
            return preferred
        else:
            print(f"Warning: Preferred TTS provider '{preferred}' is not properly configured. Searching for alternatives...")
            log_message("WARNING", f"Preferred TTS provider {preferred} failed validation, searching for alternatives")
    
    # Get all accessible providers except system and excluded providers
    available_providers = [
        provider for provider, info in sorted(accessible.items())
        if info['available'] and provider != 'system' and provider not in excluded_providers
    ]
    
    # Validate each provider thoroughly before selecting (silent mode to avoid spam)
    for provider in available_providers:
        if validate_provider_config(provider, silent=True):
            print(f"Selected {provider} as TTS provider")
            log_message("INFO", f"Selected TTS provider: {provider} (first validated)")
            return provider
        else:
            log_message("WARNING", f"TTS provider {provider} failed validation, trying next")
    
    # Try kokoro as ultimate fallback before system (if available and not excluded)
    if 'kokoro' not in excluded_providers and accessible.get('kokoro', {}).get('available'):
        if validate_provider_config('kokoro', silent=True):
            log_message("INFO", "Falling back to kokoro TTS provider")
            return 'kokoro'
        else:
            log_message("WARNING", "Kokoro TTS provider failed validation")

    # Fall back to system if available and not excluded
    if 'system' not in excluded_providers and accessible.get('system', {}).get('available'):
        if validate_provider_config('system', silent=True):
            log_message("INFO", "Falling back to system TTS provider")
            return 'system'
        else:
            log_message("WARNING", "System TTS provider failed validation")

    # No valid TTS providers available
    log_message("ERROR", "No TTS providers available or all validation failed")
    return None


def configure_tts_from_args(args) -> bool:
    """Configure TTS provider from config dictionary."""
    global disable_tts, tts_provider, use_orcabot_playback, openai_voice, polly_voice, polly_region, azure_voice, azure_region, gcloud_voice, gcloud_language_code, elevenlabs_voice_id, elevenlabs_model_id, deepgram_voice_model, kittentts_model, kittentts_voice
    
    tts_provider = args.tts_provider
    use_orcabot_playback = bool(getattr(args, "orcabot", False))
    if use_orcabot_playback:
        log_message("INFO", "Orcabot playback enabled")
        os.environ["TALKITO_ORCABOT_ENABLED"] = "1"
        if tts_provider in LOCAL_MODEL_PROVIDERS:
            log_message("WARNING", f"Orcabot playback disables local TTS provider {tts_provider}; selecting fallback")
            fallback_provider = select_best_tts_provider(excluded_providers=set(LOCAL_MODEL_PROVIDERS))
            if fallback_provider is None:
                log_message("ERROR", "No non-local TTS providers available with Orcabot playback")
                disable_tts_completely("no non-local TTS providers available with Orcabot playback", args)
                return False
            tts_provider = fallback_provider
            setattr(args, "tts_provider", fallback_provider)

    # If no provider survived selection, disable TTS instead of pretending configuration succeeded.
    if not tts_provider:
        log_message("WARNING", "No TTS provider available. Disabling TTS.")
        disable_tts = True
        if hasattr(args, 'disable_tts'):
            args.disable_tts = True
        shared_state = get_shared_state()
        shared_state.set_tts_enabled(False)
        shared_state.set_tts_config(provider=None)
        return False
    
    # Validate provider configuration
    if not validate_provider_config(tts_provider):
        log_message("WARNING", f"TTS provider {tts_provider} validation failed")
        # Fall back to best available provider
        fallback_provider = select_best_tts_provider()
        if fallback_provider is None:
            log_message("ERROR", "No TTS providers available")
            return False
        if fallback_provider != tts_provider:
            log_message("INFO", f"Falling back to TTS provider: {fallback_provider}")
            print(f"Warning: Falling back from {tts_provider} to {fallback_provider}")
            print(f"Provider-specific settings (region, voice) will be reset to defaults")
            provider = fallback_provider
            tts_provider = provider
            setattr(args, 'tts_provider', fallback_provider)
            # Reset provider-specific args to let fallback use its defaults
            if hasattr(args, 'tts_region'):
                args.tts_region = None
            if hasattr(args, 'tts_voice'):
                args.tts_voice = None
            log_message("INFO", f"Reset provider-specific args for fallback to {fallback_provider}")
            # Validate fallback provider
            if not validate_provider_config(provider):
                log_message("ERROR", f"Fallback TTS provider {fallback_provider} also failed")
                return False
        else:
            return False
    
    # Get provider info
    provider_info = TTS_PROVIDERS.get(tts_provider)
    if not provider_info:
        # System provider
        log_message("INFO", "Using system default TTS")
        return True
    
    # Handle provider-specific configuration
    if tts_provider == 'openai':
        if args.tts_voice:
            openai_voice = args.tts_voice
        log_message("INFO", f"Using OpenAI TTS with voice: {openai_voice}")
        
    elif tts_provider in ['aws', 'polly']:
        # Additional AWS validation
        try:
            import boto3
            # Use global default if region not provided via CLI
            region_to_use = args.tts_region or polly_region
            test_client = boto3.client('polly', region_name=region_to_use)
            test_client.describe_voices(LanguageCode='en-US')
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            return False
        except Exception as e:
            print(f"Error: AWS credentials not configured or invalid: {e}")
            print("Please configure AWS credentials (e.g., AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)")
            return False

        polly_region = region_to_use
        if args.tts_voice:
            polly_voice = args.tts_voice
        log_message("INFO", f"Using AWS Polly TTS with voice: {polly_voice} in region: {polly_region}")
        
    elif tts_provider == 'azure':
        # Additional Azure validation
        try:
            import azure.cognitiveservices.speech as speechsdk  # noqa: F401
        except ImportError:
            print(f"Error: {provider_info['install']} required")
            return False

        # Verify Azure API key is configured
        speech_key = os.environ.get('AZURE_SPEECH_KEY')
        if not speech_key:
            print("Error: Azure TTS requires AZURE_SPEECH_KEY environment variable")
            return False

        # Accept region from CLI args, AZURE_SPEECH_REGION, AZURE_REGION, or global default
        region_to_use = args.tts_region or os.environ.get('AZURE_SPEECH_REGION') or azure_region
        azure_region = region_to_use
        if args.tts_voice:
            azure_voice = args.tts_voice
        log_message("INFO", f"Using Microsoft Azure TTS with voice: {azure_voice} in region: {azure_region}")
        
    elif tts_provider == 'gcloud':
        # Additional Google Cloud validation
        try:
            from google.cloud import texttospeech  # noqa: F401
            texttospeech.TextToSpeechClient()  # Test instantiation to verify availability
        except ImportError:
            message = f"Error: {provider_info['install']} required"
            note = (
                "Note: On macOS, grpcio may require a clean reinstall:\n"
                "      pip uninstall grpcio && pip install grpcio --force-reinstall --no-cache-dir"
            )
            if is_orcabot_mode():
                emit_orcabot_notice(message, level="error", category="tts")
                emit_orcabot_notice(note, level="warning", category="tts")
            else:
                print(message)
                print(note)
            return False
        except Exception as e:
            message = f"Error: Google Cloud credentials not configured or invalid: {e}"
            note = "Please check your GOOGLE_APPLICATION_CREDENTIALS file"
            if is_orcabot_mode():
                emit_orcabot_notice(message, level="error", category="tts")
                emit_orcabot_notice(note, level="warning", category="tts")
            else:
                print(message)
                print(note)
            return False
            
        gcloud_language_code = args.tts_language
        if args.tts_voice:
            gcloud_voice = args.tts_voice
        log_message("INFO", f"Using Google Cloud TTS with voice: {gcloud_voice} in language: {gcloud_language_code}")
        
    elif tts_provider == 'elevenlabs':
        if args.tts_voice:
            elevenlabs_voice_id = args.tts_voice
        log_message("INFO", f"Using ElevenLabs TTS with voice ID: {elevenlabs_voice_id}")
    
    elif tts_provider == 'deepgram':
        if args.tts_voice:
            deepgram_voice_model = args.tts_voice
        log_message("INFO", f"Using Deepgram TTS with model: {deepgram_voice_model}")
    
    elif tts_provider == 'kittentts':
        # Additional KittenTTS validation
        try:
            with suppress_ai_warnings():
                from kittentts import KittenTTS  # noqa: F401
        except ImportError:
            install_cmd = TTS_PROVIDERS['kittentts']['install']
            print("Error: KittenTTS dependencies not installed")
            print(f"Please install with: {install_cmd}")
            return False
        
        if args.tts_voice:
            kittentts_voice = args.tts_voice
        log_message("INFO", f"Using KittenTTS with model: {kittentts_model} and voice: {kittentts_voice}")
        
        # Background preloading started earlier in initialization
        
    elif tts_provider == 'neutts2e':
        # Skip the expensive neutts import - it is validated during actual model loading
        log_message("DEBUG", f"Skipping {tts_provider} validation - will validate during model loading")

        global neutts2e_voice, neutts2e_model
        if args.tts_voice:
            neutts2e_voice = args.tts_voice
        log_message("INFO", f"Using NeuTTS 2E with model: {neutts2e_model} and speaker: {neutts2e_voice}")

        # Background preloading started earlier in initialization

    elif tts_provider == 'piper':
        # Skip the piper import - it is validated during actual model loading
        log_message("DEBUG", f"Skipping {tts_provider} validation - will validate during model loading")

        global piper_voice
        if args.tts_voice:
            piper_voice = args.tts_voice
        log_message("INFO", f"Using Piper TTS with voice: {piper_voice}")

        # Background preloading started earlier in initialization

    elif tts_provider == 'kokoro':
        # Skip expensive kokoro import - will validate during actual model loading
        log_message("DEBUG", "Skipping kokoro validation in configure_tts_from_dict - will validate during model loading")
        
        # Update global config variables
        global kokoro_voice, kokoro_language, kokoro_speed
        if args.tts_voice:
            kokoro_voice = args.tts_voice
        if args.tts_language:
            kokoro_language = args.tts_language
        if args.tts_rate:
            kokoro_speed = args.tts_rate
        log_message("INFO", f"Using KokoroTTS with language: {kokoro_language} and voice: {kokoro_voice}")
        
        # Background preloading started earlier in initialization
    
    # Update shared state with the actually working provider
    shared_state = get_shared_state()
    shared_state.set_tts_config(provider=tts_provider)
    
    return True


# Main entry point for testing
if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()

    # Normalize args for configure_tts_from_args (it expects tts_voice/tts_region/tts_language)
    args.tts_voice = getattr(args, 'voice', None)
    args.tts_region = getattr(args, 'region', None)
    args.tts_language = getattr(args, 'language', 'en-US')
    args.tts_rate = None  # Not supported in standalone mode

    # Auto-select provider if not specified
    if args.tts_provider is None:
        args.tts_provider = select_best_tts_provider()
        if args.tts_provider is None:
            print("No TTS provider available. TTS will be disabled.")
            exit(1)
        print(f"Auto-selected TTS provider: {args.tts_provider}")

    # Configure TTS provider (use same function as main application)
    if not configure_tts_from_args(args):
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
            engine = tts_provider  # Use the actual provider name as engine
        
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
            engine = tts_provider  # Use the actual provider name as engine
        
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
