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
Talkito - Universal TTS wrapper and voice interaction library

This package provides both a command-line interface and a Python API for:
- Text-to-Speech (TTS) with multiple providers
- Automatic Speech Recognition (ASR) with multiple providers  
- Remote communication via SMS, WhatsApp, and Slack
- Voice-enabled interaction with any command-line program
"""

from .__version__ import __version__, __author__, __email__, __license__

import warnings
warnings.filterwarnings("ignore", message=r".*in sys\.modules.*talkito\.(asr|tts|mcp|comms|profiles).*", category=RuntimeWarning)

from . import asr, comms, profiles, tts

# TTS API
from .tts import (
    detect_tts_engine,
    start_tts_worker,
    queue_for_speech,
    skip_current,
    shutdown_tts,
    wait_for_tts_to_finish,
    is_speaking,
    get_queue_size,
    get_current_speech_item,
    configure_tts_from_dict,
)

# ASR API
from .asr import (
    start_dictation,
    stop_dictation,
    configure_asr_from_dict,
    ASRConfig,
)

# Communication API
from .comms import (
    setup_communication,
    create_config_from_env,
    CommunicationManager,
    CommsConfig,
    Message,
    CommsProvider,
    TwilioSMSProvider,
    TwilioWhatsAppProvider,
    SlackProvider,
)

# Profiles API
from .profiles import (
    get_profile,
    Profile,
    PROFILES,
)

# High-level API from core.py
from .core import run_with_talkito, wrap_command, TalkitoCore

# MCP server functionality (optional - requires mcp package)
try:
    from .mcp import app as mcp_app
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp_app = None

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    
    # Modules
    "tts",
    "asr", 
    "comms",
    "profiles",
    
    # TTS functions
    "detect_tts_engine",
    "start_tts_worker",
    "queue_for_speech",
    "skip_current",
    "shutdown_tts",
    "wait_for_tts_to_finish",
    "is_speaking",
    "get_queue_size",
    "get_current_speech_item",
    "configure_tts_from_dict",
    
    # ASR functions
    "start_dictation",
    "stop_dictation",
    "configure_asr_from_dict",
    "ASRConfig",
    
    # Communication classes/functions
    "setup_communication",
    "create_config_from_env",
    "CommunicationManager",
    "CommsConfig",
    "Message",
    "CommsProvider",
    "TwilioSMSProvider",
    "TwilioWhatsAppProvider",
    "SlackProvider",

    # Profile functions/classes
    "get_profile",
    "Profile",
    "PROFILES",
    
    # High-level API
    "run_with_talkito",
    "wrap_command",
    "TalkitoCore",
    
    # MCP server
    "MCP_AVAILABLE",
    "mcp_app",
]