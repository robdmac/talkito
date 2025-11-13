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

"""Core functionality for talkito - terminal interaction and processing"""

import asyncio
import atexit
import errno
import fcntl
import hashlib
import logging
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import time
import threading
import traceback
import tty
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional, List, Tuple, Dict, Union, Deque, Any
from concurrent.futures import ThreadPoolExecutor

from . import asr, comms, tts
from .profiles import get_profile, Profile
from .logs import setup_logging, get_logger, log_message, restore_stderr, is_logging_enabled
from .state import get_shared_state, sync_communication_state_from_config
from .tts import stop_tts_immediately

# Configuration
TTS_ENGINE = "auto"  # Options: auto, espeak, festival, say, flite
SPEAK_ERRORS = True  # Whether to speak stderr output

# Constants
MAX_LINE_PREVIEW = 50
MAX_LINE_LOG = 100
STATUS_CHECK_INTERVAL = 0.1
RESIZE_DEBOUNCE_TIME = 0.5
BUFFER_SWITCH_DELAY = 0.5
SCREEN_REDRAW_THRESHOLD = 2
SCREEN_REDRAW_TIMEOUT = 5.0
DEFAULT_TERMINAL_HEIGHT = 24
DEFAULT_TERMINAL_WIDTH = 80
PTY_READ_SIZE = 16384
SIMILARITY_THRESHOLD = 0.85
RECENT_LINES_CACHE_SIZE = 50
SCREEN_CONTENT_CACHE_SIZE = 100
RESPONSE_PREFIX = '⏺'
OUTPUT_BUFFER_MAX_SIZE = 1024 * 1024  # 1MB max buffer for non-blocking writes

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHJ]')

RESET = '\033[0m'
SPACE_THEN_BACK = b'\xc2\xa0\x1b[D'
# SPACE_THEN_BACK = b' \x08'

# ANSI Key codes
RETURN = b'\x1b[B\r'
ESC = b'\x1b'
KEY_UP = b'\x1b[A'
KEY_DOWN = b'\x1b[B'
KEY_LEFT = b'\x1b[D'
KEY_RIGHT = b'\x1b[C'
KEY_SPACE = b' '
# Tap-to-talk keys - multiple options for flexibility
KEY_TAP_TO_TALK_SEQUENCES = [b'`']  # Backtick as mic button

# Filtering Configuration
SKIP_LINE_STARTS = ["?", "/", "cwd:", "#", "DEBUG:", "INFO:", "WARNING:"]
SKIP_LINE_CONTAINS = ["Press Ctrl-C", "[SKIPPED]", "? for shortcuts"]
PROGRESS_PATTERNS = [
    # r'^\s*[│⎿]\s+',  # Progress lines
    r'^\s*\.{3,}$',  # Multiple dots
    r'^\s*[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]',  # Spinner characters
    r'^\s*\d+\s+',  # Lines starting with line numbers (code diffs)
    r'^[+-]\s*\d+',  # Diff line numbers with +/-
    # r'^\s*\d+\s*[:\|]\s*',  # Line numbers with : or | separator
]

# Terminal escape sequences
RESET_SEQUENCES = [
    b'\x1b[H\x1b[2J',
    b'\x1b[2J',
    b'\x1b[3J',
    b'\x1bc',
    b'\x1b[H\x1b[J',
]
ALT_SCREEN_SEQUENCES = [
    b'\x1b[?1049h',
    b'\x1b[?1049l',
    b'\x1b[?47h',
    b'\x1b[?47l',
]

# ANSI escape code pattern - comprehensive
ANSI_CHAR_PATTERN = re.compile(r'(?:\x1B\[[0-9;?]*[a-zA-Z])+([A-Za-z])(?=(?:\x1B\[[0-9;?]*[a-zA-Z])|$)')
ANSI_MOVE_LINE_PATTERN = re.compile(r'\x1B\[(\d+);(\d+)[Hf]')
ANSI_PATTERN = re.compile(
    r'(\x1B\[[0-9;]*[a-zA-Z]|'  # Standard codes
    r'\x1B\]([0-9]+;[^\x07\x1B]*)?\x07|'  # OSC sequences
    r'\x1B[()][0-9A-Za-z]?|'  # Charset selection (includes letters like B)
    r'\x1B[>=]|'  # Keypad modes
    r'\x0F|'  # Shift in
    r'\r$|'  # Carriage return at end
    r'\x1B\[[0-9;]*[Hf]|'  # Cursor positioning
    r'\x1B\[([0-9]+[A-D]|s|u|[0-9]*[Jm])|'  # Various controls
    r'\x1B\[6n|'  # Cursor position request
    r'\x1B\[[0-9]*[GKL]|'  # Column/line operations
    r'\x1B\[[?][0-9]+[hl]|'  # Private modes
    r'\x1B\[[0-9]+;[0-9]+;[0-9]+m|'  # RGB colors
    r'\x1B\[[0-9]+(;[0-9]+)*m|'  # Generic SGR sequences
    r'\x1B\([B0]|'  # More charset selections
    r'\x1B\)0|'  # Charset selections
    r'\x1B[78]|'  # Save/restore cursor
    r'\x1B\[[0-9]*(;[0-9]+)*[HfABCDsuJKmhlr]|'  # More complete control sequences
    r'\x1B#[0-9]|'  # Line attributes
    r'\x1BP[^\\]*\\|'  # DCS sequences
    r'\x1B\[[0-9;]*~)'  # Special keys
)

# Precompiled regex patterns for clean_text() function
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
# ORPHANED_PATTERN = re.compile(
#     r"(?<![a-zA-Z0-9'])"                            # not preceded by word char or '
#     r"(?<!'\s)"                                # also not preceded by apostrophe + space
#     r"(?!a\b|I\b|s\b|d\b|ll\b|re\b|ve\b|m\b|t\b)"   # don't nuke a, I, or common contractions
#     r"[a-zA-Z]"                                     # stray letter
#     r"(?![a-zA-Z])"                                 # not followed by letter
# )

# 1) ANSI matcher (CSI, OSC BEL/ST, and 2-byte ESC)
ANSI_INLINE = r'(?:\x1B\[[0-?]*[ -/]*[@-~]|\x1B\][^\x07]*\x07|\x1B\][^\x1B]*\x1B\\|\x1B[@-Z\\-_])'

# 2) Repair broken contractions like: Here' <escapes/spaces> s  -> Here's
#    Covers: 's, 't, 'd, 'm, 've, 're, 'll  (case-insensitive)
CONTRACTION_REPAIR = re.compile(
    rf"(?<=[A-Za-z0-9])'(?:(?:\s|{ANSI_INLINE})+)(s|t|d|m|ve|re|ll)\b",
    re.IGNORECASE
)

# 3) Strip ANSI after repair
ANSI_RE = re.compile(ANSI_INLINE)

# 4) Orphaned single-letter filter (don't remove valid 'a' or 'I' or contraction tails)
ORPHANED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9'])"              # not after word char or apostrophe
    r"(?!a\b|I\b|d\b|ll\b|re\b|ve\b|m\b|t\b)"   # keep real words & contraction tails
    r"[A-Za-z]"                       # a single stray letter
    r"(?![A-Za-z0-9])"                # not followed by a letter or digit (preserves v2.0.13)
)

DASH_LINE_PATTERN = re.compile(r'──+')
ANSI_NUMBER_M_PATTERN = re.compile(r'(?<![a-zA-Z])\d{1,3}m\b')

# Pattern to filter OSC response sequences from stdin (terminal responses to queries)
# This prevents terminal responses (like color queries) from appearing as user input
OSC_RESPONSE_PATTERN = re.compile(rb'\x1B\][0-9]+;[^\x07]*\x07')

# Precompiled regex patterns for other frequent operations
BOX_CONTENT_PATTERN = re.compile(r'│\s*([^│]+)\s*│')
BOX_SEPARATOR_PATTERN = re.compile(r'^[─═╌╍]+$')
PROMPT_PATTERN = re.compile(r'^\s*[>\$#]\s*$')
SENTENCE_END_PATTERN = re.compile(r'[.!?]$')
ANSI_SIMPLE_PATTERN = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
BLOCK_DRAWING_PATTERN = re.compile(r'[\u2580-\u259F]+')  # Remove Unicode block drawing characters

# Additional precompiled patterns for performance
FILE_PATH_FILTER_PATTERN = re.compile(r'^[\w/\-_.]+\.(py|js|txt|md|json|yaml|yml|sh|c|cpp|h|java|go|rs|rb|php)$')
NUMBERS_ONLY_FILTER_PATTERN = re.compile(r'^\s*\d+(\s+\d+)*\s*$')
CURSOR_UP_PATTERN = re.compile(rb'\x1b\[(\d*)A')
CURSOR_DOWN_PATTERN = re.compile(rb'\x1b\[(\d*)B')
CURSOR_POS_PATTERN = re.compile(rb'\x1b\[(\d+);(\d+)H')
ANSI_FOR_LENGTH_PATTERN = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
ANSI_INPUT_PATTERN = re.compile(r'\x1b\[[0-9;]*[mGKHJ]')

# Global state - these are updated by TalkitoCore
current_master_fd: Optional[int] = None
current_proc: Optional[asyncio.subprocess.Process] = None
verbosity_level: int = 0
comm_manager: Optional[Any] = None  # Will be CommunicationManager when initialized

# Thread pool for blocking I/O operations
_io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='talkito-io')

def _trim_after_cursor_move(s):
    match = ANSI_MOVE_LINE_PATTERN.search(s)
    if match:
        return s[:match.start()]
    return s

# Register cleanup on exit
def _cleanup_io_executor():
    """Cleanup thread pool executor on exit"""
    _io_executor.shutdown(wait=False)

atexit.register(_cleanup_io_executor)

# Non-blocking I/O helpers
async def async_read(fd: int, size: int, timeout: float = 5.0) -> bytes:
    """Non-blocking read using thread pool"""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_io_executor, os.read, fd, size),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        log_message("WARNING", f"Read timeout on fd {fd}")
        return b""

async def async_write(fd: int, data: bytes) -> int:
    """Non-blocking write using thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_io_executor, os.write, fd, data)

async def async_stdout_write(data: bytes) -> None:
    """Non-blocking stdout write using thread pool"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_io_executor, _blocking_stdout_write, data)

def _blocking_stdout_write(data: bytes) -> None:
    """Helper for blocking stdout write"""
    sys.stdout.buffer.write(data)
    sys.stdout.flush()

# Simple state grouping to reduce globals
@dataclass
class TerminalState:
    """Groups terminal-related state"""
    line_screen_positions: Dict[int, int] = field(default_factory=dict)
    terminal_rows: int = DEFAULT_TERMINAL_HEIGHT
    terminal_cols: int = DEFAULT_TERMINAL_WIDTH
    resize_pending: bool = False
    resize_lock: threading.Lock = field(default_factory=threading.Lock)
    last_resize_time: float = 0
    screen_content_cache: Deque[str] = field(default_factory=lambda: deque(maxlen=SCREEN_CONTENT_CACHE_SIZE))
    previous_line_was_skipped: bool = False
    previous_line_was_queued: bool = False
    previous_line_was_queued_space_seperated: bool = False
    pending_speech_text: List[str] = field(default_factory=list)
    pending_text_line_number: int = 0
    pending_text_exception_match: bool = False
    pending_text_should_skip: bool = False
    last_line_number: int = -1
    last_sent_text: str = ""
    # Timer for automatic send_pending_text after delay
    pending_text_timer: Optional['threading.Timer'] = None
    pending_text_timer_lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_write_lock: threading.Lock = field(default_factory=threading.Lock)

@dataclass 
class ASRState:
    """Groups ASR-related state"""
    auto_listen_enabled: bool = True
    waiting_for_input: bool = False
    asr_auto_started: bool = False
    tap_to_talk_active: bool = False
    tap_to_talk_last_press: float = 0
    tap_to_talk_last_release: float = 0
    tap_to_talk_redraw_triggered: bool = False
    current_partial: str = ""
    last_prompt_position: int = 0
    prompt_detected: bool = False
    refresh_spaces_added: int = 0
    partial_enabled: bool = True
    question_mode: bool = False
    # Auto-submit timer fields
    last_finalized_transcript_time: float = 0
    last_partial_transcript_time: float = 0
    has_pending_transcript: bool = False

# State instances will be created by TalkitoCore
active_profile: Optional[Profile] = None  # Will be initialized to default profile
terminal = None  # Will be set by TalkitoCore
in_code_block = False  # Track if we're inside a code block (```)
asr_state: ASRState = ASRState()  # Will be set by TalkitoCore

class OutputBuffer:
    """Buffer for handling non-blocking stdout writes"""
    def __init__(self, max_size: int = OUTPUT_BUFFER_MAX_SIZE):
        self.buffer = bytearray()
        self.max_size = max_size
        self.dropped_bytes = 0
    
    def add(self, data: bytes) -> bool:
        """Add data to buffer. Returns False if buffer is full."""
        if len(self.buffer) + len(data) > self.max_size:
            self.dropped_bytes += len(data)
            log_message("WARNING", f"Output buffer full, dropping {len(data)} bytes (total dropped: {self.dropped_bytes})")
            return False
        self.buffer.extend(data)
        return True
    
    def write_to_stdout(self) -> int:
        """Try to write buffered data to stdout. Returns bytes written."""
        if not self.buffer:
            return 0
        
        try:
            written = os.write(sys.stdout.fileno(), self.buffer)
            if written > 0:
                # Remove written bytes from buffer
                self.buffer = self.buffer[written:]
            return written
        except BlockingIOError:
            return 0
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return 0
            raise
    
    def __len__(self):
        return len(self.buffer)


class LineBuffer:
    """Buffer that tracks all raw output lines with unique indices."""

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.lines: Dict[int, str] = {}  # index -> raw line content
        self.next_index = 0
        self.similarity_threshold = similarity_threshold
        # Keep a sliding window of recent lines to detect modifications
        self.recent_lines: deque = deque(maxlen=RECENT_LINES_CACHE_SIZE)  # (index, content, hash) tuples
        
    def _compute_line_hash(self, content: str) -> int:
        """Compute a simple hash for quick filtering"""
        # Use length and a few character samples for quick comparison
        if not content:
            return 0
        # Sample first, middle, and last characters + length
        return hash((len(content), content[:20], content[-20:] if len(content) > 20 else ''))

    def _get_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings"""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    def _find_similar_line(self, content: str) -> Optional[int]:
        """Find a similar line in recent history"""
        content_len = len(content)
        
        # Pre-filter: only check lines with similar length
        for idx, line_content, line_hash in self.recent_lines:
            if idx in self.lines:  # Still exists
                # Quick filters before expensive similarity check
                line_len = len(line_content)
                if abs(line_len - content_len) > content_len * 0.2:  # Skip if length differs by >20%
                    continue
                    
                # Only do expensive similarity check for potentially similar lines
                similarity = self._get_similarity(content, line_content)
                if similarity >= self.similarity_threshold:
                    return idx
        return None

    def add_or_update_line(self, raw_line: str) -> Tuple[int, str]:
        """Add a new line or update an existing one if similar."""
        # Special logging for response lines
        if '⏺' in raw_line:
            log_message("INFO", f"[LineBuffer] Adding/updating response line: '{raw_line[:100]}...'")
        
        # Check if this is similar to a recent line
        similar_idx = self._find_similar_line(raw_line)

        if similar_idx is not None:
            old_content = self.lines[similar_idx]
            if old_content == raw_line:
                # Exact match, no change
                return similar_idx, 'unchanged'
            else:
                # Similar but different - this is a modification
                self.lines[similar_idx] = raw_line
                log_message("BUFFER", f"Line {similar_idx} modified: '{old_content[:MAX_LINE_PREVIEW]}...' -> '{raw_line[:MAX_LINE_PREVIEW]}...'")
                # Update in recent lines with new hash
                line_hash = self._compute_line_hash(raw_line)
                self.recent_lines.append((similar_idx, raw_line, line_hash))
                return similar_idx, 'modified'

        # New line
        idx = self.next_index
        self.next_index += 1
        self.lines[idx] = raw_line
        line_hash = self._compute_line_hash(raw_line)
        self.recent_lines.append((idx, raw_line, line_hash))
        log_message("BUFFER", f"Line {idx} added: '{raw_line[:MAX_LINE_PREVIEW]}...'")
        return idx, 'added'

    def mark_lines_deleted(self, indices: List[int]):
        """Mark multiple lines as deleted"""
        for idx in indices:
            if idx in self.lines:
                content = self.lines[idx]
                del self.lines[idx]
                log_message("BUFFER", f"Line {idx} deleted: '{content[:MAX_LINE_PREVIEW]}...'")

    def clear(self):
        """Clear all lines (e.g., on screen clear)"""
        if self.lines:
            log_message("BUFFER", f"Clearing buffer with {len(self.lines)} lines")
        self.lines.clear()
        self.recent_lines.clear()

    def get_line(self, idx: int) -> Optional[str]:
        """Get line content by index"""
        return self.lines.get(idx)

    def get_all_lines(self) -> List[Tuple[int, str]]:
        """Get all lines with their indices, sorted by index"""
        return sorted(self.lines.items())

    def get_line_count(self) -> int:
        """Get number of lines in buffer"""
        return len(self.lines)



def send_to_comms(text: str):
    log_message("DEBUG", "send_to_comms")
    """Send text to communication channels if configured"""
    global comm_manager
    if comm_manager:
        if text.strip():
            log_message("DEBUG", f"[COMMS] Sending to comms: {text}...")
            try:
                # send_output now handles buffer logic internally with deduplication and verbosity filtering
                comm_manager.send_output(text)
            except Exception as e:
                log_message("ERROR", f"[COMMS] Failed to send to comms: {e}")
    else:
        log_message("DEBUG", "[COMMS] comm_manager not configured")


def get_comm_manager():
    """Get the global communication manager instance"""
    global comm_manager
    return comm_manager


def check_comms_input() -> Optional[str]:
    """Check for input from communication channels"""
    global comm_manager
    if comm_manager:
        try:
            return comm_manager.get_input(timeout=0)  # Non-blocking
        except Exception as e:
            log_message("ERROR", f"Failed to check comms input: {e}")
    return None


def _send_pending_text_delayed():
    """Timer callback to send pending text after delay."""
    with terminal.pending_text_timer_lock:
        # Clear the timer reference since it's about to complete
        terminal.pending_text_timer = None

    log_message("INFO", "Auto-sending pending text after 2-second timeout")
    send_pending_text()


def send_pending_text():
    # Cancel any pending timer first
    with terminal.pending_text_timer_lock:
        if terminal.pending_text_timer:
            terminal.pending_text_timer.cancel()
            terminal.pending_text_timer = None

    if terminal.pending_speech_text and ''.join(terminal.pending_speech_text).strip():
        pending_text = ''.join(terminal.pending_speech_text)
        log_message("DEBUG", f"send_pending_text [{pending_text}]")
        log_message("DEBUG", f"last_sent_text {terminal.last_sent_text}")

        # Check for duplicate sends (but allow questions and exceptions through)
        if pending_text.strip() == terminal.last_sent_text.strip():
            # Don't skip if this is a question or exception match - these should always be spoken
            if not terminal.pending_text_exception_match:
                log_message("DEBUG", f"Skipping duplicate send: '{pending_text}'")
                terminal.pending_speech_text.clear()
                terminal.pending_text_should_skip = False
                terminal.pending_text_exception_match = False
                return
            else:
                log_message("DEBUG", f"Allowing duplicate send because it's a question/exception: '{pending_text}'")
        
        # Check the stored skip decision
        if terminal.pending_text_should_skip:
            log_message("FILTER", f"Skipped buffered text by profile: '{pending_text}'")
            terminal.pending_speech_text.clear()
            terminal.pending_text_should_skip = False
            terminal.pending_text_exception_match = False
            return
        
        # Check if TTS is enabled before queueing
        shared_state = get_shared_state()
        terminal.last_sent_text = pending_text
        if shared_state.tts_enabled:

            def speech_callback(written_text: str):
                """Called when text is actually queued for speech (immediate or delayed)"""
                log_message("DEBUG", f"All checks passed for [{written_text}]. Send to comms")
                send_to_comms(written_text)

            # Pass the exception match flag from terminal state
            exception_match = getattr(terminal, 'pending_text_exception_match', False)
            # Pass constituent parts to help with duplicate detection
            constituent_parts = terminal.pending_speech_text.copy() if len(terminal.pending_speech_text) > 1 else None
            speakable_text = tts.queue_for_speech(
                pending_text,
                terminal.pending_text_line_number,
                source="output",
                exception_match=exception_match,
                writes_partial_output=active_profile.writes_partial_output,
                callback=speech_callback,
                constituent_parts=constituent_parts,
            )
            if speakable_text:
                terminal.pending_speech_text.clear()
                terminal.pending_text_exception_match = False  # Reset flag
                terminal.pending_text_should_skip = False  # Reset flag
                if speakable_text != "--ignore--":
                    speech_callback(pending_text)
        else:
            log_message("DEBUG", "TTS disabled, not sending pending text to speech")
            # Still send to comms even if TTS is disabled
            send_to_comms(pending_text)
            terminal.pending_speech_text.clear()
            terminal.pending_text_exception_match = False  # Reset flag
            terminal.pending_text_should_skip = False  # Reset flag

def queue_output(text: str, line_number: Optional[int] = None, exception_match: bool = False):
    """Queue text for TTS and communication channels with optional line tracking and exception matching."""
    log_message("DEBUG", f"queue_output [{text}] {line_number} exception_match={exception_match}")
    # if line_number < terminal.last_line_number and not active_profile.writes_partial_output:
    #     log_message("DEBUG", "queue_output skipping previously seen line number")
    # elif and text.strip():
    if text and text.strip():
        # Check if TTS is enabled before accumulating text
        shared_state = get_shared_state()
        if not shared_state.tts_enabled:
            log_message("DEBUG", "TTS disabled, not accumulating text for speech")
            # Still send to comms even if TTS is disabled
            send_to_comms(text)
            return
            
        # Store exception match flag for later use
        terminal.pending_text_exception_match = exception_match

        text = active_profile.apply_replacements(text)
        
        # Check if text should be skipped based on profile
        should_skip = active_profile and active_profile.should_skip(text, verbosity_level)
            
        # Queue for TTS (TTS worker will handle ASR pausing and cleaning of text)
        append = terminal.previous_line_was_queued and not terminal.previous_line_was_queued_space_seperated
        terminal.last_line_number = line_number
        if append:
            log_message("DEBUG", "queue_output appending")
            if text.startswith("  "):
                text = "\n" + text
            terminal.pending_speech_text.append(text)
            # When appending, keep the more permissive skip decision (False wins over True)
            terminal.pending_text_should_skip = terminal.pending_text_should_skip and should_skip
            log_message("DEBUG", f"queue_output setting should skip to be terminal.pending_text_should_skip {terminal.pending_text_should_skip} and should_skip {should_skip} = {terminal.pending_text_should_skip}")

            # Restart the 2-second timer since new text was appended
            with terminal.pending_text_timer_lock:
                if terminal.pending_text_timer:
                    terminal.pending_text_timer.cancel()
                terminal.pending_text_timer = threading.Timer(2.0, _send_pending_text_delayed)
                terminal.pending_text_timer.daemon = True
                terminal.pending_text_timer.start()
                log_message("DEBUG", "Restarted 2-second timer for pending text")
        else:
            if terminal.pending_speech_text:
                send_pending_text()
            terminal.previous_line_was_skipped = False
            terminal.previous_line_was_queued = True
            terminal.pending_speech_text = [text]
            terminal.pending_text_line_number = line_number
            terminal.pending_text_should_skip = should_skip

            # Start the 2-second timer for new pending text
            with terminal.pending_text_timer_lock:
                if terminal.pending_text_timer:
                    terminal.pending_text_timer.cancel()
                terminal.pending_text_timer = threading.Timer(2.0, _send_pending_text_delayed)
                terminal.pending_text_timer.daemon = True
                terminal.pending_text_timer.start()
                log_message("DEBUG", "Started 2-second timer for pending text")

def clean_text(text: str) -> str:
    """Strip ANSI escape codes and terminal control sequences"""

    text = _trim_after_cursor_move(text)
    text = text.replace("’", "'")

    text = re.sub(
        r"(?<=[A-Za-z0-9])'(?:(?:\s|\x1B\[[0-9;?]*[ -/]*[@-~])+)" +
        r"(s|t|d|m|ve|re|ll)\b",
        r"'\1",
        text,
        flags=re.IGNORECASE
    )

    text = ANSI_CHAR_PATTERN.sub('', text)  # Remove ANSI sequences with orphaned letters
    text = ANSI_PATTERN.sub('', text)
    text = ANSI_ESCAPE_PATTERN.sub('', text)
    text = text.replace('\x1B', '')

    # Remove various control characters
    text = text.replace('\x08', '')  # Backspace
    text = text.replace('\x0D', '')  # Carriage return

    # Remove standalone 'm' characters that are likely orphaned from ANSI codes
    # This handles cases where 'm' appears after whitespace or at start of string
    # but NOT when it's part of a word (like 'am', 'pm', 'them', etc.)
    # Also preserve contractions like "I'm", "don't", etc.
    text = ORPHANED_PATTERN.sub('', text)

    text = DASH_LINE_PATTERN.sub('', text)

    # Also remove patterns like "2m" or "22m" that might be left from ANSI codes
    text = ANSI_NUMBER_M_PATTERN.sub('', text)

    # Remove Unicode block drawing characters (▘▘ ▝▝ etc.)
    text = BLOCK_DRAWING_PATTERN.sub('', text)

    # Filter out non-printable characters
    return ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')


def strip_profile_symbols(text: str) -> str:
    """Remove profile-specific symbols from text before TTS"""
    if active_profile and active_profile.strip_symbols:
        for symbol in active_profile.strip_symbols:
            text = text.replace(symbol, '')
    return text


def is_response_line(line: str) -> bool:
    """Check if line is a response from the AI"""
    response_prefix = active_profile.response_prefix if active_profile else RESPONSE_PREFIX
    return line.startswith(response_prefix)


def extract_box_content(line: str) -> Optional[str]:
    """Extract meaningful content from box drawing lines"""
    box_match = BOX_CONTENT_PATTERN.match(line)
    if box_match:
        box_content = box_match.group(1).strip()
        if (box_content and
                not BOX_SEPARATOR_PATTERN.match(box_content) and
                not PROMPT_PATTERN.match(box_content)):
            return box_content
    return None


def is_prompt_line(line: str) -> bool:
    """Check if line is a prompt waiting for user input"""
    profile_prompt_patterns = active_profile.prompt_patterns if active_profile else []
    default_prompt_patterns = [
        r'^[>\$#:]\s*$',
        r'^>\s*.+',
        r'│\s*>\s*.*│',
        r'^\s*>\s*$',
    ]
    prompt_patterns = profile_prompt_patterns if profile_prompt_patterns else default_prompt_patterns

    try:
        return any(re.search(p, line) for p in prompt_patterns if p)
    except Exception as e:
        log_message("ERROR", f"Error in is_prompt_line: {e} - patterns: {prompt_patterns}, line: {line!r}")
        return False


def hash_content(content: str) -> str:
    """Create a hash of content for deduplication"""
    return hashlib.md5(content.encode()).hexdigest()


def is_duplicate_screen_content(content: str) -> bool:
    """Check if this content has been seen recently"""
    content_hash = hash_content(content)
    if content_hash in terminal.screen_content_cache:
        return True
    terminal.screen_content_cache.append(content_hash)
    return False


def modify_prompt_for_asr(data: bytes, input_prompts, input_replace) -> bytes:
    """Modify prompt output to show microphone emoji when ASR is active"""
    if not input_replace:
        return data
    try:
        text = data.decode('utf-8', errors='ignore')
        for input_prompt in input_prompts:
            if input_prompt in text:
                # When we find the prompt, mark that we've seen it
                asr_state.prompt_detected = True
                text = text.replace(input_prompt, input_replace, 1)
                return text.encode('utf-8')
        # log_message("WARNING", f"input prompt {input_prompt} not found in text {text}")
        return data
    except Exception:
        # If any error occurs, return original data
        return data


def should_skip_line(line: str) -> bool:
    """Check if line should be skipped based on filters"""
    # Use Profile object if available
    if active_profile:
        should_skip = active_profile.should_skip(line, verbosity_level)
        if should_skip:
            log_message("FILTER", f"Skipped by profile (verbosity={verbosity_level}): '{line}'")
            return True

    # Check default filters
    if any(line.startswith(p) for p in SKIP_LINE_STARTS):
        log_message("FILTER", f"Skipped line: '{line}'")
        return True

    if any(p in line for p in SKIP_LINE_CONTAINS):
        log_message("FILTER", f"Skipped line: '{line}'")
        return True

    # Check for progress patterns
    try:
        if any(re.match(p, line) for p in PROGRESS_PATTERNS if p):
            log_message("FILTER", f"Skipped progress line: '{line}'")
            return True
    except Exception as e:
        log_message("ERROR", f"Error in progress pattern matching: {e} - line: {line!r}")
        # Continue processing on error

    # Skip lines that are just file paths
    if FILE_PATH_FILTER_PATTERN.match(line):
        log_message("FILTER", f"Skipped file path: '{line}'")
        return True

    # Skip lines that are just numbers (single or multiple), catches cases like "599" or "599 603 1117"
    if NUMBERS_ONLY_FILTER_PATTERN.match(line):
        log_message("FILTER", f"Skipped line with only numbers: '{line}'")
        return True

    return False


def get_terminal_size() -> Tuple[int, int]:
    """Get current terminal size with fallback. Updates terminal state."""
    try:
        if sys.stdout.isatty():
            rows, cols = os.get_terminal_size()
            terminal.terminal_rows = rows
            terminal.terminal_cols = cols
            return rows, cols
    except Exception:
        pass
    return terminal.terminal_rows, terminal.terminal_cols


def track_line_position(line_index: int, current_row: int):
    """Track the screen position of a line"""
    # Get current terminal size
    rows, _ = get_terminal_size()

    # Store the position
    terminal.line_screen_positions[line_index] = current_row

    # Clean up old positions that have scrolled off screen
    # Keep a sliding window of positions
    max_visible_lines = rows - 2  # Leave room for status line
    if len(terminal.line_screen_positions) > max_visible_lines * 2:
        # Remove positions that are too far off screen
        min_visible_row = current_row - max_visible_lines
        to_remove = [idx for idx, row in terminal.line_screen_positions.items()
                     if row < min_visible_row]
        for idx in to_remove:
            del terminal.line_screen_positions[idx]


def update_line_positions_on_scroll():
    """Update all line positions when terminal scrolls"""
    # When terminal scrolls up by one line, all positions move up by one
    for idx in list(terminal.line_screen_positions.keys()):
        terminal.line_screen_positions[idx] -= 1

    # Remove any that have scrolled off the top
    terminal.line_screen_positions = {idx: row for idx, row in terminal.line_screen_positions.items()
                                      if row > 0}


class SessionRecorder:
    """Handles recording of terminal sessions"""
    def __init__(self, record_file: str):
        self.record_file = record_file
        self.recorded_data = []
        self.start_time = time.time()
        self.enabled = record_file is not None

    def record_event(self, event_type: str, data: bytes):
        """Record a session event"""
        if not self.enabled:
            return
        timestamp = time.time() - self.start_time
        self.recorded_data.append((timestamp, event_type, data))

    def save(self):
        """Save recorded data to file"""
        if not self.enabled or not self.recorded_data:
            return

        try:
            with open(self.record_file, 'wb') as f:
                for timestamp, data_type, data in self.recorded_data:
                    f.write(f"{timestamp:.6f} {data_type} {len(data):04x} ".encode('utf-8'))
                    f.write(data)
                    f.write(b"\x00")
            log_message("INFO", f"Recorded {len(self.recorded_data)} events to {self.record_file}")
        except Exception as e:
            log_message("ERROR", f"Failed to write record file {self.record_file}: {e}")

    @staticmethod
    def parse_file(replay_file: str) -> List[dict]:
        """Parse a recorded session file into entries"""
        try:
            with open(replay_file, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            log_message("ERROR", f"Replay file not found: {replay_file}")
            return []

        entries = []
        offset = 0

        while offset < len(data):
            space1 = data.find(b' ', offset)
            if space1 == -1:
                break

            space2 = data.find(b' ', space1 + 1)
            if space2 == -1:
                break

            space3 = data.find(b' ', space2 + 1)
            if space3 == -1:
                break

            try:
                timestamp = float(data[offset:space1].decode('ascii'))
                event_type = data[space1+1:space2].decode('ascii')
                data_length = int(data[space2+1:space3].decode('ascii'), 16)

                data_start = space3 + 1
                raw_bytes = data[data_start:data_start + data_length]

                entries.append({
                    'timestamp': timestamp,
                    'event_type': event_type,
                    'raw_bytes': raw_bytes
                })

                offset = data_start + data_length
                if offset < len(data) and data[offset] == 0:
                    offset += 1

            except Exception as e:
                log_message("WARNING", f"Skipping malformed entry at offset {offset}: {e}")
                offset += 1

        return entries


def parse_cursor_movements(data: bytes) -> Tuple[int, bool]:
    """Parse ANSI escape sequences to track cursor movements. Returns (cursor_row_delta, did_scroll)"""
    cursor_delta = 0
    did_scroll = False

    # Check for newline (moves cursor down)
    newline_count = data.count(b'\n')
    cursor_delta += newline_count

    # Check for carriage return + newline (common pattern)
    crlf_count = data.count(b'\r\n')
    cursor_delta -= crlf_count  # Don't double count

    # Check for cursor up
    up_matches = re.findall(rb'\x1b\[(\d*)A', data)
    for match in up_matches:
        n = int(match) if match else 1
        cursor_delta -= n

    # Check for cursor down
    down_matches = re.findall(rb'\x1b\[(\d*)B', data)
    for match in down_matches:
        n = int(match) if match else 1
        cursor_delta += n

    # ESC[H or ESC[;H - cursor home (top left)
    if b'\x1b[H' in data or b'\x1b[;H' in data:
        cursor_delta = -999  # Signal absolute positioning

    # ESC[n;mH - cursor position
    pos_matches = re.findall(rb'\x1b\[(\d+);(\d+)H', data)
    if pos_matches:
        # Take the last position if multiple
        row = int(pos_matches[-1][0])
        cursor_delta = -1000 - row  # Signal absolute positioning with row

    # Check for scroll up
    if b'\x1b[S' in data:
        did_scroll = True
    # Check for scroll down
    if b'\x1b[T' in data:
        did_scroll = True

    return cursor_delta, did_scroll


def update_cursor_position(cursor_row: int, cursor_delta: int, did_scroll: bool) -> int:
    """Update cursor position based on parsed movements. Returns new cursor row."""
    if cursor_delta <= -1000:
        # Absolute positioning
        if cursor_delta == -999:
            cursor_row = 1
        else:
            cursor_row = -cursor_delta - 1000
    else:
        # Relative movement
        cursor_row += cursor_delta
        
        # Check if we scrolled past bottom
        rows, _ = get_terminal_size()
        if cursor_row > rows:
            scrolled_lines = cursor_row - rows
            cursor_row = rows
            
            # Update all tracked positions
            for _ in range(scrolled_lines):
                update_line_positions_on_scroll()
    
    # Handle explicit scroll sequences
    if did_scroll:
        update_line_positions_on_scroll()
        
    return cursor_row


def _skip_line_and_return(buffer: List[str], line: str, detected_prompt: bool = False) -> Tuple[List[str], str, bool]:
    """Helper to skip a line and return with common pattern"""
    global comm_manager
    # Add cleaned line to processed buffer for slack context with higher verbosity (level 2)
    # Only process buffer if Slack is enabled (buffer is only used for Slack)
    cleaned_line = clean_text(line)
    if cleaned_line and cleaned_line.strip() and '───' not in cleaned_line:
        if comm_manager:
            # Check if Slack provider is enabled (buffer is only used for Slack)
            from .comms import SlackProvider
            if any(isinstance(p, SlackProvider) for p in comm_manager.providers):
                log_message("INFO", "adding line to buffer with verbosity level 2 (more verbose)")
                # Use the new comm_manager buffer with hard-coded verbosity level 2 for more verbose output
                comm_manager.add_to_buffer(cleaned_line.strip(), active_profile)

    terminal.previous_line_was_skipped = True
    send_pending_text()
    return buffer, line, detected_prompt


def _queue_text(text: str, line_number: Optional[int] = None, is_question = False) -> None:
    """Helper to queue text with exception matching"""
    exception_match = (active_profile and active_profile.matches_exception_pattern(text, verbosity_level)) or is_question
    queue_output(strip_profile_symbols(text), line_number, exception_match)


def _queue_and_return(text: str, buffer: List[str], line: str, line_number: Optional[int] = None, 
                     detected_prompt: bool = False) -> Tuple[List[str], str, bool]:
    """Helper to queue text and return"""
    _queue_text(text, line_number)
    return buffer, line, detected_prompt


def _should_skip_raw_patterns(line: str) -> bool:
    """Helper to check if line should be skipped by raw patterns"""
    if not active_profile:
        return False
    
    # Check using profile's should_skip_raw method
    if active_profile.should_skip_raw(line):
        return True
        
    # Check raw skip patterns manually
    if active_profile.raw_skip_patterns:
        return any(re.search(pattern, line) for pattern in active_profile.raw_skip_patterns)
    
    return False


def _handle_continuation_line(cleaned_line: str, line: str, buffer: List[str], line_number: Optional[int] = None) -> Optional[Tuple[List[str], str, bool]]:
    """Handle continuation line logic. Returns tuple if handled, None otherwise."""
    is_regular_continuation = active_profile and active_profile.is_continuation_line(cleaned_line)
    
    # Only treat interaction menu lines as continuations if Slack communications are active
    is_interaction_menu = False
    if active_profile and active_profile.is_interaction_menu_line(cleaned_line):
        log_message("DEBUG", "is interactive menu line")
        # Check if Slack communications are active
        shared_state = get_shared_state()
        is_interaction_menu = shared_state.slack_mode_active if shared_state else False
        
        if is_interaction_menu:
            log_message("DEBUG", f"Detected interaction menu line (Slack mode active): '{line[:MAX_LINE_PREVIEW]}...'")
        else:
            log_message("DEBUG", f"Interaction menu detected but Slack not active, treating as normal line: '{line[:MAX_LINE_PREVIEW]}...'")
    
    if not (is_regular_continuation or is_interaction_menu):
        return None
        
    if not (terminal.previous_line_was_skipped or terminal.previous_line_was_queued):
        return None

    # Check exception patterns before filtering out continuation lines
    if active_profile:
        for min_verbosity, pattern in active_profile._compiled_exceptions:
            log_message("DEBUG", f"Checking exception pattern '{pattern.pattern}' (min_verbosity={min_verbosity}) against cleaned_line: '{cleaned_line}' or line: '{line}' (verbosity_level={verbosity_level})")
            if (pattern.search(cleaned_line) or pattern.search(line)) and verbosity_level >= min_verbosity:
                log_message("DEBUG", f"Continuation line matches exception pattern, processing normally: '{line[:MAX_LINE_PREVIEW]}...'")
                return None  # Don't handle as continuation, let normal processing continue

    if is_interaction_menu:
        log_message("DEBUG", f"Processing interaction menu line as continuation: '{line[:MAX_LINE_PREVIEW]}...'")
    else:
        log_message("DEBUG", f"Processing regular continuation line: '{line[:MAX_LINE_PREVIEW]}...'")
    
    if terminal.previous_line_was_skipped:
        log_message("FILTER", f"Skipping continuation line (previous was skipped): '{line[:MAX_LINE_PREVIEW]}...'")
        return buffer, line, False
    elif terminal.previous_line_was_queued:
        if not cleaned_line.strip():
            send_pending_text()
        if is_interaction_menu:
            log_message("INFO", f"Processing interaction menu line (previous was queued): '{line[:MAX_LINE_PREVIEW]}...'")
        else:
            log_message("INFO", f"Processing continuation line (previous was queued): '{line[:MAX_LINE_PREVIEW]}...'")
        _queue_text(cleaned_line, line_number)
        return [], line, False
        
    return None


def _handle_response_line(cleaned_line: str, line: str, buffer: List[str], line_number: Optional[int] = None, asr_mode: str = "auto-input") -> Optional[Tuple[List[str], str, bool]]:
    """Handle response line logic. Returns tuple if handled, None otherwise."""
    if not is_response_line(cleaned_line):
        return None
        
    asr_state.waiting_for_input = False
    log_message("DEBUG", "Set asr_state.waiting_for_input=False in process_line (response detected)")
    check_and_enable_auto_listen(asr_mode)
    
    # Response lines should be spoken unless filtered by profile
    if active_profile and active_profile.should_skip(cleaned_line, verbosity_level):
        log_message("FILTER", f"Skipped response by profile (verbosity={verbosity_level}): '{cleaned_line}'")
        return _skip_line_and_return(buffer, line)
    
    log_message("INFO", f"Queueing for speech response line: '{cleaned_line}'")
    return _queue_and_return(cleaned_line, buffer, line, line_number)


def _process_buffer_and_queue(cleaned_line: str, line: str, buffer: List[str], line_number: Optional[int] = None) -> Tuple[List[str], str, bool]:
    """Process the buffer and handle final queuing logic."""
    filtered_buffer = [line for line in buffer if line and line.strip()]
    text_to_speak = '. '.join(filtered_buffer) if filtered_buffer else None

    # Debug suspicious short text
    if text_to_speak and len(text_to_speak) <= 5:
        log_message("WARNING", f"Short text_to_speak='{text_to_speak}' from buffer={filtered_buffer}, "
                                f"cleaned_line='{cleaned_line}', raw_line='{line[:50]}...'")

    new_buffer = [cleaned_line] if cleaned_line and cleaned_line.strip() else []

    if cleaned_line and SENTENCE_END_PATTERN.search(cleaned_line):
        if text_to_speak:
            log_message("INFO", "Queueing buffered text before sentence end")
            _queue_text(text_to_speak, line_number)
            return [], line, False
        return [], line, False
    
    if text_to_speak:
        _queue_text(text_to_speak, line_number)
    return new_buffer, line, False


def process_line(line: str, buffer: List[str], prev_line: str,
                 skip_duplicates: bool = False, line_number: Optional[int] = None, asr_mode: str = "auto-input") -> Tuple[List[str], str, bool]:
    """Process a single line and queue text internally. Returns (new_buffer, new_prev_line, detected_prompt)"""
    global in_code_block, comm_manager

    log_message("DEBUG", f"Processing line: ['{line}']")

    cleaned_line = clean_text(line)

    # Check if this is a question line - these should ALWAYS be spoken
    if active_profile and active_profile.is_question_line(cleaned_line):
        log_message("INFO", f"Detected question line: '{line[:MAX_LINE_PREVIEW]}...'")
        asr_state.question_mode = True
        _queue_text(cleaned_line, line_number, is_question=True)
        terminal.previous_line_was_skipped = True
        return buffer, line, False

    # Check if this is a continuation line
    continuation_result = _handle_continuation_line(cleaned_line, line, buffer, line_number)
    if continuation_result is not None:
        return continuation_result
    elif cleaned_line.strip(): # Reset on non empty lines
        log_message("DEBUG", f"{cleaned_line=} reset previous_line_was_queued and previous_line_was_skipped to false")
        send_pending_text()
        terminal.previous_line_was_queued = False
        terminal.previous_line_was_skipped = False
        terminal.previous_line_was_queued_space_seperated = False

    # Check for code block start/end (```)
    if cleaned_line.strip().startswith('```'):
        in_code_block = not in_code_block
        log_message("DEBUG", f"Code block mode toggled: {in_code_block}")

    # If we're in a code block, send to comms_manager with verbosity 2
    if in_code_block:
        return _skip_line_and_return(buffer, line)

    if _should_skip_raw_patterns(line):
        log_message("WARNING", f"Skipped by raw pattern: '{line[:MAX_LINE_PREVIEW]}...'")
        return _skip_line_and_return(buffer, line)

    # Detect prompts
    is_prompt = is_prompt_line(cleaned_line)
    
    # Log prompt detection for debugging
    if ">" in cleaned_line and "│" in cleaned_line:
        log_message("DEBUG", f"Checking prompt detection for line: '{cleaned_line[:50]}...' - is_prompt={is_prompt}")

    # Handle response lines
    response_result = _handle_response_line(cleaned_line, line, buffer, line_number, asr_mode)
    if response_result is not None:
        return response_result

    # Move prompt detection here, before we process the line
    if is_prompt:
        log_message("INFO", f"Detected prompt in line ({line_number}): '{cleaned_line}'")
        return _skip_line_and_return(buffer, line, True)

    if cleaned_line:
        # Skip echoed user input when we're waiting at the prompt for the next command.
        if asr_state.waiting_for_input and not is_prompt:
            log_message("FILTER", f"Suppressed input echo while waiting for prompt: '{cleaned_line}'")
            return _skip_line_and_return(buffer, line)
        # Check if extracted text should be filtered based on verbosity
        if active_profile and active_profile.should_skip(cleaned_line, verbosity_level):
            log_message("FILTER", f"Skipped extracted text by profile (verbosity={verbosity_level}): '{cleaned_line}'")
            return _skip_line_and_return(buffer, line)
        return _queue_and_return(cleaned_line, buffer, line, line_number)

    if skip_duplicates and cleaned_line:
        if is_duplicate_screen_content(cleaned_line):
            log_message("WARNING", f"Skipping duplicate content: '{cleaned_line[:MAX_LINE_PREVIEW]}...'")
            return _skip_line_and_return(buffer, line)

    box_content = extract_box_content(cleaned_line)
    if box_content and not is_prompt:
        cleaned_line = box_content

    if should_skip_line(cleaned_line):
        return _skip_line_and_return(buffer, prev_line)

    # Process buffer and handle final queuing
    return _process_buffer_and_queue(cleaned_line, line, buffer, line_number)


def detect_screen_reset(data: bytes) -> bool:
    """Detect if the terminal is doing a full screen reset/clear"""
    for seq in RESET_SEQUENCES:
        if seq in data:
            return True

    cursor_moves = data.count(b'\x1b[H') + data.count(b'\x1b[1;1H')
    if cursor_moves > 5:
        return True

    return False


def detect_clear_sequences(data: bytes) -> bool:
    """Detect if data contains clear line or similar sequences that might hide content"""
    # Check for clear line sequences
    if b'\x1b[2K' in data:  # Clear entire line
        return True
    if b'\x1b[K' in data:   # Clear to end of line
        return True
    # Check for multiple cursor up movements which might indicate content replacement
    if data.count(b'\x1b[1A') > 3:  # Multiple move up sequences
        return True
    # Check if data ends with partial escape sequence that might be a clear
    if data.endswith(b'\x1b[2') or data.endswith(b'\x1b['):
        log_message("DEBUG", "Data ends with partial escape sequence - might be truncated clear")
        return True
    return False


def clear_terminal_state(output_buffer: LineBuffer, cursor_row: int = 1) -> int:
    """Clear terminal state when screen is reset. Returns new cursor row."""
    # Process any unprocessed lines before clearing
    if output_buffer.lines:
        log_message("INFO", f"Processing {len(output_buffer.lines)} lines before clearing output buffer")
        
        # Process each line that hasn't been processed yet
        for idx, line in output_buffer.lines.items():
            if line and line.strip():
                print_line = line.strip()
                if len(print_line) > 120:
                    print_line = line.strip()[:80] + "..." + line.strip()[-30:]
                log_message("DEBUG", f"Processing line {idx} before clear: '{print_line}'")
                # Queue the line for speech before we lose it
                cleaned = clean_text(line)
                if cleaned and not active_profile.should_skip(cleaned, verbosity_level):
                    exception_match = active_profile.matches_exception_pattern(cleaned, verbosity_level)
                    queue_output(strip_profile_symbols(cleaned), idx, exception_match)
    
    output_buffer.clear()
    terminal.line_screen_positions.clear()
    log_message("INFO", "Screen reset detected - cleared buffers after processing")
    return cursor_row


def handle_alternate_screen_buffer(data: bytes) -> bool:
    """Detect and handle alternate screen buffer switches"""
    for seq in ALT_SCREEN_SEQUENCES:
        if seq in data:
            log_message("INFO", "Alternate screen buffer switch detected")
            return True

    return False


def setup_pty_with_scrollback() -> Tuple[int, int, int]:
    """Set up pseudo-terminal with proper scrollback buffer handling"""
    master_fd, slave_fd = pty.openpty()

    if sys.platform != 'win32' and sys.stdin.isatty():
        attrs = termios.tcgetattr(sys.stdin)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

    # Use consolidated terminal size function
    winsize = get_pty_winsize()
    if winsize:
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
        rows, cols, _, _ = struct.unpack('HHHH', winsize)
        log_message("INFO", f"Set PTY size to {rows}x{cols}")

    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    stdin_flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
    fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, stdin_flags | os.O_NONBLOCK)
    
    # Also make stdout non-blocking to prevent BlockingIOError
    stdout_flags = fcntl.fcntl(sys.stdout.fileno(), fcntl.F_GETFL)
    fcntl.fcntl(sys.stdout.fileno(), fcntl.F_SETFL, stdout_flags | os.O_NONBLOCK)

    return master_fd, slave_fd, stdin_flags


def get_pty_winsize() -> Optional[bytes]:
    """Get terminal window size as packed struct for PTY operations"""
    if sys.stdout.isatty():
        try:
            winsize = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b'\0' * 8)
            rows, cols, xpixel, ypixel = struct.unpack('HHHH', winsize)
            
            # Update terminal state
            terminal.terminal_rows = rows
            terminal.terminal_cols = cols
            
            # Adjust cols for margin
            if cols > 2:
                cols -= 2
            
            return struct.pack('HHHH', rows, cols, xpixel, ypixel)
        except Exception as e:
            log_message("ERROR", f"Failed to get terminal size: {e}")
    return None


def update_pty_size(master_fd: int):
    """Update PTY size to match current terminal"""
    winsize = get_pty_winsize()
    if winsize and master_fd is not None:
        try:
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
            rows, cols, _, _ = struct.unpack('HHHH', winsize)
            log_message("INFO", f"Set PTY size to {rows}x{cols}")
        except Exception as e:
            log_message("ERROR", f"Failed to update PTY size: {e}")


def debounced_winch_handler(signum, frame):
    """Handle terminal resize signal with debouncing"""
    with terminal.resize_lock:
        current_time = time.time()
        if current_time - terminal.last_resize_time > RESIZE_DEBOUNCE_TIME:
            terminal.resize_pending = True
            terminal.last_resize_time = current_time
            log_message("INFO", "Terminal resize pending")


def process_pending_resize(master_fd):
    """Process pending terminal resize if needed"""
    with terminal.resize_lock:
        if terminal.resize_pending and master_fd is not None:
            terminal.resize_pending = False
            update_pty_size(master_fd)
            # Terminal size already updated by update_pty_size via get_pty_winsize
            return True
    return False

def _log_tty_and_pgrp_state(master_fd: int, proc: asyncio.subprocess.Process):
    """Best-effort logging of process groups and TTY foreground owner."""
    try:
        parent_pid = os.getpid()
        parent_pgrp = os.getpgrp()
    except Exception:
        parent_pid = -1
        parent_pgrp = -1
    try:
        child_pid = proc.pid
        child_pgrp = os.getpgid(proc.pid)
    except Exception:
        child_pid = -1
        child_pgrp = -1
    # Who is the foreground process group for the PTY?
    try:
        fg_pgrp = os.tcgetpgrp(master_fd)
    except Exception:
        fg_pgrp = -1
    # Log it
    log_message("INFO",
        f"[signals] parent pid={parent_pid} pgrp={parent_pgrp}; "
        f"child pid={child_pid} pgrp={child_pgrp}; "
        f"tty.fg_pgrp={fg_pgrp}"
    )

async def setup_terminal_for_command(cmd: List[str]) -> Tuple[int, int]:
    """Set up PTY and spawn subprocess for command execution"""
    global current_proc, current_master_fd, _original_tty_attrs
    slave_fd = None
    _original_tty_attrs = None
    try:
        current_master_fd, slave_fd, stdin_flags = setup_pty_with_scrollback()

        env = os.environ.copy()
        env['TERM'] = os.environ.get('TERM', 'xterm-256color')
        rows, cols = get_terminal_size()
        env['LINES'] = str(rows)
        env['COLUMNS'] = str(cols - 2 if cols > 2 else cols)

        try:
            current_proc = await asyncio.create_subprocess_exec(
                *cmd,
                # preexec_fn=os.setsid,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env
            )
        except FileNotFoundError as e:
            log_message("ERROR", f"Command not found: {cmd[0]}")
            log_message("ERROR", f"Full command: {' '.join(cmd)}")
            log_message("ERROR", f"Make sure '{cmd[0]}' is installed and in your PATH")
            raise RuntimeError(f"Command '{cmd[0]}' not found. Please install it first.") from e
        # os.tcsetpgrp(master_fd, os.getpgrp())
        # try:
        #     os.tcsetpgrp(current_master_fd, os.getpgrp())
        #     log_message("INFO", f"[signals] Foreground pgrp set to parent {os.getpgrp()}")
        # except Exception as e:
        #     log_message("ERROR", f"[signals] Failed to tcsetpgrp: {e}")
        # os.close(slave_fd)
        # slave_fd = None  # Mark as closed
        _log_tty_and_pgrp_state(current_master_fd, current_proc)
        return slave_fd, stdin_flags
    except Exception:
        # Clean up on error
        if slave_fd is not None:
            try:
                os.close(slave_fd)
            except Exception:
                pass
        if current_master_fd is not None:
            try:
                os.close(current_master_fd)
            except Exception:
                pass
        raise


async def periodic_status_check(master_fd: int, asr_mode: str,
                               last_check_time: float) -> Tuple[bool, float]:
    """Handle periodic ASR status checks and auto-listen"""
    current_time = time.time()
    enabled_auto_listen = False

    # Use slower intervals for tap-to-talk since it's event-driven, not periodic
    check_interval = STATUS_CHECK_INTERVAL * 10 if asr_mode == "tap-to-talk" else STATUS_CHECK_INTERVAL
    
    if current_time - last_check_time > check_interval:
        # Check shared state and stop ASR if it's been disabled
        shared_state = get_shared_state()
        if asr_state.asr_auto_started and not shared_state.asr_enabled:
            try:
                log_message("INFO", "ASR disabled in shared state, stopping dictation")
                asr.stop_dictation()
                asr_state.asr_auto_started = False
            except Exception as e:
                log_message("ERROR", f"Failed to stop ASR after shared state disable: {e}")

        # Minimal processing for tap-to-talk mode (event-driven, not periodic)
        if asr_mode == "tap-to-talk":
            check_tap_to_talk_timeout()
            # Lightweight tap-to-talk state processing without heavy auto-listen logic
            enabled_auto_listen = process_tap_to_talk_state()
        else:
            # Full processing for auto-input mode
            if asr_state.asr_auto_started:
                # Check if TTS is speaking and stop ASR if needed
                if asr.is_recognizing() and tts.is_speaking():
                    try:
                        asr.stop_dictation()
                        asr_state.asr_auto_started = False
                        log_message("INFO", "Stopped ASR because TTS is speaking")
                    except Exception as e:
                        log_message("ERROR", f"Failed to stop ASR during TTS: {e}")
            
            enabled_auto_listen = check_and_enable_auto_listen(asr_mode)
        if enabled_auto_listen and asr_mode != "tap-to-talk":
            try:
                log_message("WARNING", "Send a non breaking space to trigger terminal activity")
                os.write(master_fd, SPACE_THEN_BACK)
            except Exception:
                pass
        return enabled_auto_listen, current_time

    # Return the original last_check_time if we didn't perform the check
    return enabled_auto_listen, last_check_time


async def process_pty_output(data: bytes, output_buffer: LineBuffer,
                            line_buffer: bytes, text_buffer: List[str],
                            prev_line: str,
                            skip_duplicates: bool, cursor_row: int,
                            asr_mode: str, recorder: SessionRecorder) -> Tuple[bytes, List[str], str, int]:
    """Process output data from PTY and queue text for speech"""
    # Log that we received data
    if b'\xe2\x8f\xba' in data:
        log_message("INFO", "[process_pty_output] Data contains response marker!")
    
    if recorder.enabled:
        recorder.record_event('OUTPUT', data)

    # Debug logging for response lines
    if b'\xe2\x8f\xba' in data:  # UTF-8 bytes for ⏺
        log_message("DEBUG", f"[process_pty_output] Data contains response marker, data length: {len(data)}")
        # Log the data around the response marker
        try:
            decoded = data.decode('utf-8', errors='ignore')
            if '⏺' in decoded:
                lines = decoded.split('\n')
                for i, line in enumerate(lines):
                    if '⏺' in line:
                        log_message("DEBUG", f"  Response line {i}: '{line[:100]}...'")
        except Exception:
            pass

    cursor_delta, did_scroll = parse_cursor_movements(data)
    cursor_row = update_cursor_position(cursor_row, cursor_delta, did_scroll)

    # Always add data to buffer first
    line_buffer += data
    
    # Check if data contains clear sequences
    has_clear_sequences = detect_clear_sequences(data)
    
    # Also log when we detect clear sequences
    if b'\x1b[2K' in data:
        log_message("DEBUG", "Data contains clear line sequences (\\x1b[2K found)")
    
    if has_clear_sequences:
        log_message("DEBUG", "Clear sequences detected - processing all buffered lines")
        
        # When clear sequences are detected, we need to process ALL complete lines
        # in the buffer immediately to avoid losing them
        line_buffer, text_buffer, prev_line, cursor_row, _ = process_line_buffer_data(
            line_buffer, output_buffer, text_buffer, prev_line,
            skip_duplicates, cursor_row, asr_mode
        )
        
        # Also process any remaining data in the buffer
        if line_buffer:
            try:
                # Don't clear line_buffer yet - let process_line_buffer_data handle it properly
                log_message("INFO", f"Processing remaining buffer after clear sequences: {len(line_buffer)} bytes")
                # Use process_line_buffer_data to properly split on newlines
                temp_buffer = line_buffer
                line_buffer, text_buffer, prev_line, cursor_row, _ = process_line_buffer_data(
                    temp_buffer, output_buffer, text_buffer, prev_line,
                    skip_duplicates, cursor_row, asr_mode
                )
                
                # If there's still a partial line without newline, keep it in buffer
                # Don't process it yet - wait for more data
                if line_buffer:
                    partial_line = line_buffer.decode('utf-8', errors='ignore')
                    if partial_line.strip():
                        log_message("INFO", f"Keeping partial line in buffer for next chunk: '{partial_line.strip()[:100]}...'")
            except Exception as e:
                log_message("ERROR", f"Error processing remaining buffer: {e}")
    else:
        # Normal processing - no clear sequences
        # Debug: Check what's in line buffer before processing
        if b'\xe2\x8f\xba' in line_buffer:
            log_message("DEBUG", f"[process_pty_output] Line buffer contains response marker before processing, buffer size: {len(line_buffer)}")
        
        line_buffer, text_buffer, prev_line, cursor_row, detected_prompt = process_line_buffer_data(
            line_buffer, output_buffer, text_buffer, prev_line,
            skip_duplicates, cursor_row, asr_mode
        )

    return line_buffer, text_buffer, prev_line, cursor_row


async def handle_stdin_input(master_fd: int, asr_mode: str):
    """Handle input from stdin and forward to PTY"""
    try:
        input_data = await async_read(sys.stdin.fileno(), 4096)
        if input_data:
            # Filter out OSC response sequences (terminal responses to queries like color info)
            # These are responses from the terminal, not actual user input
            input_data = OSC_RESPONSE_PATTERN.sub(b'', input_data)
            if not input_data:
                return  # Nothing left after filtering

            # Check for tap-to-talk keys in tap-to-talk mode (optimized for minimal interference)
            if asr_mode == "tap-to-talk":
                # Fast path: only process backtick, let everything else pass through immediately
                if b'`' in input_data:
                    # Only process backtick for tap-to-talk
                    was_already_active = asr_state.tap_to_talk_active
                    asr_state.tap_to_talk_active = True
                    asr_state.tap_to_talk_last_press = time.time()
                    
                    # Trigger screen redraw only once when first pressed
                    if not was_already_active and not asr_state.tap_to_talk_redraw_triggered:
                        try:
                            await async_write(master_fd, SPACE_THEN_BACK)
                            asr_state.tap_to_talk_redraw_triggered = True
                            log_message("DEBUG", "[TAP-TO-TALK] Triggered screen redraw for mic display")
                        except Exception as e:
                            log_message("ERROR", f"[TAP-TO-TALK] Failed to trigger screen redraw: {e}")
                    
                    # Remove backtick from input_data (don't forward to terminal)
                    input_data = input_data.replace(b'`', b'')
                    if not input_data:
                        return
                # All other keys (Ctrl-C, Esc, etc.) go straight through with no processing delay

            # Check for TTS control keys
            # control_handled = handle_tts_controls(input_data)
            # if control_handled:
            #     return

            # Forward to PTY
            if input_data:  # Only forward if there's data left after filtering
                await async_write(master_fd, input_data)

            if b'\r' in input_data or b'\n' in input_data:
                log_message("INFO", "User pressed Enter, exiting user input mode")
                # Reset pending transcript flag since user manually submitted
                asr_state.has_pending_transcript = False
                return False
            else:
                return True
    except (BlockingIOError, OSError):
        pass


def get_visual_length(text):
    """Calculate the visual length of text, excluding ANSI codes and zero-width characters."""
    # Remove ANSI escape sequences using precompiled pattern
    text_no_ansi = ANSI_SIMPLE_PATTERN.sub('', text)

    # Remove zero-width characters
    text_no_zwc = text_no_ansi.replace('\u200d', '').replace('\u200c', '').replace('\u200b', '')

    # Count actual visual characters
    # Note: This is simplified and doesn't handle all Unicode cases perfectly,
    # but it should work for our use case
    return len(text_no_zwc)

def trim_to_visual_length(text: str, target_len: int) -> str:
    """
    Trim `text` so its visual length (excluding ANSI & zero-width chars)
    is <= target_len. Preserves all ANSI codes from ANSI_PATTERN.
    """
    result = []
    visual_len = 0
    i = 0
    while i < len(text) and visual_len < target_len:
        # Match any ANSI sequence
        match = ANSI_PATTERN.match(text, i)
        if match:
            result.append(match.group())
            i = match.end()
            continue

        char = text[i]
        if char in ('\u200b', '\u200c', '\u200d'):
            # Zero-width char, keep but don’t count
            result.append(char)
        else:
            # Normal character (incl. NBSP)
            result.append(char)
            visual_len += 1
        i += 1

    return ''.join(result)


def pad_to_visual_length(text: str, target_len: int, pad_char: str = ' ') -> str:
    """
    Pad `text` so its visual length == target_len.
    ANSI codes and zero-width chars are preserved.
    """
    current_len = get_visual_length(text)
    if current_len >= target_len:
        return text
    else:
        return text + (pad_char * (target_len - current_len))

def insert_partial_transcript(data_str, asr_state, active_profile):
    """Insert the partial transcript into the terminal output while preserving formatting. Returns modified data_str."""
    # Find the input start position
    active_input_start = active_profile.input_start[0]
    if active_input_start not in data_str and len(active_profile.input_start) > 1 and active_profile.input_start[1] in data_str:
        active_input_start = active_profile.input_start[1]
    log_message("DEBUG", f"active_input_start = {active_input_start}")
    start_idx = data_str.find(active_input_start) if active_input_start else -1
    end_idx = -1
    if start_idx != -1:
        start_idx += len(active_input_start)

        # Find the right border on the same line as the input
        # Look for the next │ after the input start
        line_end = data_str.find('\r\n', start_idx)
        if line_end == -1:
            line_end = len(data_str)

        # Find the rightmost │ before the line end
        search_area = data_str[start_idx:line_end]
        right_border_pos = search_area.rfind('│')
        if right_border_pos != -1:
            # Check for ANSI codes before the │
            # Look backwards from the pipe to find where ANSI sequences start
            check_start = max(0, right_border_pos - 20)
            before_pipe = search_area[check_start:right_border_pos]

            # Find all ANSI escape sequences
            ansi_matches = list(re.finditer(r'\x1b\[[0-9;]*m', before_pipe))

            if ansi_matches:
                # Find the position where continuous ANSI codes start before the pipe
                last_ansi_end = len(before_pipe)
                for match in reversed(ansi_matches):
                    if match.end() == last_ansi_end:
                        # This ANSI code is adjacent to the previous one (or to the pipe)
                        last_ansi_end = match.start()
                    else:
                        # There's a gap, so ANSI codes are not continuous to the pipe
                        break

                if last_ansi_end < len(before_pipe):
                    # We found ANSI codes that continue to the pipe
                    end_idx = start_idx + check_start + last_ansi_end
                else:
                    # No continuous ANSI codes before pipe
                    end_idx = start_idx + right_border_pos
            else:
                # No ANSI codes before pipe
                end_idx = start_idx + right_border_pos
        else:
            # Fallback to finding any │ after start
            end_idx = data_str.find('│', start_idx)
            if end_idx == -1:
                end_idx = line_end
    else:
        start_idx = 0  # Fallback if start not in this chunk

    log_message("DEBUG", f"Found input area from {start_idx}")

    # Extract the input area
    input_area = data_str[start_idx:end_idx]
    log_message("DEBUG", f"Input area raw: {repr(input_area[:100])}")

    # Simple approach: find the last non-space character and insert after it
    last_char_pos = -1
    # Map positions from cleaned to original string
    original_to_clean = []
    clean_input = ''
    i = 0
    while i < len(input_area):
        if input_area[i] == '\x1b' and re.match(r'\x1b\[[0-9;]*[mGKHJ]', input_area[i:]):
            # Skip the ANSI sequence
            match = re.match(r'\x1b\[[0-9;]*[mGKHJ]', input_area[i:])
            i += len(match.group())
        else:
            clean_input += input_area[i]
            original_to_clean.append(i)  # i is the index in input_area corresponding to this clean char
            i += 1

    # Search backwards for last non-space character (could be punctuation)
    last_content_pos = -1
    for j in range(len(clean_input) - 1, -1, -1):
        if clean_input[j] not in ' \t\xa0\x08\u200d':  # Not a space, backspace, or zero-width character
            last_content_pos = j
            break

    # Now we need to ensure exactly one space between content and partial
    if last_content_pos >= 0:
        # We have existing content
        # Count spaces after the last content character, skipping zero-width chars
        spaces_after_content = 0
        for j in range(last_content_pos + 1, len(clean_input)):
            if clean_input[j] == ' ':
                spaces_after_content += 1
            elif clean_input[j] in '\u200d\u200c\u200b':
                # Skip zero-width characters, continue looking for spaces
                continue
            else:
                # Hit a real character, stop counting
                break

        # Position right after the last content character
        last_char_pos = original_to_clean[last_content_pos] + 1

        # We want exactly 1 space between content and partial
        if spaces_after_content == 0:
            # No spaces, add one
            insert_pos = start_idx + last_char_pos
            partial_to_show = asr_state.current_partial
        else:
            # We already have at least one space, don't add another
            insert_pos = start_idx + last_char_pos
            partial_to_show = asr_state.current_partial

        log_message("DEBUG",
                    f"Found content at pos {last_content_pos}, spaces after: {spaces_after_content}, refresh spaces: {asr_state.refresh_spaces_added}")
    else:
        # No existing text, insert at start of input
        insert_pos = start_idx
        partial_to_show = asr_state.current_partial
        log_message("DEBUG", f"No existing text, inserting at input start {insert_pos}")

    # Calculate available space
    remaining_space = end_idx - insert_pos
    log_message("DEBUG", f"Insert pos: {insert_pos}, end_idx: {end_idx}, remaining_space: {remaining_space}")

    # Calculate how much of the partial we can display
    if remaining_space > 0:
        # if len(partial_to_show) > remaining_space - 1:  # Leave room for '…'
        #     partial_to_show = partial_to_show[:remaining_space - 1] + '…'

        # Work only within the input line boundaries
        input_start_len = len(active_input_start) if active_input_start else 0
        before_input = data_str[:start_idx - input_start_len]

        input_content = data_str[start_idx:end_idx]
        after_input = data_str[end_idx:]

        # Calculate position relative to input area start
        relative_pos = insert_pos - start_idx

        # Find where the cursor marker is (reverse video ANSI codes)
        cursor_marker_start = input_content.find('\x1b[7m')

        # Debug: Show what we're working with
        log_message("DEBUG", f"Relative pos: {relative_pos}, cursor at: {cursor_marker_start}")
        log_message("DEBUG", f"Content at insert point: {repr(input_content[max(0, relative_pos - 5):relative_pos + 10])}")

        # Check if relative_pos falls inside an ANSI escape sequence
        i = 0
        adjusted_relative_pos = relative_pos
        while i < relative_pos:
            if i < len(input_content) and input_content[i] == '\x1b' and i + 1 < len(input_content) and input_content[
                i + 1] == '[':
                # Found start of ANSI sequence
                match = re.match(r'\x1b\[[0-9;]*[a-zA-Z]', input_content[i:])
                if match:
                    seq_end = i + len(match.group())
                    if i < relative_pos <= seq_end:
                        adjusted_relative_pos = seq_end
                        log_message("DEBUG",
                                    f"Adjusted insertion position from {relative_pos} to {adjusted_relative_pos} to avoid splitting ANSI sequence")
                        break
                    i = seq_end
                else:
                    i += 1
            else:
                i += 1
        relative_pos = adjusted_relative_pos

        # Calculate visual length of partial transcript
        visual_partial_length = get_visual_length(partial_to_show)

        # We need to find and remove visual_partial_length spaces
        # Start by inserting the partial at the desired position
        before_partial = input_content[:relative_pos]
        after_partial_start = input_content[relative_pos:]

        # Now scan through after_partial_start to remove exactly visual_partial_length spaces
        result = ""
        visual_spaces_removed = 0
        i = 0

        while i < len(after_partial_start):
            char = after_partial_start[i]

            if char == '\x1b' and i + 1 < len(after_partial_start) and after_partial_start[i + 1] == '[':
                # ANSI escape sequence - preserve it
                match = re.match(r'\x1b\[[0-9;]*[a-zA-Z]', after_partial_start[i:])
                if match:
                    result += match.group()
                    i += len(match.group())
                    continue
            elif char in '\u200d\u200c\u200b':
                # Zero-width character - preserve it
                result += char
                i += 1
                continue
            elif char in (' ', '\xa0') and visual_spaces_removed < visual_partial_length:
                visual_spaces_removed += 1
                i += 1
                continue

            # Regular character or space we're keeping
            result += char
            i += 1

        # Construct the new input content
        new_input = before_partial + partial_to_show + result

        # Debug logging
        log_message("DEBUG",
                    f"Visual spaces to remove: {visual_partial_length}, actually removed: {visual_spaces_removed}")
        log_message("DEBUG",
                    f"Visual length check - partial: {visual_partial_length}, new_input visual: {get_visual_length(new_input)}, original visual: {get_visual_length(input_content)}")
        # Debug the result
        log_message("DEBUG", f"Result at insert point: {repr(new_input[max(0, relative_pos - 5):relative_pos + 15])}")

        #target_visual_len = get_visual_length(input_content)
        target_visual_len = min(get_visual_length(partial_to_show), remaining_space)
        new_input = trim_to_visual_length(new_input, target_visual_len)
        new_input = pad_to_visual_length(new_input, target_visual_len)

        # Reconstruct the full string
        data_str = before_input + active_input_start + new_input + after_input

        log_message("DEBUG", f"Input area update: old_visual={get_visual_length(input_content)}, new_visual={get_visual_length(new_input)}")
        log_message("DEBUG", f"Inserted partial transcript: '{partial_to_show}' at position {insert_pos}")
        log_message("DEBUG", f"data_str now [{repr(data_str)}]")
    else:
        log_message("DEBUG", "No space available for partial transcript")

    return data_str


def handle_partial_transcript(text: str):
    """Handle partial transcript from streaming ASR"""
    log_message("INFO", f"[ASR PARTIAL] handle_partial_transcript Received partial transcript: '{text}'")
    if not asr_state.partial_enabled:
        return

    asr_state.current_partial = text # + '\u200b'
    asr_state.last_partial_transcript_time = time.time()
    asr_state.has_pending_transcript = True

    if not text.strip():
        return

    if current_master_fd is not None and asr_state.waiting_for_input:
        try:
            # Use just a space if we already have a partial transcript to avoid malformation
            if asr_state.current_partial and not asr_state.has_pending_transcript:
                # We have a partial transcript displayed and haven't executed query yet
                os.write(current_master_fd, b' ')
                log_message("DEBUG", "Using SPACE for partial transcript refresh to avoid malformation")
            else:
                # First partial or other cases - use the original SPACE_THEN_BACK
                os.write(current_master_fd, SPACE_THEN_BACK)
                log_message("DEBUG", "handle_partial_transcriptUsing SPACE_THEN_BACK for partial transcript refresh")
        except Exception as e:
            log_message("ERROR", f"handle_partial_transcript Failed to trigger screen update: {e}")


def handle_dictated_text(text: str):
    """Handle text from ASR dictation"""
    global current_master_fd
    
    # Check shared state if available
    shared_state = get_shared_state()
    if not shared_state.asr_enabled:
        log_message("DEBUG", "ASR disabled in shared state, ignoring dictated text")
        return
    
    log_message("INFO", f"[ASR TRANSCRIPTION] Received dictated text: '{text}'")
    
    # Update timing for auto-submit
    asr_state.last_finalized_transcript_time = time.time()
    asr_state.has_pending_transcript = True

    # Clear the partial transcript
    if asr_state.partial_enabled:
        handle_partial_transcript("")
        # FIXME Due to a bug with 2nd + partial transcript handling we disable it after the first final transcript.
        asr_state.partial_enabled = True
    
    if current_master_fd is not None:
        try:
            # Convert text to bytes and send to PTY
            is_execute = 'execute' in text.lower()
            if is_execute:
                text = text.replace('execute', '').replace('Execute', '').strip()

            # Prepend ESC if in question mode
            prepend_esc = False
            if asr_state.question_mode:
                prepend_esc = True
                log_message("INFO", "[ASR TRANSCRIPTION] In question mode - prepending esc.")
                asr_state.question_mode = False

            # We need to account for the refresh spaces that were added
            # If we've added refresh spaces, we already have spacing, so don't add another
            # if asr_state.refresh_spaces_added > 0:
            #     # We already have at least one space from refresh triggers
            #     text_bytes = text.encode('utf-8')
            #     log_message("DEBUG", f"Sending final text without extra space (refresh spaces: {asr_state.refresh_spaces_added})")
            # else:
                # No refresh spaces, add one for proper spacing
            text_with_space = ' ' + text
            text_bytes = text_with_space.encode('utf-8')
            if prepend_esc:
                text_bytes = ESC + text_bytes
            log_message("DEBUG", "Sending final text with added space (no refresh spaces)")
            
            os.write(current_master_fd, text_bytes)
            
            # Reset the refresh spaces counter
            asr_state.refresh_spaces_added = 0
            
            if is_execute:
                os.write(current_master_fd, RETURN)
            log_message("INFO", f"[ASR TRANSCRIPTION] Successfully sent to PTY: '{text}'")
        except Exception as e:
            log_message("ERROR", f"[ASR TRANSCRIPTION] Failed to send dictated text: {e}")
    else:
        log_message("WARNING", "[ASR TRANSCRIPTION] No master_fd available to send dictated text")


def check_tap_to_talk_timeout():
    """Check if tap-to-talk has timed out (key release simulation)"""
    TAP_TO_TALK_TIMEOUT = 0.5  # Consider key released after 500ms of no press
    
    if asr_state.tap_to_talk_active and time.time() - asr_state.tap_to_talk_last_press > TAP_TO_TALK_TIMEOUT:
        asr_state.tap_to_talk_active = False
        asr_state.tap_to_talk_last_release = time.time()
        asr_state.tap_to_talk_redraw_triggered = False  # Reset for next session
        log_message("INFO", "Tap-to-talk timeout - considering key released")
        # Trigger screen redraw to hide microphone immediately
        global current_master_fd
        if current_master_fd is not None:
            try:
                os.write(current_master_fd, SPACE_THEN_BACK)
                log_message("DEBUG", "[TAP-TO-TALK] Triggered screen redraw for mic hide")
            except Exception as e:
                log_message("ERROR", f"[TAP-TO-TALK] Failed to trigger screen redraw on timeout: {e}")
        return True
    return False


def process_tap_to_talk_state():
    """Lightweight tap-to-talk state processing (extracted from check_and_enable_auto_listen for performance)"""
    # Tap-to-talk mode: keep streaming session alive but pause/unpause processing
    if not asr_state.asr_auto_started:
        # Initialize ASR streaming session once (stays running)
        try:
            log_message("INFO", "Starting ASR streaming session for tap-to-talk mode...")
            asr.start_dictation(handle_dictated_text, handle_partial_transcript)
            asr_state.asr_auto_started = True
            # Start paused - will unpause when key is pressed
            asr.set_ignore_input(True)
            log_message("INFO", "ASR streaming started and paused (waiting for tap-to-talk)")
            return True
        except Exception as e:
            log_message("ERROR", f"Failed to start ASR streaming (tap-to-talk): {e}")
    
    # Control audio processing based on key state
    if asr_state.tap_to_talk_active:
        # Key pressed - unpause audio processing
        if asr.is_ignoring_input():
            # Reset counter for new tap-to-talk session
            asr.set_ignore_input(False)
            log_message("INFO", "ASR activated (tap-to-talk key pressed)")
    else:
        # Key released - pause audio processing and clear any pending audio chunks
        if not asr.is_ignoring_input():
            asr_state.tap_to_talk_last_release = time.time()
            asr.set_ignore_input(True)
            log_message("INFO", "ASR paused (tap-to-talk key released)")
    
    return False


def ensure_asr_cleanup():
    """Ensure ASR is properly cleaned up when MCP disables ASR or errors occur."""
    shared_state = get_shared_state()
    
    # Check if ASR should be cleaned up
    if not shared_state.asr_enabled and shared_state.asr_initialized:
        log_message("INFO", "ASR disabled but still initialized - cleaning up")
        
        try:
            # Stop active dictation if running
            if asr.is_dictation_active():
                asr.stop_dictation()
                log_message("INFO", "Stopped active dictation")
                
            # Reset ASR state
            if asr_state:
                asr_state.asr_auto_started = False
                asr_state.current_partial = ""
                
            # Mark as uninitialized
            shared_state.set_asr_initialized(False)
            log_message("INFO", "ASR cleanup completed")
            
        except Exception as e:
            log_message("ERROR", f"Error during ASR cleanup: {e}")


def ensure_asr_initialized():
    """Ensure ASR is initialized if it's enabled but not yet initialized; returns True if ASR was newly initialized."""
    shared_state = get_shared_state()
    
    # Check if ASR should be initialized
    if shared_state.asr_enabled and not shared_state.asr_initialized:
        log_message("INFO", "ASR enabled but not initialized - initializing now")
        
        try:
            # Use the provider determined by early initialization
            asr_provider = shared_state.asr_provider
            
            # If no provider is set, select the best available one
            if asr_provider is None:
                log_message("WARNING", "ASR provider is None, selecting best available provider")
                asr_provider = asr.select_best_asr_provider()
                if asr_provider:
                    shared_state.asr_provider = asr_provider
                    log_message("INFO", f"Selected ASR provider: {asr_provider}")
                else:
                    log_message("ERROR", "No ASR provider available")
                    return False

            log_message("INFO", f"Initializing ASR with provider: {asr_provider}")
            
            # Create config for the selected provider
            from .asr import ASRConfig
            asr_config = ASRConfig(provider=asr_provider)
            
            # Start ASR with the standard callbacks
            engine = asr.start_dictation(handle_dictated_text, handle_partial_transcript, config=asr_config)

            # Get the actual provider that was initialized
            actual_provider = engine.provider_config.provider if engine and engine.provider_config else asr_provider

            # Check if we're in file mode
            if shared_state.asr_mode == 'file' and shared_state.asr_source_file:
                delay = getattr(shared_state, 'asr_file_delay', 0.1)
                log_message("INFO", f"ASR in file mode, processing audio file: {shared_state.asr_source_file} (delay: {delay}s)")
                # Process the audio file directly instead of starting continuous listening
                engine.process_audio_file(shared_state.asr_source_file, delay=delay)
                log_message("INFO", f"ASR file processing complete with provider: {actual_provider}")
            else:
                # Start in paused state so auto-input logic can manage it
                asr.set_ignore_input(True)
                log_message("INFO", f"ASR initialized and paused for auto-input management with provider: {actual_provider}")

            # Update shared state with the actual provider that was successfully initialized
            shared_state.set_asr_initialized(True, provider=actual_provider)

            # Reset ASR state for clean startup
            if asr_state:
                asr_state.asr_auto_started = False

            return True
            
        except Exception as e:
            error_msg = str(e)
            log_message("ERROR", f"Failed to initialize ASR: {error_msg}")
            
            # Print error to user (consistent with other error handling in codebase)
            print(f"ASR Error: {error_msg}")
            
            # If this was a validation failure, try falling back to best available provider
            if "validation failed" in error_msg.lower():
                try:
                    fallback_provider = asr.select_best_asr_provider()
                    if fallback_provider and fallback_provider != asr_provider:
                        print(f"Falling back to ASR provider: {fallback_provider}")
                        log_message("INFO", f"Falling back to ASR provider: {fallback_provider}")
                        
                        # Update environment and shared state to use fallback
                        os.environ['TALKITO_PREFERRED_ASR_PROVIDER'] = fallback_provider
                        shared_state.asr_provider = fallback_provider
                        
                        # Try initializing with fallback provider
                        fallback_config = ASRConfig(provider=fallback_provider)  
                        engine = asr.start_dictation(handle_dictated_text, handle_partial_transcript, config=fallback_config)
                        
                        actual_provider = engine.provider_config.provider if engine and engine.provider_config else fallback_provider
                        asr.set_ignore_input(True)
                        log_message("INFO", f"ASR initialized and paused for auto-input management with provider: {actual_provider}")
                        shared_state.set_asr_initialized(True, provider=actual_provider)
                        
                        if asr_state:
                            asr_state.asr_auto_started = False
                        return True
                        
                except Exception as fallback_error:
                    log_message("ERROR", f"Fallback ASR initialization failed: {fallback_error}")
                    print(f"ASR fallback failed: {fallback_error}")
            
            # Mark as not initialized
            shared_state.set_asr_initialized(False)
            return False
            
    return False


def ensure_tts_state_sync():
    """Ensure TTS state is synchronized and clear pending text if TTS is disabled"""
    shared_state = get_shared_state()
    
    # If TTS was disabled, clear any pending speech text
    if not shared_state.tts_enabled and terminal and terminal.pending_speech_text:
        log_message("INFO", "TTS disabled, clearing pending speech text")
        terminal.pending_speech_text.clear()
        terminal.pending_text_line_number = None


def ensure_comms_initialized():
    """Ensure communication channels are initialized based on shared state"""
    global comm_manager
    shared_state = get_shared_state()
    
    # First, check if we need to create comm_manager at all
    if not comm_manager and (shared_state.slack_mode_active or shared_state.whatsapp_mode_active):
        log_message("INFO", "Communication modes active but comm_manager not initialized - creating now")
        providers = []
        if shared_state.slack_mode_active:
            providers.append('slack')
        if shared_state.whatsapp_mode_active:
            providers.append('whatsapp')
            
        try:
            comm_manager = comms.setup_communication(providers=providers)
            log_message("INFO", f"Created comm_manager with providers: {providers}")
            return True
        except Exception as e:
            log_message("ERROR", f"Failed to create comm_manager: {e}")
            return False
    
    # If we have a comm_manager but need to add providers, we need to recreate it
    # because providers can't be added dynamically
    if comm_manager:
        has_slack = any(hasattr(p, '__class__') and p.__class__.__name__ == 'SlackProvider' for p in (comm_manager.providers if hasattr(comm_manager, 'providers') else []))
        has_whatsapp = any(hasattr(p, '__class__') and p.__class__.__name__ in ['WhatsAppProvider', 'TwilioWhatsAppProvider'] for p in (comm_manager.providers if hasattr(comm_manager, 'providers') else []))
        
        needs_slack = shared_state.slack_mode_active and not has_slack
        needs_whatsapp = shared_state.whatsapp_mode_active and not has_whatsapp
        
        if needs_slack or needs_whatsapp:
            log_message("INFO", f"Need to add providers - recreating comm_manager (needs_slack={needs_slack}, needs_whatsapp={needs_whatsapp})")
            
            # Stop existing comm_manager first to avoid port conflicts
            if hasattr(comm_manager, 'stop'):
                try:
                    comm_manager.stop()
                    log_message("INFO", "Stopped existing comm_manager")
                except Exception as e:
                    log_message("WARNING", f"Failed to stop existing comm_manager: {e}")
            
            # Build provider list
            providers = []
            if shared_state.slack_mode_active:
                providers.append('slack')
            if shared_state.whatsapp_mode_active:
                providers.append('whatsapp')
                
            try:
                comm_manager = comms.setup_communication(providers=providers)
                log_message("INFO", f"Recreated comm_manager with providers: {providers}")
                return True
            except Exception as e:
                log_message("ERROR", f"Failed to recreate comm_manager: {e}")
                return False
                
    return False


def check_and_enable_auto_listen(asr_mode: str = "auto-input"):
    """Check if conditions are met to auto-enable ASR based on mode"""
    # First ensure ASR state is correct (initialize if needed, cleanup if disabled)
    ensure_asr_initialized()
    ensure_asr_cleanup()
    
    # Also ensure TTS state is synchronized
    ensure_tts_state_sync()
    
    # Ensure communication channels are initialized if needed
    ensure_comms_initialized()
        
    # Check shared state if available
    shared_state = get_shared_state()
    if not shared_state.asr_enabled:
        log_message("DEBUG", "ASR disabled in shared state, not auto-enabling")
        return False
    
    # If ASR is not initialized at this point, we can't proceed
    if not shared_state.asr_initialized:
        log_message("DEBUG", "ASR not initialized, cannot auto-enable")
        return False
        
    # If ASR was explicitly enabled via MCP when CLI had it off, use stored user preference
    if asr_mode == "off" and shared_state.asr_enabled:
        # Respect the user's stored ASR mode preference instead of blindly overriding to auto-input
        stored_mode = getattr(shared_state, 'asr_mode', 'auto-input')
        asr_mode = stored_mode
        log_message("DEBUG", f"ASR enabled via MCP, using stored user preference: {stored_mode}")
        
    # Log the current state for debugging
    is_tts_speaking = tts.is_speaking()
    is_asr_recognizing = asr.is_recognizing()
    
    log_message("DEBUG", f"check_and_enable_auto_listen: mode={asr_mode}, waiting_for_input={asr_state.waiting_for_input}, "
                        f"is_tts_speaking={is_tts_speaking}, is_asr_recognizing={is_asr_recognizing}, "
                        f"asr_auto_started={asr_state.asr_auto_started}, tap_to_talk_active={asr_state.tap_to_talk_active}")
    
    if asr_mode == "auto-input":
        # Auto-input mode: enable when waiting and TTS is done
        if asr_state.waiting_for_input and not is_tts_speaking and not is_asr_recognizing:
            if not asr_state.asr_auto_started:
                try:
                    # Check if ASR is already initialized
                    if asr.is_dictation_active():
                        # Just unpause it
                        asr.set_ignore_input(False)
                        asr_state.asr_auto_started = True
                        log_message("INFO", "Unpaused pre-initialized ASR")
                    else:
                        # Initialize it now
                        log_message("INFO", "Starting ASR dictation (auto-input mode)...")
                        asr.start_dictation(handle_dictated_text, handle_partial_transcript)
                        asr_state.asr_auto_started = True
                        log_message("INFO", "Auto-enabled ASR: AI waiting and TTS finished")
                    return True
                except Exception as e:
                    log_message("ERROR", f"Failed to auto-start ASR: {e}")
        elif not asr_state.waiting_for_input and asr_state.asr_auto_started:
            # Pause ASR when AI starts responding (don't fully stop it)
            try:
                asr.set_ignore_input(True)
                asr_state.asr_auto_started = False
                log_message("INFO", "Paused ASR: AI started responding")
            except Exception as e:
                log_message("ERROR", f"Failed to pause ASR: {e}")
                
    elif asr_mode == "tap-to-talk":
        # Tap-to-talk logic moved to process_tap_to_talk_state() for performance
        # This function is now called only from non-periodic events
        return False
    
    return False

def handle_tts_controls(input_data: bytes) -> bool:
    """Handle TTS control keys. Returns True if handled."""
    # ANSI escape sequences for arrow keys
    KEY_LEFT = b'\x1b[D'
    KEY_RIGHT = b'\x1b[C'

    if KEY_LEFT in input_data:
        # Left arrow - rewind current item (for now, just restart it)
        current = tts.get_current_speech_item()
        if current:
            tts.skip_current()
            tts.queue_for_speech(current.original_text, current.line_number, current.source)
        return True

    elif KEY_RIGHT in input_data:
        # Right arrow - skip current item
        tts.skip_current()
        return True

    return False


async def drain_pty_output(master_fd: int, line_buffer: bytes):
    """Drain any remaining output from PTY when process exits"""
    try:
        while True:
            data = await async_read(master_fd, 4096)
            if not data:
                break
            await async_stdout_write(data)
            line_buffer += data
    except Exception:
        pass


def advance_display_chars(s: str, start: int, n: int) -> int:
    """Move n printable characters forward through s, skipping ANSI sequences. Returns index after those characters."""
    i, printed = start, 0
    while i < len(s) and printed < n:
        if s[i] == '\x1b':
            m = ANSI_RE.match(s, i)
            if m:                         # skip full escape
                i = m.end()
                continue
        i += 1                            # count a real character
        printed += 1
    return i

async def run_command(cmd: List[str], asr_mode: str = "auto-input", record_file: str = None) -> int:
    """Run the command and process its output with PTY support for colors"""
    global current_master_fd, current_proc, _original_tty_attrs

    # Initialize to ensure cleanup on error
    stdin_flags = None
    try:
        # Set up terminal and spawn process
        slave_fd, stdin_flags = await setup_terminal_for_command(cmd)
        
        # Initialize recorder if requested
        recorder = SessionRecorder(record_file)
    
        if recorder.enabled:
            log_message("INFO", f"Record mode enabled - saving raw output to: {record_file}")
            asr_mode = "off"

        # Process state
        buffer = []
    
        # White input state tracking (outside the main loop)
        in_input = False
        prev_line = ""
        line_buffer = b""

        # Buffer for incomplete UTF-8 sequences
        incomplete_utf8_buffer = b""

        # Use LineBuffer to track all output
        output_buffer = LineBuffer()
        
        # Initialize non-blocking output buffer
        stdout_buffer = OutputBuffer()
        
        # Set stdout to non-blocking mode
        stdout_flags = fcntl.fcntl(sys.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdout.fileno(), fcntl.F_SETFL, stdout_flags | os.O_NONBLOCK)

        # Cursor tracking
        current_cursor_row = 1
        get_terminal_size()  # Initialize terminal dimensions

        # Terminal state tracking
        in_alternate_buffer = False
        consecutive_redraws = 0
        last_redraw_time = 0
        skip_duplicate_mode = False
        
        # For periodic status updates
        last_status_check = 0

        if sys.stdin.isatty():
            _original_tty_attrs = termios.tcgetattr(sys.stdin)
            # Store globally for signal handler
            tty.setraw(sys.stdin.fileno())

        # Main processing loop
        while True:
            # Capture runtime ASR mode changes (e.g., toggled via MCP)
            shared_state = get_shared_state()
            shared_mode = getattr(shared_state, 'asr_mode', None) or asr_mode
            if shared_mode != asr_mode:
                log_message("INFO", f"[CORE] Runtime ASR mode change detected: {asr_mode} -> {shared_mode}")
                asr_mode = shared_mode
                # Reset tap-to-talk state whenever mode changes to avoid stale flags
                if asr_state:
                    asr_state.tap_to_talk_active = False
                    asr_state.tap_to_talk_redraw_triggered = False
                    asr_state.tap_to_talk_last_press = 0
                    asr_state.tap_to_talk_last_release = time.time()
                # Ensure dictation engine respects new mode
                ensure_asr_initialized()

            if process_pending_resize(current_master_fd):
                continue
                
            # Periodic status update for dictation indicator and auto-listen check
            enabled_auto_listen, last_status_check = await periodic_status_check(
                current_master_fd, asr_mode, last_status_check)

            # Check if we need to write buffered data
            wlist = [sys.stdout] if len(stdout_buffer) > 0 else []
            
            # Reduced timeout from 0.01 to 0.001 for better responsiveness
            rlist, wlist_ready, _ = select.select([sys.stdin, current_master_fd], wlist, [], 0.001)
            
            # Try to flush buffered output if stdout is writable
            if sys.stdout in wlist_ready:
                written = stdout_buffer.write_to_stdout()
                if written > 0 and is_logging_enabled():
                    log_message("DEBUG", f"Flushed {written} bytes from output buffer")
            
            # Check for input from communication channels
            comms_input = check_comms_input()
            prepend_esc = False
            if comms_input and current_master_fd:
                try:
                    # Prepend ESC if in question mode
                    if asr_state.question_mode:
                        prepend_esc = True
                        log_message("INFO", "[COMMS] In question mode - prepending esc.")
                        asr_state.question_mode = False
                    
                    input_bytes = comms_input.encode('utf-8')
                    if prepend_esc:
                        input_bytes = ESC + input_bytes
                    await async_write(current_master_fd, input_bytes)
                    await async_write(current_master_fd, RETURN)
                    sys.stdout.flush()
                    log_message("INFO", f"Sent comms input to PTY: {comms_input}")
                except Exception as e:
                    log_message("ERROR", f"Failed to send comms input to PTY: {e}")

            # Check for auto-submit of dictated text after 3 seconds of silence
            if asr_state.has_pending_transcript and current_master_fd:
                current_time = time.time()
                time_since_finalized = current_time - asr_state.last_finalized_transcript_time
                time_since_partial = current_time - asr_state.last_partial_transcript_time
                
                # Check if we need to wait for tap-to-talk completion
                shared_state = get_shared_state()
                tap_to_talk_complete = False
                if shared_state.asr_mode == "tap-to-talk":
                    # In tap-to-talk mode, wait until key is released AND all outstanding frames are processed
                    key_released = not asr_state.tap_to_talk_active
                    outstanding_frames = asr.get_tap_to_talk_outstanding_frames() if asr else 0
                    all_frames_processed = outstanding_frames == 0
                    time_since_release = current_time - asr_state.tap_to_talk_last_release if asr_state.tap_to_talk_last_release > 0 else 0
                    min_wait_satisfied = time_since_release >= 0.1  # 100ms minimum wait after key release
                    
                    tap_to_talk_complete = key_released and all_frames_processed and min_wait_satisfied
                    log_message("DEBUG", f"[AUTO-SUBMIT] Tap-to-talk completion check: key_released={key_released}, outstanding_frames={outstanding_frames}, all_frames_processed={all_frames_processed}, time_since_release={time_since_release:.3f}s, min_wait_satisfied={min_wait_satisfied}")
                
                # Auto-submit if:
                # - It's been at least 2.5 seconds since the last finalized transcript OR
                # - It's been at least 200ms since the last finalized transcript AND no partial transcripts in 3 seconds
                # - In tap-to-talk mode: key is released AND all outstanding frames are processed
                silence_threshold = 2.5  # 2.5 seconds of actual silence
                if tap_to_talk_complete or (time_since_finalized >= silence_threshold or (time_since_finalized >= 0.2 and time_since_partial >= silence_threshold)):
                    try:
                        await async_write(current_master_fd, RETURN)
                        asr_state.has_pending_transcript = False
                        log_message("INFO", "[ASR AUTO-SUBMIT] Auto-submitted dictated text after 3 seconds of silence")
                    except Exception as e:
                        log_message("ERROR", f"[ASR AUTO-SUBMIT] Failed to auto-submit: {e}")

            if sys.stdin in rlist:
                # Handle record mode for input
                if recorder.enabled:
                    try:
                        input_data = await async_read(sys.stdin.fileno(), 4096)
                        if input_data:
                            # Filter out OSC response sequences (terminal responses)
                            input_data = OSC_RESPONSE_PATTERN.sub(b'', input_data)
                            if input_data:
                                recorder.record_event('INPUT', input_data)
                                # Forward to PTY
                                await async_write(current_master_fd, input_data)
                    except (BlockingIOError, OSError):
                        pass
                else:
                    await handle_stdin_input(current_master_fd, asr_mode)

            if current_master_fd in rlist or enabled_auto_listen:
                try:
                    data = await async_read(current_master_fd, PTY_READ_SIZE)
                    if not data:
                        break

                    # Prepend any incomplete UTF-8 bytes from previous read
                    data = incomplete_utf8_buffer + data
                    incomplete_utf8_buffer = b""

                    # Handle record mode - write raw data to file
                    if recorder.enabled:
                        recorder.record_event('OUTPUT', data)

                    if handle_alternate_screen_buffer(data):
                        in_alternate_buffer = not in_alternate_buffer
                        # Always process lines before clearing on buffer switch
                        current_cursor_row = clear_terminal_state(output_buffer, 1)

                    # Parse cursor movements before writing data
                    cursor_delta, did_scroll = parse_cursor_movements(data)

                    if detect_screen_reset(data):
                        consecutive_redraws += 1
                        current_time = time.time()

                        if consecutive_redraws > 2:
                            skip_duplicate_mode = True
                            log_message("WARNING", "Entering duplicate skip mode due to screen redraws")
                            # Always process before clearing
                            current_cursor_row = clear_terminal_state(output_buffer, 1)

                        if current_time - last_redraw_time > 5.0:
                            consecutive_redraws = 0
                            skip_duplicate_mode = False

                        last_redraw_time = current_time

                    # Check if we should modify the output for ASR indicator or partial transcript
                    output_data = data

                    try:
                        data_str = output_data.decode('utf-8')
                    except UnicodeDecodeError as e:
                        # Save incomplete bytes for next iteration
                        incomplete_utf8_buffer = output_data[e.start:]
                        output_data = output_data[:e.start]
                        data_str = output_data.decode('utf-8')

                    if not in_input and active_profile.input_start:
                        for input_start in active_profile.input_start:
                            if input_start in data_str:
                                start_idx = data_str.find(input_start) + len(input_start)
                                log_message("DEBUG", f"Found input start at position {start_idx}")
                                in_input = True
                                break

                    # Display partial transcript if available and we're in input area
                    if asr_state.current_partial and in_input:
                        data_str = insert_partial_transcript(data_str, asr_state, active_profile)
                        # data_str = data_str.replace('\u200b', ' '*len(asr_state.current_partial))

                    in_input = False

                    output_data = data_str.encode('utf-8')
                    output_data = output_data.replace(SPACE_THEN_BACK, b'')

                    # Show microphone emoji if ASR is active
                    show_mic_for_auto = asr_state.asr_auto_started and asr_state.waiting_for_input and asr_mode != "off" and not asr.is_ignoring_input()
                    show_mic_for_tap_to_talk = asr_mode == "tap-to-talk" and asr_state.tap_to_talk_active

                    # log_message("DEBUG", f"modify_prompt_for_asr against output_data {output_data}")
                    if show_mic_for_auto or show_mic_for_tap_to_talk:
                        output_data = modify_prompt_for_asr(output_data, active_profile.input_start, active_profile.input_mic_replace)
                    # elif tts.is_speaking():
                    #     output_data = modify_prompt_for_asr(output_data, active_profile.input_start,
                    #                                         active_profile.input_speaker_replace)

                    # Try direct write first
                    try:
                        written = os.write(sys.stdout.fileno(), output_data)
                        if written < len(output_data):
                            # Partial write - buffer the rest
                            stdout_buffer.add(output_data[written:])
                            if is_logging_enabled():
                                log_message("DEBUG", f"Partial write: {written}/{len(output_data)} bytes, buffered {len(output_data)-written}")
                    except BlockingIOError:
                        # Can't write at all - buffer everything
                        stdout_buffer.add(output_data)
                        if is_logging_enabled():
                            log_message("DEBUG", f"Stdout blocked, buffered {len(output_data)} bytes")
                    except OSError as e:
                        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                            # Same as BlockingIOError
                            stdout_buffer.add(output_data)
                        else:
                            log_message("ERROR", f"Error writing to stdout: {e}")
                            log_message("ERROR", f"Traceback: {traceback.format_exc()}")

                    # Process lines even in alternate buffer to ensure nothing is missed
                    # if in_alternate_buffer:
                    #     log_message("DEBUG", f"Skipping process_pty_output due to in_alternate_buffer=True")
                    #     continue

                    # Process the output data
                    
                    # Use the original data minus any incomplete UTF-8 bytes
                    data_for_processing = data
                    if incomplete_utf8_buffer:
                        # Remove the incomplete bytes from the end
                        data_for_processing = data[:-len(incomplete_utf8_buffer)]
                    
                    line_buffer, buffer, prev_line, current_cursor_row = await process_pty_output(
                        data_for_processing, output_buffer, line_buffer, buffer, prev_line,
                        skip_duplicate_mode, current_cursor_row,
                        asr_mode, recorder
                    )

                except BlockingIOError:
                    pass
                except OSError as e:
                    log_message("ERROR", f"PTY read error: {e}")
                    break
                except Exception as e:
                    log_message("ERROR", f"Unexpected error in PTY read loop: {e}")
                    log_message("ERROR", f"Traceback: {traceback.format_exc()}")

            if current_proc.returncode is not None:
                if recorder.enabled:
                    # In record mode, drain and record any remaining output
                    try:
                        while True:
                            data = await async_read(current_master_fd, PTY_READ_SIZE)
                            if not data:
                                break
                            recorder.record_event('OUTPUT', data)
                            await async_stdout_write(data)
                            # IMPORTANT: Also process this data for TTS
                            line_buffer += data

                            # Process complete lines in the drained data
                            while b'\n' in line_buffer:
                                nl_pos = line_buffer.find(b'\n')
                                line_bytes = line_buffer[:nl_pos]
                                line_buffer = line_buffer[nl_pos + 1:]

                                try:
                                    line = line_bytes.decode('utf-8')
                                except UnicodeDecodeError:
                                    # Skip malformed lines
                                    log_message("WARNING", "Skipping malformed UTF-8 line")
                                    continue
                                line_idx, action = output_buffer.add_or_update_line(line)

                                if action in ['added', 'modified']:
                                    buffer, prev_line, _ = process_line(
                                        line, buffer, prev_line, False, line_idx, asr_mode
                                    )
                    except Exception:
                        pass
                else:
                    await drain_pty_output(current_master_fd, line_buffer)
                break

        # Process any remaining content after main loop exits
        if line_buffer:
            try:
                line = line_buffer.decode('utf-8')
            except UnicodeDecodeError:
                line = line_buffer.decode('utf-8', errors='replace')
                log_message("WARNING", "Incomplete UTF-8 sequence in final line buffer")
            line_idx, action = output_buffer.add_or_update_line(line)

            if action in ['added', 'modified']:
                buffer, prev_line, _ = process_line(line, buffer, prev_line, False, line_idx, asr_mode)

        if buffer:
            # Use the last line index for any remaining buffer content
            last_idx = output_buffer.next_index - 1 if output_buffer.next_index > 0 else 0
            process_remaining_buffer(buffer, last_idx)

        # Flush any pending speech text before command completes
        send_pending_text()

        # Re-install signal handlers after subprocess completion to regain control
        try:
            log_message("DEBUG", "Re-installing signal handlers after subprocess completion")

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGWINCH, debounced_winch_handler)
            log_message("DEBUG", "Signal handlers re-installed successfully")
        except Exception as e:
            log_message("ERROR", f"Failed to re-install signal handlers: {e}")

        # Stop ASR if it was running
        try:
            asr.stop_dictation()
        except Exception:
            pass

        # Log final buffer statistics
        log_message("INFO", f"Command completed. Total lines in buffer: {output_buffer.get_line_count()}")

        # Save recorded data
        recorder.save()

        if current_proc:
            await current_proc.wait()
        
        # Restore stdout to blocking mode
        try:
            fcntl.fcntl(sys.stdout.fileno(), fcntl.F_SETFL, stdout_flags)
            
            # Flush any remaining buffered output
            while len(stdout_buffer) > 0:
                written = stdout_buffer.write_to_stdout()
                if written == 0:
                    # Force flush by writing directly in blocking mode
                    remaining = bytes(stdout_buffer.buffer)
                    if remaining:
                        await async_stdout_write(remaining)
                    break
        except Exception as e:
            log_message("ERROR", f"Error restoring stdout: {e}")

        return current_proc.returncode if current_proc else 1

    finally:
        # Clean up process if still running
        if current_proc is not None and current_proc.returncode is None:
            try:
                current_proc.terminate()
                await asyncio.wait_for(current_proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                current_proc.kill()
                await current_proc.wait()
            except Exception as e:
                log_message("ERROR", f"Failed to terminate process: {e}")
        current_proc = None
        
        # Close PTY file descriptor if it was opened
        if current_master_fd is not None:
            try:
                os.close(current_master_fd)
                log_message("INFO", "Closed master_fd in finally block")
            except Exception as e:
                log_message("ERROR", f"Failed to close master_fd: {e}")
            current_master_fd = None
        
        if _original_tty_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, _original_tty_attrs)
                log_message("INFO", "Restored terminal attributes in finally block")
            except Exception as e:
                log_message("ERROR", f"Failed to restore terminal attributes in finally: {e}")
            # Clear global reference
            _original_tty_attrs = None

        if stdin_flags is not None:
            try:
                fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, stdin_flags)
            except Exception as e:
                log_message("ERROR", f"Failed to restore stdin flags: {e}")
        
        # Ensure cursor is visible
        try:
            with terminal.terminal_write_lock:
                sys.stdout.write('\033[?25h')  # Show cursor
                sys.stdout.flush()
        except Exception:
            pass


def process_remaining_buffer(buffer: List[str], line_idx: int) -> None:
    """Process any remaining text in the buffer and queue it for speech"""
    if buffer:
        filtered_buffer = [line for line in buffer if line and line.strip()]
        if filtered_buffer:
            final_text = '. '.join(filtered_buffer)
            exception_match = active_profile and active_profile.matches_exception_pattern(final_text, verbosity_level)
            queue_output(strip_profile_symbols(final_text), line_idx, exception_match)


def process_line_buffer_data(line_buffer: bytes, output_buffer: LineBuffer,
                           text_buffer: List[str], prev_line: str,
                           skip_duplicates: bool, cursor_row: int,
                           asr_mode: str) -> Tuple[bytes, List[str], str, int, bool]:
    """Process buffered line data and handle complete lines"""
    detected_prompt = False
    
    # Debug logging for response lines in buffer
    if b'\xe2\x8f\xba' in line_buffer:  # UTF-8 bytes for ⏺
        log_message("DEBUG", f"[process_line_buffer_data] Buffer contains response line marker, buffer size: {len(line_buffer)}")
        # Show the lines in the buffer
        try:
            decoded = line_buffer.decode('utf-8', errors='ignore')
            lines = decoded.split('\n')
            log_message("DEBUG", f"  Buffer has {len(lines)} lines after split")
            for i, line in enumerate(lines):
                if line.strip():
                    log_message("DEBUG", f"  Buffer line {i}: '{line[:80]}...'")
                else:
                    log_message("DEBUG", f"  Buffer line {i}: (empty)")
        except Exception:
            pass
    
    while len(line_buffer) > 0:
        nl_pos = line_buffer.find(b'\n')
        if nl_pos == -1:
            if active_profile.needs_full_lines:
                break
            # No newline found, process remaining buffer and exit
            line_bytes = line_buffer
            line_buffer = b''
        else:
            # Newline found, process line and continue
            line_bytes = line_buffer[:nl_pos]
            line_buffer = line_buffer[nl_pos + 1:]
        
        line = line_bytes.decode('utf-8', errors='ignore')

        line_idx, action = output_buffer.add_or_update_line(line)
        
        if action == 'added':
            track_line_position(line_idx, cursor_row)

        if line.strip():
            log_message("INFO", f"Processing line {line_idx} (action={action}, row={cursor_row}): '{line.strip()}'")
        
        # Check if this is a Ctrl-C prompt from Claude
        # if "Press Ctrl-C again to exit" in line:
        #     log_message("INFO", "Detected Ctrl-C prompt - stopping all TTS")
        #     # Use skip_all instead of stop_tts_immediately to avoid shutting down the TTS system
        #     tts.skip_all()
        
        # Only process lines that have been added or modified
        if action in ['added', 'modified']:
            text_buffer, prev_line, detected_prompt = process_line(
                line, text_buffer, prev_line, skip_duplicates, line_idx, asr_mode
            )
        else:
            # For unchanged lines, don't process but preserve state
            detected_prompt = False

        response_prefix = active_profile.response_prefix
        if clean_text(line).strip().startswith(response_prefix):
            asr_state.waiting_for_input = False
            asr_state.refresh_spaces_added = 0  # Reset refresh spaces when response starts
            check_and_enable_auto_listen(asr_mode)

        if detected_prompt:
            asr_state.waiting_for_input = True
            asr_state.refresh_spaces_added = 0  # Reset refresh spaces for new prompt
            log_message("INFO", "Detected prompt")
            check_and_enable_auto_listen(asr_mode)

    return line_buffer, text_buffer, prev_line, cursor_row, detected_prompt


# Global variables for cleanup
_original_tty_attrs = None
_cleanup_done = False

def cleanup_terminal():
    """Cleanup function to restore terminal attributes"""
    global _original_tty_attrs, _cleanup_done
    
    # Prevent double cleanup
    if _cleanup_done:
        return
    _cleanup_done = True
    
    if _original_tty_attrs is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, _original_tty_attrs)
            # Can't use log_message here as logging might not be available
            _original_tty_attrs = None
        except Exception:
            # Silently ignore errors during cleanup
            pass
    
    # Restore original stderr using the centralized function
    try:
        restore_stderr()
    except Exception:
        pass
    
    # Also ensure cursor is visible
    try:
        with terminal.terminal_write_lock:
            sys.stdout.write('\033[?25h')  # Show cursor
            sys.stdout.flush()
    except Exception:
        pass
    
    # Stop communication manager if not already done
    global comm_manager
    if comm_manager:
        try:
            comm_manager.stop()
            comm_manager = None
        except Exception:
            pass

# Register cleanup function to run at exit
atexit.register(cleanup_terminal)

def signal_handler(signum, frame=None):
    """Handle shutdown signals"""
    global current_proc
    try:
        log_message("INFO", f"[signals] Handler fired: signum={signum} pid={os.getpid()} pgrp={os.getpgrp()}")
    except Exception:
        pass
    log_message("INFO", f"Received signal {signum} - stopping TTS immediately")
    # For Ctrl-C, implement a fast exit path
    if signum == signal.SIGINT:
        # First priority: restore terminal for user
        if _original_tty_attrs:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, _original_tty_attrs)
            except Exception:
                pass

        # Ensure cursor is visible
        try:
            with terminal.terminal_write_lock:
                sys.stdout.write('\033[?25h')  # Show cursor
                sys.stdout.flush()
        except Exception:
            pass

        if current_proc:
            try:
                # Forcefully kill Ollama’s process group
                os.killpg(os.getpgid(current_proc.pid), signal.SIGKILL)
            except Exception as e:
                log_message("ERROR", f"Failed to kill child: {e}")

        # Stop TTS immediately
        try:
            stop_tts_immediately()
        except Exception as e:
            log_message("ERROR", f"Failed to stop TTS immediately: {e}")

        # Quick ASR cleanup to prevent model destructor errors
        try:
            asr.stop_dictation()  # This will call _cleanup_whisper_model()
        except Exception:
            pass

        try:
            # Send SIGINT to the child's process group
            os.killpg(os.getpgid(current_proc.pid), signal.SIGINT)
            log_message("INFO", f"Forwarded SIGINT to child pid {current_proc.pid}")
            return
        except Exception as e:
            log_message("ERROR", f"Failed to forward SIGINT: {e}")

        # Quick terminal cleanup
        cleanup_terminal()

        # Exit immediately with SIGINT code
        os._exit(130)  # type: ignore[attr-defined]
    
    # For other signals, do a more graceful shutdown
    # Restore terminal attributes first
    if _original_tty_attrs:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, _original_tty_attrs)
            log_message("INFO", "Restored terminal attributes in signal handler")
        except Exception as e:
            log_message("ERROR", f"Failed to restore terminal attributes: {e}")
    
    # Ensure cursor is visible
    try:
        with terminal.terminal_write_lock:
            sys.stdout.write('\033[?25h')  # Show cursor
            sys.stdout.flush()
    except Exception:
        pass
    
    # First ensure we clean up the terminal - this is critical
    cleanup_terminal()
    
    # Then stop services gracefully
    try:
        asr.stop_dictation()
    except Exception:
        pass
    
    try:
        tts.shutdown_tts()
    except Exception:
        pass
    
    # Stop communication manager (includes zrok cleanup)
    global comm_manager
    if comm_manager:
        try:
            comm_manager.stop()
            log_message("INFO", "Communication manager stopped")
        except Exception as e:
            log_message("ERROR", f"Failed to stop comm_manager: {e}")
    
    # Exit with appropriate code using os._exit to avoid segfaults
    exit_code = 130 if signum == signal.SIGINT else 0
    os._exit(exit_code)  # type: ignore[attr-defined]


async def replay_recorded_session(args, command_name: str = None) -> Union[int, List[Tuple[float, str, int]]]:
    """Replay a recorded session file through the TTS pipeline for debugging"""
    global active_profile, terminal, asr_state, verbosity_level, comm_manager

    record_file = args.record
    capture_tts = args.capture_tts_output is not None
    disable_tts = args.disable_tts
    show_output = not args.no_output
    command_name = command_name
    verbosity = args.verbose
    log_file = args.log_file

    comms_config = build_comms_config(args)

    core = TalkitoCore(
        verbosity_level=args.verbosity,
        log_file_path=args.log_file
    )

    # Set up profile if specified
    profile = get_profile(args.profile)
    if profile:
        core.active_profile = profile

    # Ensure we have at least a default profile
    if core.active_profile is None:
        core.active_profile = get_profile('default')

    log_message("DEBUG", "Set up TTS engine")
    # Set up TTS engine
    # Check shared state first - early initialization may have already selected a provider
    shared_state = get_shared_state()

    # Use provider from shared state if it was set during early initialization
    if shared_state.tts_provider and shared_state.tts_provider != args.tts_provider:
        log_message("INFO", f"Using TTS provider from early initialization: {shared_state.tts_provider} (overriding args: {args.tts_provider})")
        args.tts_provider = shared_state.tts_provider
    
    no_tts_provider = not args.tts_provider
    if no_tts_provider:
        log_message("WARNING", "No TTS provider is configured. Keeping TTS disabled.")
        tts.disable_tts_completely("no TTS provider configured", args)
        engine = "cloud"
    elif args and args.tts_provider != 'system':
        if not tts.configure_tts_from_args(args):
            print("Error: Failed to configure TTS provider", file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError("Failed to configure TTS provider")
        engine = "cloud"
    else:
        engine = TTS_ENGINE
        if engine == "auto":
            engine = tts.detect_tts_engine()
            if engine == "none":
                print("Error: No TTS engine found. Please install espeak, festival, flite (Linux) or use macOS", file=sys.stderr)
                sys.stderr.flush()
                raise RuntimeError("No TTS engine found. Please install espeak, festival, flite (Linux) or use macOS")

    # Start TTS worker
    log_message("DEBUG", "Start TTS worker")
    auto_skip_tts = not args.dont_auto_skip_tts
    tts.start_tts_worker(engine, auto_skip_tts)

    # Start background update checker
    from .update import start_background_update_checker
    start_background_update_checker()

    # Update shared state for TTS initialization
    shared_state = get_shared_state()

    # Get the actual provider that was configured
    if args.tts_provider:
        tts_provider = args.tts_provider
    else:
        # For system/auto, get the actual engine being used
        tts_provider = engine if engine != 'cloud' else getattr(tts, 'tts_provider', 'system')

    log_message("DEBUG", "set_tts_initialized")
    shared_state.set_tts_initialized(True, tts_provider)

    # Initialize terminal state if not already done
    if terminal is None:
        terminal = TerminalState()
    
    # Initialize ASR state if not already done
    if asr_state is None:
        asr_state = ASRState()
    
    # Set verbosity level
    verbosity_level = verbosity
    
    setup_logging(log_file)
    log_message("INFO", f"Replaying recorded session: {args.replay} with verbosity level {verbosity}")
    
    # Set up communications if configured - this was missing from replay!
    if comms_config:
        log_message("INFO", "Setting up communication manager for replay session")
        try:
            # Extract enabled providers from config
            providers = []
            if hasattr(comms_config, 'sms_enabled') and comms_config.sms_enabled:
                providers.append('sms')
            if hasattr(comms_config, 'whatsapp_enabled') and comms_config.whatsapp_enabled:
                providers.append('whatsapp')
            if hasattr(comms_config, 'slack_enabled') and comms_config.slack_enabled:
                providers.append('slack')

            if providers:
                comm_manager = comms.setup_communication(providers=providers, config=comms_config)
                log_message("INFO", f"Communication manager initialized with providers: {providers}")
                
                if comm_manager:
                    has_whatsapp = any(isinstance(p, comms.TwilioWhatsAppProvider) for p in comm_manager.providers)
                    has_slack = any(isinstance(p, comms.SlackProvider) for p in comm_manager.providers)

                    sync_communication_state_from_config(
                        comms_config,
                        slack_configured=has_slack,
                        whatsapp_configured=has_whatsapp,
                    )

                    if has_slack and hasattr(comms_config, 'slack_channel') and comms_config.slack_channel:
                        shared_state = get_shared_state()
                        shared_state.set_slack_mode(True)
                        log_message("INFO", f"Slack mode activated for channel: {comms_config.slack_channel}")
            else:
                log_message("INFO", "No communication providers enabled in config")
        except Exception as e:
            log_message("ERROR", f"Failed to set up communication manager: {e}")
            # Don't fail the replay if communication setup fails
    else:
        log_message("DEBUG", "No communication config provided for replay")
    
    # Set the disable_tts flag in tts module if requested
    if disable_tts:
        tts.disable_tts = True
        log_message("INFO", "TTS disabled by disable_tts parameter")
    
    recorded_data = []
    record_start_time = time.time() if record_file else None
    is_record_mode = record_file is not None
    
    if is_record_mode:
        log_message("INFO", f"Capture mode enabled - saving replay output to: {record_file}")
    
    # Set up profile based on command name if provided
    global active_profile
    if command_name:
        active_profile = get_profile(command_name)
        if active_profile.name != 'default':
            log_message("INFO", f"Using {command_name} profile for replay")
        else:
            log_message("INFO", f"Profile '{command_name}' not found, using default profile")
    else:
        active_profile = get_profile('default')
        log_message("INFO", "No profile set for replay, using default")
    
    # Configure TTS provider
    # engine = configure_tts_engine(tts_config, auto_skip_tts)
    # if not engine:
    #     return 1

    # Parse recorded session
    entries = SessionRecorder.parse_file(args.replay)
    if not entries:
        print(f"Error: No valid entries found in replay file: {args.replay}", file=sys.stderr)
        return 1
    
    buffer = []
    prev_line = ""
    output_buffer = LineBuffer()  # Create line buffer for tracking changes
    line_buffer = b""  # Persistent line buffer like in run_command
    skip_duplicate_mode = False
    consecutive_redraws = 0
    last_redraw_time = 0
    in_alternate_buffer = False
    buffer_switch_time = 0
    
    # Cursor tracking
    current_cursor_row = 1
    get_terminal_size()  # Initialize terminal dimensions

    print(f"Replaying {len(entries)} session entries...")
    
    for entry in entries:
        if entry['event_type'] == 'OUTPUT':
            # Use raw bytes for everything
            raw_bytes = entry['raw_bytes']
            
            # Capture the output if in record mode
            if is_record_mode:
                timestamp = time.time() - record_start_time
                recorded_data.append((timestamp, 'OUTPUT', raw_bytes))

            if show_output:
                # Write raw bytes to stdout so user can see the replay
                sys.stdout.buffer.write(raw_bytes)
                sys.stdout.flush()
            
            # Process output data like run_command does
            if handle_alternate_screen_buffer(raw_bytes):
                in_alternate_buffer = not in_alternate_buffer
                current_cursor_row = clear_terminal_state(output_buffer, 1)

            # Parse cursor movements
            cursor_delta, did_scroll = parse_cursor_movements(raw_bytes)
            
            if detect_screen_reset(raw_bytes):
                consecutive_redraws += 1
                current_time = time.time()
                
                if consecutive_redraws > 2:
                    skip_duplicate_mode = True
                    current_cursor_row = clear_terminal_state(output_buffer, 1)
                
                if current_time - last_redraw_time > 5.0:
                    consecutive_redraws = 0
                    skip_duplicate_mode = False
                
                last_redraw_time = current_time
            
            # Update cursor position tracking
            current_cursor_row = update_cursor_position(current_cursor_row, cursor_delta, did_scroll)
            
            if time.time() - buffer_switch_time < 0.5:
                continue
                
            if in_alternate_buffer:
                continue
            
            # Add to line buffer and process complete lines
            line_buffer += raw_bytes
            line_buffer, buffer, prev_line, current_cursor_row, detected_prompt = process_line_buffer_data(
                line_buffer, output_buffer, buffer, prev_line,
                skip_duplicate_mode, current_cursor_row, "off"
            )
        
        elif entry['event_type'] == 'INPUT':
            # Use raw bytes
            raw_bytes = entry['raw_bytes']
            
            # Capture the input if in record mode
            if is_record_mode:
                timestamp = time.time() - record_start_time
                recorded_data.append((timestamp, 'INPUT', raw_bytes))
            

    # Process any remaining line buffer content
    if line_buffer:
        line = line_buffer.decode('utf-8', errors='ignore')
        line_idx, action = output_buffer.add_or_update_line(line)
        
        if action in ['added', 'modified']:
            buffer, prev_line, _ = process_line(
                line, buffer, prev_line, skip_duplicate_mode, line_idx, "off"
            )
    
    # Process any remaining buffer
    if buffer:
        last_idx = output_buffer.next_index - 1 if output_buffer.next_index > 0 else 0
        process_remaining_buffer(buffer, last_idx)
    
    log_message("DEBUG", f"Replay completed. Total lines in buffer: {output_buffer.get_line_count()}")
    
    # Write recorded data to file if in record mode
    if is_record_mode and recorded_data:
        try:
            with open(record_file, 'wb') as f:
                for timestamp, data_type, data in recorded_data:
                    # Write entry header: timestamp EVENT_TYPE data_length data
                    f.write(f"{timestamp:.6f} {data_type} {len(data):04x} ".encode('utf-8'))
                    # Write raw data
                    f.write(data)
                    # Write separator
                    f.write(b"\n")
            log_message("INFO", f"Captured {len(recorded_data)} replay events to {record_file}")
            print(f"Captured {len(recorded_data)} replay events to {record_file}")
        except Exception as e:
            log_message("ERROR", f"Failed to write record file {record_file}: {e}")
            print(f"Error writing record file: {e}")
    
    if capture_tts:
        # Wait for all TTS to be processed and captured
        tts.wait_for_tts_to_finish(timeout=300)  # Wait up to 5 minutes
        # Don't shutdown TTS when capturing - we might want to run multiple captures
        return [(0.0, item.text, item.line_number) for item in tts.speech_history]
    else:
        # Wait for all TTS to finish playing
        print("\nWaiting for speech to complete...")
        tts.wait_for_tts_to_finish(timeout=300)  # Wait up to 5 minutes
        
        tts.shutdown_tts()
        return 0


# TalkitoCore class for library usage
class TalkitoCore:
    """Core talkito functionality as a reusable class"""
    
    def __init__(self, verbosity_level: int = 0, log_file_path: Optional[str] = None):
        self.verbosity_level = verbosity_level
        self.logger: Optional[logging.Logger] = None
        self.current_master_fd: Optional[int] = None
        self.current_proc: Optional[asyncio.subprocess.Process] = None
        self.comm_manager = None
        self.terminal = TerminalState()
        self.asr_state = ASRState()
        self.active_profile: Optional[Profile] = None  # Will be initialized to default profile
        
        # Set up logging
        self.setup_logging(log_file_path)
    
    def setup_logging(self, log_file_path: Optional[str] = None):
        """Set up logging configuration
        
        Args:
            log_file_path: Optional path to log file. If provided, enables logging.
        """
        # Use the imported setup_logging function
        setup_logging(log_file_path)
        self.logger = get_logger(__name__)  # Get logger using centralized function
    
    def log_message(self, level: str, message: str):
        """Log a message if logging is enabled"""
        log_message(level, message, __name__)  # Use centralized log_message with module name
    
    async def run_command(self, args: List[str], asr_mode: str = "auto-input", 
                         record_file: str = None) -> int:
        """Run a command with TTS and optional ASR support"""
        # Update globals that are used by helper functions
        global current_master_fd, current_proc, verbosity_level, active_profile
        global terminal, asr_state, comm_manager
        
        # Initialize terminal and asr_state if not already done
        if terminal is None:
            terminal = TerminalState()
        if asr_state is None:
            asr_state = ASRState()
            
        current_master_fd = self.current_master_fd
        current_proc = self.current_proc  
        verbosity_level = self.verbosity_level
        active_profile = self.active_profile
        terminal = self.terminal
        asr_state = self.asr_state
        # Only update comm_manager if not already set (e.g., from run_with_talkito)
        if comm_manager is None:
            comm_manager = self.comm_manager
        
        # Call the existing run_command function
        result = await run_command(args, asr_mode, record_file)
        
        # Update instance variables
        self.current_master_fd = current_master_fd
        self.current_proc = current_proc
        
        return result


def get_comms_config_from_args(args) -> comms.CommsConfig:
    """Return a CommsConfig snapshot that reflects env vars plus CLI overrides."""
    config = comms.create_config_from_env()

    def _split_list(value: Optional[str]) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]

    if getattr(args, "sms_recipients", None):
        config.sms_recipients = _split_list(args.sms_recipients)
    if getattr(args, "whatsapp_recipients", None):
        config.whatsapp_recipients = _split_list(args.whatsapp_recipients)
    if getattr(args, "slack_channel", None):
        config.slack_channel = args.slack_channel
    if getattr(args, "webhook_port", None):
        config.webhook_port = args.webhook_port

    return config


def build_comms_config(args) -> Optional[comms.CommsConfig]:
    """Build communication configuration from command line arguments"""
    config = get_comms_config_from_args(args)

    # Check if any communication is configured (after applying overrides)
    has_sms = config.twilio_account_sid and config.sms_recipients
    has_whatsapp = config.twilio_whatsapp_number and config.whatsapp_recipients
    has_slack = config.slack_bot_token and config.slack_app_token and config.slack_channel

    if not any([has_sms, has_whatsapp, has_slack]):
        # No communication configured
        return None

    # Auto-detect based on configuration
    config.sms_enabled = has_sms
    config.whatsapp_enabled = has_whatsapp
    config.slack_enabled = has_slack

    # Apply user preference from environment if set
    preferred = os.environ.get('TALKITO_PREFERRED_COMMS_PROVIDERS', 'auto')
    if preferred and preferred != 'auto':
        if preferred == 'none':
            # Disable all providers
            config.slack_enabled = False
            config.whatsapp_enabled = False
            config.sms_enabled = False
        elif preferred == 'slack':
            # Only enable slack
            config.slack_enabled = has_slack
            config.whatsapp_enabled = False
            config.sms_enabled = False
        elif preferred == 'whatsapp':
            # Only enable whatsapp
            config.slack_enabled = False
            config.whatsapp_enabled = has_whatsapp
            config.sms_enabled = False
        elif preferred == 'both':
            # Enable both slack and whatsapp
            config.slack_enabled = has_slack
            config.whatsapp_enabled = has_whatsapp
            config.sms_enabled = False

    return config

# High-level API functions
async def run_with_talkito(command: List[str], args) -> int:
    """Run a command with talkito functionality
    
    Args:
        command: Command and arguments to run
        **kwargs: Additional options (verbosity, tts_config, asr_config, etc.)
    
    Returns:
        Exit code of the command
    """
    # Set up signal handlers (if not already installed)
    try:
        current_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGINT, current_sigint)  # Restore current handler
        if current_sigint == signal.SIG_DFL:
            log_message("DEBUG", "Installing signal handlers at the start of run_with_talkito")
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGWINCH, debounced_winch_handler)
    except Exception as e:
        log_message("ERROR", f"Failed to re-install signal handlers: {e}")

    # Handle TTS disable
    if args.disable_tts:
        tts.disable_tts = True
        get_shared_state().set_tts_enabled(False)

    comms_config = build_comms_config(args)
    
    core = TalkitoCore(
        verbosity_level=args.verbosity,
        log_file_path=args.log_file
    )
    
    # Set up profile if specified
    profile = get_profile(args.profile)
    if profile:
        core.active_profile = profile

    if not profile.supported:
        print(f"Sorry, TalkiTo does not currently work with {profile.name}.")
        if profile.warning:
            print("\n"+profile.warning+"\n")
        print("Please get in touch with me if you want to help resolve this!")
        exit(0)
    elif profile.warning:
        print(profile.warning)
    
    # Ensure we have at least a default profile
    if core.active_profile is None:
        core.active_profile = get_profile('default')

    log_message("DEBUG", "Set up TTS engine")
    # Set up TTS engine
    # Check shared state first - early initialization may have already selected a provider
    shared_state = get_shared_state()

    # Use provider from shared state if it was set during early initialization
    if shared_state.tts_provider and shared_state.tts_provider != args.tts_provider:
        log_message("INFO", f"Using TTS provider from early initialization: {shared_state.tts_provider} (overriding args: {args.tts_provider})")
        args.tts_provider = shared_state.tts_provider
    
    no_tts_provider = not args.tts_provider
    if no_tts_provider:
        log_message("WARNING", "No TTS provider is configured. Keeping TTS disabled.")
        tts.disable_tts_completely("no TTS provider configured", args)
        engine = "cloud"
    elif args and args.tts_provider != 'system':
        if not tts.configure_tts_from_args(args):
            print("Error: Failed to configure TTS provider", file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError("Failed to configure TTS provider")
        engine = "cloud"
    else:
        engine = TTS_ENGINE
        if engine == "auto":
            engine = tts.detect_tts_engine()
            if engine == "none":
                print("Error: No TTS engine found. Please install espeak, festival, flite (Linux) or use macOS", file=sys.stderr)
                sys.stderr.flush()
                raise RuntimeError("No TTS engine found. Please install espeak, festival, flite (Linux) or use macOS")
    
    # Start TTS worker
    log_message("DEBUG", "Start TTS worker")
    auto_skip_tts = not args.dont_auto_skip_tts
    tts.start_tts_worker(engine, auto_skip_tts)
    
    # Start background update checker
    from .update import start_background_update_checker
    start_background_update_checker()
    
    # Update shared state for TTS initialization
    shared_state = get_shared_state()
    
    # Get the actual provider that was configured
    if args.tts_provider:
        tts_provider = args.tts_provider
    else:
        # For system/auto, get the actual engine being used
        tts_provider = engine if engine != 'cloud' else getattr(tts, 'tts_provider', 'system')

    asr.configure_asr_from_args(args)

    log_message("DEBUG", "set_tts_initialized")
    shared_state.set_tts_initialized(True, tts_provider)
    
    if shared_state.asr_mode != 'off':
        # Enable ASR in shared state - centralized functions will handle initialization
        log_message("INFO", f"Enabling ASR for {shared_state.asr_mode} mode")
        shared_state.set_asr_enabled(True)
    else:
        # Explicitly disable ASR when mode is 'off'
        log_message("INFO", "Disabling ASR due to --asr-mode off")
        shared_state.set_asr_enabled(False)
    
    # Set up communications if configured
    if comms_config:
        global comm_manager
        # Extract enabled providers from config
        providers = []
        if hasattr(comms_config, 'sms_enabled') and comms_config.sms_enabled:
            providers.append('sms')
        if hasattr(comms_config, 'whatsapp_enabled') and comms_config.whatsapp_enabled:
            providers.append('whatsapp')
        if hasattr(comms_config, 'slack_enabled') and comms_config.slack_enabled:
            providers.append('slack')

        # Only setup communication if there are enabled providers
        if providers:
            comm_manager = comms.setup_communication(providers=providers, config=comms_config)
        else:
            comm_manager = None
        core.comm_manager = comm_manager
        
        # Update shared state with configured providers
        if comm_manager:
            has_whatsapp = any(isinstance(p, comms.TwilioWhatsAppProvider) for p in comm_manager.providers)
            has_slack = any(isinstance(p, comms.SlackProvider) for p in comm_manager.providers)

            sync_communication_state_from_config(
                comms_config,
                slack_configured=has_slack,
                whatsapp_configured=has_whatsapp,
            )

            # Save the state
            from .state import save_shared_state
            save_shared_state()

    log_message("DEBUG", "Running command")
    try:
        # Run the command
        return await core.run_command(
            command,
            asr_mode=shared_state.asr_mode,
            record_file=args.record,
        )
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT)
        return 130  # Standard exit code for SIGINT


def wrap_command(command: List[str], **kwargs):
    """Synchronous wrapper for run_with_talkito"""
    import asyncio
    try:
        return asyncio.run(run_with_talkito(command, **kwargs))
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT)
        return 130  # Standard exit code for SIGINT


def parse_test_arguments():
    """Parse command-line arguments for process_line testing"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Test talkito process_line function',
        usage='python -m talkito.core [options] <line_to_test>'
    )
    parser.add_argument('line', nargs='+',
                       help='Line of text to test with process_line')
    parser.add_argument('--log-file', type=str, default='~/.talkito.log',
                       help='Log file path (default: ~/.talkito.log)')
    parser.add_argument('--profile', '-p', type=str, default='default',
                       help='Profile to use for testing (default: default)')
    parser.add_argument('--verbosity', '-v', type=int, default=2,
                       help='Verbosity level (0-2, default: 2)')
    parser.add_argument('--line-number', type=int, default=1,
                       help='Line number to use for testing (default: 1)')
    parser.add_argument('--asr-mode', type=str, default='auto-input',
                       choices=['auto-input', 'tap-to-talk', 'off'],
                       help='ASR mode for testing (default: auto-input)')
    return parser.parse_args()


def test_process_line():
    """Main entry point for process_line testing"""
    args = parse_test_arguments()

    # Initialize minimal state required for process_line
    global terminal, active_profile, asr_state, verbosity_level

    # Initialize terminal state
    if terminal is None:
        terminal = TerminalState()

    # Initialize ASR state
    if asr_state is None:
        asr_state = ASRState()

    # Set up profile
    active_profile = get_profile(args.profile)

    # Set verbosity level
    verbosity_level = args.verbosity

    # Set up logging
    import os
    log_path = os.path.expanduser(args.log_file)
    setup_logging(log_path)

    # Get the test line
    test_line = ' '.join(args.line)

    print(f"Testing line: {test_line}")
    print(f"Profile: {args.profile}, Verbosity: {verbosity_level}")
    print("-" * 80)

    # Call process_line with empty initial state
    buffer = []
    prev_line = ""

    new_buffer, new_prev_line, detected_prompt = process_line(
        test_line,
        buffer,
        prev_line,
        skip_duplicates=False,
        line_number=args.line_number,
        asr_mode=args.asr_mode
    )

    print("\nResult:")
    print(f"  Buffer: \n{new_buffer}\n")
    print(f"  Prev line: \n{new_prev_line}\n")
    print(f"  Detected prompt: \n{detected_prompt}\n")
    print(f"\nCheck {log_path} for detailed processing logs")


if __name__ == '__main__':
    test_process_line()
