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
Core functionality for talkito - terminal interaction and processing
"""

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
import tty
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional, List, Tuple, Dict, Union, Deque, Any

from . import tts
from .profiles import get_profile, Profile
from .logs import setup_logging, get_logger, log_message, restore_stderr, log_debug, is_logging_enabled
from .state import get_shared_state

try:
    from . import asr
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False

try:
    from . import comms
    COMMS_AVAILABLE = True
except ImportError:
    COMMS_AVAILABLE = False

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

# ANSI Key codes
RETURN = b'\x1b[B\r'
ESC = b'\x1b'
KEY_UP = b'\x1b[A'
KEY_DOWN = b'\x1b[B'
KEY_LEFT = b'\x1b[D'
KEY_RIGHT = b'\x1b[C'
KEY_SPACE = b' '
# Tap-to-talk keys - multiple options for flexibility
KEY_TAP_TO_TALK_SEQUENCES = [
    b'\x1b',  # Escape (Right Option/Alt when pressed alone)
    b'\xc2\xa7',  # § (section sign - common on Mac keyboards)
    b'\x1b[2~',  # Insert key (Linux/Windows)
    b'`',  # Backtick key as fallback
]

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

# Common regex patterns
FILE_PATH_PATTERN = r'^[\w/\-_.]+\.(py|js|txt|md|json|yaml|yml|sh|c|cpp|h|java|go|rs|rb|php)$'
NUMBERS_ONLY_PATTERN = r'^\s*\d+(\s+\d+)*\s*$'
BOX_CONTENT_PATTERN = r'│\s*([^│]+)\s*│'
BOX_SEPARATOR_PATTERN = r'^[─═╌╍]+$'
PROMPT_PATTERN = r'^\s*[>\$#]\s*$'
SENTENCE_END_PATTERN = r'[.!?]$'

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

# Global state - these are updated by TalkitoCore
current_master_fd: Optional[int] = None
current_proc: Optional[asyncio.subprocess.Process] = None
verbosity_level: int = 0
comm_manager: Optional[Any] = None  # Will be CommunicationManager when initialized

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
    pending_speech_text: str = ""
    pending_text_line_number: int = 0
    last_line_number: int = -1

@dataclass 
class ASRState:
    """Groups ASR-related state"""
    auto_listen_enabled: bool = True
    waiting_for_input: bool = False
    asr_auto_started: bool = False
    tap_to_talk_active: bool = False
    tap_to_talk_last_press: float = 0
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
        self.recent_lines: deque = deque(maxlen=RECENT_LINES_CACHE_SIZE)  # (index, content) pairs

    def _get_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings"""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    def _find_similar_line(self, content: str) -> Optional[int]:
        """Find a similar line in recent history"""
        for idx, line_content in self.recent_lines:
            if idx in self.lines:  # Still exists
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
                # Update in recent lines
                self.recent_lines.append((similar_idx, raw_line))
                return similar_idx, 'modified'

        # New line
        idx = self.next_index
        self.next_index += 1
        self.lines[idx] = raw_line
        self.recent_lines.append((idx, raw_line))
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


# Override the imported setup_logging to add module-specific setup
def setup_logging(log_file_path: Optional[str] = None):
    """Set up logging and pass logger to modules that need it
    
    Args:
        log_file_path: Optional path to log file. If provided, enables logging.
    """
    # Import the base setup_logging function with a different name to avoid recursion
    from .logs import setup_logging as base_setup_logging
    
    # Use centralized logging setup
    base_setup_logging(log_file_path)

def send_to_comms(text: str):
    log_message("DEBUG", "send_to_comms")
    """Send text to communication channels if configured"""
    global comm_manager
    if comm_manager:
        if text.strip():
            log_message("DEBUG", f"[COMMS] Sending to comms: {text}...")
            try:
                # send_output already checks shared state for channels internally
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

def send_pending_text():
    if terminal.pending_speech_text.strip():
        log_message("DEBUG", f"send_pending_text [{terminal.pending_speech_text}]")
        
        # Check if TTS is enabled before queueing
        shared_state = get_shared_state()
        if shared_state.tts_enabled:
            speakable_text = tts.queue_for_speech(terminal.pending_speech_text, terminal.pending_text_line_number)
            if speakable_text:
                log_message("DEBUG", f"All checks passed for [{speakable_text}]. Set previous_line_was_queued to True and send to comms")
                terminal.pending_speech_text = ""
                send_to_comms(speakable_text)
        else:
            log_message("DEBUG", "TTS disabled, not sending pending text to speech")
            terminal.pending_speech_text = ""
            # Still send to comms even if TTS is disabled
            send_to_comms(terminal.pending_speech_text)

def queue_output(text: str, line_number: Optional[int] = None):
    """Queue text for both TTS and communication channels"""
    log_message("DEBUG", f"queue_output [{text}] {line_number}")
    if line_number < terminal.last_line_number:
        log_message("DEBUG", f"queue_output skipping previously seen line number")
    elif text and text.strip():
        # Check if TTS is enabled before accumulating text
        shared_state = get_shared_state()
        if not shared_state.tts_enabled:
            log_message("DEBUG", "TTS disabled, not accumulating text for speech")
            # Still send to comms even if TTS is disabled
            send_to_comms(text)
            return
            
        # Queue for TTS (TTS worker will handle ASR pausing and cleaning of text)
        append = terminal.previous_line_was_queued and not terminal.previous_line_was_queued_space_seperated
        terminal.last_line_number = line_number
        if append:
            log_message("DEBUG", f"queue_output appending")
            terminal.pending_speech_text = terminal.pending_speech_text + text
        else:
            if terminal.pending_speech_text:
                send_pending_text()
            terminal.previous_line_was_skipped = False
            terminal.previous_line_was_queued = True
            terminal.pending_speech_text = text
            terminal.pending_text_line_number = line_number

def clean_text(text: str) -> str:
    """Strip ANSI escape codes and terminal control sequences"""
    # First, remove all ANSI escape sequences
    text = ANSI_PATTERN.sub('', text)
    text = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', text)
    text = text.replace('\x1B', '')
    
    # Remove various control characters
    text = text.replace('\x08', '')  # Backspace
    text = text.replace('\x0D', '')  # Carriage return
    
    # Remove standalone 'm' characters that are likely orphaned from ANSI codes
    # This handles cases where 'm' appears after whitespace or at start of string
    # but NOT when it's part of a word (like 'am', 'pm', 'them', etc.)
    # Also preserve contractions like "I'm", "don't", etc.
    text = re.sub(r'(?<![a-zA-Z0-9\'])m(?![a-zA-Z])', '', text)
    
    # Also remove patterns like "2m" or "22m" that might be left from ANSI codes
    text = re.sub(r'(?<![a-zA-Z])\d{1,3}m\b', '', text)
    
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
    box_match = re.match(BOX_CONTENT_PATTERN, line)
    if box_match:
        box_content = box_match.group(1).strip()
        if (box_content and
                not re.match(BOX_SEPARATOR_PATTERN, box_content) and
                not re.match(PROMPT_PATTERN, box_content)):
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
        return (any(re.search(p, line) for p in prompt_patterns if p) or line.strip() == '>')
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


def modify_prompt_for_asr(data: bytes, input_prompt, input_mic_replace) -> bytes:
    """Modify prompt output to show microphone emoji when ASR is active"""
    try:
        text = data.decode('utf-8', errors='ignore')
        if input_prompt in text:
            # When we find the prompt, mark that we've seen it
            asr_state.prompt_detected = True
            text = text.replace(input_prompt, input_mic_replace)
            return text.encode('utf-8')
        log_message("WARNING", f"input prompt {input_prompt} not found in text {text}")
        return data
    except:
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
    if re.match(r'^[\w/\-_.]+\.(py|js|txt|md|json|yaml|yml|sh|c|cpp|h|java|go|rs|rb|php)$', line):
        log_message("FILTER", f"Skipped file path: '{line}'")
        return True

    # Skip lines that are just numbers (single or multiple), catches cases like "599" or "599 603 1117"
    if re.match(r'^\s*\d+(\s+\d+)*\s*$', line):
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
    except:
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


def process_line(line: str, buffer: List[str], prev_line: str,
                 skip_duplicates: bool = False, line_number: Optional[int] = None, asr_mode: str = "auto-input") -> Tuple[
    Optional[str], List[str], str, bool]:
    """Process a single line and return (text_to_speak, new_buffer, new_prev_line, detected_prompt)"""

    # Check if this is a question line - these should ALWAYS be spoken
    if active_profile and active_profile.is_question_line(line):
        log_message("INFO", f"Detected question line: '{line[:MAX_LINE_PREVIEW]}...'")
        asr_state.question_mode = True
        cleaned_line = clean_text(line)
        queue_output(strip_profile_symbols(cleaned_line), line_number)
        terminal.previous_line_was_skipped = True
        return None, buffer, line, False

    cleaned_line = clean_text(line)

    # Check if this is a continuation line
    if active_profile and active_profile.is_continuation_line(cleaned_line) and (terminal.previous_line_was_skipped or terminal.previous_line_was_queued):
        log_message("DEBUG", f"Detected continuation line: '{line[:MAX_LINE_PREVIEW]}...'")
        if terminal.previous_line_was_skipped:
            log_message("FILTER", f"Skipping continuation line (previous was skipped): '{line[:MAX_LINE_PREVIEW]}...'")
            return None, buffer, line, False
        elif terminal.previous_line_was_queued:
            if not cleaned_line.strip():
                send_pending_text()
            log_message("INFO", f"Processing continuation line (previous was queued): '{line[:MAX_LINE_PREVIEW]}...'")
            return cleaned_line, [], line, False
    elif cleaned_line.strip(): # Reset on non empty lines
        log_message("DEBUG", f"{cleaned_line=} reset previous_line_was_queued and previous_line_was_skipped to false")
        send_pending_text()
        terminal.previous_line_was_queued = False
        terminal.previous_line_was_skipped = False
        terminal.previous_line_was_queued_space_seperated = False

    if active_profile and active_profile.should_skip_raw(line):
        log_message("WARNING", f"Skipped by raw pattern: '{line[:MAX_LINE_PREVIEW]}...'")
        terminal.previous_line_was_skipped = True
        send_pending_text()
        return None, buffer, line, False
    elif active_profile and active_profile.raw_skip_patterns:
        for pattern in active_profile.raw_skip_patterns:
            if re.search(pattern, line):
                log_message("WARNING", f"Skipped by raw pattern: '{line[:MAX_LINE_PREVIEW]}...'")
                terminal.previous_line_was_skipped = True
                send_pending_text()
                return None, buffer, line, False

    # Detect prompts
    is_prompt = (active_profile and active_profile.input_start and active_profile.input_start in line) or is_prompt_line(cleaned_line)
    
    # Log prompt detection for debugging
    if ">" in cleaned_line and "│" in cleaned_line:
        log_message("DEBUG", f"Checking prompt detection for line: '{cleaned_line[:50]}...' - is_prompt={is_prompt}")

    if is_response_line(cleaned_line):
        asr_state.waiting_for_input = False
        log_message("DEBUG", f"Set asr_state.waiting_for_input=False in process_line (response detected)")
        check_and_enable_auto_listen(asr_mode)
        
        # Response lines should be spoken unless filtered by profile
        # Use the profile's should_skip method which properly handles verbosity and exceptions
        if active_profile and active_profile.should_skip(cleaned_line, verbosity_level):
            log_message("FILTER", f"Skipped response by profile (verbosity={verbosity_level}): '{cleaned_line}'")
            terminal.previous_line_was_skipped = True
            send_pending_text()
            return None, buffer, line, False
        
        log_message("INFO", f"Queueing for speech response line: '{cleaned_line}'")
        queue_output(strip_profile_symbols(cleaned_line), line_number)
        return None, buffer, line, False

    # Move prompt detection here, before we process the line
    if is_prompt:
        log_message("INFO", f"Detected prompt in line ({line_number}): '{cleaned_line}'")
        terminal.previous_line_was_skipped = True
        send_pending_text()
        return None, buffer, line, True

    if cleaned_line:
        # Check if extracted text should be filtered based on verbosity
        if active_profile and active_profile.should_skip(cleaned_line, verbosity_level):
            log_message("FILTER", f"Skipped extracted text by profile (verbosity={verbosity_level}): '{cleaned_line}'")
            terminal.previous_line_was_skipped = True
            send_pending_text()
            return None, buffer, line, False
        queue_output(strip_profile_symbols(cleaned_line), line_number)
        return None, buffer, line, False

    if skip_duplicates and cleaned_line:
        if is_duplicate_screen_content(cleaned_line):
            log_message("WARNING", f"Skipping duplicate content: '{cleaned_line[:MAX_LINE_PREVIEW]}...'")
            terminal.previous_line_was_skipped = True
            send_pending_text()
            return None, buffer, line, False

    box_content = extract_box_content(cleaned_line)
    if box_content and not is_prompt:
        cleaned_line = box_content

    if should_skip_line(cleaned_line):
        terminal.previous_line_was_skipped = True
        send_pending_text()
        return None, buffer, prev_line, False

    # if re.match(r'^\s+\S', line) and prev_line:
    #     log_message("WARNING", f"Processing continuation line")
    #     continuation_text = clean_text(line).lstrip()
    #     if continuation_text.strip():
    #         buffer.append(continuation_text)
    #     # Don't update state here - buffer append doesn't determine final state
    #     return None, buffer, line, False

    filtered_buffer = [line for line in buffer if line and line.strip()]
    text_to_speak = '. '.join(filtered_buffer) if filtered_buffer else None

    # Debug suspicious short text
    if text_to_speak and len(text_to_speak) <= 5:
        log_message("WARNING", f"Short text_to_speak='{text_to_speak}' from buffer={filtered_buffer}, "
                              f"cleaned_line='{cleaned_line}', raw_line='{line[:50]}...'")

    buffer = []

    new_buffer = [cleaned_line] if cleaned_line and cleaned_line.strip() else []

    if cleaned_line and re.search(SENTENCE_END_PATTERN, cleaned_line):
        if text_to_speak:
            log_message("INFO", f"Queueing buffered text before sentence end")
            queue_output(strip_profile_symbols(text_to_speak), line_number)
            return cleaned_line, [], line, False
        return cleaned_line, [], line, False
    
    return text_to_speak, new_buffer, line, False


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
                log_message("DEBUG", f"Processing line {idx} before clear: '{line[:80]}...'")
                # Queue the line for speech before we lose it
                cleaned = clean_text(line)
                if cleaned and not active_profile.should_skip(cleaned, verbosity_level):
                    queue_output(strip_profile_symbols(cleaned), idx)
    
    output_buffer.clear()
    terminal.line_screen_positions.clear()
    log_message("INFO", "Screen reset detected - cleared buffers after processing")
    return cursor_row


def handle_alternate_screen_buffer(data: bytes) -> bool:
    """Detect and handle alternate screen buffer switches"""
    for seq in ALT_SCREEN_SEQUENCES:
        if seq in data:
            log_message("INFO", f"Alternate screen buffer switch detected")
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


async def setup_terminal_for_command(cmd: List[str]) -> Tuple[int, int, int, asyncio.subprocess.Process]:
    """Set up PTY and spawn subprocess for command execution"""
    master_fd, slave_fd, stdin_flags = setup_pty_with_scrollback()

    env = os.environ.copy()
    env['TERM'] = os.environ.get('TERM', 'xterm-256color')
    rows, cols = get_terminal_size()
    env['LINES'] = str(rows)
    env['COLUMNS'] = str(cols - 2 if cols > 2 else cols)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env
    )

    os.close(slave_fd)
    return master_fd, slave_fd, stdin_flags, proc


async def periodic_status_check(master_fd: int, asr_mode: str,
                               last_check_time: float) -> Tuple[bool, float]:
    """Handle periodic ASR status checks and auto-listen"""
    current_time = time.time()
    enabled_auto_listen = False

    if current_time - last_check_time > STATUS_CHECK_INTERVAL:
        log_message("DEBUG", f"Periodic status check: ASR_AVAILABLE={ASR_AVAILABLE}, "
                            f"asr_auto_started={asr_state.asr_auto_started}, asr_mode={asr_mode}")

        # Check shared state and stop ASR if it's been disabled
        shared_state = get_shared_state()
        if ASR_AVAILABLE and asr_state.asr_auto_started and not shared_state.asr_enabled:
            try:
                log_message("INFO", "ASR disabled in shared state, stopping dictation")
                asr.stop_dictation()
                asr_state.asr_auto_started = False
            except Exception as e:
                log_message("ERROR", f"Failed to stop ASR after shared state disable: {e}")

        if ASR_AVAILABLE and asr_state.asr_auto_started and asr_mode != "off":
            log_message("DEBUG", "Calling show_speech_status from periodic check")
            
            # Check if TTS is speaking and stop ASR if needed
            if asr.is_recognizing() and tts.is_speaking():
                try:
                    asr.stop_dictation()
                    asr_state.asr_auto_started = False
                    log_message("INFO", "Stopped ASR because TTS is speaking")
                except Exception as e:
                    log_message("ERROR", f"Failed to stop ASR during TTS: {e}")

        if asr_mode == "tap-to-talk":
            check_tap_to_talk_timeout()

        enabled_auto_listen = check_and_enable_auto_listen(asr_mode)
        if enabled_auto_listen:
            try:
                log_message("WARNING", "Send a non breaking space to trigger terminal activity")
                os.write(master_fd, SPACE_THEN_BACK)
            except:
                pass
        return enabled_auto_listen, current_time

    # Return the original last_check_time if we didn't perform the check
    return enabled_auto_listen, last_check_time


def configure_tts_engine(tts_config: dict, auto_skip_tts: bool) -> str:
    """Configure and validate TTS engine"""
    if tts_config and tts_config.get('provider') != 'system':
        if not tts.configure_tts_from_dict(tts_config):
            log_message("ERROR", "Failed to configure TTS provider")
            return None
        engine = "cloud"
    else:
        engine = TTS_ENGINE
        if engine == "auto":
            engine = tts.detect_tts_engine()
            if engine == "none":
                print("Error: No TTS engine found. Please install one of: espeak, festival, flite (Linux) or use macOS with 'say' command", file=sys.stderr)
                log_message("ERROR", "No TTS engine found")
                return None
            print(f"Using TTS engine: {engine}")
            log_message("INFO", f"Using TTS engine: {engine}")

    tts.start_tts_worker(engine, auto_skip_tts)
    
    # Update shared state
    from .state import get_shared_state
    shared_state = get_shared_state()
    
    # Get the actual provider that was configured
    if tts_config and tts_config.get('provider'):
        provider = tts_config.get('provider')
    else:
        # For system/auto, get the actual engine being used
        provider = engine if engine != 'cloud' else getattr(tts, 'tts_provider', 'system')
    
    shared_state.set_tts_initialized(True, provider)
    
    return engine


async def process_pty_output(data: bytes, output_buffer: LineBuffer,
                            line_buffer: bytes, text_buffer: List[str],
                            prev_line: str,
                            skip_duplicates: bool, cursor_row: int,
                            asr_mode: str, recorder: SessionRecorder) -> Tuple[bytes, List[str], str, int]:
    """Process output data from PTY and queue text for speech"""
    # Log that we received data
    if is_logging_enabled():
        log_message("DEBUG", f"[process_pty_output] Called with {len(data)} bytes of data")
    if b'\xe2\x8f\xba' in data:
        log_message("INFO", f"[process_pty_output] Data contains response marker!")
    
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
        except:
            pass

    cursor_delta, did_scroll = parse_cursor_movements(data)
    cursor_row = update_cursor_position(cursor_row, cursor_delta, did_scroll)

    # Always add data to buffer first
    line_buffer += data
    
    # Check if data contains clear sequences
    has_clear_sequences = detect_clear_sequences(data)
    
    # Also log when we detect clear sequences
    if b'\x1b[2K' in data:
        log_message("DEBUG", f"Data contains clear line sequences (\\x1b[2K found)")
    
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
        input_data = os.read(sys.stdin.fileno(), 4096)
        if input_data:
            # Check for tap-to-talk keys in tap-to-talk mode
            if asr_mode == "tap-to-talk":
                for tap_key in KEY_TAP_TO_TALK_SEQUENCES:
                    if tap_key in input_data:
                        asr_state.tap_to_talk_active = True
                        asr_state.tap_to_talk_last_press = time.time()
                        log_message("INFO", f"Tap-to-talk key pressed: {tap_key}")
                        # Don't forward tap-to-talk key to PTY
                        # Remove the tap key from input_data
                        input_data = input_data.replace(tap_key, b'')
                        if not input_data:
                            return
                        break

            # Check for TTS control keys
            # control_handled = handle_tts_controls(input_data)
            # if control_handled:
            #     return

            # Forward to PTY
            if input_data:  # Only forward if there's data left after filtering
                os.write(master_fd, input_data)

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
    # Remove ANSI escape sequences
    import re
    text_no_ansi = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

    # Remove zero-width characters
    text_no_zwc = text_no_ansi.replace('\u200d', '').replace('\u200c', '').replace('\u200b', '')

    # Count actual visual characters
    # Note: This is simplified and doesn't handle all Unicode cases perfectly,
    # but it should work for our use case
    return len(text_no_zwc)


def insert_partial_transcript(data_str, asr_state, active_profile):
    """Insert the partial transcript into the terminal output while preserving formatting. Returns modified data_str."""
    # Find the input start position
    start_idx = data_str.find(active_profile.input_start) if active_profile and active_profile.input_start else -1
    if start_idx != -1:
        start_idx += len(active_profile.input_start)

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
        input_start_len = len(active_profile.input_start) if active_profile and active_profile.input_start else 0
        before_input = data_str[:start_idx - input_start_len]
        input_content = data_str[start_idx:end_idx]
        after_input = data_str[end_idx:]

        # Calculate position relative to input area start
        relative_pos = insert_pos - start_idx

        # Find where the cursor marker is (reverse video ANSI codes)
        cursor_marker_start = input_content.find('\x1b[7m')

        # Debug: Show what we're working with
        log_message("DEBUG", f"Relative pos: {relative_pos}, cursor at: {cursor_marker_start}")
        log_message("DEBUG",
                    f"Content at insert point: {repr(input_content[max(0, relative_pos - 5):relative_pos + 10])}")

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
            elif char == ' ' and visual_spaces_removed < visual_partial_length:
                # This is a space we should remove
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

        # Ensure we maintain the exact same length
        if len(new_input) > len(input_content):
            new_input = new_input[:len(input_content)]
        elif len(new_input) < len(input_content):
            # Pad at the end if needed
            new_input += ' ' * (len(input_content) - len(new_input))

        # Reconstruct the full string
        data_str = before_input + active_profile.input_start + new_input + after_input

        log_message("DEBUG", f"Input area update: old_len={len(input_content)}, new_len={len(new_input)}")
        log_message("DEBUG", f"Inserted partial transcript: '{partial_to_show}' at position {insert_pos}")
    else:
        log_message("DEBUG", "No space available for partial transcript")

    return data_str


def handle_partial_transcript(text: str):
    """Handle partial transcript from streaming ASR"""
    log_message("INFO", f"[ASR PARTIAL] Received partial transcript: '{text}'")
    if not asr_state.partial_enabled:
        return

    asr_state.current_partial = text # + '\u200b'
    asr_state.last_partial_transcript_time = time.time()

    if not text.strip():
        return

    if current_master_fd is not None and asr_state.waiting_for_input:
        try:
            os.write(current_master_fd, '>'.encode('utf-8'))
        except Exception as e:
            log_message("ERROR", f"Failed to trigger screen update: {e}")


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
        log_message("INFO", "Tap-to-talk timeout - considering key released")
        return True
    return False


def ensure_asr_cleanup():
    """Ensure ASR is properly cleaned up when disabled.
    
    This function handles ASR cleanup when:
    - MCP disables ASR
    - ASR encounters errors requiring cleanup
    """
    if not ASR_AVAILABLE:
        return
        
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
    """Ensure ASR is initialized if it's enabled but not yet initialized.
    
    This centralized function handles ASR initialization for all cases:
    - Initial CLI startup with ASR enabled
    - MCP enabling ASR after startup
    - Re-initialization after ASR errors
    
    Returns:
        bool: True if ASR was newly initialized, False otherwise
    """
    if not ASR_AVAILABLE:
        return False
        
    shared_state = get_shared_state()
    
    # Check if ASR should be initialized
    if shared_state.asr_enabled and not shared_state.asr_initialized:
        log_message("INFO", "ASR enabled but not initialized - initializing now")
        
        try:
            # Get the configured provider if available, otherwise select the best
            if hasattr(asr, '_stored_config') and asr._stored_config:
                asr_provider = asr._stored_config.provider
                log_message("INFO", f"Using configured ASR provider: {asr_provider}")
            else:
                asr_provider = asr.select_best_asr_provider()
                log_message("INFO", f"Selected best ASR provider: {asr_provider}")
            
            log_message("INFO", f"Initializing ASR with provider: {asr_provider}")
            
            # Start ASR with the standard callbacks
            asr.start_dictation(handle_dictated_text, handle_partial_transcript)
            
            # Start in paused state so auto-input logic can manage it
            asr.set_ignore_input(True)
            log_message("INFO", "ASR initialized and paused for auto-input management")
            
            # Update shared state
            shared_state.set_asr_initialized(True, provider=asr_provider)
            
            # Reset ASR state for clean startup
            if asr_state:
                asr_state.asr_auto_started = False
                
            return True
            
        except Exception as e:
            log_message("ERROR", f"Failed to initialize ASR: {e}")
            # Mark as not initialized so we can retry later
            shared_state.set_asr_initialized(False)
            return False
            
    return False


def ensure_tts_state_sync():
    """Ensure TTS state is synchronized and clear pending text if TTS is disabled"""
    shared_state = get_shared_state()
    
    # If TTS was disabled, clear any pending speech text
    if not shared_state.tts_enabled and terminal and terminal.pending_speech_text:
        log_message("INFO", "TTS disabled, clearing pending speech text")
        terminal.pending_speech_text = ""
        terminal.pending_text_line_number = None


def ensure_comms_initialized():
    """Ensure communication channels are initialized based on shared state"""
    if not COMMS_AVAILABLE:
        return False
        
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
    if not ASR_AVAILABLE:
        return False
    
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
        
    # If ASR was explicitly enabled via MCP when CLI had it off, treat as auto-input
    if asr_mode == "off" and shared_state.asr_enabled:
        asr_mode = "auto-input"
        log_message("DEBUG", f"ASR enabled via MCP, overriding CLI mode to auto-input")
        
    # Log the current state for debugging
    is_tts_speaking = tts.is_speaking()
    is_asr_recognizing = asr.is_recognizing() if ASR_AVAILABLE else False
    
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
        # Tap-to-talk mode: only enable when key is held
        if asr_state.tap_to_talk_active and not is_asr_recognizing:
            if not asr_state.asr_auto_started:
                try:
                    log_message("INFO", "Starting ASR dictation (tap-to-talk mode)...")
                    asr.start_dictation(handle_dictated_text, handle_partial_transcript)
                    asr_state.asr_auto_started = True
                    return True
                except Exception as e:
                    log_message("ERROR", f"Failed to start ASR (tap-to-talk): {e}")
        elif not asr_state.tap_to_talk_active and asr_state.asr_auto_started:
            # Stop ASR when key is released
            try:
                asr.stop_dictation()
                asr_state.asr_auto_started = False
                log_message("INFO", "Stopped ASR: tap-to-talk key released")
            except Exception as e:
                log_message("ERROR", f"Failed to stop ASR (tap-to-talk): {e}")
    
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
            data = os.read(master_fd, 4096)
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
            line_buffer += data
    except:
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
    master_fd = None
    proc = None
    
    try:
        # Set up terminal and spawn process
        master_fd, slave_fd, stdin_flags, proc = await setup_terminal_for_command(cmd)
        current_master_fd = master_fd
        current_proc = proc
        
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
        buffer_switch_time = 0
        consecutive_redraws = 0
        last_redraw_time = 0
        skip_duplicate_mode = False
        
        # For periodic status updates
        last_status_check = 0

        old_tty_attrs = None
        if sys.stdin.isatty():
            old_tty_attrs = termios.tcgetattr(sys.stdin)
            # Store globally for signal handler
            _original_tty_attrs = old_tty_attrs
            tty.setraw(sys.stdin.fileno())

        # Main processing loop
        while True:
            if process_pending_resize(master_fd):
                continue
                
            # Periodic status update for dictation indicator and auto-listen check
            enabled_auto_listen, last_status_check = await periodic_status_check(
                master_fd, asr_mode, last_status_check)

            # Check if we need to write buffered data
            wlist = [sys.stdout] if len(stdout_buffer) > 0 else []
            
            # Reduced timeout from 0.01 to 0.001 for better responsiveness
            rlist, wlist_ready, _ = select.select([sys.stdin, master_fd], wlist, [], 0.001)
            
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
                    os.write(current_master_fd, input_bytes)
                    os.write(current_master_fd, RETURN)
                    sys.stdout.flush()
                    log_message("INFO", f"Sent comms input to PTY: {comms_input}")
                except Exception as e:
                    log_message("ERROR", f"Failed to send comms input to PTY: {e}")

            # Check for auto-submit of dictated text after 3 seconds of silence
            if asr_state.has_pending_transcript and current_master_fd:
                current_time = time.time()
                time_since_last_finalized = current_time - asr_state.last_finalized_transcript_time
                time_since_last_partial = current_time - asr_state.last_partial_transcript_time
                
                # Auto-submit if:
                # - It's been at least 3 seconds since the last finalized transcript
                # - No partial transcripts have been received in the last 3 seconds
                if time_since_last_finalized >= 0.2 and time_since_last_partial >= 1.0:
                    try:
                        os.write(current_master_fd, RETURN)
                        asr_state.has_pending_transcript = False
                        log_message("INFO", "[ASR AUTO-SUBMIT] Auto-submitted dictated text after 3 seconds of silence")
                    except Exception as e:
                        log_message("ERROR", f"[ASR AUTO-SUBMIT] Failed to auto-submit: {e}")

            if sys.stdin in rlist:
                # Handle record mode for input
                if recorder.enabled:
                    try:
                        input_data = os.read(sys.stdin.fileno(), 4096)
                        if input_data:
                            recorder.record_event('INPUT', input_data)
                            # Forward to PTY
                            os.write(master_fd, input_data)
                    except (BlockingIOError, OSError):
                        pass
                else:
                    await handle_stdin_input(master_fd, asr_mode)

            if master_fd in rlist or enabled_auto_listen:
                try:
                    data = os.read(master_fd, PTY_READ_SIZE)
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
                        buffer_switch_time = time.time()
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
                        if active_profile.input_start and active_profile.input_start in data_str:
                            start_idx = data_str.find(active_profile.input_start) + len(active_profile.input_start)
                            log_message("DEBUG", f"Found input start at position {start_idx}")
                            in_input = True

                    # Display partial transcript if available and we're in input area
                    if asr_state.current_partial and in_input:
                        log_message("DEBUG", f"partial_needs_display {asr_state.current_partial}")
                        data_str = insert_partial_transcript(data_str, asr_state, active_profile)
                        # data_str = data_str.replace('\u200b', ' '*len(asr_state.current_partial))

                    in_input = False

                    output_data = data_str.encode('utf-8')
                    output_data = output_data.replace(SPACE_THEN_BACK, b'')

                    # Show microphone emoji if ASR is active
                    if asr_state.asr_auto_started and asr_state.waiting_for_input and asr_mode != "off" and not asr.is_ignoring_input():
                        if is_logging_enabled():
                            log_message("DEBUG", f"ASR mic conditions met - asr_auto_started={asr_state.asr_auto_started}, waiting_for_input={asr_state.waiting_for_input}, asr_mode={asr_mode}")
                        output_data = modify_prompt_for_asr(output_data, active_profile.input_start, active_profile.input_mic_replace)

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
                            import traceback
                            log_message("ERROR", f"Traceback: {traceback.format_exc()}")
                    
                    log_message("DEBUG", f"After stdout write/flush")

                    # Removed buffer switch delay to ensure all lines are processed

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
                    import traceback
                    log_message("ERROR", f"Traceback: {traceback.format_exc()}")

            if proc.returncode is not None:
                if recorder.enabled:
                    # In record mode, drain and record any remaining output
                    try:
                        while True:
                            data = os.read(master_fd, PTY_READ_SIZE)
                            if not data:
                                break
                            recorder.record_event('OUTPUT', data)
                            sys.stdout.buffer.write(data)
                            sys.stdout.flush()
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
                                    log_message("WARNING", f"Skipping malformed UTF-8 line")
                                    continue
                                line_idx, action = output_buffer.add_or_update_line(line)

                                if action in ['added', 'modified']:
                                    text_to_speak, buffer, prev_line, _ = process_line(
                                        line, buffer, prev_line, False, line_idx, asr_mode
                                    )
                                    if text_to_speak:
                                        queue_output(strip_profile_symbols(text_to_speak), line_idx)
                    except:
                        pass
                else:
                    await drain_pty_output(master_fd, line_buffer)
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
                text_to_speak, buffer, prev_line, _ = process_line(line, buffer, prev_line, False, line_idx, asr_mode)
                if text_to_speak:
                    queue_output(strip_profile_symbols(text_to_speak), line_idx)

        if buffer:
            # Use the last line index for any remaining buffer content
            last_idx = output_buffer.next_index - 1 if output_buffer.next_index > 0 else 0
            process_remaining_buffer(buffer, last_idx)

        # Flush any pending speech text before command completes
        send_pending_text()

        tts.wait_for_tts_to_finish()
        
        # Stop ASR if it was running
        if ASR_AVAILABLE:
            try:
                asr.stop_dictation()
            except Exception:
                pass

        # Log final buffer statistics
        log_message("INFO", f"Command completed. Total lines in buffer: {output_buffer.get_line_count()}")

        # Save recorded data
        recorder.save()

        if proc:
            await proc.wait()
        
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
                        sys.stdout.buffer.write(remaining)
                        sys.stdout.flush()
                    break
        except Exception as e:
            log_message("ERROR", f"Error restoring stdout: {e}")

        return proc.returncode if proc else 1

    finally:
        # Close PTY file descriptor if it was opened
        if master_fd is not None:
            try:
                os.close(master_fd)
                log_message("INFO", "Closed master_fd in finally block")
            except Exception as e:
                log_message("ERROR", f"Failed to close master_fd: {e}")
            current_master_fd = None
        
        if old_tty_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, old_tty_attrs)
                log_message("INFO", "Restored terminal attributes in finally block")
            except Exception as e:
                log_message("ERROR", f"Failed to restore terminal attributes in finally: {e}")
            # Clear global reference
            _original_tty_attrs = None

        try:
            fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, stdin_flags)
        except Exception as e:
            log_message("ERROR", f"Failed to restore stdin flags: {e}")
        
        # Ensure cursor is visible
        try:
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
            queue_output(strip_profile_symbols(final_text), line_idx)


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
        except:
            pass
    
    while b'\n' in line_buffer and not detected_prompt:
        nl_pos = line_buffer.find(b'\n')
        line_bytes = line_buffer[:nl_pos]
        line_buffer = line_buffer[nl_pos + 1:]
        
        line = line_bytes.decode('utf-8', errors='ignore')

        line_idx, action = output_buffer.add_or_update_line(line)
        
        if action == 'added':
            track_line_position(line_idx, cursor_row)

        if line.strip():
            log_message("INFO", f"Processing line {line_idx} (action={action}, row={cursor_row}): '{line.strip()[:100]}...'")
        
        # Only process lines that have been added or modified
        if action in ['added', 'modified']:
            text_to_speak, text_buffer, prev_line, detected_prompt = process_line(
                line, text_buffer, prev_line, skip_duplicates, line_idx, asr_mode
            )
        else:
            # For unchanged lines, don't process but preserve state
            text_to_speak = None
            detected_prompt = False

        response_prefix = active_profile.response_prefix
        if clean_text(line).strip().startswith(response_prefix):
            asr_state.waiting_for_input = False
            asr_state.refresh_spaces_added = 0  # Reset refresh spaces when response starts
            check_and_enable_auto_listen(asr_mode)

        if detected_prompt:
            asr_state.waiting_for_input = True
            asr_state.refresh_spaces_added = 0  # Reset refresh spaces for new prompt
            log_message("INFO", f"Detected prompt")
            check_and_enable_auto_listen(asr_mode)
        
        if text_to_speak:
            queue_output(strip_profile_symbols(text_to_speak), line_idx)
    
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

def signal_handler(signum):
    """Handle shutdown signals"""
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
            sys.stdout.write('\033[?25h')  # Show cursor
            sys.stdout.flush()
        except Exception:
            pass
        
        # Stop TTS immediately
        try:
            from .tts import stop_tts_immediately
            stop_tts_immediately()
        except Exception:
            pass
        
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
        sys.stdout.write('\033[?25h')  # Show cursor
        sys.stdout.flush()
    except Exception:
        pass
    
    # First ensure we clean up the terminal - this is critical
    cleanup_terminal()
    
    # Then stop services gracefully
    if ASR_AVAILABLE:
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


async def replay_recorded_session(replay_file: str, auto_skip_tts: bool = True, tts_config: dict = None, record_file: str = None, capture_tts: bool = False, disable_tts: bool = False, show_output: bool = True, command_name: str = None, verbosity: int = 0, log_file: str = None) -> Union[int, List[Tuple[float, str, int]]]:
    """Replay a recorded session file through the TTS pipeline for debugging"""
    global active_profile, terminal, asr_state, verbosity_level
    
    # Initialize terminal state if not already done
    if terminal is None:
        terminal = TerminalState()
    
    # Initialize ASR state if not already done
    if asr_state is None:
        asr_state = ASRState()
    
    # Set verbosity level
    verbosity_level = verbosity
    
    setup_logging(log_file)
    log_message("INFO", f"Replaying recorded session: {replay_file} with verbosity level {verbosity}")
    
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
    engine = configure_tts_engine(tts_config, auto_skip_tts)
    if not engine:
        return 1

    # Set up TTS capture if requested
    tts_outputs = []

    # Parse recorded session
    entries = SessionRecorder.parse_file(replay_file)
    if not entries:
        print(f"Error: No valid entries found in replay file: {replay_file}", file=sys.stderr)
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
                buffer_switch_time = time.time()
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
            text_to_speak, buffer, prev_line, _ = process_line(
                line, buffer, prev_line, skip_duplicate_mode, line_idx, "off"
            )
            if text_to_speak:
                queue_output(strip_profile_symbols(text_to_speak), line_idx)
    
    # Process any remaining buffer
    if buffer:
        last_idx = output_buffer.next_index - 1 if output_buffer.next_index > 0 else 0
        process_remaining_buffer(buffer, last_idx)
    
    print(f"Replay completed. Total lines in buffer: {output_buffer.get_line_count()}")
    
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
        tts.shutdown_tts()
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
        # Use the global setup_logging function
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


# High-level API functions
async def run_with_talkito(command: List[str], **kwargs) -> int:
    """Run a command with talkito functionality
    
    Args:
        command: Command and arguments to run
        **kwargs: Additional options (verbosity, tts_config, asr_config, etc.)
    
    Returns:
        Exit code of the command
    """
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGWINCH, debounced_winch_handler)
    
    core = TalkitoCore(
        verbosity_level=kwargs.get('verbosity', 0),
        log_file_path=kwargs.get('log_file')
    )
    
    # Set up profile if specified
    if 'profile' in kwargs:
        profile = get_profile(kwargs['profile'])
        if profile:
            core.active_profile = profile
    
    # Ensure we have at least a default profile
    if core.active_profile is None:
        core.active_profile = get_profile('default')
    
    # Configure TTS based on kwargs
    tts_config = kwargs.get('tts_config', {})
    auto_skip_tts = kwargs.get('auto_skip_tts', False)
    
    # Set up TTS engine
    if tts_config and tts_config.get('provider') != 'system':
        if not tts.configure_tts_from_dict(tts_config):
            raise RuntimeError("Failed to configure TTS provider")
        engine = "cloud"
    else:
        engine = TTS_ENGINE
        if engine == "auto":
            engine = tts.detect_tts_engine()
            if engine == "none":
                raise RuntimeError("No TTS engine found. Please install espeak, festival, flite (Linux) or use macOS")
    
    # Start TTS worker
    tts.start_tts_worker(engine, auto_skip_tts)
    
    # Start background update checker
    from .update import start_background_update_checker
    start_background_update_checker()
    
    # Update shared state for TTS initialization
    from .state import get_shared_state
    shared_state = get_shared_state()
    
    # Get the actual provider that was configured
    if tts_config and tts_config.get('provider'):
        provider = tts_config.get('provider')
    else:
        # For system/auto, get the actual engine being used
        provider = engine if engine != 'cloud' else getattr(tts, 'tts_provider', 'system')
    
    shared_state.set_tts_initialized(True, provider)
    
    # Configure ASR based on kwargs
    if 'asr_config' in kwargs and ASR_AVAILABLE:
        asr.configure_asr_from_dict(kwargs['asr_config'])
    
    # Set ASR state based on mode
    asr_mode = kwargs.get('asr_mode', 'auto-input')
    if ASR_AVAILABLE:
        if asr_mode != 'off':
            # Enable ASR in shared state - centralized functions will handle initialization
            log_message("INFO", f"Enabling ASR for {asr_mode} mode")
            shared_state.set_asr_enabled(True)
        else:
            # Explicitly disable ASR when mode is 'off'
            log_message("INFO", "Disabling ASR due to --asr-mode off")
            shared_state.set_asr_enabled(False)
    
    # Set up communications if configured
    if 'comms_config' in kwargs and COMMS_AVAILABLE:
        global comm_manager
        config = kwargs['comms_config']
        # Extract enabled providers from config
        providers = []
        if hasattr(config, 'sms_enabled') and config.sms_enabled:
            providers.append('sms')
        if hasattr(config, 'whatsapp_enabled') and config.whatsapp_enabled:
            providers.append('whatsapp')
        if hasattr(config, 'slack_enabled') and config.slack_enabled:
            providers.append('slack')

        comm_manager = comms.setup_communication(providers=providers, config=config)
        core.comm_manager = comm_manager

    try:
        # Run the command
        return await core.run_command(
            command,
            asr_mode=asr_mode,
            record_file=kwargs.get('record_file')
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


