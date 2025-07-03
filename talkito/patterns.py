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
patterns.py - Centralized regex patterns for performance optimization
All regex patterns are compiled once at module load time
"""

import re

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

# Common text processing patterns
FILE_PATH_PATTERN = re.compile(r'^[\w/\-_.]+\.(py|js|txt|md|json|yaml|yml|sh|c|cpp|h|java|go|rs|rb|php)$')
NUMBERS_ONLY_PATTERN = re.compile(r'^\s*\d+(\s+\d+)*\s*$')
BOX_CONTENT_PATTERN = re.compile(r'│\s*([^│]+)\s*│')
BOX_SEPARATOR_PATTERN = re.compile(r'^[─═╌╍]+$')
PROMPT_PATTERN = re.compile(r'^\s*[>\$#]\s*$')
SENTENCE_END_PATTERN = re.compile(r'[.!?]$')

# Progress patterns (compiled versions)
PROGRESS_PATTERNS = [
    re.compile(r'^\s*\.{3,}$'),  # Multiple dots
    re.compile(r'^\s*[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]'),  # Spinner characters
    re.compile(r'^\s*\d+\s+'),  # Lines starting with line numbers (code diffs)
    re.compile(r'^[+-]\s*\d+'),  # Diff line numbers with +/-
]

# Additional patterns for text cleaning
ORPHANED_M_PATTERN = re.compile(r'(?<![a-zA-Z0-9\'])m(?![a-zA-Z])')
ANSI_NUMBER_M_PATTERN = re.compile(r'(?<![a-zA-Z])\d{1,3}m\b')
CONTINUATION_LINE_PATTERN = re.compile(r'^\s+\S')

# TTS-specific patterns
PATH_PATTERN = re.compile(r'[~./\\]?[\w./\\-]+(?:\.[a-zA-Z]+)?')
URL_PATTERN = re.compile(r'https?://[^\s]+|www\.[^\s]+')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
IP_PATTERN = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
HEX_PATTERN = re.compile(r'\b0x[0-9a-fA-F]+\b')
UUID_PATTERN = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')
HASH_PATTERN = re.compile(r'\b[a-f0-9]{32,64}\b')  # MD5 or SHA hashes

# Punctuation cleaning patterns
MULTI_PERIOD_PATTERN = re.compile(r'\.{2,}')
PERIOD_COMMA_PATTERN = re.compile(r'\.,')
COMMA_PERIOD_PATTERN = re.compile(r',\.')
MULTI_PUNCTUATION_PATTERN = re.compile(r'([.!?]){2,}')

# WhatsApp formatting patterns (from comms.py)
WHATSAPP_FORMATTING_PATTERNS = [
    (re.compile(r'```([^`]+)```'), r'"\1"'),  # Code blocks
    (re.compile(r'`([^`]+)`'), r'"\1"'),      # Inline code
    (re.compile(r'\*\*([^*]+)\*\*'), r'\1'),  # Bold
    (re.compile(r'\*([^*]+)\*'), r'\1'),      # Italic
    (re.compile(r'~~([^~]+)~~'), r'\1'),      # Strikethrough
    (re.compile(r'^#+\s+'), ''),              # Headers
    (re.compile(r'^\s*[-*]\s+'), '• '),       # Bullet points
    (re.compile(r'^\s*\d+\.\s+'), ''),        # Numbered lists
]

# Profile-specific patterns (can be extended by profiles)
def compile_profile_patterns(patterns):
    """Compile a list of pattern strings into regex objects"""
    compiled = []
    for pattern in patterns:
        if pattern:  # Skip empty patterns
            try:
                compiled.append(re.compile(pattern))
            except re.error as e:
                # Log error but don't fail
                from .logs import log_message
                log_message("ERROR", f"Failed to compile pattern '{pattern}': {e}")
    return compiled