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

"""Program-specific profiles for talkito.py - Contains pattern matching rules and behavior customization for different CLI programs."""

import re
import argparse
from typing import Dict, List, Tuple, Optional, Pattern, Union, Sequence
from dataclasses import dataclass, field


@dataclass
class Profile:
    """Represents a program-specific profile with compiled patterns"""
    
    # Basic settings
    supported: bool
    name: str
    warning: Optional[str] = None
    writes_partial_output: bool = False
    needs_full_lines: Optional[bool] = False
    response_prefix: str = '⏺'
    continuation_prefix: Optional[str] = None
    question_prefix: Optional[str] = None
    
    # Raw patterns (applied before text cleaning)
    raw_skip_patterns: List[str] = field(default_factory=list)
    
    # Cleaned text patterns
    # skip_patterns can be either strings (verbosity 0) or (verbosity_level, pattern) tuples
    skip_patterns: Sequence[Union[str, Tuple[int, str]]] = field(default_factory=list)
    speak_patterns: List[str] = field(default_factory=list)
    prompt_patterns: List[str] = field(default_factory=list)
    replacements: List[tuple[str, str]] = field(default_factory=list)
    # exception_patterns: lines containing these patterns are never skipped, regardless of verbosity
    # Format: List of (max_verbosity, pattern) - exception applies up to this verbosity level
    exception_patterns: List[Tuple[int, str]] = field(default_factory=list)
    
    # Behavior settings
    skip_progress: List[str] = field(default_factory=list)
    strip_symbols: List[str] = field(default_factory=list)

    # Input handling
    input_start: List[str] = field(default_factory=list)
    input_mic_replace: Optional[str] = None
    input_speaker_replace: Optional[str] = None

    # Compiled patterns (cached)
    _compiled_raw_skip: List[Pattern] = field(default_factory=list, init=False)
    _compiled_skip: List[Tuple[int, Pattern]] = field(default_factory=list, init=False)  # (verbosity, pattern)
    _compiled_speak: List[Pattern] = field(default_factory=list, init=False)
    _compiled_prompt: List[Pattern] = field(default_factory=list, init=False)
    _compiled_extract: List[Tuple[Pattern, int]] = field(default_factory=list, init=False)
    _compiled_exceptions: List[Tuple[int, Pattern]] = field(default_factory=list, init=False)  # (max_verbosity, pattern)
    _compiled_continuation: Optional[Pattern] = field(default=None, init=False)
    _compiled_question: Optional[Pattern] = field(default=None, init=False)
    _compiled_interaction_menu: List[Pattern] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Compile all regex patterns for efficiency"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile all regex patterns"""
        self._compiled_raw_skip = [re.compile(p) for p in self.raw_skip_patterns]
        
        # Handle skip_patterns which can be strings or (verbosity, pattern) tuples
        self._compiled_skip = []
        for p in self.skip_patterns:
            if isinstance(p, str):
                # Default to verbosity 4 (always filter) for string patterns
                self._compiled_skip.append((4, re.compile(p)))
            else:
                # Tuple format: (verbosity_level, pattern)
                verbosity, pattern = p
                self._compiled_skip.append((verbosity, re.compile(pattern)))
        
        self._compiled_speak = [re.compile(p) for p in self.speak_patterns]
        self._compiled_prompt = [re.compile(p) for p in self.prompt_patterns]
        
        # Compile exception patterns
        self._compiled_exceptions = []
        for min_verbosity, pattern in self.exception_patterns:
            self._compiled_exceptions.append((min_verbosity, re.compile(pattern)))
        
        # Compile continuation prefix pattern
        if self.continuation_prefix:
            self._compiled_continuation = re.compile(self.continuation_prefix)
        
        # Compile question prefix pattern
        if self.question_prefix:
            self._compiled_question = re.compile(self.question_prefix)
        
        # Compile interaction menu patterns (for Claude Code interactive menus)
        interaction_menu_patterns = [
            r'❯\s*\d+\.',           # ❯ 1. Yes
            r'\s*\d+\.\s*Yes',      # 1. Yes 
            r'\s*\d+\.\s*No',       # 2. No
            r'don\'t ask again',    # don't ask again options
            r'tell Claude what',    # tell Claude what to do differently
        ]
        self._compiled_interaction_menu = [re.compile(p, re.IGNORECASE) for p in interaction_menu_patterns]

    def should_skip_raw(self, line: str) -> bool:
        """Check if line should be skipped based on raw patterns"""
        return any(p.search(line) for p in self._compiled_raw_skip)
    
    def get_raw_skip_reason(self, line: str) -> Optional[str]:
        """Get the reason why a line would be skipped by raw patterns"""
        for i, pattern in enumerate(self._compiled_raw_skip):
            if pattern.search(line):
                return f"Raw skip pattern: {self.raw_skip_patterns[i]}"
        return None

    def apply_replacements(self, line: str) -> str:
        """Apply replacements based on raw patterns"""
        for replacement in self.replacements:
            line = line.replace(replacement[0], replacement[1])
        return line
    
    def should_skip(self, line: str, verbosity: int = 0) -> bool:
        """Check if line should be skipped based on cleaned patterns and verbosity level"""
        if self.is_question_line(line):
            return False

        # Check exception patterns first - these override all skip rules
        for min_verbosity, pattern in self._compiled_exceptions:
            if pattern.search(line) and verbosity >= min_verbosity:
                return False
        
        # # Check if it's a response line
        # if line.startswith(self.response_prefix):
        #     return False
        
        # Check speak patterns (should NOT skip)
        if any(p.search(line) for p in self._compiled_speak):
            return False
        
        # Check prompt patterns (should skip UI chrome like prompts)
        if any(p.search(line) for p in self._compiled_prompt):
            return True
        
        # Check skip patterns with verbosity
        for min_verbosity, pattern in self._compiled_skip:
            if pattern.search(line):
                # Skip if current verbosity is less than the required verbosity
                # min_verbosity=1 means skip unless -v or higher (skip at verbosity 0)
                # min_verbosity=2 means skip unless -vv or higher (skip at verbosity 0,1)
                # min_verbosity=3 means skip unless -vvv or higher (skip at verbosity 0,1,2)
                # min_verbosity=4 means always skip (would need -vvvv which we don't allow)
                if verbosity < min_verbosity:
                    return True
        
        # Check skip progress words
        if any(word in line for word in self.skip_progress):
            return True
        
        return False
    
    def matches_exception_pattern(self, line: str, verbosity: int = 0) -> bool:
        """Check if line matches an exception pattern (should be spoken even if it matches skip patterns)"""
        for min_verbosity, pattern in self._compiled_exceptions:
            if pattern.search(line) and verbosity >= min_verbosity:
                return True
        return False
    
    def get_skip_reason(self, line: str, verbosity: int = 0) -> Optional[str]:
        """Get the reason why a line would be skipped, or None if it wouldn't be skipped"""
        # Check exception patterns first - these override all skip rules
        for min_verbosity, pattern in self._compiled_exceptions:
            if pattern.search(line) and verbosity >= min_verbosity:
                return None  # Exception pattern matched - line won't be skipped
        
        # Check speak patterns (should NOT skip)
        for i, p in enumerate(self._compiled_speak):
            if p.search(line):
                return None  # Speak pattern matched - line won't be skipped
        
        # Check prompt patterns (should skip)
        for i, p in enumerate(self._compiled_prompt):
            if p.search(line):
                return f"Prompt pattern: {self.prompt_patterns[i]}"
        
        # Check skip patterns with verbosity
        for i, (min_verbosity, pattern) in enumerate(self._compiled_skip):
            if pattern.search(line):
                if verbosity < min_verbosity:
                    # Find the original pattern string
                    original_pattern = None
                    for j, p in enumerate(self.skip_patterns):
                        if isinstance(p, str) and j == i:
                            original_pattern = p
                            break
                        elif isinstance(p, tuple) and j == i:
                            original_pattern = p[1]
                            break
                    return f"Skip pattern (verbosity {min_verbosity}): {original_pattern or pattern.pattern}"
        
        # Check skip progress words
        for word in self.skip_progress:
            if word in line:
                return f"Skip progress word: '{word}'"
        
        return None

    def is_continuation_line(self, line: str) -> bool:
        """Check if line is a continuation of the previous line"""
        if line.strip().endswith(":"):
            return False
        if self.is_question_line(line):
            return False
        if self._compiled_continuation:
            return self._compiled_continuation.match(line) is not None
        return False

    def is_question_line(self, line: str) -> bool:
        """Check if line is a question that should always be spoken"""
        if self._compiled_question:
            return self._compiled_question.search(line) is not None
        return False
    
    def is_interaction_menu_line(self, line: str) -> bool:
        """Check if line is part of an interactive menu (options following a question)"""
        return any(pattern.search(line) for pattern in self._compiled_interaction_menu)

    def is_input_start(self, line: str) -> bool:
        """Check if line is the start of an input block"""
        if self.input_start and any(s in line for s in self.input_start):
            return True
        return any(p.search(line) for p in self._compiled_prompt)

    def extract_text(self, line: str) -> Optional[str]:
        """Extract text based on extraction patterns"""
        for pattern, group_idx in self._compiled_extract:
            match = pattern.search(line)
            if match:
                try:
                    return match.group(group_idx)
                except IndexError:
                    continue
        return None
    
    def strip_symbols_from(self, text: str) -> str:
        """Strip profile-specific symbols from text"""
        for symbol in self.strip_symbols:
            text = text.replace(symbol, '')
        return text.strip()


# Common patterns that should be applied to all profiles
COMMON_SKIP_PATTERNS = [
    # Timestamp patterns - filter log lines with timestamps in first 20 chars
    (2, r'^.{0,20}\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'),  # [YYYY-MM-DD HH:MM:SS format
    (2, r'^.{0,20}\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'),    # YYYY-MM-DD HH:MM:SS format (no brackets)
    (2, r'^.{0,20}\d{2}:\d{2}:\d{2}'),                         # HH:MM:SS format in first 20 chars
    (2, r'^.{0,20}\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}'),    # YYYY/MM/DD HH:MM:SS format
    (2, r'^.{0,20}\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}'),    # MM/DD/YYYY HH:MM:SS format

    # Level 2: Filter unless -vv (code)
    (2, r'^  \+'),
    (2, r'^  \['),
    (2, r'(import|include|require|use)\s+\w+…\)'),  # Import statements (Python/JS/Rust/PHP)
    (2, r'     '),                                  # Indent usually means code
    (2, r'  - '),                                   # Indent with dash usually means code
    (2, r'^\s{2,}\d+\s{2,}[a-zA-Z_#/]'),            # Line numbers + code/comments (added / for C++)
    (2, r'(/usr/bin/|#!/)'),                        # Shebang path prefix or shebang start
    (2, r'^\s{2,}(import|include|require|use|from|#include)\s+'), # Indented imports (multi-lang)
    (2, r'│'),                                      # Lines with pipes anywhere
    (2, r'^(#|//|/\*|\*)'),                         # Comments: #, //, /*, or continuation *
    (2, r'^\s*\['),

    (3, r'^\s*\/'),
    (3, r'^\/help'),
    (3, r'sc to interrupt|sc interrupt|enter send'),  # UI hints
    (3, r'ctrl\s*\+r\s*to\s*expand'),  # UI hints
    (3, r'^\s*⏺\s+Task\('),  # Claude Code task indicator
    (3, r'^Task\([^)]+\)'),  # Task() without prefix
    (3, r'Read \d+ lines'),  # Skip "Read X lines" messages
    (3, r'Edit file'),
    (3, r'\)…'),
    (3, r'cwd:'),
    (3, r'⎿'),  # subheadings underneath an edit block like "Wrote 122 lines to x.py"
    (3, r'(Bash|Read|Edit|Write|Grep|Task|MultiEdit|NotebookEdit|WebFetch|TodoWrite|Update|Modify|Create|Search)\s*\('),
    (3, r'^[^A-Za-z0-9]*[A-Za-z][a-z]+(?:-[a-z]+)*(?:\.\.\.|…|\.)'),
    (3, r'^∴'),

    (4, r'^\^C '),
    (4, r'^░█'),
    (4, r'Press Ctrl-C'),
    (4, r'^\s*ctrl\+c '),
    (4, r'xai:function_call'),
    (4, r'Error File content|\[Pasted text'),
    (4, r'^[|│]\s*>\s*'),
    (4, r'(╭|╮|╯|╰)'),
    (4, r'\? for shortcuts'),
    (4, r'\(node:'),
    (4, r'^\['),
    (4, r'⏵⏵ auto-accept edits'),
    (4, r'===|▀▀▀▀|………|╌╌'),
]


# Profile definitions
CLAUDE_PROFILE = Profile(
    supported=True,
    name='claude',
    needs_full_lines=True,
    response_prefix='⏺',
    continuation_prefix=r'^(\s+[-\w()\'"]|  [a-z]\w*\.|[a-z]\w*\. )',
    question_prefix=r'^\s*Do you',
    raw_skip_patterns=[
        r'\[38;5;153m│.*\[38;5;246m\d+',      # Box drawing + line numbers
        r'\[38;5;246m\d+\s*\[39m',            # Direct line numbers
        r'\x1b\[48;5;237m\x1b\[38;5;231m',    # User input (bg:237, fg:231)
    ],
    exception_patterns=[
        (0, r'Claude Code v'),       # ✻ Welcome to Claude Code
    ],
    skip_patterns=COMMON_SKIP_PATTERNS + [
        # Level 1: Filter unless -v (tips, hints, usage info, single-word status)
        (1, r'Tip:'),                         # Tips
        (1, r'usage limit'),                  # Usage limit messages
        (1, r'to use best available model'),  # Model suggestions
        (1, r'Update Todos'),                 # Update Todos

        # Level 3: Filter unless -vvv (tool calling details, implementation details)
        (3, r'Claude needs your permission'), # Claude needs your permission to use X
        (3, r'talkito:'),
        (3, r'^│'),
        (3, r'^\s*/'),

        (4, r'^\s*>\s*'),
    ],
    skip_progress=['Forming', 'Exploring'],
    strip_symbols=['⏺'],
    prompt_patterns=[
        r'^│\s*>\s*',        # Line starting with box character and prompt
        r'^>\s*.+',          # Line starting with >
        r'^\s*│\s*>\s*',     # Line starting with optional spaces, box, prompt
    ],
    input_start=['>'],
    input_mic_replace='🎤',
    input_speaker_replace='📢',
)


CODEX_PROFILE = Profile(
    supported=True,
    name='codex',
    response_prefix='⏺',
    continuation_prefix=r'^(\s+[-\w()\'"]|  [a-z]\w*\.|[a-z]\w*\. )',
    question_prefix=r'│ Do',
    raw_skip_patterns=[
        r'\[38;5;153m│.*\[38;5;246m\d+',  # Box drawing + line numbers
        r'\[38;5;246m\d+\s*\[39m',  # Direct line numbers
    ],
    exception_patterns=[
        (0, r'>_ OpenAI Codex'),
    ],
    skip_patterns=COMMON_SKIP_PATTERNS + [
        (2, r'^    '),
        (3, r'talkito:'),
        (3, r'^\s*[└□✔]'),
        (3, r'^[^\s•]'),  # Skip lines not starting with space/tab or •
        (3, r'• (Ran|Explored|Edited|Added|Updated|Called)'),
        (3, r'^› '),
        (3, r'esc to '),
        (3, r'\? for shortcuts'),
        (3, r'^>>'),

        (4, r'^✨⬆️'),
        (4, r'^[A-Za-z][a-z]'),
    ],
    skip_progress=[],
    strip_symbols=[],
    prompt_patterns=[
        r'^▌',
    ],
    input_start=[';3H'],
    input_mic_replace=';3H🎤',
    input_speaker_replace=';3H📢',
)


OPENCODE_PROFILE = Profile(
    supported=False,
    name='opencode',
    warning="""There's a bit of a performance issue when running with OpenCode due to a lot of redundant processing caused by handling the partial outputs and it noticeably slows down with TalkiTo running. Feel free to flip the supported boolean to True but perhaps don't use the local whisper asr and tts models when using OpenCode.""",
    writes_partial_output=True,
    response_prefix='',
    continuation_prefix=r'^[A-Z][a-z]',
    question_prefix=r'│ Do',
    raw_skip_patterns=[
        r'\[38;5;153m│.*\[38;5;246m\d+',      # Box drawing + line numbers
        r'\[38;5;246m\d+\s*\[39m',            # Direct line numbers
    ],
    replacements=[('█▀▀█ █▀▀█ █▀▀ █▀▀▄ █▀▀ █▀▀█ █▀▀▄ █▀▀', 'OpenCode')],
    exception_patterns=[
        (0, r'│\s*✻ Welcome'),                # Welcome messages
    ],
    skip_patterns=COMMON_SKIP_PATTERNS + [
        # Level 1: Filter unless -v (tips, hints, usage info)
        (1, r'Tip:'),
        (1, r'usage limit'),
        (1, r'to use best available model'),
        (1, r'Update Todos'),

        (2, r'^    '),

        # Level 4: Always filter
        (4, r'^\s*>\s*'),
        (4, r'^v[0-9]'),
        (4, r'^ opencode v'),
        (4, r'^ Build '),
        (4, r'BUILD AGENT'),
    ],
    skip_progress=['Forming', 'Exploring'],
    strip_symbols=['⏺'],
    prompt_patterns=[
        r'^\s*┃',
        r'^\s*│',
        r'^>\s*.+',
        r'^\s*│\s*>\s*',
    ],
    input_start=['│ > ', '┃ >'],
    input_mic_replace='│ 🎤 ',
)

AIDER_PROFILE = Profile(
    supported=False,
    name='aider',
    warning="""Have a problem with how Aider prints its output to the terminal in a non linear fashion.
Unlikely to be able to support Aider any time soon.""",
    skip_patterns=COMMON_SKIP_PATTERNS + [
        (3, r'^Main model'),
        (3, r'^Editor model'),
        (3, r'^Weak model'),
        (3, r'^Git repo'),
        (3, r'^Repo-map'),
        (3, r'^Tokens:'),
        (3, r'^session.'),
    ],
    prompt_patterns=[
        r'^\s*>\s*$',  # Aider uses simple > prompt
    ],
)


OLLAMA_PROFILE = Profile(
    supported=True,
    name='ollama',
    warning="""Warning: There is a bug where you cannot ctr-c to stop an ongoing response. Still investigating this.""",
    needs_full_lines=True,
    skip_patterns=COMMON_SKIP_PATTERNS + [
        (3, r'Ctrl \+ d')
    ],
    prompt_patterns=[
        r'>>>',
    ],
    input_start=['\x1b', '>>>', '>>>'],
)


PIPET_PROFILE = Profile(
    supported=True,
    name='pipet',
    skip_patterns=COMMON_SKIP_PATTERNS + [
        (1, r'^Analyzing package'),
        (1, r'^Package:'),
        (1, r'^Version:'),
        (1, r'^License:'),
    ],
)


PYTHON_PROFILE = Profile(
    supported=True,
    name='python',
    skip_patterns=COMMON_SKIP_PATTERNS,
    prompt_patterns=[
        r'^>>>\s*',  # Python REPL prompt
        r'^\.\.\.\s*',  # Python continuation prompt
    ],
)


MYSQL_PROFILE = Profile(
    supported=True,
    name='mysql',
    skip_patterns=COMMON_SKIP_PATTERNS,
    prompt_patterns=[
        r'^mysql>\s*',  # MySQL prompt
        r'^->\s*',  # MySQL continuation
    ],
)


PSQL_PROFILE = Profile(
    supported=True,
    name='psql',
    skip_patterns=COMMON_SKIP_PATTERNS,
    prompt_patterns=[
        r'^\w+=#\s*',  # PostgreSQL prompt (database=#)
        r'^\w+=-#\s*',  # PostgreSQL continuation
    ],
)


# Default empty profile for when no specific profile is set
DEFAULT_PROFILE = Profile(
    supported=True,
    name='default',
    response_prefix='',
    raw_skip_patterns=[],
    skip_patterns=COMMON_SKIP_PATTERNS,
    speak_patterns=[],
    prompt_patterns=[],
    skip_progress=[],
    strip_symbols=[],
    input_start=[],
    input_mic_replace='',
)

# Profile registry
PROFILES: Dict[str, Profile] = {
    'claude': CLAUDE_PROFILE,
    'codex': CODEX_PROFILE,
    'aider': AIDER_PROFILE,
    'ollama': OLLAMA_PROFILE,
    'pipet': PIPET_PROFILE,
    'python': PYTHON_PROFILE,
    'mysql': MYSQL_PROFILE,
    'psql': PSQL_PROFILE,
    'opencode': OPENCODE_PROFILE,
    'default': DEFAULT_PROFILE,
}


def get_profile(name: str) -> Profile:
    """Get a profile by name, returns default profile if not found"""
    return PROFILES.get(name, DEFAULT_PROFILE)


def register_profile(profile: Profile):
    """Register a new profile"""
    PROFILES[profile.name] = profile


def parse_arguments():
    """Parse command-line arguments for profile testing"""
    parser = argparse.ArgumentParser(
        description='Test talkito profile filtering',
        usage='python -m talkito.profiles [options] [line_to_test]'
    )
    parser.add_argument('--profile', '-p', type=str, default='default',
                       choices=list(PROFILES.keys()),
                       help='Profile to use for testing (default: default)')
    parser.add_argument('--verbosity', '-v', action='count', default=0,
                       help='Increase verbosity (can be used multiple times, up to -vvv)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available profiles')
    parser.add_argument('line', nargs='*', 
                       help='Line of text to test against the profile')
    return parser.parse_args()


def test_line(profile: Profile, line: str, verbosity: int):
    """Test a line against a profile and show the results"""
    print(f"\nTesting line: '{line}'")
    print(f"Profile: {profile.name}, Verbosity: {verbosity}")
    print("-" * 50)
    
    # Test raw skip
    raw_skip_reason = profile.get_raw_skip_reason(line)
    if raw_skip_reason:
        print("✗ RAW SKIP: Line matches raw skip pattern")
        print(f"  - {raw_skip_reason}")
        return
    
    # Test cleaned skip
    skip_reason = profile.get_skip_reason(line, verbosity)
    if skip_reason:
        print(f"✗ SKIP: Line would be skipped at verbosity {verbosity}")
        print(f"  - {skip_reason}")
    else:
        print(f"✓ SPEAK: Line would be spoken at verbosity {verbosity}")
    
    # Additional checks
    if profile.is_continuation_line(line):
        print("  - Is continuation line")
    
    if profile.is_question_line(line):
        print("  - Is question line")
    
    if profile.is_input_start(line):
        print("  - Is input start")
    
    # Test symbol stripping
    stripped = profile.strip_symbols_from(line)
    if stripped != line:
        print(f"  - After symbol stripping: '{stripped}'")


def main():
    """Main entry point for profile testing"""
    args = parse_arguments()
    
    # Handle --list option
    if args.list:
        print("Available profiles:")
        for name in sorted(PROFILES.keys()):
            print(f"  - {name}")
        return
    
    # Get the selected profile
    profile = get_profile(args.profile)
    
    # Test a line if provided
    if args.line:
        line = ' '.join(args.line)
        test_line(profile, line, args.verbosity)
    else:
        # Interactive mode - read from stdin
        print(f"Profile: {args.profile}, Verbosity: {args.verbosity}")
        print("Enter lines to test (Ctrl+D to exit):")
        print("-" * 50)
        
        try:
            while True:
                line = input("> ")
                if line.strip():
                    test_line(profile, line, args.verbosity)
                    print()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")


if __name__ == "__main__":
    main()
