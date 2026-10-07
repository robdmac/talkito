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

"""Minimal virtual terminal for TUIs that paint words into place rather than writing lines."""

import codecs
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

# One complete escape sequence at the start of the pending input
CSI_PATTERN = re.compile(r'\x1b\[([?<>=]?)([0-9;:]*)([ -/]*)([@-~])')
# OSC, DCS, APC and PM strings run until BEL or ST
STRING_PATTERN = re.compile(r'\x1b[\]P_^].*?(?:\x07|\x1b\\)', re.DOTALL)
# Time source for settling rows; replays swap in one driven by the bytes fed, so results don't depend
# on how fast the replay happens to run
clock = time.monotonic

# Rows that open a list item are never the tail of a wrapped line
LIST_ITEM_PATTERN = re.compile(r'^\s*(?:[-*•◦▪›>]|\d+[.)])\s')


def char_width(char: str) -> int:
    """Number of cells a character occupies."""
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1


class Screen:
    """A grid of cells that applies cursor addressing, erasing and scrolling."""

    def __init__(self, rows: int, cols: int):
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self.cursor_row = 0
        self.cursor_col = 0
        self.saved_cursor = (0, 0)
        self.scroll_top = 0
        self.scroll_bottom = self.rows - 1
        self.grid = [self._blank_row() for _ in range(self.rows)]
        self.wrap_pending = False
        self.in_update = False   # inside a synchronized update, so the grid is half painted
        self.updates = 0         # synchronized updates completed, a frame count independent of chunking
        self.closed_lines: Optional[List[str]] = None   # what the alternate screen showed as it closed
        self._pending = ''
        self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def _blank_row(self) -> List[str]:
        return [' '] * self.cols

    def resize(self, rows: int, cols: int):
        """Change the grid size, keeping what fits."""
        rows, cols = max(1, rows), max(1, cols)
        grid = [(row + [' '] * cols)[:cols] for row in self.grid[-rows:]]
        while len(grid) < rows:
            grid.append([' '] * cols)
        self.rows, self.cols, self.grid = rows, cols, grid
        self.scroll_top, self.scroll_bottom = 0, rows - 1
        self.cursor_row = min(self.cursor_row, rows - 1)
        self.cursor_col = min(self.cursor_col, cols - 1)

    def lines(self) -> List[str]:
        """The visible text, one string per row with trailing blanks removed."""
        return [''.join(cell for cell in row if cell is not None).rstrip() for row in self.grid]

    def feed(self, data: bytes):
        """Apply a chunk of terminal output, holding back any incomplete sequence."""
        text = self._pending + self._decoder.decode(data)
        self._pending = ''
        i = 0
        length = len(text)
        while i < length:
            char = text[i]
            if char < ' ' or char == '\x7f':
                self.wrap_pending = False
            if char == '\x1b':
                consumed = self._escape(text, i)
                if consumed == 0:
                    self._pending = text[i:]
                    return
                i += consumed
                continue
            if char == '\r':
                self.cursor_col = 0
            elif char in '\n\x0b\x0c':
                self._line_feed()
            elif char == '\x08':
                self.cursor_col = max(0, self.cursor_col - 1)
            elif char == '\t':
                self.cursor_col = min(self.cols - 1, (self.cursor_col // 8 + 1) * 8)
            elif char >= ' ' and char != '\x7f':
                self._put(char)
            i += 1

    def _escape(self, text: str, i: int) -> int:
        """Apply the escape sequence at i and return its length, or 0 if it is incomplete."""
        if i + 1 >= len(text):
            return 0
        kind = text[i + 1]
        if kind == '[':
            match = CSI_PATTERN.match(text, i)
            if not match:
                # Incomplete unless something that cannot belong to a CSI shows up
                tail = text[i + 2:]
                return 0 if re.fullmatch(r'[?<>=]?[0-9;:]*[ -/]*', tail) else 2
            self._csi(match.group(1), match.group(2), match.group(3), match.group(4))
            return match.end() - i
        if kind in ']P_^':
            match = STRING_PATTERN.match(text, i)
            return match.end() - i if match else 0
        if kind in '()*+#%':
            return 3 if i + 2 < len(text) else 0
        if kind == '7':
            self.saved_cursor = (self.cursor_row, self.cursor_col)
        elif kind == '8':
            self.cursor_row, self.cursor_col = self.saved_cursor
        elif kind == 'D':
            self._line_feed()
        elif kind == 'E':
            self.cursor_col = 0
            self._line_feed()
        elif kind == 'M':
            self._reverse_line_feed()
        elif kind == 'c':
            self.__init__(self.rows, self.cols)
        return 2

    def _csi(self, private: str, params: str, intermediate: str, final: str):
        """Apply a control sequence."""
        if intermediate:
            return
        values = [int(p) if p.isdigit() else 0 for p in re.split('[;:]', params)] if params else []

        def arg(index: int = 0, default: int = 1) -> int:
            value = values[index] if index < len(values) else 0
            return value or default

        if private:
            if private == '?' and final in 'hl' and 2026 in values:
                if self.in_update and final == 'l':
                    self.updates += 1
                self.in_update = final == 'h'
            if private == '?' and final in 'hl' and any(v in (47, 1047, 1049) for v in values):
                if final == 'l':
                    self.closed_lines = self.lines()
                # Entering or leaving the alternate screen starts from a blank grid either way
                self.grid = [self._blank_row() for _ in range(self.rows)]
                if final == 'h':
                    self.cursor_row = self.cursor_col = 0
            return

        if final in 'Hf':
            self.cursor_row = min(self.rows - 1, arg(0) - 1)
            self.cursor_col = min(self.cols - 1, arg(1) - 1)
        elif final == 'A':
            self.cursor_row = max(0, self.cursor_row - arg())
        elif final in 'Be':
            self.cursor_row = min(self.rows - 1, self.cursor_row + arg())
        elif final in 'Ca':
            self.cursor_col = min(self.cols - 1, self.cursor_col + arg())
        elif final == 'D':
            self.cursor_col = max(0, self.cursor_col - arg())
        elif final == 'E':
            self.cursor_row = min(self.rows - 1, self.cursor_row + arg())
            self.cursor_col = 0
        elif final == 'F':
            self.cursor_row = max(0, self.cursor_row - arg())
            self.cursor_col = 0
        elif final in 'G`':
            self.cursor_col = min(self.cols - 1, arg() - 1)
        elif final == 'd':
            self.cursor_row = min(self.rows - 1, arg() - 1)
        elif final == 'K':
            row = self.grid[self.cursor_row]
            mode = values[0] if values else 0
            start, end = {0: (self.cursor_col, self.cols), 1: (0, self.cursor_col + 1)}.get(mode, (0, self.cols))
            for col in range(start, end):
                row[col] = ' '
        elif final == 'J':
            mode = values[0] if values else 0
            if mode == 0:
                self._csi('', '0', '', 'K')
                rows = range(self.cursor_row + 1, self.rows)
            elif mode == 1:
                self._csi('', '1', '', 'K')
                rows = range(0, self.cursor_row)
            else:
                rows = range(self.rows)
            for r in rows:
                self.grid[r] = self._blank_row()
        elif final == 'X':
            row = self.grid[self.cursor_row]
            for col in range(self.cursor_col, min(self.cols, self.cursor_col + arg())):
                row[col] = ' '
        elif final == 'P':
            row = self.grid[self.cursor_row]
            del row[self.cursor_col:self.cursor_col + arg()]
            row.extend([' '] * (self.cols - len(row)))
        elif final == '@':
            row = self.grid[self.cursor_row]
            row[self.cursor_col:self.cursor_col] = [' '] * arg()
            del row[self.cols:]
        elif final == 'L' and self.scroll_top <= self.cursor_row <= self.scroll_bottom:
            for _ in range(arg()):
                del self.grid[self.scroll_bottom]
                self.grid.insert(self.cursor_row, self._blank_row())
        elif final == 'M' and self.scroll_top <= self.cursor_row <= self.scroll_bottom:
            for _ in range(arg()):
                del self.grid[self.cursor_row]
                self.grid.insert(self.scroll_bottom, self._blank_row())
        elif final == 'S':
            for _ in range(arg()):
                self._scroll_up()
        elif final == 'T':
            for _ in range(arg()):
                self._scroll_down()
        elif final == 'r':
            top = arg(0) - 1
            bottom = (values[1] if len(values) > 1 and values[1] else self.rows) - 1
            if 0 <= top < bottom < self.rows:
                self.scroll_top, self.scroll_bottom = top, bottom
            self.cursor_row = self.cursor_col = 0

    def _put(self, char: str):
        width = char_width(char)
        if width == 0:
            return
        if self.wrap_pending or self.cursor_col + width > self.cols:
            self.cursor_col = 0
            self._line_feed()
        self.wrap_pending = False
        row = self.grid[self.cursor_row]
        row[self.cursor_col] = char
        if width == 2:
            row[self.cursor_col + 1] = None   # the right half of a wide character
        self.cursor_col += width
        if self.cursor_col >= self.cols:
            # Writing the last column leaves the cursor there; only the next character wraps
            self.cursor_col = self.cols - 1
            self.wrap_pending = True

    def _line_feed(self):
        if self.cursor_row == self.scroll_bottom:
            self._scroll_up()
        else:
            self.cursor_row = min(self.rows - 1, self.cursor_row + 1)

    def _reverse_line_feed(self):
        if self.cursor_row == self.scroll_top:
            self._scroll_down()
        else:
            self.cursor_row = max(0, self.cursor_row - 1)

    def _scroll_up(self):
        del self.grid[self.scroll_top]
        self.grid.insert(self.scroll_bottom, self._blank_row())

    def _scroll_down(self):
        del self.grid[self.scroll_bottom]
        self.grid.insert(self.scroll_top, self._blank_row())


class ScreenTranscript:
    """Turns a repainting TUI's screen into the newline-separated lines an older TUI would have written.

    A row is only released once something has been painted below it in the transcript, so a row that
    is still being streamed word by word is never read half finished. The last row is released when
    it is the turn's footer, or once the screen has sat unchanged for a while.

    The screen is read once per frame: at the end of each synchronized update for TUIs that mark
    them, or after a short pause in output for those that don't (quiet_seconds).
    """

    IDLE_SECONDS = 2.0

    def __init__(self, rows: int, cols: int, composer_pattern: Optional[str] = None,
                 live_pattern: Optional[str] = None, turn_end_pattern: Optional[str] = None,
                 margin: int = 0, settle_frames: int = 0, right_margin: int = 0,
                 quiet_seconds: float = 0.0):
        self.screen = Screen(rows, cols)
        self.composer_pattern = re.compile(composer_pattern) if composer_pattern else None
        self.live_pattern = re.compile(live_pattern) if live_pattern else None
        self.turn_end_pattern = re.compile(turn_end_pattern) if turn_end_pattern else None
        self.margin = margin   # columns of left padding the TUI puts before every transcript row
        self.right_margin = right_margin   # columns it leaves unused at the right before wrapping
        # TUIs that lay out the next row before finishing the one above need rows to sit unchanged
        # for a number of frames before they count as complete
        self.settle_frames = settle_frames
        self.quiet_seconds = quiet_seconds
        self.quiet_frames = 0
        self.unread_output = False
        self.last_output = 0.0
        self.row_since: Dict[str, int] = {}
        self.released: List[str] = []
        self.last_turn_end = ''
        self.held_row = ''
        self.held_since = 0.0
        self._last_flush_check = 0.0

    def resize(self, rows: int, cols: int):
        if (rows, cols) != (self.screen.rows, self.screen.cols):
            self.screen.resize(rows, cols)

    def transcript_rows(self, lines: Optional[List[str]] = None) -> List[str]:
        """Rows above the composer, without the live status row."""
        rows = self.screen.lines() if lines is None else lines
        if self.composer_pattern:
            composer = [i for i, row in enumerate(rows) if self.composer_pattern.search(row)]
            if composer:
                rows = rows[:composer[-1]]
        if self.live_pattern:
            rows = ['' if self.live_pattern.search(row) else row for row in rows]
        return rows

    def _frame(self) -> int:
        return self.quiet_frames if self.quiet_seconds > 0 else self.screen.updates

    def feed(self, data: bytes, now: Optional[float] = None) -> Tuple[List[str], bool]:
        """Apply output; return the transcript rows just completed and whether a turn just ended."""
        now = clock() if now is None else now
        self.screen.feed(data)
        self.last_output = now
        released = self._release_closed()
        if self.quiet_seconds > 0:
            # Read once the output pauses, which is when a frame has been painted in full
            self.unread_output = True
            return released, False
        if self.screen.in_update:
            return released, False
        more, turn_ended = self._read(now)
        return released + more, turn_ended

    def _release_closed(self) -> List[str]:
        """Everything left on an alternate screen as it closed, which nothing will paint over now."""
        if self.screen.closed_lines is None:
            return []
        rows = self.transcript_rows(self.screen.closed_lines)
        self.screen.closed_lines = None
        return self._release(rows)

    def _read(self, now: float) -> Tuple[List[str], bool]:
        """Take a frame from the screen; return the rows it completes and whether a turn just ended."""
        rows = self.transcript_rows()
        frame = self._frame()
        self.row_since = {row: self.row_since.get(row, frame) for row in rows if row.strip()}
        last = max((i for i, row in enumerate(rows) if row.strip()), default=-1)
        if last < 0:
            return [], False

        turn_ended = False
        if self.turn_end_pattern and self.turn_end_pattern.search(rows[last]) and rows[last] != self.last_turn_end:
            self.last_turn_end = rows[last]
            turn_ended = True

        # The last painted row may still be growing; everything above it is complete once settled, or
        # straight away when it is the turn's footer, since nothing above that is still being painted
        released = self._release(rows[:last] if turn_ended else self._settled(rows[:last]))

        if rows[last] != self.held_row:
            self.held_row = rows[last]
            self.held_since = now
        return released, turn_ended

    def _settled(self, complete: List[str]) -> List[str]:
        """The leading rows of complete that have stopped changing."""
        if self.settle_frames <= 0:
            return complete
        frame = self._frame()
        for index, row in enumerate(complete):
            if row.strip() and frame - self.row_since.get(row, frame) < self.settle_frames:
                return complete[:index]
        return complete

    def flush_idle(self, now: Optional[float] = None) -> Tuple[List[str], bool]:
        """Read a frame once output pauses, and release the last row once the screen has gone quiet."""
        final = now == float('inf')
        now = clock() if now is None else now
        # Called on every pass of the read loop, so only look a few times a second
        if not final and now - self._last_flush_check < min(0.1, self.quiet_seconds or 0.1):
            return [], False
        self._last_flush_check = now
        if self.screen.in_update:
            return [], False

        released, turn_ended = [], False
        if self.unread_output and (final or now - self.last_output >= self.quiet_seconds):
            self.unread_output = False
            self.quiet_frames += 1
            released, turn_ended = self._read(now)

        if not self.held_row:
            return released, turn_ended
        rows = self.transcript_rows()
        last = max((i for i, row in enumerate(rows) if row.strip()), default=-1)
        if last < 0:
            return released, turn_ended
        complete = self._settled(rows[:last])
        # Once the screen has gone quiet nothing is still being painted, settled or not
        if now - self.held_since >= self.IDLE_SECONDS and not (self.released and self.released[-1] == self.held_row):
            complete = rows[:last + 1]
        return released + self._release(complete), turn_ended

    def _release(self, complete: List[str]) -> List[str]:
        if not any(row.strip() for row in complete):
            return []
        start = self._new_rows_start(complete)
        fresh = complete[start:]
        self.released.extend(row for row in fresh if row.strip())
        del self.released[:-500]

        # A row carrying on a line the TUI wrapped keeps a single space of indent, which the line
        # reader joins on rather than starting a new sentence
        marked = []
        for index, row in enumerate(fresh, start):
            wrapped = index > 0 and self.is_wrapped(complete[index - 1], row)
            if self.margin and not row[:self.margin].strip():
                row = row[self.margin:]
            if wrapped:
                row = ' ' + row.lstrip()
            marked.append(row)

        # Blank rows only matter between pieces of text
        while marked and not marked[0].strip():
            marked = marked[1:]
        return marked

    def is_wrapped(self, previous: str, row: str) -> bool:
        """Whether row is the overflow of previous, judged by the next word not fitting after it."""
        if not previous.strip() or not row.strip() or LIST_ITEM_PATTERN.match(row):
            return False
        first_word = row.split()[0]
        # talkito gives the program two columns fewer than the terminal, and it wraps at its own width
        return len(previous) + 1 + len(first_word) > self.screen.cols - 3 - self.right_margin

    def _new_rows_start(self, complete: List[str]) -> int:
        """Index in complete of the first row not yet released, aligning on what was released last."""
        released = self.released
        if not released:
            return 0
        painted = [i for i, row in enumerate(complete) if row.strip()]
        # Find the latest place the tail of what was released appears on screen
        for size in range(min(3, len(released)), 0, -1):
            tail = released[-size:]
            for k in range(len(painted) - size, -1, -1):
                if [complete[j] for j in painted[k:k + size]] == tail:
                    return painted[k + size - 1] + 1
        # Nothing matches, so the screen has moved on entirely; skip rows released recently
        recent = set(released[-200:])
        for index in painted:
            if complete[index] not in recent:
                return index
        return len(complete)
