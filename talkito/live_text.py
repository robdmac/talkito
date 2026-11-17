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

"""Live dictation drawn in grey at the input caret, over cells the program underneath doesn't know about."""

import re
import time
from typing import List, Tuple

from .screen import Screen, char_width, sgr

LIVE_FOREGROUND = '90'   # grey, a step back from text the program shows as typed
QUIET_SECONDS = 0.03     # how long the program must stop writing before it's drawn over
# Box and block drawing ends the input field: the live text stops short of a box's edge
FIELD_EDGE = re.compile(r'[─-▟]')


class LiveText:
    """Speech heard but not yet final, shown after the caret until the final words are typed in its place.

    The program never sees these cells change, so whatever was painted over is put back from the virtual
    screen when the text changes or goes, leaving the terminal as the program believes it is.
    """

    def __init__(self):
        self.text = ''
        self.drawn: List[Tuple[int, int]] = []   # cells painted over, to put back
        self.dirty = False
        self.last_output = 0.0

    def set(self, text: str):
        """Show this text in place of what was shown."""
        text = ' '.join(text.split())
        if text != self.text:
            self.text = text
            self.dirty = True

    def on_output(self):
        """The program wrote to the screen, perhaps over the live text or moving the caret."""
        self.last_output = time.monotonic()
        if self.text or self.drawn:
            self.dirty = True

    def frame(self, screen: Screen) -> bytes:
        """What to write to show the text as it is now, once the program has stopped writing."""
        if not self.dirty or not screen.at_rest or time.monotonic() - self.last_output < QUIET_SECONDS:
            return b''
        self.dirty = False
        out = []
        cells = []
        if self.text:
            row, start = screen.cursor_row, screen.cursor_col
            end = start
            # Up to the field's edge, leaving the last column, which would wrap
            while end < screen.cols - 1 and not FIELD_EDGE.match(screen.grid[row][end] or ''):
                end += 1
            shown = self._fit(self.text, end - start)
            out.append(f'\x1b[{row + 1};{start + 1}H')
            col, last_style = start, None
            # The text, then blanks over the rest of the field, so a placeholder doesn't show past it
            for char in list(shown) + [' '] * (end - start):
                width = char_width(char)
                if col + width > end:
                    break
                style = self._grey(screen.styles[row][col])
                if style != last_style:
                    out.append(sgr(style))
                    last_style = style
                out.append(char)
                cells.extend((row, c) for c in range(col, col + width))
                col += width
        kept = set(cells)
        for r, c in self.drawn:
            if (r, c) not in kept and r < screen.rows and c < screen.cols:
                cell = screen.paint(r, c)
                if cell:
                    out.append(f'\x1b[{r + 1};{c + 1}H{cell}')
        self.drawn = cells
        if not out:
            return b''
        # Back to where the program left the cursor, in the style it was writing with
        out.append(f'\x1b[{screen.cursor_row + 1};{screen.cursor_col + 1}H{sgr(screen.style)}')
        return ('\x1b[?2026h' + ''.join(out) + '\x1b[?2026l').encode('utf-8')

    @staticmethod
    def _grey(style):
        """A cell's style with grey text, keeping its background so the field still looks like itself."""
        return (LIVE_FOREGROUND, style[1], '')

    @staticmethod
    def _fit(text: str, width: int) -> str:
        """The text, or its end after an ellipsis when it's too long for the field: the newest words matter."""
        if sum(char_width(c) for c in text) <= width:
            return text
        tail = ''
        for char in reversed(text):
            if sum(char_width(c) for c in tail) + char_width(char) > width - 1:
                break
            tail = char + tail
        return '…' + tail
