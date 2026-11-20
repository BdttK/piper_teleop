"""termios 单键读取(直接在终端里用,无需 DISPLAY)。"""

from __future__ import annotations

import select
import sys
import termios
import tty

_KEYMAP = {
    " ": "space",
    "\r": "space",
    "\n": "space",
    "\x7f": "backspace",
    "\x08": "backspace",
    "\x1b": "esc",
    "\x03": "ctrl-c",
}


class KeyReader:
    def __init__(self):
        self._is_tty = sys.stdin.isatty()
        self._old_attrs = None

    def __enter__(self) -> "KeyReader":
        if self._is_tty:
            self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc_info) -> None:
        if self._old_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def poll(self, timeout: float = 0.2) -> str | None:
        if not self._is_tty:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        ch = sys.stdin.read(1)
        return _KEYMAP.get(ch, ch)
