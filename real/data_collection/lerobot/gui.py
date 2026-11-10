"""Standalone LeRobot Chinese GUI with an interactive PTY terminal."""

from __future__ import annotations

import argparse
import codecs
import errno
import fcntl
import html
import json
import os
import pty
import select
import shlex
import signal
import struct
import subprocess
import sys
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PyQt5.QtCore import QEvent, QSignalBlocker, QObject, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QFontDatabase, QIntValidator, QTextCursor, QKeyEvent
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Runtime parameter model

@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    section: str
    kind: str = "text"
    default: Any = ""
    modes: frozenset[str] | None = None
    choices: tuple[str, ...] = ()
    placeholder: str = ""
    readonly: bool = False
    optional: bool = False


# The GUI owns this runtime command model.
FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("robot.type", "Robot type", "Task", "text", "franka"),
    FieldSpec("control.root", "Output directory", "Task", "path", ""),
    FieldSpec("control.num_episodes", "Episodes", "Recording", "int", "40"),
    FieldSpec("control.reset_time_s", "Reset seconds", "Recording", "float", "0"),
    FieldSpec("control.resume", "Resume", "Saving", "bool", False),
    FieldSpec("control.fps", "Frame rate", "Saving", "int", "30"),
)

FIELD_BY_KEY = {spec.key: spec for spec in FIELD_SPECS}
DEFAULT_VALUES = {spec.key: spec.default for spec in FIELD_SPECS}

FIELD_DESCRIPTIONS: dict[str, str] = {
    "robot.type": "This UI does not expose low-level settings such as IPs or joint limits.",
    "control.type": "Always uses the record control mode: the program collects "
    "and saves the dataset following plan.csv.",
    "control.fps": "Target frame rate for the control loop and data collection. "
    "Higher values put more load on the cameras, disk and robot communication.",
    "control.root": "Local directory where the dataset is saved (required), "
    "e.g. outputs/date_id.",
    "control.resume": "Whether to resume recording on top of an existing dataset.",
    "control.reset_time_s": "Time (seconds) between episodes for the operator "
    "to reset the scene.",
    "control.num_episodes": "Number of episodes to record; recording stops once "
    "the count is reached.",
}


def _value_for_cli(spec: FieldSpec, value: Any) -> str:
    return ("true" if bool(value) else "false") if spec.kind == "bool" else str(value).strip()


def build_cli_args(values: dict[str, Any]) -> list[str]:
    """Build only the parameters owned by this runtime GUI."""

    args: list[str] = [
        "--control.type=record",
        "--control.single_task=DUMMY",  # overwritten by the plan.csv action column
        '--control.tags=["tutorial"]',
        "--control.repo_id=brave/bananas",
        "--control.warmup_time_s=2",
        "--control.episode_time_s=-1",
        "--control.push_to_hub=false",
        "--control.num_image_writer_threads_per_camera=8",
        "--control.video=false",
    ]
    for spec in FIELD_SPECS:
        value = values.get(spec.key, spec.default)
        if spec.optional and spec.kind == "bool" and not bool(value):
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        args.append(f"--{spec.key}={_value_for_cli(spec, value)}")
    return args


def build_command(values: dict[str, Any], project_root: Path) -> str:
    """Build the complete command sent directly to the interactive PTY."""

    project_root = project_root.expanduser().resolve()
    script_relative = f"{project_root.name}/scripts/control_robot.py"
    args = " ".join(shlex.quote(item) for item in build_cli_args(values))
    return (
        f"cd {shlex.quote(str(project_root.parent))} && "
        f"PYTHONPATH=\"$(pwd)\" python {shlex.quote(script_relative)} {args}"
    )


def format_preview(values: dict[str, Any], project_root: Path, width: int = 96) -> str:
    command = build_command(values, project_root)
    parts = command.split(" && ")
    return f"{parts[0]} &&\n{parts[1]}" if len(parts) == 2 else command


def known_fields() -> Iterable[FieldSpec]:
    return FIELD_SPECS


def describe_field(spec: FieldSpec) -> str:
    return FIELD_DESCRIPTIONS[spec.key]


# Inherited launch environment

def runtime_environment_label(environ: Mapping[str, str] | None = None) -> str:
    """Return a short label without scanning or changing Conda environments."""

    values = os.environ if environ is None else environ
    name = values.get("CONDA_DEFAULT_ENV", "").strip()
    if name:
        return name
    prefix = values.get("CONDA_PREFIX", "").strip()
    if prefix:
        return Path(prefix).name or "Current Conda environment"
    return "Current shell environment"


def inherited_shell_command(environ: Mapping[str, str] | None = None) -> str:
    """Start an interactive Bash that inherits the GUI launch environment."""

    values = os.environ if environ is None else environ
    lines = [
        "export TERM=xterm-256color",
        "export COLORTERM=truecolor",
        "export LANG=${LANG:-C.UTF-8}",
    ]
    environment_name = values.get("CONDA_DEFAULT_ENV", "").strip()
    if environment_name:
        prompt = f"({environment_name}) \\u@\\h:\\w\\$ "
        lines.append(f"export PS1={shlex.quote(prompt)}")
    lines.append("exec /bin/bash -i")
    return "; ".join(lines)


# PTY and terminal widget

class TerminalBuffer:
    """Render the useful subset of ANSI terminal control sequences locally."""

    def __init__(self, columns: int = 120, rows: int = 36, history_limit: int = 3000) -> None:
        self.columns = max(20, columns)
        self.rows = max(8, rows)
        self.history_limit = history_limit
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._saved_cursor = (0, 0)
        self._state = "normal"
        self._csi = ""
        self._osc = ""
        self.history: list[str] = []
        self.lines: list[list[str]] = []
        self.row = 0
        self.column = 0
        self.reset()

    def _blank_line(self) -> list[str]:
        return [" "] * self.columns

    def reset(self) -> None:
        self.history.clear()
        self.lines = [self._blank_line() for _ in range(self.rows)]
        self.row = 0
        self.column = 0
        self._saved_cursor = (0, 0)
        self._state = "normal"
        self._csi = ""
        self._osc = ""

    def clear(self) -> None:
        self.reset()

    def resize(self, columns: int, rows: int) -> None:
        columns = max(20, columns)
        rows = max(8, rows)
        if columns != self.columns:
            for line in self.lines:
                if columns > self.columns:
                    line.extend([" "] * (columns - self.columns))
                else:
                    del line[columns:]
            self.columns = columns
            self.column = min(self.column, columns - 1)

        if rows < self.rows:
            remove_count = self.rows - rows
            self.history.extend("".join(self.lines[i]).rstrip() for i in range(remove_count))
            self.history = self.history[-self.history_limit :]
            self.lines = self.lines[remove_count:]
            self.row = max(0, self.row - remove_count)
        elif rows > self.rows:
            self.lines.extend(self._blank_line() for _ in range(rows - self.rows))
        self.rows = rows

    def _scroll_if_needed(self) -> None:
        if self.row < self.rows:
            return
        self.history.append("".join(self.lines.pop(0)).rstrip())
        self.history = self.history[-self.history_limit :]
        self.lines.append(self._blank_line())
        self.row = self.rows - 1

    def _line_feed(self) -> None:
        self.row += 1
        self._scroll_if_needed()

    def _put(self, char: str) -> None:
        if self.column >= self.columns:
            self.column = 0
            self._line_feed()
        self.lines[self.row][self.column] = char
        self.column += 1

    def _erase_line(self, mode: int) -> None:
        line = self.lines[self.row]
        if mode == 2:
            start, end = 0, self.columns
        elif mode == 1:
            start, end = 0, min(self.columns, self.column + 1)
        else:
            start, end = min(self.columns, self.column), self.columns
        for index in range(start, end):
            line[index] = " "

    def _erase_screen(self, mode: int) -> None:
        if mode == 2 or mode == 3:
            self.lines = [self._blank_line() for _ in range(self.rows)]
            self.history.clear()
            return
        if mode == 1:
            for index in range(self.row + 1):
                self.lines[index] = self._blank_line()
            for index in range(self.column + 1):
                self.lines[self.row][index] = " "
            return
        self._erase_line(0)
        for index in range(self.row + 1, self.rows):
            self.lines[index] = self._blank_line()

    @staticmethod
    def _params(raw: str) -> list[int]:
        raw = raw.lstrip("?=>")
        if not raw:
            return [1]
        result: list[int] = []
        for part in raw.split(";"):
            try:
                result.append(int(part) if part else 1)
            except ValueError:
                result.append(1)
        return result or [1]

    def _handle_csi(self, raw: str, final: str) -> None:
        params = self._params(raw)
        first = params[0] if params else 1
        second = params[1] if len(params) > 1 else 1
        if final == "A":
            self.row = max(0, self.row - first)
        elif final == "B":
            self.row = min(self.rows - 1, self.row + first)
        elif final == "C":
            self.column = min(self.columns - 1, self.column + first)
        elif final == "D":
            self.column = max(0, self.column - first)
        elif final in {"H", "f"}:
            self.row = max(0, min(self.rows - 1, first - 1))
            self.column = max(0, min(self.columns - 1, second - 1))
        elif final == "G":
            self.column = max(0, min(self.columns - 1, first - 1))
        elif final == "d":
            self.row = max(0, min(self.rows - 1, first - 1))
        elif final == "J":
            self._erase_screen(0 if raw == "" else first)
        elif final == "K":
            self._erase_line(0 if raw == "" else first)
        elif final == "s":
            self._saved_cursor = (self.row, self.column)
        elif final == "u":
            self.row, self.column = self._saved_cursor
        elif final == "P":
            count = max(1, first)
            line = self.lines[self.row]
            del line[self.column : min(self.columns, self.column + count)]
            line.extend([" "] * (self.columns - len(line)))
        elif final == "@":
            count = max(1, first)
            line = self.lines[self.row]
            line[self.column : self.column] = [" "] * count
            del line[self.columns :]
        elif final == "X":
            count = max(1, first)
            for index in range(self.column, min(self.columns, self.column + count)):
                self.lines[self.row][index] = " "
        elif final == "c":
            self.reset()
        # SGR color/style codes and private modes are intentionally ignored.

    def _handle_escape(self, char: str) -> None:
        if char == "[":
            self._state = "csi"
            self._csi = ""
        elif char == "]":
            self._state = "osc"
            self._osc = ""
        elif char == "7":
            self._saved_cursor = (self.row, self.column)
            self._state = "normal"
        elif char == "8":
            self.row, self.column = self._saved_cursor
            self._state = "normal"
        elif char == "D":
            self._line_feed()
            self._state = "normal"
        elif char == "M":
            if self.row == 0:
                self.lines.insert(0, self._blank_line())
                self.lines.pop()
            else:
                self.row -= 1
            self._state = "normal"
        elif char == "E":
            self.column = 0
            self._line_feed()
            self._state = "normal"
        elif char == "c":
            self.reset()
        else:
            self._state = "normal"

    def feed(self, data: bytes) -> None:
        for char in self._decoder.decode(data):
            if self._state == "osc":
                if char == "\x07":
                    self._state = "normal"
                elif char == "\x1b":
                    self._state = "osc_end"
                else:
                    self._osc += char
                continue
            if self._state == "osc_end":
                self._state = "normal" if char == "\\" else "osc"
                continue
            if self._state == "csi":
                if "@" <= char <= "~":
                    self._handle_csi(self._csi, char)
                    self._state = "normal"
                elif char == "\x1b":
                    self._state = "escape"
                else:
                    self._csi += char
                continue
            if self._state == "escape":
                self._handle_escape(char)
                continue
            if char == "\x1b":
                self._state = "escape"
            elif char in {"\x00", "\x0e", "\x0f", "\x07"}:
                continue
            elif char == "\r":
                self.column = 0
            elif char == "\n":
                self._line_feed()
            elif char == "\b":
                self.column = max(0, self.column - 1)
            elif char == "\t":
                spaces = 8 - (self.column % 8)
                for _ in range(spaces):
                    self._put(" ")
            elif ord(char) >= 32:
                self._put(char)

    def render(self) -> str:
        rows = self.history + ["".join(line).rstrip() for line in self.lines]
        while rows and rows[-1] == "":
            rows.pop()
        return "\n".join(rows)


class PtyReader(QThread):
    data_received = pyqtSignal(bytes)
    closed = pyqtSignal()
    read_error = pyqtSignal(str)

    def __init__(self, master_fd: int) -> None:
        super().__init__()
        self.master_fd = master_fd
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            while not self._stop_requested:
                ready, _, _ = select.select([self.master_fd], [], [], 0.15)
                if not ready:
                    continue
                try:
                    data = os.read(self.master_fd, 8192)
                except OSError as exc:
                    if exc.errno not in {errno.EIO, errno.EBADF} and not self._stop_requested:
                        self.read_error.emit(str(exc))
                    break
                if not data:
                    break
                self.data_received.emit(data)
        except (OSError, ValueError) as exc:
            if not self._stop_requested:
                self.read_error.emit(str(exc))
        finally:
            self.closed.emit()


class PtySession(QObject):
    """Own a shell connected to a PTY and expose bidirectional signals."""

    output = pyqtSignal(bytes)
    started = pyqtSignal()
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, cwd: Path, shell_command: str) -> None:
        super().__init__()
        self.cwd = cwd.expanduser().resolve()
        self.shell_command = shell_command
        self.master_fd: int | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: PtyReader | None = None
        self._finished = False

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.process is not None:
            return
        if not self.cwd.exists():
            self.error.emit(f"Working directory does not exist: {self.cwd}")
            return
        master_fd, slave_fd = pty.openpty()
        env = os.environ.copy()
        env.update(
            {
                "TERM": "xterm-256color",
                "COLORTERM": "truecolor",
                "PYTHONUNBUFFERED": "1",
            }
        )
        try:
            self.process = subprocess.Popen(
                ["/bin/bash", "-i", "-c", self.shell_command],
                cwd=str(self.cwd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            os.close(master_fd)
            os.close(slave_fd)
            self.error.emit(f"Failed to start PTY: {exc}")
            return
        finally:
            try:
                os.close(slave_fd)
            except OSError:
                pass

        self.master_fd = master_fd
        self.reader = PtyReader(master_fd)
        self.reader.data_received.connect(self.output.emit)
        self.reader.read_error.connect(self.error.emit)
        self.reader.closed.connect(self._on_reader_closed)
        self.reader.start()
        self.resize(120, 36)
        self.started.emit()

    def _on_reader_closed(self) -> None:
        if self.process is None or self._finished:
            return
        try:
            return_code = self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._send_signal(signal.SIGKILL)
            return_code = self.process.wait()
        self._finished = True
        self.finished.emit(int(return_code))

    def send_bytes(self, data: bytes) -> None:
        if self.master_fd is None or not data:
            return
        try:
            os.write(self.master_fd, data)
        except OSError as exc:
            if exc.errno not in {errno.EIO, errno.EBADF}:
                self.error.emit(str(exc))

    def send_text(self, text: str) -> None:
        self.send_bytes(text.encode("utf-8", errors="replace"))

    def send_command(self, command: str) -> None:
        self.send_text(command + "\r")

    def _send_signal(self, signum: signal.Signals) -> None:
        if self.process is None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signum)
        except (OSError, ProcessLookupError):
            try:
                self.process.send_signal(signum)
            except OSError:
                pass

    def interrupt(self) -> None:
        self.send_bytes(b"\x03")

    def resize(self, columns: int, rows: int) -> None:
        if self.master_fd is None:
            return
        try:
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
            self._send_signal(signal.SIGWINCH)
        except OSError:
            pass

    def stop(self) -> None:
        if self.process is None:
            return
        if self.reader is not None:
            self.reader.stop()
        self._send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            self._send_signal(signal.SIGKILL)
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        if self.reader is not None and self.reader.isRunning():
            self.reader.wait(1000)
        if not self._finished:
            self._finished = True
            self.finished.emit(int(self.process.returncode or 0))


class TerminalView(QPlainTextEdit):
    """Interactive terminal surface with familiar keyboard controls."""

    bytes_written = pyqtSignal(bytes)
    resized = pyqtSignal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSize(18)
        self.setFont(fixed_font)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setTabStopWidth(self.fontMetrics().horizontalAdvance(" ") * 4)
        self.setCursorWidth(2)
        self.setFocusPolicy(Qt.StrongFocus)
        self.buffer = TerminalBuffer(history_limit=10000)
        self.setPlaceholderText("Connecting to the terminal...")

    def feed(self, data: bytes) -> None:
        scrollbar = self.verticalScrollBar()
        follow_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        previous_value = scrollbar.value()
        self.buffer.feed(data)
        self.setPlainText(self.buffer.render())
        if follow_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(previous_value, scrollbar.maximum()))

    def clear_terminal(self) -> None:
        self.buffer.clear()
        self.clear()

    def copy_all(self) -> None:
        QApplication.clipboard().setText(self.toPlainText())

    def paste_clipboard(self) -> None:
        self._send(QApplication.clipboard().text().encode("utf-8", errors="replace"))

    def _send(self, data: bytes) -> None:
        self.bytes_written.emit(data)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        modifiers = event.modifiers()
        ctrl = bool(modifiers & Qt.ControlModifier)
        alt = bool(modifiers & Qt.AltModifier)
        shift = bool(modifiers & Qt.ShiftModifier)

        if ctrl:
            control_code: bytes | None = None
            if Qt.Key_A <= key <= Qt.Key_Z:
                control_code = bytes([key - Qt.Key_A + 1])
            else:
                control_code = {
                    Qt.Key_Space: b"\x00",
                    Qt.Key_BracketLeft: b"\x1b",
                    Qt.Key_Backslash: b"\x1c",
                    Qt.Key_BracketRight: b"\x1d",
                    Qt.Key_AsciiCircum: b"\x1e",
                    Qt.Key_Underscore: b"\x1f",
                }.get(key)
            if control_code is not None:
                self._send(b"\x1b" + control_code if alt else control_code)
                return

        arrow_keys = {
            Qt.Key_Up: "A",
            Qt.Key_Down: "B",
            Qt.Key_Right: "C",
            Qt.Key_Left: "D",
        }
        if key in arrow_keys:
            modifier_code = 1 + (1 if shift else 0) + (2 if alt else 0) + (4 if ctrl else 0)
            direction = arrow_keys[key]
            sequence = f"\x1b[1;{modifier_code}{direction}" if modifier_code != 1 else f"\x1b[{direction}"
            self._send(sequence.encode("ascii"))
            return

        if key == Qt.Key_Tab:
            self._send(b"\x1b[Z" if shift else b"\t")
            return

        special_keys = {
            Qt.Key_Return: b"\r",
            Qt.Key_Enter: b"\r",
            Qt.Key_Backspace: b"\x7f",
            Qt.Key_Delete: b"\x1b[3~",
            Qt.Key_Home: b"\x1b[H",
            Qt.Key_End: b"\x1b[F",
            Qt.Key_Escape: b"\x1b",
            Qt.Key_Insert: b"\x1b[2~",
            Qt.Key_PageUp: b"\x1b[5~",
            Qt.Key_PageDown: b"\x1b[6~",
        }
        function_keys = {
            Qt.Key_F1: b"\x1bOP",
            Qt.Key_F2: b"\x1bOQ",
            Qt.Key_F3: b"\x1bOR",
            Qt.Key_F4: b"\x1bOS",
            Qt.Key_F5: b"\x1b[15~",
            Qt.Key_F6: b"\x1b[17~",
            Qt.Key_F7: b"\x1b[18~",
            Qt.Key_F8: b"\x1b[19~",
            Qt.Key_F9: b"\x1b[20~",
            Qt.Key_F10: b"\x1b[21~",
            Qt.Key_F11: b"\x1b[23~",
            Qt.Key_F12: b"\x1b[24~",
        }
        for number, sequence in {
            13: b"\x1b[25~",
            14: b"\x1b[26~",
            15: b"\x1b[28~",
            16: b"\x1b[29~",
            17: b"\x1b[31~",
            18: b"\x1b[32~",
            19: b"\x1b[33~",
            20: b"\x1b[34~",
        }.items():
            qt_key = getattr(Qt, f"Key_F{number}", None)
            if qt_key is not None:
                function_keys[qt_key] = sequence

        if key in function_keys:
            self._send(b"\x1b" + function_keys[key] if alt else function_keys[key])
            return
        if key in special_keys:
            self._send(b"\x1b" + special_keys[key] if alt else special_keys[key])
            return

        text = event.text()
        if not text and Qt.Key_A <= key <= Qt.Key_Z:
            text = chr(key)
        if text:
            encoded = text.encode("utf-8", errors="replace")
            self._send(b"\x1b" + encoded if alt or modifiers & Qt.MetaModifier else encoded)
            return

        # Do not let Qt's read-only text widget swallow an otherwise valid
        # key event. Unknown keys have no universal xterm representation, but
        # sending an escape prefix preserves the terminal's meta-key behavior.
        if alt or modifiers & Qt.MetaModifier:
            self._send(b"\x1b")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        char_width = max(1, self.fontMetrics().horizontalAdvance("M"))
        line_height = max(1, self.fontMetrics().lineSpacing())
        columns = max(20, self.viewport().width() // char_width)
        rows = max(8, self.viewport().height() // line_height)
        self.buffer.resize(columns, rows)
        self.resized.emit(columns, rows)


# Main window

STYLE = """
* {
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif;
    color: #18212b;
}
QMainWindow, QWidget {
    background: #ffffff;
}
#header {
    background: #ffffff;
    border-bottom: 1px solid #e6e9ee;
}
#brandTitle {
    color: #101828;
    font-size: 21px;
    font-weight: 700;
}
#brandSubtitle {
    color: #7a8694;
    font-size: 12px;
}
#statusText {
    color: #52606d;
    font-size: 12px;
}
#statusDot {
    color: #94a3b8;
    font-size: 15px;
}
#statusDot[status="ready"] { color: #16a34a; }
#statusDot[status="running"] { color: #2563eb; }
#statusDot[status="warning"] { color: #d97706; }
#statusDot[status="error"] { color: #dc2626; }
#envBadge {
    color: #365314;
    background: #ecfccb;
    border: 1px solid #bef264;
    border-radius: 5px;
    padding: 5px 9px;
    font-size: 12px;
}
#sidebar {
    background: #f7f8fa;
    border-bottom: 1px solid #e6e9ee;
}
#configSection {
    background: #ffffff;
    border: 1px solid #e3e7ed;
    border-radius: 8px;
}
#editHint {
    color: #9a6700;
    background: #fff8e6;
    border: 1px solid #f5d48a;
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 11px;
}
#sidebarHeading {
    color: #101828;
    font-size: 15px;
    font-weight: 700;
}
#muted {
    color: #7a8694;
    font-size: 12px;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #e3e7ed;
    border-radius: 8px;
    margin-top: 11px;
    padding: 14px 10px 10px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #52606d;
    font-size: 12px;
    font-weight: 700;
    background: #ffffff;
}
QLabel.fieldLabel {
    color: #52606d;
    font-size: 12px;
}
QLabel.fieldLabel:disabled {
    color: #a9b3bf;
}
QLineEdit, QComboBox, QPlainTextEdit {
    min-height: 28px;
    border: 1px solid #d5dce5;
    border-radius: 6px;
    padding: 3px 8px;
    background: #ffffff;
    selection-background-color: #bfdbfe;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #4d8bff;
}
QLineEdit:disabled, QComboBox:disabled {
    background: #f1f3f5;
    color: #9aa5b1;
}
QLineEdit:read-only {
    background: #f5f7fa;
    color: #667085;
}
QPlainTextEdit:focus {
    border: 1px solid #4d8bff;
}
QPlainTextEdit[locked="true"] {
    background: #f5f7fa;
    color: #667085;
}
QCheckBox {
    spacing: 7px;
    color: #52606d;
    min-height: 28px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #c6ced8;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
}
QCheckBox[locked="true"] {
    color: #98a2b3;
}
QCheckBox[locked="true"]::indicator {
    background: #f5f7fa;
    border-color: #d0d5dd;
}
QPushButton, QToolButton {
    min-height: 30px;
    border: 1px solid #d5dce5;
    border-radius: 6px;
    padding: 4px 10px;
    background: #ffffff;
    color: #344054;
}
QPushButton:hover, QToolButton:hover {
    background: #f2f6fb;
    border-color: #a9b8ca;
}
QPushButton:pressed, QToolButton:pressed {
    background: #e7eef8;
}
#primaryButton {
    background: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
    font-weight: 700;
}
#primaryButton:hover { background: #1d4ed8; }
#dangerButton {
    color: #b42318;
    border-color: #f0b4ae;
    background: #fffafa;
}
#dangerButton:hover { background: #fff1f0; }
#terminalHeader {
    background: #ffffff;
    border-bottom: 1px solid #e6e9ee;
}
#terminalTitle {
    color: #101828;
    font-size: 14px;
    font-weight: 700;
}
#terminalMeta {
    color: #7a8694;
    font-size: 12px;
}
#terminal {
    background: #111827;
    color: #dbeafe;
    border: 0;
    border-radius: 0;
    padding: 12px;
    selection-background-color: #334155;
    selection-color: #f8fafc;
}
#commandPreview {
    background: #f8fafc;
    border: 1px solid #e3e7ed;
    border-radius: 7px;
    color: #334155;
    padding: 8px;
}
#helpTitle {
    color: #344054;
    font-size: 12px;
    font-weight: 700;
}
#helpBrowser {
    background: #ffffff;
    border: 1px solid #e3e7ed;
    border-radius: 7px;
    color: #52606d;
    padding: 4px;
}
#footerText {
    color: #7a8694;
    font-size: 12px;
}
QScrollArea { border: 0; background: transparent; }
QSplitter::handle { background: #e6e9ee; }
"""


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root.expanduser().resolve()
        self.values: dict[str, Any] = dict(DEFAULT_VALUES)
        self.session: PtySession | None = None
        self.runtime_environment = runtime_environment_label()
        self.field_widgets: dict[str, QWidget] = {}
        self.field_labels: dict[str, QLabel] = {}
        self.field_rows: dict[str, QWidget] = {}
        self.locked_fields: set[str] = set(FIELD_BY_KEY)
        self._last_checkbox_press: dict[str, float] = {}

        self.setWindowTitle("LeRobot Console")
        self.setMinimumSize(1120, 720)
        self.resize(1420, 900)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._populate_values()
        self._update_command_preview()
        self._update_mode_visibility()
        QTimer.singleShot(80, self._start_terminal)

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("central")
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(70)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 0, 20, 0)
        header_layout.setSpacing(12)

        brand = QVBoxLayout()
        brand.setSpacing(1)
        title = QLabel("LeRobot Console")
        title.setObjectName("brandTitle")
        subtitle = QLabel("Parameters · Environment · Interactive terminal")
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header_layout.addLayout(brand)
        header_layout.addStretch(1)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setProperty("status", "warning")
        self.status_text = QLabel("Connecting to the terminal")
        self.status_text.setObjectName("statusText")
        header_layout.addWidget(self.status_dot)
        header_layout.addWidget(self.status_text)
        self.env_badge = QLabel(self.runtime_environment)
        self.env_badge.setObjectName("envBadge")
        header_layout.addWidget(self.env_badge)
        restart = QPushButton("Restart terminal")
        restart.setToolTip("Close the current PTY and reopen Bash with the environment the GUI was started with")
        restart.clicked.connect(self._restart_terminal)
        header_layout.addWidget(restart)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        root_layout.addWidget(splitter, 1)

        sidebar = self._build_sidebar()
        sidebar.setObjectName("sidebar")
        splitter.addWidget(sidebar)
        splitter.addWidget(self._build_terminal_area())
        splitter.setSizes([380, 490])

        footer = QFrame()
        footer.setFixedHeight(30)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 0, 20, 0)
        self.footer_text = QLabel("Ready")
        self.footer_text.setObjectName("footerText")
        footer_layout.addWidget(self.footer_text)
        footer_layout.addStretch(1)
        hint = QLabel("Terminal shortcuts are handled by the PTY · Ctrl+C interrupts the current command")
        hint.setObjectName("footerText")
        footer_layout.addWidget(hint)
        root_layout.addWidget(footer)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebar")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 12, 18, 10)
        layout.setSpacing(8)

        heading_row = QHBoxLayout()
        heading = QLabel("Run configuration")
        heading.setObjectName("sidebarHeading")
        heading_row.addWidget(heading)
        edit_hint = QLabel("Locked by default · double-click a field to edit")
        edit_hint.setObjectName("editHint")
        edit_hint.setToolTip("A single click cannot change a value; double-click a field to unlock it")
        heading_row.addWidget(edit_hint)
        heading_row.addStretch(1)
        layout.addLayout(heading_row)

        columns = QHBoxLayout()
        columns.setSpacing(10)

        left = QVBoxLayout()
        left.setSpacing(7)
        left.addWidget(self._build_parameter_section("Task"))
        left.addStretch(1)
        columns.addLayout(left, 1)

        middle = QVBoxLayout()
        middle.setSpacing(7)
        middle.addWidget(self._build_parameter_section("Recording"))
        middle.addStretch(1)
        columns.addLayout(middle, 2)

        right = QVBoxLayout()
        right.setSpacing(7)
        right.addWidget(self._build_parameter_section("Saving"))
        right.addStretch(1)
        columns.addLayout(right, 1)

        help_column = QVBoxLayout()
        help_column.setSpacing(5)
        self.help_title = QLabel("Parameter help")
        self.help_title.setObjectName("helpTitle")
        help_column.addWidget(self.help_title)
        self.help_browser = QTextBrowser()
        self.help_browser.setObjectName("helpBrowser")
        self.help_browser.setOpenExternalLinks(False)
        self.help_browser.setMinimumWidth(245)
        self.help_browser.setMinimumHeight(145)
        help_column.addWidget(self.help_browser, 1)

        columns.addLayout(help_column, 1)

        layout.addLayout(columns, 1)
        return panel

    def _build_parameter_section(self, section: str) -> QWidget:
        group = QFrame()
        group.setObjectName("configSection")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 7, 10, 8)
        layout.setSpacing(3)
        title = QLabel(section)
        title.setObjectName("muted")
        layout.addWidget(title)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(3)
        for spec in self._field_specs_in_order():
            if spec.section != section:
                continue
            label, widget, row = self._make_field(spec)
            form.addRow(label, row)
            self.field_widgets[spec.key] = widget
            self.field_labels[spec.key] = label
            self.field_rows[spec.key] = row
        layout.addLayout(form)
        return group

    def _build_terminal_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(16, 14, 18, 8)
        layout.setSpacing(9)

        preview_row = QHBoxLayout()
        preview_label = QLabel("Terminal")
        preview_label.setObjectName("terminalTitle")
        preview_row.addWidget(preview_label)
        preview_row.addStretch(1)
        preview_button = QPushButton("View command")
        preview_button.setCheckable(True)
        preview_button.setToolTip("Show the command generated from the current parameters")
        preview_row.addWidget(preview_button)
        layout.addLayout(preview_row)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setObjectName("commandPreview")
        self.command_preview.setReadOnly(True)
        self.command_preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.command_preview.setFixedHeight(92)
        preview_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        preview_font.setPointSize(9)
        self.command_preview.setFont(preview_font)
        self.command_preview.setVisible(False)
        preview_button.toggled.connect(self.command_preview.setVisible)
        layout.addWidget(self.command_preview)

        terminal_header = QFrame()
        terminal_header.setObjectName("terminalHeader")
        terminal_header_layout = QHBoxLayout(terminal_header)
        terminal_header_layout.setContentsMargins(0, 0, 0, 5)
        terminal_title = QLabel("Interactive terminal")
        terminal_title.setObjectName("terminalTitle")
        terminal_header_layout.addWidget(terminal_title)
        self.terminal_meta = QLabel("PTY · Bash")
        self.terminal_meta.setObjectName("terminalMeta")
        terminal_header_layout.addWidget(self.terminal_meta)
        terminal_header_layout.addStretch(1)
        self.terminal = TerminalView()
        self.terminal.setObjectName("terminal")
        self.terminal.bytes_written.connect(self._write_terminal_bytes)
        self.terminal.resized.connect(self._resize_terminal)
        clear_button = QPushButton("Clear")
        clear_button.setToolTip("Clear the terminal display history without stopping the running process")
        clear_button.clicked.connect(lambda: self.terminal.clear_terminal())
        bottom_button = QPushButton("Scroll to bottom")
        bottom_button.clicked.connect(lambda: self.terminal.verticalScrollBar().setValue(self.terminal.verticalScrollBar().maximum()))
        copy_button = QPushButton("Copy all")
        copy_button.clicked.connect(self.terminal.copy_all)
        paste_button = QPushButton("Paste")
        paste_button.clicked.connect(self.terminal.paste_clipboard)
        interrupt_button = QPushButton("Interrupt")
        interrupt_button.setObjectName("dangerButton")
        interrupt_button.setToolTip("Send Ctrl+C to the current PTY without closing the terminal")
        interrupt_button.clicked.connect(self._interrupt_command)
        start_button = QPushButton("▶  Run command")
        start_button.setObjectName("primaryButton")
        start_button.setToolTip("Send the command generated from the current parameters to the interactive terminal")
        start_button.clicked.connect(self._run_command)
        terminal_header_layout.addWidget(clear_button)
        terminal_header_layout.addWidget(bottom_button)
        terminal_header_layout.addWidget(copy_button)
        terminal_header_layout.addWidget(paste_button)
        terminal_header_layout.addWidget(interrupt_button)
        terminal_header_layout.addWidget(start_button)
        layout.addWidget(terminal_header)
        layout.addWidget(self.terminal, 1)
        return area

    @staticmethod
    def _path_row(line_edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row_layout.addWidget(line_edit, 1)
        button = QToolButton()
        button.setText("…")
        button.setToolTip("Browse path")
        button.clicked.connect(callback)
        row_layout.addWidget(button)
        return row

    @staticmethod
    def _field_specs_in_order() -> list[FieldSpec]:
        return list(FIELD_SPECS)

    def _make_field(self, spec: FieldSpec) -> tuple[QLabel, QWidget, QWidget]:
        label = QLabel(spec.label)
        label.setObjectName("fieldLabel")
        label.setProperty("class", "fieldLabel")
        label.setToolTip(self._tooltip_for(spec))

        if spec.kind == "bool":
            widget: QWidget = QCheckBox("Enabled")
            widget.setProperty("locked", True)
            widget.setToolTip(self._tooltip_for(spec))
            widget.stateChanged.connect(self._field_changed)
            row = widget
        elif spec.kind == "combo":
            combo = QComboBox()
            combo.addItems(list(spec.choices))
            combo.setToolTip(self._tooltip_for(spec))
            combo.currentTextChanged.connect(self._field_changed)
            widget = combo
            row = combo
        elif spec.kind == "multiline":
            edit = QPlainTextEdit()
            edit.setPlaceholderText(spec.placeholder)
            edit.setToolTip(self._tooltip_for(spec))
            edit.setMinimumHeight(48)
            edit.setMaximumHeight(58)
            edit.setProperty("locked", True)
            edit.setReadOnly(True)
            edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
            edit.textChanged.connect(self._field_changed)
            widget = edit
            row = edit
        else:
            edit = QLineEdit()
            edit.setPlaceholderText(spec.placeholder)
            edit.setReadOnly(True)
            edit.setToolTip(self._tooltip_for(spec))
            if spec.kind == "int":
                edit.setValidator(QIntValidator(-999999, 999999, edit))
            elif spec.kind == "float":
                edit.setValidator(QDoubleValidator(-999999.0, 999999.0, 6, edit))
            edit.textChanged.connect(self._field_changed)
            widget = edit
            row = self._path_row(edit, lambda key=spec.key: self._browse_parameter_path(key)) if spec.kind == "path" else edit

        widget.installEventFilter(self)
        label.installEventFilter(self)
        return label, widget, row

    def _tooltip_for(self, spec: FieldSpec) -> str:
        return f"{spec.key}\n{describe_field(spec)}"

    def _populate_values(self) -> None:
        self._lock_all_fields()
        for key, widget in self.field_widgets.items():
            value = self.values.get(key, DEFAULT_VALUES.get(key, ""))
            blocker = QSignalBlocker(widget)
            try:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                    else:
                        widget.setCurrentText(str(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                    widget.setCursorPosition(0)
                elif isinstance(widget, QPlainTextEdit):
                    widget.setPlainText(str(value))
                    widget.moveCursor(QTextCursor.Start)
            finally:
                del blocker
        first = next(iter(self.field_widgets), None)
        if first:
            self._show_help(first)

    def _lock_all_fields(self) -> None:
        self.locked_fields = set(self.field_widgets)
        for widget in self.field_widgets.values():
            widget.removeEventFilter(self)
            widget.installEventFilter(self)
            if isinstance(widget, QCheckBox):
                widget.setProperty("locked", True)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
            elif isinstance(widget, (QLineEdit, QPlainTextEdit)):
                if isinstance(widget, QPlainTextEdit):
                    widget.setProperty("locked", True)
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)
                widget.setReadOnly(True)

    def _current_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, widget in self.field_widgets.items():
            if isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText().strip()
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text()
            elif isinstance(widget, QPlainTextEdit):
                values[key] = widget.toPlainText()

        return values

    def _field_changed(self, *_args) -> None:
        self.values = self._current_values()
        self._update_command_preview()
        self._update_mode_visibility()

    def _update_command_preview(self) -> None:
        self.command_preview.setPlainText(format_preview(self._current_values(), self.project_root))

    def _update_mode_visibility(self) -> None:
        mode = str(self._current_values().get("control.type", "record"))
        for key, widget in self.field_widgets.items():
            spec = FIELD_BY_KEY[key]
            enabled = not spec.modes or mode in spec.modes
            widget.setEnabled(enabled)
            self.field_labels[key].setEnabled(enabled)
            self.field_rows[key].setEnabled(enabled)

    def eventFilter(self, watched, event) -> bool:
        for key, widget in self.field_widgets.items():
            is_field = watched is widget or watched is self.field_labels[key] or watched is self.field_rows[key]
            if not is_field:
                continue
            if event.type() == QEvent.MouseButtonDblClick and key in self.locked_fields:
                self._unlock_field(key)
                return True
            if event.type() == QEvent.MouseButtonPress:
                self._show_help(key)
                if key in self.locked_fields and isinstance(widget, QCheckBox):
                    now = time.monotonic()
                    previous = self._last_checkbox_press.get(key, 0.0)
                    if now - previous <= QApplication.doubleClickInterval() / 1000.0:
                        self._unlock_field(key)
                        self._last_checkbox_press.pop(key, None)
                    else:
                        self._last_checkbox_press[key] = now
                    return True
            elif event.type() == QEvent.FocusIn:
                self._show_help(key)
            break
        return super().eventFilter(watched, event)

    def _unlock_field(self, key: str) -> None:
        widget = self.field_widgets[key]
        self.locked_fields.discard(key)
        if isinstance(widget, QCheckBox):
            widget.removeEventFilter(self)
            widget.setProperty("locked", False)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        elif isinstance(widget, (QLineEdit, QPlainTextEdit)):
            if isinstance(widget, QPlainTextEdit):
                widget.setProperty("locked", False)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
            widget.setReadOnly(False)
            widget.setFocus()
        self._set_footer(f"Unlocked \"{FIELD_BY_KEY[key].label}\"; you can edit it now")

    def _show_help(self, key: str) -> None:
        spec = FIELD_BY_KEY[key]
        value = self._current_values().get(key, spec.default)
        kind_name = {
            "bool": "Switch",
            "int": "Integer",
            "float": "Number",
            "json": "JSON text",
            "path": "Path",
            "combo": "Choice",
            "text": "Text",
            "multiline": "Multiline text",
        }.get(spec.kind, spec.kind)
        default = "On" if spec.default is True else "Off" if spec.default is False else str(spec.default)
        current = "On" if value is True else "Off" if value is False else str(value)
        self.help_title.setText(f"Parameter help · {spec.label}")
        self.help_browser.setHtml(
            "<div style='line-height:1.45'>"
            f"<div><b>Argument</b>　<code>--{html.escape(spec.key)}=...</code></div>"
            f"<div><b>Type</b>　{html.escape(kind_name)}　　<b>Default</b>　{html.escape(default)}</div>"
            f"<div><b>Current</b>　{html.escape(current)}</div>"
            f"<p style='margin:7px 0 0 0'>{html.escape(describe_field(spec))}</p>"
            "</div>"
        )

    def _start_terminal(self) -> None:
        if self.session and self.session.is_running:
            return
        if self.session:
            self.session.stop()
            self.session = None
        shell_command = inherited_shell_command()
        # Start from the package parent so the generated command uses the same
        # relative import and working-directory behavior as the project layout.
        self.session = PtySession(self.project_root.parent, shell_command)
        self.session.output.connect(self._terminal_output)
        self.session.started.connect(self._terminal_started)
        self.session.finished.connect(self._terminal_finished)
        self.session.error.connect(self._terminal_error)
        self.session.start()

    def _restart_terminal(self) -> None:
        if self.session:
            self.session.stop()
            self.session = None
        self.terminal.clear_terminal()
        self._set_status("Restarting terminal", "warning")
        QTimer.singleShot(120, self._start_terminal)

    def _terminal_started(self) -> None:
        self._set_status("Terminal ready", "ready")
        self.terminal.setFocus()

    def _terminal_finished(self, return_code: int) -> None:
        self._set_status(f"Terminal exited ({return_code})", "warning" if return_code == 0 else "error")
        self._set_footer("The terminal process has ended; click \"Restart terminal\" to reconnect")

    def _terminal_error(self, message: str) -> None:
        self._set_status("Terminal error", "error")
        self._set_footer(message)

    def _terminal_output(self, data: bytes) -> None:
        self.terminal.feed(data)

    def _write_terminal_bytes(self, data: bytes) -> None:
        if self.session:
            self.session.send_bytes(data)

    def _resize_terminal(self, columns: int, rows: int) -> None:
        if self.session:
            self.session.resize(columns, rows)

    def _run_command(self) -> None:
        root = self.project_root
        if not (root / "scripts" / "control_robot.py").exists():
            QMessageBox.warning(self, "Invalid project path",
                                "scripts/control_robot.py was not found in the project root.")
            return
        values = self._current_values()
        command = build_command(values, root)
        if not self.session or not self.session.is_running:
            self._start_terminal()
            QTimer.singleShot(500, lambda: self.session.send_command(command) if self.session else None)
        else:
            self.session.send_command(command)
        self._set_status("Command sent", "running")
        self._set_footer("The full command has been entered in the terminal; the terminal remains interactive")

    @staticmethod
    def _validate_values(values: dict[str, Any]) -> str | None:
        required = {
            "robot.type": "Robot type",
            "control.type": "Control mode",
        }
        for key, label in required.items():
            if not str(values.get(key, "")).strip():
                return f"{label} must not be empty."
        for spec in FIELD_BY_KEY.values():
            if spec.kind != "json":
                continue
            raw = str(values.get(spec.key, "")).strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return f"{spec.label} must be valid JSON: {exc.msg}"
            if not isinstance(parsed, (list, dict)):
                return f"{spec.label} must be a JSON array or object."
        return None

    def _interrupt_command(self) -> None:
        if self.session:
            self.session.interrupt()
            self._set_status("Ctrl+C sent", "warning")
            self._set_footer("Requesting interruption of the current command; the terminal stays open")

    def _browse_parameter_path(self, key: str) -> None:
        widget = self.field_widgets.get(key)
        if not isinstance(widget, QLineEdit):
            return
        selected = QFileDialog.getExistingDirectory(self, f"Choose {FIELD_BY_KEY[key].label}", widget.text())
        if selected:
            widget.setText(selected)

    def _set_status(self, text: str, status: str) -> None:
        self.status_text.setText(text)
        self.status_dot.setProperty("status", status)
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)

    def _set_footer(self, text: str) -> None:
        self.footer_text.setText(text)

    def closeEvent(self, event) -> None:
        if self.session:
            self.session.stop()
            self.session = None
        event.accept()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LeRobot script console")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args(argv)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("LeRobot Console")
    app.setOrganizationName("LeRobot")
    app.setStyle("Fusion")
    window = MainWindow(args.project_root)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())


