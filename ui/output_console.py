"""
ui/output_console.py
OutputConsole — bottom dock widget for AxisForge.

Three tabs:
  Output  : stdout from user sections + execution headers.
            Format per section: '─── § Title — 0.12s ───'
            followed by captured stdout. Nothing else.
  Errors  : filtered tracebacks from failed sections.
            Only user-code frames shown, not internal runner frames.
  Log     : application events (file opened, project loaded, saved, reset).

Design rules:
  - No solver intermediate data here. That belongs in .txt files.
  - OutputConsole never calls solvers or parses results.
  - All content arrives via public slots — no internal state beyond text.
  - Monospace font throughout.
  - Colours from Catppuccin Mocha palette (consistent with app stylesheet).

Public slots (connect from MainWindow / ScriptEditor):
  append_section_result(SectionResult)
  append_log(message: str)
  clear_output()
  clear_errors()
  clear_all()

Signals: none — output only.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout, QPushButton, QTabWidget,
    QPlainTextEdit, QVBoxLayout, QWidget,
)

from runner import SectionResult


# ---------------------------------------------------------------------------
# Catppuccin Mocha — local palette (subset used here)
# ---------------------------------------------------------------------------

_BG_BASE      = "#1e1e2e"
_BG_MANTLE    = "#181825"
_BG_SURFACE   = "#313244"
_FG_TEXT      = "#cdd6f4"
_FG_SUBTEXT   = "#6c7086"
_FG_GREEN     = "#a6e3a1"   # ok / success
_FG_YELLOW    = "#f9e2af"   # warning
_FG_RED       = "#f38ba8"   # error
_FG_BLUE      = "#89b4fa"   # section header / accent
_FG_PEACH     = "#fab387"   # timing

_FONT_FAMILY  = "Consolas, 'Courier New', monospace"
_FONT_SIZE    = 11


# ---------------------------------------------------------------------------
# _ConsoleEdit — read-only QPlainTextEdit with coloured append
# ---------------------------------------------------------------------------

class _ConsoleEdit(QPlainTextEdit):
    """
    Read-only plain text editor with coloured append methods.

    Uses QTextCharFormat for per-run colour without HTML overhead.
    Capped at MAX_BLOCKS blocks to prevent unbounded memory growth.
    """

    MAX_BLOCKS = 5000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_BLOCKS)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(_FONT_SIZE)
        self.setFont(font)

        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {_BG_MANTLE};
                color: {_FG_TEXT};
                border: none;
                padding: 4px;
                selection-background-color: {_BG_SURFACE};
            }}
        """)

    # ------------------------------------------------------------------
    # Coloured append
    # ------------------------------------------------------------------

    def append_plain(self, text: str, colour: str = _FG_TEXT) -> None:
        """Append text in the given hex colour. Adds newline."""
        if not text:
            return
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text + "\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def append_separator(self, text: str = "", colour: str = _FG_BLUE) -> None:
        """Append a section header separator line."""
        self.append_plain(text, colour)

    def append_blank(self) -> None:
        """Append an empty line."""
        self.append_plain("", _FG_TEXT)


# ---------------------------------------------------------------------------
# OutputConsole
# ---------------------------------------------------------------------------

class OutputConsole(QWidget):
    """
    Three-tab output dock for AxisForge.

    Usage
    -----
    Instantiate and set as widget of a QDockWidget.
    Connect SectionRunner / ScriptEditor signals to the public slots.

    Example
    -------
        console = OutputConsole()
        dock = QDockWidget("Output")
        dock.setWidget(console)
        script_editor.section_executed.connect(console.append_section_result)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def append_section_result(self, result: SectionResult) -> None:
        """
        On failure: write traceback to Errors tab and switch to it.
        On success with validation warnings: write to Errors tab.
        On clean success with stdout: write print output to Log tab.
        """
        if result.success and not result.validation_errors:
            if result.stdout.strip():
                self._log.append_plain(result.format_header(), _FG_BLUE)
                for line in result.stdout.splitlines():
                    self._log.append_plain(line, _FG_TEXT)
                self._log.append_blank()
                self._tabs.setCurrentIndex(1)
            return

        if result.validation_errors:
            self._errors.append_plain(
                f"§ {result.title}  —  validation", _FG_YELLOW
            )
            for err in result.validation_errors:
                self._errors.append_plain(f"  ⚠ {err}", _FG_YELLOW)
            self._errors.append_blank()
            self._tabs.setCurrentIndex(0)
            return

        # Failure
        self._write_error(result)
        self._tabs.setCurrentIndex(0)

    def append_log(self, message: str) -> None:
        """Append a timestamped application event to the Log tab."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append_plain(f"[{ts}]  {message}", _FG_SUBTEXT)

    def clear_output(self) -> None:
        self._errors.clear()  # kept for API compat — clears errors

    def clear_errors(self) -> None:
        self._errors.clear()

    def clear_all(self) -> None:
        self._errors.clear()
        self._log.clear()

    def switch_to_output(self) -> None:
        self._tabs.setCurrentIndex(0)  # Errors is now index 0

    def switch_to_errors(self) -> None:
        self._tabs.setCurrentIndex(0)

    def switch_to_log(self) -> None:
        self._tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # --- Errors tab (index 0) ---
        self._errors = _ConsoleEdit()
        self._tabs.addTab(self._errors, "Errors")

        # --- Log tab (index 1) ---
        self._log = _ConsoleEdit()
        self._tabs.addTab(self._log, "Log")

        toolbar = self._build_toolbar()

        root.addWidget(toolbar)
        root.addWidget(self._tabs)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(28)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(54)
        clear_btn.setToolTip("Clear current tab")
        clear_btn.clicked.connect(self._on_clear_current)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setFixedWidth(64)
        clear_all_btn.setToolTip("Clear all tabs")
        clear_all_btn.clicked.connect(self.clear_all)

        layout.addStretch()
        layout.addWidget(clear_btn)
        layout.addWidget(clear_all_btn)
        return bar

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background: {_BG_BASE};
                color: {_FG_TEXT};
                font-family: {_FONT_FAMILY};
                font-size: {_FONT_SIZE}px;
            }}
            QTabWidget::pane {{
                border: none;
                background: {_BG_MANTLE};
            }}
            QTabBar::tab {{
                background: {_BG_BASE};
                color: {_FG_SUBTEXT};
                padding: 4px 14px;
                border: none;
                border-right: 1px solid {_BG_SURFACE};
            }}
            QTabBar::tab:selected {{
                background: {_BG_MANTLE};
                color: {_FG_TEXT};
                border-bottom: 2px solid {_FG_BLUE};
            }}
            QTabBar::tab:hover {{
                color: {_FG_TEXT};
            }}
            QPushButton {{
                background: {_BG_SURFACE};
                color: {_FG_SUBTEXT};
                border: none;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: #45475a;
                color: {_FG_TEXT};
            }}
        """)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def append_inspect(self, name: str, repr_str: str) -> None:
        """Show variable repr in Log tab (called from MainWindow on inspect)."""
        self._log.append_plain(f"inspect: {name}", _FG_BLUE)
        for line in repr_str.splitlines():
            self._log.append_plain(f"  {line}", _FG_TEXT)
        self._log.append_blank()
        self._tabs.setCurrentIndex(1)

    def _write_error(self, result: SectionResult) -> None:
        """Write filtered traceback to the Errors tab."""
        # Section header in errors tab
        self._errors.append_plain(
            f"§ {result.title}  —  {result.error_type}", _FG_RED
        )
        # Traceback frames
        for line in result.traceback_lines:
            if line.startswith("File "):
                self._errors.append_plain(f"  {line}", _FG_SUBTEXT)
            else:
                self._errors.append_plain(f"  {line}", _FG_RED)
        # Final exception line with message
        self._errors.append_plain(
            f"  → {result.error_type}: {result.error_message}", _FG_RED
        )
        self._errors.append_blank()

    def _on_clear_current(self) -> None:
        idx = self._tabs.currentIndex()
        if idx == 0:
            self.clear_errors()
        else:
            self._log.clear()