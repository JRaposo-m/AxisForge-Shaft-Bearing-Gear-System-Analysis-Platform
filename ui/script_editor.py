"""
ui/script_editor.py
ScriptEditor — central widget for AxisForge.

Replaces the cell-based CellList with a continuous .py file editor
with # %% section execution, matching the MATLAB Editor / Spyder model.

Layout:
  ┌─────────────────────────────────────────────────────────────┐
  │  [▶ Section]  [▶▶ All]  [⟳ Reset]   filename.py   [*]     │
  ├────┬────────────────────────────────────────────────────────┤
  │ LN │  Python source — syntax highlighted                    │
  │    │  # %% Section title        ← section bar (2px accent) │
  │    │  code...                                               │
  │    │                                                        │
  │    │  # %% Next section         ← next section bar         │
  │    │  code...                                               │
  └────┴────────────────────────────────────────────────────────┘

Keyboard shortcuts:
  Ctrl+Enter        Execute current section
  Shift+Enter       Execute current section + advance cursor to next
  Ctrl+Shift+Enter  Execute all sections
  Ctrl+S            Save file
  Ctrl+Shift+E      Open file in system default editor

Section detection:
  Lines starting with '# %%' delimit sections.
  The active section (containing the cursor) is highlighted with a
  2px left border in the line number gutter.

Signals:
  section_executed(SectionResult)
      Emitted after any section finishes executing.
  file_saved(str)
      Emitted after Ctrl+S with the absolute file path.
  file_changed()
      Emitted when document content changes (unsaved state).

Design rules:
  - ScriptEditor never calls solvers directly.
  - All execution goes through SectionRunner via SectionWorker (QThread).
  - File on disk is the source of truth — no hidden state.
  - MainWindow connects section_executed to OutputConsole and WorkspacePanel.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import (
    QObject, QRect, QSize, Qt, QThread, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QFont, QKeySequence, QPainter,
    QSyntaxHighlighter, QTextCharFormat, QTextCursor,
    QTextOption,
)
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from runner import Section, SectionResult, SectionRunner


# ---------------------------------------------------------------------------
# Catppuccin Mocha palette
# ---------------------------------------------------------------------------

_BG_BASE       = "#1e1e2e"
_BG_MANTLE     = "#181825"
_BG_SURFACE    = "#313244"
_BG_GUTTER     = "#181825"
_FG_TEXT       = "#cdd6f4"
_FG_SUBTEXT    = "#6c7086"
_FG_BLUE       = "#89b4fa"
_FG_GREEN      = "#a6e3a1"
_FG_RED        = "#f38ba8"
_FG_YELLOW     = "#f9e2af"
_ACCENT_ACTIVE = "#89b4fa"   # active section bar

_FONT_FAMILY   = "Consolas, 'Courier New', monospace"
_FONT_SIZE     = 12


# ---------------------------------------------------------------------------
# PythonHighlighter — reutilizado do code_editor.py
# ---------------------------------------------------------------------------

class PythonHighlighter(QSyntaxHighlighter):
    """Pygments-based Python syntax highlighter."""

    _FORMATS: dict[str, QTextCharFormat] = {}

    _TOKEN_COLORS: dict[str, str] = {
        "keyword":   "#cba6f7",
        "builtin":   "#89dceb",
        "string":    "#a6e3a1",
        "comment":   "#6c7086",
        "number":    "#fab387",
        "operator":  "#89b4fa",
        "decorator": "#f38ba8",
        "class":     "#f9e2af",
        "function":  "#89b4fa",
        "default":   "#cdd6f4",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_formats()
        try:
            from pygments.lexers import PythonLexer
            self._lexer = PythonLexer()
        except ImportError:
            self._lexer = None

    def _build_formats(self) -> None:
        for key, color in self._TOKEN_COLORS.items():
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            if key in ("keyword", "class", "function"):
                fmt.setFontWeight(700)
            self._FORMATS[key] = fmt

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        if self._lexer is None:
            return
        try:
            from pygments import lex
            from pygments.token import (
                Token, Keyword, Name, String, Comment, Number, Operator,
            )
        except ImportError:
            return
        pos = 0
        for ttype, value in lex(text, self._lexer):
            length = len(value)
            fmt = self._format_for(ttype, Token, Keyword, Name,
                                   String, Comment, Number, Operator)
            if fmt:
                self.setFormat(pos, length, fmt)
            pos += length

    def _format_for(self, ttype, Token, Keyword, Name,   # noqa: N803
                    String, Comment, Number, Operator) -> QTextCharFormat | None:
        f = self._FORMATS
        if ttype in Keyword:            return f["keyword"]
        if ttype in (Name.Builtin, Name.Builtin.Pseudo): return f["builtin"]
        if ttype in (Name.Class,):      return f["class"]
        if ttype in (Name.Function, Name.Function.Magic): return f["function"]
        if ttype in (Name.Decorator,):  return f["decorator"]
        if ttype in String:             return f["string"]
        if ttype in Comment:            return f["comment"]
        if ttype in Number:             return f["number"]
        if ttype in Operator:           return f["operator"]
        return None


# ---------------------------------------------------------------------------
# SectionWorker — executes one Section in a QThread
# ---------------------------------------------------------------------------

class SectionWorker(QObject):
    """Runs SectionRunner.execute() off the main thread."""

    finished = Signal(object)   # SectionResult

    def __init__(self, runner: SectionRunner, section: Section) -> None:
        super().__init__()
        self._runner = runner
        self._section = section

    def run(self) -> None:
        result = self._runner.execute(self._section)
        self.finished.emit(result)


class _AllSectionsWorker(QObject):
    """Runs all sections sequentially in a single QThread. Stops on first error."""

    section_done = Signal(object)   # SectionResult per section
    finished     = Signal()

    def __init__(self, runner: SectionRunner, sections: list) -> None:
        super().__init__()
        self._runner   = runner
        self._sections = sections

    def run(self) -> None:
        for section in self._sections:
            result = self._runner.execute(section)
            self.section_done.emit(result)
            if not result.success:
                break
        self.finished.emit()


# ---------------------------------------------------------------------------
# LineNumberGutter — left margin widget with line numbers + section bars
# ---------------------------------------------------------------------------

class LineNumberGutter(QWidget):
    """
    Paints line numbers and a 2px accent bar for the active section.

    Active section: the # %% block that contains the cursor.
    Section bars are drawn in _ACCENT_ACTIVE on the leftmost 2px.
    """

    _GUTTER_PADDING = 6   # px right-padding inside gutter

    def __init__(self, editor: "CodeEdit") -> None:
        super().__init__(editor)
        self._editor = editor
        self._active_section: Section | None = None

    def set_active_section(self, section: Section | None) -> None:
        self._active_section = section
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:  # noqa: N802
        self._editor.paint_gutter(self, event)


# ---------------------------------------------------------------------------
# CodeEdit — QPlainTextEdit with gutter
# ---------------------------------------------------------------------------

class CodeEdit(QPlainTextEdit):
    """
    QPlainTextEdit with:
      - Line number gutter (LineNumberGutter)
      - Active section highlighting (2px left bar)
      - # %% section awareness (cursor position → section index)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._gutter = LineNumberGutter(self)
        self._sections: list[Section] = []
        self._active_section: Section | None = None

        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(_FONT_SIZE)
        self.setFont(font)
        self.setTabStopDistance(28)   # 4-space equivalent at 12pt

        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {_BG_BASE};
                color: {_FG_TEXT};
                border: none;
                selection-background-color: {_BG_SURFACE};
            }}
        """)

        # Update gutter width when block count changes
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._on_cursor_moved)
        self._update_gutter_width()

    # ------------------------------------------------------------------
    # Gutter geometry
    # ------------------------------------------------------------------

    def gutter_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        char_w = self.fontMetrics().horizontalAdvance("9")
        return 2 + char_w * digits + self._gutter._GUTTER_PADDING + 2  # +2 for bar

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(
            QRect(cr.left(), cr.top(), self.gutter_width(), cr.height())
        )

    def _update_gutter_width(self) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_gutter_width()

    def paint_gutter(self, gutter: LineNumberGutter, event) -> None:
        """Called by LineNumberGutter.paintEvent."""
        painter = QPainter(gutter)
        painter.fillRect(event.rect(), QColor(_BG_GUTTER))

        block = self.firstVisibleBlock()
        block_num = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        line_h = self.fontMetrics().height()
        char_w = self.fontMetrics().horizontalAdvance("9")
        digits = max(3, len(str(self.blockCount())))
        num_w = char_w * digits
        bar_w = 2

        active = self._active_section

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                # Section bar
                if active is not None:
                    line_1based = block_num + 1
                    if active.line_start <= line_1based <= active.line_end:
                        painter.fillRect(
                            0, top, bar_w, line_h, QColor(_ACCENT_ACTIVE)
                        )

                # Line number
                painter.setPen(QColor(_FG_SUBTEXT))
                painter.drawText(
                    bar_w, top, num_w, line_h,
                    Qt.AlignmentFlag.AlignRight, str(block_num + 1)
                )

            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_num += 1

    # ------------------------------------------------------------------
    # Section tracking
    # ------------------------------------------------------------------

    def update_sections(self, sections: list[Section]) -> None:
        self._sections = sections
        self._on_cursor_moved()

    def current_section(self) -> Section | None:
        """Return the Section containing the current cursor position."""
        if not self._sections:
            return None
        line = self.textCursor().blockNumber() + 1   # 1-based
        # Walk sections in reverse — find last section whose line_start <= cursor
        current = self._sections[0]
        for s in self._sections:
            if s.line_start <= line:
                current = s
        return current

    def _on_cursor_moved(self) -> None:
        sec = self.current_section()
        if sec is not self._active_section:
            self._active_section = sec
            self._gutter.set_active_section(sec)


# ---------------------------------------------------------------------------
# ScriptEditor
# ---------------------------------------------------------------------------

class ScriptEditor(QWidget):
    """
    Central widget: continuous .py editor with # %% section execution.

    Signals
    -------
    section_executed(SectionResult)
        Emitted after every section execution (success or failure).
    file_saved(str)
        Emitted with absolute path after save.
    file_changed()
        Emitted on any document modification (unsaved indicator).
    """

    section_executed = Signal(object)   # SectionResult
    file_saved       = Signal(str)
    file_changed     = Signal()
    namespace_reset  = Signal()

    def __init__(
        self,
        runner: SectionRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner  = runner or SectionRunner()
        self._file_path: str = ""
        self._is_modified: bool = False
        self._sections: list[Section] = []
        self._thread: QThread | None = None
        self._worker: SectionWorker | None = None

        self._build_ui()
        self._connect_signals()
        self._apply_style()

        # Start with an empty untitled buffer
        self._set_modified(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_file(self, path: str | Path) -> None:
        """Load a .py file into the editor. Replaces current content."""
        p = Path(path)
        if not p.is_file():
            return
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Open File", str(e))
            return

        self._file_path = str(p)
        self._editor.setPlainText(text)
        self._reparse()
        self._set_modified(False)
        self._update_title()

    def save_file(self) -> bool:
        """Save current content to disk. Returns True on success."""
        if not self._file_path:
            return False
        try:
            Path(self._file_path).write_text(
                self._editor.toPlainText(), encoding="utf-8"
            )
        except OSError as e:
            QMessageBox.critical(self, "Save", str(e))
            return False
        self._set_modified(False)
        self.file_saved.emit(self._file_path)
        return True

    def runner(self) -> SectionRunner:
        return self._runner

    def current_text(self) -> str:
        return self._editor.toPlainText()

    def file_path(self) -> str:
        return self._file_path

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Toolbar
        root.addWidget(self._build_toolbar())

        # Editor
        self._editor = CodeEdit()
        PythonHighlighter(self._editor.document())
        root.addWidget(self._editor)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(36)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Run Section
        self._btn_run_section = QPushButton("▶  Run Section")
        self._btn_run_section.setToolTip("Execute current section  (Ctrl+Enter)")
        self._btn_run_section.setFixedHeight(26)

        # Run All
        self._btn_run_all = QPushButton("▶▶  Run All")
        self._btn_run_all.setToolTip("Execute all sections  (Ctrl+Shift+Enter)")
        self._btn_run_all.setFixedHeight(26)

        # Reset
        self._btn_reset = QPushButton("⟳  Reset")
        self._btn_reset.setToolTip("Clear namespace and reset runner")
        self._btn_reset.setFixedHeight(26)

        # File label
        self._lbl_file = QLabel("untitled")
        self._lbl_file.setStyleSheet(f"color: {_FG_SUBTEXT}; font-size: 11px;")
        self._lbl_file.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        # Modified indicator
        self._lbl_modified = QLabel("")
        self._lbl_modified.setFixedWidth(12)
        self._lbl_modified.setStyleSheet(f"color: {_FG_YELLOW}; font-size: 14px;")

        layout.addWidget(self._btn_run_section)
        layout.addWidget(self._btn_run_all)
        layout.addWidget(self._btn_reset)
        layout.addStretch()
        layout.addWidget(self._lbl_file)
        layout.addWidget(self._lbl_modified)

        return bar

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background: {_BG_BASE};
                color: {_FG_TEXT};
                font-family: {_FONT_FAMILY};
                font-size: {_FONT_SIZE}px;
            }}
            QWidget#toolbar {{
                background: {_BG_MANTLE};
                border-bottom: 1px solid {_BG_SURFACE};
            }}
            QPushButton {{
                background: {_BG_SURFACE};
                color: {_FG_TEXT};
                border: none;
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: #45475a;
            }}
            QPushButton:pressed {{
                background: #585b70;
            }}
        """)

        # Run Section button — blue accent
        self._btn_run_section.setStyleSheet(f"""
            QPushButton {{
                background: #1e4a7a;
                color: {_FG_BLUE};
                border: 1px solid {_FG_BLUE};
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background: #2a5a8a; }}
            QPushButton:pressed {{ background: #1a3a6a; }}
        """)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._btn_run_section.clicked.connect(self._run_current_section)
        self._btn_run_all.clicked.connect(self._run_all)
        self._btn_reset.clicked.connect(self._reset)
        self._editor.document().contentsChanged.connect(self._on_content_changed)
        self._editor.document().contentsChanged.connect(self._reparse)

        # Keyboard shortcuts
        from PySide6.QtGui import QShortcut
        from PySide6.QtCore import Qt as _Qt
        QShortcut(QKeySequence("Ctrl+Return"),       self, self._run_current_section)
        QShortcut(QKeySequence("Shift+Return"),      self, self._run_current_and_advance)
        QShortcut(QKeySequence("Ctrl+Shift+Return"), self, self._run_all)
        QShortcut(QKeySequence("Ctrl+Shift+E"),      self, self._open_external)
        # Ctrl+S: ApplicationShortcut so it works regardless of which widget has focus
        _sc_save = QShortcut(QKeySequence("Ctrl+S"), self)
        _sc_save.setContext(_Qt.ShortcutContext.ApplicationShortcut)
        _sc_save.activated.connect(self.save_file)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_current_section(self) -> None:
        section = self._editor.current_section()
        if section is None:
            return
        self._execute(section)

    def _run_current_and_advance(self) -> None:
        section = self._editor.current_section()
        if section is None:
            return
        self._execute(section)
        self._advance_to_next_section(section)

    def _run_all(self) -> None:
        if not self._sections:
            return
        if self._thread and self._thread.isRunning():
            return
        self._set_running(True)

        thread = QThread(self)
        worker = _AllSectionsWorker(self._runner, list(self._sections))
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.section_done.connect(self._on_section_done_run_all)
        worker.finished.connect(thread.quit)

        # Keep strong references until thread fully stops
        self._thread = thread
        self._worker = worker

        thread.finished.connect(self._on_run_all_finished)
        thread.start()

    def _execute(self, section: Section) -> None:
        """Execute one section in a QThread."""
        if self._thread and self._thread.isRunning():
            return

        self._set_running(True)

        thread = QThread(self)
        worker = SectionWorker(self._runner, section)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_section_finished)
        worker.finished.connect(thread.quit)

        # Keep strong references until thread fully stops
        self._thread = thread
        self._worker = worker

        thread.start()

    def _advance_to_next_section(self, current: Section) -> None:
        """Move cursor to the first line of the next section."""
        idx = current.index + 1
        if idx >= len(self._sections):
            return
        next_sec = self._sections[idx]
        doc = self._editor.document()
        block = doc.findBlockByLineNumber(next_sec.line_start)
        if block.isValid():
            cursor = QTextCursor(block)
            self._editor.setTextCursor(cursor)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_section_finished(self, result: SectionResult) -> None:
        """Slot for single-section execution (Run Section)."""
        self._set_running(False)
        if self._thread:
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.section_executed.emit(result)

    @Slot(object)
    def _on_section_done_run_all(self, result: SectionResult) -> None:
        """Slot for each section during Run All — emits signal but does NOT re-enable buttons."""
        self.section_executed.emit(result)

    @Slot()
    def _on_run_all_finished(self) -> None:
        """Called when the Run All thread finishes — re-enables buttons."""
        self._set_running(False)
        if self._thread:
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _on_content_changed(self) -> None:
        if not self._is_modified:
            self._set_modified(True)
        self.file_changed.emit()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _reparse(self) -> None:
        """Re-parse sections from current document text."""
        source = self._editor.toPlainText()
        self._sections = self._runner.parse(source)
        self._editor.update_sections(self._sections)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._runner.reset()
        self._thread = None
        self._worker = None
        self._set_running(False)
        self.namespace_reset.emit()

    def _set_running(self, running: bool) -> None:
        self._btn_run_section.setEnabled(not running)
        self._btn_run_all.setEnabled(not running)
        if running:
            self._btn_run_section.setText("…")
        else:
            self._btn_run_section.setText("▶  Run Section")

    def _set_modified(self, modified: bool) -> None:
        self._is_modified = modified
        self._lbl_modified.setText("●" if modified else "")

    def _update_title(self) -> None:
        if self._file_path:
            name = Path(self._file_path).name
        else:
            name = "untitled"
        self._lbl_file.setText(name)

    def _open_external(self) -> None:
        """Open current file in system default editor."""
        if not self._file_path:
            return
        if sys.platform == "win32":
            os.startfile(self._file_path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self._file_path])  # noqa: S603
        else:
            subprocess.Popen(["xdg-open", self._file_path])  # noqa: S603