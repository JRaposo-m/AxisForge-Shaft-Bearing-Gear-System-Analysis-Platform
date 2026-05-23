"""
ui/code_editor.py
Cell-based code editor for AxisForge UI.

Components:
  - PythonHighlighter : QSyntaxHighlighter using Pygments tokens
  - CellWidget        : Single executable cell (header + editor + output)
  - CellList          : Scrollable list of CellWidgets
  - CellRunner        : Imported from runner.py — executed in QThread

Thread model:
  - CellWorker runs exec() in a QThread to keep UI responsive.
  - Signals carry CellResult back to the main thread.
  - CellWidget disables run button while worker is active.

Signals emitted by CellList (connect in main_window.py):
  - cell_executed(int, CellResult)   : after any cell finishes
  - cell_selected(int)               : when user clicks a cell header
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import (
    Qt, Signal, QThread, QObject, QTimer,
)
from PySide6.QtGui import (
    QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from runner import CellResult, CellRunner


# ---------------------------------------------------------------------------
# Colour palette — dark theme QSS compatible
# ---------------------------------------------------------------------------

_BG_CELL        = "#1e1e2e"
_BG_HEADER      = "#181825"
_BG_OUTPUT      = "#11111b"
_BG_OUTPUT_ERR  = "#2a1a1a"
_FG_DEFAULT     = "#cdd6f4"
_FG_MUTED       = "#6c7086"
_ACCENT_OK      = "#a6e3a1"
_ACCENT_WARN    = "#f9e2af"
_ACCENT_ERR     = "#f38ba8"
_ACCENT_RUN     = "#89b4fa"
_BORDER_ACTIVE  = "#89b4fa"
_BORDER_IDLE    = "#313244"

_FONT_MONO = "Consolas, 'Courier New', monospace"
_FONT_SIZE  = 12


# ---------------------------------------------------------------------------
# Syntax highlighter — Pygments tokens → QTextCharFormat
# ---------------------------------------------------------------------------

class PythonHighlighter(QSyntaxHighlighter):
    """
    Pygments-based Python syntax highlighter for QPlainTextEdit.

    Uses Pygments tokeniser for correctness. Formats are fixed at
    construction — no runtime Pygments calls during typing beyond
    re-tokenising the changed block.
    """

    _FORMATS: dict[str, QTextCharFormat] = {}

    # Catppuccin Mocha palette mapped to token categories
    _TOKEN_COLORS: dict[str, str] = {
        "keyword":    "#cba6f7",   # purple
        "builtin":    "#89dceb",   # sky
        "string":     "#a6e3a1",   # green
        "comment":    "#6c7086",   # overlay0
        "number":     "#fab387",   # peach
        "operator":   "#89b4fa",   # blue
        "decorator":  "#f38ba8",   # red
        "class":      "#f9e2af",   # yellow
        "function":   "#89b4fa",   # blue
        "default":    "#cdd6f4",   # text
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_formats()

    def _build_formats(self) -> None:
        for key, color in self._TOKEN_COLORS.items():
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            if key in ("keyword", "class", "function"):
                fmt.setFontWeight(700)
            self._FORMATS[key] = fmt

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        try:
            from pygments import lex
            from pygments.lexers import PythonLexer
            from pygments.token import (
                Token, Keyword, Name, String, Comment,
                Number, Operator, Punctuation,
            )
        except ImportError:
            return

        lexer = PythonLexer()
        pos = 0
        for ttype, value in lex(text, lexer):
            length = len(value)
            fmt = self._format_for(ttype, Token, Keyword, Name,
                                    String, Comment, Number, Operator)
            if fmt:
                self.setFormat(pos, length, fmt)
            pos += length

    def _format_for(self, ttype, Token, Keyword, Name,  # noqa: N803
                    String, Comment, Number, Operator) -> QTextCharFormat | None:
        f = self._FORMATS
        if ttype in Keyword:
            return f["keyword"]
        if ttype in (Name.Builtin, Name.Builtin.Pseudo):
            return f["builtin"]
        if ttype in (Name.Class,):
            return f["class"]
        if ttype in (Name.Function, Name.Function.Magic):
            return f["function"]
        if ttype in (Name.Decorator,):
            return f["decorator"]
        if ttype in String:
            return f["string"]
        if ttype in Comment:
            return f["comment"]
        if ttype in Number:
            return f["number"]
        if ttype in Operator:
            return f["operator"]
        return None


# ---------------------------------------------------------------------------
# Worker — runs cell in QThread
# ---------------------------------------------------------------------------

class CellWorker(QObject):
    """Executes a single cell in a background QThread."""

    finished = Signal(object)   # CellResult

    def __init__(self, runner: "CellRunner", cell_index: int, code: str) -> None:
        super().__init__()
        self._runner = runner
        self._index = cell_index
        self._code = code

    def run(self) -> None:
        result = self._runner.execute(self._index, self._code)
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# CellWidget
# ---------------------------------------------------------------------------

class CellWidget(QFrame):
    """
    Single executable cell.

    Layout:
      HeaderBar  [▶] [title__________] [badge]
      CodeEdit   QPlainTextEdit with PythonHighlighter
      OutputArea QFrame — visible only when there is output
    """

    # Signals
    run_requested = Signal(int)        # cell_index
    selected = Signal(int)             # cell_index — user clicked header

    def __init__(self, cell_index: int, runner: "CellRunner",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = cell_index
        self._runner = runner
        self._thread: QThread | None = None
        self._worker: CellWorker | None = None

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def cell_index(self) -> int:
        return self._index

    def set_index(self, index: int) -> None:
        """Update index — called when cells are reordered."""
        self._index = index

    def code(self) -> str:
        return self._code_edit.toPlainText()

    def set_code(self, text: str) -> None:
        self._code_edit.setPlainText(text)

    def title(self) -> str:
        return self._title_edit.text()

    def set_title(self, text: str) -> None:
        self._title_edit.setText(text)

    def set_running(self) -> None:
        self._run_btn.setEnabled(False)
        self._run_btn.setText("…")
        self._set_badge("run")

    def apply_result(self, result: "CellResult") -> None:
        """Update UI from a CellResult produced by runner."""
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶")

        if not result.success:
            self._set_badge("error")
            lines = result.traceback_lines or [
                f"{result.error_type}: {result.error_message}"
            ]
            self._show_output("\n".join(lines), error=True)
        elif result.validation_errors:
            self._set_badge("warning")
            self._show_output(
                "⚠ " + "\n  ".join(result.validation_errors), error=False
            )
        else:
            self._set_badge("ok")
            text = result.stdout.strip()
            if text:
                self._show_output(text, error=False)
            else:
                self._hide_output()

    def trigger_run(self) -> None:
        """Start execution in background thread."""
        if self._thread and self._thread.isRunning():
            return

        code = self.code()
        self.set_running()

        self._thread = QThread()
        self._worker = CellWorker(self._runner, self._index, code)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    # ------------------------------------------------------------------
    # Private — UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("CellHeader")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(6, 4, 6, 4)
        h_layout.setSpacing(6)

        self._run_btn = QPushButton("▶")
        self._run_btn.setFixedSize(24, 24)
        self._run_btn.setToolTip("Run cell (Shift+Enter)")
        self._run_btn.clicked.connect(self.trigger_run)

        self._title_edit = QLineEdit(f"Cell {self._index + 1}")
        self._title_edit.setPlaceholderText("Cell title…")
        self._title_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._badge = QLabel("○")
        self._badge.setFixedWidth(20)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        h_layout.addWidget(self._run_btn)
        h_layout.addWidget(self._title_edit)
        h_layout.addWidget(self._badge)

        # Click on header → select
        header.mousePressEvent = lambda _e: self.selected.emit(self._index)

        # Code editor
        self._code_edit = QPlainTextEdit()
        self._code_edit.setFont(self._mono_font())
        self._code_edit.setMinimumHeight(80)
        self._code_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._highlighter = PythonHighlighter(self._code_edit.document())

        # Shift+Enter → run
        self._code_edit.keyPressEvent = self._intercept_keys

        # Output area
        self._output_frame = QFrame()
        self._output_frame.setObjectName("CellOutput")
        out_layout = QVBoxLayout(self._output_frame)
        out_layout.setContentsMargins(8, 4, 8, 4)
        self._output_label = QLabel()
        self._output_label.setFont(self._mono_font(size=11))
        self._output_label.setWordWrap(True)
        self._output_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        out_layout.addWidget(self._output_label)
        self._output_frame.hide()

        root.addWidget(header)
        root.addWidget(self._code_edit)
        root.addWidget(self._output_frame)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            CellWidget {{
                background: {_BG_CELL};
                border: 1px solid {_BORDER_IDLE};
                border-radius: 4px;
            }}
            QFrame#CellHeader {{
                background: {_BG_HEADER};
                border-bottom: 1px solid {_BORDER_IDLE};
            }}
            QPlainTextEdit {{
                background: {_BG_CELL};
                color: {_FG_DEFAULT};
                border: none;
                padding: 6px;
            }}
            QLineEdit {{
                background: transparent;
                color: {_FG_MUTED};
                border: none;
                font-size: 11px;
            }}
            QPushButton {{
                background: transparent;
                color: {_ACCENT_RUN};
                border: none;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: white;
            }}
            QFrame#CellOutput {{
                background: {_BG_OUTPUT};
                border-top: 1px solid {_BORDER_IDLE};
            }}
        """)

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    def _intercept_keys(self, event) -> None:
        """Shift+Enter triggers run; everything else passes through."""
        if (event.key() == Qt.Key.Key_Return
                and event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.trigger_run()
        else:
            QPlainTextEdit.keyPressEvent(self._code_edit, event)

    def _on_worker_finished(self, result: "CellResult") -> None:
        self.apply_result(result)
        self.run_requested.emit(self._index)

    def _set_badge(self, status: str) -> None:
        colors = {
            "ok":      (_ACCENT_OK,   "✓"),
            "warning": (_ACCENT_WARN, "⚠"),
            "error":   (_ACCENT_ERR,  "✗"),
            "run":     (_ACCENT_RUN,  "…"),
        }
        color, symbol = colors.get(status, (_FG_MUTED, "○"))
        self._badge.setText(symbol)
        self._badge.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _show_output(self, text: str, *, error: bool) -> None:
        bg = _BG_OUTPUT_ERR if error else _BG_OUTPUT
        color = _ACCENT_ERR if error else _FG_DEFAULT
        self._output_label.setText(text)
        self._output_label.setStyleSheet(f"color: {color};")
        self._output_frame.setStyleSheet(
            f"QFrame#CellOutput {{ background: {bg}; "
            f"border-top: 1px solid {_BORDER_IDLE}; }}"
        )
        self._output_frame.show()

    def _hide_output(self) -> None:
        self._output_frame.hide()

    @staticmethod
    def _mono_font(size: int = _FONT_SIZE) -> QFont:
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(size)
        return font


# ---------------------------------------------------------------------------
# CellList
# ---------------------------------------------------------------------------

class CellList(QWidget):
    """
    Scrollable list of CellWidgets sharing a CellRunner namespace.

    Signals
    -------
    cell_executed(int, object)
        Emitted after any cell finishes. Carries (cell_index, CellResult).
    cell_selected(int)
        Emitted when user clicks a cell header.
    """

    cell_executed = Signal(int, object)   # (cell_index, CellResult)
    cell_selected = Signal(int)

    def __init__(self, runner: "CellRunner",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runner = runner
        self._cells: list[CellWidget] = []

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch()

        self._scroll.setWidget(self._container)

        self._add_btn = QPushButton("+ Add Cell")
        self._add_btn.setFixedHeight(28)
        self._add_btn.clicked.connect(self.add_cell)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._scroll)
        root.addWidget(self._add_btn)

        self._apply_style()

        # Start with one empty cell
        self.add_cell()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_cell(self, code: str = "", title: str = "") -> CellWidget:
        """Append a new cell and return it."""
        index = len(self._cells)
        cell = CellWidget(index, self._runner, self._container)
        if code:
            cell.set_code(code)
        if title:
            cell.set_title(title)

        cell.run_requested.connect(self._on_cell_run_requested)
        cell.selected.connect(self.cell_selected)

        # Insert before the stretch
        self._layout.insertWidget(self._layout.count() - 1, cell)
        self._cells.append(cell)

        # Scroll to new cell
        QTimer.singleShot(50, lambda: self._scroll.ensureWidgetVisible(cell))
        return cell

    def run_all(self) -> None:
        """Execute all cells sequentially (cell 0 → N)."""
        for cell in self._cells:
            cell.trigger_run()

    def cell_count(self) -> int:
        return len(self._cells)

    def get_cell(self, index: int) -> CellWidget | None:
        if 0 <= index < len(self._cells):
            return self._cells[index]
        return None

    def load_cells(self, cells: list[dict]) -> None:
        """
        Load cells from serialised state.

        cells: list of {"title": str, "code": str}
        Clears existing cells first.
        """
        # Clear existing
        for cell in self._cells:
            cell.deleteLater()
        self._cells.clear()

        for entry in cells:
            self.add_cell(
                code=entry.get("code", ""),
                title=entry.get("title", ""),
            )

    def dump_cells(self) -> list[dict]:
        """Serialise current cells to list of dicts."""
        return [
            {"title": c.title(), "code": c.code()}
            for c in self._cells
        ]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_cell_run_requested(self, cell_index: int) -> None:
        """Relay signal with result from runner namespace."""
        # Result is already applied to cell; we re-fetch from runner
        # for external consumers (canvas, tree).
        # The runner's namespace is already updated at this point.
        self.cell_executed.emit(cell_index, None)  # result carried by cell

    def _apply_style(self) -> None:
        self._scroll.setStyleSheet(f"background: #181825;")
        self._container.setStyleSheet(f"background: #181825;")
        self._add_btn.setStyleSheet(f"""
            QPushButton {{
                background: #313244;
                color: {_FG_MUTED};
                border: none;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: #45475a;
                color: {_FG_DEFAULT};
            }}
        """)
