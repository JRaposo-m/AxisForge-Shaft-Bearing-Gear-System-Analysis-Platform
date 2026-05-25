"""
ui/workspace_panel.py
WorkspacePanel — left dock widget showing namespace variables.

Behaviour:
  - Displays all user-defined variables in the SectionRunner namespace.
  - Excludes seeded names (Shaft, np, etc.) and _private names.
  - Updates after every SectionResult via refresh(namespace, stale_indices).
  - Variables from stale sections shown in muted colour.
  - Clicking a matplotlib Figure/Axes row opens a PlotWindow.
  - Clicking any other result dataclass emits inspect_requested(name, obj)
    so MainWindow can display it in the OutputConsole.

Columns:
  Name  | Type  | Value (truncated)

Signals:
  inspect_requested(str, object)
      Emitted when user clicks a non-Figure variable.
      Carries (variable_name, object).

Design rules:
  - No solver calls here.
  - No knowledge of specific result types — uses type name strings only,
    except for matplotlib Figure detection (import guarded).
  - PlotWindow is a lightweight QMainWindow child — not modal.
  - Modular: adding new result types requires no changes here.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel,
    QMainWindow, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Catppuccin Mocha palette (subset)
# ---------------------------------------------------------------------------

_BG_BASE      = "#1e1e2e"
_BG_MANTLE    = "#181825"
_BG_SURFACE   = "#313244"
_FG_TEXT      = "#cdd6f4"
_FG_SUBTEXT   = "#6c7086"
_FG_STALE     = "#45475a"
_FG_GREEN     = "#a6e3a1"
_FG_BLUE      = "#89b4fa"
_FG_PEACH     = "#fab387"
_FG_YELLOW    = "#f9e2af"

_FONT_FAMILY  = "Consolas, 'Courier New', monospace"
_FONT_SIZE    = 11

# Maximum characters shown in Value column before truncation
_VALUE_MAX_CHARS = 60

# Type names that trigger PlotWindow on click
_FIGURE_TYPES = {"Figure", "Axes", "AxesSubplot"}

# Type name prefixes that are result dataclasses — shown with accent colour
_RESULT_SUFFIXES = ("Result", "Section")


# ---------------------------------------------------------------------------
# PlotWindow — lightweight matplotlib figure viewer
# ---------------------------------------------------------------------------

class PlotWindow(QMainWindow):
    """
    Standalone window that embeds a matplotlib Figure.

    Created on demand when the user clicks a Figure variable.
    Closing the window does not affect the figure object in the namespace.
    """

    def __init__(
        self,
        fig: Any,
        title: str = "Plot",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"AxisForge — {title}")
        self.resize(800, 600)
        self._embed(fig)

    def _embed(self, fig: Any) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            canvas = FigureCanvasQTAgg(fig)
            self.setCentralWidget(canvas)
            canvas.draw()
        except Exception as exc:
            label = QLabel(f"Cannot display figure:\n{exc}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setCentralWidget(label)


# ---------------------------------------------------------------------------
# WorkspacePanel
# ---------------------------------------------------------------------------

class WorkspacePanel(QWidget):
    """
    MATLAB-style workspace panel showing namespace variables.

    Connect to SectionRunner via refresh() after each section execution.
    """

    inspect_requested = Signal(str, object)   # (name, obj)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stale_indices: set[int] = set()
        self._name_to_section: dict[str, int] = {}
        self._plot_windows: list[PlotWindow] = []   # keep references alive

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(
        self,
        namespace: dict[str, Any],
        stale_section_indices: list[int] | None = None,
        name_to_section: dict[str, int] | None = None,
    ) -> None:
        """
        Rebuild the table from the current namespace snapshot.

        Parameters
        ----------
        namespace : dict
            Output of SectionRunner.get_namespace() — already filtered
            (no seeded names, no _privates).
        stale_section_indices : list[int]
            Section indices considered stale. Variables written by these
            sections are shown in muted colour.
        name_to_section : dict[str, int]
            Maps variable name → section index that last wrote it.
            Used to determine staleness per variable.
        """
        self._stale_indices = set(stale_section_indices or [])
        self._name_to_section = name_to_section or {}

        self._table.setRowCount(0)
        self._table.setSortingEnabled(False)

        for name, obj in sorted(namespace.items()):
            self._add_row(name, obj)

        self._table.setSortingEnabled(True)
        self._update_count_label(len(namespace))

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._update_count_label(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header label
        self._count_label = QLabel("Workspace")
        self._count_label.setFixedHeight(24)
        self._count_label.setContentsMargins(8, 0, 0, 0)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Name", "Type", "Value"])
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(False)

        # Column widths
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self._table.cellDoubleClicked.connect(self._on_double_click)

        root.addWidget(self._count_label)
        root.addWidget(self._table)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background: {_BG_BASE};
                color: {_FG_TEXT};
                font-family: {_FONT_FAMILY};
                font-size: {_FONT_SIZE}px;
            }}
            QLabel {{
                background: {_BG_MANTLE};
                color: {_FG_SUBTEXT};
                font-size: 10px;
                border-bottom: 1px solid {_BG_SURFACE};
            }}
            QTableWidget {{
                background: {_BG_BASE};
                color: {_FG_TEXT};
                border: none;
                gridline-color: {_BG_SURFACE};
                selection-background-color: {_BG_SURFACE};
                selection-color: {_FG_TEXT};
            }}
            QHeaderView::section {{
                background: {_BG_MANTLE};
                color: {_FG_SUBTEXT};
                border: none;
                border-bottom: 1px solid {_BG_SURFACE};
                padding: 3px 6px;
                font-size: 10px;
            }}
            QScrollBar:vertical {{
                background: {_BG_MANTLE};
                width: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {_BG_SURFACE};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_row(self, name: str, obj: Any) -> None:
        """Add one variable row to the table."""
        type_name = type(obj).__name__
        value_str = self._format_value(obj, type_name)
        is_stale = self._is_stale(name)
        is_figure = type_name in _FIGURE_TYPES
        is_result = type_name.endswith(_RESULT_SUFFIXES)

        row = self._table.rowCount()
        self._table.insertRow(row)

        # Choose colour
        if is_stale:
            colour = _FG_STALE
        elif is_figure:
            colour = _FG_PEACH
        elif is_result:
            colour = _FG_GREEN
        else:
            colour = _FG_TEXT

        for col, text in enumerate([name, type_name, value_str]):
            item = QTableWidgetItem(text)
            item.setForeground(QColor(colour))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            # Store name in first column for retrieval on click
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, name)
            self._table.setItem(row, col, item)

        # Row height
        self._table.setRowHeight(row, 20)

    def _format_value(self, obj: Any, type_name: str) -> str:
        """Short single-line representation of a variable."""
        if type_name in _FIGURE_TYPES:
            return "— double-click to open plot —"

        try:
            r = repr(obj)
        except Exception:
            return "(repr failed)"

        # Collapse multiline repr to first line
        first_line = r.split("\n")[0]
        if len(first_line) > _VALUE_MAX_CHARS:
            return first_line[:_VALUE_MAX_CHARS] + "…"
        return first_line

    def _is_stale(self, name: str) -> bool:
        section_idx = self._name_to_section.get(name)
        if section_idx is None:
            return False
        return section_idx in self._stale_indices

    def _update_count_label(self, count: int) -> None:
        self._count_label.setText(
            f"  Workspace — {count} variable{'s' if count != 1 else ''}"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_double_click(self, row: int, _col: int) -> None:
        name_item = self._table.item(row, 0)
        if name_item is None:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        if name is None:
            return

        # Retrieve the live object — not stored in table, always from caller
        # The table stores name; the actual object must be re-fetched.
        # We emit inspect_requested and let MainWindow decide.
        # Special case: Figure — attempt to open PlotWindow directly if
        # the object is accessible via the signal receiver.
        type_item = self._table.item(row, 1)
        type_name = type_item.text() if type_item else ""

        if type_name in _FIGURE_TYPES:
            # Emit inspect_requested — MainWindow fetches obj and calls
            # open_plot(name, obj) on this panel.
            self.inspect_requested.emit(name, None)
        else:
            self.inspect_requested.emit(name, None)

    def open_plot(self, name: str, fig: Any) -> None:
        """
        Open a PlotWindow for the given matplotlib figure.

        Called by MainWindow after receiving inspect_requested for a Figure.
        Keeps a reference to prevent garbage collection.
        """
        win = PlotWindow(fig, title=name, parent=self.window())
        self._plot_windows.append(win)
        win.show()
        win.raise_()
