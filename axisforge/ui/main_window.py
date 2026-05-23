"""
ui/main_window.py
AxisForge main application window.

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  MenuBar                      [▶ Run All]  [⟳ Reset]   │
  ├──────────────┬──────────────────────────────────────────┤
  │              │  Tab: Canvas  |  Tab: Editor             │
  │  Component   │                                          │
  │  Tree        │  [ShaftCanvas / CellList]                │
  │              │                                          │
  ├──────────────┴──────────────────────────────────────────┤
  │  Status bar: ✓ valid | 1450 rpm | 2 bearings | 0 errors │
  └─────────────────────────────────────────────────────────┘

Signal routing (all in _connect_signals):
  CellList.cell_executed    → _on_cell_executed → canvas + tree update
  canvas.component_selected → tree.select_by_label
  tree.component_selected   → canvas.select_component

Separation guarantee:
  - Solvers are NEVER called here.
  - All exec() goes through runner.py via CellList.
  - Canvas and tree read from MechanicalSystem, never from solvers.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QSplitter,
    QStatusBar, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from runner import CellRunner
from code_editor import CellList
from shaft_canvas import ShaftCanvas
from component_tree import ComponentTree


# ---------------------------------------------------------------------------
# Application stylesheet — dark theme, no external dependencies
# ---------------------------------------------------------------------------

_APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
}
QMenuBar {
    background: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item:selected {
    background: #313244;
}
QMenu {
    background: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
}
QMenu::item:selected {
    background: #313244;
}
QTabWidget::pane {
    border: 1px solid #313244;
    background: #1e1e2e;
}
QTabBar::tab {
    background: #181825;
    color: #6c7086;
    padding: 6px 16px;
    border: 1px solid #313244;
    border-bottom: none;
}
QTabBar::tab:selected {
    background: #1e1e2e;
    color: #cdd6f4;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover {
    color: #cdd6f4;
}
QSplitter::handle {
    background: #313244;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}
QStatusBar {
    background: #181825;
    color: #6c7086;
    border-top: 1px solid #313244;
    font-size: 11px;
}
QPushButton {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 12px;
}
QPushButton:hover {
    background: #45475a;
}
QPushButton:pressed {
    background: #585b70;
}
QScrollBar:vertical {
    background: #181825;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #181825;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 4px;
}
"""


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """AxisForge main application window."""

    def __init__(self) -> None:
        super().__init__()

        self._runner = CellRunner()
        self._last_validation_errors: list[str] = []

        self.setWindowTitle("AxisForge — Shaft · Bearing · Gear Analysis")
        self.setMinimumSize(1100, 650)
        self.resize(1280, 760)

        self._build_ui()
        self._build_menu()
        self._build_statusbar()
        self._connect_signals()

        self._status_idle()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        # Root layout
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Toolbar
        toolbar = self._build_toolbar()
        root.addWidget(toolbar)

        # Main splitter: tree | content
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(3)

        # Left panel — component tree
        self._tree = ComponentTree()
        self._tree.setMinimumWidth(180)
        self._tree.setMaximumWidth(320)
        self._main_splitter.addWidget(self._tree)

        # Right panel — tabs (canvas / editor)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Canvas tab
        self._canvas = ShaftCanvas()
        self._tabs.addTab(self._canvas, "  Canvas  ")

        # Editor tab
        self._cell_list = CellList(self._runner)
        self._tabs.addTab(self._cell_list, "  Editor  ")

        self._main_splitter.addWidget(self._tabs)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([220, 1060])

        root.addWidget(self._main_splitter)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet("background: #181825; border-bottom: 1px solid #313244;")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Logo / title
        title = QLabel("⚙ AxisForge")
        title.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa;")
        layout.addWidget(title)
        layout.addStretch()

        # Run All button
        self._run_all_btn = QPushButton("▶  Run All")
        self._run_all_btn.setToolTip("Execute all cells in order (Ctrl+Shift+Enter)")
        self._run_all_btn.setStyleSheet(
            "QPushButton { background: #1e4a7a; color: #89b4fa; "
            "border: 1px solid #89b4fa; border-radius: 4px; padding: 4px 14px; }"
            "QPushButton:hover { background: #2a5a8a; }"
        )
        self._run_all_btn.clicked.connect(self._on_run_all)

        # Reset button
        self._reset_btn = QPushButton("⟳  Reset")
        self._reset_btn.setToolTip("Clear namespace and reset all cells")
        self._reset_btn.clicked.connect(self._on_reset)

        # Switch to editor on run
        self._editor_btn = QPushButton("Editor →")
        self._editor_btn.setToolTip("Switch to editor tab")
        self._editor_btn.clicked.connect(
            lambda: self._tabs.setCurrentIndex(1)
        )

        layout.addWidget(self._editor_btn)
        layout.addWidget(self._reset_btn)
        layout.addWidget(self._run_all_btn)

        return bar

    def _build_menu(self) -> None:
        menu = self.menuBar()

        # File
        file_menu = menu.addMenu("File")
        file_menu.addAction(self._make_action(
            "New Session", self._on_reset, "Ctrl+N"
        ))
        file_menu.addSeparator()
        file_menu.addAction(self._make_action(
            "Exit", self.close, "Ctrl+Q"
        ))

        # Run
        run_menu = menu.addMenu("Run")
        run_menu.addAction(self._make_action(
            "Run All Cells", self._on_run_all, "Ctrl+Shift+Return"
        ))

        # View
        view_menu = menu.addMenu("View")
        view_menu.addAction(self._make_action(
            "Canvas", lambda: self._tabs.setCurrentIndex(0), "Ctrl+1"
        ))
        view_menu.addAction(self._make_action(
            "Editor", lambda: self._tabs.setCurrentIndex(1), "Ctrl+2"
        ))

    def _build_statusbar(self) -> None:
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        self._status_valid = QLabel()
        self._status_speed = QLabel()
        self._status_components = QLabel()
        self._status_errors = QLabel()

        for w in (self._status_valid, self._status_speed,
                  self._status_components, self._status_errors):
            w.setFont(QFont("Consolas", 10))
            self._statusbar.addWidget(w)
            self._statusbar.addWidget(self._sep())

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Cell executed → refresh canvas + tree + status
        self._cell_list.cell_executed.connect(self._on_cell_executed)

        # Canvas click → tree selection
        self._canvas.component_selected.connect(self._on_canvas_selected)

        # Tree click → canvas selection
        self._tree.component_selected.connect(self._on_tree_selected)

        # Cell selected → switch to editor tab (if not already)
        self._cell_list.cell_selected.connect(
            lambda _: self._tabs.setCurrentIndex(1)
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_cell_executed(self, cell_index: int, _result) -> None:
        """Refresh visual state after any cell runs."""
        systems = self._runner.get_systems()
        self._canvas.render_systems(systems)
        self._tree.update_systems(systems)
        self._update_status(systems)

        # Auto-switch to canvas if a system appeared
        if systems and self._tabs.currentIndex() == 1:
            # Briefly flash canvas tab indicator — don't force switch
            pass

    def _on_run_all(self) -> None:
        self._cell_list.run_all()

    def _on_reset(self) -> None:
        self._runner.reset()
        self._canvas.clear()
        self._tree.update_systems([])
        self._status_idle()

    def _on_canvas_selected(self, label: str) -> None:
        if label:
            self._tree.select_by_label(label)

    def _on_tree_selected(self, label: str) -> None:
        if label:
            self._canvas.select_component(label)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _update_status(self, systems: list) -> None:
        if not systems:
            self._status_idle()
            return

        # Aggregate across all systems
        total_bearings = sum(len(s.bearings) for s in systems)
        total_gears    = sum(len(s.gears)    for s in systems)
        rpms           = [s.speed_rpm for s in systems if s.speed_rpm > 0]
        all_errors: list[str] = []
        for s in systems:
            all_errors.extend(s.validate())

        valid = len(all_errors) == 0
        self._status_valid.setText(
            "✓ Valid" if valid else f"✗ {len(all_errors)} error(s)"
        )
        self._status_valid.setStyleSheet(
            "color: #a6e3a1;" if valid else "color: #f38ba8;"
        )

        speed_txt = f"{rpms[0]:.0f} rpm" if rpms else "— rpm"
        self._status_speed.setText(speed_txt)
        self._status_speed.setStyleSheet("color: #6c7086;")

        self._status_components.setText(
            f"{total_bearings} bearing(s)  {total_gears} gear(s)"
        )
        self._status_components.setStyleSheet("color: #6c7086;")

        if all_errors:
            self._status_errors.setText(all_errors[0][:60])
            self._status_errors.setStyleSheet("color: #f38ba8;")
        else:
            self._status_errors.setText("")

    def _status_idle(self) -> None:
        self._status_valid.setText("○ No system")
        self._status_valid.setStyleSheet("color: #6c7086;")
        self._status_speed.setText("—")
        self._status_components.setText("")
        self._status_errors.setText("")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_action(label: str, slot, shortcut: str | None = None) -> QAction:
        action = QAction(label)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        return action

    @staticmethod
    def _sep() -> QLabel:
        sep = QLabel("  |  ")
        sep.setStyleSheet("color: #313244;")
        return sep


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("AxisForge")
    app.setStyleSheet(_APP_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
