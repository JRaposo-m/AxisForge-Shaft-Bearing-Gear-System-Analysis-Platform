"""
ui/main_window.py
AxisForge main application window.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  MenuBar                                                     │
  ├─────────────────┬────────────────────────────────────────────┤
  │                 │                                            │
  │  PROJECT        │         SCRIPT EDITOR                     │
  │  EXPLORER       │         (central widget)                  │
  │  (dock left)    │                                            │
  ├─────────────────┤                                            │
  │                 │                                            │
  │  WORKSPACE      │                                            │
  │  (dock left)    │                                            │
  │                 │                                            │
  ├─────────────────┴────────────────────────────────────────────┤
  │  OUTPUT CONSOLE  [Output] [Errors] [Log]   (dock bottom)    │
  └──────────────────────────────────────────────────────────────┘

Signal routing:
  ScriptEditor.section_executed  → OutputConsole.append_section_result
  ScriptEditor.section_executed  → _refresh_workspace
  ScriptEditor.file_saved        → OutputConsole.append_log
  ScriptEditor.file_changed      → (title update)
  ProjectExplorer.file_open_requested  → ScriptEditor.open_file
  ProjectExplorer.txt_open_requested   → _open_txt_viewer
  WorkspacePanel.inspect_requested     → _on_inspect_requested

Separation guarantee:
  - Solvers are NEVER called here.
  - MainWindow only routes signals and manages dock layout.
  - No engineering logic in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog,
    QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QStatusBar, QVBoxLayout, QWidget,
)

from runner import SectionResult, SectionRunner  # SectionResult used for type annotation
from script_editor import ScriptEditor
from output_console import OutputConsole
from workspace_panel import WorkspacePanel
from project_explorer import ProjectExplorer
from project_wizard import NewProjectWizard

try:
    import sys as _sys, pathlib as _pathlib
    _ui_dir    = _pathlib.Path(__file__).parent
    _repo_root = _ui_dir.parent
    _axisforge  = _repo_root / "axisforge"
    for _p in (str(_repo_root), str(_axisforge)):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    from core.project_gen import ProjectManager
except Exception:
    ProjectManager = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Application stylesheet
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
QMenuBar::item:selected { background: #313244; }
QMenu {
    background: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
}
QMenu::item:selected { background: #313244; }
QMenu::separator { background: #313244; height: 1px; margin: 3px 6px; }

QDockWidget {
    color: #cdd6f4;
    font-size: 11px;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background: #181825;
    padding: 4px 8px;
    border-bottom: 1px solid #313244;
    text-align: left;
}
QDockWidget::close-button, QDockWidget::float-button {
    border: none;
    background: transparent;
}

QStatusBar {
    background: #181825;
    color: #6c7086;
    border-top: 1px solid #313244;
    font-size: 11px;
}
QScrollBar:vertical {
    background: #181825;
    width: 6px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #181825;
    height: 6px;
}
QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 3px;
}
"""


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """AxisForge main application window."""

    def __init__(self) -> None:
        super().__init__()
        self._runner = SectionRunner()

        self.setWindowTitle("AxisForge")
        self.setMinimumSize(1100, 680)
        self.resize(1400, 820)

        self._build_central()
        self._build_docks()
        self._build_menu()
        self._build_statusbar()
        self._connect_signals()
        self._restore_layout()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_central(self) -> None:
        """ScriptEditor is the central widget — always visible."""
        self._editor = ScriptEditor(runner=self._runner)
        self.setCentralWidget(self._editor)

    def _build_docks(self) -> None:
        # --- Project Explorer (left, top) ---
        self._explorer = ProjectExplorer()
        self._dock_project = QDockWidget("Project")
        self._dock_project.setWidget(self._explorer)
        self._dock_project.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_project)

        # --- Workspace Panel (left, below project) ---
        self._workspace = WorkspacePanel()
        self._dock_workspace = QDockWidget("Workspace")
        self._dock_workspace.setWidget(self._workspace)
        self._dock_workspace.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_workspace)
        self.splitDockWidget(
            self._dock_project, self._dock_workspace,
            Qt.Orientation.Vertical
        )

        # --- Output Console (bottom) ---
        self._console = OutputConsole()
        self._dock_output = QDockWidget("Output")
        self._dock_output.setWidget(self._console)
        self._dock_output.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_output)

        # Default dock sizes
        self.resizeDocks(
            [self._dock_project, self._dock_workspace],
            [280, 200],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self._dock_output],
            [180],
            Qt.Orientation.Vertical,
        )

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("File")
        file_menu.addAction(self._action(
            "New Project…", self._new_project, "Ctrl+N"
        ))
        file_menu.addSeparator()
        file_menu.addAction(self._action(
            "Open Project Folder…", self._open_project, "Ctrl+Shift+O"
        ))
        file_menu.addAction(self._action(
            "Open File…", self._open_file, "Ctrl+O"
        ))
        file_menu.addAction(self._action(
            "Save", self._editor.save_file, "Ctrl+S"
        ))
        file_menu.addSeparator()
        file_menu.addAction(self._action("Exit", self.close, "Ctrl+Q"))

        # Run
        run_menu = mb.addMenu("Run")
        run_menu.addAction(self._action(
            "Run Section", self._editor._run_current_section, "Ctrl+Return"
        ))
        run_menu.addAction(self._action(
            "Run All", self._editor._run_all, "Ctrl+Shift+Return"
        ))
        run_menu.addSeparator()
        run_menu.addAction(self._action(
            "Reset Namespace", self._editor._reset
        ))

        # View
        view_menu = mb.addMenu("View")
        view_menu.addAction(self._action(
            "Project Explorer",
            lambda: self._toggle_dock(self._dock_project),
        ))
        view_menu.addAction(self._action(
            "Workspace",
            lambda: self._toggle_dock(self._dock_workspace),
        ))
        view_menu.addAction(self._action(
            "Output Console",
            lambda: self._toggle_dock(self._dock_output),
        ))

    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._status_file  = QLabel("")
        self._status_ns    = QLabel("")

        for w in (self._status_file, self._status_ns):
            w.setStyleSheet("color: #6c7086; font-size: 11px;")
            sb.addWidget(w)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Section executed → console + workspace
        self._editor.section_executed.connect(self._console.append_section_result)
        self._editor.section_executed.connect(self._on_section_executed)

        # File saved → log
        self._editor.file_saved.connect(
            lambda p: self._console.append_log(f"Saved: {p}")
        )

        # File changed → title
        self._editor.file_changed.connect(self._update_title)

        # Project explorer → editor
        self._explorer.file_open_requested.connect(self._open_py_file)
        self._explorer.txt_open_requested.connect(self._open_txt_viewer)

        # Workspace inspect → fetch obj and route
        self._workspace.inspect_requested.connect(self._on_inspect_requested)

        # Namespace reset → clear workspace + log
        self._editor.namespace_reset.connect(self._on_namespace_reset)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_section_executed(self, result: SectionResult) -> None:
        """Refresh workspace after section execution."""
        ns    = self._runner.get_namespace()
        stale = self._runner.stale_after(result.section_index)

        self._workspace.refresh(
            namespace=ns,
            stale_section_indices=stale,
            name_to_section=self._runner.get_name_to_section_map(),
        )
        self._update_status_ns(len(ns))
        self._scan_new_elements()

    def _on_namespace_reset(self) -> None:
        """Clear workspace and log after runner reset."""
        self._workspace.clear()
        self._console.append_log("Namespace reset.")
        self._update_status_ns(0)

    def _on_inspect_requested(self, name: str, _obj) -> None:
        """User double-clicked a workspace variable."""
        ns  = self._runner.get_namespace()
        obj = ns.get(name)
        if obj is None:
            return

        type_name = type(obj).__name__
        if type_name in {"Figure", "Axes", "AxesSubplot"}:
            self._workspace.open_plot(name, obj)
        else:
            self._console.append_inspect(name, repr(obj))
            self._console.switch_to_log()

    def _open_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Open Project Folder", str(Path.home())
        )
        if not folder:
            return
        import os
        os.chdir(folder)
        self._explorer.set_root(folder)
        self._console.append_log(f"Project: {folder}")
        self._status_file.setText(f"  {Path(folder).name}")
        # Re-seed create_gear_pair with new root
        self._runner.reset()

    def _new_project(self) -> None:
        wizard = NewProjectWizard(parent=self)
        if wizard.exec() != NewProjectWizard.DialogCode.Accepted:
            return
        name, parent = wizard.result_data

        if ProjectManager is None:
            QMessageBox.critical(
                self, "New Project",
                "ProjectManager unavailable — check axisforge/core/project_gen.py."
            )
            return

        try:
            project_root = ProjectManager.create_project(name, parent)
        except FileExistsError as exc:
            QMessageBox.warning(self, "New Project", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "New Project", f"Could not create project:\n{exc}")
            return

        import os
        os.chdir(project_root)
        self._explorer.set_root(str(project_root))
        self._console.append_log(f"Project created: {project_root}")
        self._status_file.setText(f"  {name}")

        # Re-seed namespace — create_gear_pair needs the new root
        self._runner.reset()

        # Open full_pipeline.py in the editor
        pipeline = project_root / "scripts" / "full_pipeline.py"
        if pipeline.exists():
            self._open_py_file(str(pipeline))

    def _scan_new_elements(self) -> None:
        """
        Compare MechanicalSystem components in the namespace against disk state.
        Generate skeleton files for any new bearings or gears detected.

        Runs synchronously in the UI thread after each section execution.
        Errors are logged to the OutputConsole — never raised.

        Convention: MechanicalSystem.name must match the shaft directory name
        (e.g. MechanicalSystem(name="shaft_1") → shafts/shaft_1/).
        """
        if ProjectManager is None:
            return

        import os
        root = Path(os.getcwd())
        if not (root / "shafts").is_dir():
            return  # not an AxisForge project root

        try:
            state = ProjectManager.scan_project(root)
        except Exception:
            return

        project_name = state.project_name  # read from project.axf
        systems = self._runner.get_systems()
        tree_dirty = False

        for sys_obj in systems:
            shaft_name = sys_obj.name

            for bearing in sys_obj.bearings:
                label = bearing.label
                if label in state.bearing_labels.get(shaft_name, set()):
                    continue
                try:
                    result = ProjectManager.generate_bearing_file(
                        shaft_name, label, root, project_name
                    )
                    self._console.append_log(
                        f"[ProjectGen] Gerado: {result.relative_to(root)}"
                    )
                    tree_dirty = True
                except FileExistsError as exc:
                    self._console.append_log(f"[ProjectGen] ERRO: {exc}")

            for gear in sys_obj.gears:
                label = gear.label
                if label in state.gear_labels.get(shaft_name, set()):
                    continue
                try:
                    result = ProjectManager.generate_gear_file(
                        shaft_name, label, root, project_name
                    )
                    self._console.append_log(
                        f"[ProjectGen] Gerado: {result.relative_to(root)}"
                    )
                    tree_dirty = True
                except FileExistsError as exc:
                    self._console.append_log(f"[ProjectGen] ERRO: {exc}")

        if tree_dirty:
            self._explorer.set_root(str(root))

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", str(Path.home()),
            "Python files (*.py);;All files (*)"
        )
        if path:
            self._open_py_file(path)

    def _open_py_file(self, path: str) -> None:
        self._editor.open_file(path)
        self._console.append_log(f"Opened: {path}")
        self._update_title()

    def _open_txt_viewer(self, path: str) -> None:
        """Open a .txt result file in a simple read-only dock or window."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Open", str(e))
            return

        win = QWidget(self, Qt.WindowType.Window)
        win.setWindowTitle(Path(path).name)
        win.resize(720, 540)
        layout = QVBoxLayout(win)
        layout.setContentsMargins(0, 0, 0, 0)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        view.setStyleSheet(
            "background: #181825; color: #cdd6f4; "
            "font-family: Consolas, monospace; font-size: 11px; border: none;"
        )
        layout.addWidget(view)
        win.show()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_title(self) -> None:
        fp = self._editor.file_path()
        name = Path(fp).name if fp else "untitled"
        modified = "● " if self._editor._is_modified else ""
        self.setWindowTitle(f"AxisForge — {modified}{name}")
        self._status_file.setText(f"  {name}")

    def _update_status_ns(self, count: int) -> None:
        self._status_ns.setText(
            f"  {count} variable{'s' if count != 1 else ''} in namespace"
        )

    def _toggle_dock(self, dock: QDockWidget) -> None:
        dock.setVisible(not dock.isVisible())

    def _restore_layout(self) -> None:
        settings = QSettings("AxisForge", "Layout")
        state = settings.value("windowState")
        geo   = settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        if state:
            self.restoreState(state)

    def closeEvent(self, event) -> None:  # noqa: N802
        settings = QSettings("AxisForge", "Layout")
        settings.setValue("windowState", self.saveState())
        settings.setValue("geometry",    self.saveGeometry())

        if self._editor._is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "The current file has unsaved changes. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()

    def _action(
        self,
        label: str,
        slot,
        shortcut: str | None = None,
    ) -> QAction:
        act = QAction(label, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        return act


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