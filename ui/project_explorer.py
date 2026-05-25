"""
ui/project_explorer.py
ProjectExplorer — left dock widget showing the project folder tree.

Behaviour:
  - Empty state until set_root() is called (no filesystem shown by default).
  - Displays the project folder using QFileSystemModel (real filesystem).
  - No system icons — custom delegate draws plain text only.
  - Double-click on .py  → emits file_open_requested(path).
  - Double-click on .txt → emits txt_open_requested(path).
  - Right-click context menu → New File / New Folder / Rename / Copy Path.
  - QFileSystemWatcher monitors root; tree refreshes automatically when
    files change externally (VSCode, terminal, etc.).

Signals:
  file_open_requested(str)   .py double-clicked — absolute path
  txt_open_requested(str)    .txt double-clicked — absolute path
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    QDir, QFileSystemWatcher, QModelIndex,
    QSortFilterProxyModel, Qt, Signal,
)
from PySide6.QtGui import QAction, QColor, QCursor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QFileSystemModel, QInputDialog,
    QLabel, QMenu, QMessageBox, QApplication,
    QStyledItemDelegate, QTreeView, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Catppuccin Mocha palette
# ---------------------------------------------------------------------------

_BG_BASE     = "#1e1e2e"
_BG_MANTLE   = "#181825"
_BG_SURFACE  = "#313244"
_FG_TEXT     = "#cdd6f4"
_FG_SUBTEXT  = "#6c7086"
_FG_BLUE     = "#89b4fa"

_FONT_SIZE   = 11


# ---------------------------------------------------------------------------
# NoIconDelegate — strips system icons, renders plain text only
# ---------------------------------------------------------------------------

class NoIconDelegate(QStyledItemDelegate):
    """
    Item delegate that suppresses the system file/folder icon.
    Renders only the display text in the correct colour.
    """

    def initStyleOption(self, option, index) -> None:  # noqa: N802
        super().initStyleOption(option, index)
        # Remove icon entirely
        option.icon = None  # type: ignore[assignment]
        option.decorationSize.setWidth(0)
        option.decorationSize.setHeight(0)


# ---------------------------------------------------------------------------
# ProjectExplorer
# ---------------------------------------------------------------------------

class ProjectExplorer(QWidget):
    """
    Filesystem tree rooted at a project folder.

    Empty until set_root() is called.
    """

    file_open_requested = Signal(str)
    txt_open_requested  = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._root_path: str = ""
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_fs_changed)

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, path: str | Path) -> None:
        """Set the project root folder. Does nothing if path does not exist."""
        p = Path(path)
        if not p.is_dir():
            return

        root_str = str(p)

        if self._watcher.directories():
            self._watcher.removePaths(self._watcher.directories())

        self._root_path = root_str
        root_index = self._model.setRootPath(root_str)
        proxy_index = self._proxy.mapFromSource(root_index)
        self._tree.setRootIndex(proxy_index)
        self._tree.setVisible(True)
        self._placeholder.setVisible(False)
        self._watcher.addPath(root_str)

    def root_path(self) -> str:
        return self._root_path

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Placeholder shown when no project is open
        self._placeholder = QLabel("No project open\nFile → Open Project Folder…")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_FG_SUBTEXT}; font-size: 11px; padding: 24px;"
        )
        root.addWidget(self._placeholder)

        # Filesystem model — root set later by set_root()
        self._model = QFileSystemModel(self)
        self._model.setRootPath("")   # required but tree root set separately
        self._model.setFilter(
            QDir.Filter.AllDirs
            | QDir.Filter.Files
            | QDir.Filter.NoDotAndDotDot
        )

        # Proxy for sorting
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        # Tree view
        self._tree = QTreeView(self)
        self._tree.setModel(self._proxy)
        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        # No icons — plain text only
        self._tree.setItemDelegate(NoIconDelegate(self._tree))

        # Hide all columns except Name
        self._tree.setColumnHidden(1, True)
        self._tree.setColumnHidden(2, True)
        self._tree.setColumnHidden(3, True)
        self._tree.header().setVisible(False)

        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._tree.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._tree.setUniformRowHeights(True)
        self._tree.setIndentation(14)
        self._tree.setAnimated(False)

        # Hidden until set_root() called
        self._tree.setVisible(False)
        root.addWidget(self._tree)

        # Connect
        self._tree.doubleClicked.connect(self._on_double_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background: {_BG_BASE};
                color: {_FG_TEXT};
                font-family: Consolas, 'Courier New', monospace;
                font-size: {_FONT_SIZE}px;
            }}
            QTreeView {{
                background: {_BG_BASE};
                color: {_FG_TEXT};
                border: none;
                selection-background-color: {_BG_SURFACE};
                selection-color: {_FG_TEXT};
            }}
            QTreeView::item {{ padding: 2px 0; }}
            QTreeView::item:hover {{ background: {_BG_MANTLE}; }}
            QTreeView::branch {{ background: {_BG_BASE}; }}
            QScrollBar:vertical {{
                background: {_BG_MANTLE};
                width: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {_BG_SURFACE};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_double_clicked(self, proxy_index: QModelIndex) -> None:
        source_index = self._proxy.mapToSource(proxy_index)
        file_path = self._model.filePath(source_index)
        if self._model.isDir(source_index):
            return
        suffix = Path(file_path).suffix.lower()
        if suffix == ".py":
            self.file_open_requested.emit(file_path)
        elif suffix == ".txt":
            self.txt_open_requested.emit(file_path)

    def _on_context_menu(self, pos) -> None:
        proxy_index = self._tree.indexAt(pos)
        source_index = self._proxy.mapToSource(proxy_index)
        is_valid = proxy_index.isValid()

        if is_valid:
            file_path = self._model.filePath(source_index)
            is_dir = self._model.isDir(source_index)
            parent_dir = file_path if is_dir else str(Path(file_path).parent)
        else:
            file_path = ""
            is_dir = False
            parent_dir = self._root_path

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background: {_BG_MANTLE};
                color: {_FG_TEXT};
                border: 1px solid {_BG_SURFACE};
                font-size: {_FONT_SIZE}px;
            }}
            QMenu::item:selected {{ background: {_BG_SURFACE}; }}
            QMenu::separator {{ background: {_BG_SURFACE}; height: 1px; margin: 3px 0; }}
        """)

        act_new_file = QAction("New File", self)
        act_new_file.triggered.connect(lambda: self._new_file(parent_dir))
        menu.addAction(act_new_file)

        act_new_folder = QAction("New Folder", self)
        act_new_folder.triggered.connect(lambda: self._new_folder(parent_dir))
        menu.addAction(act_new_folder)

        if is_valid:
            menu.addSeparator()
            act_rename = QAction("Rename", self)
            act_rename.triggered.connect(lambda: self._rename(file_path))
            menu.addAction(act_rename)
            menu.addSeparator()
            act_copy = QAction("Copy Path", self)
            act_copy.triggered.connect(
                lambda: QApplication.clipboard().setText(file_path)
            )
            menu.addAction(act_copy)

        menu.exec(QCursor.pos())

    def _on_fs_changed(self, _path: str) -> None:
        pass   # QFileSystemModel handles refresh internally

    # ------------------------------------------------------------------
    # Context menu actions
    # ------------------------------------------------------------------

    def _new_file(self, parent_dir: str) -> None:
        name, ok = QInputDialog.getText(
            self, "New File", "File name:", text="untitled.py"
        )
        if not ok or not name.strip():
            return
        dest = Path(parent_dir) / name.strip()
        if dest.exists():
            QMessageBox.warning(self, "New File", f"'{dest.name}' already exists.")
            return
        try:
            dest.touch()
        except OSError as e:
            QMessageBox.critical(self, "New File", str(e))

    def _new_folder(self, parent_dir: str) -> None:
        name, ok = QInputDialog.getText(
            self, "New Folder", "Folder name:", text="new_folder"
        )
        if not ok or not name.strip():
            return
        dest = Path(parent_dir) / name.strip()
        try:
            dest.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            QMessageBox.warning(self, "New Folder", f"'{dest.name}' already exists.")
        except OSError as e:
            QMessageBox.critical(self, "New Folder", str(e))

    def _rename(self, file_path: str) -> None:
        old = Path(file_path)
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=old.name
        )
        if not ok or not new_name.strip() or new_name.strip() == old.name:
            return
        dest = old.parent / new_name.strip()
        if dest.exists():
            QMessageBox.warning(self, "Rename", f"'{dest.name}' already exists.")
            return
        try:
            old.rename(dest)
        except OSError as e:
            QMessageBox.critical(self, "Rename", str(e))