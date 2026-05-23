"""
ui/component_tree.py
Component tree panel — reads from MechanicalSystem, emits selection signals.

Design:
  - QTreeWidget populated from MechanicalSystem instances.
  - No internal state — tree rebuilt from system on every update.
  - Selection is bidirectional: selecting here emits component_selected(label),
    which main_window routes to canvas.select_component().
  - External selection (from canvas click) calls select_by_label() here.
  - Supports multiple systems (multi-shaft roadmap, Ano 2).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QFrame, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Colours (Catppuccin Mocha)
# ---------------------------------------------------------------------------

_C_BG       = "#1e1e2e"
_C_HEADER   = "#181825"
_C_TEXT     = "#cdd6f4"
_C_MUTED    = "#6c7086"
_C_BEARING  = "#89b4fa"
_C_GEAR     = "#a6e3a1"
_C_LOAD     = "#f9e2af"
_C_SHAFT    = "#585b70"
_C_SYSTEM   = "#cba6f7"
_C_SELECT   = "#313244"


# ---------------------------------------------------------------------------
# Item categories
# ---------------------------------------------------------------------------

_CAT_SYSTEM   = "system"
_CAT_SHAFT    = "shaft"
_CAT_SECTIONS = "sections"
_CAT_BEARING  = "bearing"
_CAT_GEAR     = "gear"
_CAT_LOAD     = "load"


# ---------------------------------------------------------------------------
# ComponentTree
# ---------------------------------------------------------------------------

class ComponentTree(QWidget):
    """
    Hierarchical view of all MechanicalSystem components.

    Signals
    -------
    component_selected(str)
        Label of the selected component, or "" for deselect / category nodes.
    """

    component_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._systems: list[Any] = []
        self._label_to_item: dict[str, QTreeWidgetItem] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_systems(self, systems: list[Any]) -> None:
        """Rebuild tree from a new list of MechanicalSystem instances."""
        self._systems = systems
        self._label_to_item.clear()
        self._tree.clear()

        if not systems:
            self._show_placeholder()
            return

        self._placeholder.hide()
        self._tree.show()

        for system in systems:
            self._add_system_node(system)

        self._tree.expandAll()

    def select_by_label(self, label: str) -> None:
        """
        Highlight a tree item by component label.
        Called from canvas or editor — does NOT re-emit component_selected.
        """
        self._tree.blockSignals(True)
        self._tree.clearSelection()
        if label in self._label_to_item:
            item = self._label_to_item[label]
            item.setSelected(True)
            self._tree.scrollToItem(item)
        self._tree.blockSignals(False)

    # ------------------------------------------------------------------
    # Private — UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QFrame()
        header.setFixedHeight(28)
        header.setStyleSheet(f"background: {_C_HEADER};")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(8, 4, 8, 4)
        title = QLabel("Components")
        title.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {_C_MUTED};")
        h_layout.addWidget(title)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setFont(QFont("Consolas", 10))
        self._tree.setIndentation(14)
        self._tree.setAnimated(True)
        self._tree.setStyleSheet(self._tree_stylesheet())
        self._tree.itemClicked.connect(self._on_item_clicked)

        # Placeholder
        self._placeholder = QLabel("No system.\nRun a cell to populate.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")

        root.addWidget(header)
        root.addWidget(self._tree)
        root.addWidget(self._placeholder)
        self._tree.hide()

    def _show_placeholder(self) -> None:
        self._tree.hide()
        self._placeholder.show()

    # ------------------------------------------------------------------
    # Private — tree population
    # ------------------------------------------------------------------

    def _add_system_node(self, system: Any) -> None:
        sys_item = QTreeWidgetItem([system.name])
        sys_item.setForeground(0, QColor(_C_SYSTEM))
        sys_item.setFont(0, QFont("Consolas", 10, QFont.Weight.Bold))
        sys_item.setData(0, Qt.ItemDataRole.UserRole, ("system", system.name))
        self._tree.addTopLevelItem(sys_item)

        # Shaft node
        shaft_item = self._category_node(sys_item, "⬛ Shaft", _C_SHAFT)
        shaft = system.shaft
        for i, section in enumerate(shaft.sections):
            label = section.label or f"§{i+1}"
            text = f"{label}  L={section.length:.0f} ⌀{section.diameter:.0f}"
            child = QTreeWidgetItem([text])
            child.setForeground(0, QColor(_C_SHAFT))
            child.setData(0, Qt.ItemDataRole.UserRole, (_CAT_SHAFT, label))
            shaft_item.addChild(child)

        # Bearings node
        if system.bearings:
            bear_item = self._category_node(sys_item, "◈ Bearings", _C_BEARING)
            for bearing in system.bearings:
                text = (f"{bearing.label}  x={bearing.position:.0f} mm  "
                        f"[{bearing.arrangement}]")
                child = QTreeWidgetItem([text])
                child.setForeground(0, QColor(_C_BEARING))
                child.setData(0, Qt.ItemDataRole.UserRole,
                              (_CAT_BEARING, bearing.label))
                bear_item.addChild(child)
                self._label_to_item[bearing.label] = child

        # Gears node
        if system.gears:
            gear_item = self._category_node(sys_item, "⚙ Gears", _C_GEAR)
            for gear in system.gears:
                text = (f"{gear.label}  x={gear.position:.0f} mm  "
                        f"Wt={gear.tangential_force:.0f} N")
                child = QTreeWidgetItem([text])
                child.setForeground(0, QColor(_C_GEAR))
                child.setData(0, Qt.ItemDataRole.UserRole,
                              (_CAT_GEAR, gear.label))
                gear_item.addChild(child)
                self._label_to_item[gear.label] = child

        # Loads node
        if system.loads:
            load_item = self._category_node(sys_item, "↓ Loads", _C_LOAD)
            for load in system.loads:
                lbl = getattr(load, "label", "") or type(load).__name__
                mag = getattr(load, "magnitude", 0)
                pos = getattr(load, "position", 0)
                text = f"{lbl}  x={pos:.0f} mm  {mag:.0f} N"
                child = QTreeWidgetItem([text])
                child.setForeground(0, QColor(_C_LOAD))
                child.setData(0, Qt.ItemDataRole.UserRole, (_CAT_LOAD, lbl))
                load_item.addChild(child)
                self._label_to_item[lbl] = child

        # System summary
        summary = QTreeWidgetItem([
            f"  {system.speed_rpm:.0f} rpm  "
            f"|  L = {shaft.total_length:.0f} mm"
        ])
        summary.setForeground(0, QColor(_C_MUTED))
        summary.setFont(0, QFont("Consolas", 9))
        sys_item.addChild(summary)

    @staticmethod
    def _category_node(parent: QTreeWidgetItem,
                       text: str, color: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([text])
        item.setForeground(0, QColor(color))
        item.setFont(0, QFont("Consolas", 10))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        parent.addChild(item)
        return item

    # ------------------------------------------------------------------
    # Private — selection
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        category, label = data
        if category in (_CAT_BEARING, _CAT_GEAR, _CAT_LOAD):
            self.component_selected.emit(label)
        else:
            self.component_selected.emit("")

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------

    @staticmethod
    def _tree_stylesheet() -> str:
        return f"""
            QTreeWidget {{
                background: {_C_BG};
                color: {_C_TEXT};
                border: none;
                outline: none;
            }}
            QTreeWidget::item {{
                padding: 2px 4px;
            }}
            QTreeWidget::item:selected {{
                background: {_C_SELECT};
                color: {_C_TEXT};
            }}
            QTreeWidget::item:hover {{
                background: #2a2a3e;
            }}
            QTreeWidget::branch {{
                background: {_C_BG};
            }}
        """
