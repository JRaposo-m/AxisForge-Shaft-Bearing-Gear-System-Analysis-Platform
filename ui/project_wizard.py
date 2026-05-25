"""
ui/project_wizard.py
New Project wizard dialog — minimal PySide6.

Fields:
  - Project name (QLineEdit)
  - Parent folder (QLineEdit + Browse button)

Returns (name, parent_path) via .result_data after exec() == Accepted.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout,
)


class NewProjectWizard(QDialog):
    """
    Simple dialog: Project Name + parent folder.

    Usage
    -----
    wizard = NewProjectWizard(parent=self)
    if wizard.exec() == QDialog.DialogCode.Accepted:
        name, path = wizard.result_data
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(420)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        self.result_data: tuple[str, Path] | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Project name
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. demo_reducer_1stage")
        form.addRow("Project name:", self._name_edit)

        # Parent folder
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Choose parent folder…")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse_btn)

        folder_widget_container = QLabel()  # dummy — addRow accepts widget only
        form.addRow("Location:", self._make_row_widget(self._folder_edit, browse_btn))

        layout.addLayout(form)

        # Preview label
        self._preview = QLabel()
        self._preview.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._preview.setWordWrap(True)
        layout.addWidget(self._preview)

        self._name_edit.textChanged.connect(self._update_preview)
        self._folder_edit.textChanged.connect(self._update_preview)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _make_row_widget(line_edit: QLineEdit, btn: QPushButton):
        """Wrap QLineEdit + QPushButton into a single widget for QFormLayout."""
        from PySide6.QtWidgets import QWidget
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit)
        row.addWidget(btn)
        return container

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select parent folder", str(Path.home())
        )
        if folder:
            self._folder_edit.setText(folder)

    def _update_preview(self) -> None:
        name = self._name_edit.text().strip()
        folder = self._folder_edit.text().strip()
        if name and folder:
            preview_path = Path(folder) / name
            self._preview.setText(f"Will create: {preview_path}")
        else:
            self._preview.setText("")

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        folder = self._folder_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "New Project", "Project name cannot be empty.")
            return

        # Basic name validation — no path separators, no spaces
        invalid_chars = set(r'/\:*?"<>| ')
        bad = [c for c in name if c in invalid_chars]
        if bad:
            QMessageBox.warning(
                self, "New Project",
                f"Project name contains invalid characters: {' '.join(set(bad))}"
            )
            return

        if not folder:
            QMessageBox.warning(self, "New Project", "Please choose a parent folder.")
            return

        parent = Path(folder)
        if not parent.is_dir():
            QMessageBox.warning(
                self, "New Project",
                f"Parent folder does not exist:\n{parent}"
            )
            return

        if (parent / name).exists():
            QMessageBox.warning(
                self, "New Project",
                f"A directory named '{name}' already exists in:\n{parent}"
            )
            return

        self.result_data = (name, parent)
        self.accept()
