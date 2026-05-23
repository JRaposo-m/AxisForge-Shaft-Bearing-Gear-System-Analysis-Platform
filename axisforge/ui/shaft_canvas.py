"""
ui/shaft_canvas.py
2D shaft schematic canvas — QPainter rendering, single XY view.

Renders:
  - Shaft sections (cylinders in elevation, diameter to scale)
  - Bearings (ISO 1101-style triangle symbols)
  - Gear elements (circle with pitch diameter, force arrow)
  - Radial/axial loads (arrows with magnitude label)
  - Reaction labels (R_A, R_B)

Design:
  - Reads MechanicalSystem directly — no intermediate ViewModel yet.
  - Selection: component label → highlighted in yellow.
  - Hover: shows tooltip with component properties.
  - No matplotlib — pure QPainter for native interactivity.
  - Multi-shaft ready: render(systems: list[MechanicalSystem]) API.

Phase 2 scope: single XY view. XZ and top views deferred to Phase 3.
"""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import (
    QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
    QFontMetrics,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_C_BG           = QColor("#1e1e2e")
_C_SHAFT        = QColor("#585b70")
_C_SHAFT_FILL   = QColor("#313244")
_C_BEARING      = QColor("#89b4fa")
_C_GEAR         = QColor("#a6e3a1")
_C_LOAD         = QColor("#f9e2af")
_C_REACTION     = QColor("#cba6f7")
_C_AXIS         = QColor("#45475a")
_C_TEXT         = QColor("#cdd6f4")
_C_TEXT_MUTED   = QColor("#6c7086")
_C_HIGHLIGHT    = QColor("#f9e2af")
_C_HOVER        = QColor("#f5c2e7")


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_MARGIN_H   = 60    # px left/right margin
_MARGIN_V   = 40    # px top/bottom margin
_AXIS_Y     = 0.5   # fraction of canvas height for shaft centreline
_MAX_D_PX   = 80    # max diameter in pixels (largest section)
_ARROW_LEN  = 40    # force arrow length px
_FONT_SIZE  = 9


# ---------------------------------------------------------------------------
# Hit area record
# ---------------------------------------------------------------------------

class _HitArea:
    """Associates a drawn rectangle with a component label."""
    __slots__ = ("rect", "label", "tooltip")

    def __init__(self, rect: QRectF, label: str, tooltip: str) -> None:
        self.rect = rect
        self.label = label
        self.tooltip = tooltip


# ---------------------------------------------------------------------------
# ShaftCanvas
# ---------------------------------------------------------------------------

class ShaftCanvas(QWidget):
    """
    2D shaft schematic — XY plane.

    Signals
    -------
    component_selected(str)
        Emitted when user clicks a component. Carries component label.
        Empty string if background clicked (deselect).
    """

    component_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._systems: list[Any] = []          # list[MechanicalSystem]
        self._selected_label: str = ""
        self._hovered_label: str = ""
        self._hit_areas: list[_HitArea] = []

        self.setMinimumHeight(200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMouseTracking(True)
        self.setStyleSheet(f"background: #1e1e2e;")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_systems(self, systems: list[Any]) -> None:
        """Update the canvas with a new list of MechanicalSystem objects."""
        self._systems = systems
        self._hit_areas.clear()
        self.update()

    def select_component(self, label: str) -> None:
        """Highlight a component by label (called from tree/editor)."""
        self._selected_label = label
        self.update()

    def clear(self) -> None:
        self._systems = []
        self._hit_areas.clear()
        self._selected_label = ""
        self.update()

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), _C_BG)

        if not self._systems:
            self._draw_empty(painter)
            return

        # Rebuild hit areas on every paint (canvas may have resized)
        self._hit_areas.clear()

        for system in self._systems:
            self._draw_system(painter, system)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        hit = self._hit_at(pos)
        if hit:
            self._hovered_label = hit.label
            QToolTip.showText(event.globalPosition().toPoint(), hit.tooltip, self)
        else:
            self._hovered_label = ""
            QToolTip.hideText()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        hit = self._hit_at(pos)
        if hit:
            self._selected_label = hit.label
            self.component_selected.emit(hit.label)
        else:
            self._selected_label = ""
            self.component_selected.emit("")
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(600, 220)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_empty(self, painter: QPainter) -> None:
        painter.setPen(QPen(_C_TEXT_MUTED))
        painter.setFont(QFont("Consolas", 11))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "No system defined.\nRun a cell with MechanicalSystem to render.",
        )

    def _draw_system(self, painter: QPainter, system: Any) -> None:
        w = self.width()
        h = self.height()

        shaft = system.shaft
        total_mm = shaft.total_length
        if total_mm <= 0:
            return

        # Layout parameters
        x0 = _MARGIN_H                    # canvas x for shaft left end
        x1 = w - _MARGIN_H               # canvas x for shaft right end
        canvas_span = x1 - x0            # pixels available for shaft length
        cy = int(h * _AXIS_Y)            # centreline y

        def mm_to_px(mm: float) -> float:
            return x0 + (mm / total_mm) * canvas_span

        # Max diameter for scaling
        max_d = max((s.diameter for s in shaft.sections), default=50.0)

        def d_to_px(d_mm: float) -> float:
            return max(6.0, (d_mm / max_d) * _MAX_D_PX)

        # ---- Axis line ----
        pen = QPen(_C_AXIS, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x0), cy, int(x1), cy)

        # ---- Shaft sections ----
        cx = x0
        for section in shaft.sections:
            sx = cx
            ex = cx + (section.length / total_mm) * canvas_span
            r_px = d_to_px(section.diameter) / 2

            rect = QRectF(sx, cy - r_px, ex - sx, 2 * r_px)
            painter.setBrush(QBrush(_C_SHAFT_FILL))
            painter.setPen(QPen(_C_SHAFT, 1.5))
            painter.drawRect(rect)

            # Diameter label
            label_text = f"⌀{section.diameter:.0f}"
            painter.setPen(QPen(_C_TEXT_MUTED))
            painter.setFont(QFont("Consolas", _FONT_SIZE - 1))
            painter.drawText(
                QRectF(sx, cy + r_px + 2, ex - sx, 14),
                Qt.AlignmentFlag.AlignHCenter,
                label_text,
            )

            cx = ex

        # ---- Bearings ----
        for bearing in system.bearings:
            bx = mm_to_px(bearing.position)
            r_px = d_to_px(
                self._section_diameter_at(shaft, bearing.position)
            ) / 2
            selected = bearing.label == self._selected_label
            hovered  = bearing.label == self._hovered_label
            self._draw_bearing(painter, bx, cy, r_px, bearing.label,
                               bearing.arrangement, selected, hovered)
            # Hit area
            hit_rect = QRectF(bx - 14, cy - r_px - 20, 28, r_px + 36)
            tip = (f"Bearing {bearing.label}\n"
                   f"Position: {bearing.position:.1f} mm\n"
                   f"C = {bearing.C:.0f} N  C₀ = {bearing.C0:.0f} N\n"
                   f"Arrangement: {bearing.arrangement}")
            self._hit_areas.append(_HitArea(hit_rect, bearing.label, tip))

        # ---- Gear elements ----
        for gear in system.gears:
            gx = mm_to_px(gear.position)
            r_px = d_to_px(
                self._section_diameter_at(shaft, gear.position)
            ) / 2
            selected = gear.label == self._selected_label
            hovered  = gear.label == self._hovered_label
            self._draw_gear(painter, gx, cy, r_px, gear.label, selected, hovered)
            hit_rect = QRectF(gx - 20, cy - r_px - 12, 40, 2 * r_px + 24)
            tip = (f"Gear {gear.label}\n"
                   f"Position: {gear.position:.1f} mm\n"
                   f"Wt = {gear.tangential_force:.0f} N\n"
                   f"Wr = {gear.radial_force:.0f} N\n"
                   f"d = {gear.pitch_diameter:.1f} mm")
            self._hit_areas.append(_HitArea(hit_rect, gear.label, tip))

        # ---- Dimension line ----
        self._draw_dimension(painter, x0, x1, cy + _MAX_D_PX // 2 + 24,
                             total_mm)

        # ---- System name ----
        painter.setPen(QPen(_C_TEXT_MUTED))
        painter.setFont(QFont("Consolas", _FONT_SIZE))
        painter.drawText(int(x0), 16, system.name)

    def _draw_bearing(self, painter: QPainter, cx: float, cy: float,
                      shaft_r: float, label: str, arrangement: str,
                      selected: bool, hovered: bool) -> None:
        """Bearing symbol: square with X, centred on shaft at cx, above and below."""
        color = _C_HIGHLIGHT if selected else (_C_HOVER if hovered else _C_BEARING)
        half = 14  # half-size of square in px

        # Square spans the shaft vertically — top above centreline, bottom below
        top_y = cy - shaft_r - half
        rect = QRectF(cx - half, top_y, 2 * half, 2 * half)

        painter.setPen(QPen(color, 2.0))
        painter.setBrush(QBrush(color.darker(220)))
        painter.drawRect(rect)

        # X inside the square
        painter.drawLine(QPointF(rect.left(), rect.top()),
                         QPointF(rect.right(), rect.bottom()))
        painter.drawLine(QPointF(rect.right(), rect.top()),
                         QPointF(rect.left(), rect.bottom()))

        # Label below square
        painter.setPen(QPen(color))
        painter.setFont(QFont("Consolas", _FONT_SIZE, QFont.Weight.Bold))
        painter.drawText(
            QRectF(cx - 20, rect.bottom() + 3, 40, 14),
            Qt.AlignmentFlag.AlignHCenter,
            label,
        )

    def _draw_gear(self, painter: QPainter, cx: float, cy: float,
                   shaft_r: float, label: str,
                   selected: bool, hovered: bool) -> None:
        """Gear symbol: rectangle wider than the shaft, centred on cx."""
        color = _C_HIGHLIGHT if selected else (_C_HOVER if hovered else _C_GEAR)

        gear_half_w = 18   # half-width of gear rectangle px
        gear_half_h = shaft_r + 10  # slightly taller than shaft

        rect = QRectF(cx - gear_half_w, cy - gear_half_h,
                      2 * gear_half_w, 2 * gear_half_h)

        painter.setPen(QPen(color, 2.0))
        painter.setBrush(QBrush(color.darker(260)))
        painter.drawRect(rect)

        # Label above rectangle
        painter.setPen(QPen(color))
        painter.setFont(QFont("Consolas", _FONT_SIZE, QFont.Weight.Bold))
        painter.drawText(
            QRectF(cx - 20, rect.top() - 14, 40, 14),
            Qt.AlignmentFlag.AlignHCenter,
            label,
        )

    @staticmethod
    def _draw_arrowhead(painter: QPainter, x: float, y: float,
                        direction: str, color: QColor) -> None:
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        size = 5
        path = QPainterPath()
        if direction == "down":
            path.moveTo(x, y)
            path.lineTo(x - size, y - size * 1.6)
            path.lineTo(x + size, y - size * 1.6)
        elif direction == "up":
            path.moveTo(x, y)
            path.lineTo(x - size, y + size * 1.6)
            path.lineTo(x + size, y + size * 1.6)
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    @staticmethod
    def _draw_dimension(painter: QPainter, x0: float, x1: float,
                        y: float, length_mm: float) -> None:
        """Horizontal dimension line with total shaft length."""
        painter.setPen(QPen(_C_TEXT_MUTED, 1))
        painter.setFont(QFont("Consolas", _FONT_SIZE - 1))
        tick = 4
        painter.drawLine(QPointF(x0, y - tick), QPointF(x0, y + tick))
        painter.drawLine(QPointF(x1, y - tick), QPointF(x1, y + tick))
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        mid = (x0 + x1) / 2
        painter.drawText(
            QRectF(mid - 40, y + 2, 80, 14),
            Qt.AlignmentFlag.AlignHCenter,
            f"{length_mm:.0f} mm",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hit_at(self, pos: QPointF) -> _HitArea | None:
        for hit in reversed(self._hit_areas):  # topmost first
            if hit.rect.contains(pos):
                return hit
        return None

    @staticmethod
    def _section_diameter_at(shaft: Any, position: float) -> float:
        """Return diameter of shaft section at axial position [mm]."""
        cumulative = 0.0
        for section in shaft.sections:
            cumulative += section.length
            if position <= cumulative + 1e-6:
                return section.diameter
        return shaft.sections[-1].diameter if shaft.sections else 50.0