"""
mechanical_system/systems/shaft_system.py

Single-shaft assembly container for AxisForge (parallel-axis transmissions).

Two classes:
  - GearElement : thin wrapper binding a gear geometry object to a kinematic
                  role ("driver"/"driven") on a shaft. Carries no geometry of
                  its own; `position` delegates to the underlying gear.
  - ShaftSystem : autonomous per-shaft container of bearings, gears and loads.
                  Has NO knowledge of other shafts. Its `shaft_position`
                  (y, z global offset of the shaft axis) is normally set by
                  GearSystem.resolve(), not by the user.

Conventions (AxisForge):
  - explicit __init__ (no dataclass)
  - validate() -> list[str], never raises internally
  - validate_or_raise() wrapper
  - SI-ish shop units: mm for positions, N for forces, N*mm for torque/moment
  - GUI-independent, numerically transparent
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from axisforge.config import MIN_BEARING_SEPARATION_MM, MIN_SHOULDER_CLEARANCE_MM, FLOATING_BEARING_CLEARANCE_MM

from axisforge.core.loads import (
    Load, RadialLoad, AxialLoad, TorqueLoad, ExternalMoment,
)

# Type-only imports: annotations are strings (PEP 563), so these never need to
# resolve at runtime — keeps ShaftSystem importable without dragging in the
# whole geometry/config stack.
if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.machine_elements.Shaft.shaft import Shaft
    from axisforge.core.machine_elements.Bearings.bearing import Bearing

# Default design life. Sourced from config when available; the fallback keeps
# the module importable standalone (e.g. for unit tests).
try:  # pragma: no cover
    from config import DEFAULT_DESIGN_LIFE_HOURS
except Exception:  # pragma: no cover
    DEFAULT_DESIGN_LIFE_HOURS = 20_000.0



# ===========================================================================
# GearElement
# ===========================================================================

class GearElement:
    """
    Binds a gear geometry object to its kinematic role on a shaft.

    Parameters
    ----------
    gear         : geometry object exposing `.position` (SpurGear / HelicalGear
                   / InternalGear). No geometry is duplicated here.
    role         : "driver" | "driven"
    rotation_dir : +1 / -1 — set explicitly ONLY for the kinematic SOURCE gear
                   of a GearSystem. Left None everywhere else and propagated by
                   GearSystem during resolve().
    label        : reporting tag.
    """

    _VALID_ROLES = ("driver", "driven")

    def __init__(self, gear, role: str,
                 rotation_dir: int | None = None, label: str = ""):
        self.gear = gear
        self.role = role
        self.rotation_dir = rotation_dir
        self.label = label

    @property
    def position(self) -> float:
        return self.gear.position

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or "GearElement"

        if self.role not in self._VALID_ROLES:
            errors.append(
                f"{tag}: role must be one of {self._VALID_ROLES}, got {self.role!r}"
            )
        if self.rotation_dir not in (None, 1, -1):
            errors.append(
                f"{tag}: rotation_dir must be +1, -1 or None, got {self.rotation_dir!r}"
            )
        # Delegate to the underlying gear if it can validate itself.
        gear_validate = getattr(self.gear, "validate", None)
        if callable(gear_validate):
            errors.extend(f"{tag}.gear: {e}" for e in gear_validate())
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    def __repr__(self) -> str:
        return (f"GearElement(role={self.role!r}, position={self.position:.2f} mm, "
                f"rotation_dir={self.rotation_dir!r}, label={self.label!r})")


# ===========================================================================
# ShaftSystem
# ===========================================================================

class ShaftSystem:
    """
    Autonomous single-shaft container.

    Retains ALL bearings / gears / loads placed on the shaft, persistently,
    for downstream per-shaft consumption by StaticsSolver, StressSolver and
    BearingLifeSolver. Nothing is summarised or discarded.
    """

    def __init__(self, shaft: "Shaft", name: str = "System_1",
                 speed_rpm: float = 0.0,
                 design_life_hours: float = DEFAULT_DESIGN_LIFE_HOURS,
                 shaft_position: tuple[float, float] = (0.0, 0.0),
                 shaft_origin_x: float = 0.0, label: str = ""):
        self.shaft = shaft
        self.name = name
        self.speed_rpm = speed_rpm
        self.design_life_hours = design_life_hours
        # (y, z) global offset of the shaft axis. Overwritten by
        # GearSystem.resolve(); the user rarely sets this directly.
        self.shaft_position = shaft_position
        self.shaft_origin_x = shaft_origin_x
        self.label = label

        self._bearings: list["Bearing"] = []
        self._gears: list[GearElement] = []
        self._loads: list[Load] = []

    # ------------------------------------------------------------------
    # Construction helpers (fail fast on out-of-bounds placement)
    # ------------------------------------------------------------------

    def _check_axial_bounds(self, position: float, what: str) -> None:
        L = self.shaft.total_length
        if not (0.0 <= position <= L):
            raise ValueError(
                f"{self.name}: {what} position {position:.4f} mm is outside "
                f"the shaft extent [0, {L:.4f}] mm"
            )

    def add_bearing(self, bearing: "Bearing") -> "ShaftSystem":
        self._check_axial_bounds(bearing.position, "bearing")
        self._bearings.append(bearing)
        return self

    def add_gear(self, gear: GearElement) -> "ShaftSystem":
        self._check_axial_bounds(gear.position, "gear")
        self._gears.append(gear)
        return self

    def add_load(self, load: Load) -> "ShaftSystem":
        self._check_axial_bounds(load.position, "load")
        self._loads.append(load)
        return self

    # ------------------------------------------------------------------
    # Gear-mesh load injection (called by GearSystem.resolve)
    # ------------------------------------------------------------------

    def set_gear_loads(self, loads: list[Load]) -> None:
        """
        Replace every load with source=="gear_mesh" by `loads`, preserving all
        user (and any other non-gear-mesh) loads. Idempotent: safe to call
        repeatedly after re-resolve() without accumulating duplicates.
        """
        self._loads = [ld for ld in self._loads if ld.source != "gear_mesh"]
        self._loads.extend(loads)

    # ------------------------------------------------------------------
    # Sorted accessors (by axial position)
    # ------------------------------------------------------------------

    @property
    def bearings(self) -> list["Bearing"]:
        return sorted(self._bearings, key=lambda b: b.position)

    @property
    def gears(self) -> list[GearElement]:
        return sorted(self._gears, key=lambda g: g.position)

    @property
    def loads(self) -> list[Load]:
        return sorted(self._loads, key=lambda ld: ld.position)

    @property
    def support_positions(self) -> list[float]:
        return [b.position for b in self.bearings]

    # ------------------------------------------------------------------
    # Type-filtered accessors (full objects, position-sorted)
    # StaticsSolver calls .component(plane) itself — no plane pre-filtering,
    # since every RadialLoad contributes to both XY and XZ via theta_deg.
    # ------------------------------------------------------------------

    @property
    def radial_loads(self) -> list[RadialLoad]:
        return [ld for ld in self.loads if isinstance(ld, RadialLoad)]

    @property
    def axial_loads(self) -> list[AxialLoad]:
        return [ld for ld in self.loads if isinstance(ld, AxialLoad)]

    @property
    def torque_loads(self) -> list[TorqueLoad]:
        return [ld for ld in self.loads if isinstance(ld, TorqueLoad)]

    @property
    def external_moments(self) -> list[ExternalMoment]:
        return [ld for ld in self.loads if isinstance(ld, ExternalMoment)]
    
    # ------------------------------------------------------------------
    # Axial extent helpers (overlap / shoulder checks)
    # ------------------------------------------------------------------

    def gear_extent(self, ge: GearElement) -> tuple[float, float]:
        """
        Axial [lo, hi] footprint of a gear element.
        Convention: position is the CENTRE of the face width (b).
        b == 0.0 (not yet defined) -> treated as a point.
        """
        b = getattr(ge.gear, "b", 0.0)
        p = ge.position
        return (p - b / 2.0, p + b / 2.0)

    def bearing_extent(self, bearing: "Bearing") -> tuple[float, float]:
        """
        Axial [lo, hi] footprint of a bearing.
        Convention: position is the CENTRE of the bearing width (b).
        b == 0.0 (not yet defined) -> treated as a point.
        """
        width = getattr(bearing, "b", 0.0)
        p = bearing.position
        return (p - width / 2.0, p + width / 2.0)

    def _all_axial_elements(self) -> list[tuple[float, float, str, str]]:
        """
        Every gear + bearing on this shaft as (lo, hi, kind, tag),
        sorted by lo. Shared by overlap and shoulder checks.
        """
        items: list[tuple[float, float, str, str]] = []
        for ge in self._gears:
            lo, hi = self.gear_extent(ge)
            tag = ge.label or getattr(ge.gear, "label", "") or "gear"
            items.append((lo, hi, "gear", tag))
        for b in self._bearings:
            lo, hi = self.bearing_extent(b)
            tag = b.label or b.designation or "bearing"
            items.append((lo, hi, "bearing", tag))
        items.sort(key=lambda t: t[0])
        return items

    def _overlap_errors(self) -> list[str]:
        """
        Check pairwise overlap between all gears and bearings on this
        shaft. Elements are 1D intervals; sorting by `lo` and comparing
        only adjacent pairs is sufficient once sorted.

        Point-vs-point pairs (both b/width == 0.0) fall back to
        MIN_BEARING_SEPARATION_mm as a purely numerical minimum
        separation, not an engineering clearance.
        """
        errors: list[str] = []
        tag = self.name or self.label or "ShaftSystem"

        items = self._all_axial_elements()
        for (lo1, hi1, kind1, tag1), (lo2, hi2, kind2, tag2) in zip(items, items[1:]):
            is_point_pair = (hi1 - lo1) == 0.0 and (hi2 - lo2) == 0.0
            gap = lo2 - hi1  # <= 0 => overlap or touching

            if is_point_pair:
                if abs(lo2 - lo1) < MIN_BEARING_SEPARATION_MM:
                    errors.append(
                        f"{tag}: {kind1} '{tag1}' and {kind2} '{tag2}' coincide "
                        f"within {MIN_BEARING_SEPARATION_MM} mm "
                        f"({lo1:.4f} vs {lo2:.4f} mm) — width not yet defined, "
                        f"treated as point separation only."
                    )
            elif gap < 0.0:
                errors.append(
                    f"{tag}: {kind1} '{tag1}' [{lo1:.3f}, {hi1:.3f}] mm overlaps "
                    f"{kind2} '{tag2}' [{lo2:.3f}, {hi2:.3f}] mm by {-gap:.3f} mm"
                )
        return errors

    def _shoulder_coincidence_errors(
        self, margin: float | None = None
    ) -> list[str]:
        """
        Check shoulder coincidence separately for gears and bearings.

        Gears: any axial coincidence with a shoulder is flagged — not a
        standard mounting pattern for AxisForge's scope.

        Bearings: axial coincidence with a shoulder is the NORMAL axial
        locating scheme (the shoulder is the abutment face for the inner
        ring) and is NOT flagged by itself. It is only an error if the
        shoulder's diameter_large exceeds the bearing's permissible
        abutment diameter (Bearing.da_max) — i.e. the shoulder step would
        foul the inner race fillet / rolling track. If da_max is not
        defined on the bearing, coincidence is reported as unverifiable
        rather than silently accepted.

        margin : float | None
            Axial keep-out margin used only for the GEAR check.
            Defaults to MIN_SHOULDER_CLEARANCE_mm if None.
        """
        errors: list[str] = []
        tag = self.name or self.label or "ShaftSystem"
        keep_out = margin if margin is not None else MIN_SHOULDER_CLEARANCE_MM

        shoulders = self.shaft.shoulders()
        if not shoulders:
            return errors

        # --- gears: axial coincidence alone is an error ---
        for ge in self._gears:
            lo, hi = self.gear_extent(ge)
            elem_tag = ge.label or getattr(ge.gear, "label", "") or "gear"
            for x_shoulder, _ in shoulders:
                if (lo - keep_out) <= x_shoulder <= (hi + keep_out):
                    errors.append(
                        f"{tag}: gear '{elem_tag}' [{lo:.3f}, {hi:.3f}] mm "
                        f"coincides with shoulder at x={x_shoulder:.3f} mm "
                        f"(keep-out margin={keep_out:.3f} mm)"
                    )

        # --- bearings: coincidence is normal; only d1 violation is an error ---
        for b in self._bearings:
            lo, hi = self.bearing_extent(b)
            elem_tag = b.label or b.designation or "bearing"
            for x_shoulder, shoulder in shoulders:
                if not (lo < x_shoulder < hi):
                    continue  # no axial overlap at all — nothing to check
                
                if hi > x_shoulder or lo < x_shoulder:
                    # bearing overlaps shoulder axially — check inner
                    errors.append(
                        f"{tag}: bearing '{elem_tag}' [{lo:.3f}, {hi:.3f}] mm "
                        f"overlaps shoulder at x={x_shoulder:.3f} mm — "
                    )
                else:
                    arrangement = getattr(b, "arrangement", "locating")
                    if arrangement == "locating":
                        d1 = getattr(b, "d1", None)
                        if not (lo == x_shoulder or hi == x_shoulder):
                            errors.append(
                                f"{tag}: bearing '{elem_tag}' [{lo:.3f}, {hi:.3f}] mm "
                                f"as an arrangement='locating' bearing must abut a shoulder at x={x_shoulder:.3f} mm "
                            )
                        elif d1 is not None and shoulder.diameter_large > d1:
                            errors.append(
                                f"{tag}: bearing '{elem_tag}' abuts shoulder at "
                                f"x={x_shoulder:.3f} mm with diameter_large="
                                f"{shoulder.diameter_large:.3f} mm > d1={d1:.3f} mm "
                                f"— shoulder fouls the inner race."
                            )

                        else:
                            errors.append(
                                f"{tag}: bearing '{elem_tag}' abuts shoulder at "
                                f"x={x_shoulder:.3f} mm but d1 (max permissible "
                                f"shoulder diameter, SKF catalog) is not defined on "
                                f"the bearing — cannot verify inner-race clearance." 
                            )
                    elif arrangement == "floating":
                        if not (lo-FLOATING_BEARING_CLEARANCE_MM == x_shoulder == hi+FLOATING_BEARING_CLEARANCE_MM):
                            errors.append(
                                f"{tag}: bearing '{elem_tag}' [{lo:.3f}, {hi:.3f}] mm "
                                f"as an arrangement='floating' bearing must have a margin of '{FLOATING_BEARING_CLEARANCE_MM:.3f}' mm"
                                f"from the shoulder at x={x_shoulder:.3f} mm "
                            )

                    # falta o non-locating arrangement mas preciso de verificar as suas condições

        return errors
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.name or self.label or "ShaftSystem"

        # 1 — delegate to owned components
        errors.extend(f"{tag}.shaft: {e}" for e in self.shaft.validate())
        for b in self._bearings:
            errors.extend(f"{tag}.bearing: {e}" for e in b.validate())
        for g in self._gears:
            errors.extend(f"{tag}: {e}" for e in g.validate())
        for ld in self._loads:
            errors.extend(f"{tag}: {e}" for e in ld.validate())

        # 2 — structural checks
        L = self.shaft.total_length

        if len(self._bearings) < 2:
            errors.append(
                f"{tag}: needs >= 2 bearings for a determinate support, "
                f"got {len(self._bearings)}"
            )

        if self.speed_rpm < 0:
            errors.append(f"{tag}: speed_rpm must be >= 0, got {self.speed_rpm}")

        # positions within [0, L]
        for b in self._bearings:
            if not (0.0 <= b.position <= L):
                errors.append(
                    f"{tag}: bearing position {b.position:.4f} mm outside [0, {L:.4f}]"
                )
        for g in self._gears:
            if not (0.0 <= g.position <= L):
                errors.append(
                    f"{tag}: gear position {g.position:.4f} mm outside [0, {L:.4f}]"
                )
        for ld in self._loads:
            if not (0.0 <= ld.position <= L):
                errors.append(
                    f"{tag}: load position {ld.position:.4f} mm outside [0, {L:.4f}]"
                )

        # no two bearings coincident (div-by-zero guard for the 1D FEM)
        sorted_pos = sorted(b.position for b in self._bearings)
        for p_lo, p_hi in zip(sorted_pos, sorted_pos[1:]):
            if (p_hi - p_lo) < MIN_BEARING_SEPARATION_MM:
                errors.append(
                    f"{tag}: two bearings within {MIN_BEARING_SEPARATION_MM} mm "
                    f"({p_lo:.6f} vs {p_hi:.6f} mm) — cannot form a support span"
                )
        
        # overlap between gears/bearings (any pair, not just bearing-bearing)
        errors.extend(self._overlap_errors())

        # gear/bearing footprint must not coincide with a shaft shoulder
        errors.extend(self._shoulder_coincidence_errors())

        # helical thrust needs an axial reaction path
        has_axial_mesh = any(
            isinstance(ld, AxialLoad) and ld.source == "gear_mesh"
            for ld in self._loads
        )
        if has_axial_mesh:
            has_fixed = any(
                b.arrangement == "fixed" and b.Ka is not None
                for b in self._bearings
            )
            if not has_fixed:
                errors.append(
                    f"{tag}: a gear-mesh axial load is present but no bearing has "
                    f"arrangement=='fixed' with Ka defined — the helical thrust "
                    f"has no axial reaction path"
                )

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError(
                f"ShaftSystem '{self.name}' validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # State freeze (optional; useful before solvers run)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Freeze the current shaft state into a plain dict for later per-shaft
        consumption. Objects are referenced (not deep-copied); the containers
        are copied so subsequent add/set calls don't mutate the snapshot.
        """
        return {
            "name": self.name,
            "label": self.label,
            "speed_rpm": self.speed_rpm,
            "design_life_hours": self.design_life_hours,
            "shaft_position": self.shaft_position,
            "shaft_origin_x": self.shaft_origin_x,
            "total_length": self.shaft.total_length,
            "shaft": self.shaft,
            "bearings": list(self.bearings),
            "gears": list(self.gears),
            "loads": list(self.loads),
            "support_positions": list(self.support_positions),
        }

    to_dict = snapshot  # alias

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        y, z = self.shaft_position
        lines = [
            f"── ShaftSystem '{self.name}' ─────────────────────────────",
            f"  length        : {self.shaft.total_length:.2f} mm",
            f"  speed         : {self.speed_rpm:.2f} rpm",
            f"  design life   : {self.design_life_hours:.0f} h",
            f"  axis offset   : (y={y:.3f}, z={z:.3f}) mm   origin_x={self.shaft_origin_x:.3f} mm",
            f"  bearings      : {len(self._bearings)}  @ {[f'{p:.1f}' for p in self.support_positions]}",
            f"  gears         : {len(self._gears)}  @ {[f'{g.position:.1f}' for g in self.gears]}",
            f"  loads         : {len(self._loads)}  "
            f"(user={sum(1 for l in self._loads if l.source=='user')}, "
            f"gear_mesh={sum(1 for l in self._loads if l.source=='gear_mesh')})",
            "──────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"ShaftSystem(name={self.name!r}, "
                f"L={self.shaft.total_length:.1f} mm, "
                f"bearings={len(self._bearings)}, gears={len(self._gears)}, "
                f"loads={len(self._loads)})")