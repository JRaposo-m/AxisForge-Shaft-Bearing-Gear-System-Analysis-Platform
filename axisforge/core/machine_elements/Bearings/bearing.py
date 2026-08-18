"""
core/machine_elements/Bearings/bearing.py

Rolling bearing base class — catalog data, ISO 281 equivalent load,
mounting arrangement.

Internal geometry slots are NOT declared here. Each subclass declares its
own geometry attributes in __init__ and overrides has_internal_geometry().
Contact angle is part of that internal geometry too now — alpha_0, set by
each subclass's setup_internal_geometry() (mirrored from its Geometry
object), is the single source of truth for contact angle. There is no
separate contact-angle input at this base level anymore.

References:
  - ISO 281:2007  — dynamic load rating, life calculation, X/Y factors
  - ISO 76:2006   — static load rating
"""

from __future__ import annotations
import numpy as np
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType


class Bearing:

    def __init__(self,
                 d: float,
                 D: float,
                 bearing_type: BearingType = BearingType.DEEP_GROOVE_BALL,
                 designation: str = "",
                 b: float = 0.0,
                 C: float = 0.0,
                 C0: float = 0.0,
                 X: float = 1.0,
                 Y: float = 0.0,
                 arrangement: str = "locating",
                 label: str = "",
                 position: float = 0.0):
        """
        Parameters
        ----------
        d                 : bore diameter [mm]
        D                 : outer diameter [mm]
        bearing_type      : BearingType enum
        designation       : manufacturer designation, e.g. "6208"
        b                 : bearing width [mm]
        C                 : dynamic load rating [N]   (ISO 281)
        C0                : static load rating [N]    (ISO 76)
        X, Y              : dynamic equivalent load factors (ISO 281)
        arrangement       : "locating" | "floating" | "non-locating"
        label             : identifier for reporting/traceability
        position          : axial coordinate along the shaft [mm]

        Internal geometry (Dw, Dpw, Z, ri, re, alpha_0, ...) is NOT
        declared here. Each subclass declares its own geometry slots in
        __init__ and populates alpha_0 from setup_internal_geometry().
        """
        # --- metadata ---
        self.label        = label
        self.designation  = designation
        self.bearing_type = bearing_type

        # --- catalog / rating data ---
        self.b   = b
        self.d   = d
        self.D   = D
        self.C   = C
        self.C0  = C0
        self.X   = X
        self.Y   = Y
        self.dm  = 0.5 * (d + D)

        # --- mounting ---
        self.position    = position
        self.arrangement = arrangement

    # ------------------------------------------------------------------
    # Derived / convenience
    # ------------------------------------------------------------------

    def is_locating(self) -> bool:
        return self.arrangement == "locating"

    def equivalent_dynamic_load(self, Fr: float, Fa: float) -> float:
        """P = X·Fr + Y·Fa   (ISO 281)"""
        return self.X * Fr + self.Y * Fa

    def has_internal_geometry(self) -> bool:
        """
        True if setup_internal_geometry() has been called.
        Base returns False — subclass overrides with its own sentinel check.
        """
        return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.designation or self.__class__.__name__

        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.C < 0:
            errors.append(f"{tag}: C must be >= 0, got {self.C}")
        if self.C0 < 0:
            errors.append(f"{tag}: C0 must be >= 0, got {self.C0}")
        if self.arrangement not in ("locating", "floating", "non-locating"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', or "
                f"'non-locating', got '{self.arrangement}'"
            )
        if not (0.0 < self.X <= 1.0):
            errors.append(f"{tag}: X must be in (0, 1], got {self.X}")
        if self.Y < 0:
            errors.append(f"{tag}: Y must be >= 0, got {self.Y}")

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or self.designation or "Bearing"
        lines = [
            f"── {tag} ─────────────────────────────────",
            f"  type        : {self.bearing_type.name}",
            f"  position    : {self.position:.2f} mm",
            f"  arrangement : {self.arrangement}",
            f"  d / D       : {self.d:.1f} mm / {self.D:.1f} mm",
            f"  b           : {self.b:.1f} mm",
            f"  C / C0      : {self.C:.0f} N / {self.C0:.0f} N",
            f"  X / Y       : {self.X:.3f} / {self.Y:.3f}",
            f"  geometry    : {'set' if self.has_internal_geometry() else 'not set'}",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.bearing_type.name}, "
            f"designation='{self.designation}', "
            f"position={self.position:.2f} mm, "
            f"arrangement='{self.arrangement}')"
        )