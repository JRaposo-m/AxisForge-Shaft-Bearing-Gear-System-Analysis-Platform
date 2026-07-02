"""
core/mechanical_elements/Bearings/bearing.py

Rolling bearing: catalog input data for a single bearing position.

References:
  - ISO 281:2007   — dynamic load rating, life calculation, X/Y factors
  - ISO 76:2006    — static load rating
"""

from __future__ import annotations
import numpy as np

from .bearing_types import BearingType


class Bearing:

    def __init__(self,
                 position: float,
                 bearing_type: BearingType = BearingType.DEEP_GROOVE_BALL,
                 designation: str = "",
                 C: float = 0.0,
                 C0: float = 0.0,
                 arrangement: str = "fixed",
                 contact_angle_deg: float = 0.0,
                 X: float = 1.0,
                 Y: float = 0.0,
                 Kr: float = 1.0e5,
                 Ka: float | None = None,
                 label: str = ""):
        """
        Parameters
        ----------
        position          : axial coordinate along the shaft [mm]
        bearing_type       : BearingType enum — catalog family
        designation         : manufacturer designation, e.g. "6208" (informational only)
        C                  : dynamic load rating [N]        (ISO 281)
        C0                 : static load rating [N]         (ISO 76)
        arrangement         : "fixed" (locating, restrains axial) | "floating"
        contact_angle_deg   : nominal contact angle [°]      (0 = radial)
        X, Y                : dynamic equivalent load factors (ISO 281 Table)
                              static input for now — later derived from Fa/C0
        Kr                 : radial stiffness [N/mm]         (FEM 1D input)
        Ka                 : axial stiffness [N/mm]          (required if arrangement="fixed")
        label               : identifier for reporting/traceability
        """
        # --- metadata ---
        self.label       = label
        self.designation = designation

        # --- catalog / rating data ---
        self.bearing_type = bearing_type
        self.C            = C
        self.C0           = C0
        self.contact_angle_deg = contact_angle_deg
        self.contact_angle      = np.radians(contact_angle_deg)

        # --- ISO 281 equivalent load factors (static input, fase 1) ---
        self.X = X
        self.Y = Y

        # --- mounting / FEM 1D ---
        self.position    = position
        self.arrangement = arrangement
        self.Kr          = Kr
        self.Ka          = Ka

    # ------------------------------------------------------------------
    # Derived / convenience
    # ------------------------------------------------------------------

    def is_locating(self) -> bool:
        """True if this bearing restrains axial displacement (fixed side)."""
        return self.arrangement == "fixed"

    def equivalent_dynamic_load(self, Fr: float, Fa: float) -> float:
        """
        P = X·Fr + Y·Fa   (ISO 281)

        X, Y are taken as fixed inputs for this phase — no lookup on Fa/C0
        is performed yet.
        """
        return self.X * Fr + self.Y * Fa

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
        if self.arrangement not in ("fixed", "floating"):
            errors.append(
                f"{tag}: arrangement must be 'fixed' or 'floating', got '{self.arrangement}'"
            )
        if not (0.0 <= self.contact_angle_deg < 90.0):
            errors.append(
                f"{tag}: contact_angle_deg must be in [0, 90), got {self.contact_angle_deg}"
            )
        if not (0.0 < self.X <= 1.0):
            errors.append(f"{tag}: X (radial factor) must be in (0, 1], got {self.X}")
        if self.Y < 0:
            errors.append(f"{tag}: Y (axial factor) must be >= 0, got {self.Y}")
        if self.Kr <= 0:
            errors.append(f"{tag}: Kr must be > 0, got {self.Kr}")
        if self.arrangement == "fixed" and self.Ka is None:
            errors.append(f"{tag}: arrangement='fixed' requires Ka to be defined")
        if self.Ka is not None and self.Ka <= 0:
            errors.append(f"{tag}: Ka must be > 0 when defined, got {self.Ka}")

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
            f"  C / C0      : {self.C:.0f} N / {self.C0:.0f} N",
            f"  X / Y       : {self.X:.3f} / {self.Y:.3f}",
            f"  Kr / Ka     : {self.Kr:.2e} N/mm / "
            f"{'—' if self.Ka is None else f'{self.Ka:.2e}'} N/mm",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Bearing(type={self.bearing_type.name}, position={self.position:.2f} mm, "
            f"arrangement='{self.arrangement}', Kr={self.Kr:.2e} N/mm)"
        )