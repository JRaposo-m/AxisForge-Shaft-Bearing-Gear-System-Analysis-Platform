"""
models/bearing_result.py
BearingLifeResult — immutable output of BearingLifeSolver.

Phase 1 scope:
  - ISO 281 basic L10 life only (no aISO modification factor).
  - Equivalent dynamic load: P = X*Fr + Y*Fa (Phase 1: X=1, Y=0 → P = Fr).
  - Equivalent static load:  P0 = X0*Fr + Y0*Fa (Phase 1: X0=0.6, Y0=0.5 — DGBB values).
  - Static safety factor:    S0 = C0 / P0.

References:
  ISO 281:2007 §6 (basic life), §7 (equivalent load)
  SKF General Catalogue pub. 10000 EN, §2.1
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BearingLifeResult:
    """
    Life and safety assessment for a single rolling bearing.

    Attributes
    ----------
    bearing_label : str
        Bearing identifier (from Bearing.label).
    bearing_position : float
        Axial position on shaft [mm].
    Fr : float
        Resultant radial force on bearing [N].
    Fa : float
        Axial (thrust) force on bearing [N].
    P : float
        Equivalent dynamic load [N].  Phase 1: P = Fr (X=1, Y=0).
    P0 : float
        Equivalent static load [N].   Phase 1: P0 = 0.6*Fr + 0.5*Fa.
    L10 : float
        Basic rating life [10⁶ rev].  L10 = (C/P)^p.
    L10h : float
        Basic rating life [h].        L10h = L10×10⁶ / (60×n).
    S0 : float
        Static safety factor [-].     S0 = C0 / P0.
    C_over_P : float
        Dynamic load ratio C/P [-].
    life_target_hours : float
        Design life target [h] passed to solver.
    meets_life_target : bool
        True if L10h >= life_target_hours.
    meets_static_safety : bool
        True if S0 >= STATIC_SAFETY_FACTOR_MIN (1.0).
    X : float
        Radial load factor used for P.
    Y : float
        Axial load factor used for P.
    X0 : float
        Radial load factor used for P0.
    Y0 : float
        Axial load factor used for P0.
    life_exponent : float
        p — 3.0 for ball, 10/3 for roller.
    speed_rpm : float
        Operating speed used for L10h [rpm].
    """
    bearing_label: str
    bearing_position: float     # [mm]
    Fr: float                   # [N]
    Fa: float                   # [N]
    P: float                    # [N]
    P0: float                   # [N]
    L10: float                  # [10⁶ rev]
    L10h: float                 # [h]
    S0: float                   # [-]
    C_over_P: float             # [-]
    life_target_hours: float    # [h]
    meets_life_target: bool
    meets_static_safety: bool
    X: float
    Y: float
    X0: float
    Y0: float
    life_exponent: float
    speed_rpm: float            # [rpm]

    @property
    def is_safe(self) -> bool:
        """True if both life target and static safety are met."""
        return self.meets_life_target and self.meets_static_safety

    @property
    def life_ratio(self) -> float:
        """L10h / life_target_hours — margin above (>1) or below (<1) target."""
        if self.life_target_hours <= 0.0:
            return float("inf")
        return self.L10h / self.life_target_hours

    def __repr__(self) -> str:
        return (
            f"BearingLifeResult(label={self.bearing_label!r}, "
            f"Fr={self.Fr:.0f}N, L10h={self.L10h:.0f}h, "
            f"S0={self.S0:.2f}, safe={self.is_safe})"
        )
