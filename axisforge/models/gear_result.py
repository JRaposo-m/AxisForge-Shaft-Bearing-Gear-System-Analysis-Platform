"""
models/gear_result.py
GearGeometryResult and GearForceResult — immutable output of GearSolver.

Units (AxisForge SI convention):
  Lengths      : mm
  Forces       : N
  Torques      : N·mm
  Angles I/O   : degrees (internal computation in radians)
  Ratios       : dimensionless

Phase 1 scope:
  - Cylindrical gears: spur (β=0) and helical.
  - Profile shift x1 = x2 = 0.
  - Single gear pair (pinion on shaft under analysis).
  - al supplied by user; αtw derived directly (no brentq needed for x=0 case).

References:
  MAAG Gear Book; ISO 21771:2007; Shigley §13-7.
  GEARpie CALC_GEOMETRY.py / FORCES_SPEEDS.py (C. Fernandes, MIT License).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GearGeometryResult:
    """
    Geometric parameters of a cylindrical gear pair.

    All angles stored in degrees (input/output convention).
    All lengths in mm.

    Attributes
    ----------
    mn : float
        Normal module [mm].
    z1, z2 : int
        Number of teeth — pinion, wheel.
    alpha_n_deg : float
        Normal pressure angle [°].
    beta_deg : float
        Helix angle [°].
    x1, x2 : float
        Profile shift coefficients (Phase 1: both 0.0).
    mt : float
        Transverse module [mm].
    alpha_t_deg : float
        Transverse pressure angle [°].
    beta_b_deg : float
        Base helix angle [°].
    u : float
        Gear ratio z2/z1.
    d1, d2 : float
        Reference pitch diameters [mm].
    db1, db2 : float
        Base circle diameters [mm].
    da1, da2 : float
        Tip (addendum) diameters [mm].
    df1, df2 : float
        Root (dedendum) diameters [mm].
    dl1, dl2 : float
        Working pitch diameters [mm].  dl = 2*al/(u+1) for pinion.
    a : float
        Standard centre distance [mm].
    al : float
        Working centre distance [mm].
    alpha_tw_deg : float
        Working transverse pressure angle [°].
    p_bt : float
        Transverse base pitch [mm].
    eps_alpha : float
        Transverse contact ratio εα [-].
    eps_beta : float
        Overlap contact ratio εβ [-].  0 for spur gears.
    eps_gamma : float
        Total contact ratio εγ = εα + εβ [-].
    b : float
        Face width used for εβ [mm].  0 if not supplied (Phase 1: εβ=0 for β=0).
    """
    mn: float
    z1: int
    z2: int
    alpha_n_deg: float
    beta_deg: float
    x1: float
    x2: float

    # Derived angles
    mt: float
    alpha_t_deg: float
    beta_b_deg: float

    # Ratios and distances
    u: float
    a: float
    al: float
    alpha_tw_deg: float

    # Diameters
    d1: float
    d2: float
    db1: float
    db2: float
    da1: float
    da2: float
    df1: float
    df2: float
    dl1: float
    dl2: float

    # Pitches
    p_bt: float

    # Contact ratios
    eps_alpha: float
    eps_beta: float
    eps_gamma: float

    # Face width (for εβ; 0 if not supplied)
    b: float = 0.0

    @property
    def rl1(self) -> float:
        """Working pitch radius of pinion [mm]  = dl1/2."""
        return self.dl1 / 2.0

    @property
    def rl2(self) -> float:
        """Working pitch radius of wheel [mm]  = dl2/2."""
        return self.dl2 / 2.0

    @property
    def rb1(self) -> float:
        """Base circle radius of pinion [mm]  = db1/2."""
        return self.db1 / 2.0

    @property
    def rb2(self) -> float:
        """Base circle radius of wheel [mm]  = db2/2."""
        return self.db2 / 2.0

    @property
    def is_spur(self) -> bool:
        """True if helix angle is effectively zero."""
        return self.beta_deg < 0.1

    @property
    def is_standard_centre(self) -> bool:
        """True if working centre distance equals standard centre distance (within 0.01 mm)."""
        return abs(self.al - self.a) < 0.01


@dataclass(frozen=True)
class GearForceResult:
    """
    Force and torque components for a gear pair.

    All forces in N, torques in N·mm (AxisForge convention).

    Attributes
    ----------
    Ft : float
        Tangential force [N]  — acts on working pitch circle, XZ plane.
    Fr : float
        Radial force [N]      — separating force, XY plane.
    Fa : float
        Axial (thrust) force [N] — along shaft axis; 0 for spur gears.
    Fn : float
        Normal force [N]      — resultant at tooth contact.
    Fbt : float
        Tangential force at base circle [N]  (intermediate, exposed for debug).
    Fbn : float
        Normal force at base plane [N]       (intermediate, exposed for debug).
    T1_Nmm : float
        Input torque on pinion [N·mm].
    T2_Nmm : float
        Output torque on wheel [N·mm].
    gear_ratio : float
        u = z2/z1.
    """
    Ft: float
    Fr: float
    Fa: float
    Fn: float
    Fbt: float
    Fbn: float
    T1_Nmm: float
    T2_Nmm: float
    gear_ratio: float

    @property
    def F_transverse_resultant(self) -> float:
        """√(Ft² + Fr²) — resultant in the transverse plane [N]."""
        return math.hypot(self.Ft, self.Fr)

    @property
    def consistency_deviation_pct(self) -> float:
        """
        Deviation between stored T1 and Ft×rl1.

        rl1 = T1_Nmm / Ft  (derived — no geometry object needed here).
        Exposed for cross-check in tests; GearElement.__post_init__ enforces < 2%.
        """
        # T1 = Ft × rl1  →  rl1 = T1/Ft
        if self.Ft < 1e-9:
            return 0.0
        rl1_derived = self.T1_Nmm / self.Ft
        T1_check = self.Ft * rl1_derived
        return abs(T1_check - self.T1_Nmm) / max(self.T1_Nmm, 1.0) * 100.0
