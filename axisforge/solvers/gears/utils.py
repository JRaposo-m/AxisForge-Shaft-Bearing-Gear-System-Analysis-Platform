"""
solvers/gears/utils.py
Pure mathematical helpers for cylindrical gear geometry.

All functions are stateless. No solver classes here — only building blocks
used by geometry.py, and reused by strength.py and profile_shift.py (Phase 2+).

Unit convention (AxisForge SI):
  Angles  : radians internally; degrees at public API boundary (in geometry.py)
  Lengths : mm
  Forces  : N

References:
  MAAG Gear Book, 2nd ed.
  ISO 21771:2007 — Gears — Cylindrical involute gears and gear pairs.
  Shigley 10th ed. §13-7, §13-10.
  GEARpie CALC_GEOMETRY.py (C. Fernandes, MIT License).
"""
from __future__ import annotations

import math
from typing import Optional

from scipy.optimize import brentq

# Standard rack proportions (ISO 53 / DIN 867)
HAP: float = 1.0    # addendum coefficient
HFP: float = 1.25   # dedendum coefficient
# k = 0: no tip shortening (Phase 1; Phase 2 computes k from centre modification)


# ---------------------------------------------------------------------------
# Involute function
# ---------------------------------------------------------------------------

def involute(alpha: float) -> float:
    """
    Involute function: inv(α) = tan(α) − α  [rad].

    Parameters
    ----------
    alpha : float
        Angle in radians.
    """
    return math.tan(alpha) - alpha


def solve_alpha_tw(
    alpha_n: float,
    alpha_t: float,
    x1: float,
    x2: float,
    z1: int,
    z2: int,
    db1: float,
    db2: float,
) -> tuple[float, float]:
    """
    Solve for working pressure angle αtw and centre distance al
    from the involute equation (ISO 21771 §7).

      inv(αtw) = inv(αt) + 2·tan(αn)·(x1+x2)/(z1+z2)
      al = (db1 + db2) / (2·cos(αtw))

    Uses scipy.optimize.brentq to invert the involute function.

    Parameters
    ----------
    alpha_n : float
        Normal pressure angle [rad].
    alpha_t : float
        Transverse pressure angle [rad].
    x1, x2 : float
        Profile shift coefficients [-].
    z1, z2 : int
        Number of teeth — pinion, wheel.
    db1, db2 : float
        Base circle diameters [mm].

    Returns
    -------
    (al [mm], alpha_tw [rad])

    Raises
    ------
    ValueError
        If brentq cannot solve the involute equation (invalid profile shifts).
    """

    inv_target = (
        math.tan(alpha_t)
        - alpha_t + 2.0 * math.tan(alpha_n) * (x1 + x2) / (z1 + z2)
    )

    def _residual(atw: float) -> float:
        return involute(atw) - inv_target

    lo = alpha_t / 2.0 if alpha_t > 1e-6 else 1e-6
    hi = math.radians(89.0)

    try:
        alpha_tw = brentq(_residual, lo, hi, xtol=1e-12, rtol=1e-12)
    except ValueError as exc:
        raise ValueError(
            f"Could not solve involute equation for x1={x1}, x2={x2}. "
            f"inv_target={inv_target:.8f}. Check profile shift values."
        ) from exc

    al = (db1 + db2) / (2.0 * math.cos(alpha_tw))
    return al, alpha_tw


# ---------------------------------------------------------------------------
# Contact ratios
# ---------------------------------------------------------------------------

def contact_ratio_alpha(
    z1: int,
    z2: int,
    da1: float,
    da2: float,
    db1: float,
    db2: float,
    alpha_tw: float,
    p_bt: float,
) -> float:
    """
    Transverse contact ratio εα (ISO 21771).

      εa1 = z1·(tan(αa1) − tan(αtw)) / (2π)
      εa2 = z2·(tan(αa2) − tan(αtw)) / (2π)
      αai = arccos(dbi / dai)

    Parameters
    ----------
    z1, z2 : int
        Number of teeth.
    da1, da2 : float
        Tip diameters [mm].
    db1, db2 : float
        Base diameters [mm].
    alpha_tw : float
        Working transverse pressure angle [rad].
    p_bt : float
        Transverse base pitch [mm].  (Unused in this formulation; kept for
        API symmetry with contact_ratio_beta and future AE-length calculation.)
    """
    cos_aa1 = min(max(db1 / da1, -1.0), 1.0)
    cos_aa2 = min(max(db2 / da2, -1.0), 1.0)
    alpha_a1 = math.acos(cos_aa1)
    alpha_a2 = math.acos(cos_aa2)

    tan_atw = math.tan(alpha_tw)
    eps_a1 = z1 * (math.tan(alpha_a1) - tan_atw) / (2.0 * math.pi)
    eps_a2 = z2 * (math.tan(alpha_a2) - tan_atw) / (2.0 * math.pi)
    return eps_a1 + eps_a2


def contact_ratio_beta(b: float, beta_b: float, p_bt: float) -> float:
    """
    Overlap contact ratio εβ = b·tan(βb)/p_bt (ISO 21771).

    Returns 0.0 if b ≤ 0 or p_bt ≤ 0 (spur gear or no face width supplied).

    Parameters
    ----------
    b : float
        Face width [mm].
    beta_b : float
        Base helix angle [rad].
    p_bt : float
        Transverse base pitch [mm].
    """
    if b <= 0.0 or p_bt <= 0.0:
        return 0.0
    return b * math.tan(beta_b) / p_bt


# ---------------------------------------------------------------------------
# Undercut limit
# ---------------------------------------------------------------------------

def undercut_z_min(alpha_n_deg: float, beta_deg: float) -> int:
    """
    Minimum number of teeth to avoid undercut for x=0.

      z_min = floor(2 / sin²(αt))

    z=17 is safe for α=20°, β=0° (Shigley §13-10, Tab. 13-11).
    Profile shift x > 0 mitigates undercut below this limit.

    Parameters
    ----------
    alpha_n_deg : float
        Normal pressure angle [°].
    beta_deg : float
        Helix angle [°].

    Returns
    -------
    int
        Minimum tooth count without undercut.
    """
    alpha_n = math.radians(alpha_n_deg)
    beta    = math.radians(beta_deg)
    alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
    sin_at  = math.sin(alpha_t)
    if sin_at < 1e-9:
        return 999
    return int(2.0 / sin_at ** 2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_geometry_inputs(
    mn: float,
    z1: int,
    z2: int,
    alpha_n_deg: float,
    beta_deg: float,
    al: Optional[float],
    x1: float,
    x2: float,
) -> None:
    """
    Validate raw inputs for GearSolver.compute_geometry().

    Raises
    ------
    ValueError
        On any invalid parameter.
    """
    if mn <= 0.0:
        raise ValueError(f"mn must be > 0, got {mn}")
    if z1 <= 0:
        raise ValueError(f"z1 must be ≥ 1, got {z1}")
    if z2 <= 0:
        raise ValueError(f"z2 must be ≥ 1, got {z2}")
    if not (0.0 < alpha_n_deg < 90.0):
        raise ValueError(f"alpha_n_deg must be in (0, 90), got {alpha_n_deg}")
    if not (0.0 <= beta_deg < 90.0):
        raise ValueError(f"beta_deg must be in [0, 90), got {beta_deg}")
    if al is not None and al <= 0.0:
        raise ValueError(f"al must be > 0, got {al}")
    if x1 < -2.0 or x1 > 2.0:
        raise ValueError(f"x1={x1} outside plausible range [-2, 2]")
    if x2 < -2.0 or x2 > 2.0:
        raise ValueError(f"x2={x2} outside plausible range [-2, 2]")
