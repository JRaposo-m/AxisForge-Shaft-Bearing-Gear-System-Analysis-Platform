"""axisforge/solvers/machine_elements/bearings/load_distribution/single_row/iso_16281/ball_bearing/postprocessing.py

Post-processing for POINT-CONTACT (ball) bearings -- everything that
CONSUMES an already-computed BallBearingResult to derive a further
quantity, rather than producing one itself. None of this runs
scipy.optimize -- the solve is already done. Imports ball_bearing.results
only; never imports solver.py (one-directional: postprocessing depends on
the solve output shape, not the other way around).

Single-row only (BallBearingResult.rows has length 1, see
contracts/bearing/ball_bearing_results.py) -- Q_j/phi_j_global/
contact_distribution/DynamicEquivalentRollingElementLoad.from_bearing_result()
still return a length-1 list each, so callers keep indexing [0] the same
way regardless of how this scoping evolves later.

BasicReferenceRatingLife/combine_row_L10r/DynamicEquivalentReferenceLoad
(eq.29-31) are ball-specific -- Sec 4.3.3/4.3.4. Roller's own rating life
is a different, lamina-based formula (Sec 5.3.1.2) in
roller_bearing_postprocessing.py; it does not reuse this code.

Row combination (for a bearing composed of >=2 of these results elsewhere,
e.g. a thrust multi-row bearing) uses Zaretsky, E.V., "Rolling Bearing Life
Prediction, Theory, and Application", NASA/TP-2013-215305/REV1, 2016,
eq.(49a)/(49b): 1/L^e = 1/L1^e + ... + 1/Li^e (e=10/9 ball). Pref
(eq.30-31) is back-derived from L10r, not an independent equivalent-load
formula.

References
----------
ISO/TS 16281:2008 Sec 4.2 eq.(12)-(15), Sec 4.3.2 eq.(25)-(28),
Sec 4.3.3 eq.(29), Sec 4.3.4 eq.(30)-(31)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.results.bearings.load_distribution.single_row.ball_bearing_results import (
    BallLoadDistributionResult,
    BallBearingResult,
)


# ---------------------------------------------------------------------------
# Per-element contact force / global angular position -- length-1 list.
# ---------------------------------------------------------------------------

def Q_j(bearing: Bearing, result: BallBearingResult) -> list[np.ndarray]:
    """Per-element contact force [N], Q_j = cp * delta_j^1.5."""
    row = result.rows[0]
    return [bearing.cp * np.maximum(row.delta_j, 0.0) ** 1.5]


def phi_j_global(bearing: Bearing, result: BallBearingResult) -> list[np.ndarray]:
    """Ball angular positions in the global frame [rad], wrapped to [0, 2*pi)."""
    row = result.rows[0]
    return [(bearing.phi_j + row.phi_Fr) % (2.0 * np.pi)]


def contact_distribution(bearing: Bearing, result: BallBearingResult,
                         frame: str = "global") -> list[np.ndarray]:
    """(angle, contact force) pairs -- ndarray(Z, 2): [phi, Q_j] columns.

    frame="global" (default): phi = phi_j_global(...). frame="local": phi
    as stored (0 aligned with the resultant radial force plane).
    """
    if frame == "global":
        phi = phi_j_global(bearing, result)[0]
    elif frame == "local":
        phi = bearing.phi_j
    else:
        raise ValueError(f"contact_distribution: frame must be 'global' or 'local', got {frame!r}")

    Q = Q_j(bearing, result)[0]
    return [np.column_stack((phi, Q))]


# ---------------------------------------------------------------------------
# BallBearingStiffness -- a property of the whole bearing.
# ---------------------------------------------------------------------------

class BallBearingStiffness:
    """Secant stiffness of a point-contact bearing, XZ / XY / axial axes.

    Built from a converged BallLoadDistributionResult by projecting delta_r
    back onto the global axes:

        delta_r_xz = delta_r * cos(phi_Fr)
        delta_r_xy = delta_r * sin(phi_Fr)
        Kr_xz = Fr_xz / delta_r_xz
        Kr_xy = Fr_xy / delta_r_xy
        Ka    = Fa    / delta_a

    Any stiffness component is set to float('inf') when the corresponding
    displacement is negligible (rigid in that direction).

    Ka_regime: "no_load" (Fa=0) / "engaged" (delta_a>=0) /
    "closing_clearance" (delta_a<0).

    Fr_xz, Fr_xy, Fa, delta_a are not stored here -- they already live on
    BearingNodeData / the source result; only what from_result() computes
    is kept.
    """

    __slots__ = (
        "label",
        "delta_r_xz", "Kr_xz",
        "delta_r_xy", "Kr_xy",
        "Ka", "Ka_regime",
    )

    def __init__(self, label,
                 delta_r_xz, Kr_xz,
                 delta_r_xy, Kr_xy,
                 Ka=None, Ka_regime=None):
        self.label      = label
        self.delta_r_xz = delta_r_xz
        self.Kr_xz      = Kr_xz
        self.delta_r_xy = delta_r_xy
        self.Kr_xy      = Kr_xy
        self.Ka         = Ka
        self.Ka_regime  = Ka_regime

    @classmethod
    def from_result(cls,
                    label: str,
                    result: BallLoadDistributionResult,
                    Fr_xz: float,
                    Fr_xy: float,
                    Fa: float,
                    eps: float = 1e-9) -> "BallBearingStiffness":
        delta_r_xz = result.delta_r * np.cos(result.phi_Fr)
        delta_r_xy = result.delta_r * np.sin(result.phi_Fr)
        delta_a    = result.delta_a

        Kr_xz = (Fr_xz / delta_r_xz) if abs(delta_r_xz) > eps else float("inf")
        Kr_xy = (Fr_xy / delta_r_xy) if abs(delta_r_xy) > eps else float("inf")

        if Fa == 0.0:
            Ka, regime = float("inf"), "no_load"
        elif abs(delta_a) > eps:
            Ka     = Fa / delta_a
            regime = "engaged" if delta_a >= 0.0 else "closing_clearance"
        else:
            Ka, regime = float("inf"), "engaged"

        return cls(label=label,
                   delta_r_xz=delta_r_xz, Kr_xz=Kr_xz,
                   delta_r_xy=delta_r_xy, Kr_xy=Kr_xy,
                   Ka=Ka, Ka_regime=regime)


def bearing_stiffness(bearing: Bearing, result: BallBearingResult,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      eps: float = 1e-9) -> BallBearingStiffness:
    """Secant stiffness (Kr_xz, Kr_xy, Ka) of the whole bearing.

    Fr_xz/Fr_xy/Fa are the TOTAL external loads on the bearing.
    """
    return BallBearingStiffness.from_result(
        label=bearing.label, result=result.rows[0],
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa, eps=eps,
    )


# ---------------------------------------------------------------------------
# DynamicEquivalentRollingElementLoad -- ISO/TS 16281 Sec 4.3.2, eq.(25)-(28)
# ---------------------------------------------------------------------------

_P_ROTATING   = 3.0        # eq.(25)/(28) -- ring rotating relative to the load
_P_STATIONARY = 10.0 / 3.0  # eq.(26)/(27) -- ring stationary relative to the load


@dataclass(frozen=True)
class DynamicEquivalentRollingElementLoad:
    """Dynamic equivalent load per rolling element, inner/outer raceway.

    Rotating relative to the load -- eq.(25) inner, eq.(28) outer:
        Q_e = ( (1/Z) * sum_j Q_j^3 )^(1/3)
    Stationary relative to the load -- eq.(26) inner, eq.(27) outer:
        Q_e = ( (1/Z) * sum_j Q_j^(10/3) )^(3/10)

    inner_rotating / outer_rotating are independent flags (not mutually
    exclusive).

    Attributes
    ----------
    label            str
    Q_ei             float [N]   dynamic equivalent load, inner raceway
    Q_ee             float [N]   dynamic equivalent load, outer raceway
    inner_rotating   bool
    outer_rotating   bool
    Q_j              ndarray(Z,) [N]   per-element contact forces used
    """
    label          : str
    Q_ei           : float
    Q_ee           : float
    inner_rotating : bool
    outer_rotating : bool
    Q_j            : np.ndarray

    @classmethod
    def from_distribution(cls, bearing: Bearing, result: BallLoadDistributionResult,
                          inner_rotating: bool = True,
                          outer_rotating: bool = False,
                          label: str = "") -> "DynamicEquivalentRollingElementLoad":
        """Single-row primitive -- does all the eq.(25)-(28) math."""
        Q = bearing.cp * np.maximum(result.delta_j, 0.0) ** 1.5
        _lbl = label or bearing.label

        Q_ei = _equivalent(Q, inner_rotating)
        Q_ee = _equivalent(Q, outer_rotating)

        return cls(label=_lbl, Q_ei=Q_ei, Q_ee=Q_ee,
                   inner_rotating=inner_rotating, outer_rotating=outer_rotating,
                   Q_j=Q)

    @classmethod
    def from_bearing_result(cls,
                            bearing: Bearing,
                            result: BallBearingResult,
                            inner_rotating: bool = True,
                            outer_rotating: bool = False,
                            label: str = "") -> list["DynamicEquivalentRollingElementLoad"]:
        """Always returns a length-1 list -- calls from_distribution() once."""
        lbl = label or getattr(bearing, "label", "")
        return [cls.from_distribution(
            bearing, result.rows[0],
            inner_rotating=inner_rotating, outer_rotating=outer_rotating,
            label=lbl,
        )]


def _equivalent(Q: np.ndarray, rotating: bool) -> float:
    """eq.(25)/(28) if rotating relative to the load, else eq.(26)/(27)."""
    p = _P_ROTATING if rotating else _P_STATIONARY
    return float(np.mean(Q ** p) ** (1.0 / p))


# ---------------------------------------------------------------------------
# BasicReferenceRatingLife / DynamicEquivalentReferenceLoad -- eq.(29)-(31)
# ---------------------------------------------------------------------------

_E_BALL = 10.0 / 9.0   # Weibull slope, point contact -- Zaretsky eq.(49a)/(49b)


@dataclass(frozen=True)
class BasicReferenceRatingLife:
    """L10r for one row/raceway pair -- eq.(29)."""
    label : str
    L10r  : float
    Q_ci  : float
    Q_ei  : float
    Q_ce  : float
    Q_ee  : float

    @classmethod
    def from_loads(cls, label: str, Q_ci: float, Q_ei: float,
                   Q_ce: float, Q_ee: float) -> "BasicReferenceRatingLife":
        if Q_ci <= 0.0 or Q_ei <= 0.0 or Q_ce <= 0.0 or Q_ee <= 0.0:
            raise ValueError(
                f"BasicReferenceRatingLife.from_loads({label!r}): all loads must be positive."
            )
        L10r = ((Q_ci / Q_ei) ** (-10.0 / 3.0)
                + (Q_ce / Q_ee) ** (-10.0 / 3.0)) ** (-9.0 / 10.0)
        return cls(label=label, L10r=L10r, Q_ci=Q_ci, Q_ei=Q_ei, Q_ce=Q_ce, Q_ee=Q_ee)


def combine_row_L10r(L10r_rows: list[float], e: float = _E_BALL) -> float:
    """Bearing-level L10r from n rows -- Zaretsky eq.(49a): L^-e = sum(Li^-e)."""
    if len(L10r_rows) < 2:
        raise ValueError(f"combine_row_L10r needs >= 2 rows; got {len(L10r_rows)}.")
    if any(L <= 0.0 for L in L10r_rows):
        raise ValueError(f"All L10r_rows must be positive; got {list(L10r_rows)}.")
    return sum(L ** (-e) for L in L10r_rows) ** (-1.0 / e)


def basic_reference_rating_life(
    label: str,
    Q_ci_rows: list[float], Q_ei_rows: list[float],
    Q_ce_rows: list[float], Q_ee_rows: list[float],
    e: float = _E_BALL,
) -> tuple[list[BasicReferenceRatingLife], float]:
    """Per-row L10r (eq.29) + combined bearing L10r (combine_row_L10r() for n>=2)."""
    n = len(Q_ci_rows)
    if not (len(Q_ei_rows) == len(Q_ce_rows) == len(Q_ee_rows) == n):
        raise ValueError("basic_reference_rating_life: row lists must all be the same length.")
    if n == 0:
        raise ValueError("basic_reference_rating_life: got 0 rows.")

    per_row = [
        BasicReferenceRatingLife.from_loads(
            label=f"{label}-row{j}" if n > 1 else label,
            Q_ci=Q_ci_rows[j], Q_ei=Q_ei_rows[j],
            Q_ce=Q_ce_rows[j], Q_ee=Q_ee_rows[j],
        )
        for j in range(n)
    ]
    L10r_bearing = (per_row[0].L10r if n == 1
                    else combine_row_L10r([r.L10r for r in per_row], e=e))
    return per_row, L10r_bearing


@dataclass(frozen=True)
class DynamicEquivalentReferenceLoad:
    """Pref -- eq.(30)-(31): Pref_r = Cr/L10r^(1/3), Pref_a = Ca/L10r^(1/3)."""
    label  : str
    Pref_r : float | None
    Pref_a : float | None

    @classmethod
    def from_L10r(cls, label: str, L10r_bearing: float,
                  Cr: float | None = None, Ca: float | None = None
                  ) -> "DynamicEquivalentReferenceLoad":
        if Cr is None and Ca is None:
            raise ValueError(f"DynamicEquivalentReferenceLoad.from_L10r({label!r}): need Cr and/or Ca.")
        if L10r_bearing <= 0.0:
            raise ValueError(f"L10r_bearing must be positive; got {L10r_bearing}.")
        denom = L10r_bearing ** (1.0 / 3.0)
        return cls(label=label,
                   Pref_r=(Cr / denom) if Cr is not None else None,
                   Pref_a=(Ca / denom) if Ca is not None else None)