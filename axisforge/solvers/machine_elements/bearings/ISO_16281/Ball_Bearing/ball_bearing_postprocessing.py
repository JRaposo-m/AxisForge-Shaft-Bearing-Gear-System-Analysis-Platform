"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_postprocessing.py

Post-processing for POINT-CONTACT (ball) bearings — everything that CONSUMES
an already-computed BallLoadDistributionResult (or BallBearingStiffness) to
derive a further quantity, rather than producing one itself.

Scope
-----
Q_j(), phi_j_global(), contact_distribution(), bearing_stiffness() and
DynamicEquivalentRollingElementLoad all take a converged
BallLoadDistributionResult as input. None of them run scipy.optimize — the
solve is already done by the time any of this file's functions are called.
This file imports ball_bearing.py only for typing (Bearing), never for its
solver class, and ball_bearing.py never imports this file — the dependency
is one-directional, post-processing depends on the solve output shape, not
the other way around.

BallLoadDistributionResult is imported from ball_bearing_results.py (the
local library), not from the global library.py. BallBearingStiffness is
defined directly in THIS file, not imported from library.py either — same
reasoning as BallLoadDistributionResult: everything before the final,
cross-type aggregation step (BearingResultsLibrary) belongs local, so this
module never needs a real import from library.py at all. The secant
stiffness projection (delta_r -> Kr_xz/Kr_xy/Ka) is identical for point and
line contact, so RollerBearingStiffness in
Roller_Bearing/roller_bearing_postprocessing.py duplicates the same
formula rather than either side importing the other or importing a shared
base from library.py — consistent with how RollerLoadDistributionResult /
BallLoadDistributionResult are each self-contained rather than sharing a
base.

References
----------
ISO/TS 16281:2008 §4.2 eq.(12)-(15) (kinematics), §4.3.2 eq.(25)-(28)
(dynamic equivalent load)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
    BallLoadDistributionResult,
)


# ---------------------------------------------------------------------------
# Per-element contact force / global angular position
# ---------------------------------------------------------------------------

def Q_j(bearing: Bearing, result: BallLoadDistributionResult) -> np.ndarray:
    """
    Per-element contact force [N], point-contact law Q_j = cp * delta_j^1.5.

    delta_j is already zero-floored by the solver for unloaded elements, so
    Q_j is zero there too — the np.maximum(..., 0.0) below is a defensive
    floor for any caller that hands in a delta_j not produced by this
    package's own solver, not something the solver itself needs.
    """
    return bearing.cp * np.maximum(result.delta_j, 0.0) ** 1.5


def phi_j_global(bearing: Bearing, result: BallLoadDistributionResult) -> np.ndarray:
    """
    Ball angular positions in the global frame [rad], wrapped to [0, 2*pi).

    bearing.phi_j is local (0 aligned with the resultant radial force plane);
    phi_Fr rotates that plane back into the global frame.
    """
    return (bearing.phi_j + result.phi_Fr) % (2.0 * np.pi)


def contact_distribution(bearing: Bearing, result: BallLoadDistributionResult,
                         frame: str = "global") -> np.ndarray:
    """
    Per-element (angle, contact force) pairs, shape (Z, 2): [phi, Q_j].

    frame="global" (default): phi = phi_j_global(bearing, result).
    frame="local": phi = bearing.phi_j as stored (0 aligned with the
    resultant radial force plane) — mainly for debugging the solve itself.
    """
    if frame == "global":
        phi = phi_j_global(bearing, result)
    elif frame == "local":
        phi = bearing.phi_j
    else:
        raise ValueError(f"contact_distribution: frame must be 'global' or 'local', got {frame!r}")

    return np.column_stack((phi, Q_j(bearing, result)))


# ---------------------------------------------------------------------------
# BallBearingStiffness — self-contained, no import from library.py
# ---------------------------------------------------------------------------

class BallBearingStiffness:
    """
    Secant stiffness of a point-contact (ball) bearing, decomposed onto the
    XZ / XY / axial axes. Fully self-contained: no import, subclassing, or
    other runtime dependency on library.py or on Roller_Bearing — see the
    module docstring for why this is duplicated rather than shared.

    Built from a converged BallLoadDistributionResult by projecting delta_r
    back onto the global axes:

        delta_r_xz = delta_r * cos(phi_Fr)
        delta_r_xy = delta_r * sin(phi_Fr)
        Kr_xz = Fr_xz / delta_r_xz
        Kr_xy = Fr_xy / delta_r_xy
        Ka    = Fa    / delta_a

    Any stiffness component is set to float('inf') when the corresponding
    displacement is negligible (rigid in that direction).

    Fr_xz, Fr_xy, Fa and delta_a are NOT stored on this object — Fr_xz/Fr_xy/Fa
    already live in BearingNodeData (SimpleFEMResultsLibrary) and delta_a
    already lives on the BallLoadDistributionResult that from_result() was
    built from; the caller has all three in scope by construction, since it
    had to pass them in to call from_result() in the first place. Keeping
    them here too would just be a second copy of data that already exists
    elsewhere. Only what from_result() actually computes — the projected
    displacements and the stiffnesses/regime derived from them — is stored.

    Ka_regime
    ---------
    "no_load"           Fa = 0
    "engaged"           delta_a >= 0 — axial contact active
    "closing_clearance" delta_a < 0  — axial clearance not yet closed

    Attributes
    ----------
    label      str
    delta_r_xz float [mm]      delta_r projected onto the XZ plane
    Kr_xz      float [N/mm]
    delta_r_xy float [mm]      delta_r projected onto the XY plane
    Kr_xy      float [N/mm]
    Ka         float | None   [N/mm]
    Ka_regime  str   | None
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
        """
        Build from a converged BallLoadDistributionResult.

        Fr_xz, Fr_xy, Fa are used only to compute Kr_xz/Kr_xy/Ka here — they
        are not retained on the returned object (see class docstring).
        """
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


def bearing_stiffness(bearing: Bearing, result: BallLoadDistributionResult,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      eps: float = 1e-9) -> BallBearingStiffness:
    """Secant stiffness (Kr_xz, Kr_xy, Ka) from a converged BallLoadDistributionResult."""
    return BallBearingStiffness.from_result(
        label=bearing.label, result=result,
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa, eps=eps,
    )


# ---------------------------------------------------------------------------
# DynamicEquivalentRollingElementLoad — ISO/TS 16281 §4.3.2, eq.(25)-(28)
# ---------------------------------------------------------------------------

_P_ROTATING   = 3.0        # eq.(25)/(28) — ring rotating relative to the load
_P_STATIONARY = 10.0 / 3.0  # eq.(26)/(27) — ring stationary relative to the load


@dataclass(frozen=True)
class DynamicEquivalentRollingElementLoad:
    """
    Dynamic equivalent load per rolling element, inner/outer raceway —
    ISO/TS 16281 §4.3.2, eq.(25)-(28).

    Both raceways use the same pair of power-mean forms; which one applies
    depends only on whether that raceway rotates relative to the bearing
    load direction, not on which raceway (inner/outer) it is:

    Rotating relative to the load — eq.(25) for inner, eq.(28) for outer:

        Q_e = ( (1/Z) * sum_j Q_j^3 )^(1/3)

    Stationary relative to the load — eq.(26) for inner, eq.(27) for outer:

        Q_e = ( (1/Z) * sum_j Q_j^(10/3) )^(3/10)

    The standard notes that for a normal load distribution the difference
    between the rotating and stationary forms is under 2% and can generally
    be neglected — but both are implemented here rather than assuming one.

    inner_rotating / outer_rotating are independent flags (not mutually
    exclusive) so both a rotating-inner/stationary-outer arrangement and a
    rotating-load case (e.g. unbalance) can be represented.

    Attributes
    ----------
    label            str
    Q_ei             float [N]   dynamic equivalent load, inner raceway
    Q_ee             float [N]   dynamic equivalent load, outer raceway
    inner_rotating   bool
    outer_rotating   bool
    Q_j              ndarray(Z,) [N]   per-element contact forces used to derive
                                       Q_ei/Q_ee — kept on the result so a caller
                                       doesn't have to recompute Q_j(bearing, result)
                                       separately (e.g. for a debug printout).
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
        Q = Q_j(bearing, result)
        _lbl = label or bearing.label

        Q_ei = _equivalent(Q, inner_rotating)
        Q_ee = _equivalent(Q, outer_rotating)

        return cls(label=_lbl, Q_ei=Q_ei, Q_ee=Q_ee,
                   inner_rotating=inner_rotating, outer_rotating=outer_rotating,
                   Q_j=Q)


def _equivalent(Q: np.ndarray, rotating: bool) -> float:
    """eq.(25)/(28) if rotating relative to the load, else eq.(26)/(27)."""
    p = _P_ROTATING if rotating else _P_STATIONARY
    return float(np.mean(Q ** p) ** (1.0 / p))