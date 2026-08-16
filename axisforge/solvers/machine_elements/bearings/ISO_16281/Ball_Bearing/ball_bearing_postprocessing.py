"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Radial_Ball_Bearing/radial_ball_bearing_postprocessing.py

Post-processing for POINT-CONTACT (ball) bearings — everything that CONSUMES
an already-computed LoadDistributionResult (or BearingStiffness) to derive a
further quantity, rather than producing a LoadDistributionResult itself.

Scope
-----
Q_j(), phi_j_global(), contact_distribution(), bearing_stiffness() and
DynamicEquivalentRollingElementLoad all take a converged LoadDistributionResult
as input. None of them run scipy.optimize — the solve is already done by the
time any of this file's functions are called. This file imports
radial_ball_bearing.py only for typing (Bearing), never for its solver
class, and radial_ball_bearing.py never imports this file — the dependency
is one-directional, post-processing depends on the solve output shape, not
the other way around.

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
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    LoadDistributionResult,
    BearingStiffness,
)


# ---------------------------------------------------------------------------
# Per-element contact force / global angular position
# ---------------------------------------------------------------------------

def Q_j(bearing: Bearing, result: LoadDistributionResult) -> np.ndarray:
    """
    Per-element contact force [N], point-contact law Q_j = cp * delta_j^1.5.

    delta_j is already zero-floored by the solver for unloaded elements, so
    Q_j is zero there too — the np.maximum(..., 0.0) below is a defensive
    floor for any caller that hands in a delta_j not produced by this
    package's own solver, not something the solver itself needs.
    """
    return bearing.cp * np.maximum(result.delta_j, 0.0) ** 1.5


def phi_j_global(bearing: Bearing, result: LoadDistributionResult) -> np.ndarray:
    """
    Ball angular positions in the global frame [rad], wrapped to [0, 2*pi).

    bearing.phi_j is local (0 aligned with the resultant radial force plane);
    phi_Fr rotates that plane back into the global frame.
    """
    return (bearing.phi_j + result.phi_Fr) % (2.0 * np.pi)


def contact_distribution(bearing: Bearing, result: LoadDistributionResult,
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
# Secant stiffness — thin wrapper around the single implementation in library.py
# ---------------------------------------------------------------------------

def bearing_stiffness(bearing: Bearing, result: LoadDistributionResult,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      eps: float = 1e-9) -> BearingStiffness:
    """Secant stiffness (Kr_xz, Kr_xy, Ka) from a converged LoadDistributionResult."""
    return BearingStiffness.from_result(
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
    def from_distribution(cls, bearing: Bearing, result: LoadDistributionResult,
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