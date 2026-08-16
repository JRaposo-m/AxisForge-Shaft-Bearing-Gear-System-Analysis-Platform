"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing_postprocessing.py

Post-processing for LINE-CONTACT (radial cylindrical roller) bearings —
everything that CONSUMES an already-computed RollerLoadDistributionResult
(or BearingStiffness) to derive a further quantity, rather than producing
one itself. Mirrors Ball_Bearing/ball_bearing_postprocessing.py.

Scope
-----
Q_j(), phi_j_global(), contact_distribution(), lamina_distribution(),
bearing_stiffness(), stress_riser_factor() and LaminaDynamicEquivalentLoad
all take a converged RollerLoadDistributionResult as input. None of them
run scipy.optimize — the solve is already done by the time any of this
file's functions are called. This file imports roller_bearing.py only for
typing and for RollerLoadDistributionResult (needed to read q_jk/x_k off
the result); roller_bearing.py never imports this file — the dependency
is one-directional, post-processing depends on the solve output shape,
not the other way around.

Dynamic equivalent load — now per lamina, not per roller (correction)
-----------------------------------------------------------------------
An earlier version of this file computed a single per-bearing Q_ei/Q_ee
from Q_j (each roller's force summed over its laminae), using exponents
4 (rotating) / 9/2 (stationary) attributed to Harris & Kotzalas because
the ISO/TS 16281 excerpt available at the time stopped before §5.3. That
was the wrong quantity: ISO/TS 16281 §5.3.4, eq.(61)-(64) defines the
dynamic equivalent load PER LAMINA k (q_kei, q_kee — arrays over k, not a
scalar per bearing), built from q_j,k directly (not pre-summed over k into
Q_j), and weighted by the edge-stress concentration factors f_i[j,k] /
f_e[j,k] of eq.(58)-(59). The exponents themselves (4 rotating, 9/2
stationary — eq.61 vs eq.62) turned out to match what was already here;
what was missing was the per-lamina indexing and the f_i/f_e weighting.
LaminaDynamicEquivalentLoad below replaces the old per-roller class.

f_i[j,k]/f_e[j,k] (eq.58-59) need the actual local Hertzian contact
pressure P_Hi,j,k / P_He,j,k, which this package does not compute (only
the ISO/TS 16281 §5.2 lamina LOAD q_j,k via the load-deflection law, not
contact stress/pressure). eq.(60) gives an explicit first-order
APPROXIMATION for exactly this situation:

    f_i[k] = f_e[k] = 1 - 0.01 / ln(1.985 * |2k - n_s - 1| / (2*n_s - 2))

which needs only k and n_s — implemented as stress_riser_factor() below.
Per the standard's own note, eq.(60) is "valid for an approximated profile
obtained with the aid of Equations (42), (43) and (44)" — exactly
roller_bearing.ISO16281RollerSolver.roller_profile() — "and providing the
conditions of medium load and a total misalignment of the bearing of less
than 4' [~1.16e-3 rad] are fulfilled. For a general calculation, the use
of the methods described in References [5], [6] or [7] is recommended."
Neither this file nor its caller checks those conditions — the ISO/TS
16281 excerpt available in this session does not include Reference
[5]/[6]/[7]'s methods, so there is no fallback implemented here either;
verify the load/misalignment regime is appropriate before trusting the
approximation, or supply real Hertzian-pressure-based f_i/f_e instead.

Because f_i[k] = f_e[k] under this approximation (no j-dependence),
LaminaDynamicEquivalentLoad's q_kei and q_kee differ from each other only
through inner_rotating vs outer_rotating (i.e. only if one raceway rotates
relative to the load and the other doesn't) — the underlying q_j,k data
and f[k] weighting are otherwise identical between the two.

This class stops at eq.(64). Combining q_kei[k]/q_kee[k] across all n_s
laminae into a single bearing rating life is ISO/TS 16281 §5.4 (Basic
rating life), which the excerpt available in this session does not
include — no combination rule is invented here.

References
----------
ISO/TS 16281:2008 §5.2, eq.(36)-(46); §5.3.2-.4, eq.(56)-(64)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import BearingStiffness
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing import (
    RollerLoadDistributionResult,
)


def _require_roller_result(result, fn_name: str) -> None:
    if not isinstance(result, RollerLoadDistributionResult):
        raise TypeError(
            f"{fn_name}: expected a RollerLoadDistributionResult (with q_jk/x_k), "
            f"got {type(result).__name__}. This post-processing module only "
            f"consumes results produced by ISO16281RollerSolver."
        )


# ---------------------------------------------------------------------------
# Per-element (per-roller) contact force / global angular position
# ---------------------------------------------------------------------------

def Q_j(bearing: Bearing, result: RollerLoadDistributionResult) -> np.ndarray:
    """
    Total contact force per roller [N], summed over its n_s laminae:

        Q_j = sum_k q_j,k

    This is the line-contact analogue of eq.(45)'s inner sum, per roller
    (before the cos(phi_j) projection that eq.(45) applies to get Fr).
    """
    _require_roller_result(result, "Q_j")
    return np.sum(result.q_jk, axis=1)


def phi_j_global(bearing: Bearing, result: RollerLoadDistributionResult) -> np.ndarray:
    """
    Roller angular positions in the global frame [rad], wrapped to [0, 2*pi).

    bearing.phi_j is local (0 aligned with the resultant radial force
    plane); phi_Fr rotates that plane back into the global frame.
    """
    return (bearing.phi_j + result.phi_Fr) % (2.0 * np.pi)


def contact_distribution(bearing: Bearing, result: RollerLoadDistributionResult,
                         frame: str = "global") -> np.ndarray:
    """
    Per-roller (angle, total contact force) pairs, shape (Z, 2): [phi, Q_j].

    frame="global" (default): phi = phi_j_global(bearing, result).
    frame="local": phi = bearing.phi_j as stored — mainly for debugging the
    solve itself.
    """
    if frame == "global":
        phi = phi_j_global(bearing, result)
    elif frame == "local":
        phi = bearing.phi_j
    else:
        raise ValueError(f"contact_distribution: frame must be 'global' or 'local', got {frame!r}")

    return np.column_stack((phi, Q_j(bearing, result)))


def lamina_distribution(bearing: Bearing, result: RollerLoadDistributionResult,
                        j: int) -> np.ndarray:
    """
    Per-lamina (position, load) pairs for a single roller j, shape (n_s, 2):
    [x_k, q_j,k] — the pressure-profile-along-the-roller-length view the
    §5.2.3 profile function (roller_profile()) exists to keep well-behaved
    at the ends. Useful for plotting/inspecting edge loading on a specific
    roller (e.g. the most heavily loaded one, at phi_j closest to phi_Fr).

    j : index into bearing.phi_j / result.q_jk's first axis, NOT an angle.
    """
    _require_roller_result(result, "lamina_distribution")
    Z = result.q_jk.shape[0]
    if not (0 <= j < Z):
        raise IndexError(f"lamina_distribution: j={j} out of range for Z={Z} rollers.")
    return np.column_stack((result.x_k, result.q_jk[j, :]))


# ---------------------------------------------------------------------------
# Secant stiffness — thin wrapper around the single implementation in library.py
# ---------------------------------------------------------------------------

def bearing_stiffness(bearing: Bearing, result: RollerLoadDistributionResult,
                      Fr_xz: float, Fr_xy: float,
                      eps: float = 1e-9) -> BearingStiffness:
    """
    Secant stiffness (Kr_xz, Kr_xy) from a converged RollerLoadDistributionResult.

    Fa is always 0.0 here — radial roller bearings (NU/N-type) carry no
    axial load, so Ka comes back as float('inf') with regime "no_load"
    (see BearingStiffness.from_result), which is the physically correct
    statement rather than an edge case to special-case around.
    """
    return BearingStiffness.from_result(
        label=bearing.label, result=result,
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=0.0, eps=eps,
    )


# ---------------------------------------------------------------------------
# Stress-riser (edge-stress concentration) — ISO/TS 16281 §5.3.3, eq.(60)
# ---------------------------------------------------------------------------

_STRESS_RISER_LOG_ARG_EPS = 1e-12   # floor for ln() argument at the exact bearing mid-plane


def stress_riser_factor(n_s: int) -> np.ndarray:
    """
    Approximate stress-riser function f_i[k] = f_e[k] = f[k], k=1..n_s —
    eq.(60), a first-order approximation of the Hertz-pressure-based
    f_i[j,k]/f_e[j,k] of eq.(58)-(59) that needs only n_s. See the module
    docstring's "Dynamic equivalent load" section for the validity
    conditions (roller_profile()-shaped profile, medium load, total
    misalignment < 4') — not checked here.

        f[k] = 1 - 0.01 / ln(1.985 * |2k - n_s - 1| / (2*n_s - 2))

    At the roller mid-plane (only reachable when n_s is odd, k=(n_s+1)/2),
    the ln() argument is exactly 0; floored to _STRESS_RISER_LOG_ARG_EPS
    there rather than raising, since f[k] -> 1 in that limit anyway (no
    edge effect at the centre) — the floor just avoids -inf/divide-by-zero
    on the way to that same answer.

    Returns
    -------
    ndarray(n_s,) — f[k] for k=1..n_s, in lamina order (index 0 = k=1,
    matching bearing.x_k's ordering from lamina_positions()-style setup).
    """
    k = np.arange(1, n_s + 1)
    ratio = 1.985 * np.abs(2 * k - n_s - 1) / (2.0 * n_s - 2.0)
    ratio = np.maximum(ratio, _STRESS_RISER_LOG_ARG_EPS)
    return 1.0 - 0.01 / np.log(ratio)


# ---------------------------------------------------------------------------
# LaminaDynamicEquivalentLoad — ISO/TS 16281 §5.3.4, eq.(61)-(64)
# ---------------------------------------------------------------------------

_P_ROTATING   = 4.0    # eq.(61)/(64) — raceway rotating relative to the load
_P_STATIONARY = 4.5    # eq.(62)/(63) — raceway stationary relative to the load


@dataclass(frozen=True)
class LaminaDynamicEquivalentLoad:
    """
    Dynamic equivalent load PER LAMINA k, inner/outer raceway — ISO/TS
    16281 §5.3.4, eq.(61)-(64), using stress_riser_factor()'s eq.(60)
    approximation for f_i[j,k]/f_e[j,k]. See the module docstring for why
    this replaced an earlier, wrong per-roller version, and why q_kei and
    q_kee only differ via inner_rotating/outer_rotating under this
    approximation (f_i[k] = f_e[k]).

        q_kei[k] = f[k] * ( (1/Z) * sum_j q_j,k^p )^(1/p)
        q_kee[k] = f[k] * ( (1/Z) * sum_j q_j,k^p )^(1/p)
        p = 4 if that raceway rotates relative to the load, else 4.5

    One value PER LAMINA (n_s,) — NOT one per bearing. Compare against
    RollerElementCapacity.radial()'s q_ci/q_ce (eq.56-57, also per lamina),
    not against Q_ci/Q_ce (whole-roller). Combining across laminae into a
    single bearing life is ISO/TS 16281 §5.4, not implemented here (see
    module docstring).

    Attributes
    ----------
    label            str
    q_kei            ndarray(n_s,) [N]   per-lamina dynamic equivalent load, inner
    q_kee            ndarray(n_s,) [N]   per-lamina dynamic equivalent load, outer
    inner_rotating   bool
    outer_rotating   bool
    f_k              ndarray(n_s,)       stress-riser factor used, eq.(60)
    """
    label          : str
    q_kei          : np.ndarray
    q_kee          : np.ndarray
    inner_rotating : bool
    outer_rotating : bool
    f_k            : np.ndarray

    @classmethod
    def from_distribution(cls, bearing: Bearing, result: RollerLoadDistributionResult,
                          inner_rotating: bool = True,
                          outer_rotating: bool = False,
                          label: str = "") -> "LaminaDynamicEquivalentLoad":
        _require_roller_result(result, "LaminaDynamicEquivalentLoad.from_distribution")
        q_jk = result.q_jk                     # (Z, n_s)
        Z, n_s = q_jk.shape
        f_k = stress_riser_factor(n_s)
        _lbl = label or bearing.label

        q_kei = f_k * _lamina_mean(q_jk, inner_rotating)
        q_kee = f_k * _lamina_mean(q_jk, outer_rotating)

        return cls(label=_lbl, q_kei=q_kei, q_kee=q_kee,
                   inner_rotating=inner_rotating, outer_rotating=outer_rotating,
                   f_k=f_k)


def _lamina_mean(q_jk: np.ndarray, rotating: bool) -> np.ndarray:
    """
    ( (1/Z) * sum_j q_j,k^p )^(1/p) per lamina k — the core mean inside
    eq.(61)/(63) (before the f_i/f_e[j,k] weighting, applied separately
    by the caller since it factors out under the eq.60 approximation).
    """
    p = _P_ROTATING if rotating else _P_STATIONARY
    Z = q_jk.shape[0]
    return (np.sum(q_jk ** p, axis=0) / Z) ** (1.0 / p)