"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing_postprocessing.py

Post-processing for LINE-CONTACT roller bearings -- everything that
CONSUMES an already-computed RollerBearingResult (or RollerBearingStiffness)
to derive a further quantity, rather than producing one itself. Mirrors
Ball_Bearing/ball_bearing_postprocessing.py.

Scope
-----
Q_j(), phi_j_global(), contact_distribution(), bearing_stiffness(),
stress_riser_factor() and LaminaDynamicEquivalentLoad all take a converged
RollerBearingResult as input (lamina_distribution() and
LaminaDynamicEquivalentLoad.from_distribution() are the two exceptions --
see their own docstrings below for why they stay row-level primitives).
None of them run scipy.optimize -- the solve is already done by the time
any of this file's functions are called. This file imports both
RollerLoadDistributionResult and RollerBearingResult from
roller_bearing_results.py (not from roller_bearing_solver.py -- the solving
module is not a dependency of postprocessing at all); roller_bearing_solver.py
never imports this file -- the dependency is one-directional, post-processing
depends on the solve output shape, not the other way around.

RECONSTRUCTED, this turn -- row-aware, mirroring Ball_Bearing, prepared
ahead of need for a FUTURE thrust roller family
-----------------------------------------------------------------------------
Q_j, phi_j_global and contact_distribution take a RollerBearingResult and
ALWAYS return a list -- one entry per row, length 1 today for
CylindricalRollerFamily (radial NU/N-type, the only family in core/ today).
This mirrors the exact same treatment in
Ball_Bearing/ball_bearing_postprocessing.py, via the same _row_views()
pattern: single-row reads straight off `bearing`, a (currently unreachable)
multi-row branch would read per-row geometry off `bearing.rows[j]` dicts
the same way a future thrust roller family would have to produce them
(mirroring MultiRowThrustBallFamily.rows on the ball side) -- untested,
since nothing produces that shape today, but kept consistent rather than
reinvented later.

IMPORTANT -- this multi-row branch is NOT for CylindricalRollerFamily's own
`i` (row-count capacity multiplier, mirroring DeepGrooveBallFamily): that
stays a single raceway, single delta_r, never produces `bearing.rows`. The
`rows`-based shape this file is prepared for belongs to a future THRUST
roller family only (needle or cylindrical thrust roller -- not yet built in
core/), exactly like MultiRowThrustBallFamily is a separate family from
DeepGrooveBallFamily on the ball side, not an alternate mode of it. See
roller_bearing_results.py's module docstring for the full reasoning.

LaminaDynamicEquivalentLoad gets the same treatment as
DynamicEquivalentRollingElementLoad did on the ball side:
from_distribution() (the low-level, strictly-single-row primitive that does
all the actual eq.(61)-(64) math) is UNCHANGED; from_bearing_result() is a
new, thin, row-aware wrapper that calls it once per row and always returns
a list.

lamina_distribution() is the one function DELIBERATELY left row-level
(taking a raw RollerLoadDistributionResult, not the wrapped
RollerBearingResult) rather than made row-aware -- it already takes a roller
index j as a drill-down parameter; adding a second, row-of-the-bearing axis
on top would give it two unrelated indices to juggle for no benefit, since
its whole purpose is "let me look at ONE specific roller's lamina profile",
which is naturally scoped to one row's result already in the caller's hand.
A caller with a RollerBearingResult picks the row first
(`bearing_result.rows[0]` today; `bearing_result.rows[k]` for a future
multi-row one) and passes that row straight in, same as before.

RollerBearingStiffness is defined directly in THIS file, not imported from
the global library.py: everything before the final, cross-type aggregation
step (BearingResultsLibrary) belongs local, so this module never needs a
real import from library.py. The secant stiffness projection (delta_r ->
Kr_xz/Kr_xy/Ka) is identical for point and line contact, so this duplicates
the same formula as BallBearingStiffness in
Ball_Bearing/ball_bearing_postprocessing.py rather than either side
importing the other or importing a shared base from library.py -- consistent
with how RollerLoadDistributionResult / BallLoadDistributionResult are each
self-contained rather than sharing a base. bearing_stiffness() always reads
result.rows[0] -- stiffness is a property of the WHOLE bearing (the rigid
ring), never per-row, same reasoning as the ball side's bearing_stiffness().
Fa is passed as 0.0 here unconditionally -- correct for CylindricalRollerFamily
(radial, no axial load); revisit if this function is ever reused for a
future thrust roller family, where Fa would be the primary load.

Dynamic equivalent load -- per lamina, not per roller
-----------------------------------------------------------------------
ISO/TS 16281 §5.3.4, eq.(61)-(64) defines the dynamic equivalent load PER
LAMINA k (q_kei, q_kee -- arrays over k, not a scalar per bearing), built
from q_j,k directly (not pre-summed over k into Q_j), and weighted by the
edge-stress concentration factors f_i[j,k] / f_e[j,k] of eq.(58)-(59).

f_i[j,k]/f_e[j,k] (eq.58-59) need the actual local Hertzian contact
pressure P_Hi,j,k / P_He,j,k, which this package does not compute (only
the ISO/TS 16281 §5.2 lamina LOAD q_j,k via the load-deflection law, not
contact stress/pressure). eq.(60) gives an explicit first-order
APPROXIMATION for exactly this situation:

    f_i[k] = f_e[k] = 1 - 0.01 / ln(1.985 * |2k - n_s - 1| / (2*n_s - 2))

which needs only k and n_s -- implemented as stress_riser_factor() below.
Per the standard's own note, eq.(60) is "valid for an approximated profile
obtained with the aid of Equations (42), (43) and (44)" -- exactly
roller_bearing_solver.ISO16281RollerSolver.roller_profile() -- "and providing
the conditions of medium load and a total misalignment of the bearing of
less than 4' [~1.16e-3 rad] are fulfilled. For a general calculation, the
use of the methods described in References [5], [6] or [7] is recommended."
Neither this file nor its caller checks those conditions -- the ISO/TS
16281 excerpt available in this session does not include Reference
[5]/[6]/[7]'s methods, so there is no fallback implemented here either;
verify the load/misalignment regime is appropriate before trusting the
approximation, or supply real Hertzian-pressure-based f_i/f_e instead.

Because f_i[k] = f_e[k] under this approximation (no j-dependence),
LaminaDynamicEquivalentLoad's q_kei and q_kee differ from each other only
through inner_rotating vs outer_rotating (i.e. only if one raceway rotates
relative to the load and the other doesn't) -- the underlying q_j,k data
and f[k] weighting are otherwise identical between the two.

Basic reference rating life / Pref -- ISO/TS 16281 §5.3.5-.6, eq.(65)-(67)
-----------------------------------------------------------------------
Same structure as the ball side: BasicReferenceRatingLife.from_loads() is
the single-row primitive (eq.65 -- sums q_kci/q_kei/q_kce/q_kee over all
n_s laminae into that row's L10r), combine_row_L10r() combines rows
(Zaretsky eq.49a, e=9/8 line contact), basic_reference_rating_life() is
the row-aware wrapper. q_kci/q_kce (per-lamina capacity) are NOT computed
here -- see RollerElementCapacity.radial() (eq.56-57).

References
----------
ISO/TS 16281:2008 §5.2, eq.(36)-(46); §5.3.2-.6, eq.(56)-(67)
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.load_distribution.ISO_16281.Roller_Bearing.results import (
    RollerLoadDistributionResult,
    RollerBearingResult,
)


def _require_roller_result(result, fn_name: str) -> None:
    if not isinstance(result, RollerLoadDistributionResult):
        raise TypeError(
            f"{fn_name}: expected a RollerLoadDistributionResult (one row -- "
            f"with q_jk/x_k), got {type(result).__name__}. Pass one row of a "
            f"RollerBearingResult (e.g. bearing_result.rows[0]), not the "
            f"RollerBearingResult itself."
        )


# ---------------------------------------------------------------------------
# Row views -- the one place "single-row vs multi-row" is decided. Mirrors
# Ball_Bearing/ball_bearing_postprocessing.py's _row_views() exactly; the
# multi-row branch is UNEXERCISED today (see module docstring -- it is
# prepared for a future thrust roller family, not for CylindricalRollerFamily's
# own `i`) but kept structurally identical so that family slots in without
# this file changing shape again.
# ---------------------------------------------------------------------------

def _row_views(bearing: Bearing, result: RollerBearingResult) -> list[Any]:
    """
    One bearing-like view per row, index-aligned with result.rows.

    Single-row (result.is_multirow is False -- the only case that exists
    today): [bearing] itself -- no row structure to unwrap, `bearing`
    already carries Z/Dwe/Lwe/Dpw/phi_j/s/n_s/x_k/cL/cs/alpha_0/label
    directly.

    Multi-row (unexercised today -- no thrust roller family exists yet to
    produce this shape, see roller_bearing_results.py's module docstring):
    would read bearing.rows[j] (a plain dict, mirroring
    MultiRowThrustBallFamily.rows on the ball side) wrapped as a
    SimpleNamespace per row.
    """
    if not result.is_multirow:
        return [bearing]
    label = getattr(bearing, "label", "multirow")
    return [
        SimpleNamespace(**row, label=f"{label}-row{j}")
        for j, row in enumerate(bearing.rows)
    ]


# ---------------------------------------------------------------------------
# Per-element (per-roller) contact force / global angular position --
# always a list now, one entry per row (length 1 for single-row, i.e.
# always, today).
# ---------------------------------------------------------------------------

def Q_j(bearing: Bearing, result: RollerBearingResult) -> list[np.ndarray]:
    """
    Per-row, total contact force per roller [N], summed over its n_s
    laminae -- one ndarray(Z_row,) per row:

        Q_j = sum_k q_j,k

    This is the line-contact analogue of eq.(45)'s inner sum, per roller
    (before the cos(phi_j) projection that eq.(45) applies to get Fr).
    """
    views = _row_views(bearing, result)
    out = []
    for view, row in zip(views, result.rows):
        _require_roller_result(row, "Q_j")
        out.append(np.sum(row.q_jk, axis=1))
    return out


def phi_j_global(bearing: Bearing, result: RollerBearingResult) -> list[np.ndarray]:
    """
    Per-row roller angular positions in the global frame [rad], wrapped to
    [0, 2*pi) -- one ndarray(Z_row,) per row.

    Each row's own phi_j is local (0 aligned with the resultant radial
    force plane); that row's own phi_Fr rotates it back into the global
    frame.
    """
    views = _row_views(bearing, result)
    return [
        (view.phi_j + row.phi_Fr) % (2.0 * np.pi)
        for view, row in zip(views, result.rows)
    ]


def contact_distribution(bearing: Bearing, result: RollerBearingResult,
                         frame: str = "global") -> list[np.ndarray]:
    """
    Per-row (angle, total contact force) pairs -- one ndarray(Z_row, 2) per
    row: [phi, Q_j] columns, length 1 list for a single-row bearing (always,
    today).

    frame="global" (default): phi = phi_j_global(bearing, result)[row].
    frame="local": phi = that row's own phi_j as stored -- mainly for
    debugging the solve itself.
    """
    if frame == "global":
        phis = phi_j_global(bearing, result)
    elif frame == "local":
        phis = [view.phi_j for view in _row_views(bearing, result)]
    else:
        raise ValueError(f"contact_distribution: frame must be 'global' or 'local', got {frame!r}")

    Qs = Q_j(bearing, result)
    return [np.column_stack((phi, Q)) for phi, Q in zip(phis, Qs)]


def lamina_distribution(bearing: Bearing, result: RollerLoadDistributionResult,
                        j: int) -> np.ndarray:
    """
    Per-lamina (position, load) pairs for a single roller j, shape (n_s, 2):
    [x_k, q_j,k] -- the pressure-profile-along-the-roller-length view the
    §5.2.3 profile function (roller_profile()) exists to keep well-behaved
    at the ends. Useful for plotting/inspecting edge loading on a specific
    roller (e.g. the most heavily loaded one, at phi_j closest to phi_Fr).

    Deliberately NOT row-aware, unlike Q_j/phi_j_global/contact_distribution
    above -- `result` here is ONE row's RollerLoadDistributionResult, not
    the wrapped RollerBearingResult. Adding a second index for "which row"
    on top of `j` ("which roller") would give this function two unrelated
    axes to juggle for no benefit -- pick the row first
    (`bearing_result.rows[0]` today; `bearing_result.rows[k]` for a future
    multi-row one) and pass it straight in, same as calling this always
    worked before this file's row-aware reconstruction.

    j : index into bearing.phi_j / result.q_jk's first axis, NOT an angle.
    """
    _require_roller_result(result, "lamina_distribution")
    Z = result.q_jk.shape[0]
    if not (0 <= j < Z):
        raise IndexError(f"lamina_distribution: j={j} out of range for Z={Z} rollers.")
    return np.column_stack((result.x_k, result.q_jk[j, :]))


# ---------------------------------------------------------------------------
# RollerBearingStiffness -- a property of the WHOLE bearing, not per-row.
# Self-contained, no import from library.py.
# ---------------------------------------------------------------------------

class RollerBearingStiffness:
    """
    Secant stiffness of a line-contact (radial cylindrical roller) bearing,
    decomposed onto the XZ / XY / axial axes. Fully self-contained: no
    import, subclassing, or other runtime dependency on library.py or on
    Ball_Bearing -- see the module docstring for why this is duplicated
    rather than shared.

    Built from a converged RollerLoadDistributionResult (ONE row's worth --
    see bearing_stiffness() below for why row 0 is always the right one to
    pass here) by projecting delta_r back onto the global axes:

        delta_r_xz = delta_r * cos(phi_Fr)
        delta_r_xy = delta_r * sin(phi_Fr)
        Kr_xz = Fr_xz / delta_r_xz
        Kr_xy = Fr_xy / delta_r_xy
        Ka    = Fa    / delta_a

    Any stiffness component is set to float('inf') when the corresponding
    displacement is negligible (rigid in that direction).

    Ka_regime
    ---------
    "no_load"           Fa = 0 -- always the case here, since radial roller
                        bearings (NU/N-type) carry no axial load
    "engaged"           delta_a >= 0 -- axial contact active
    "closing_clearance" delta_a < 0  -- axial clearance not yet closed

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
                    result: RollerLoadDistributionResult,
                    Fr_xz: float,
                    Fr_xy: float,
                    Fa: float,
                    eps: float = 1e-9) -> "RollerBearingStiffness":
        """
        Build from a converged RollerLoadDistributionResult -- ONE row's
        result, not a RollerBearingResult. See bearing_stiffness() below
        for the public entry point that unwraps row 0 for you.

        Fr_xz, Fr_xy, Fa are used only to compute Kr_xz/Kr_xy/Ka here -- they
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


def bearing_stiffness(bearing: Bearing, result: RollerBearingResult,
                      Fr_xz: float, Fr_xy: float,
                      eps: float = 1e-9) -> RollerBearingStiffness:
    """
    Secant stiffness (Kr_xz, Kr_xy) of the WHOLE bearing -- a single
    RollerBearingStiffness regardless of row count.

    Always reads result.rows[0] (equivalently result.delta_r): today the
    only row there is; for a future thrust roller multi-row bearing, every
    row would need to share the same rigid-ring displacement at convergence
    for row 0 to remain the correct representative -- same idealization as
    the ball side's bearing_stiffness(), not yet exercised here (see module
    docstring).

    Fa is always 0.0 here -- correct for CylindricalRollerFamily (radial
    NU/N-type carries no axial load, so Ka comes back as float('inf') with
    regime "no_load", the physically correct statement rather than an edge
    case to special-case around). A future thrust roller family would need
    this reconsidered -- Fa would be the primary load there, not 0.
    """
    return RollerBearingStiffness.from_result(
        label=bearing.label, result=result.rows[0],
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=0.0, eps=eps,
    )


# ---------------------------------------------------------------------------
# Stress-riser (edge-stress concentration) -- ISO/TS 16281 §5.3.3, eq.(60)
# ---------------------------------------------------------------------------

_STRESS_RISER_LOG_ARG_EPS = 1e-12   # floor for ln() argument at the exact bearing mid-plane


def stress_riser_factor(n_s: int) -> np.ndarray:
    """
    Approximate stress-riser function f_i[k] = f_e[k] = f[k], k=1..n_s --
    eq.(60), a first-order approximation of the Hertz-pressure-based
    f_i[j,k]/f_e[j,k] of eq.(58)-(59) that needs only n_s. See the module
    docstring's "Dynamic equivalent load" section for the validity
    conditions (roller_profile()-shaped profile, medium load, total
    misalignment < 4') -- not checked here. Not row-aware -- n_s is a
    scalar property of one raceway's lamina model, not affected by row
    count.

        f[k] = 1 - 0.01 / ln(1.985 * |2k - n_s - 1| / (2*n_s - 2))

    At the roller mid-plane (only reachable when n_s is odd, k=(n_s+1)/2),
    the ln() argument is exactly 0; floored to _STRESS_RISER_LOG_ARG_EPS
    there rather than raising, since f[k] -> 1 in that limit anyway (no
    edge effect at the centre) -- the floor just avoids -inf/divide-by-zero
    on the way to that same answer.

    Returns
    -------
    ndarray(n_s,) -- f[k] for k=1..n_s, in lamina order (index 0 = k=1,
    matching bearing.x_k's ordering from lamina_positions()-style setup).
    """
    k = np.arange(1, n_s + 1)
    ratio = 1.985 * np.abs(2 * k - n_s - 1) / (2.0 * n_s - 2.0)
    ratio = np.maximum(ratio, _STRESS_RISER_LOG_ARG_EPS)
    return 1.0 - 0.01 / np.log(ratio)


# ---------------------------------------------------------------------------
# LaminaDynamicEquivalentLoad -- ISO/TS 16281 §5.3.4, eq.(61)-(64)
# ---------------------------------------------------------------------------

_P_ROTATING   = 4.0    # eq.(61)/(64) -- raceway rotating relative to the load
_P_STATIONARY = 4.5    # eq.(62)/(63) -- raceway stationary relative to the load


@dataclass(frozen=True)
class LaminaDynamicEquivalentLoad:
    """
    Dynamic equivalent load PER LAMINA k, inner/outer raceway -- ISO/TS
    16281 §5.3.4, eq.(61)-(64), using stress_riser_factor()'s eq.(60)
    approximation for f_i[j,k]/f_e[j,k]. ONE raceway's (one row's) result --
    see from_bearing_result() below for the row-aware entry point that
    returns a list of these, one per row. See the module docstring for why
    q_kei and q_kee only differ via inner_rotating/outer_rotating under
    this approximation (f_i[k] = f_e[k]).

        q_kei[k] = f[k] * ( (1/Z) * sum_j q_j,k^p )^(1/p)
        q_kee[k] = f[k] * ( (1/Z) * sum_j q_j,k^p )^(1/p)
        p = 4 if that raceway rotates relative to the load, else 4.5

    One value PER LAMINA (n_s,) -- NOT one per bearing. Compare against
    RollerElementCapacity.radial()'s q_ci/q_ce (eq.56-57, also per lamina),
    not against Q_ci/Q_ce (whole-roller). Combining across laminae into a
    single bearing life is basic_reference_rating_life() below.

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
        """
        Low-level, strictly single-row primitive -- ONE bearing-like object
        and ONE RollerLoadDistributionResult. Does all the actual
        eq.(61)-(64) math; from_bearing_result() below never reimplements
        it, only calls it once per row.
        """
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

    @classmethod
    def from_bearing_result(cls,
                            bearing: Bearing,
                            result: RollerBearingResult,
                            inner_rotating: bool = True,
                            outer_rotating: bool = False,
                            label: str = "") -> list["LaminaDynamicEquivalentLoad"]:
        """
        Row-aware entry point -- ALWAYS returns a list: length 1 today for
        CylindricalRollerFamily (see module docstring), length i for a
        future thrust roller bearing's i rows, index-aligned with
        result.rows / bearing.rows. Mirrors
        DynamicEquivalentRollingElementLoad.from_bearing_result() on the
        ball side exactly.

        Per row, this only ever calls from_distribution() once -- it does
        not merge q_j,k arrays across rows (each row has its own raceway;
        eq.(61)-(64) is defined per raceway PER ROW) and it does not scale
        anything by row count.

        inner_rotating/outer_rotating are forwarded unchanged to every
        row's from_distribution() call -- the whole bearing (all rows)
        shares one physical rotating/stationary arrangement, it is not a
        per-row property.
        """
        views = _row_views(bearing, result)
        lbl   = label or getattr(bearing, "label", "")
        multi = result.is_multirow
        return [
            cls.from_distribution(
                view, row,
                inner_rotating=inner_rotating, outer_rotating=outer_rotating,
                label=(f"{lbl}-row{j}" if multi else lbl),
            )
            for j, (view, row) in enumerate(zip(views, result.rows))
        ]


def _lamina_mean(q_jk: np.ndarray, rotating: bool) -> np.ndarray:
    """
    ( (1/Z) * sum_j q_j,k^p )^(1/p) per lamina k -- the core mean inside
    eq.(61)/(63) (before the f_i/f_e[j,k] weighting, applied separately
    by the caller since it factors out under the eq.60 approximation).
    """
    p = _P_ROTATING if rotating else _P_STATIONARY
    Z = q_jk.shape[0]
    return (np.sum(q_jk ** p, axis=0) / Z) ** (1.0 / p)


# ---------------------------------------------------------------------------
# BasicReferenceRatingLife / DynamicEquivalentReferenceLoad -- eq.(65)-(67),
# ISO/TS 16281 §5.3.5 (L10r), §5.3.6 (Pref)
# ---------------------------------------------------------------------------

_E_ROLLER = 9.0 / 8.0   # Weibull slope, line contact -- Zaretsky eq.(49a)/(49b)


@dataclass(frozen=True)
class BasicReferenceRatingLife:
    """L10r for one row/raceway pair -- eq.(65), summed over n_s laminae."""
    label : str
    L10r  : float
    q_kci : np.ndarray
    q_kei : np.ndarray
    q_kce : np.ndarray
    q_kee : np.ndarray

    @classmethod
    def from_loads(cls, label: str, q_kci: np.ndarray, q_kei: np.ndarray,
                   q_kce: np.ndarray, q_kee: np.ndarray) -> "BasicReferenceRatingLife":
        if np.any(q_kci <= 0.0) or np.any(q_kce <= 0.0):
            raise ValueError(
                f"BasicReferenceRatingLife.from_loads({label!r}): q_kci/q_kce (capacity) must be positive."
            )
        if np.any(q_kei < 0.0) or np.any(q_kee < 0.0):
            raise ValueError(
                f"BasicReferenceRatingLife.from_loads({label!r}): q_kei/q_kee must be non-negative."
            )
        # (q_kci/q_kei)^-4.5 == (q_kei/q_kci)^4.5 -- rewritten this way so an
        # unloaded lamina (q_kei/q_kee = 0, e.g. a crowned profile's zero-
        # load edge) contributes 0 to the sum instead of dividing by zero.
        terms = (q_kei / q_kci) ** 4.5 + (q_kee / q_kce) ** 4.5
        L10r = float(np.sum(terms) ** (-8.0 / 9.0))
        return cls(label=label, L10r=L10r, q_kci=q_kci, q_kei=q_kei, q_kce=q_kce, q_kee=q_kee)


def combine_row_L10r(L10r_rows: list[float], e: float = _E_ROLLER) -> float:
    """Bearing-level L10r from n rows -- Zaretsky eq.(49a): L^-e = sum(Li^-e)."""
    if len(L10r_rows) < 2:
        raise ValueError(f"combine_row_L10r needs >= 2 rows; got {len(L10r_rows)}.")
    if any(L <= 0.0 for L in L10r_rows):
        raise ValueError(f"All L10r_rows must be positive; got {list(L10r_rows)}.")
    return sum(L ** (-e) for L in L10r_rows) ** (-1.0 / e)


def basic_reference_rating_life(
    label: str,
    q_kci_rows: list[np.ndarray], q_kei_rows: list[np.ndarray],
    q_kce_rows: list[np.ndarray], q_kee_rows: list[np.ndarray],
    e: float = _E_ROLLER,
) -> tuple[list[BasicReferenceRatingLife], float]:
    """Per-row L10r (eq.65) + combined bearing L10r (combine_row_L10r() for n>=2)."""
    n = len(q_kci_rows)
    if not (len(q_kei_rows) == len(q_kce_rows) == len(q_kee_rows) == n):
        raise ValueError("basic_reference_rating_life: row lists must all be the same length.")
    if n == 0:
        raise ValueError("basic_reference_rating_life: got 0 rows.")

    per_row = [
        BasicReferenceRatingLife.from_loads(
            label=f"{label}-row{j}" if n > 1 else label,
            q_kci=q_kci_rows[j], q_kei=q_kei_rows[j],
            q_kce=q_kce_rows[j], q_kee=q_kee_rows[j],
        )
        for j in range(n)
    ]
    L10r_bearing = (per_row[0].L10r if n == 1
                    else combine_row_L10r([r.L10r for r in per_row], e=e))
    return per_row, L10r_bearing


@dataclass(frozen=True)
class DynamicEquivalentReferenceLoad:
    """Pref -- eq.(66)-(67): Pref_r = Cr/L10r^(3/10), Pref_a = Ca/L10r^(3/10)."""
    label  : str
    Pref_r : float | None
    Pref_a : float | None

    @classmethod
    def from_L10r(cls, label: str, L10r: float,
                  Cr: float | None = None, Ca: float | None = None
                  ) -> "DynamicEquivalentReferenceLoad":
        if Cr is None and Ca is None:
            raise ValueError(f"DynamicEquivalentReferenceLoad.from_L10r({label!r}): need Cr and/or Ca.")
        if L10r <= 0.0:
            raise ValueError(f"L10r must be positive; got {L10r}.")
        denom = L10r ** (3.0 / 10.0)
        return cls(label=label,
                   Pref_r=(Cr / denom) if Cr is not None else None,
                   Pref_a=(Ca / denom) if Ca is not None else None)