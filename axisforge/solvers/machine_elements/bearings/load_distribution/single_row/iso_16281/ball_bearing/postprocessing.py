"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_postprocessing.py

Post-processing for POINT-CONTACT (ball) bearings -- everything that
CONSUMES an already-computed BallBearingResult (single-row or multi-row,
uniformly) to derive a further quantity, rather than producing one itself.

Scope
-----
Q_j(), phi_j_global(), contact_distribution(), bearing_stiffness() and
DynamicEquivalentRollingElementLoad all take a converged BallBearingResult
as input. None of them run scipy.optimize -- the solve is already done by
the time any of this file's functions are called. This file imports
ball_bearing_solver.py only for typing (Bearing), never for its solver
class, and ball_bearing_solver.py never imports this file -- the
dependency is one-directional, post-processing depends on the solve output
shape, not the other way around.

RECONSTRUCTED, this turn -- one file, row-aware throughout
------------------------------------------------------------
This file used to only know about single-row bearings; a second file,
ball_bearing_multirow_postprocessing.py, duplicated the orchestration
needed to call this file's DynamicEquivalentRollingElementLoad once per
row for a multi-row bearing. That file is RETIRED -- its one function,
multirow_dynamic_equivalent_load(), is folded in here as
DynamicEquivalentRollingElementLoad.from_bearing_result(), which now
handles single-row (1 row) and multi-row (i rows) through the exact same
code path.

Every function below now takes (bearing, result: BallBearingResult) rather
than (bearing, result: BallLoadDistributionResult) -- BallBearingResult.rows
is always a list (length 1 or i, see ball_bearing_results.py), so "is this
multi-row" is answered once, by _row_views() below, and nothing downstream
has to branch on it again. Q_j / phi_j_global / contact_distribution /
DynamicEquivalentRollingElementLoad.from_bearing_result() ALWAYS return a
list now -- length 1 for single-row, length i for multi-row -- there is no
more asymmetry where a single-row caller gets a bare object and a
multi-row caller gets a list; callers index [0] explicitly for the
single-row case, same convention as BallBearingResult.rows itself and as
BearingResultsLibrary's own list fields in the global library.py.

The one exception is bearing_stiffness()/BallBearingStiffness: stiffness
is a property of the WHOLE bearing (the rigid ring), never per-row, so it
keeps returning a single object regardless of row count -- see its own
docstring below for why row 0 is always the correct representative.

DynamicEquivalentRollingElementLoad.from_distribution() -- the low-level,
strictly-single-row primitive taking ONE bearing-like object and ONE
BallLoadDistributionResult -- is UNCHANGED and still does all the actual
ISO/TS 16281 eq.(25)-(28) math. from_bearing_result() is a new, thin
wrapper that calls it once per row; it does not reimplement it.

BasicReferenceRatingLife/combine_row_L10r/DynamicEquivalentReferenceLoad
(eq.29-31) below are ball-specific: eq.(29)'s exponents (10/3, 9/10) are
§4.3.3 of ISO/TS 16281, the point-contact section. Roller's own rating
life is a different formula entirely (§5.3.1.2, lamina-based, not yet
implemented) -- it will NOT reuse this code, it gets its own version in
roller_bearing_postprocessing.py.

Row combination uses Zaretsky, E.V., "Rolling Bearing Life Prediction,
Theory, and Application", NASA/TP-2013-215305/REV1, 2016, eq.(49a)/(49b):
1/L^e = 1/L1^e + ... + 1/Li^e (e=10/9 ball). Pref (eq.30-31) is back-
derived from L10r, not an independent equivalent-load formula.

References
----------
ISO/TS 16281:2008 §4.2 eq.(12)-(15) (kinematics), §4.3.2 eq.(25)-(28)
(dynamic equivalent load), §4.3.3 eq.(29) (L10r), §4.3.4 eq.(30)-(31) (Pref)
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.ball_bearing.results import (
    BallLoadDistributionResult,
    BallBearingResult,
)


# ---------------------------------------------------------------------------
# Row views -- the one place "single-row vs multi-row" is decided
# ---------------------------------------------------------------------------

def _row_views(bearing: Bearing, result: BallBearingResult) -> list[Any]:
    """
    One bearing-like view per row, index-aligned with result.rows.

    Single-row (result.is_multirow is False): [bearing] itself -- there is
    no row structure to unwrap, `bearing` already carries A/alpha_0/phi_j/
    Ri/cp/Dpw/Z/label directly.

    Multi-row: bearing.rows[j] (a plain dict, see MultiRowThrustBallFamily)
    wrapped as a SimpleNamespace per row -- same construction
    ball_bearing_multirow_solver.py already uses to hand each row to
    ISO16281BallSolver.solve_contact(). Each row's own point-contact
    geometry, not the bearing's own top-level attributes (which do not
    exist for a multi-row bearing -- see that family's own docstring).

    Raises AttributeError (via bearing.rows) if result.is_multirow is True
    but `bearing` does not actually expose `.rows` -- i.e. if the caller
    passed a `result` that was not produced by solving THIS `bearing`.
    """
    if not result.is_multirow:
        return [bearing]
    label = getattr(bearing, "label", "multirow")
    return [
        SimpleNamespace(**row, label=f"{label}-row{j}")
        for j, row in enumerate(bearing.rows)
    ]


# ---------------------------------------------------------------------------
# Per-element contact force / global angular position -- always a list now,
# one entry per row (length 1 for single-row).
# ---------------------------------------------------------------------------

def Q_j(bearing: Bearing, result: BallBearingResult) -> list[np.ndarray]:
    """
    Per-row, per-element contact force [N], point-contact law
    Q_j = cp * delta_j^1.5 -- one ndarray(Z_row,) per row.

    delta_j is already zero-floored by the solver for unloaded elements, so
    Q_j is zero there too -- the np.maximum(..., 0.0) below is a defensive
    floor for any caller that hands in a delta_j not produced by this
    package's own solver, not something the solver itself needs.
    """
    views = _row_views(bearing, result)
    return [
        view.cp * np.maximum(row.delta_j, 0.0) ** 1.5
        for view, row in zip(views, result.rows)
    ]


def phi_j_global(bearing: Bearing, result: BallBearingResult) -> list[np.ndarray]:
    """
    Per-row ball angular positions in the global frame [rad], wrapped to
    [0, 2*pi) -- one ndarray(Z_row,) per row.

    Each row's own phi_j is local (0 aligned with the resultant radial
    force plane); that row's own phi_Fr rotates it back into the global
    frame -- for a multi-row bearing every row shares the same phi_Fr (one
    applied-force direction for the whole ring, see
    ball_bearing_multirow_solver.py), but each row's BallLoadDistribution
    Result carries its own copy, so this reads it per-row rather than
    assuming row 0's applies to all.
    """
    views = _row_views(bearing, result)
    return [
        (view.phi_j + row.phi_Fr) % (2.0 * np.pi)
        for view, row in zip(views, result.rows)
    ]


def contact_distribution(bearing: Bearing, result: BallBearingResult,
                         frame: str = "global") -> list[np.ndarray]:
    """
    Per-row (angle, contact force) pairs -- one ndarray(Z_row, 2) per row:
    [phi, Q_j] columns, length 1 list for a single-row bearing.

    frame="global" (default): phi = phi_j_global(bearing, result)[row].
    frame="local": phi = that row's own phi_j as stored (0 aligned with the
    resultant radial force plane) -- mainly for debugging the solve itself.
    """
    if frame == "global":
        phis = phi_j_global(bearing, result)
    elif frame == "local":
        phis = [view.phi_j for view in _row_views(bearing, result)]
    else:
        raise ValueError(f"contact_distribution: frame must be 'global' or 'local', got {frame!r}")

    Qs = Q_j(bearing, result)
    return [np.column_stack((phi, Q)) for phi, Q in zip(phis, Qs)]


# ---------------------------------------------------------------------------
# BallBearingStiffness -- a property of the WHOLE bearing, not per-row.
# Self-contained, no import from library.py.
# ---------------------------------------------------------------------------

class BallBearingStiffness:
    """
    Secant stiffness of a point-contact (ball) bearing, decomposed onto the
    XZ / XY / axial axes. Fully self-contained: no import, subclassing, or
    other runtime dependency on library.py or on Roller_Bearing -- see the
    module docstring for why this is duplicated rather than shared.

    Built from a converged BallLoadDistributionResult (ONE row's worth --
    see bearing_stiffness() below for why row 0 is always the right one to
    pass here, single-row or multi-row alike) by projecting delta_r back
    onto the global axes:

        delta_r_xz = delta_r * cos(phi_Fr)
        delta_r_xy = delta_r * sin(phi_Fr)
        Kr_xz = Fr_xz / delta_r_xz
        Kr_xy = Fr_xy / delta_r_xy
        Ka    = Fa    / delta_a

    Any stiffness component is set to float('inf') when the corresponding
    displacement is negligible (rigid in that direction).

    Fr_xz, Fr_xy, Fa and delta_a are NOT stored on this object -- Fr_xz/Fr_xy/Fa
    already live in BearingNodeData (SimpleFEMResultsLibrary) and delta_a
    already lives on the BallLoadDistributionResult that from_result() was
    built from; the caller has all three in scope by construction, since it
    had to pass them in to call from_result() in the first place. Keeping
    them here too would just be a second copy of data that already exists
    elsewhere. Only what from_result() actually computes -- the projected
    displacements and the stiffnesses/regime derived from them -- is stored.

    Ka_regime
    ---------
    "no_load"           Fa = 0
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
                    result: BallLoadDistributionResult,
                    Fr_xz: float,
                    Fr_xy: float,
                    Fa: float,
                    eps: float = 1e-9) -> "BallBearingStiffness":
        """
        Build from a converged BallLoadDistributionResult -- ONE row's
        result, not a BallBearingResult. See bearing_stiffness() below for
        the public entry point that unwraps row 0 for you.

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


def bearing_stiffness(bearing: Bearing, result: BallBearingResult,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      eps: float = 1e-9) -> BallBearingStiffness:
    """
    Secant stiffness (Kr_xz, Kr_xy, Ka) of the WHOLE bearing -- a single
    BallBearingStiffness regardless of row count.

    Always reads result.rows[0] (equivalently result.delta_r/result.delta_a):
    for a multi-row bearing, every row shares the same rigid-ring
    displacement at convergence (the co-located-rows idealization -- see
    ball_bearing_multirow_solver.py's module docstring), so row 0 is the
    representative displacement for the WHOLE ring; for a single-row
    bearing row 0 is simply the only row there is. Fr_xz/Fr_xy/Fa are the
    TOTAL external loads on the bearing (not a per-row share) either way --
    stiffness is total force / shared displacement, not a per-row quantity.
    """
    return BallBearingStiffness.from_result(
        label=bearing.label, result=result.rows[0],
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa, eps=eps,
    )


# ---------------------------------------------------------------------------
# DynamicEquivalentRollingElementLoad -- ISO/TS 16281 §4.3.2, eq.(25)-(28)
# ---------------------------------------------------------------------------

_P_ROTATING   = 3.0        # eq.(25)/(28) -- ring rotating relative to the load
_P_STATIONARY = 10.0 / 3.0  # eq.(26)/(27) -- ring stationary relative to the load


@dataclass(frozen=True)
class DynamicEquivalentRollingElementLoad:
    """
    Dynamic equivalent load per rolling element, inner/outer raceway --
    ISO/TS 16281 §4.3.2, eq.(25)-(28). ONE raceway's (one row's) result --
    see from_bearing_result() below for the row-aware entry point that
    returns a list of these, one per row.

    Both raceways use the same pair of power-mean forms; which one applies
    depends only on whether that raceway rotates relative to the bearing
    load direction, not on which raceway (inner/outer) it is:

    Rotating relative to the load -- eq.(25) for inner, eq.(28) for outer:

        Q_e = ( (1/Z) * sum_j Q_j^3 )^(1/3)

    Stationary relative to the load -- eq.(26) for inner, eq.(27) for outer:

        Q_e = ( (1/Z) * sum_j Q_j^(10/3) )^(3/10)

    The standard notes that for a normal load distribution the difference
    between the rotating and stationary forms is under 2% and can generally
    be neglected -- but both are implemented here rather than assuming one.

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
                                       Q_ei/Q_ee -- kept on the result so a caller
                                       doesn't have to recompute Q_j separately
                                       (e.g. for a debug printout).
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
        """
        Low-level, strictly single-row primitive -- ONE bearing-like object
        (needs .cp) and ONE BallLoadDistributionResult. Does all the actual
        eq.(25)-(28) math; from_bearing_result() below never reimplements
        it, only calls it once per row.
        """
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
        """
        Row-aware entry point -- ALWAYS returns a list: length 1 for a
        single-row bearing, length i for a multi-row bearing's i rows,
        index-aligned with result.rows / bearing.rows. This replaces the
        old split between calling from_distribution() directly (single-row)
        and the separate multirow_dynamic_equivalent_load() function
        (multi-row, now retired) -- one call, either row count.

        Per row, this only ever calls from_distribution() once -- it does
        not merge Q_j arrays across rows (each row has its own raceway
        rotation history; eq.(25)-(28) is defined per raceway PER ROW, see
        the retired multirow_dynamic_equivalent_load()'s docstring for why
        that would be physically meaningless) and it does not scale
        anything by row count (no "x2" anywhere, for any i).

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