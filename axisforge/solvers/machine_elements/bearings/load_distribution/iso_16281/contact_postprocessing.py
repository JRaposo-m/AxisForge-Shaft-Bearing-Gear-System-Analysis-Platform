"""
axisforge/solvers/machine_elements/bearings/load_distribution/single_row/iso_16281/contact_postprocessing.py

Single- and multi-row postprocessing, point and line contact. Shared
pieces on top (ContactBearingStiffness, combine_row_L10r,
DynamicEquivalentReferenceLoadBase, _row_views), then point-contact
(ball) functions/classes, then line-contact (roller).

_row_views(bearing, result) is what makes every *_Q_j / *_phi_j_global /
*_contact_distribution / *_from_bearing_result function below work
unmodified whether `result` wraps 1 row (single-row bearing) or N rows
(multi-row, shared-displacement) -- a single row returns the scalar/array
it always returned; N rows return a list of them, one per row. This is
the same generalization the pre-refactor multirow postprocessing files
used under the same name, folded in here instead of kept in a parallel
file, per the same "abstract/shared on top, concretes below, one file"
shape as contact_solver.py.

Shared-displacement state (delta_r, delta_a, psi, phi_Fr) is stored ONCE
on the *BearingResult, not per row -- functions below read it from
`result`, and per-row data (delta_j, q_jk, x_k, ...) from `result.rows[i]`.

Never imports contact_solver.py's solver classes -- only the *Result
shapes they produce. contact_solver.py never imports this file -- one-
directional, postprocessing depends on the solve output shape, not the
other way around.

BasicReferenceRatingLife and DynamicEquivalentRollingElementLoad/
LaminaDynamicEquivalentLoad stay separate per contact type (not merged
into the shared section): their formulas/data shapes (scalar per element
vs array per lamina, roller's extra stress_riser_factor weighting) differ
enough that forcing a shared base would hide more than it'd save.

References
----------
ISO/TS 16281:2008 Sec 4.2 eq.(12)-(15), Sec 4.3.2-.4 eq.(25)-(31) -- point contact
ISO/TS 16281:2008 Sec 5.2 eq.(36)-(46), Sec 5.3.2-.6 eq.(56)-(67) -- line contact
Zaretsky, E.V., "Rolling Bearing Life Prediction, Theory, and Application",
NASA/TP-2013-215305/REV1, 2016, eq.(49a)/(49b) -- row combination
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.results.bearings.load_distribution.load_distribution_results import (
    BallLoadDistributionResult, BallBearingResult,
    RollerLoadDistributionResult, RollerBearingResult,
)


# =====================================================================
# ---- Shared: ContactBearingStiffness, combine_row_L10r, Pref base,
# ----         _row_views ----------------------------------------------
# =====================================================================

class ContactBearingStiffness:
    """Secant stiffness of a contact bearing (point or line),
    XZ / XY / axial axes -- identical projection either way:

        delta_r_xz = delta_r * cos(phi_Fr);  Kr_xz = Fr_xz / delta_r_xz
        delta_r_xy = delta_r * sin(phi_Fr);  Kr_xy = Fr_xy / delta_r_xy
        Ka = Fa / delta_a

    float('inf') where the corresponding displacement is negligible.
    Ka_regime: "no_load" (Fa=0) / "engaged" (delta_a>=0) /
    "closing_clearance" (delta_a<0).

    TODO (review): inf when a projected displacement is ~0 (e.g.
    phi_Fr ~ 90 deg -> Kr_xz = inf) is known to be wrong."""

    __slots__ = ("label", "delta_r_xz", "Kr_xz", "delta_r_xy", "Kr_xy", "Ka", "Ka_regime")

    def __init__(self, label, delta_r_xz, Kr_xz, delta_r_xy, Kr_xy, Ka=None, Ka_regime=None):
        self.label      = label
        self.delta_r_xz = delta_r_xz
        self.Kr_xz      = Kr_xz
        self.delta_r_xy = delta_r_xy
        self.Kr_xy      = Kr_xy
        self.Ka         = Ka
        self.Ka_regime  = Ka_regime

    @classmethod
    def from_result(cls, label: str, result, Fr_xz: float, Fr_xy: float, Fa: float,
                    eps: float = 1e-9) -> "ContactBearingStiffness":
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

        return cls(label=label, delta_r_xz=delta_r_xz, Kr_xz=Kr_xz,
                   delta_r_xy=delta_r_xy, Kr_xy=Kr_xy, Ka=Ka, Ka_regime=regime)


def bearing_stiffness(bearing: Bearing, result, Fr_xz: float, Fr_xy: float,
                      Fa: float = 0.0, eps: float = 1e-9) -> ContactBearingStiffness:
    """Secant stiffness (Kr_xz, Kr_xy, Ka) of the whole bearing -- shared
    by ball (pass Fa explicitly) and roller (Fa defaults 0.0 -- radial
    roller bearings carry no axial load). delta_r/delta_a/phi_Fr are read
    from the *BearingResult itself: stored once per bearing (shared
    displacement), not per row -- so 1 or N rows give the same call."""
    return ContactBearingStiffness.from_result(
        label=bearing.label, result=result,
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa, eps=eps,
    )


def combine_row_L10r(L10r_rows: list[float], e: float) -> float:
    """Bearing-level L10r from n rows -- Zaretsky eq.(49a): L^-e = sum(Li^-e).
    e has no default -- caller passes its own contact type's Weibull
    exponent explicitly (_E_BALL below or _E_ROLLER below)."""
    if len(L10r_rows) < 2:
        raise ValueError(f"combine_row_L10r needs >= 2 rows; got {len(L10r_rows)}.")
    if any(L <= 0.0 for L in L10r_rows):
        raise ValueError(f"All L10r_rows must be positive; got {list(L10r_rows)}.")
    return sum(L ** (-e) for L in L10r_rows) ** (-1.0 / e)


def _row_views(bearing, result):
    """Per-row (bearing_row, result_row) pairs.

    A single-row bearing (no .rows attribute) is represented as ONE pair
    -- (bearing, result.rows[0]) -- so every function below that loops
    over _row_views() behaves exactly as before for single-row callers.
    A multi-row bearing.rows entry may be a plain dict (as produced by a
    family/fixture) or an object; dicts are wrapped in a SimpleNamespace
    so the same attribute-based access (rb.cp, rb.phi_j, ...) works
    either way -- same convention as MultiRowSolverBase._row_views() in
    contact_solver.py."""
    if not hasattr(bearing, "rows"):
        return [(bearing, result.rows[0])]
    row_bearings = [SimpleNamespace(**row) if isinstance(row, dict) else row for row in bearing.rows]
    return list(zip(row_bearings, result.rows))


@dataclass(frozen=True)
class DynamicEquivalentReferenceLoadBase:
    """Pref = C / L10r^exponent -- shared shape, exponent set per contact
    type by the concrete subclass below (BallDynamicEquivalentReferenceLoad:
    1/3, eq.30-31; RollerDynamicEquivalentReferenceLoad: 3/10, eq.66-67)."""
    label  : str
    Pref_r : float | None
    Pref_a : float | None

    _PREF_EXPONENT: ClassVar[float] = 0.0

    @classmethod
    def from_L10r(cls, label: str, L10r: float,
                  Cr: float | None = None, Ca: float | None = None
                  ) -> "DynamicEquivalentReferenceLoadBase":
        if cls._PREF_EXPONENT <= 0.0:
            raise NotImplementedError(f"{cls.__name__} must set _PREF_EXPONENT.")
        if Cr is None and Ca is None:
            raise ValueError(f"{cls.__name__}.from_L10r({label!r}): need Cr and/or Ca.")
        if L10r <= 0.0:
            raise ValueError(f"L10r must be positive; got {L10r}.")
        denom = L10r ** cls._PREF_EXPONENT
        return cls(label=label,
                   Pref_r=(Cr / denom) if Cr is not None else None,
                   Pref_a=(Ca / denom) if Ca is not None else None)


# =====================================================================
# ---- Point contact (ball) -- Sec 4.2/4.3, eq.(12)-(31) ----------------
# =====================================================================

def ball_Q_j(bearing: Bearing, result: BallBearingResult):
    """Per-element contact force [N], Q_j = cp * delta_j^1.5.

    Single-row -> ndarray(Z,). Multi-row -> list[ndarray(Z,)], one per
    row, in bearing.rows order."""
    views = _row_views(bearing, result)
    out = [rb.cp * np.maximum(rr.delta_j, 0.0) ** 1.5 for rb, rr in views]
    return out[0] if len(out) == 1 else out


def ball_phi_j_global(bearing: Bearing, result: BallBearingResult):
    """Ball angular positions in the global frame [rad], wrapped to
    [0, 2*pi). Single-row -> ndarray(Z,); multi-row -> list thereof."""
    views = _row_views(bearing, result)
    out = [(rb.phi_j + result.phi_Fr) % (2.0 * np.pi) for rb, rr in views]
    return out[0] if len(out) == 1 else out


def ball_contact_distribution(bearing: Bearing, result: BallBearingResult,
                              frame: str = "global"):
    """(angle, contact force) pairs -- ndarray(Z, 2): [phi, Q_j] columns.
    Single-row -> one ndarray; multi-row -> list of ndarrays, one per row."""
    views = _row_views(bearing, result)
    out = []
    for rb, rr in views:
        if frame == "global":
            phi = (rb.phi_j + result.phi_Fr) % (2.0 * np.pi)
        elif frame == "local":
            phi = rb.phi_j
        else:
            raise ValueError(f"ball_contact_distribution: frame must be 'global' or 'local', got {frame!r}")
        Q = rb.cp * np.maximum(rr.delta_j, 0.0) ** 1.5
        out.append(np.column_stack((phi, Q)))
    return out[0] if len(out) == 1 else out


_BALL_P_ROTATING   = 3.0        # eq.(25)/(28) -- ring rotating relative to the load
_BALL_P_STATIONARY = 10.0 / 3.0  # eq.(26)/(27) -- ring stationary relative to the load


@dataclass(frozen=True)
class DynamicEquivalentRollingElementLoad:
    """Dynamic equivalent load per rolling element, inner/outer raceway --
    eq.(25)-(28). inner_rotating/outer_rotating are independent flags."""
    label          : str
    Q_ei           : float
    Q_ee           : float
    inner_rotating : bool
    outer_rotating : bool
    Q_j            : np.ndarray

    @classmethod
    def from_distribution(cls, bearing: Bearing, result: BallLoadDistributionResult,
                          inner_rotating: bool = True, outer_rotating: bool = False,
                          label: str = "") -> "DynamicEquivalentRollingElementLoad":
        Q = bearing.cp * np.maximum(result.delta_j, 0.0) ** 1.5
        _lbl = label or bearing.label
        return cls(label=_lbl,
                   Q_ei=_ball_equivalent(Q, inner_rotating),
                   Q_ee=_ball_equivalent(Q, outer_rotating),
                   inner_rotating=inner_rotating, outer_rotating=outer_rotating, Q_j=Q)

    @classmethod
    def from_bearing_result(cls, bearing: Bearing, result: BallBearingResult,
                            inner_rotating: bool = True, outer_rotating: bool = False,
                            label: str = "") -> list["DynamicEquivalentRollingElementLoad"]:
        lbl = label or getattr(bearing, "label", "")
        views = _row_views(bearing, result)
        n = len(views)
        return [
            cls.from_distribution(rb, rr, inner_rotating=inner_rotating,
                                  outer_rotating=outer_rotating,
                                  label=f"{lbl}-row{j}" if n > 1 else lbl)
            for j, (rb, rr) in enumerate(views)
        ]


def _ball_equivalent(Q: np.ndarray, rotating: bool) -> float:
    p = _BALL_P_ROTATING if rotating else _BALL_P_STATIONARY
    return float(np.mean(Q ** p) ** (1.0 / p))


_E_BALL = 10.0 / 9.0   # Weibull slope, point contact


@dataclass(frozen=True)
class BallBasicReferenceRatingLife:
    """L10r for one row/raceway pair -- eq.(29)."""
    label : str
    L10r  : float
    Q_ci  : float
    Q_ei  : float
    Q_ce  : float
    Q_ee  : float

    @classmethod
    def from_loads(cls, label: str, Q_ci: float, Q_ei: float,
                   Q_ce: float, Q_ee: float) -> "BallBasicReferenceRatingLife":
        if Q_ci <= 0.0 or Q_ei <= 0.0 or Q_ce <= 0.0 or Q_ee <= 0.0:
            raise ValueError(f"BallBasicReferenceRatingLife.from_loads({label!r}): all loads must be positive.")
        L10r = ((Q_ci / Q_ei) ** (-10.0/3.0) + (Q_ce / Q_ee) ** (-10.0/3.0)) ** (-9.0/10.0)
        return cls(label=label, L10r=L10r, Q_ci=Q_ci, Q_ei=Q_ei, Q_ce=Q_ce, Q_ee=Q_ee)


def ball_basic_reference_rating_life(
    label: str,
    Q_ci_rows: list[float], Q_ei_rows: list[float],
    Q_ce_rows: list[float], Q_ee_rows: list[float],
) -> tuple[list[BallBasicReferenceRatingLife], float]:
    """Per-row L10r (eq.29) + combined bearing L10r (combine_row_L10r() for n>=2).
    Already row-list-shaped in its own signature -- multi-row just means
    passing n>1 entries per *_rows list; no change needed here."""
    n = len(Q_ci_rows)
    if not (len(Q_ei_rows) == len(Q_ce_rows) == len(Q_ee_rows) == n):
        raise ValueError("ball_basic_reference_rating_life: row lists must all be the same length.")
    if n == 0:
        raise ValueError("ball_basic_reference_rating_life: got 0 rows.")

    per_row = [
        BallBasicReferenceRatingLife.from_loads(
            label=f"{label}-row{j}" if n > 1 else label,
            Q_ci=Q_ci_rows[j], Q_ei=Q_ei_rows[j], Q_ce=Q_ce_rows[j], Q_ee=Q_ee_rows[j],
        ) for j in range(n)
    ]
    L10r_bearing = per_row[0].L10r if n == 1 else combine_row_L10r([r.L10r for r in per_row], e=_E_BALL)
    return per_row, L10r_bearing


@dataclass(frozen=True)
class BallDynamicEquivalentReferenceLoad(DynamicEquivalentReferenceLoadBase):
    """Pref -- eq.(30)-(31): Pref_r = Cr/L10r^(1/3), Pref_a = Ca/L10r^(1/3)."""
    _PREF_EXPONENT: ClassVar[float] = 1.0 / 3.0


# =====================================================================
# ---- Line contact (roller) -- Sec 5.2/5.3, eq.(36)-(67) ----------------
# =====================================================================

def roller_Q_j(bearing: Bearing, result: RollerBearingResult):
    """Total contact force per roller [N], summed over its n_s laminae:
    Q_j = sum_k q_j,k. Single-row -> ndarray(Z,); multi-row -> list thereof."""
    views = _row_views(bearing, result)
    out = [np.sum(rr.q_jk, axis=1) for _, rr in views]
    return out[0] if len(out) == 1 else out


def roller_phi_j_global(bearing: Bearing, result: RollerBearingResult):
    """Roller angular positions in the global frame [rad], wrapped to
    [0, 2*pi). Single-row -> ndarray(Z,); multi-row -> list thereof."""
    views = _row_views(bearing, result)
    out = [(rb.phi_j + result.phi_Fr) % (2.0 * np.pi) for rb, rr in views]
    return out[0] if len(out) == 1 else out


def roller_contact_distribution(bearing: Bearing, result: RollerBearingResult,
                                frame: str = "global"):
    """(angle, total contact force) pairs -- ndarray(Z, 2): [phi, Q_j]
    columns. Single-row -> one ndarray; multi-row -> list of ndarrays."""
    views = _row_views(bearing, result)
    out = []
    for rb, rr in views:
        if frame == "global":
            phi = (rb.phi_j + result.phi_Fr) % (2.0 * np.pi)
        elif frame == "local":
            phi = rb.phi_j
        else:
            raise ValueError(f"roller_contact_distribution: frame must be 'global' or 'local', got {frame!r}")
        Q = np.sum(rr.q_jk, axis=1)
        out.append(np.column_stack((phi, Q)))
    return out[0] if len(out) == 1 else out


def roller_lamina_distribution(bearing: Bearing, result: RollerLoadDistributionResult,
                               j: int) -> np.ndarray:
    """Per-lamina (position, load) pairs for a single roller j, shape
    (n_s, 2): [x_k, q_j,k]. `result` is ONE row's RollerLoadDistributionResult
    (e.g. bearing_result.rows[0], or bearing_result.rows[i] for row i of a
    multi-row bearing) -- deliberately NOT row-aware itself, unlike the
    functions above: it already operates at the single-row level by
    design, the caller picks the row."""
    Z = result.q_jk.shape[0]
    if not (0 <= j < Z):
        raise IndexError(f"roller_lamina_distribution: j={j} out of range for Z={Z} rollers.")
    return np.column_stack((result.x_k, result.q_jk[j, :]))


_STRESS_RISER_LOG_ARG_EPS = 1e-12   # floor for ln() argument at the exact bearing mid-plane


def stress_riser_factor(n_s: int) -> np.ndarray:
    """Approximate stress-riser f_i[k] = f_e[k] = f[k], k=1..n_s -- eq.(60).
    Valid for a profile from roller_profile() (eq.42-44), medium load, and
    total misalignment < 4' -- not checked here."""
    k = np.arange(1, n_s + 1)
    ratio = 1.985 * np.abs(2 * k - n_s - 1) / (2.0 * n_s - 2.0)
    ratio = np.maximum(ratio, _STRESS_RISER_LOG_ARG_EPS)
    return 1.0 - 0.01 / np.log(ratio)


_ROLLER_P_ROTATING   = 4.0    # eq.(61)/(64) -- raceway rotating relative to the load
_ROLLER_P_STATIONARY = 4.5    # eq.(62)/(63) -- raceway stationary relative to the load


@dataclass(frozen=True)
class LaminaDynamicEquivalentLoad:
    """Dynamic equivalent load PER LAMINA k, inner/outer raceway -- eq.(61)-(64),
    using stress_riser_factor()'s eq.(60) approximation. One value per
    lamina (n_s,), not per bearing."""
    label          : str
    q_kei          : np.ndarray
    q_kee          : np.ndarray
    inner_rotating : bool
    outer_rotating : bool
    f_k            : np.ndarray

    @classmethod
    def from_distribution(cls, bearing: Bearing, result: RollerLoadDistributionResult,
                          inner_rotating: bool = True, outer_rotating: bool = False,
                          label: str = "") -> "LaminaDynamicEquivalentLoad":
        q_jk = result.q_jk
        Z, n_s = q_jk.shape
        f_k = stress_riser_factor(n_s)
        _lbl = label or bearing.label
        return cls(label=_lbl,
                   q_kei=f_k * _lamina_mean(q_jk, inner_rotating),
                   q_kee=f_k * _lamina_mean(q_jk, outer_rotating),
                   inner_rotating=inner_rotating, outer_rotating=outer_rotating, f_k=f_k)

    @classmethod
    def from_bearing_result(cls, bearing: Bearing, result: RollerBearingResult,
                            inner_rotating: bool = True, outer_rotating: bool = False,
                            label: str = "") -> list["LaminaDynamicEquivalentLoad"]:
        lbl = label or getattr(bearing, "label", "")
        views = _row_views(bearing, result)
        n = len(views)
        return [
            cls.from_distribution(rb, rr, inner_rotating=inner_rotating,
                                  outer_rotating=outer_rotating,
                                  label=f"{lbl}-row{j}" if n > 1 else lbl)
            for j, (rb, rr) in enumerate(views)
        ]


def _lamina_mean(q_jk: np.ndarray, rotating: bool) -> np.ndarray:
    p = _ROLLER_P_ROTATING if rotating else _ROLLER_P_STATIONARY
    Z = q_jk.shape[0]
    return (np.sum(q_jk ** p, axis=0) / Z) ** (1.0 / p)


_E_ROLLER = 9.0 / 8.0   # Weibull slope, line contact


@dataclass(frozen=True)
class RollerBasicReferenceRatingLife:
    """L10r for one row/raceway pair -- eq.(65), summed over n_s laminae."""
    label : str
    L10r  : float
    q_kci : np.ndarray
    q_kei : np.ndarray
    q_kce : np.ndarray
    q_kee : np.ndarray

    @classmethod
    def from_loads(cls, label: str, q_kci: np.ndarray, q_kei: np.ndarray,
                   q_kce: np.ndarray, q_kee: np.ndarray) -> "RollerBasicReferenceRatingLife":
        if np.any(q_kci <= 0.0) or np.any(q_kce <= 0.0):
            raise ValueError(f"RollerBasicReferenceRatingLife.from_loads({label!r}): q_kci/q_kce must be positive.")
        if np.any(q_kei < 0.0) or np.any(q_kee < 0.0):
            raise ValueError(f"RollerBasicReferenceRatingLife.from_loads({label!r}): q_kei/q_kee must be non-negative.")
        terms = (q_kei / q_kci) ** 4.5 + (q_kee / q_kce) ** 4.5
        L10r = float(np.sum(terms) ** (-8.0/9.0))
        return cls(label=label, L10r=L10r, q_kci=q_kci, q_kei=q_kei, q_kce=q_kce, q_kee=q_kee)


def roller_basic_reference_rating_life(
    label: str,
    q_kci_rows: list[np.ndarray], q_kei_rows: list[np.ndarray],
    q_kce_rows: list[np.ndarray], q_kee_rows: list[np.ndarray],
) -> tuple[list[RollerBasicReferenceRatingLife], float]:
    n = len(q_kci_rows)
    if not (len(q_kei_rows) == len(q_kce_rows) == len(q_kee_rows) == n):
        raise ValueError("roller_basic_reference_rating_life: row lists must all be the same length.")
    if n == 0:
        raise ValueError("roller_basic_reference_rating_life: got 0 rows.")

    per_row = [
        RollerBasicReferenceRatingLife.from_loads(
            label=f"{label}-row{j}" if n > 1 else label,
            q_kci=q_kci_rows[j], q_kei=q_kei_rows[j], q_kce=q_kce_rows[j], q_kee=q_kee_rows[j],
        ) for j in range(n)
    ]
    L10r_bearing = per_row[0].L10r if n == 1 else combine_row_L10r([r.L10r for r in per_row], e=_E_ROLLER)
    return per_row, L10r_bearing


@dataclass(frozen=True)
class RollerDynamicEquivalentReferenceLoad(DynamicEquivalentReferenceLoadBase):
    """Pref -- eq.(66)-(67): Pref_r = Cr/L10r^(3/10), Pref_a = Ca/L10r^(3/10)."""
    _PREF_EXPONENT: ClassVar[float] = 3.0 / 10.0