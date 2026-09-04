"""axisforge/solvers/machine_elements/bearings/load_distribution/single_row/iso_16281/roller_bearing/postprocessing.py

Post-processing for LINE-CONTACT roller bearings -- everything that
CONSUMES an already-computed RollerBearingResult (or RollerBearingStiffness)
to derive a further quantity, rather than producing one itself. Mirrors
ball_bearing/postprocessing.py. Imports roller_bearing.results only; never
imported by solver.py (one-directional).

Single-row only (RollerBearingResult.rows has length 1, see
contracts/bearing/roller_bearing_results.py) -- CylindricalRollerFamily's
own `i` is a capacity multiplier on one raceway, not a rows list. Q_j,
phi_j_global, contact_distribution and LaminaDynamicEquivalentLoad.
from_bearing_result() return a length-1 list, mirroring the ball side.

lamina_distribution() takes a raw RollerLoadDistributionResult directly
(one row), not the wrapped RollerBearingResult -- it already indexes a
roller j; the caller picks the row first (result.rows[0]) and passes it in.

RollerBearingStiffness is defined here, not shared with the ball side --
identical projection formula (delta_r -> Kr_xz/Kr_xy/Ka), duplicated the
same way RollerLoadDistributionResult / BallLoadDistributionResult are each
self-contained. Fa is always 0.0 -- CylindricalRollerFamily (radial)
carries no axial load.

Dynamic equivalent load -- per lamina, not per roller
------------------------------------------------------
ISO/TS 16281 Sec 5.3.4, eq.(61)-(64) defines the dynamic equivalent load
PER LAMINA k, built from q_j,k directly, weighted by the edge-stress
concentration factors f_i[j,k]/f_e[j,k] of eq.(58)-(59). Those need the
actual local Hertzian contact pressure, which this package does not
compute -- eq.(60) gives an explicit first-order approximation instead:

    f_i[k] = f_e[k] = 1 - 0.01 / ln(1.985 * |2k - n_s - 1| / (2*n_s - 2))

implemented as stress_riser_factor() below. Valid "for an approximated
profile obtained with the aid of Equations (42), (43) and (44)" (roller_
profile()) "and providing the conditions of medium load and a total
misalignment of the bearing of less than 4' [~1.16e-3 rad] are fulfilled" --
not checked here; verify the regime before trusting the approximation.

Basic reference rating life / Pref -- Sec 5.3.5-.6, eq.(65)-(67) -- same
structure as the ball side: BasicReferenceRatingLife.from_loads() is the
single-row primitive (eq.65, sums q_kci/q_kei/q_kce/q_kee over all n_s
laminae), combine_row_L10r() combines rows (Zaretsky eq.49a, e=9/8 line
contact). q_kci/q_kce (per-lamina capacity) are not computed here -- see
RollerElementCapacity.radial() (eq.56-57).

References
----------
ISO/TS 16281:2008 Sec 5.2, eq.(36)-(46); Sec 5.3.2-.6, eq.(56)-(67)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.results.bearings.load_distribution.single_row.roller_bearing_results import (
    RollerLoadDistributionResult,
    RollerBearingResult,
)


def _require_roller_result(result, fn_name: str) -> None:
    if not isinstance(result, RollerLoadDistributionResult):
        raise TypeError(
            f"{fn_name}: expected a RollerLoadDistributionResult (one row -- "
            f"with q_jk/x_k), got {type(result).__name__}. Pass result.rows[0], "
            f"not the RollerBearingResult itself."
        )


# ---------------------------------------------------------------------------
# Per-element (per-roller) contact force / global angular position --
# length-1 list.
# ---------------------------------------------------------------------------

def Q_j(bearing: Bearing, result: RollerBearingResult) -> list[np.ndarray]:
    """Total contact force per roller [N], summed over its n_s laminae:
    Q_j = sum_k q_j,k -- the line-contact analogue of eq.(45)'s inner sum,
    per roller (before the cos(phi_j) projection eq.(45) applies to get Fr).
    """
    row = result.rows[0]
    _require_roller_result(row, "Q_j")
    return [np.sum(row.q_jk, axis=1)]


def phi_j_global(bearing: Bearing, result: RollerBearingResult) -> list[np.ndarray]:
    """Roller angular positions in the global frame [rad], wrapped to [0, 2*pi)."""
    row = result.rows[0]
    return [(bearing.phi_j + row.phi_Fr) % (2.0 * np.pi)]


def contact_distribution(bearing: Bearing, result: RollerBearingResult,
                         frame: str = "global") -> list[np.ndarray]:
    """(angle, total contact force) pairs -- ndarray(Z, 2): [phi, Q_j] columns.

    frame="global" (default): phi = phi_j_global(...). frame="local": phi
    as stored.
    """
    if frame == "global":
        phi = phi_j_global(bearing, result)[0]
    elif frame == "local":
        phi = bearing.phi_j
    else:
        raise ValueError(f"contact_distribution: frame must be 'global' or 'local', got {frame!r}")

    Q = Q_j(bearing, result)[0]
    return [np.column_stack((phi, Q))]


def lamina_distribution(bearing: Bearing, result: RollerLoadDistributionResult,
                        j: int) -> np.ndarray:
    """Per-lamina (position, load) pairs for a single roller j, shape (n_s, 2):
    [x_k, q_j,k] -- the pressure-profile-along-the-roller-length view the
    Sec 5.2.3 profile function (roller_profile()) exists to keep well-behaved
    at the ends. `result` is ONE row's RollerLoadDistributionResult (e.g.
    bearing_result.rows[0]), not the wrapped RollerBearingResult.

    j : index into bearing.phi_j / result.q_jk's first axis, NOT an angle.
    """
    _require_roller_result(result, "lamina_distribution")
    Z = result.q_jk.shape[0]
    if not (0 <= j < Z):
        raise IndexError(f"lamina_distribution: j={j} out of range for Z={Z} rollers.")
    return np.column_stack((result.x_k, result.q_jk[j, :]))


# ---------------------------------------------------------------------------
# RollerBearingStiffness -- a property of the whole bearing.
# ---------------------------------------------------------------------------

class RollerBearingStiffness:
    """Secant stiffness of a line-contact (radial cylindrical roller)
    bearing, XZ / XY / axial axes.

    Built from a converged RollerLoadDistributionResult by projecting
    delta_r back onto the global axes:

        delta_r_xz = delta_r * cos(phi_Fr)
        delta_r_xy = delta_r * sin(phi_Fr)
        Kr_xz = Fr_xz / delta_r_xz
        Kr_xy = Fr_xy / delta_r_xy
        Ka    = Fa    / delta_a

    Any stiffness component is set to float('inf') when the corresponding
    displacement is negligible.

    Ka_regime: "no_load" (always here -- radial roller bearings carry no
    axial load), "engaged", "closing_clearance".
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
    """Secant stiffness (Kr_xz, Kr_xy) of the whole bearing. Fa is always
    0.0 -- CylindricalRollerFamily carries no axial load.
    """
    return RollerBearingStiffness.from_result(
        label=bearing.label, result=result.rows[0],
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=0.0, eps=eps,
    )


# ---------------------------------------------------------------------------
# Stress-riser (edge-stress concentration) -- ISO/TS 16281 Sec 5.3.3, eq.(60)
# ---------------------------------------------------------------------------

_STRESS_RISER_LOG_ARG_EPS = 1e-12   # floor for ln() argument at the exact bearing mid-plane


def stress_riser_factor(n_s: int) -> np.ndarray:
    """Approximate stress-riser function f_i[k] = f_e[k] = f[k], k=1..n_s --
    eq.(60), a first-order approximation of the Hertz-pressure-based
    f_i[j,k]/f_e[j,k] of eq.(58)-(59) needing only n_s. See module docstring
    for validity conditions (not checked here).

        f[k] = 1 - 0.01 / ln(1.985 * |2k - n_s - 1| / (2*n_s - 2))

    At the roller mid-plane (n_s odd, k=(n_s+1)/2) the ln() argument is
    exactly 0; floored to _STRESS_RISER_LOG_ARG_EPS rather than raising,
    since f[k] -> 1 in that limit anyway.

    Returns
    -------
    ndarray(n_s,) -- f[k] for k=1..n_s, index 0 = k=1.
    """
    k = np.arange(1, n_s + 1)
    ratio = 1.985 * np.abs(2 * k - n_s - 1) / (2.0 * n_s - 2.0)
    ratio = np.maximum(ratio, _STRESS_RISER_LOG_ARG_EPS)
    return 1.0 - 0.01 / np.log(ratio)


# ---------------------------------------------------------------------------
# LaminaDynamicEquivalentLoad -- ISO/TS 16281 Sec 5.3.4, eq.(61)-(64)
# ---------------------------------------------------------------------------

_P_ROTATING   = 4.0    # eq.(61)/(64) -- raceway rotating relative to the load
_P_STATIONARY = 4.5    # eq.(62)/(63) -- raceway stationary relative to the load


@dataclass(frozen=True)
class LaminaDynamicEquivalentLoad:
    """Dynamic equivalent load PER LAMINA k, inner/outer raceway -- eq.(61)-(64),
    using stress_riser_factor()'s eq.(60) approximation for f_i[j,k]/f_e[j,k].
    q_kei and q_kee differ from each other only via inner_rotating/
    outer_rotating (f_i[k] = f_e[k] under this approximation).

        q_kei[k] = f[k] * ( (1/Z) * sum_j q_j,k^p )^(1/p)
        q_kee[k] = f[k] * ( (1/Z) * sum_j q_j,k^p )^(1/p)
        p = 4 if that raceway rotates relative to the load, else 4.5

    One value PER LAMINA (n_s,) -- NOT one per bearing. Compare against
    RollerElementCapacity.radial()'s q_ci/q_ce (eq.56-57, also per lamina).

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
        """Single-row primitive -- does all the eq.(61)-(64) math."""
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
        """Always returns a length-1 list -- calls from_distribution() once."""
        lbl = label or getattr(bearing, "label", "")
        return [cls.from_distribution(
            bearing, result.rows[0],
            inner_rotating=inner_rotating, outer_rotating=outer_rotating,
            label=lbl,
        )]


def _lamina_mean(q_jk: np.ndarray, rotating: bool) -> np.ndarray:
    """( (1/Z) * sum_j q_j,k^p )^(1/p) per lamina k -- the core mean inside
    eq.(61)/(63), before the f_i/f_e[j,k] weighting.
    """
    p = _P_ROTATING if rotating else _P_STATIONARY
    Z = q_jk.shape[0]
    return (np.sum(q_jk ** p, axis=0) / Z) ** (1.0 / p)


# ---------------------------------------------------------------------------
# BasicReferenceRatingLife / DynamicEquivalentReferenceLoad -- eq.(65)-(67),
# ISO/TS 16281 Sec 5.3.5 (L10r), Sec 5.3.6 (Pref)
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
        # unloaded lamina contributes 0 instead of dividing by zero.
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