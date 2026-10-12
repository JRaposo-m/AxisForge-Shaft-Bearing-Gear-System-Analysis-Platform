"""Stiffness and reference rating life from an ISO/TS 16281 load distribution.

From a solved bearing result this module derives the secant stiffness of
the bearing, the dynamic equivalent loads of the rolling elements (ball) or
laminae (roller), the basic reference rating life L10r of ISO/TS 16281, 4.3
and 5.3, and the dynamic equivalent reference load Pref. The functions
accept single-row and multi-row results; row lives are combined with a
strict-series rule that is not part of ISO/TS 16281. The modified reference
rating life L_nmr (Eq. (32)-(33) and (68)-(71)) is not implemented.
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
    """Secant stiffness of a bearing in the two shaft planes and axially.

    Ratio of load to displacement at the solved operating point, for point and
    line contact alike::

        delta_r_xz = delta_r*cos(phi_Fr);   Kr_xz = Fr_xz / delta_r_xz
        delta_r_xy = delta_r*sin(phi_Fr);   Kr_xy = Fr_xy / delta_r_xy
        Ka = Fa / delta_a

    Parameters
    ----------
    label : str
        Bearing label.
    delta_r_xz : float
        Component of the radial displacement delta_r in the XZ plane [mm].
    Kr_xz : float
        Radial secant stiffness in the XZ plane [N/mm].
    delta_r_xy : float
        Component of the radial displacement delta_r in the XY plane [mm].
    Kr_xy : float
        Radial secant stiffness in the XY plane [N/mm].
    Ka : float, optional
        Axial secant stiffness [N/mm].
    Ka_regime : str, optional
        "no_load" (Fa = 0), "engaged" (delta_a >= 0) or "closing_clearance"
        (delta_a < 0).

    Warnings
    --------
    Known limitation (to review): a stiffness is float("inf") when its
    displacement is negligible, e.g. phi_Fr near 90 deg gives Kr_xz = inf.
    inf is a placeholder, not a stiffness: when phi_Fr is near 90 deg both
    Fr_xz and delta_r_xz vanish and the ratio is undetermined (the other
    plane gives Fr/delta_r).

    Secant, not tangent: delta_r contains the take-up of the radial clearance,
    so F/delta is not dF/d(delta).

    Ka = inf whenever Fa = 0 (regime "no_load"), whatever delta_a. With
    Fa != 0 and a negligible delta_a, Ka = inf and the regime is "engaged".

    delta_a is measured from the reference position of ISO_TS_16281_2008
    Eq. (12) (contact angle alpha_0), not from the axially centred ring; Ka is
    therefore not a stiffness about the centred position and can be negative.

    Notes
    -----
    When phi_Fr = arctan2(Fr_xy, Fr_xz), as set by ``SolverBase.solve``, Kr_xz
    and Kr_xy are both equal to Fr/delta_r wherever both projections are
    non-zero.

    References
    ----------
    .. [1] Hernot2000, Eq. (12)-(13), (16): secant stiffness matrix
       [K]{delta} = {F} of an angular-contact ball bearing (constant contact
       angle, no clearance), tangent matrix n*[K]. The per-axis scalar ratio
       used here, without the coupling term, is not printed in any library
       source; not in ISO/TS 16281.
    """

    __slots__ = ("label", "delta_r_xz", "Kr_xz", "delta_r_xy", "Kr_xy", "Ka", "Ka_regime")

    def __init__(self, label, delta_r_xz, Kr_xz, delta_r_xy, Kr_xy, Ka=None, Ka_regime=None):
        """Store the stiffness values (see the class docstring)."""
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
        """Secant stiffness from a solved bearing result and the applied loads.

        Parameters
        ----------
        label : str
            Bearing label.
        result
            Bearing-level result with delta_r [mm], delta_a [mm] and phi_Fr [rad].
        Fr_xz, Fr_xy : float
            Radial load components in the two shaft planes [N].
        Fa : float
            Axial load [N].
        eps : float, optional
            Displacement magnitude [mm] at or below which the stiffness is set to
            float("inf"). Default 1e-9.

        Returns
        -------
        ContactBearingStiffness
            Kr_xz, Kr_xy, Ka [N/mm] and the axial regime; equations and limitations
            in the class docstring.
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

        return cls(label=label, delta_r_xz=delta_r_xz, Kr_xz=Kr_xz,
                   delta_r_xy=delta_r_xy, Kr_xy=Kr_xy, Ka=Ka, Ka_regime=regime)


def bearing_stiffness(bearing: Bearing, result, Fr_xz: float, Fr_xy: float,
                      Fa: float = 0.0, eps: float = 1e-9) -> ContactBearingStiffness:
    """Secant stiffness (Kr_xz, Kr_xy, Ka) of a whole bearing, 1 or N rows.

    Parameters
    ----------
    bearing : Bearing
        Only its label is used.
    result
        Bearing-level result with delta_r [mm], delta_a [mm] and phi_Fr [rad].
    Fr_xz, Fr_xy : float
        Radial load components in the two shaft planes [N].
    Fa : float, optional
        Axial load [N]. Default 0.0 (radial roller bearings).
    eps : float, optional
        Displacement threshold [mm]. Default 1e-9.

    Returns
    -------
    ContactBearingStiffness
        See ``ContactBearingStiffness`` for equations and limitations.
    """
    return ContactBearingStiffness.from_result(
        label=bearing.label, result=result,
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa, eps=eps,
    )


def combine_row_L10r(L10r_rows: list[float], e: float) -> float:
    """Rating life of a multi-row bearing from the lives of its rows.

    Parameters
    ----------
    L10r_rows : list[float]
        Life of each row, all positive, at least two rows. In this module
        L10r is in [10^6 rev].
    e : float
        Weibull slope [-], the same for all rows. No default.

    Returns
    -------
    float
        (sum_i L_i^(-e))^(-1/e), in the unit of the inputs.

    Raises
    ------
    ValueError
        With fewer than two rows, or a life that is not positive.

    Warnings
    --------
    Not part of ISO/TS 16281: the standard prints no equation that combines
    row lives.

    Notes
    -----
    Strict series reliability, 1/L^e = sum_i 1/L_i^e. The source prints
    Eq. (49a) for rows of balls "assuming all rows carry equal load"; the form
    for unequal lives is Eq. (C11) (Appendix C), derived for a Weibull slope
    common to all rows. Same product-law algebra as Harris2007_vol1,
    Eq. 11.79, printed there for the two raceways of one row.

    References
    ----------
    .. [1] Zaretsky2016, Eq. (49a); Appendix C, Eq. (C11).
    .. [2] Harris2007_vol1, 11.4, Eq. 11.79.
    """
    if len(L10r_rows) < 2:
        raise ValueError(f"combine_row_L10r needs >= 2 rows; got {len(L10r_rows)}.")
    if any(L <= 0.0 for L in L10r_rows):
        raise ValueError(f"All L10r_rows must be positive; got {list(L10r_rows)}.")
    return sum(L ** (-e) for L in L10r_rows) ** (-1.0 / e)


def _row_views(bearing, result):
    """Pair each bearing row with its result row.

    Returns
    -------
    list[tuple]
        (bearing_row, result_row) pairs. A bearing without a ``rows``
        attribute gives one pair, (bearing, result.rows[0]). A row given as a
        dict is wrapped in a SimpleNamespace, so that ``row.cp``,
        ``row.phi_j`` and the other attributes can be read either way.
    """
    if not hasattr(bearing, "rows"):
        return [(bearing, result.rows[0])]
    row_bearings = [SimpleNamespace(**row) if isinstance(row, dict) else row for row in bearing.rows]
    return list(zip(row_bearings, result.rows))


@dataclass(frozen=True)
class DynamicEquivalentReferenceLoadBase:
    """Dynamic equivalent reference load of a bearing, Pref = C / L10r^w.

    Constant bearing load that gives the basic reference rating life L10r with
    the basic dynamic load rating C. The exponent w (``_PREF_EXPONENT``) is set
    by the subclass: 1/3 for ball bearings, 3/10 for roller bearings. The base
    class itself cannot be evaluated (``from_L10r`` raises).

    Parameters
    ----------
    label : str
        Bearing label.
    Pref_r : float or None
        Dynamic equivalent reference radial load [N]; None without Cr.
    Pref_a : float or None
        Dynamic equivalent reference axial load [N]; None without Ca.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.3.4, Eq. (30)-(31); 5.3.6, Eq. (66)-(67).
    """
    label  : str
    Pref_r : float | None
    Pref_a : float | None

    _PREF_EXPONENT: ClassVar[float] = 0.0

    @classmethod
    def from_L10r(cls, label: str, L10r: float,
                  Cr: float | None = None, Ca: float | None = None
                  ) -> "DynamicEquivalentReferenceLoadBase":
        """Dynamic equivalent reference load from L10r and the load rating.

        Parameters
        ----------
        label : str
            Bearing label.
        L10r : float
            Basic reference rating life [10^6 rev], positive.
        Cr : float, optional
            Basic dynamic radial load rating [N].
        Ca : float, optional
            Basic dynamic axial load rating [N].

        Returns
        -------
        DynamicEquivalentReferenceLoadBase
            Instance of the calling subclass with Pref_r = Cr / L10r^w and
            Pref_a = Ca / L10r^w [N], each None when its rating is not given.

        Raises
        ------
        NotImplementedError
            If the class does not set a positive ``_PREF_EXPONENT``.
        ValueError
            If neither Cr nor Ca is given, or L10r is not positive.
        """
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
    """Contact load of every ball, Q_j = cp * delta_j^1.5.

    Parameters
    ----------
    bearing : Bearing
        Needs cp [N/mm^1.5] (per row for a multi-row bearing).
    result : BallBearingResult
        Solved bearing-level result.

    Returns
    -------
    np.ndarray or list[np.ndarray]
        Q_j [N], shape (Z,); for a multi-row bearing a list with one array per
        row, in the order of ``bearing.rows``.

    Notes
    -----
    Recomputed from the stored delta_j (floored at 0), not read from the Q_j
    that the solver stored in the result row.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.2.1, Eq. (10).
    """
    views = _row_views(bearing, result)
    out = [rb.cp * np.maximum(rr.delta_j, 0.0) ** 1.5 for rb, rr in views]
    return out[0] if len(out) == 1 else out


def ball_phi_j_global(bearing: Bearing, result: BallBearingResult):
    """Angular position of every ball in the global frame.

    Parameters
    ----------
    bearing : Bearing
        Needs phi_j [rad], the local angular positions (zero on the line of
        the resultant radial load).
    result : BallBearingResult
        Solved bearing-level result; gives phi_Fr [rad].

    Returns
    -------
    np.ndarray or list[np.ndarray]
        (phi_j + phi_Fr) mod 2*pi [rad], in [0, 2*pi), shape (Z,); one array
        per row for a multi-row bearing.

    Notes
    -----
    phi_Fr = arctan2(Fr_xy, Fr_xz) is measured from +Z toward +Y (see
    ``SolverBase.solve``).
    """
    views = _row_views(bearing, result)
    out = [(rb.phi_j + result.phi_Fr) % (2.0 * np.pi) for rb, rr in views]
    return out[0] if len(out) == 1 else out


def ball_contact_distribution(bearing: Bearing, result: BallBearingResult,
                              frame: str = "global"):
    """Angular position and contact load of every ball, as a two-column table.

    Parameters
    ----------
    bearing : Bearing
        Needs phi_j [rad] and cp [N/mm^1.5] (per row for a multi-row bearing).
    result : BallBearingResult
        Solved bearing-level result.
    frame : str, optional
        "global" (default): angle (phi_j + phi_Fr) mod 2*pi; "local": phi_j,
        zero on the line of the resultant radial load.

    Returns
    -------
    np.ndarray or list[np.ndarray]
        Shape (Z, 2): column 0 the angle [rad], column 1 the load
        Q_j = cp * delta_j^1.5 [N]; one array per row for a multi-row bearing.

    Raises
    ------
    ValueError
        If ``frame`` is neither "global" nor "local".
    """
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
    """Dynamic equivalent rolling-element load of a ball-bearing row.

    One equivalent load for the inner raceway and one for the outer raceway,
    from the loads of all balls of the row.

    Parameters
    ----------
    label : str
        Bearing or row label.
    Q_ei : float
        Dynamic equivalent rolling-element load on the inner ring [N].
    Q_ee : float
        Dynamic equivalent rolling-element load on the outer ring [N].
    inner_rotating : bool
        True if the inner ring rotates relative to the load.
    outer_rotating : bool
        True if the outer ring rotates relative to the load.
    Q_j : np.ndarray
        Contact load of every ball [N], shape (Z,).

    Notes
    -----
    Equation as coded, for each ring::

        Q_e = ((1/Z) * sum_j(Q_j^p))^(1/p)

    with p = 3 for a ring that rotates relative to the load (Eq. (25), (28))
    and p = 10/3 for a ring that is stationary relative to the load (Eq. (26),
    (27)). The mean runs over all Z balls, the unloaded ones with Q_j = 0. The
    two flags are independent of each other.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.3.2, Eq. (25)-(28).
    """
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
        """Equivalent rolling-element loads of one ball row.

        Parameters
        ----------
        bearing : Bearing
            Bearing or row with cp [N/mm^1.5] and a label.
        result : BallLoadDistributionResult
            Result of one row; gives delta_j [mm].
        inner_rotating, outer_rotating : bool, optional
            Ring rotation flags of the class. Defaults True, False.
        label : str, optional
            Label of the output; the bearing label if empty.

        Returns
        -------
        DynamicEquivalentRollingElementLoad
            Q_ei, Q_ee [N] and Q_j [N], recomputed as cp * max(delta_j, 0)^1.5.
        """
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
        """Equivalent rolling-element loads of every row of a ball bearing.

        Parameters
        ----------
        bearing : Bearing
            Single-row or multi-row ball bearing.
        result : BallBearingResult
            Solved bearing-level result.
        inner_rotating, outer_rotating : bool, optional
            Ring rotation flags of the class. Defaults True, False.
        label : str, optional
            Base label; the bearing label if empty.

        Returns
        -------
        list[DynamicEquivalentRollingElementLoad]
            One entry per row; labels get the suffix "-row<j>" with several rows.
        """
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
    """Power mean of the ball loads of one row.

    Parameters
    ----------
    Q : np.ndarray
        Contact load of every ball [N], shape (Z,).
    rotating : bool
        True for a ring rotating relative to the load (p = 3), False for a
        stationary ring (p = 10/3).

    Returns
    -------
    float
        ((1/Z) * sum_j(Q_j^p))^(1/p) [N] (ISO_TS_16281_2008, 4.3.2,
        Eq. (25)-(28)).
    """
    p = _BALL_P_ROTATING if rotating else _BALL_P_STATIONARY
    return float(np.mean(Q ** p) ** (1.0 / p))


_E_BALL = 10.0 / 9.0   # Weibull slope, point contact: Harris2007_vol1, 11.4 (PDF p. 226); ISO_TR_1281 Part 1, 6.2


@dataclass(frozen=True)
class BallBasicReferenceRatingLife:
    """Basic reference rating life of one ball-bearing row.

    Parameters
    ----------
    label : str
        Bearing or row label.
    L10r : float
        Basic reference rating life [10^6 rev].
    Q_ci : float
        Rolling-element load for the basic dynamic load rating of the inner
        ring [N].
    Q_ei : float
        Dynamic equivalent rolling-element load on the inner ring [N].
    Q_ce : float
        Rolling-element load for the basic dynamic load rating of the outer
        ring [N].
    Q_ee : float
        Dynamic equivalent rolling-element load on the outer ring [N].

    Notes
    -----
    Equation as coded::

        L10r = ((Q_ci/Q_ei)^(-10/3) + (Q_ce/Q_ee)^(-10/3))^(-9/10)   Eq. (29)

    The modified reference rating life (Eq. (32)-(33)) is not implemented.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.3.3, Eq. (29).
    """
    label : str
    L10r  : float
    Q_ci  : float
    Q_ei  : float
    Q_ce  : float
    Q_ee  : float

    @classmethod
    def from_loads(cls, label: str, Q_ci: float, Q_ei: float,
                   Q_ce: float, Q_ee: float) -> "BallBasicReferenceRatingLife":
        """Basic reference rating life of a ball row from its four loads.

        Parameters
        ----------
        label : str
            Bearing or row label.
        Q_ci, Q_ei, Q_ce, Q_ee : float
            The four loads of the class [N].

        Returns
        -------
        BallBasicReferenceRatingLife
            L10r [10^6 rev] by Eq. (29), with the four loads.

        Raises
        ------
        ValueError
            If any of the four loads is not positive. A row without any loaded
            ball (Q_ei = Q_ee = 0) therefore raises instead of giving an infinite
            life.
        """
        if Q_ci <= 0.0 or Q_ei <= 0.0 or Q_ce <= 0.0 or Q_ee <= 0.0:
            raise ValueError(f"BallBasicReferenceRatingLife.from_loads({label!r}): all loads must be positive.")
        L10r = ((Q_ci / Q_ei) ** (-10.0/3.0) + (Q_ce / Q_ee) ** (-10.0/3.0)) ** (-9.0/10.0)
        return cls(label=label, L10r=L10r, Q_ci=Q_ci, Q_ei=Q_ei, Q_ce=Q_ce, Q_ee=Q_ee)


def ball_basic_reference_rating_life(
    label: str,
    Q_ci_rows: list[float], Q_ei_rows: list[float],
    Q_ce_rows: list[float], Q_ee_rows: list[float],
) -> tuple[list[BallBasicReferenceRatingLife], float]:
    """Basic reference rating life of each ball row and of the whole bearing.

    Parameters
    ----------
    label : str
        Bearing label; with more than one row the row labels get the suffix
        "-row<j>".
    Q_ci_rows, Q_ce_rows : list[float]
        Rolling-element loads for the basic dynamic load rating of the inner
        and outer ring [N], one value per row.
    Q_ei_rows, Q_ee_rows : list[float]
        Dynamic equivalent rolling-element loads on the inner and outer ring
        [N], one value per row.

    Returns
    -------
    per_row : list[BallBasicReferenceRatingLife]
        Life of each row, Eq. (29).
    L10r_bearing : float
        Life of the bearing [10^6 rev]: the row life for one row; for several
        rows ``combine_row_L10r`` with e = 10/9.

    Raises
    ------
    ValueError
        If the lists differ in length or are empty; from ``from_loads``.

    Warnings
    --------
    The combination of row lives is not part of ISO/TS 16281 (see
    ``combine_row_L10r``). e = 10/9 is the Lundberg-Palmgren Weibull slope
    for ball bearings [3]. The sources use it to combine the two raceways of a
    row (Harris2007_vol1, Eq. 11.80), not rows under unequal loads.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.3.3, Eq. (29).
    .. [2] Zaretsky2016, Eq. (49a).
    .. [3] Harris2007_vol1, 11.4 (PDF p. 226), Eq. 11.79-11.80; ISO_TR_1281
       (Part 1), 6.2.
    """
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
    """Dynamic equivalent reference load of a ball bearing.

    Pref_r = Cr / L10r^(1/3) and Pref_a = Ca / L10r^(1/3) [N], with L10r in
    [10^6 rev]. Fields and constructor as in
    ``DynamicEquivalentReferenceLoadBase``.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.3.4, Eq. (30)-(31).
    """
    _PREF_EXPONENT: ClassVar[float] = 1.0 / 3.0


# =====================================================================
# ---- Line contact (roller) -- Sec 5.2/5.3, eq.(36)-(67) ----------------
# =====================================================================

def roller_Q_j(bearing: Bearing, result: RollerBearingResult):
    """Total contact load of every roller, Q_j = sum_k(q_jk).

    Parameters
    ----------
    bearing : Bearing
        Used only to tell single-row from multi-row.
    result : RollerBearingResult
        Solved bearing-level result; each row gives q_jk [N].

    Returns
    -------
    np.ndarray or list[np.ndarray]
        Q_j [N], shape (Z,), the sum over the n_s laminae of each roller; one
        array per row for a multi-row bearing.
    """
    views = _row_views(bearing, result)
    out = [np.sum(rr.q_jk, axis=1) for _, rr in views]
    return out[0] if len(out) == 1 else out


def roller_phi_j_global(bearing: Bearing, result: RollerBearingResult):
    """Angular position of every roller in the global frame.

    Parameters
    ----------
    bearing : Bearing
        Needs the local angular positions phi_j [rad].
    result : RollerBearingResult
        Solved bearing-level result; gives phi_Fr [rad].

    Returns
    -------
    np.ndarray or list[np.ndarray]
        (phi_j + phi_Fr) mod 2*pi [rad], as ``ball_phi_j_global``.
    """
    views = _row_views(bearing, result)
    out = [(rb.phi_j + result.phi_Fr) % (2.0 * np.pi) for rb, rr in views]
    return out[0] if len(out) == 1 else out


def roller_contact_distribution(bearing: Bearing, result: RollerBearingResult,
                                frame: str = "global"):
    """Angular position and total load of every roller, as a two-column table.

    Parameters
    ----------
    bearing : Bearing
        Needs phi_j [rad] (per row for a multi-row bearing).
    result : RollerBearingResult
        Solved bearing-level result.
    frame : str, optional
        "global" (default) or "local", as in ``ball_contact_distribution``.

    Returns
    -------
    np.ndarray or list[np.ndarray]
        Shape (Z, 2): angle [rad] and roller load Q_j = sum_k(q_jk) [N]; one
        array per row for a multi-row bearing.

    Raises
    ------
    ValueError
        If ``frame`` is neither "global" nor "local".
    """
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
    """Lamina positions and lamina loads of one roller, as a two-column table.

    Parameters
    ----------
    bearing : Bearing
        Not used.
    result : RollerLoadDistributionResult
        Result of one row (the caller picks the row of a multi-row bearing).
    j : int
        Roller index, 0 <= j < Z.

    Returns
    -------
    np.ndarray
        Shape (n_s, 2): column 0 the lamina position x_k [mm], column 1 the
        lamina load q_jk [N] of roller j.

    Raises
    ------
    IndexError
        If ``j`` is out of range.
    """
    Z = result.q_jk.shape[0]
    if not (0 <= j < Z):
        raise IndexError(f"roller_lamina_distribution: j={j} out of range for Z={Z} rollers.")
    return np.column_stack((result.x_k, result.q_jk[j, :]))


_STRESS_RISER_LOG_ARG_EPS = 1e-12   # floor for the ln() argument at the middle lamina (odd n_s)


def stress_riser_factor(n_s: int) -> np.ndarray:
    """Stress-riser factor of each lamina for the edge stress of a roller.

    First approximation of the standard for the stress concentration toward
    the roller ends; the same value is used for the inner and the outer ring.

    Parameters
    ----------
    n_s : int
        Number of laminae per roller. n_s = 1 gives NaN (0/0, NumPy
        warning), no exception.

    Returns
    -------
    np.ndarray
        f[k] [-] for k = 1..n_s, shape (n_s,).

    Warnings
    --------
    The validity conditions of the standard are not checked here: roller
    profile of Eq. (42)-(44), medium load, total misalignment of the bearing
    below 4' (as printed).

    The argument of the logarithm is floored at 1e-12, which is not in the
    standard. For an odd n_s the argument of the middle lamina is zero and the
    floor gives f = 1 - 0.01/ln(1e-12), about 1.00036.

    Notes
    -----
    Equation as coded::

        f[k] = 1 - 0.01 / ln(1.985 * abs((2*k - n_s - 1) / (2*n_s - 2)))   Eq. (60)

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.3.3, Eq. (60).
    """
    k = np.arange(1, n_s + 1)
    ratio = 1.985 * np.abs(2 * k - n_s - 1) / (2.0 * n_s - 2.0)
    ratio = np.maximum(ratio, _STRESS_RISER_LOG_ARG_EPS)
    return 1.0 - 0.01 / np.log(ratio)


_ROLLER_P_ROTATING   = 4.0    # eq.(61)/(64) -- raceway rotating relative to the load
_ROLLER_P_STATIONARY = 4.5    # eq.(62)/(63) -- raceway stationary relative to the load


@dataclass(frozen=True)
class LaminaDynamicEquivalentLoad:
    """Dynamic equivalent load of each lamina of a roller-bearing row.

    One equivalent load per lamina k for the inner raceway and one for the
    outer raceway, from the loads of that lamina on all rollers of the row.

    Parameters
    ----------
    label : str
        Bearing or row label.
    q_kei : np.ndarray
        Dynamic equivalent load of each lamina on the inner ring [N], shape
        (n_s,).
    q_kee : np.ndarray
        Dynamic equivalent load of each lamina on the outer ring [N], shape
        (n_s,).
    inner_rotating : bool
        True if the inner ring rotates relative to the load.
    outer_rotating : bool
        True if the outer ring rotates relative to the load.
    f_k : np.ndarray
        Stress-riser factor of each lamina [-], shape (n_s,).

    Notes
    -----
    Equation as coded, for each ring::

        q_ke[k] = f[k] * ((1/Z) * sum_j(q_jk^p))^(1/p)

    with p = 4 for a ring that rotates relative to the load (Eq. (61), (64))
    and p = 4.5 for a ring that is stationary relative to the load (Eq. (62),
    (63)). The mean runs over all Z rollers.

    The standard writes the factor f[j,k] inside the powered term. The code
    applies f[k] outside, which gives the same value only because the factor
    of Eq. (60) does not depend on the roller j, and uses the same f[k] for
    both rings. A stress-based factor f[j,k] (Eq. (58)-(59)) could not be
    taken outside the sum; it is not implemented.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.3.3, Eq. (58)-(60); 5.3.4, Eq. (61)-(64).
    """
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
        """Equivalent lamina loads of one roller row.

        Parameters
        ----------
        bearing : Bearing
            Bearing or row; only its label is used.
        result : RollerLoadDistributionResult
            Result of one row; gives q_jk [N], shape (Z, n_s).
        inner_rotating, outer_rotating : bool, optional
            Ring rotation flags of the class. Defaults True, False.
        label : str, optional
            Label of the output; the bearing label if empty.

        Returns
        -------
        LaminaDynamicEquivalentLoad
            q_kei, q_kee [N] and f_k [-] from ``stress_riser_factor(n_s)``.
        """
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
        """Equivalent lamina loads of every row of a roller bearing.

        Parameters
        ----------
        bearing : Bearing
            Single-row or multi-row roller bearing.
        result : RollerBearingResult
            Solved bearing-level result.
        inner_rotating, outer_rotating : bool, optional
            Ring rotation flags of the class. Defaults True, False.
        label : str, optional
            Base label; the bearing label if empty.

        Returns
        -------
        list[LaminaDynamicEquivalentLoad]
            One entry per row; labels get the suffix "-row<j>" with several rows.
        """
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
    """Power mean over the rollers of the load of each lamina.

    Parameters
    ----------
    q_jk : np.ndarray
        Lamina loads [N], shape (Z, n_s).
    rotating : bool
        True for a ring rotating relative to the load (p = 4), False for a
        stationary ring (p = 4.5).

    Returns
    -------
    np.ndarray
        ((1/Z) * sum_j(q_jk^p))^(1/p) [N], shape (n_s,); the term of
        ISO_TS_16281_2008, 5.3.4, Eq. (61)-(64) without the stress-riser
        factor.
    """
    p = _ROLLER_P_ROTATING if rotating else _ROLLER_P_STATIONARY
    Z = q_jk.shape[0]
    return (np.sum(q_jk ** p, axis=0) / Z) ** (1.0 / p)


_E_ROLLER = 9.0 / 8.0   # Weibull slope, line contact: Harris2007_vol1, 11.4 (PDF p. 226); ISO_TR_1281 Part 1, 6.5


@dataclass(frozen=True)
class RollerBasicReferenceRatingLife:
    """Basic reference rating life of one roller-bearing row.

    Parameters
    ----------
    label : str
        Bearing or row label.
    L10r : float
        Basic reference rating life [10^6 rev].
    q_kci : np.ndarray
        Basic dynamic load rating of each lamina at the inner ring [N], shape
        (n_s,).
    q_kei : np.ndarray
        Dynamic equivalent load of each lamina on the inner ring [N].
    q_kce : np.ndarray
        Basic dynamic load rating of each lamina at the outer ring [N].
    q_kee : np.ndarray
        Dynamic equivalent load of each lamina on the outer ring [N].

    Notes
    -----
    Equation as coded::

        L10r = (sum_k((q_kei/q_kci)^4.5 + (q_kee/q_kce)^4.5))^(-8/9)

    This is Eq. (65), which the standard writes with the inverse ratios and
    the exponent -4.5. The exponent 4.5 is fixed; it does not depend on the
    ring rotation flags. The modified reference rating life (Eq. (68)-(71)) is
    not implemented.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.3.5, Eq. (65).
    """
    label : str
    L10r  : float
    q_kci : np.ndarray
    q_kei : np.ndarray
    q_kce : np.ndarray
    q_kee : np.ndarray

    @classmethod
    def from_loads(cls, label: str, q_kci: np.ndarray, q_kei: np.ndarray,
                   q_kce: np.ndarray, q_kee: np.ndarray) -> "RollerBasicReferenceRatingLife":
        """Basic reference rating life of a roller row from its lamina loads.

        Parameters
        ----------
        label : str
            Bearing or row label.
        q_kci, q_kce : np.ndarray
            Lamina ratings of the class [N], shape (n_s,), positive.
        q_kei, q_kee : np.ndarray
            Lamina equivalent loads of the class [N], shape (n_s,), not negative.

        Returns
        -------
        RollerBasicReferenceRatingLife
            L10r [10^6 rev] by Eq. (65), with the four arrays. A row without any
            loaded lamina gives L10r = inf; the ball equivalent raises instead.

        Raises
        ------
        ValueError
            If a lamina rating is not positive or an equivalent load is negative.
        """
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
    """Basic reference rating life of each roller row and of the whole bearing.

    Parameters
    ----------
    label : str
        Bearing label; with more than one row the row labels get the suffix
        "-row<j>".
    q_kci_rows, q_kce_rows : list[np.ndarray]
        Basic dynamic load rating of each lamina at the inner and outer ring
        [N], one array per row.
    q_kei_rows, q_kee_rows : list[np.ndarray]
        Dynamic equivalent load of each lamina on the inner and outer ring
        [N], one array per row.

    Returns
    -------
    per_row : list[RollerBasicReferenceRatingLife]
        Life of each row, Eq. (65).
    L10r_bearing : float
        Life of the bearing [10^6 rev]: the row life for one row; for several
        rows ``combine_row_L10r`` with e = 9/8.

    Raises
    ------
    ValueError
        If the lists differ in length or are empty; from ``from_loads``.

    Warnings
    --------
    The combination of row lives is not part of ISO/TS 16281, and the source
    of the strict-series rule prints it for rows of balls (see
    ``combine_row_L10r``). e = 9/8 is the Lundberg-Palmgren Weibull slope
    for roller bearings [3]. The sources use it to combine the two raceways of
    a row (Harris2007_vol1, Eq. 11.148), not rows under unequal loads.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.3.5, Eq. (65).
    .. [2] Zaretsky2016, Eq. (49a).
    .. [3] Harris2007_vol1, 11.4 (PDF p. 226), Eq. 11.148; ISO_TR_1281
       (Part 1), 6.5.
    """
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
    """Dynamic equivalent reference load of a roller bearing.

    Pref_r = Cr / L10r^(3/10) and Pref_a = Ca / L10r^(3/10) [N], with L10r in
    [10^6 rev]. Fields and constructor as in
    ``DynamicEquivalentReferenceLoadBase``.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.3.6, Eq. (66)-(67).
    """
    _PREF_EXPONENT: ClassVar[float] = 3.0 / 10.0