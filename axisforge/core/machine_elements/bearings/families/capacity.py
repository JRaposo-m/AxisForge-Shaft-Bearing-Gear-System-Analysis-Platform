# =====================================================================
# families/capacity.py
# =====================================================================
"""Basic dynamic load ratings and per-rolling-element load capacities of rolling bearings.

One calculator class per contact type (point = ball, line = roller) and duty
(radial, single-row thrust at 90 deg or not). Each gives the basic dynamic
load rating (Cr or Ca) with the formulae of ISO_TR_1281, clause 6, the
capacities Q_ci, Q_ce of one rolling element at the inner and outer ring
contact with ISO_TS_16281_2008, 4.3 and 5.3, and (rollers) the capacity per
lamina; the multi-row thrust combination is in two mixins. The static rating
C0 is not implemented.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
import math


# =====================================================================
# ---- contracts -----------------------------------------------------
# =====================================================================

# ---------------------------------------------------------------------
# ---- radial duty ----------------------------------------------------
# ---------------------------------------------------------------------

class RadialCapacityCalculator(ABC):
    """Contract of the capacity calculators for radial duty.

    A radial calculator gives the basic dynamic radial load rating Cr, the
    static rating C0 and the capacities (Q_ci, Q_ce) of one rolling element.
    Implemented by PointContactCapacityRadial and LineContactCapacityRadial.
    """

    @property
    @abstractmethod
    def Cr(self) -> float:
        """Basic dynamic radial load rating of the bearing [N].

        ISO 281:2007 (not in the library; (source to find)). The concrete
        calculators use the formulae of ISO_TR_1281, clause 6.
        """

    @property
    def dynamic_rating(self) -> float:
        """Basic dynamic load rating [N]: alias of ``Cr``, same name as in the thrust contract."""
        return self.Cr

    @property
    @abstractmethod
    def C0(self) -> float:
        """Basic static radial load rating of the bearing [N].

        ISO 76 (not in the library; (source to find)). Every concrete
        calculator raises NotImplementedError.
        """

    @property
    @abstractmethod
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one rolling element at the inner and outer ring contact [N].

        ISO_TS_16281_2008, 4.3 (ball) and 5.3 (roller).
        """

# ---------------------------------------------------------------------
# ---- thrust duty ----------------------------------------------------
# ---------------------------------------------------------------------

class ThrustCapacityCalculator(ABC):
    """Contract of the capacity calculators for thrust (axial) duty.

    A thrust calculator gives the basic dynamic axial load rating Ca, the
    static rating C0 and the capacities (Q_ci, Q_ce) of one rolling element.
    Implemented by the PointContactCapacityThrust_* and
    LineContactCapacityThrust_* classes.
    """

    @property
    @abstractmethod
    def Ca(self) -> float:
        """Basic dynamic axial load rating of the bearing [N].

        ISO 281:2007 (not in the library; (source to find)). The concrete
        calculators use the formulae of ISO_TR_1281, clause 6.
        """

    @property
    def dynamic_rating(self) -> float:
        """Basic dynamic load rating [N]: alias of ``Ca``.

        Lets ``combine_multirow`` (and any generic code) read the dynamic
        rating without knowing whether it is radial (Cr) or thrust (Ca).
        """
        return self.Ca

    @property
    @abstractmethod
    def C0(self) -> float:
        """Basic static axial load rating of the bearing [N].

        ISO 76 (not in the library; (source to find)). Every concrete
        calculator raises NotImplementedError.
        """

    @property
    @abstractmethod
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one rolling element at the inner and outer ring contact [N].

        ISO_TS_16281_2008, 4.3 (ball) and 5.3 (roller).
        """

# ---------------------------------------------------------------------
# ---- Multi row helper -----------------------------------------------
# ---------------------------------------------------------------------


class _MultirowCombinablePointContact:
    """Mixin: dynamic axial rating of a ball bearing with several rows, from the single-row ratings.

    Mixed into the two single-row thrust ball calculators
    (PointContactCapacityThrust_Non_90deg and _90deg); called by
    ThrustBallMultiRowFamily.assemble_geometry.

    Notes
    -----
    Equation as coded (Ca_k = ``dynamic_rating`` of an instance built from row
    k, Z_k = ``row["Z"]``)::

        Ca = (Z_1 + ... + Z_n) * [ sum_k (Z_k / Ca_k)^(10/3) ]^(-3/10)     [N]

    Same symbols as the standard. Assumes that the load on a row is
    proportional to the number of balls of that row.

    References
    ----------
    .. [1] ISO_TR_1281, 6.4, Formula (29) (2021 edition numbering).
    .. [2] Harris2007_vol1, ch. 11, Eq. 11.141.
    """

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        """Dynamic axial load rating Ca of the multi-row bearing [N].

        Parameters
        ----------
        rows : Sequence[dict]
            One dict of constructor keyword arguments of ``cls`` per row
            (Z, Dw, alpha_0, ri, re, gamma, lam, eta, optionally i); at least 2.

        Returns
        -------
        float
            Ca of the whole bearing [N].

        Raises
        ------
        ValueError
            If fewer than 2 rows are given.
        """
        if len(rows) < 2:
            raise ValueError(
                f"combine_multirow requires at least 2 rows; got {len(rows)}. "
                f"Use .Ca directly for a single row."
            )
        instances = [cls(**row) for row in rows]
        Ca_rows = [inst.dynamic_rating for inst in instances]
        Z_rows = [row["Z"] for row in rows]

        total_Z = sum(Z_rows)
        bracket = sum((z / ca) ** (10.0 / 3.0) for z, ca in zip(Z_rows, Ca_rows))
        return total_Z * bracket ** (-3.0 / 10.0)


class _MultirowCombinableLineContact:
    """Mixin: dynamic axial rating of a roller bearing with several rows, from the single-row ratings.

    Mixed into the two single-row thrust roller calculators
    (LineContactCapacityThrust_Non_90deg and _90deg); called by
    RollerThrustMultiRowFamily.assemble_geometry. Same structure as the ball
    mixin with the line-contact exponents 9/2 and -2/9 and the row weight
    Z_k * Lwe_k.

    Notes
    -----
    Equation as coded (Ca_k = ``dynamic_rating`` of an instance built from row
    k, Z_k = ``row["Z"]``, Lwe_k = ``row["Lwe"]``)::

        Ca = (sum_k Z_k Lwe_k) * [ sum_k (Z_k Lwe_k / Ca_k)^(9/2) ]^(-2/9)     [N]

    Assumes that the load on a row is proportional to the product of roller
    length and number of rollers of that row.

    References
    ----------
    .. [1] ISO_TR_1281, 6.7, Formula (46), printed for thrust roller bearings.
    """

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        """Dynamic axial load rating Ca of the multi-row bearing [N].

        Parameters
        ----------
        rows : Sequence[dict]
            One dict of constructor keyword arguments of ``cls`` per row
            (Z, Dwe, Lwe, alpha_0, gamma, lambda_v, eta, optionally i); at
            least 2.

        Returns
        -------
        float
            Ca of the whole bearing [N].

        Raises
        ------
        ValueError
            If fewer than 2 rows are given.
        """
        if len(rows) < 2:
            raise ValueError(
                f"combine_multirow requires at least 2 rows; got {len(rows)}. "
                f"Use .Ca directly for a single row."
            )
        instances = [cls(**row) for row in rows]
        Ca_rows = [inst.dynamic_rating for inst in instances]
        Z_rows = [row["Z"] for row in rows]
        Lwe_rows = [row["Lwe"] for row in rows]

        weighted = [z * lwe for z, lwe in zip(Z_rows, Lwe_rows)]
        total_weighted = sum(weighted)
        bracket = sum((w / ca) ** (9.0 / 2.0) for w, ca in zip(weighted, Ca_rows))
        return total_weighted * bracket ** (-2.0 / 9.0)
    
# =====================================================================
# ---- point contact / radial duty-------------------------------------
# =====================================================================

class PointContactCapacityRadial(RadialCapacityCalculator):
    """Basic dynamic radial load rating and element capacities of a radial ball bearing.

    Instantiate once with the geometry; read ``Cr``, ``Q_elements`` as needed
    (shared terms are computed once and cached). Used by DeepGrooveBallFamily,
    AngularContactFamily and SelfAligningBallFamily.

    Parameters
    ----------
    Z : int
        Number of balls per row [-].
    Dw : float
        Ball diameter [mm].
    alpha_0 : float
        Initial contact angle [rad].
    ri : float
        Cross-sectional radius of the inner raceway groove [mm].
    re : float
        Cross-sectional radius of the outer raceway groove [mm]. ``inf`` is
        accepted by ``Cr`` (magneto-bearing limit) but not by ``Q_elements``.
    gamma : float
        Auxiliary parameter Dw cos(contact angle)/Dpw [-], in (0, 1); used as
        passed.
    reduction_factor : float
        Reduction factor lambda of the rating [-].
    i : int, optional
        Number of rows of balls [-].

    Raises
    ------
    ValueError
        If Z, Dw or i is not positive, if 2 ri or 2 re <= Dw, or if gamma is
        not in (0, 1).

    Notes
    -----
    Equations as coded (Dw in mm, Cr and Q in N)::

        groove = (2 ri / (2 ri - Dw))^0.41
        g_term = gamma^0.3 (1 - gamma)^1.39 / (1 + gamma)^(1/3)
        G      = (ri / re) (2 re - Dw) / (2 ri - Dw)      (2 ri / (2 ri - Dw) if re = inf)
        f_c    = 98.0665 * 0.41 * reduction_factor * groove * g_term
                 * [1 + (1.04 G^0.41 ((1 - gamma)/(1 + gamma))^1.72)^(10/3)]^(-3/10)
        Cr     = f_c (i cos(alpha_0))^0.7 Z^(2/3) Dw^1.8            Dw <= 25.4 mm
        Cr     = 3.647 f_c (i cos(alpha_0))^0.7 Z^(2/3) Dw^1.4      Dw >  25.4 mm
        B      = 1.044 G^0.41 ((1 - gamma)/(1 + gamma))^1.72
        Q_ci   = Cr / (0.407 Z cos(alpha_0) i^0.7) * (1 + B^(10/3))^0.3
        Q_ce   = Cr / (0.389 Z cos(alpha_0) i^0.7) * (1 + B^(-10/3))^0.3

    Code names and symbols of the standards: reduction_factor = lambda,
    ri, re = r_i, r_e, Dw = D_w, i = i, Z = Z. 98.0665 is "0,089 A_1" (value
    for Cr in N with Dw in mm); the 0.41 is a separate factor of Formula (15).
    f_c carries the units that give Cr in N with Dw in mm. The two Q
    equations are Eq. (19)-(20) with the form of Cor. 1:2009 (cos(alpha) to
    the power 1 and i^0,7 in the denominator).

    The ``re = inf`` form of G is the limit of the printed ratio for
    r_e -> infinity (ISO_TR_1281, Table 1 lists r_e = infinity for separable
    (magneto) bearings; the limit is not printed). No family passes ``inf``.

    Warnings
    --------
    Cr is computed as f_c (...) without a rating factor b_m. This is the form
    printed in ISO_TR_1281, clause 6 (Formulae (13)-(14)); Harris2007_vol1,
    Eq. 11.172 and Table 11.11 print C = b_m f_c (...). ISO 281:2007 is not in
    the library ((source to find)).

    Differs from ISO_TR_1281, Formulae (13)-(14), and ISO_TS_16281_2008,
    Eq. (19)-(20): Cr, Q_ci and Q_ce use cos(alpha_0) of the initial contact
    angle where the standards print the nominal contact angle alpha. ``gamma``
    is used as passed (the families pass the value computed with alpha_0, see
    BallBearingGeometry).

    ``Q_elements`` does not handle ``re = inf`` (the ratio G becomes nan),
    unlike ``Cr``.

    References
    ----------
    .. [1] ISO_TR_1281, 6.2, Formulae (13)-(15), Table 1.
    .. [2] ISO_TS_16281_2008, 4.3.1.2, Eq. (19)-(20), with Cor. 1:2009.
    """

    _A1_0089_N = 98.0665      # 0,089*A_1 [N, Dw in mm]: ISO_TR_1281, 6.2, sentence after Formula (15)
    _DW_THRESHOLD_MM = 25.4   # ISO_TR_1281, Formulae (13)/(14): Dw^1.8 up to 25,4 mm, 3,647 * Dw^1.4 above

    def __init__(self, Z: int, Dw: float, alpha_0: float, ri: float, re: float,
                 gamma: float, reduction_factor: float, i: int = 1):
        """Store the arguments; see the class docstring (it states what is validated)."""
        if Z <= 0 or Dw <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dw and i must all be positive; got Z={Z}, Dw={Dw}, i={i}.")
        if 2.0 * ri <= Dw or 2.0 * re <= Dw:
            raise ValueError("groove radius too small relative to Dw.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dw, self.alpha_0 = Z, Dw, alpha_0
        self.ri, self.re, self.gamma = ri, re, gamma
        self.reduction_factor, self.i = reduction_factor, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Cr and (indirectly) C0 --
    @property
    def _fc(self) -> float:
        """Rating factor f_c of ISO_TR_1281, Formula (15), including lambda."""
        if "_fc" not in self._cache:
            groove_conformity = (2.0 * self.ri / (2.0 * self.ri - self.Dw)) ** 0.41
            gamma_term = self.gamma ** 0.3 * (1.0 - self.gamma) ** 1.39 / (1.0 + self.gamma) ** (1.0 / 3.0)

            if math.isinf(self.re):
                radii_ratio = 2.0 * self.ri / (2.0 * self.ri - self.Dw)
            else:
                radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))

            bracket = 1.04 * radii_ratio ** 0.41 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
            correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

            self._cache["_fc"] = (self._A1_0089_N * 0.41 * self.reduction_factor
                                   * groove_conformity * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def _q_bracket(self) -> float:
        """Brace B of Q_ci, Q_ce [-]: 1.044 G^0.41 ((1 - gamma)/(1 + gamma))^1.72.

        ISO_TS_16281_2008, 4.3.1.2, Eq. (19)-(20). The brace of f_c in
        ``_fc`` is the same expression with 1.04 instead of 1.044.
        """
        if "_q_bracket" not in self._cache:
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))
            self._cache["_q_bracket"] = (1.044 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
                                          * radii_ratio ** 0.41)
        return self._cache["_q_bracket"]

    @property
    def Cr(self) -> float:
        """Basic dynamic radial load rating Cr [N] of the ball bearing.

        ISO_TR_1281, 6.2, Formulae (13)-(14): Dw^1.8 for Dw <= 25.4 mm,
        3.647 * Dw^1.4 above. No rating factor b_m (see the class Warnings).
        """
        cos_term = (self.i * math.cos(self.alpha_0)) ** 0.7
        if self.Dw <= self._DW_THRESHOLD_MM:
            return self._fc * cos_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * cos_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.4

    @property
    def C0(self) -> float:
        """Basic static radial load rating [N]; not implemented, always raises NotImplementedError (ISO 76, (source to find))."""
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one ball at the inner and outer ring contact [N].

        Computed from this instance's own ``Cr``; there is no parameter to
        pass a different Cr. Equations in the class Notes; ISO_TS_16281_2008,
        4.3.1.2, Eq. (19)-(20), with Cor. 1:2009.
        """
        Cr = self.Cr
        cos_alpha_07 = np.cos(self.alpha_0) * self.i ** 0.7
        Q_ci = (Cr / (0.407 * self.Z * cos_alpha_07)) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Cr / (0.389 * self.Z * cos_alpha_07)) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce


# =====================================================================
# ---- point contact / thrust duty-------------------------------------
# =====================================================================

class PointContactCapacityThrust_Non_90deg(ThrustCapacityCalculator, _MultirowCombinablePointContact):
    """Basic dynamic axial load rating and element capacities of a single-row thrust ball bearing, contact angle other than 90 deg.

    Used by ThrustBallSingleRowFamily when alpha_0 is not 90 deg, and by
    ThrustBallMultiRowFamily for each row (``combine_multirow``).

    Parameters
    ----------
    Z : int
        Number of balls [-].
    Dw : float
        Ball diameter [mm].
    alpha_0 : float
        Initial contact angle [rad].
    ri : float
        Cross-sectional radius of the inner (shaft washer) groove [mm].
    re : float
        Cross-sectional radius of the outer (housing washer) groove [mm].
    gamma : float
        Auxiliary parameter Dw cos(contact angle)/Dpw [-], in (0, 1); used as
        passed.
    lam : float
        Reduction factor lambda [-].
    eta : float
        Reduction factor eta for the unequal load distribution among the balls
        [-].
    i : int, optional
        Number of rows [-]. Validated and stored; it appears in no equation of
        this class.

    Raises
    ------
    ValueError
        If Z, Dw or i is not positive, if 2 ri or 2 re <= Dw, or if gamma is
        not in (0, 1).

    Notes
    -----
    Equations as coded (Dw in mm, Ca and Q in N)::

        groove = (2 ri / (2 ri - Dw))^0.41
        g_term = gamma^0.3 (1 - gamma)^1.39 / (1 + gamma)^(1/3)
        G      = (ri / re) (2 re - Dw) / (2 ri - Dw)
        f_c    = 98.0665 * lam * eta * groove * g_term
                 * [1 + (G^0.41 ((1 - gamma)/(1 + gamma))^1.72)^(10/3)]^(-3/10)
        Ca     = f_c cos(alpha_0)^0.7 tan(alpha_0) Z^(2/3) Dw^1.8         Dw <= 25.4 mm
        Ca     = 3.647 f_c cos(alpha_0)^0.7 tan(alpha_0) Z^(2/3) Dw^1.4   Dw >  25.4 mm
        B      = 1.044 G^0.41 ((1 - gamma)/(1 + gamma))^1.72
        Q_ci   = Ca / (Z sin(alpha_0)) * (1 + B^(10/3))^0.3
        Q_ce   = Ca / (Z sin(alpha_0)) * (1 + B^(-10/3))^0.3

    Code names and symbols of the standards: lam = lambda, eta = eta,
    ri, re = r_i, r_e, Dw = D_w. 98.0665 is "0,089 A_1" (Ca in N, Dw in mm).

    Warnings
    --------
    Differs from ISO_TS_16281_2008, 4.3.1.3, Eq. (21)-(22): the brace B is
    coded with the factor 1.044; the standard prints the brace
    ((1-gamma)/(1+gamma))^1,72 G^0,41 without 1,044 (the radial Eq. (19)-(20)
    have it).

    Ca is computed as f_c (...) without a rating factor b_m, the form
    printed in ISO_TR_1281, 6.3.1 (Formulae (18)-(19)); Harris2007_vol1, Eq. 11.172 and Table 11.11 print
    C = b_m f_c (...). ISO 281:2007 is not in the library ((source to find)).

    Ca, Q_ci and Q_ce use cos(alpha_0) and sin(alpha_0) of the initial contact
    angle where the standards print the nominal contact angle alpha.

    References
    ----------
    .. [1] ISO_TR_1281, 6.3.1, Formulae (18)-(20); Table 1 (lam, eta).
    .. [2] ISO_TS_16281_2008, 4.3.1.3, Eq. (21)-(22).
    """

    _A1_0089_N = 98.0665      # 0,089*A_1 [N, Dw in mm]: ISO_TR_1281, 6.3.1, sentence after Formula (20)
    _DW_THRESHOLD_MM = 25.4   # ISO_TR_1281, Formulae (18)/(19): Dw^1.8 up to 25,4 mm, 3,647 * Dw^1.4 above

    def __init__(self, Z: int, Dw: float, alpha_0: float, ri: float, re: float,
                 gamma: float, lam: float, eta: float, i: int = 1):
        """Store the arguments; see the class docstring (it states what is validated)."""
        if Z <= 0 or Dw <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dw and i must all be positive; got Z={Z}, Dw={Dw}, i={i}.")
        if 2.0 * ri <= Dw or 2.0 * re <= Dw:
            raise ValueError("groove radius too small relative to Dw.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dw, self.alpha_0 = Z, Dw, alpha_0
        self.ri, self.re, self.gamma = ri, re, gamma
        self.lam, self.eta, self.i = lam, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        """Rating factor f_c of ISO_TR_1281, Formula (20), including lambda and eta."""
        if "_fc" not in self._cache:
            groove_conformity = (2.0 * self.ri / (2.0 * self.ri - self.Dw)) ** 0.41
            gamma_term = self.gamma ** 0.3 * (1.0 - self.gamma) ** 1.39 / (1.0 + self.gamma) ** (1.0 / 3.0)

            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))

            bracket = radii_ratio ** 0.41 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
            correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

            self._cache["_fc"] = (self._A1_0089_N * self.lam * self.eta
                                   * groove_conformity * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        """Basic dynamic axial load rating Ca [N] of the thrust ball bearing, alpha != 90 deg.

        ISO_TR_1281, 6.3.1, Formulae (18)-(19): Dw^1.8 for Dw <= 25.4 mm,
        3.647 * Dw^1.4 above. No rating factor b_m (see the class Warnings).
        """
        cos_term = np.cos(self.alpha_0) ** 0.7
        tan_term = np.tan(self.alpha_0)
        if self.Dw <= self._DW_THRESHOLD_MM:
            return self._fc * cos_term * tan_term * self.Z ** (2.0/3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * cos_term * tan_term * self.Z ** (2.0/3.0) * self.Dw ** 1.4 

    @property
    def _q_bracket(self) -> float:
        """Brace B of Q_ci, Q_ce [-]: 1.044 G^0.41 ((1 - gamma)/(1 + gamma))^1.72.

        Coded with 1.044, the factor of the radial Eq. (19)-(20); the thrust
        Eq. (21)-(22) of ISO_TS_16281_2008, 4.3.1.3 print the brace without it
        (see the class Warnings). The brace of f_c in ``_fc`` has no 1.04.
        """
        if "_q_bracket" not in self._cache:
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))
            self._cache["_q_bracket"] = (1.044 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
                                          * radii_ratio ** 0.41)
        return self._cache["_q_bracket"]

    @property
    def C0(self) -> float:
        """Basic static axial load rating [N]; not implemented, always raises NotImplementedError (ISO 76, (source to find))."""
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one ball at the inner and outer ring contact [N].

        Computed from this instance's own ``Ca``. Equations in the class
        Notes; ISO_TS_16281_2008, 4.3.1.3, Eq. (21)-(22) (brace differs, see
        the class Warnings).
        """
        Ca = self.Ca
        sin_alpha_0 = np.sin(self.alpha_0)
        Q_ci = (Ca / (self.Z * sin_alpha_0)) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / (self.Z * sin_alpha_0)) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce


class PointContactCapacityThrust_90deg(ThrustCapacityCalculator, _MultirowCombinablePointContact):
    """Basic dynamic axial load rating and element capacities of a single-row thrust ball bearing, contact angle 90 deg.

    Used by ThrustBallSingleRowFamily when alpha_0 is 90 deg (the default of
    that family), and by ThrustBallMultiRowFamily for each such row
    (``combine_multirow``).

    Parameters
    ----------
    Z : int
        Number of balls [-].
    Dw : float
        Ball diameter [mm].
    alpha_0 : float
        Initial contact angle [rad]. Stored; it appears in no equation of this
        class.
    ri : float
        Cross-sectional radius of the inner (shaft washer) groove [mm].
    re : float
        Cross-sectional radius of the outer (housing washer) groove [mm].
    gamma : float
        Auxiliary parameter Dw/Dpw [-], in (0, 1); used as passed.
    lam : float
        Reduction factor lambda [-].
    eta : float
        Reduction factor eta for the unequal load distribution among the balls
        [-].
    i : int, optional
        Number of rows [-]. Validated and stored; it appears in no equation of
        this class.

    Raises
    ------
    ValueError
        If Z, Dw or i is not positive, if 2 ri or 2 re <= Dw, or if gamma is
        not in (0, 1).

    Notes
    -----
    Equations as coded (Dw in mm, Ca and Q in N)::

        groove = (2 ri / (2 ri - Dw))^0.41
        G      = (ri / re) (2 re - Dw) / (2 ri - Dw)
        f_c    = 98.0665 * lam * eta * groove * gamma^0.3 * [1 + (G^0.41)^(10/3)]^(-3/10)
        Ca     = f_c Z^(2/3) Dw^1.8              Dw <= 25.4 mm
        Ca     = 3.647 f_c Z^(2/3) Dw^1.4        Dw >  25.4 mm
        Q_ci   = (Ca / Z) * (1 + (G^0.41)^(10/3))^0.3
        Q_ce   = (Ca / Z) * (1 + (G^0.41)^(-10/3))^0.3

    Code names and symbols of the standards: lam = lambda, eta = eta,
    ri, re = r_i, r_e, Dw = D_w; gamma = D_w/D_pw as in Formula (25). 98.0665
    is "0,089 A_1" (Ca in N, Dw in mm).

    Warnings
    --------
    Ca is computed as f_c (...) without a rating factor b_m, the form
    printed in ISO_TR_1281, 6.3.2 (Formulae (23)-(24)); Harris2007_vol1, Eq. 11.172 and Table 11.11 print
    C = b_m f_c (...). ISO 281:2007 is not in the library ((source to find)).

    References
    ----------
    .. [1] ISO_TR_1281, 6.3.2, Formulae (23)-(25); Table 1 (lam, eta).
    .. [2] ISO_TS_16281_2008, 4.3.1.4, Eq. (23)-(24).
    """

    _A1_0089_N = 98.0665      # 0,089*A_1 [N, Dw in mm]: ISO_TR_1281, 6.3.2, sentence after Formula (25)
    _DW_THRESHOLD_MM = 25.4   # ISO_TR_1281, Formulae (23)/(24): Dw^1.8 up to 25,4 mm, 3,647 * Dw^1.4 above

    def __init__(self, Z: int, Dw: float, alpha_0: float, ri: float, re: float,
                 gamma: float, lam: float, eta: float, i: int = 1):
        """Store the arguments; see the class docstring (it states what is validated)."""
        if Z <= 0 or Dw <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dw and i must all be positive; got Z={Z}, Dw={Dw}, i={i}.")
        if 2.0 * ri <= Dw or 2.0 * re <= Dw:
            raise ValueError("groove radius too small relative to Dw.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dw, self.alpha_0 = Z, Dw, alpha_0
        self.ri, self.re, self.gamma = ri, re, gamma
        self.lam, self.eta, self.i = lam, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        """Rating factor f_c of ISO_TR_1281, Formula (25), including lambda and eta."""
        if "_fc" not in self._cache:
            groove_conformity = (2.0 * self.ri / (2.0 * self.ri - self.Dw)) ** 0.41
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))

            bracket = radii_ratio ** 0.41
            correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

            self._cache["_fc"] = (self._A1_0089_N * self.lam * self.eta
                                   * groove_conformity * (self.gamma ** 0.3) * correction)
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        """Basic dynamic axial load rating Ca [N] of the thrust ball bearing, alpha = 90 deg.

        ISO_TR_1281, 6.3.2, Formulae (23)-(24): Dw^1.8 for Dw <= 25.4 mm,
        3.647 * Dw^1.4 above. No rating factor b_m (see the class Warnings).
        """
        if self.Dw <= self._DW_THRESHOLD_MM:
            return self._fc * self.Z ** (2.0/3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * self.Z ** (2.0/3.0) * self.Dw ** 1.4 

    @property
    def _q_bracket(self) -> float:
        """Brace of Q_ci, Q_ce [-]: G^0.41, ISO_TS_16281_2008, 4.3.1.4, Eq. (23)-(24)."""
        if "_q_bracket" not in self._cache:
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))
            self._cache["_q_bracket"] = radii_ratio ** 0.41
        return self._cache["_q_bracket"]

    @property
    def C0(self) -> float:
        """Basic static axial load rating [N]; not implemented, always raises NotImplementedError (ISO 76, (source to find))."""
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one ball at the inner and outer ring contact [N].

        Computed from this instance's own ``Ca``; equations in the class
        Notes; ISO_TS_16281_2008, 4.3.1.4, Eq. (23)-(24).
        """
        Ca = self.Ca
        Q_ci = (Ca / self.Z) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / self.Z) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce

# =====================================================================
# ---- line contact / radial duty--------------------------------------
# =====================================================================


class LineContactCapacityRadial(RadialCapacityCalculator):
    """Basic dynamic radial load rating and element capacities of a radial roller bearing.

    Cr, the capacities Q_ci, Q_ce of one roller and the capacities q_ci, q_ce
    of one lamina of the roller. Used by CylindricalRollerFamily.

    Parameters
    ----------
    Z : int
        Number of rollers per row [-].
    Dwe : float
        Mean roller diameter [mm].
    Lwe : float
        Effective contact length of the roller [mm].
    alpha_0 : float
        Initial contact angle [rad].
    gamma : float
        Auxiliary parameter Dwe cos(contact angle)/Dpw [-], in (0, 1); used
        as passed.
    lambda_v : float
        Product lambda*nu of the reduction factor and the exponent-variation
        factor [-] (0.83 for the radial roller bearing).
    n_s : float
        Number of laminae of the roller [-]; used only by ``per_lamina``.
    i : int, optional
        Number of rows of rollers [-].

    Raises
    ------
    ValueError
        If Z, Dwe, Lwe or i is not positive, or if gamma is not in (0, 1).

    Notes
    -----
    Equations as coded (Dwe, Lwe in mm, Cr and Q in N)::

        f_c  = 551.13373 * 0.377 * lambda_v * gamma^(2/9) (1 - gamma)^(29/27) / (1 + gamma)^(1/4)
               * {1 + [1.04 ((1 - gamma)/(1 + gamma))^(143/108)]^(9/2)}^(-2/9)
        Cr   = f_c (i Lwe cos(alpha_0))^(7/9) Z^(3/4) Dwe^(29/27)
        N    = 1.038 ((1 - gamma)/(1 + gamma))^(143/108)
        Q_ci = (1/lambda_v) Cr / (0.378 Z cos(alpha_0) i^(7/9)) * (1 + N^(9/2))^(2/9)
        Q_ce = (1/lambda_v) Cr / (0.364 Z cos(alpha_0) i^(7/9)) * (1 + N^(-9/2))^(2/9)
        q_ci = Q_ci (1 / n_s)^(7/9),   q_ce = Q_ce (1 / n_s)^(7/9)

    Code names and symbols of the standards: lambda_v = lambda*nu, Dwe = D_we,
    Lwe = L_we, i = i. 551.13373 is "0,483 B_1" (Cr in N, Dwe in mm).

    Warnings
    --------
    Cr is computed as f_c (...) without a rating factor b_m. This is the form
    printed in ISO_TR_1281, 6.5 (Formula (33)); Harris2007_vol1, Eq. 11.172
    and Table 11.11 print C = b_m f_c (...). ISO 281:2007 is not in the
    library ((source to find)).

    Differs from ISO_TR_1281, Formula (33), and ISO_TS_16281_2008,
    Eq. (47)-(48): cos(alpha_0) of the initial contact angle is used where the
    standards print the nominal contact angle alpha.

    References
    ----------
    .. [1] ISO_TR_1281, 6.5, Formulae (33)-(34); Table 2 (lambda*nu).
    .. [2] ISO_TS_16281_2008, 5.3.1.2, Eq. (47)-(48), (49); 5.3.2,
       Eq. (56)-(57).
    """

    _B1_0483_N = 551.13373    # 0,483*B_1 [N, Dwe in mm]: ISO_TR_1281, 6.5, sentence after Formula (34)

    def __init__(self, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                 gamma: float, lambda_v: float, n_s: float, i: int = 1):
        """Store the arguments; see the class docstring (it states what is validated)."""
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dwe, Lwe and i must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dwe, self.Lwe, self.alpha_0 = Z, Dwe, Lwe, alpha_0
        self.gamma, self.lambda_v, self.n_s, self.i = gamma, lambda_v, n_s, i
        self._cache: dict[str, float] = {}

    @property
    def _fc(self) -> float:
        """Rating factor f_c of ISO_TR_1281, Formula (34), including lambda*nu."""
        if "_fc" not in self._cache:
            gamma_term = (self.gamma ** (2.0 / 9.0) * (1.0 - self.gamma) ** (29.0 / 27.0)
                        / (1.0 + self.gamma) ** (1.0 / 4.0))
            bracket = 1.04 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)
            correction = (1.0 + bracket ** (9.0 / 2.0)) ** (-2.0 / 9.0)

            self._cache["_fc"] = (self._B1_0483_N * 0.377 * self.lambda_v * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def Cr(self) -> float:
        """Basic dynamic radial load rating Cr [N] of the roller bearing.

        ISO_TR_1281, 6.5, Formula (33). No rating factor b_m (see the class
        Warnings).
        """
        cos_term = (self.i * self.Lwe * np.cos(self.alpha_0)) ** (7.0 / 9.0)
        return self._fc * cos_term * self.Z ** (3.0/4.0) * self.Dwe ** (29.0 / 27.0)

    @property
    def C0(self) -> float:
        """Basic static radial load rating [N]; not implemented, always raises NotImplementedError (no message; ISO 76, (source to find))."""
        raise NotImplementedError

    @property
    def _numer(self) -> float:
        """Factor N = 1.038 ((1 - gamma)/(1 + gamma))^(143/108) of Q_ci, Q_ce [-] (ISO_TS_16281_2008, Eq. (47)-(48))."""
        return 1.038 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)

    @property
    def _denom(self) -> float:
        """Factor cos(alpha_0) i^(7/9) of the denominator of Q_ci, Q_ce [-]."""
        return np.cos(self.alpha_0) * (self.i ** (7.0 / 9.0))

    @property
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one roller at the inner and outer ring contact [N].

        ISO_TS_16281_2008, 5.3.1.2, Eq. (47)-(48); equations in the class Notes.
        """
        Q_ci = (1.0 / self.lambda_v) * (self.Cr / (0.378 * self.Z * self._denom)) * (
            1.0 + self._numer ** (9.0 / 2.0)) ** (2.0 / 9.0)
        Q_ce = (1.0 / self.lambda_v) * (self.Cr / (0.364 * self.Z * self._denom)) * (
            1.0 + self._numer ** (-9.0 / 2.0)) ** (2.0 / 9.0)
        return Q_ci, Q_ce

    @property
    def per_lamina(self) -> tuple[float, float]:
        """Load capacities (q_ci, q_ce) of one lamina of a roller [N].

        ``Q * (1 / n_s)^(7/9)``; ISO_TS_16281_2008, 5.3.2, Eq. (56)-(57).
        """
        Q_ci, Q_ce = self.Q_elements
        q_ci = Q_ci * (1.0 / self.n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / self.n_s) ** (7.0 / 9.0)
        return q_ci, q_ce

# =====================================================================
# ---- line contact / thrust duty--------------------------------------
# =====================================================================

class LineContactCapacityThrust_Non_90deg(ThrustCapacityCalculator, _MultirowCombinableLineContact):
    """Basic dynamic axial load rating and element capacities of a single-row thrust roller bearing, contact angle other than 90 deg.

    Used by ThrustCylindricalRollerFamily when alpha_0 is not 90 deg, and by
    RollerThrustMultiRowFamily for each row (``combine_multirow``).

    Parameters
    ----------
    Z : int
        Number of rollers [-].
    Dwe : float
        Mean roller diameter [mm].
    Lwe : float
        Effective contact length of the roller [mm].
    alpha_0 : float
        Initial contact angle [rad].
    gamma : float
        Auxiliary parameter Dwe cos(contact angle)/Dpw [-] (ISO_TR_1281,
        clause 4, roller bearings with alpha != 90 deg), in (0, 1); used as
        passed. No family supplies it today (ThrustCylindricalRollerFamily
        cannot assemble).
    lambda_v : float
        Product lambda*nu [-] (0.73 for the thrust roller bearing).
    eta : float
        Reduction factor eta for the unequal load distribution among the
        rollers [-].
    i : int, optional
        Number of rows [-]. Validated and stored; it appears in no equation of
        this class.

    Raises
    ------
    ValueError
        If Z, Dwe, Lwe or i is not positive, or if gamma is not in (0, 1).

    Notes
    -----
    Equations as coded (Dwe, Lwe in mm, Ca and Q in N)::

        f_c  = 551.13373 * lambda_v * eta * gamma^(2/9) (1 - gamma)^(29/27) / (1 + gamma)^(1/4)
               * {1 + [((1 - gamma)/(1 + gamma))^(143/108)]^(9/2)}^(-2/9)
        Ca   = f_c (Lwe cos(alpha_0))^(7/9) tan(alpha_0) Z^(3/4) Dwe^(29/27)
        N    = ((1 - gamma)/(1 + gamma))^(143/108)
        Q_ci = (1/lambda_v) Ca / (Z sin(alpha_0)) * (1 + N^(9/2))^(2/9)
        Q_ce = (1/lambda_v) Ca / (Z sin(alpha_0)) * (1 + N^(-9/2))^(2/9)

    Code names and symbols of the standards: lambda_v = lambda*nu, eta = eta,
    Dwe = D_we, Lwe = L_we. 551.13373 is "0,483 B_1" (Ca in N, Dwe in mm). The
    prefactor 0.377 and the factors 1.04 and 1.038 of the radial equations do
    not appear, as in the printed Formula (38) and Eq. (50)-(51).

    Warnings
    --------
    Ca is computed as f_c (...) without a rating factor b_m, the form printed
    in ISO_TR_1281, 6.6.1 (Formula (37)); Harris2007_vol1, Eq. 11.172 and
    Table 11.11 print C = b_m f_c (...). ISO 281:2007 is not in the library
    ((source to find)).

    Ca, Q_ci and Q_ce use cos(alpha_0) and sin(alpha_0) of the initial contact
    angle where the standards print the nominal contact angle alpha.

    ``per_lamina`` reads ``self.n_s``, which this class neither takes nor
    stores: it raises AttributeError (so
    ThrustCylindricalRollerFamily.per_lamina_dynamic_capacity fails).

    References
    ----------
    .. [1] ISO_TR_1281, 6.6.1, Formulae (37)-(38); Table 2 (lambda*nu, eta).
    .. [2] ISO_TS_16281_2008, 5.3.1.3, Eq. (50)-(51), (52).
    """

    # 0,483*B_1 [N, Dwe in mm]: ISO_TR_1281, 6.6.1, sentence after Formula (38)
    _B1_0483_N = 551.13373     

    def __init__(self, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                 gamma: float, lambda_v: float, eta: float, i: int = 1):
        """Store the arguments; see the class docstring (it states what is validated)."""
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0 or i <= 0:
            raise ValueError(f"Z, Dwe, Lwe and i must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dwe, self.Lwe, self.alpha_0 = Z, Dwe, Lwe, alpha_0
        self.gamma, self.lambda_v, self.eta, self.i = gamma, lambda_v, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        """Rating factor f_c of ISO_TR_1281, Formula (38), including lambda*nu and eta."""
        if "_fc" not in self._cache:
            gamma_term = (self.gamma ** (2.0 / 9.0) * (1.0 - self.gamma) ** (29.0 / 27.0)
                          / (1.0 + self.gamma) ** (1.0 / 4.0))

            bracket = ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)
            correction = (1.0 + bracket ** (9.0 / 2.0)) ** (-2.0 / 9.0)

            self._cache["_fc"] = (self._B1_0483_N * self.lambda_v * self.eta
                                   * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        """Basic dynamic axial load rating Ca [N] of the thrust roller bearing, alpha != 90 deg.

        ISO_TR_1281, 6.6.1, Formula (37). No rating factor b_m (see the class
        Warnings).
        """
        cos_term = (self.Lwe * np.cos(self.alpha_0)) ** (7.0 / 9.0)
        tan_term = np.tan(self.alpha_0)
        return self._fc * cos_term * tan_term * self.Z ** (3.0 / 4.0) * self.Dwe ** (29.0 / 27.0) 

    @property
    def _numer(self) -> float:
        """Factor N = ((1 - gamma)/(1 + gamma))^(143/108) of Q_ci, Q_ce [-] (ISO_TS_16281_2008, Eq. (50)-(51))."""
        return ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)

    @property
    def _denom(self) -> float:
        """Factor Z sin(alpha_0) of the denominator of Q_ci, Q_ce [-]."""
        return (self.Z * np.sin(self.alpha_0))

    @property
    def C0(self) -> float:
        """Basic static axial load rating [N]; not implemented, always raises NotImplementedError (ISO 76, (source to find))."""
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one roller at the inner and outer ring contact [N].

        Computed from this instance's own ``Ca``; ISO_TS_16281_2008, 5.3.1.3,
        Eq. (50)-(51); equations in the class Notes.
        """
        Q_ci = (1.0 / self.lambda_v) * (self.Ca / self._denom) * (
            1.0 + self._numer ** (9.0 / 2.0)) ** (2.0 / 9.0)
        
        Q_ce = (1.0 / self.lambda_v) * (self.Ca / self._denom) * (
            1.0 + self._numer ** (-9.0 / 2.0)) ** (2.0 / 9.0)
        return Q_ci, Q_ce

    @property
    def per_lamina(self) -> tuple[float, float]:
        """Load capacities (q_ci, q_ce) of one lamina of a roller [N].

        ``Q * (1 / n_s)^(7/9)``, ISO_TS_16281_2008, 5.3.2, Eq. (56)-(57).
        Reads ``self.n_s``, which this class never sets: raises AttributeError.
        """
        Q_ci, Q_ce = self.Q_elements
        q_ci = Q_ci * (1.0 / self.n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / self.n_s) ** (7.0 / 9.0)
        return q_ci, q_ce


class LineContactCapacityThrust_90deg(ThrustCapacityCalculator, _MultirowCombinableLineContact):
    """Basic dynamic axial load rating and element capacities of a single-row thrust roller bearing, contact angle 90 deg.

    Used by ThrustCylindricalRollerFamily when alpha_0 is 90 deg, and by
    RollerThrustMultiRowFamily for each such row (``combine_multirow``).

    Parameters
    ----------
    Z : int
        Number of rollers [-].
    Dwe : float
        Mean roller diameter [mm].
    Lwe : float
        Effective contact length of the roller [mm].
    alpha_0 : float
        Initial contact angle [rad]. Stored; it appears in no equation of this
        class.
    gamma : float
        Auxiliary parameter Dwe/Dpw [-], in (0, 1); used as passed.
    lambda_v : float
        Product lambda*nu [-] (0.73 for the thrust roller bearing).
    eta : float
        Reduction factor eta for the unequal load distribution among the
        rollers [-].
    i : int, optional
        Number of rows [-]. Validated and stored; it appears in no equation of
        this class.

    Raises
    ------
    ValueError
        If Z, Dwe, Lwe or i is not positive, or if gamma is not in (0, 1).

    Notes
    -----
    Equations as coded (Dwe, Lwe in mm, Ca and Q in N)::

        f_c  = 472.45388 * lambda_v * eta * gamma^(2/9)
        Ca   = f_c Lwe^(7/9) Z^(3/4) Dwe^(29/27)
        Q_ci = Q_ce = (1/lambda_v) (Ca / Z) 2^(2/9)

    Code names and symbols of the standards: lambda_v = lambda*nu, eta = eta,
    Dwe = D_we, Lwe = L_we. 472.45388 is "0,41 B_1" (Ca in N, Dwe in mm).

    Warnings
    --------
    Ca is computed as f_c (...) without a rating factor b_m, the form printed
    in ISO_TR_1281, 6.6.2 (Formula (41)); Harris2007_vol1, Eq. 11.172 and
    Table 11.11 print C = b_m f_c (...). ISO 281:2007 is not in the library
    ((source to find)).

    ``per_lamina`` reads ``self.n_s``, which this class neither takes nor
    stores: it raises AttributeError.

    References
    ----------
    .. [1] ISO_TR_1281, 6.6.2, Formulae (41)-(42); Table 2 (lambda*nu, eta).
    .. [2] ISO_TS_16281_2008, 5.3.1.4, Eq. (53)-(54), (55).
    """

    # 0,41*B_1 [N, Dwe in mm]: ISO_TR_1281, 6.6.2, sentence after Formula (42)
    _B1_041_N  = 472.45388   

    def __init__(self, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                 gamma: float, lambda_v: float, eta: float, i: int = 1):
        """Store the arguments; see the class docstring (it states what is validated)."""
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0 or i <= 0:
            raise ValueError(f"Z, Dwe, Lwe and i must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dwe, self.Lwe, self.alpha_0 = Z, Dwe, Lwe, alpha_0
        self.gamma, self.lambda_v, self.eta, self.i = gamma, lambda_v, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        """Rating factor f_c of ISO_TR_1281, Formula (42), including lambda*nu and eta."""
        if "_fc" not in self._cache:

            self._cache["_fc"] = (self._B1_041_N * self.lambda_v * self.eta
                                   * self.gamma ** (2.0 / 9.0))
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        """Basic dynamic axial load rating Ca [N] of the thrust roller bearing, alpha = 90 deg.

        ISO_TR_1281, 6.6.2, Formula (41). No rating factor b_m (see the class
        Warnings).
        """
        return self._fc * self.Lwe ** (7.0 / 9.0) * self.Z ** (3.0 / 4.0) * self.Dwe ** (29.0 / 27.0) 

    @property
    def C0(self) -> float:
        """Basic static axial load rating [N]; not implemented, always raises NotImplementedError (ISO 76, (source to find))."""
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one roller, equal at both contacts [N].

        ISO_TS_16281_2008, 5.3.1.4, Eq. (53)-(54): (1/(lambda*nu)) (Ca/Z) 2^(2/9).
        """
        Q = (1.0 / self.lambda_v) * (self.Ca / self.Z) * 2.0 ** (2.0 / 9.0)
        return Q, Q

    @property
    def per_lamina(self) -> tuple[float, float]:
        """Load capacities (q_ci, q_ce) of one lamina of a roller [N].

        ``Q * (1 / n_s)^(7/9)``, ISO_TS_16281_2008, 5.3.2, Eq. (56)-(57).
        Reads ``self.n_s``, which this class never sets: raises AttributeError.
        """
        Q_ci, Q_ce = self.Q_elements
        q_ci = Q_ci * (1.0 / self.n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / self.n_s) ** (7.0 / 9.0)
        return q_ci, q_ce