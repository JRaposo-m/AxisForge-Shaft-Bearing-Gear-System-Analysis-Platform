"""Contact geometry of a rolling element between its inner and outer raceway.

Radii of curvature of the rolling element (solid a) and of the raceway
(solid b) in the rolling direction (x) and in the transverse direction (y),
for ball and cylindrical roller bearings, with gamma (read by the capacity
and contact-stiffness calculations), the groove ratios and the free end play
(stored on the Bearing; no AxisForge calculation reads them). Notation of
Harris2007_vol1, chapter 2.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np



# ---------------------------------------------------------------------
# ---- shared low-level geometry helpers (same across radial/thrust) --
# ---------------------------------------------------------------------

def contact_angle_and_clearance(A: float, *, s: float | None = None,
                                 alpha_0_deg: float | None = None) -> tuple[float, float]:
    """Initial contact angle and diametral clearance of a ball bearing, one from the other.

    Exactly one of ``s`` and ``alpha_0_deg`` must be given; the other is
    computed. Shared by the ball families: DeepGrooveBallFamily gives ``s``,
    AngularContactFamily, SelfAligningBallFamily and ThrustBallSingleRowFamily
    give ``alpha_0_deg``.

    Parameters
    ----------
    A : float
        Distance between the curvature centres of the two raceway grooves
        [mm], A = ri + re - Dw.
    s : float, optional
        Diametral internal clearance [mm] (measured on the diameter).
    alpha_0_deg : float, optional
        Initial (free) contact angle [deg].

    Returns
    -------
    alpha_0 : float
        Initial contact angle [rad].
    s : float
        Diametral internal clearance [mm].

    Raises
    ------
    ValueError
        If both or neither of ``s`` and ``alpha_0_deg`` are given.

    Notes
    -----
    Equations as coded::

        alpha_0 = arccos(1 - s / (2 A))          given s
        alpha_0 = radians(alpha_0_deg)           given the angle in degrees
        s       = 2 A (1 - cos(alpha_0))         given the angle

    Code names and symbols of the sources: s = P_d (diametral clearance),
    alpha_0 = alpha deg (free contact angle) of Harris, A = A. The third
    equation is the inverse of the first and is not printed in the sources.
    The range of ``s`` (``s <= 4 A`` for a real angle) is not checked here.

    References
    ----------
    .. [1] Harris2007_vol1, 2.2.2, Eq. 2.9 (alpha deg = arccos(1 - P_d/(2A))).
    .. [2] ISO_TS_16281_2008, 4.2.2 (alpha_0 = arccos[1 - s/(2A)], text before
       Eq. (12); s is the "diametrically measured radial operating clearance").
    """
    if (s is None) == (alpha_0_deg is None):
        raise ValueError("Exactly one of 's' or 'alpha_0_deg' must be provided, not both or neither.")
    if s is not None:
        return np.arccos(1.0 - s / (2.0 * A)), s
    alpha_0 = np.radians(alpha_0_deg)
    return alpha_0, 2.0 * A * (1.0 - np.cos(alpha_0))


# ---------------------------------------------------------------------
# ---- Geometry definition -------------------------------------
# ---------------------------------------------------------------------

class Geometry(ABC):
    """Radii of curvature and geometry parameters of one rolling element between its raceways.

    Contract for the geometry of a rolling element (solid a) and of its inner
    and outer raceway (solid b). An infinite radius (``float("inf")``) means a
    flat or straight surface in that direction, e.g. the transverse radius of a
    straight cylindrical roller.

    Notes
    -----
    Notation: a = rolling element, b = raceway, x = rolling direction,
    y = transverse direction (groove cross-section), as in Harris2007_vol1,
    chapter 2. All radii are in [mm].

    Conventions as coded (all subclasses):

    - r_ax, r_ay and the rolling-direction raceway radii r_bx_inner,
      r_bx_outer are positive magnitudes, also at the outer raceway.
    - The transverse raceway radii r_by_inner, r_by_outer of the ball are
      negative (concave groove): ``-ri``, ``-re``. For the roller they are
      ``+inf``.
    - All radii are evaluated at the initial contact angle alpha_0 only, not at
      the operating angle of each rolling element.

    Warnings
    --------
    Differs from Harris2007_vol1, 2.2.4: Harris keeps every radius positive
    (Eq. 2.24, "the radius of curvature is always positive") and puts the sign
    on the curvature rho = 1/r, convex positive and concave negative; at the
    outer raceway both curvatures are negative. Here the sign is on the radius
    for the transverse direction only, and the outer-raceway rolling-direction
    radius is positive. A consumer that needs signed curvatures must convert
    them itself.

    References
    ----------
    .. [1] Harris2007_vol1, 2.2.4, Eq. 2.24 (sign convention of curvature).
    """

    @property
    @abstractmethod
    def gamma(self) -> float:
        """Auxiliary parameter gamma = (rolling-element diameter x cos(contact angle)) / pitch diameter [-]."""
        ...

    @property
    @abstractmethod
    def free_end_play(self) -> float:
        """Free end play, relative axial movement of the rings under zero load [mm]."""
        ...

    @property
    @abstractmethod
    def f_i(self) -> float:
        """Inner raceway groove radius over rolling-element diameter, r/D [-]."""
        ...

    @property
    @abstractmethod
    def f_o(self) -> float:
        """Outer raceway groove radius over rolling-element diameter, r/D [-]."""
        ...

    @property
    @abstractmethod
    def r_ax(self) -> float:
        """Radius of the rolling element in the rolling direction [mm]."""
        ...

    @property
    @abstractmethod
    def r_ay(self) -> float:
        """Radius of the rolling element in the transverse direction [mm]."""
        ...

    @property
    @abstractmethod
    def r_bx_inner(self) -> float:
        """Inner-raceway radius in the rolling direction [mm]."""
        ...

    @property
    @abstractmethod
    def r_by_inner(self) -> float:
        """Inner-raceway radius in the transverse direction [mm]."""
        ...

    @property
    @abstractmethod
    def r_bx_outer(self) -> float:
        """Outer-raceway radius in the rolling direction [mm]."""
        ...

    @property
    @abstractmethod
    def r_by_outer(self) -> float:
        """Outer-raceway radius in the transverse direction [mm]."""
        ...

# ---------------------------------------------------------------------
# ---- Ball Bearing Geometry definition -------------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BallBearingGeometry(Geometry):
    """Geometry of a ball between the inner and the outer raceway groove.

    Radii of curvature of the ball and of both grooves at the initial contact
    angle, gamma, groove ratios and free end play. Used by the radial and thrust
    ball families.

    Parameters
    ----------
    Dw : float
        Ball diameter [mm].
    Dpw : float
        Pitch diameter of the ball set [mm].
    A : float
        Distance between the curvature centres of the two grooves [mm],
        A = ri + re - Dw. Used only by ``free_end_play``; not checked against
        ``ri``, ``re`` and ``Dw``.
    s : float
        Diametral internal clearance [mm]. Used only by ``free_end_play``; not
        checked against ``A`` and ``alpha_0``.
    ri : float
        Cross-sectional radius of the inner raceway groove [mm].
    re : float
        Cross-sectional radius of the outer raceway groove [mm].
    alpha_0 : float
        Initial contact angle [rad].

    Notes
    -----
    Equations as coded::

        gamma       = Dw cos(alpha_0) / Dpw      (Dw / Dpw if abs(alpha_0 - pi/2) <= about 1.6e-5 rad: np.isclose with atol 1e-9 and the default rtol 1e-5)
        P_e         = sqrt(4 A s - s^2)                                   [mm]
        f_i, f_o    = ri / Dw, re / Dw                                    [-]
        r_ax = r_ay = Dw / 2                                              [mm]
        r_bx_inner  = (Dpw - Dw cos(alpha_0)) / (2 cos(alpha_0))          [mm]
        r_bx_outer  = (Dpw + Dw cos(alpha_0)) / (2 cos(alpha_0))          [mm]
        r_by_inner  = -ri,   r_by_outer = -re                             [mm]

    Code names and symbols of Harris2007_vol1: Dw = D, Dpw = d_m (pitch
    diameter), ri, re = r_i, r_o (groove radii), f_i, f_o = f_i, f_o,
    alpha_0 = alpha (contact angle), P_e = P_e (free endplay), s = P_d,
    r_bx_inner = r_II1 at the inner raceway, r_bx_outer = r_II1 at the outer
    raceway, r_by_* = r_II2, r_ax = r_ay = r_I1 = r_I2.
    The free end play equals Harris Eq. 2.12, P_e = 2 A sin(alpha_0), after the
    angle is eliminated with Eq. 2.9 (algebraic substitution).

    Warnings
    --------
    Differs from ISO_TS_16281_2008, clause 3 and ISO_TR_1281, clause 4: both
    define gamma with the nominal contact angle alpha; the code uses the
    initial angle alpha_0. At alpha_0 = 90 deg the code uses Dw/Dpw, the form
    ISO_TR_1281, 6.3.2 prints (gamma = D_w/D_pw); ISO_TS_16281_2008, clause 3
    prints only D_w cos(alpha)/D_pw.

    Differs from Harris2007_vol1, 2.2.4, Eq. 2.24: Harris keeps r_II2 = f D
    positive and gives the curvature rho_II2 = -1/(f D); here
    ``r_by_inner = -ri`` and ``r_by_outer = -re`` carry the sign on the radius.
    The outer-raceway rolling-direction radius is positive here, while the
    curvature rho_II1 of Harris is negative there. A consumer that needs signed
    curvatures must convert. Radii are evaluated at alpha_0 only.

    At alpha_0 = pi/2 (thrust ball bearing) ``r_bx_inner`` and ``r_bx_outer``
    divide by cos(pi/2) = 6.1e-17 and evaluate to about 1e17 mm, finite and
    not inf. ThrustBallSingleRowFamily replaces them by ``float("inf")`` in the
    ``r_inner``, ``r_outer`` it stores; that is exact only at 90 deg.

    References
    ----------
    .. [1] Harris2007_vol1, 2.2.1, Eq. 2.5 (r = f D); 2.2.2, Eq. 2.9, 2.12;
       2.2.4, Eq. 2.24, 2.27.
    .. [2] ISO_TS_16281_2008, clause 3 (gamma); ISO_TR_1281, clause 4 and
       6.3.2 (gamma).
    """
    Dw: float
    Dpw: float
    A: float
    s: float
    ri: float
    re: float
    alpha_0: float


    @property
    def gamma(self) -> float:
        """Auxiliary parameter gamma = Dw cos(alpha_0) / Dpw [-] (Dw / Dpw at alpha_0 = 90 deg)."""
        if np.isclose(self.alpha_0, np.pi / 2, atol=1e-9):
            return self.Dw / self.Dpw
        return self.Dw * np.cos(self.alpha_0) / self.Dpw

    @property
    def free_end_play(self) -> float:
        """Free end play P_e = sqrt(4 A s - s^2) [mm]."""
        return (4 * self.A * self.s - self.s**2)**0.5
    @property
    def f_i(self) -> float:
        """Inner groove radius over ball diameter, ri / Dw [-]."""
        return self.ri / self.Dw
    @property
    def f_o(self) -> float:
        """Outer groove radius over ball diameter, re / Dw [-]."""
        return self.re / self.Dw

    @property
    def r_ax(self) -> float:
        """Ball radius in the rolling direction, Dw / 2 [mm]."""
        return self.Dw/2.0

    @property
    def r_ay(self) -> float:
        """Ball radius in the transverse direction, Dw / 2 [mm]."""
        return self.Dw/2.0

    @property
    def r_bx_inner(self) -> float:
        """Inner-raceway radius in the rolling direction at alpha_0 [mm], positive.

        (Dpw - Dw cos(alpha_0)) / (2 cos(alpha_0)).
        """
        return (self.Dpw - self.Dw * np.cos(self.alpha_0)) / (2 * np.cos(self.alpha_0))

    @property
    def r_by_inner(self) -> float:
        """Inner-groove radius in the transverse direction, -ri [mm], negative (concave)."""
        return -1 * self.ri

    @property
    def r_bx_outer(self) -> float:
        """Outer-raceway radius in the rolling direction at alpha_0 [mm], positive.

        (Dpw + Dw cos(alpha_0)) / (2 cos(alpha_0)).
        """
        return (self.Dpw + self.Dw * np.cos(self.alpha_0)) / (2 * np.cos(self.alpha_0))

    @property
    def r_by_outer(self) -> float:
        """Outer-groove radius in the transverse direction, -re [mm], negative (concave)."""
        return -1 * self.re


# ---------------------------------------------------------------------
# ---- Roller Bearing Geometry definition -----------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CylindricalRollerBearingGeometry(Geometry):
    """Geometry of a straight cylindrical roller between the inner and outer raceway.

    Radii of curvature of the roller and of both raceways (rolling direction
    only; transverse radii are infinite), gamma and the lamina positions of the
    ISO/TS 16281 lamina model. Used by CylindricalRollerFamily, whose contact
    angle is zero.

    Parameters
    ----------
    Dwe : float
        Mean roller diameter [mm].
    Dpw : float
        Pitch diameter of the roller set [mm].
    Lwe : float
        Effective contact length of the roller [mm].
    n_s : int
        Number of laminae the roller is divided into [-].

    Notes
    -----
    Equations as coded::

        gamma         = Dwe / Dpw                                   [-]
        x_k           = (Lwe / n_s) (k + 1/2) - Lwe / 2,  k = 0 .. n_s - 1   [mm]
        r_ax          = Dwe / 2                                     [mm]
        r_ay          = r_by_inner = r_by_outer = inf               [mm]
        r_bx_inner    = (Dpw - Dwe) / 2                             [mm]
        r_bx_outer    = (Dpw + Dwe) / 2                             [mm]

    Code names and symbols: Dwe = D_we, Lwe = L_we, Dpw = D_pw, n_s = n_s,
    x_k = x_k (ISO_TS_16281_2008, clause 3: distance between the centre of
    lamina k and the roller centre). ISO_TS_16281_2008, 5.2.2 divides the roller
    into n_s identical laminae (at least 30) but prints no formula for x_k; the
    midpoint formula is the code's. The radii are the rolling-direction radii at
    zero contact angle; ``float("inf")`` for the transverse radii follows
    Harris2007_vol1, 2.4.2 ("R = r_i = r_o => infinity").

    Warnings
    --------
    gamma = Dwe/Dpw has no cos(alpha). ISO_TR_1281, clause 4 prints
    D_we cos(alpha)/D_pw for alpha != 90 deg and D_we/D_pw for alpha = 90 deg.
    The two forms coincide for the radial roller bearing (alpha = 0), the only
    case in which this class is used; ThrustCylindricalRollerFamily does not use
    it.

    The roller is a straight cylinder: ``r_ay`` is ``inf``, there is no crowning
    or end relief. Radii are positive magnitudes with the
    rolling-direction outer-raceway radius positive, and are evaluated at zero
    contact angle (see the Geometry class docstring).

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.2.2 (laminae); clause 3 (x_k).
    .. [2] ISO_TR_1281, clause 4 (gamma).
    .. [3] Harris2007_vol1, 2.4.2, Eq. 2.42-2.43 (roller-raceway curvature).
    """

    Dwe: float
    Dpw: float
    Lwe: float
    n_s: int

    @property
    def gamma(self) -> float:
        """Auxiliary parameter gamma = Dwe / Dpw [-]."""
        return self.Dwe / self.Dpw

    @property
    def lamina_positions(self) -> np.ndarray:
        """Positions x_k of the lamina centres from the roller mid-length [mm].

        Midpoints of n_s equal laminae, strictly inside (-Lwe/2, Lwe/2):
        ``x_k = (Lwe / n_s) (k + 1/2) - Lwe / 2``, k = 0 .. n_s - 1.
        ISO_TS_16281_2008, 5.2.2 (laminae).
        """
        lamina_length = self.Lwe / self.n_s
        return lamina_length * (np.arange(self.n_s) + 0.5) - self.Lwe / 2.0
    
    @property
    def free_end_play(self) -> float:
        """Free end play [mm]; not implemented, always raises NotImplementedError."""
        raise NotImplementedError(
            "CylindricalRollerGeometry.free_end_play not implemented yet -- "
            "cylindrical roller bearings don't have a groove-conformity clearance "
            "the same way ball bearings do; radial clearance is given directly "
            "by tolerance class (C0-C5), not derived here."
        )

    @property
    def f_i(self) -> float:
        """Inner groove ratio [-]; not implemented, always raises NotImplementedError."""
        raise NotImplementedError(
            "CylindricalRollerGeometry.f_i not implemented yet"
        )


    @property
    def f_o(self) -> float:
        """Outer groove ratio [-]; not implemented, always raises NotImplementedError."""
        raise NotImplementedError(
            "CylindricalRollerGeometry.f_o not implemented yet"
        )

    @property
    def r_ax(self):
        """Roller radius in the rolling direction, Dwe / 2 [mm]."""
        return self.Dwe/2

    @property
    def r_ay(self):
        """Roller radius in the transverse direction [mm]; inf (straight generatrix, no crowning)."""
        return float("inf")

    @property
    def r_bx_inner(self) -> float:
        """Inner-raceway radius in the rolling direction, (Dpw - Dwe) / 2 [mm], positive."""
        return (self.Dpw - self.Dwe) / 2

    @property
    def r_by_inner(self) -> float:
        """Inner-raceway radius in the transverse direction [mm]; inf (flat raceway)."""
        return float("inf")

    @property
    def r_bx_outer(self) -> float:
        """Outer-raceway radius in the rolling direction, (Dpw + Dwe) / 2 [mm], positive."""
        return (self.Dpw + self.Dwe) / 2

    @property
    def r_by_outer(self) -> float:
        """Outer-raceway radius in the transverse direction [mm]; inf (flat raceway)."""
        return float("inf")