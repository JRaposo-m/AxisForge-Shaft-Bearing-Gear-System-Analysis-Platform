"""Contact spring constants of ISO/TS 16281 for rolling bearings.

Point contact (ball between the two raceways): spring constant cp from the
Hertzian deflection of ISO/TS 16281, 4.2.1. Line contact (roller): spring
constant cl, lamina constant cs and reference roller profile P(x_k) of 5.2.
The bearing families evaluate these once at assembly and store the values on
the bearing; the load-distribution solvers only read them.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# ---- shared low-level geometry helpers (same across radial/thrust) --
# ---------------------------------------------------------------------

def _chi_equation(chi: float, F_rho: float) -> float:
    """Residual of the equation that defines the contact-ellipse axis ratio.

    Parameters
    ----------
    chi : float
        Ratio of semi-major to semi-minor axis of the contact ellipse [-].
        chi = 1 divides by zero.
    F_rho : float
        Relative curvature difference F(rho) of the contact [-].

    Returns
    -------
    float
        1 - 2/(chi^2 - 1) * (K/E - 1) - F_rho [-]; zero at the solution.

    Notes
    -----
    K and E are the complete elliptic integrals of the first and second kind,
    Eq. (3) and (4). They are evaluated with scipy.special.ellipk and ellipe,
    which take the parameter m = 1 - 1/chi^2 (not the modulus).

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.2.1, Eq. (2)-(4).
    """
    m = 1.0 - 1.0 / chi**2
    return 1.0 - (2.0 / (chi**2 - 1.0)) * (ellipk(m) / ellipe(m) - 1.0) - F_rho


def _chi(F_rho: float) -> float:
    """Axis ratio chi of the contact ellipse for a given curvature difference.

    Parameters
    ----------
    F_rho : float
        Relative curvature difference F(rho) of the contact [-].

    Returns
    -------
    float
        chi [-], semi-major over semi-minor axis: root of ``_chi_equation``.

    Raises
    ------
    ValueError
        From scipy.optimize.brentq, when the residual has the same sign at
        both ends of the bracket (root outside [1.0001, 1000]).

    Notes
    -----
    The standard gives only the equation. The bracket [1.0001, 1000] and the
    default tolerances of brentq are choices of this code.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.2.1, Eq. (2).
    """
    return brentq(_chi_equation, 1.0001, 1000.0, args=(F_rho,))

# ---------------------------------------------------------------------
# ---- point contact --------------------------------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PointContactStiffness:
    """Spring constant of a ball between the two raceways (Hertzian point contact).

    Load-deflection relation of one ball, Q = cp * delta^(3/2), where delta is
    the total elastic deflection of the inner and the outer contact together.
    The ball-bearing families build this object at assembly and store ``cp``
    and ``raceway_contact_radius`` (as ``Ri``) on the bearing.

    Parameters
    ----------
    Dw : float
        Ball diameter [mm].
    Dpw : float
        Pitch diameter of the ball set [mm].
    gamma : float
        Auxiliary parameter Dw*cos(contact angle)/Dpw [-]. Used as passed; it
        is not checked against Dw, Dpw and alpha_0 (see Warnings).
    ri : float
        Cross-sectional groove radius of the inner raceway [mm].
    re : float
        Cross-sectional groove radius of the outer raceway [mm].
    alpha_0 : float
        Initial contact angle [rad]: the "initial contact angle" of
        ISO/TS 16281, not the nominal contact angle of ISO 281. Used only in
        ``raceway_contact_radius``.
    e1 : float
        Modulus of elasticity of the rolling element [MPa].
    e2 : float
        Modulus of elasticity of the raceways, both rings [MPa].
    nu1 : float
        Poisson's ratio of the rolling element [-].
    nu2 : float
        Poisson's ratio of the raceways [-].

    Warnings
    --------
    Differs from ISO_TS_16281_2008, clause 3: the standard defines
    gamma = Dw*cos(alpha)/Dpw with the nominal contact angle alpha. This class
    takes ``gamma`` as an argument. The AxisForge families pass
    gamma = Dw*cos(alpha_0)/Dpw with alpha_0 the initial contact angle, except
    at alpha_0 = 90 deg (thrust ball bearing), where they pass Dw/Dpw instead
    of 0. cp is one constant per bearing and is not re-evaluated at the
    operating contact angle alpha_j of each ball. ISO_TR_1281 (Part 1), 6.3.2
    prints gamma = Dw/Dpw for alpha = 90 deg, but only in the load-rating
    factor f_c (Formula (22)), whose raceway-curvature term has no gamma. Here
    that value enters Eq. (5)-(8) as a raceway curvature in the rolling
    direction, although the thrust family sets that radius to infinity.

    Differs from ISO_TS_16281_2008, Eq. (11): generalised to two materials.
    The standard has one material, factor E/(1 - nu_E^2). The code uses
    e_prime = 2 / ((1 - nu1^2)/e1 + (1 - nu2^2)/e2), which gives the same value
    when both materials are equal.

    Notes
    -----
    Equations as coded, with g = gamma::

        sum_rho_i = (2/Dw) * (2 + g/(1 - g) - Dw/(2*ri))               Eq. (5)
        sum_rho_e = (2/Dw) * (2 - g/(1 + g) - Dw/(2*re))               Eq. (6)
        F_i = (g/(1 - g) + Dw/(2*ri)) / (2 + g/(1 - g) - Dw/(2*ri))    Eq. (7)
        F_e = (-g/(1 + g) + Dw/(2*re)) / (2 - g/(1 + g) - Dw/(2*re))   Eq. (8)
        chi_i, chi_e = roots of Eq. (2) for F_i, F_e
        T_i = K(chi_i) * cbrt(sum_rho_i / (chi_i^2 * E(chi_i)))
        T_e = K(chi_e) * cbrt(sum_rho_e / (chi_e^2 * E(chi_e)))
        cp  = 1.48 * e_prime * (T_i + T_e)^(-3/2)                      Eq. (11)
        Ri  = Dpw/2 + (ri - Dw/2) * cos(alpha_0)                       Eq. (14)

    Code names and symbols of the standard: Dw = D_w, Dpw = D_pw, ri = r_i,
    re = r_e, curvature_sum_* = sum of rho, curvature_diff_* = F(rho),
    chi_* = chi, cp = c_P, raceway_contact_radius = R_i.

    The factor 1.48 is the value printed in Eq. (11). The deflection follows
    Hertzian theory (4.2.1); the standard treats the bearing statically and
    neglects centripetal and gyroscopic forces (4.1).

    No property is cached: each access to ``cp`` solves Eq. (2) twice.

    References
    ----------
    .. [1] ISO_TS_16281_2008, clause 3 (symbols); 4.2.1, Eq. (1)-(11); 4.2.2,
       Eq. (14).
    .. [2] Harris2007_vol1, 2.2.4, Eq. 2.27-2.31 (curvature sum and difference
       of ball-raceway contacts); 7.4.2 states that the load-deflection factor
       K_n is a function of the final contact angle (thrust-loaded
       angular-contact bearing).
    """

    Dw: float
    Dpw: float
    gamma: float
    ri: float
    re: float
    alpha_0: float
    e1: float
    e2: float
    nu1: float
    nu2: float

    
    @property
    def raceway_contact_radius(self) -> float:
        """Radius of the locus of the inner raceway groove curvature centres, R_i.

        Distance between the centre of curvature of the inner raceway groove and
        the bearing axis. The ball solver uses it as the lever arm of the tilt.

        Returns
        -------
        float
            Dpw/2 + (ri - Dw/2)*cos(alpha_0) [mm] (ISO_TS_16281_2008, 4.2.2,
            Eq. (14)).
        """
        return self.Dpw / 2.0 + (self.ri - self.Dw / 2.0) * np.cos(self.alpha_0)

    @property
    def curvature_sum_inner(self) -> float:
        """Curvature sum of the contact between ball and inner raceway.

        Returns
        -------
        float
            Sum of rho at the inner contact [1/mm] (ISO_TS_16281_2008, 4.2.1,
            Eq. (5)).
        """
        g = self.gamma
        return (2.0 / self.Dw) * (2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri))

    @property
    def curvature_sum_outer(self) -> float:
        """Curvature sum of the contact between ball and outer raceway.

        Returns
        -------
        float
            Sum of rho at the outer contact [1/mm] (ISO_TS_16281_2008, 4.2.1,
            Eq. (6)).
        """
        g = self.gamma
        return (2.0 / self.Dw) * (2.0 - g / (1.0 + g) - self.Dw / (2.0 * self.re))

    @property
    def curvature_diff_inner(self) -> float:
        """Relative curvature difference of the contact between ball and inner raceway.

        Returns
        -------
        float
            F_i(rho) [-] (ISO_TS_16281_2008, 4.2.1, Eq. (7)).
        """
        g = self.gamma
        num = g / (1.0 - g) + self.Dw / (2.0 * self.ri)
        den = 2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri)
        return num / den

    @property
    def curvature_diff_outer(self) -> float:
        """Relative curvature difference of the contact between ball and outer raceway.

        Returns
        -------
        float
            F_e(rho) [-] (ISO_TS_16281_2008, 4.2.1, Eq. (8)).
        """
        g = self.gamma
        num = -g / (1.0 + g) + self.Dw / (2.0 * self.re)
        den = 2.0 - g / (1.0 + g) - self.Dw / (2.0 * self.re)
        return num / den

    @property
    def chi_inner(self) -> float:
        """Axis ratio of the contact ellipse at the inner raceway.

        Returns
        -------
        float
            chi_i [-], semi-major over semi-minor axis: root of Eq. (2) for
            ``curvature_diff_inner``.
        """
        return _chi(self.curvature_diff_inner)

    @property
    def chi_outer(self) -> float:
        """Axis ratio of the contact ellipse at the outer raceway.

        Returns
        -------
        float
            chi_e [-], semi-major over semi-minor axis: root of Eq. (2) for
            ``curvature_diff_outer``.
        """
        return _chi(self.curvature_diff_outer)

    @property
    def _inner_term(self) -> float:
        """Inner-raceway term of the deflection bracket of Eq. (11).

        Returns
        -------
        float
            K(chi_i) * cbrt(sum_rho_i / (chi_i^2 * E(chi_i))) [mm^(-1/3)].
        """
        xi = self.chi_inner
        mi = 1.0 - 1.0 / xi**2
        Ki, Ei = ellipk(mi), ellipe(mi)
        return Ki * np.cbrt(self.curvature_sum_inner / (xi**2 * Ei))

    @property
    def _outer_term(self) -> float:
        """Outer-raceway term of the deflection bracket of Eq. (11).

        Returns
        -------
        float
            K(chi_e) * cbrt(sum_rho_e / (chi_e^2 * E(chi_e))) [mm^(-1/3)].
        """
        xe = self.chi_outer
        me = 1.0 - 1.0 / xe**2
        Ke, Ee = ellipk(me), ellipe(me)
        return Ke * np.cbrt(self.curvature_sum_outer / (xe**2 * Ee))

    @property
    def cp(self) -> float:
        """Spring constant of one ball for both raceway contacts together, c_P.

        The equation and the two departures from the standard (angle in gamma, two
        materials) are in the class docstring.

        Returns
        -------
        float
            cp [N/mm^1.5]: Q [N] = cp * delta^1.5 with delta [mm] the total
            elastic deflection of both contacts (ISO_TS_16281_2008, 4.2.1,
            Eq. (10)-(11)).
        """
        e_prime = 2.0 / ((1.0 - self.nu1**2)/self.e1 + (1.0 - self.nu2**2)/self.e2)   # = E/(1-nu^2) for equal materials
        return 1.48 * e_prime * (self._inner_term + self._outer_term) ** (-3.0 / 2.0)


class SelfAligningPointContactStiffness(PointContactStiffness):
    """Placeholder for a ball on a spherical outer raceway (self-aligning bearing).

    Same fields and inner-raceway terms as ``PointContactStiffness``. The
    outer-raceway term is not implemented: ``_outer_term`` raises, and so does
    ``cp``. No AxisForge family builds this class: the code of
    ``SelfAligningBallFamily`` calls the plain ``PointContactStiffness`` (Eq. (6)
    and (8)) with the outer radius re = 0.5*(1 + 1/gamma)*Dw of the family
    (Eq. (78)). That call is not reached today (``SelfAligningBallFamily.
    assemble_geometry`` uses ri, re before assigning them). If reached, Eq. (8)
    gives F_e = 0, i.e. chi_e = 1 (circular contact), outside the bracket of
    ``_chi``: brentq raises ValueError.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 6.6, Eq. (78); 4.2.1, Eq. (6), (8).
       Closed form of the outer term for chi_e = 1: (source to find).
    """

    @property
    def _outer_term(self) -> float:
        """Outer-raceway term for a spherical outer raceway: not implemented.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "Closed-form circular-contact (chi_e=1) outer-race term not "
            "provided yet -- see class docstring."
        )


# ---------------------------------------------------------------------
# ---- Line contact ---------------------------------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class LineContactStiffness:
    """Spring constants of a roller (line contact) and reference roller profile.

    Load-deflection relation of one roller, Q = cl * delta^(10/9), with delta
    the total deflection of both raceway contacts, and its lamina form
    q = cs * delta^(10/9) for one of n_s equal slices of the roller.
    ``CylindricalRollerFamily`` builds this object at assembly and stores
    ``cl`` (as ``cL``), ``cs`` and the profile (as ``P_xk``) on the bearing.

    Parameters
    ----------
    Lwe : float
        Effective roller length [mm].
    n_s : float
        Number of laminae [-]. Not checked here.
    Dwe : float
        Roller diameter [mm].
    x_k : np.ndarray
        Position of the centre of each lamina along the roller axis, measured
        from the roller mid-length [mm].
    _LOG_ARG_EPS : float
        Lower bound of the logarithm argument in the profile [-]. The family
        passes 1e-12.

    Warnings
    --------
    Steel only: the standard gives the constant 35948 "for contacting parts
    made of steel". No modulus of elasticity enters and the material is not
    checked.

    ``ThrustCylindricalRollerFamily`` calls this class with another signature
    (Dwe, Dpw, alpha_0, Lwe, n_s) and reads ``gamma`` and ``lamina_positions``,
    which do not exist here: that family cannot be assembled today.

    Differs from ISO_TS_16281_2008, clause 3: the unit of c_s. The standard
    prints c_L "to the power of 10/9" and c_s "newtons per millimetre to the
    power of 8/9" (review item).

    Notes
    -----
    Equations as coded::

        cl = 35948 * Lwe^(8/9)        Eq. (35), Lwe in mm
        cs = cl / n_s                 Eq. (37)

    Code names and symbols of the standard: cl = c_L, cs = c_s, Lwe = L_we,
    Dwe = D_we, reference_roller_profile = P(x_k). cl is in [N/mm^(10/9)]
    (clause 3). cs is used in [N/mm^(10/9)] too, the unit that follows from
    Eq. (36)-(37); clause 3 prints "newtons per millimetre to the power of
    8/9" for c_s.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.2.1, Eq. (34)-(35); 5.2.2, Eq. (36)-(37);
       5.2.3, Eq. (42)-(44).
    """

    Lwe: float
    n_s: float
    Dwe: float
    x_k: np.ndarray
    _LOG_ARG_EPS: float

    @property
    def cl(self) -> float:
        """Spring constant of one roller for both raceway contacts together, c_L.

        Returns
        -------
        float
            35948 * Lwe^(8/9) [N/mm^(10/9)], Lwe in mm (ISO_TS_16281_2008, 5.2.1,
            Eq. (35)); steel only.
        """
        return 35948.0 * self.Lwe ** (8.0 / 9.0)

    @property
    def cs(self) -> float:
        """Spring constant of one roller lamina, c_s.

        Returns
        -------
        float
            cl / n_s [N/mm^(10/9)] (ISO_TS_16281_2008, 5.2.2, Eq. (37)).
        """
        return self.cl / self.n_s

    @property
    def reference_roller_profile(self) -> np.ndarray:
        """Reference profile of the roller: profile depth P(x_k) at each lamina.

        Returns
        -------
        np.ndarray
            P(x_k) [mm], same shape as ``x_k``; zero at the roller mid-length and
            increasing toward the roller ends.

        Notes
        -----
        Equations as coded::

            Lwe <= 2.5*Dwe:
                P = 0.00035 * Dwe * ln(1 / (1 - (2*x_k/Lwe)^2))            Eq. (42)
            Lwe > 2.5*Dwe, with c = Lwe - 2.5*Dwe:
                P = 0                              for |x_k| <= c/2        Eq. (43)
                P = 0.0005 * Dwe
                    * ln(1 / (1 - ((2*|x_k| - c) / (2.5*Dwe))^2))  otherwise  Eq. (44)

        The argument 1 - (...)^2 is floored at ``_LOG_ARG_EPS`` before the
        logarithm. This floor is not in the standard; it keeps P finite when a
        lamina centre is at the roller end. The profile is even in x_k.

        The standard gives these profiles as approximate values for steel rollers
        and usual applications. The roller solver subtracts 2*P(x_k) from the
        lamina approach (Eq. (41)).

        References
        ----------
        .. [1] ISO_TS_16281_2008, 5.2.3, Eq. (42)-(44).
        """
        P = np.zeros_like(self.x_k)
        if self.Lwe <= 2.5 * self.Dwe:
            arg = 1.0 - (2.0 * self.x_k / self.Lwe) ** 2
            arg = np.maximum(arg, self._LOG_ARG_EPS)
            P = 0.000350 * self.Dwe * np.log(1.0 / arg)
        else:
            half_flat = (self.Lwe - 2.5 * self.Dwe) / 2.0
            edge = np.abs(self.x_k) > half_flat
            if np.any(edge):
                xe = self.x_k[edge]
                arg = 1.0 - ((2.0 * np.abs(xe) - (self.Lwe - 2.5 * self.Dwe)) / (2.5 * self.Dwe)) ** 2
                arg = np.maximum(arg, self._LOG_ARG_EPS)
                P[edge] = 0.000500 * self.Dwe * np.log(1.0 / arg)
        return P    