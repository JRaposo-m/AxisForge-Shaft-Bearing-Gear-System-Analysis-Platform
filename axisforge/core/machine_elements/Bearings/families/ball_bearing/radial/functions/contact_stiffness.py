"""
core/machine_elements/Bearings/families/ball/radial/functions/contact_stiffness.py

Point-contact ("ball") Hertz contact math -- ISO/TS 16281 Sec 5 eq.(2)-(11).
Pure functions, no state, no catalog/family knowledge, no table-specific
constants (those are subtype data -- see ../subtypes/).
"""
from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq

LOAD_DEFLECTION_EXPONENT = 3.0 / 2.0

# NOTE: reference raceway radii (ri, re from Dw) and the reduction factor
# (lambda) are NOT here. They're ISO 281:2007 Table 1 inputs -- one row
# per bearing type (radial contact groove, angular contact groove,
# self-aligning, thrust, ...). That's subtype data, not generic math --
# each subtype in ../subtypes/ declares its own table-row constants and
# calls hertz_spring_constant() below with the ri/re it derived from them.


def contact_angle_and_clearance(A: float, *, s: float | None = None,
                                 alpha_0_deg: float | None = None) -> tuple[float, float]:
    """Exactly one of s / alpha_0_deg required. Returns (alpha_0 [rad], s [mm])."""
    if (s is None) == (alpha_0_deg is None):
        raise ValueError("Exactly one of 's' or 'alpha_0_deg' must be provided, not both or neither.")
    if s is not None:
        return np.arccos(1.0 - s / (2.0 * A)), s
    alpha_0 = np.radians(alpha_0_deg)
    return alpha_0, 2.0 * A * (1.0 - np.cos(alpha_0))


def gamma(Dw: float, Dpw: float, alpha_0: float) -> float:
    """Dw*cos(alpha_0)/Dpw -- cos term dropped at alpha_0=90deg (thrust)."""
    if np.isclose(alpha_0, np.pi / 2, atol=1e-9):
        return Dw / Dpw
    return Dw * np.cos(alpha_0) / Dpw


def raceway_contact_radius(Dpw: float, ri: float, Dw: float, alpha_0: float) -> float:
    """Ri -- inner raceway radius to contact."""
    return Dpw / 2.0 + (ri - Dw / 2.0) * np.cos(alpha_0)


def curvature_sum_inner(Dw: float, ri: float, g: float) -> float:
    """Sum(rho)_i   eq.(5)"""
    return (2.0 / Dw) * (2.0 + g / (1.0 - g) - Dw / (2.0 * ri))


def curvature_sum_outer(Dw: float, re: float, g: float) -> float:
    """Sum(rho)_e   eq.(6)"""
    return (2.0 / Dw) * (2.0 - g / (1.0 + g) - Dw / (2.0 * re))


def curvature_diff_inner(Dw: float, ri: float, g: float) -> float:
    """F_i(rho)   eq.(7)"""
    num = g / (1.0 - g) + Dw / (2.0 * ri)
    den = 2.0 + g / (1.0 - g) - Dw / (2.0 * ri)
    return num / den


def curvature_diff_outer(Dw: float, re: float, g: float) -> float:
    """F_e(rho)   eq.(8)"""
    num = -g / (1.0 + g) + Dw / (2.0 * re)
    den = 2.0 - g / (1.0 + g) - Dw / (2.0 * re)
    return num / den


def _chi_equation(chi: float, F_rho: float) -> float:
    m = 1.0 - 1.0 / chi**2
    return 1.0 - (2.0 / (chi**2 - 1.0)) * (ellipk(m) / ellipe(m) - 1.0) - F_rho


def _chi(F_rho: float) -> float:
    """Solve eq.(2) for chi via brentq."""
    return brentq(_chi_equation, 1.0001, 1000.0, args=(F_rho,))


def hertz_spring_constant(Dw: float, ri: float, re: float, E: float, nu: float,
                           alpha_0: float, Dpw: float) -> float:
    """c_p [N/mm^(3/2)] -- eq.(9)-(11)."""
    g = gamma(Dw, Dpw, alpha_0)
    Sr_i = curvature_sum_inner(Dw, ri, g)
    Sr_e = curvature_sum_outer(Dw, re, g)
    xi = _chi(curvature_diff_inner(Dw, ri, g))
    xe = _chi(curvature_diff_outer(Dw, re, g))

    E_star = E / (1.0 - nu**2)
    mi, me = 1.0 - 1.0 / xi**2, 1.0 - 1.0 / xe**2
    Ki, Ei = ellipk(mi), ellipe(mi)
    Ke, Ee = ellipk(me), ellipe(me)

    bi = Ki * np.cbrt(Sr_i / (xi**2 * Ei))
    be = Ke * np.cbrt(Sr_e / (xe**2 * Ee))
    return 1.48 * E_star * (bi + be) ** (-3.0 / 2.0)

class SelfAligningContactStiffness:
    """
    Special-case Hertz contact math for self-aligning ball bearings.

    Why this exists: ISO 281:2007 Table 1 defines this subtype's outer
    raceway radius as re = 0.5*(1/gamma + 1)*Dw (see
    ../subtypes/self_aligning.py). Substituting that into
    curvature_diff_outer() above gives F_e(rho) == 0 identically, for any
    valid Dw/Dpw/alpha_0 -- not a numerical edge case, an exact algebraic
    cancellation (Dw/(2*re) = gamma/(1+gamma) = -(-gamma/(1+gamma)), so the
    numerator is always exactly zero). Physically: the outer raceway is
    spherical and conforms exactly to the ball curvature, which is the
    whole point of "self-aligning".

    hertz_spring_constant() above solves eq.(2) for chi via brentq assuming
    elliptical contact (chi > 1, bracket [1.0001, 1000]). At F_e(rho)=0 the
    correct root is chi_e=1 -- circular contact, the degenerate limit of
    the ellipse -- which sits exactly on the boundary brentq can't bracket
    (confirmed: brentq raises "f(a) and f(b) must have different signs").

    NOT implemented -- needs the closed-form circular-contact Hertz
    stiffness expression (chi=1 case) for the outer race, combined with the
    normal elliptical-contact solve (unchanged) for the inner race. Left
    stubbed deliberately -- fill in outer_race_spring_term() once you have
    that closed form, then hertz_spring_constant() below just combines it
    with the existing inner-race term.
    """

    @staticmethod
    def outer_race_spring_term(Dw: float, re: float, E: float, nu: float,
                                alpha_0: float, Dpw: float) -> float:
        """
        'be' term (chi_e=1, circular contact) -- NOT implemented. Once
        filled in, this replaces the `be = Ke * np.cbrt(...)` line from the
        general hertz_spring_constant() for this subtype only.
        """
        raise NotImplementedError(
            "Closed-form circular-contact (chi=1) outer-race term not "
            "provided yet -- see SelfAligningContactStiffness docstring."
        )

    @staticmethod
    def hertz_spring_constant(Dw: float, ri: float, re: float, E: float, nu: float,
                               alpha_0: float, Dpw: float) -> float:
        """
        c_p [N/mm^(3/2)] for self-aligning ball bearings -- inner race via
        the normal elliptical solve (same as hertz_spring_constant() above),
        outer race via outer_race_spring_term() (not implemented yet).
        """
        g = gamma(Dw, Dpw, alpha_0)
        Sr_i = curvature_sum_inner(Dw, ri, g)
        xi = _chi(curvature_diff_inner(Dw, ri, g))

        E_star = E / (1.0 - nu**2)
        mi = 1.0 - 1.0 / xi**2
        Ki, Ei = ellipk(mi), ellipe(mi)
        bi = Ki * np.cbrt(Sr_i / (xi**2 * Ei))

        be = SelfAligningContactStiffness.outer_race_spring_term(Dw, re, E, nu, alpha_0, Dpw)
        return 1.48 * E_star * (bi + be) ** (-3.0 / 2.0)