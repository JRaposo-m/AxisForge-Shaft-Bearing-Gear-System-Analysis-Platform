"""
core/machine_elements/Bearings/families/ball/thrust/functions/contact_stiffness.py

Point-contact ("ball") Hertz contact math -- ISO/TS 16281 Sec 5 eq.(2)-(11).
Pure functions, no state, no catalog/family knowledge, no table-specific
constants (those are subtype data -- see ../subtypes/).

Byte-identical to ../../radial/functions/contact_stiffness.py -- the Hertz
math is duty-agnostic (gamma() already special-cases alpha_0=90deg, i.e.
pure thrust, since it was first written). Duplicated rather than imported
across the radial/thrust boundary to keep each duty folder self-contained
(subtypes/ imports via `from ..functions import contact_stiffness`, same
relative pattern in both places) -- same reasoning already applied to
subtype-level table constants, extended here to keep the folder structure
uniform. If the two ever need to diverge (they shouldn't, it's textbook
Hertz theory), that's exactly why they're not shared.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq

LOAD_DEFLECTION_EXPONENT = 3.0 / 2.0


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