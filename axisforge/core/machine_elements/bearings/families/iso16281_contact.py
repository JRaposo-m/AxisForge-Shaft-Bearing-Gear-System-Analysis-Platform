"""
core/machine_elements/bearings/families/contact.py


Precisa de revisao, mas foi mudado porque como agora vou implementar slipPY
no repositorio que faz a bridge estes calculos exclusivos de iso16281 que sao 
simplificações passam a viver aqui de modo a nao serem atributos dos rolamentos 
que nao faz sentido uma vez que forçavam uma construção mais limitada 

Every contact-stiffness calculation, tagged by physical contact type
(point/line) AND duty (radial/thrust) -- both dimensions matter here,
Hertz math genuinely differs across both, not just across subtypes.

This contact follows the ISO 16281 formulations, for future it may be obtained
in the AxisForge-Design-Studies this is because there slippy will be used
and different approaches for contact may be used
"""
from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq

from axisforge.core.machine_elements.bearings.families.bearing_properties import RadialSurfaces


# ---------------------------------------------------------------------
# ---- shared low-level geometry helpers (same across radial/thrust) --
# ---------------------------------------------------------------------

def contact_angle_and_clearance(A: float, *, s: float | None = None,
                                 alpha_0_deg: float | None = None) -> tuple[float, float]:
    """Exactly one of s / alpha_0_deg required. Returns (alpha_0 [rad], s [mm]).
    Shared by every point-contact radial family -- DeepGrooveBallFamily
    supplies s, AngularContactFamily supplies alpha_0_deg, each family
    just picks which idiom makes sense for it."""
    if (s is None) == (alpha_0_deg is None):
        raise ValueError("Exactly one of 's' or 'alpha_0_deg' must be provided, not both or neither.")
    if s is not None:
        return np.arccos(1.0 - s / (2.0 * A)), s
    alpha_0 = np.radians(alpha_0_deg)
    return alpha_0, 2.0 * A * (1.0 - np.cos(alpha_0))

def _chi_equation(chi: float, F_rho: float) -> float:
    m = 1.0 - 1.0 / chi**2
    return 1.0 - (2.0 / (chi**2 - 1.0)) * (ellipk(m) / ellipe(m) - 1.0) - F_rho


def _chi(F_rho: float) -> float:
    return brentq(_chi_equation, 1.0001, 1000.0, args=(F_rho,))


# ---------------------------------------------------------------------
# ---- point contact / radial duty -------------------------------------
# ---------------------------------------------------------------------

class PointContactStiffness:
    """Hertzian point contact -- ISO/TS 16281 Sec 5 eq.(2)-(11). Works for
    radial AND thrust duty (alpha_0 is just an input). Takes both contacting
    materials directly (E1/nu1, E2/nu2) -- resolving Material/SurfacePair
    objects into these scalars is the caller's job, not this class's."""

    def __init__(self, Dw: float, ri: float, re: float,
                 E1: float, nu1: float, E2: float, nu2: float,
                 alpha_0: float, Dpw: float):
        self.Dw, self.ri, self.re = Dw, ri, re
        self.E1, self.nu1 = E1, nu1
        self.E2, self.nu2 = E2, nu2
        self.alpha_0, self.Dpw = alpha_0, Dpw
        self._cache: dict[str, float] = {}

    @property
    def duty(self) -> str:
        return "thrust" if np.pi/4 < self.alpha_0 <= np.pi/2 else "radial"

    @property
    def gamma(self) -> float:
        if "gamma" not in self._cache:
            if np.isclose(self.alpha_0, np.pi / 2, atol=1e-9):
                self._cache["gamma"] = self.Dw / self.Dpw
            else:
                self._cache["gamma"] = self.Dw * np.cos(self.alpha_0) / self.Dpw
        return self._cache["gamma"]

    @property
    def raceway_contact_radius(self) -> float:
        return self.Dpw / 2.0 + (self.ri - self.Dw / 2.0) * np.cos(self.alpha_0)

    @property
    def curvature_sum_inner(self) -> float:
        g = self.gamma
        return (2.0 / self.Dw) * (2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri))

    @property
    def curvature_sum_outer(self) -> float:
        g = self.gamma
        return (2.0 / self.Dw) * (2.0 - g / (1.0 + g) - self.Dw / (2.0 * self.re))

    @property
    def curvature_diff_inner(self) -> float:
        g = self.gamma
        num = g / (1.0 - g) + self.Dw / (2.0 * self.ri)
        den = 2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri)
        return num / den

    @property
    def curvature_diff_outer(self) -> float:
        g = self.gamma
        num = -g / (1.0 + g) + self.Dw / (2.0 * self.re)
        den = 2.0 - g / (1.0 + g) - self.Dw / (2.0 * self.re)
        return num / den

    @property
    def chi_inner(self) -> float:
        if "chi_i" not in self._cache:
            self._cache["chi_i"] = _chi(self.curvature_diff_inner)
        return self._cache["chi_i"]

    @property
    def chi_outer(self) -> float:
        if "chi_e" not in self._cache:
            self._cache["chi_e"] = _chi(self.curvature_diff_outer)
        return self._cache["chi_e"]

    @property
    def _inner_term(self) -> float:
        xi = self.chi_inner
        mi = 1.0 - 1.0 / xi**2
        Ki, Ei = ellipk(mi), ellipe(mi)
        return Ki * np.cbrt(self.curvature_sum_inner / (xi**2 * Ei))

    @property
    def _outer_term(self) -> float:
        xe = self.chi_outer
        me = 1.0 - 1.0 / xe**2
        Ke, Ee = ellipk(me), ellipe(me)
        return Ke * np.cbrt(self.curvature_sum_outer / (xe**2 * Ee))

    @property
    def cp(self) -> float:
        """c_p [N/mm^(3/2)] -- ISO/TS 16281 eq.(11), Hertz reduced modulus
        generalized to two dissimilar materials. NOT the assembled bearing
        stiffness -- only feeds Q = c_p * delta^(3/2) in the load-
        distribution solver."""
        E_star = 1.0 / ((1.0 - self.nu1**2) / self.E1 + (1.0 - self.nu2**2) / self.E2)
        return 1.48 * E_star * (self._inner_term + self._outer_term) ** (-3.0 / 2.0)

class SelfAligningPointContactStiffness(PointContactStiffness):
    """Specialization of PointContactStiffnessRadial: the spherical
    outer race makes curvature_diff_outer cancel to zero exactly (see
    SelfAligningBallFamily docstring in family.py), which lands the
    correct root on chi_e=1 -- outside the elliptical solve's bracket.
    NOT implemented -- needs the closed-form circular-contact term."""

    @property
    def _outer_term(self) -> float:
        raise NotImplementedError(
            "Closed-form circular-contact (chi_e=1) outer-race term not "
            "provided yet -- see class docstring."
        )
    # .cp is inherited unchanged -- it calls _outer_term, which is the
    # only piece this subtype actually overrides.


# ---------------------------------------------------------------------
# ---- line contact / radial + thrust -- SKETCH ONLY --------------------
# ---------------------------------------------------------------------

class LineContactStiffness:
    CONTACT_TYPE = "line"
    DUTY = "radial"

    def __init__(self, Dwe: float, Dpw: float, alpha_0: float,
                 Lwe: float, n_s: int):

        self.Dwe, self.Dpw, self.alpha_0 = Dwe, Dpw, alpha_0
        self.Lwe, self.n_s = Lwe, n_s
        self._cache: dict[str, float] = {}

    @property
    def lamina_positions(self) -> np.ndarray:
        """x_k -- lamina midpoints, strictly inside (-Lwe/2, Lwe/2). Sec 5.2.2."""
        lamina_length = self.Lwe / self.n_s
        return lamina_length * (np.arange(self.n_s) + 0.5) - self.Lwe / 2.0

    @property
    def gamma(self) -> float:
        if "gamma" not in self._cache:
            if np.isclose(self.alpha_0, np.pi / 2, atol=1e-9):
                self._cache["gamma"] = self.Dwe / self.Dpw
            else:
                self._cache["gamma"] = self.Dwe * np.cos(self.alpha_0) / self.Dpw
        return self._cache["gamma"]

    @property
    def cl(self) -> float:
        """c_L [N/mm^(10/9)] -- the line-contact load-deflection constant,
        eq.(35). Like ``PointContactStiffness.cp``, this is not itself a
        stiffness: it feeds the load-deflection relation the load-
        distribution solver (or a Hertz-based approach) actually solves
        with -- that step happens outside this class."""
        c_L = 35948.0 * self.Lwe ** (8.0 / 9.0)
        return c_L

    @property
    def cs(self) -> float:
        """c_s [N/mm^(10/9)] -- ``cl`` split evenly across the ``n_s``
        laminae, eq.(35). Same caveat as ``cl``: a load-deflection
        constant, not an assembled stiffness."""
        c_s = self.cl / self.n_s
        return c_s