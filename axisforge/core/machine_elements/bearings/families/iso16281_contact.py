# =====================================================================
# families/iso16281_contact.py
# =====================================================================
"""
core/machine_elements/bearings/families/contact.py

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
    """Hertzian point contact -- ISO/TS 16281 Sec 5 eq.(2)-(11). Works
    for radial AND thrust duty: alpha_0 is just an input, and gamma
    already handles the alpha_0=90deg (thrust) case explicitly. The
    two duties differ only in which alpha_0/ri/re a family supplies,
    never in this class's own math."""

    def __init__(self, Dw: float, ri: float, re: float, E: float, nu: float,
                 alpha_0: float, Dpw: float):
        self.Dw, self.ri, self.re = Dw, ri, re
        self.E, self.nu, self.alpha_0, self.Dpw = E, nu, alpha_0, Dpw
        self._cache: dict[str, float] = {}

    @property
    def duty(self) -> str:
        """Derived, not fixed -- varies per instance depending on alpha_0."""
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
    def stiffness(self) -> float:
        """c_p [N/mm^(3/2)] -- the value everything above exists to produce."""
        E_star = self.E / (1.0 - self.nu**2)
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
    # .stiffness is inherited unchanged -- it calls _outer_term, which is
    # the only piece this subtype actually overrides.


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
    def stiffness(self) -> float:
        """(c_L) [N/mm^(10/9)] --  eq.(35)."""
        c_L = 35948.0 * self.Lwe ** (8.0 / 9.0)
        return c_L

    @property
    def lamina_stiffness(self) -> float:
        """(c_s) [N/mm^(10/9)] --  eq.(35)."""
        c_s = self.stiffness / self.n_s
        return c_s 
