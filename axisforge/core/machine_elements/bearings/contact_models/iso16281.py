"""
axisforge/core/machine_elements/bearings/contact_models/iso16281.py

ISO/TS 16281's contact-stiffness and bearing-kinematics formulas.
PointContactStiffness/LineContactStiffness take curvatures + materials
only -- they know nothing about Dw/ri/re/alpha_0. The kinematics helpers
below are ISO/TS 16281's own formulas too, but geometric/kinematic, not
Hertz-stiffness concerns.
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
    m = 1.0 - 1.0 / chi**2
    return 1.0 - (2.0 / (chi**2 - 1.0)) * (ellipk(m) / ellipe(m) - 1.0) - F_rho


def _chi(F_rho: float) -> float:
    return brentq(_chi_equation, 1.0001, 1000.0, args=(F_rho,))

# ---------------------------------------------------------------------
# ---- point contact --------------------------------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PointContactStiffness:
    """Hertzian point contact -- ISO/TS 16281 Sec 5 eq.(2)-(11), generalized
    to two dissimilar materials. Knows only geometry + materials."""

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
        return _chi(self.curvature_diff_inner)

    @property
    def chi_outer(self) -> float:
        return _chi(self.curvature_diff_outer)

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
        e_star = 1.0 / ((1.0 - self.nu1**2) / self.e1 + (1.0 - self.nu2**2) / self.e2)
        return 1.48 * e_star * (self._inner_term + self._outer_term) ** (-3.0 / 2.0)


class SelfAligningPointContactStiffness(PointContactStiffness):
    @property
    def _outer_term(self) -> float:
        raise NotImplementedError(
            "Closed-form circular-contact (chi_e=1) outer-race term not "
            "provided yet -- see class docstring."
        )


# ---------------------------------------------------------------------
# ---- Line contact ---------------------------------------------------
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class LineContactStiffness:
    Lwe: float
    n_s: float

    @property
    def lamina_positions(self) -> np.ndarray:
        lamina_length = self.Lwe / self.n_s
        return lamina_length * (np.arange(self.n_s) + 0.5) - self.Lwe / 2.0

    @property
    def cl(self) -> float:
        return 35948.0 * self.Lwe ** (8.0 / 9.0)

    @property
    def cs(self) -> float:
        return self.cl / self.n_s