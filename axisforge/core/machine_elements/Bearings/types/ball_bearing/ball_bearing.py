"""
core/machine_elements/Bearings/types/ball_bearing/ball_bearing.py

Base class for point-contact ("ball") bearing calculations.
ISO/TS 16281 eq.(2)-(11).

This class knows the Hertz point-contact math and NOTHING else: no
catalog data, no bearing family, no §6 reference relations. It's the
foundation every subtype in ball_bearing/subtype/ (DeepGrooveBallBearing,
AngularContactBallBearing, ...) builds on for its calculations.

All parameters in mm / MPa. Angles in radians internally.

References:
  - ISO/TS 16281:2008 §5 eq.(2)-(11) — point contact load distribution
"""

from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq


class BallBearingGeometry:
    """
    Point contact geometry — generic math, no family knowledge.

    Attributes populated by setup():
        ri, re, Dw, Dpw, Z, s, E, nu
        A, alpha_0, Ri, phi_j
        cp  (after hertz_spring_constant())
    """

    load_deflection_exponent: float = 3.0 / 2.0

    def setup(self,
              ri: float,
              re: float,
              Dw: float,
              Dpw: float,
              Z: int,
              E: float,
              nu: float = 0.3,
              *,
              s: float | None = None,
              alpha_0_deg: float | None = None) -> None:
        """
        Cache internal geometry.

        Parameters
        ----------
        ri, re      : inner/outer groove radii [mm]
        Dw          : rolling element diameter [mm]
        Dpw         : pitch circle diameter [mm]
        Z           : number of rolling elements
        E           : Young's modulus [MPa]
        nu          : Poisson's ratio

        Clearance specification — exactly one required:
            s           : diametral operating clearance [mm]
                          → alpha_0 = arccos(1 − s / (2·A))
            alpha_0_deg : free contact angle [°]
                          → s = 2·A·(1 − cos(alpha_0))
        """
        if (s is None) == (alpha_0_deg is None):
            raise ValueError(
                "Exactly one of 's' or 'alpha_0_deg' must be provided, not both or neither."
            )

        self.ri  = ri;   self.re  = re
        self.Dw  = Dw;   self.Dpw = Dpw
        self.Z   = Z
        self.E   = E;    self.nu  = nu

        self.A = ri + re - Dw

        if s is not None:
            self.s       = s
            self.alpha_0 = np.arccos(1.0 - s / (2.0 * self.A))
        else:
            self.alpha_0 = np.radians(alpha_0_deg)
            self.s       = 2.0 * self.A * (1.0 - np.cos(self.alpha_0))

        self.Ri    = Dpw / 2.0 + (ri - Dw / 2.0) * np.cos(self.alpha_0)
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        self._gamma = None   # geometry changed -- invalidate cache


    # ------------------------------------------------------------------
    # gamma -- ISO/TS 16281 eq.(5)-(8), reusable as-is by ISO 281
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> float:
        """
        gamma = Dw * cos(alpha_0) / Dpw -- ball-to-pitch-diameter ratio
        feeding the curvature sum/difference formulas below (eq.5-8), and
        reusable as-is by ISO 281 life calculations. Lazily computed and
        cached on first access; purely geometric (Dw, Dpw, alpha_0 don't
        change after setup()).

        alpha_0 = 90 deg (pure thrust ball bearing) is a special case: cos(pi/2)
        doesn't round to a clean 0.0 in floating point (~6e-17, sign depends on
        the rounding path). At exactly 90 deg the cos(alpha) term is dropped
        and gamma reduces to Dw/Dpw -- same convention as RollerBearingGeometry.gamma.
        """
        if self._gamma is None:
            if np.isclose(self.alpha_0, np.pi / 2, atol=1e-9):
                self._gamma = self.Dw / self.Dpw
            else:
                self._gamma = self.Dw * np.cos(self.alpha_0) / self.Dpw
        return self._gamma

    # ------------------------------------------------------------------
    # Curvature sums and differences — ISO/TS 16281 eq.(5)–(8)
    # ------------------------------------------------------------------

    def curvature_sum_inner(self) -> float:
        """Sum(rho)_i = (2/Dw)*(2 + gamma/(1-gamma) - Dw/(2*ri))   eq.(5)"""
        g = self.gamma
        return (2.0 / self.Dw) * (2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri))

    def curvature_sum_outer(self) -> float:
        """Sum(rho)_e = (2/Dw)*(2 - gamma/(1+gamma) - Dw/(2*re))   eq.(6)"""
        g = self.gamma
        return (2.0 / self.Dw) * (2.0 - g / (1.0 + g) - self.Dw / (2.0 * self.re))

    def curvature_diff_inner(self) -> float:
        """F_i(rho) = (gamma/(1-gamma) + Dw/(2*ri)) / (2 + gamma/(1-gamma) - Dw/(2*ri))   eq.(7)"""
        g = self.gamma
        num = g / (1.0 - g) + self.Dw / (2.0 * self.ri)
        den = 2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri)
        return num / den

    def curvature_diff_outer(self) -> float:
        """F_e(rho) = (-gamma/(1+gamma) + Dw/(2*re)) / (2 - gamma/(1+gamma) - Dw/(2*re))   eq.(8)"""
        g = self.gamma
        num = -g / (1.0 + g) + self.Dw / (2.0 * self.re)
        den = 2.0 - g / (1.0 + g) - self.Dw / (2.0 * self.re)
        return num / den

    # ------------------------------------------------------------------
    # Elliptic auxiliary — ISO/TS 16281 eq.(2)
    # ------------------------------------------------------------------

    @staticmethod
    def _chi_equation(chi: float, F_rho: float) -> float:
        """1 − 2/(χ²−1)·[K(χ)/E(χ) − 1] − F(ρ) = 0"""
        m = 1.0 - 1.0 / chi**2
        K = ellipk(m)
        E = ellipe(m)
        return 1.0 - (2.0 / (chi**2 - 1.0)) * (K / E - 1.0) - F_rho

    def _chi(self, F_rho: float) -> float:
        """Solve eq.(2) for χ via brentq."""
        return brentq(self._chi_equation, 1.0001, 1000.0, args=(F_rho,))

    # ------------------------------------------------------------------
    # Hertzian spring constant — ISO/TS 16281 eq.(9)–(11)
    # ------------------------------------------------------------------

    def hertz_spring_constant(self) -> float:
        """
        c_p  [N/mm^(3/2)]

        Requires setup() to have been called first.
        Result cached as self.cp.
        """
        Sr_i = self.curvature_sum_inner()
        Sr_e = self.curvature_sum_outer()
        Fi   = self.curvature_diff_inner()
        Fe   = self.curvature_diff_outer()
        xi   = self._chi(Fi)
        xe   = self._chi(Fe)

        E_star = self.E / (1.0 - self.nu**2)

        mi, me = 1.0 - 1.0 / xi**2, 1.0 - 1.0 / xe**2
        Ki, Ei = ellipk(mi), ellipe(mi)
        Ke, Ee = ellipk(me), ellipe(me)

        bi = Ki * np.cbrt(Sr_i / (xi**2 * Ei))
        be = Ke * np.cbrt(Sr_e / (xe**2 * Ee))

        self.cp = 1.48 * E_star * (bi + be) ** (-3.0 / 2.0)
        return self.cp