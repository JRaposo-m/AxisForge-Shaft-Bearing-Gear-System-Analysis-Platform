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

    # ------------------------------------------------------------------
    # Curvature sums and differences — ISO/TS 16281 eq.(5)–(8)
    # ------------------------------------------------------------------

    def _gamma(self) -> float:
        """γ = Dw·cos(α) / Dpw"""
        return self.Dw * np.cos(self.alpha_0) / self.Dpw

    def curvature_sum_inner(self) -> float:
        """Σρᵢ = (2/Dw)·(2 + γ/(1−γ) − Dw/(2rᵢ))   eq.(5)"""
        g = self._gamma()
        return (2.0 / self.Dw) * (2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri))

    def curvature_sum_outer(self) -> float:
        """Σρₑ = (2/Dw)·(2 − γ/(1+γ) − Dw/(2rₑ))   eq.(6)"""
        g = self._gamma()
        return (2.0 / self.Dw) * (2.0 - g / (1.0 + g) - self.Dw / (2.0 * self.re))

    def curvature_diff_inner(self) -> float:
        """Fᵢ(ρ) = (γ/(1−γ) + Dw/(2rᵢ)) / (2 + γ/(1−γ) − Dw/(2rᵢ))   eq.(7)"""
        g = self._gamma()
        num = g / (1.0 - g) + self.Dw / (2.0 * self.ri)
        den = 2.0 + g / (1.0 - g) - self.Dw / (2.0 * self.ri)
        return num / den

    def curvature_diff_outer(self) -> float:
        """Fₑ(ρ) = (−γ/(1+γ) + Dw/(2rₑ)) / (2 − γ/(1+γ) − Dw/(2rₑ))   eq.(8)"""
        g = self._gamma()
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