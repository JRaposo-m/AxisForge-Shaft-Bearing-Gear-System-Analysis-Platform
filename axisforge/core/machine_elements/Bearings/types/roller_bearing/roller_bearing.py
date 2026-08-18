"""
core/machine_elements/Bearings/types/roller_bearing/roller_bearing.py

Base class for line-contact ("roller") bearing calculations.
ISO/TS 16281 eq.(34)-(37), lamina positions per §5.2.2.

This class knows the line-contact math and NOTHING else: no catalog data,
no bearing family, no §6 reference relations. It's the foundation every
subtype in roller_bearing/subtype/ (CylindricalRollerBearing, and later
tapered/spherical roller) builds on for its calculations.

All parameters in mm / MPa. Angles in radians internally.

References:
  - ISO/TS 16281:2008 §5.2, eq.(34)-(46) — internal load distribution, line contact
"""

from __future__ import annotations
import numpy as np


class RollerBearingGeometry:
    load_deflection_exponent: float = 10.0 / 9.0
    _gamma: float | None = None   # cache for the gamma property

    def setup(self, Dwe, Lwe, Dpw, Z, s, n_s=30, alpha_0_deg=0.0) -> None:
        self.Dwe = Dwe
        self.Lwe = Lwe
        self.Dpw = Dpw
        self.Z   = Z
        self.s   = s
        self.n_s = n_s
        self.alpha_0 = np.radians(alpha_0_deg)

        lamina_length = Lwe / n_s
        self.x_k = lamina_length * (np.arange(n_s) + 0.5) - Lwe / 2.0
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        self._gamma = None   # geometry changed -- invalidate cache

    def hertz_spring_constant(self) -> float:
        self.cL = 35948.0 * self.Lwe ** (8.0 / 9.0)
        self.cs = self.cL / self.n_s
        return self.cL

    # ------------------------------------------------------------------
    # gamma -- ISO/TS 16281 §5.3.1.2, used throughout eq.(47)-(52)
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> float:
        """
        gamma = Dwe * cos(alpha_0) / Dpw -- roller-to-pitch-diameter ratio,
        used throughout ISO/TS 16281 Sec 5.3.1.2 (eq.47-52) and reusable as-is
        by ISO 281 life calculations. Lazily computed and cached on first
        access; purely geometric (Dwe, Dpw, alpha_0 don't change after
        setup()) -- kept at the bearing level so every consumer (solver,
        ISO 281, debug utilities) reads the same cached value instead of
        recomputing it.

        alpha_0 = 90 deg (pure thrust) is a special case: cos(pi/2) doesn't
        round to a clean 0.0 in floating point (~6e-17, sign depends on the
        rounding path), which would either collapse gamma to a meaningless
        near-zero value or falsely trip the (0,1) check below. At exactly
        90 deg the cos(alpha) term is dropped and gamma reduces to Dwe/Dpw.

        Raises
        ------
        ValueError : if gamma is outside (0, 1) -- checked once here instead
                    of being repeated at every capacity call site.
        """
        if self._gamma is None:
            if np.isclose(self.alpha_0, np.pi / 2, atol=1e-9):
                g = self.Dwe / self.Dpw
            else:
                g = self.Dwe * np.cos(self.alpha_0) / self.Dpw
            if not (0.0 < g < 1.0):
                raise ValueError(
                    f"gamma = Dwe*cos(alpha_0)/Dpw = {g:.6f} is outside "
                    f"(0, 1) -- check Dwe/Dpw/alpha_0."
                )
            self._gamma = g
        return self._gamma