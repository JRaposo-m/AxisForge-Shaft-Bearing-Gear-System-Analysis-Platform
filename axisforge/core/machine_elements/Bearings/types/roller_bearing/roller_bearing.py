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
    """
    Line contact geometry — generic math, no family knowledge.

    Attributes populated by setup():
        Dwe, Lwe, Dpw, Z, n_s, s, alpha_0
        x_k, phi_j
        cL, cs  (after hertz_spring_constant())
    """

    load_deflection_exponent: float = 10.0 / 9.0

    def setup(self,
              Dwe: float,
              Lwe: float,
              Dpw: float,
              Z: int,
              s: float,
              n_s: int = 30,
              alpha_0_deg: float = 0.0) -> None:
        """
        Cache internal geometry.

        Parameters
        ----------
        Dwe         : roller diameter [mm]
        Lwe         : effective roller length [mm]
        Dpw         : pitch circle diameter [mm]
        Z           : number of rollers
        s           : diametral operating clearance [mm]
        n_s         : number of laminae (>= 30 per ISO/TS 16281 §5.2.2)
        alpha_0_deg : nominal contact angle [deg] — 0.0 for a radial
                      cylindrical roller bearing (NU/N-type). Non-zero
                      values are accepted here for future tapered/spherical
                      subtypes.
        """
        self.Dwe = Dwe
        self.Lwe = Lwe
        self.Dpw = Dpw
        self.Z   = Z
        self.s   = s
        self.n_s = n_s
        self.alpha_0 = np.radians(alpha_0_deg)

        # Lamina midpoints, eq.(38)-figure 3 — strictly inside (-Lwe/2, Lwe/2)
        lamina_length = Lwe / n_s
        self.x_k = lamina_length * (np.arange(n_s) + 0.5) - Lwe / 2.0

        # Roller angular positions — one per rolling element, NOT per lamina.
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    # ------------------------------------------------------------------
    # Line contact spring constants — ISO/TS 16281 eq.(34)-(37)
    # ------------------------------------------------------------------

    def hertz_spring_constant(self) -> float:
        """
        c_L  [N/mm^(10/9)] — eq.(35), contacting parts made of steel.
        Also derives c_s [N/mm^(10/9)] — eq.(37) — the per-lamina spring
        constant the lamina model (eq.36) uses.

        Requires setup() to have been called first (needs self.Lwe, self.n_s).
        Results cached as self.cL, self.cs.
        """
        self.cL = 35948.0 * self.Lwe ** (8.0 / 9.0)   # eq.(35)
        self.cs = self.cL / self.n_s                   # eq.(37)
        return self.cL