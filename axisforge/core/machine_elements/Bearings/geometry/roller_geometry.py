"""
Bearings/geometry/roller_geometry.py

Line contact internal geometry for roller bearings (CR, TR, SR).
ISO/TS 16281 line contact formulation.

Phase 1: simplified c_l only.
Phase 2: full crowning correction, tilt, roller profile.
"""

from __future__ import annotations
import numpy as np
from .base import BearingGeometry


class RollerBearingGeometry(BearingGeometry):
    """
    Line contact geometry.

    Attributes populated by setup():
        Dw, Dpw, Z, Le, s, E, nu
        phi_j
        cl  (after hertz_spring_constant())
    """

    load_deflection_exponent: float = 10.0 / 9.0

    def __init__(self, contact_angle: float = 0.0):
        self.contact_angle = contact_angle

    def setup(self,
              Dw: float,
              Dpw: float,
              Z: int,
              Le: float,
              s: float,
              E: float,
              nu: float = 0.3) -> None:
        """
        Parameters
        ----------
        Dw  : roller diameter [mm]
        Dpw : pitch circle diameter [mm]
        Z   : number of rollers
        Le  : effective roller contact length [mm]
        s   : radial clearance [mm]
        E   : Young's modulus [MPa]
        nu  : Poisson's ratio
        """
        self.Dw  = Dw;  self.Dpw = Dpw
        self.Z   = Z;   self.Le  = Le
        self.s   = s
        self.E   = E;   self.nu  = nu

        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    def hertz_spring_constant(self) -> float:
        """
        c_l  [N/mm^(10/9)]

        Phase 1 approximation — full ISO/TS 16281 line contact deferred to Phase 2.
        """
        # Approximate per Palmgren: c_l ≈ 35500 · Le^(8/9)
        self.cl = 35500.0 * self.Le ** (8.0 / 9.0)
        return self.cl
