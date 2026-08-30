"""
core/machine_elements/bearings/families/ball_bearing/radial/functions/capacity.py

Cr, Ca -- ISO 281 dynamic/static capacity for radial ball bearings, and
per-element capacity (Q_ci, Q_ce) derived from an already-known Cr.

Two distinct classes live here, from two different standards:

  - BearingCapacity: computes the OVERALL bearing Cr/Ca from geometry +
    f_c formulas (ISO 1281-1:2021 Sec 6.2 eq. (15)).

  - Calculations for the static load rating are not yet implemented -- see ISO 76:2006 Sec 5.  

  - RollingElementCapacity: given an ALREADY-KNOWN Cr (the catalog value,
    or whatever the caller supplies), computes the per-rolling-element
    capacity Q_ci (inner) / Q_ce (outer) -- ISO/TS 16281:2008 Sec 4.3.1.2
    eq.(19)-(20). 

References:
  ISO 281:2007 Sec 6.1 and ISO 1281-1:2021 Sec 6.2         -- dynamic load rating, radial ball bearings
  ISO 76:2006   Sec 5                                      -- static load rating, radial ball bearings
  ISO/TS 16281:2008 Sec 4.3.1.2 eq.(19)-(20) -- per-element capacity, radial
"""
from __future__ import annotations
import numpy as np
import math
import warnings



class BearingCapacity:
    """Overall bearing Cr, Ca -- ISO 281:2007 Sec 6.1 / ISO 76:2006 Sec 5."""

    _A1_0089_N = 98.0665    # "0,089*A1", Formula (15) note -- Cr in N, Dw in mm
    _DW_THRESHOLD_MM = 25.4  # Formula (13)/(14) switch point

    @staticmethod
    def _fc(ri: float, re: float, Dw: float, gamma: float, reduction_factor: float) -> float:

        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma = Dw*cos(alpha)/Dpw must be in (0, 1); got {gamma}.")
        if 2.0 * ri <= Dw:
            raise ValueError(f"2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")

        groove_conformity = (2.0 * ri / (2.0 * ri - Dw)) ** 0.41
        gamma_term = gamma ** 0.3 * (1.0 - gamma) ** 1.39 / (1.0 + gamma) ** (1.0 / 3.0)

        if math.isinf(re):
            radii_ratio = 2.0 * ri / (2.0 * ri - Dw)
        else:
            if 2.0 * re <= Dw:
                raise ValueError(f"2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")
            radii_ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))

        bracket = 1.04 * radii_ratio ** 0.41 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72
        correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

        return (BearingCapacity._A1_0089_N * 0.41 * reduction_factor
                * groove_conformity * gamma_term * correction)

    @staticmethod
    def _basic_dynamic_rating(fc: float, i: int, alpha_0: float, Z: int, Dw: float) -> float:
        """Formula (13) (Dw <= 25,4 mm) / Formula (14) (Dw > 25,4 mm). alpha_0 in radians."""
        cos_term = (i * math.cos(alpha_0)) ** 0.7
        if Dw <= BearingCapacity._DW_THRESHOLD_MM:
            return fc * cos_term * Z ** (2.0 / 3.0) * Dw ** 1.8
        return 3.647 * fc * cos_term * Z ** (2.0 / 3.0) * Dw ** 1.4

    @classmethod
    def dynamic(cls, Z: int, Dw: float, alpha_0: float, ri: float, re: float,
                gamma: float, reduction_factor: float, i: int = 1) -> float:
        """
        Cr [N] -- ISO 281:2007 Formula (13)/(14), fc from Formula (15).
        alpha_0 in RADIANS. ri/re/gamma/reduction_factor already resolved
        on the assembled bearing -- no Table 1 lookup here.
        NOT VALIDATED -- see _fc() docstring.
        """
        if Z <= 0 or Dw <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dw and i must all be positive; got Z={Z}, Dw={Dw}, i={i}.")
        fc = cls._fc(ri, re, Dw, gamma, reduction_factor)
        return cls._basic_dynamic_rating(fc, i, alpha_0, Z, Dw)

    @staticmethod
    def static(Z, Dw, alpha_0, reduction_factor, i=1) -> float:
        raise NotImplementedError("f_0 formula not provided yet -- see ISO 76:2006 Sec 5")


class RollingElementCapacity:
    """Per-rolling-element dynamic capacity -- ISO/TS 16281:2008 Sec 4.3.1.2."""

    @staticmethod
    def _geometry_bracket(gamma: float, ri: float, re: float, Dw: float) -> float:
        """
        Groove geometry factor for Q_ci/Q_ce -- ISO/TS 16281 Sec 4.3.1.2-.3:

            1.044 * ((1-gamma)/(1+gamma))^1.72 * (ri/re * (2re-Dw)/(2ri-Dw))^0.41
        """
        radii_ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))
        return 1.044 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72 * radii_ratio ** 0.41

    @classmethod
    def radial(cls, Z: int, alpha_0: float, ri: float, re: float, Dw: float,
               gamma: float, Cr: float, i: int = 1) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- per-rolling-element dynamic capacity, radial ball
        bearings. ISO/TS 16281 Sec 4.3.1.2 eq.(19)-(20).

        Cr is supplied by the caller (bearing.C, or whatever Cr the analysis
        wants to check against) -- this method does not resolve Cr itself
        (see BearingCapacity.dynamic() above, not yet implemented).
        """
        if 2.0 * ri <= Dw:
            raise ValueError(f"2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) -- inner groove radius too small.")
        if 2.0 * re <= Dw:
            raise ValueError(f"2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) -- outer groove radius too small.")

        bracket      = cls._geometry_bracket(gamma, ri, re, Dw)
        cos_alpha_07 = np.cos(alpha_0) * i ** 0.7

        Q_ci = (Cr / (0.407 * Z * cos_alpha_07)) * (1.0 + bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Cr / (0.389 * Z * cos_alpha_07)) * (1.0 + bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce