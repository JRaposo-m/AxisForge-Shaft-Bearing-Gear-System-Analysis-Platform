"""
core/machine_elements/Bearings/families/ball/thrust/functions/capacity.py

Cr, Ca -- ISO 281 dynamic/static capacity for thrust ball bearings, and
per-element capacity (Q_ci, Q_ce) derived from an already-known Ca.

Same split as the radial side (../../radial/functions/capacity.py):

  - BearingCapacity: OVERALL bearing Ca (thrust duty rates on Ca, not Cr),
    ISO 281:2007 Formula (20)/(21) -- a different, eta-based fc than the
    radial side's Formula (15). NOT provided yet -- stubbed, not fabricated.

  - RollingElementCapacity: given an ALREADY-KNOWN Ca, computes the
    per-rolling-element capacity Q_ci (inner) / Q_ce (outer):
      .thrust_nonzero_alpha() -- ISO/TS 16281 Sec 4.3.1.3 eq.(21)-(22)
      .thrust_90deg()         -- ISO/TS 16281 Sec 4.3.1.4 eq.(23)-(24)
    Both ported from the solver-side draft's RollingElementCapacity
    dataclass (ball_bearing_solver.py) -- dataclass bookkeeping (label,
    Cr/Ca storage, bearing_class) dropped, same as the radial side.

References:
  ISO 281:2007  Formula (20)/(21)         -- static/dynamic rating, thrust ball bearings
  ISO/TS 16281:2008 Sec 4.3.1.3 eq.(21)-(22) -- per-element capacity, thrust, alpha != 90deg
  ISO/TS 16281:2008 Sec 4.3.1.4 eq.(23)-(24) -- per-element capacity, thrust, alpha = 90deg
"""
from __future__ import annotations
import numpy as np


class BearingCapacity:
    """Overall bearing Ca -- ISO 281:2007 Formula (20)/(21), thrust ball bearings."""

    @staticmethod
    def dynamic(Z: int, Dw: float, alpha_0: float, reduction_factor: float,
                i: int = 1) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (20)/(21). Uses an eta reduction
        factor (function of alpha_0), NOT the same fc as the radial side's
        Formula (15) -- not the same formula with alpha_0=90 substituted in.

        NOT implemented -- eta/Formula (20)-(21) text not provided yet.
        """
        raise NotImplementedError("Formula (20)/(21) not provided yet -- see ISO 281:2007, thrust ball bearings")

    @staticmethod
    def static(Z: int, Dw: float, alpha_0: float, reduction_factor: float,
               i: int = 1) -> float:
        """Ca0 [N] -- ISO 76:2006, thrust ball bearings. Not filled in."""
        raise NotImplementedError("Static rating formula not provided yet -- see ISO 76:2006, thrust ball bearings")


class RollingElementCapacity:
    """Per-rolling-element dynamic capacity, thrust duty -- ISO/TS 16281:2008 Sec 4.3.1.3/.4."""

    @staticmethod
    def _geometry_bracket(gamma: float, ri: float, re: float, Dw: float) -> float:
        """
        Groove geometry factor for Q_ci/Q_ce, alpha != 90deg -- same form as
        the radial side's _geometry_bracket (ISO/TS 16281 Sec 4.3.1.2-.3):

            1.044 * ((1-gamma)/(1+gamma))^1.72 * (ri/re * (2re-Dw)/(2ri-Dw))^0.41
        """
        radii_ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))
        return 1.044 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72 * radii_ratio ** 0.41

    @classmethod
    def thrust_nonzero_alpha(cls, Z: int, alpha_0: float, ri: float, re: float,
                              Dw: float, gamma: float, Ca: float) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- thrust ball bearing, alpha_0 != 90deg.
        ISO/TS 16281 Sec 4.3.1.3 eq.(21)-(22).

        Ca is supplied by the caller -- this method does not resolve Ca
        itself (see BearingCapacity.dynamic() above, not yet implemented).
        """
        if 2.0 * ri <= Dw:
            raise ValueError(f"2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) -- inner groove radius too small.")
        if 2.0 * re <= Dw:
            raise ValueError(f"2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) -- outer groove radius too small.")

        bracket = cls._geometry_bracket(gamma, ri, re, Dw)
        sin_a   = np.sin(alpha_0)

        Q_ci = (Ca / (Z * sin_a)) * (1.0 + bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / (Z * sin_a)) * (1.0 + bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce

    @staticmethod
    def thrust_90deg(Z: int, ri: float, re: float, Dw: float, Ca: float) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- thrust ball bearing, alpha_0 = 90deg.
        ISO/TS 16281 Sec 4.3.1.4 eq.(23)-(24).

        At alpha_0=90deg, gamma=0; the (1-gamma)/(1+gamma) term vanishes and
        the geometry bracket reduces to the groove radii ratio alone --
        computed directly here rather than through _geometry_bracket(gamma=0, ...),
        matching the solver-side draft exactly.
        """
        if 2.0 * ri <= Dw:
            raise ValueError(f"2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) -- inner groove radius too small.")
        if 2.0 * re <= Dw:
            raise ValueError(f"2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) -- outer groove radius too small.")

        ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))

        Q_ci = (Ca / Z) * (1.0 + (ratio ** 0.41) ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / Z) * (1.0 + (ratio ** 0.41) ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce