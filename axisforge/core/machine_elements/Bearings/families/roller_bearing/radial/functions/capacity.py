"""
core/machine_elements/Bearings/families/roller/radial/functions/capacity.py

Cr, Ca -- ISO 281 dynamic/static capacity for radial roller bearings, and
per-roller / per-lamina capacity derived from an already-known Cr.

Two classes, same split as the ball side
(../../ball/radial/functions/capacity.py):

  - BearingCapacity: OVERALL bearing Cr/Ca, ISO 281:2007 Sec 6.2 / ISO 76
    Sec 6. Still stubbed -- f_c (Dwe*cos(alpha)/dm table) not provided yet.

  - RollingElementCapacity: given an ALREADY-KNOWN Cr, computes:
      .radial()     -- whole-roller Q_ci/Q_ce, ISO/TS 16281 Sec 5.3.1.2
                        eq.(47)-(49)
      .per_lamina() -- per-lamina q_ci/q_ce, ISO/TS 16281 Sec 5.3.2
                        eq.(56)-(57). This is what a lamina's dynamic
                        equivalent load is meant to be compared against --
                        Q_ci/Q_ce are whole-roller figures and are not the
                        right denominator for a per-lamina check.
    Ported from the solver-side draft's RollerElementCapacity dataclass --
    the dataclass bookkeeping (label, Cr storage, ...) is dropped, same as
    the ball side, since it's no longer needed downstream. Thrust variants
    (thrust_nonzero_alpha, thrust_90deg, eq.(50)-(55), lambda_v=0.73) were
    in that same draft but are NOT ported here -- not radial duty, they
    belong in families/roller/thrust/functions/capacity.py once that
    family exists (same reasoning as the ball side).

lambda_v is NOT stored/defaulted here. ISO 281:2007 Table 2 currently
gives one value for all "Radial roller bearings", but that's still
subtype input, not generic math -- same reasoning as the ball side's
RI_OVER_DW/REDUCTION_FACTOR living in subtypes/, not functions/. Each
roller subtype (cylindrical_roller.py today, tapered/spherical/needle
later) declares its own LAMBDA_V_RADIAL and passes it in -- this module
never assumes the value, so a future subtype with a different table row
just calls .radial() the same way with its own number.

References:
  ISO 281:2007 Sec 6.2            -- dynamic load rating, radial roller bearings
  ISO 76:2006   Sec 6             -- static load rating, radial roller bearings
  ISO/TS 16281:2008 Sec 5.3.1.2 eq.(47)-(49) -- per-roller capacity, radial
  ISO/TS 16281:2008 Sec 5.3.2   eq.(56)-(57) -- per-lamina capacity
"""
from __future__ import annotations
import numpy as np


class BearingCapacity:
    """Overall bearing Cr, Ca -- ISO 281:2007 Sec 6.2 / ISO 76:2006 Sec 6."""

    @staticmethod
    def dynamic(Z: int, Dwe: float, Lwe: float, alpha_0: float, i: int = 1) -> float:
        """Cr [N] -- ISO 281 Sec 6.2. Needs f_c (table per Dwe*cos(alpha)/dm) -- not filled in."""
        raise NotImplementedError("f_c factor not defined yet -- see ISO 281 Sec 6.2")

    @staticmethod
    def static(Z: int, Dwe: float, Lwe: float, alpha_0: float, i: int = 1) -> float:
        """Ca [N] -- ISO 76 Sec 6. Needs f_0 -- not filled in."""
        raise NotImplementedError("f_0 factor not defined yet -- see ISO 76 Sec 6")


class RollingElementCapacity:
    """Per-roller / per-lamina dynamic capacity -- ISO/TS 16281:2008 Sec 5.3.1.2 / 5.3.2."""

    @classmethod
    def radial(cls, Z: int, alpha_0: float, gamma: float, Cr: float,
               lambda_v: float, i: int = 1) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- whole-roller dynamic capacity, radial roller
        bearings. ISO/TS 16281 Sec 5.3.1.2 eq.(47)-(48).

        Cr is supplied by the caller (bearing.C, or whatever Cr the
        analysis wants to check against) -- this method does not resolve
        Cr itself (see BearingCapacity.dynamic() above, not yet implemented).

        lambda_v : no default here -- always the subtype's own value (see
        module docstring). ISO 281:2007 Table 2 -- 0.83 for cylindrical
        roller today (subtypes/cylindrical_roller.py's LAMBDA_V_RADIAL).
        """
        base    = 1.038 * ((1.0 - gamma) / (1.0 + gamma)) ** (143.0 / 108.0)
        denom_a = np.cos(alpha_0) * (i ** (7.0 / 9.0))

        Q_ci = (1.0 / lambda_v) * (Cr / (0.378 * Z * denom_a)) * (1.0 + base ** (9.0 / 2.0)) ** (2.0 / 9.0)
        Q_ce = (1.0 / lambda_v) * (Cr / (0.364 * Z * denom_a)) * (1.0 + base ** (-9.0 / 2.0)) ** (2.0 / 9.0)
        return Q_ci, Q_ce

    @staticmethod
    def per_lamina(Q_ci: float, Q_ce: float, n_s: int) -> tuple[float, float]:
        """
        (q_ci, q_ce) [N] -- per-lamina dynamic load rating, ISO/TS 16281
        Sec 5.3.2 eq.(56)-(57).
        """
        q_ci = Q_ci * (1.0 / n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / n_s) ** (7.0 / 9.0)
        return q_ci, q_ce