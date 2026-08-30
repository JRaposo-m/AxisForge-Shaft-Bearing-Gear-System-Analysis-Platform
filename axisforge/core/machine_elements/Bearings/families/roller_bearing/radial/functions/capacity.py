"""
core/machine_elements/bearings/families/roller_bearing/radial/functions/capacity.py

Cr -- ISO 281 dynamic capacity for radial roller bearings, and
per-roller / per-lamina capacity derived from an already-known Cr.

Two classes, same split as the ball side
(../../ball/radial/functions/capacity.py):

  - BearingCapacity: OVERALL bearing Cr/Ca, ISO 281:2007 Sec 6.2 / ISO 76
    Sec 6.
      .dynamic()  -- Cr, Formula (33)/(34).
        BearingCapacity.dynamic() shape: a private _fc() staticmethod
        (Formula (34)) feeding a classmethod that assembles Formula (33).

      .static()   -- C, ISO 76 Sec 6. Still stubbed -- f_0 not provided
        yet, same caution as the ball side (no fabricated table values).

  - RollingElementCapacity: given an ALREADY-KNOWN Cr, computes:
      .radial()     -- whole-roller Q_ci/Q_ce, ISO/TS 16281 Sec 5.3.1.2
                        eq.(47)-(49)

      .per_lamina() -- per-lamina q_ci/q_ce, ISO/TS 16281 Sec 5.3.2
                        eq.(56)-(57). 

BearingCapacity.dynamic() 
---------------------------------------------------------------------
ISO 281:2007 Formula (33) gives Cr directly from geometry + f_c:

    Cr = f_c * (i * Lwe * cos(alpha))^(7/9) * Z^(3/4) * Dwe^(29/27)

Formula (34) gives f_c:

    f_c = 0,483*B1 * 0,377 * lambda 
          * [gamma^(2/9) * (1-gamma)^(29/27) / (1+gamma)^(1/4)]
          * {1 + [1,04 * ((1-gamma)/(1+gamma))^(143/108)]^(9/2)}^(-2/9)

with 0,483*B1 = 551,133 73 (the constant that makes Cr come out in
newtons -- per the note under Formula (34)). 

Each roller subtype (cylindrical_roller.py today, tapered/spherical/needle
later) declares its own LAMBDA_V_RADIAL and passes it in.

References:
  ISO 281:2007 Sec 6.2 Formula (33)-(34) -- dynamic load rating, radial roller bearings
  ISO 76:2006   Sec 6             -- static load rating, radial roller bearings
  ISO/TS 16281:2008 Sec 5.3.1.2 eq.(47)-(49) -- per-roller capacity, radial
  ISO/TS 16281:2008 Sec 5.3.2   eq.(56)-(57) -- per-lamina capacity
"""
from __future__ import annotations
import numpy as np


class BearingCapacity:
    """Overall bearing Cr, C -- ISO 281:2007 Sec 6.2 / ISO 76:2006 Sec 6."""

    # "0,483*B1", Formula (34) note -- value to use so Cr comes out in newtons.
    _B1_0483_N = 551.13373

    @staticmethod
    def _fc(gamma: float, reduction_factor: float) -> float:
        """
        f_c -- ISO 281:2007 Formula (34).

        gamma : Dwe*cos(alpha)/Dpw, already resolved on the assembled bearing.
        reduction_factor : Formula (34)'s lambda -- ISO 281:2007 Table 2
            ("Radial roller bearings") value, owned by the subtype

        nu : Formula (34)'s other factor -- also subtype/table input.
        """
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma = Dwe*cos(alpha)/Dpw must be in (0, 1); got {gamma}.")

        gamma_term = (gamma ** (2.0 / 9.0) * (1.0 - gamma) ** (29.0 / 27.0)
                      / (1.0 + gamma) ** (1.0 / 4.0))
        bracket = 1.04 * ((1.0 - gamma) / (1.0 + gamma)) ** (143.0 / 108.0)
        correction = (1.0 + bracket ** (9.0 / 2.0)) ** (-2.0 / 9.0)

        return (BearingCapacity._B1_0483_N * 0.377 * reduction_factor * gamma_term * correction)

    @classmethod
    def dynamic(cls, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                gamma: float, reduction_factor: float,
                i: int = 1) -> float:
        """
        Cr [N] -- ISO 281:2007 Formula (33), f_c from Formula (34).
        """
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0.0 or i <= 0:
            raise ValueError(
                f"Z, Dwe, Lwe and i must all be positive; got "
                f"Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}."
            )
        fc = cls._fc(gamma, reduction_factor)
        cos_term = (i * Lwe * np.cos(alpha_0)) ** (7.0 / 9.0)
        return fc * cos_term * Z ** (3.0 / 4.0) * Dwe ** (29.0 / 27.0)

    @staticmethod
    def static(Z: int, Dwe: float, Lwe: float, alpha_0: float, i: int = 1) -> float:
        """C [N] -- ISO 76 Sec 6. Needs f_0 -- not filled in."""
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
        Cr itself (see BearingCapacity.dynamic() above).

        lambda_v : no default here -- always the subtype's own value.
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