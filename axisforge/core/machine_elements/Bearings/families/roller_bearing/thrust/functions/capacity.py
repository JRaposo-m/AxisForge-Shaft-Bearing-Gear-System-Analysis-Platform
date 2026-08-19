"""
core/machine_elements/Bearings/families/roller/thrust/functions/capacity.py

Cr, Ca -- ISO 281 dynamic/static capacity for thrust roller bearings, and
per-roller / per-lamina capacity derived from an already-known Ca.

Same split as the radial side (../../radial/functions/capacity.py):

  - BearingCapacity: OVERALL bearing Ca -- ISO 281:2007, thrust roller
    bearings (eta-based, ISO 281:2007 Table 2 Table No. 10:
    eta = 1 - 0.15*sin(alpha)). NOT provided yet -- stubbed, not fabricated.

  - RollingElementCapacity: given an ALREADY-KNOWN Ca:
      .thrust_nonzero_alpha() -- ISO/TS 16281 Sec 5.3.1.3(ish) eq.(50)-(51)
      .thrust_90deg()         -- eq.(52)-(53)ish, alpha_0 = 90deg
      .per_lamina()           -- eq.(56)-(57), same as the radial side
    Ported from the solver-side draft's RollerElementCapacity dataclass --
    dataclass bookkeeping dropped, same as everywhere else.

    lambda_v is NOT stored/defaulted here -- same reasoning as the radial
    side: it's subtype input (ISO 281:2007 Table 2, Table No. 10 gives
    0.73 for "Thrust roller bearings" as a category, but that's still the
    subtype's own value to declare and pass in, not baked into this module).
    ThrustCylindricalRollerFamily and ThrustNeedleRollerFamily both declare
    LAMBDA_V_THRUST = 0.73 independently.

    NOTE (verified against the pasted source, not silently "fixed"):
    thrust_nonzero_alpha()'s `base` term does NOT carry the leading 1.038
    coefficient that the radial side's eq.(47)-(48) `base` does -- ported
    exactly as given. Flagging this because it looks like it could be a
    typo/omission in the original draft, but I'm not changing it without
    you confirming against the actual ISO/TS 16281 clause.

References:
  ISO 281:2007 Table 2, Table No. 10 -- reduction factor, thrust roller bearings
  ISO/TS 16281:2008 Sec 5.3.x eq.(50)-(53) -- per-roller capacity, thrust
  ISO/TS 16281:2008 Sec 5.3.2 eq.(56)-(57) -- per-lamina capacity
"""
from __future__ import annotations
import numpy as np


class BearingCapacity:
    """Overall bearing Ca -- ISO 281:2007, thrust roller bearings."""

    @staticmethod
    def dynamic(Z: int, Dwe: float, Lwe: float, alpha_0: float, i: int = 1) -> float:
        """
        Ca [N] -- ISO 281:2007, thrust roller bearings. Uses an eta
        reduction factor (Table 2, Table No. 10: eta = 1 - 0.15*sin(alpha)),
        NOT the same fc as the radial side.

        NOT implemented -- full formula text not provided yet.
        """
        raise NotImplementedError("Formula not provided yet -- see ISO 281:2007, thrust roller bearings")

    @staticmethod
    def static(Z: int, Dwe: float, Lwe: float, alpha_0: float, i: int = 1) -> float:
        """Ca0 [N] -- ISO 76:2006, thrust roller bearings. Not filled in."""
        raise NotImplementedError("Static rating formula not provided yet -- see ISO 76:2006, thrust roller bearings")


class RollingElementCapacity:
    """Per-roller / per-lamina dynamic capacity, thrust duty -- ISO/TS 16281:2008."""

    @staticmethod
    def thrust_nonzero_alpha(Z: int, alpha_0: float, gamma: float, Ca: float,
                              lambda_v: float) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- thrust roller bearing, alpha_0 != 90deg.

        Ca is supplied by the caller -- this method does not resolve Ca
        itself (see BearingCapacity.dynamic() above, not yet implemented).
        lambda_v : no default -- subtype's own value (Table 2, Table No. 10
        gives 0.73 for thrust roller bearings today).
        """
        base    = ((1.0 - gamma) / (1.0 + gamma)) ** (143.0 / 108.0)
        denom_a = Z * np.sin(alpha_0)

        Q_ci = (1.0 / lambda_v) * (Ca / denom_a) * (1.0 + base ** (9.0 / 2.0)) ** (2.0 / 9.0)
        Q_ce = (1.0 / lambda_v) * (Ca / denom_a) * (1.0 + base ** (-9.0 / 2.0)) ** (2.0 / 9.0)
        return Q_ci, Q_ce

    @staticmethod
    def thrust_90deg(Z: int, Ca: float, lambda_v: float) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- thrust roller bearing, alpha_0 = 90deg. Q_ci ==
        Q_ce here -- the bracket collapses to a fixed 2**(2/9) constant at
        90deg (no gamma/geometry dependence left), same shape as the ball
        side's thrust_90deg().
        """
        Q = (1.0 / lambda_v) * (Ca / Z) * 2.0 ** (2.0 / 9.0)
        return Q, Q

    @staticmethod
    def per_lamina(Q_ci: float, Q_ce: float, n_s: int) -> tuple[float, float]:
        """(q_ci, q_ce) [N] -- per-lamina dynamic load rating, eq.(56)-(57).
        Same formula as the radial side -- duplicated here for the same
        self-containment reason as contact_stiffness.py."""
        q_ci = Q_ci * (1.0 / n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / n_s) ** (7.0 / 9.0)
        return q_ci, q_ce