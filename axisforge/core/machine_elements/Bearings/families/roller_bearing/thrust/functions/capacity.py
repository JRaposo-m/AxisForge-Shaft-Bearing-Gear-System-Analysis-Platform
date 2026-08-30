"""
core/machine_elements/Bearings/families/roller/thrust/functions/capacity.py

Ca -- ISO 281 dynamic capacity for thrust roller bearings, and
per-roller / per-lamina capacity derived from an already-known Ca.

Same split as the radial side (../../radial/functions/capacity.py):

  - BearingCapacity: OVERALL bearing Ca.
      .dynamic_nonzero_alpha() -- ISO 281:2007 Formula (37), fc from
        Formula (38) -- Sec 6.6.1, single row, contact angle alpha != 90deg.
      .dynamic_90deg()         -- ISO 281:2007 Formula (41), fc from
        Formula (42) -- Sec 6.6.2, single row, contact angle alpha = 90deg.
      .dynamic_multirow()      -- ISO 281:2007 Formula (46) -- combines n
        already-known single-row Ca's into the overall bearing Ca.
      .static()                -- ISO 76, thrust roller bearings. NOT
        provided yet -- stubbed, not fabricated.

    Formula (35)/(36) and (39)/(40) are the general (e, c, h) forms; this
    module implements only the already-substituted e=9/8, c=31/3, h=7/3
    results -- Formula (37)/(38) and (41)/(42) -- same as how the radial
    side only implements Formula (33)/(34), not the general form behind it.


  - RollingElementCapacity: given an ALREADY-KNOWN Ca:
      .thrust_nonzero_alpha() -- ISO/TS 16281 Sec 5.3.1.3(ish) eq.(50)-(51)
      .thrust_90deg()         -- eq.(52)-(53)ish, alpha_0 = 90deg
      .per_lamina()           -- eq.(56)-(57), same as the radial side


References:
  ISO 281:2007 Sec 6.6.1, Formula (35)-(38) -- Ca, single row, alpha != 90deg
  ISO 281:2007 Sec 6.6.2, Formula (39)-(42) -- Ca, single row, alpha = 90deg
  ISO 281:2007          , Formula (46)      -- Ca, two or more rows of rollers
  ISO 281:2007 Table 2, Table No. 10 -- reduction factor, thrust roller bearings
  ISO/TS 16281:2008 Sec 5.3.x eq.(50)-(53) -- per-roller capacity, thrust
  ISO/TS 16281:2008 Sec 5.3.2 eq.(56)-(57) -- per-lamina capacity
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


class BearingCapacity:
    """Overall bearing Ca -- ISO 281:2007 Sec 6.6.1 (single row, alpha != 90deg),
    Sec 6.6.2 (single row, alpha = 90deg), and Formula (46) (multi-row combination)."""

    _B1_0483_N = 551.13373   # "0,483*B1", Formula (38) note -- Ca in N. Same B1
                              # material/geometry constant as the radial roller
                              # side's Formula (34) -- same value, 551,133 73.
    _B1_041_N  = 472.45388   # "0,41*B1", Formula (42) note -- Ca in N, alpha_0 = 90deg case.

    # ------------------------------------------------------------------
    # Sec 6.6.1 -- single row, contact angle alpha != 90deg
    # ------------------------------------------------------------------
    @classmethod
    def _fc_nonzero_alpha(cls, gamma: float, reduction_factor: float,
                          eta: float) -> float:
        """
        f_c -- Formula (38), thrust roller bearings, alpha_0 != 90deg.
        """
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma = Dwe*cos(alpha)/Dpw must be in (0, 1); got {gamma}.")

        gamma_term = (gamma ** (2.0 / 9.0) * (1.0 - gamma) ** (29.0 / 27.0)
                      / (1.0 + gamma) ** (1.0 / 4.0))
        bracket = ((1.0 - gamma) / (1.0 + gamma)) ** (143.0 / 108.0)
        correction = (1.0 + bracket ** (9.0 / 2.0)) ** (-2.0 / 9.0)

        return cls._B1_0483_N * reduction_factor * eta * gamma_term * correction

    @classmethod
    def dynamic_nonzero_alpha(cls, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                               gamma: float, reduction_factor: float,
                               eta: float) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (37), f_c from Formula (38).
        alpha_0 in RADIANS.

        Single row only (Sec 6.6.1) -- no i parameter, same reasoning as
        the ball thrust side's dynamic_nonzero_alpha(): a bearing with two
        or more rows calls this once per row (each with its own Z/Dwe/Lwe/
        geometry) and combines the results with dynamic_multirow() below
        (Formula (46)).
        """
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0.0:
            raise ValueError(f"Z, Dwe and Lwe must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}.")

        fc = cls._fc_nonzero_alpha(gamma, reduction_factor, eta)
        cos_term = (Lwe * np.cos(alpha_0)) ** (7.0 / 9.0)
        tan_term = np.tan(alpha_0)

        return fc * cos_term * tan_term * Z ** (3.0 / 4.0) * Dwe ** (29.0 / 27.0)

    # ------------------------------------------------------------------
    # Sec 6.6.2 -- single row, contact angle alpha_0 = 90deg
    # ------------------------------------------------------------------
    @classmethod
    def _fc_90deg(cls, gamma: float, reduction_factor: float, eta: float) -> float:
        """
        f_c -- Formula (42), thrust roller bearings, alpha_0 = 90deg.

        Structurally simpler than _fc_nonzero_alpha(): the (1-gamma)/(1+gamma)
        factors and the {1+[...]^(9/2)}^(-2/9) correction term both drop
        out entirely, leaving a bare gamma**(2/9) -- this is Formula (42)
        as given, not an approximation of Formula (38).
        """
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma = Dwe/Dpw must be in (0, 1); got {gamma}.")

        return cls._B1_041_N * reduction_factor * eta * gamma ** (2.0 / 9.0)

    @classmethod
    def dynamic_90deg(cls, Z: int, Dwe: float, Lwe: float, gamma: float,
                       reduction_factor: float, eta: float) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (41), f_c from Formula (42).

        Single row only (Sec 6.6.2) -- no alpha_0 argument (fixed at 90deg
        by definition, so no cos/tan(alpha) term exists in Formula (39),
        unlike Formula (35) for the alpha != 90deg case) and no i
        parameter, same reasoning as dynamic_nonzero_alpha().
        """
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0.0:
            raise ValueError(f"Z, Dwe and Lwe must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}.")

        fc = cls._fc_90deg(gamma, reduction_factor, eta)
        return fc * Lwe ** (7.0 / 9.0) * Z ** (3.0 / 4.0) * Dwe ** (29.0 / 27.0)

    # ------------------------------------------------------------------
    # Formula (46) -- thrust roller bearings with two or more rows
    # ------------------------------------------------------------------
    @staticmethod
    def dynamic_multirow(Z_rows: Sequence[int], Lwe_rows: Sequence[float],
                          Ca_rows: Sequence[float]) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (46).

            Ca = (Z1*Lwe1 + Z2*Lwe2 + ... + Zn*Lwen)
                 * [sum_j (Zj*Lwej/Caj)**(9/2)]**(-2/9)

        Ca_rows[j] is the SINGLE-ROW Ca of row j -- obtained by calling
        dynamic_nonzero_alpha() or dynamic_90deg() once per row, with that
        row's own Z/Dwe/Lwe/geometry. This method does not resolve Ca
        itself, same "given already-known inputs" pattern as
        RollingElementCapacity below and as the ball side's
        dynamic_multirow() -- the caller loops over rows, this just
        combines the results. Rows need not be identical.

        Only valid for n >= 2 rows; a single row is dynamic_nonzero_alpha()/
        dynamic_90deg() directly, not this method.
        """
        if not (len(Z_rows) == len(Lwe_rows) == len(Ca_rows)):
            raise ValueError(
                f"Z_rows, Lwe_rows and Ca_rows must be the same length; got "
                f"{len(Z_rows)}, {len(Lwe_rows)} and {len(Ca_rows)}."
            )
        if len(Z_rows) < 2:
            raise ValueError(
                f"dynamic_multirow() needs >= 2 rows; got {len(Z_rows)}. "
                "Use dynamic_nonzero_alpha()/dynamic_90deg() directly for a single row."
            )
        if any(Zj <= 0 for Zj in Z_rows):
            raise ValueError(f"All Z_rows must be positive; got {list(Z_rows)}.")
        if any(Lj <= 0.0 for Lj in Lwe_rows):
            raise ValueError(f"All Lwe_rows must be positive; got {list(Lwe_rows)}.")
        if any(Caj <= 0.0 for Caj in Ca_rows):
            raise ValueError(f"All Ca_rows must be positive; got {list(Ca_rows)}.")

        ZL_rows = [Zj * Lj for Zj, Lj in zip(Z_rows, Lwe_rows)]
        total_ZL = sum(ZL_rows)
        bracket = sum((ZLj / Caj) ** (9.0 / 2.0) for ZLj, Caj in zip(ZL_rows, Ca_rows))
        return total_ZL * bracket ** (-2.0 / 9.0)

    @staticmethod
    def static(Z: int, Dwe: float, Lwe: float, alpha_0: float, i: int = 1) -> float:
        """Ca0 [N] -- ISO 76, thrust roller bearings. Not filled in."""
        raise NotImplementedError("Static rating formula not provided yet -- see ISO 76, thrust roller bearings")


class RollingElementCapacity:
    """Per-roller / per-lamina dynamic capacity, thrust duty -- ISO/TS 16281:2008."""

    @staticmethod
    def thrust_nonzero_alpha(Z: int, alpha_0: float, gamma: float, Ca: float,
                              lambda_v: float) -> tuple[float, float]:
        """
        (Q_ci, Q_ce) [N] -- thrust roller bearing, alpha_0 != 90deg.

        Ca is supplied by the caller (bearing.C, or whatever Ca the
        analysis wants to check against) -- this method does not resolve
        Ca itself (see BearingCapacity.dynamic_nonzero_alpha() above).
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