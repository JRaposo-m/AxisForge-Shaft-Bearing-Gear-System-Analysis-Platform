"""
core/machine_elements/Bearings/families/ball/thrust/functions/capacity.py

Cr, Ca -- ISO 281 dynamic/static capacity for thrust ball bearings, and
per-element capacity (Q_ci, Q_ce) derived from an already-known Ca.

Same split as the radial side (../../radial/functions/capacity.py), with
one structural difference worth flagging: on the radial side, i (number of
rows) is folded into ONE formula as (i*cos(alpha))**0.7 -- see radial's
_basic_dynamic_rating(). Thrust bearings do NOT work that way. ISO 281:2007
Sec 6.3 (single row, Formulae (16)-(25)) never contains i at all; ISO
281:2007 Sec 6.4 (two or more rows, Formula (26)-(29)) is a SEPARATE
combination law: each row's own Ca is computed independently from Sec 6.3,
then the per-row Ca's are combined via a product-law-of-probability formula
(Formula (29)) -- not by substituting i into a single closed-form formula.
That's why BearingCapacity below has three dynamic() methods instead of one
i-parametrised method: dynamic_nonzero_alpha() and dynamic_90deg() are
Sec 6.3 (single row only, no i), and dynamic_multirow() is Sec 6.4 -- it
combines already-known per-row Ca values, it does not compute them itself.

  - BearingCapacity: OVERALL bearing Ca (thrust duty rates on Ca, not Cr).
      .dynamic_nonzero_alpha() -- ISO 281:2007 Formula (18)/(19), fc from
        Formula (20) -- Sec 6.3.1, single row, contact angle alpha != 90deg.
      .dynamic_90deg()         -- ISO 281:2007 Formula (23)/(24), fc from
        Formula (25) -- Sec 6.3.2, single row, contact angle alpha = 90deg.
      .dynamic_multirow()      -- ISO 281:2007 Formula (29) -- Sec 6.4,
        combines n already-known single-row Ca's (from the two methods
        above, one call per row) into the overall bearing Ca.
      .static()                -- ISO 76:2006, thrust ball bearings.
        NOT provided yet -- stubbed, not fabricated.

    The "0,089*A1" constant in Formula (20)/(25) is 98,0665 -- an initial
    transcription read it as 98 066,5 (1000x too high), which pushed
    fc/Ca three orders of magnitude above catalog/literature order of
    magnitude; fixed via _A1_0089_N. The radial side's _fc() carries the
    same "98_066.5" constant under a "NOT VALIDATED, ~1000x too high" flag
    -- almost certainly the same error, worth fixing there too.

    dynamic_multirow() (Formula (29)) is self-consistent independently of
    that fix: for n identical rows it reduces to i**0.7 * Ca_single_row,
    matching the (i*cos(alpha))**0.7 relation used on the radial side.

  - RollingElementCapacity: given an ALREADY-KNOWN Ca, computes the
    per-rolling-element capacity Q_ci (inner) / Q_ce (outer):
      .thrust_nonzero_alpha() -- ISO/TS 16281 Sec 4.3.1.3 eq.(21)-(22)
      .thrust_90deg()         -- ISO/TS 16281 Sec 4.3.1.4 eq.(23)-(24)
    Both ported from the solver-side draft's RollingElementCapacity
    dataclass (ball_bearing_solver.py) -- dataclass bookkeeping (label,
    Cr/Ca storage, bearing_class) dropped, same as the radial side.

References:
  ISO 281:2007  Sec 6.3, Formula (16)-(20)   -- Ca, single row, alpha != 90deg
  ISO 281:2007  Sec 6.3, Formula (21)-(25)   -- Ca, single row, alpha = 90deg
  ISO 281:2007  Sec 6.4, Formula (26)-(29)   -- Ca, two or more rows of balls
  ISO/TS 16281:2008 Sec 4.3.1.3 eq.(21)-(22) -- per-element capacity, thrust, alpha != 90deg
  ISO/TS 16281:2008 Sec 4.3.1.4 eq.(23)-(24) -- per-element capacity, thrust, alpha = 90deg
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


class BearingCapacity:
    """Overall bearing Ca -- ISO 281:2007 Sec 6.3 (single row) and Sec 6.4 (multi-row)."""

    _A1_0089_N = 98.0665      # "0,089*A1", Formula (20)/(25) note -- Ca in N, Dw in mm
    _DW_THRESHOLD_MM = 25.4   # Formula (18)/(19) and (23)/(24) switch point

    # ------------------------------------------------------------------
    # Sec 6.3.1 -- single row, contact angle alpha != 90deg
    # ------------------------------------------------------------------
    @classmethod
    def _fc_nonzero_alpha(cls, ri: float, re: float, Dw: float, gamma: float,
                           lam: float, eta: float) -> float:
        """
        fc -- Formula (20), thrust ball bearings, alpha_0 != 90deg.

        lam (lambda) is the general reduction factor (as in 6.2, radial
        ball bearings); eta is the thrust-specific reduction factor noted
        in 6.3.1 as "designated as eta" -- both are distinct inputs here,
        unlike the radial side's single reduction_factor.

        0,089*A1 = 98,0665 (not 98 066,5 -- that earlier reading was off by
        1000x and is what drove fc/Ca three orders of magnitude too high;
        fixed via _A1_0089_N). The radial side's _fc() carries the same
        "98_066.5" constant under the same "NOT VALIDATED, ~1000x too high"
        flag -- almost certainly the identical transcription error, worth
        fixing there too.
        """
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma = Dw*cos(alpha)/Dpw must be in (0, 1); got {gamma}.")
        if 2.0 * ri <= Dw:
            raise ValueError(f"2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")
        if 2.0 * re <= Dw:
            raise ValueError(f"2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")

        groove_conformity = (2.0 * ri / (2.0 * ri - Dw)) ** 0.41
        gamma_term = gamma ** 0.3 * (1.0 - gamma) ** 1.39 / (1.0 + gamma) ** (1.0 / 3.0)
        radii_ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))
        bracket = radii_ratio ** 0.41 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72
        correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

        return cls._A1_0089_N * lam * eta * groove_conformity * gamma_term * correction

    @classmethod
    def dynamic_nonzero_alpha(cls, Z: int, Dw: float, alpha_0: float, ri: float,
                               re: float, gamma: float, lam: float, eta: float) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (18) (Dw <= 25,4 mm) / Formula (19)
        (Dw > 25,4 mm), fc from Formula (20). alpha_0 in RADIANS.

        Single row only (Sec 6.3.1) -- no i parameter, ISO 281 does not use
        one here. For a bearing with two or more rows, call this once per
        row (each with its own Z/Dw/geometry) then combine the results with
        dynamic_multirow() -- see Sec 6.4.
        """
        if Z <= 0 or Dw <= 0.0:
            raise ValueError(f"Z and Dw must both be positive; got Z={Z}, Dw={Dw}.")

        fc = cls._fc_nonzero_alpha(ri, re, Dw, gamma, lam, eta)
        cos_term = np.cos(alpha_0) ** 0.7
        tan_term = np.tan(alpha_0)

        if Dw <= cls._DW_THRESHOLD_MM:
            return fc * cos_term * tan_term * Z ** (2.0 / 3.0) * Dw ** 1.8
        return 3.647 * fc * cos_term * tan_term * Z ** (2.0 / 3.0) * Dw ** 1.4

    # ------------------------------------------------------------------
    # Sec 6.3.2 -- single row, contact angle alpha = 90deg
    # ------------------------------------------------------------------
    @classmethod
    def _fc_90deg(cls, ri: float, re: float, Dw: float, gamma: float,
                  lam: float, eta: float) -> float:
        """
        fc -- Formula (25), thrust ball bearings, alpha_0 = 90deg.

        Structurally simpler than _fc_nonzero_alpha(): the (1-gamma)/(1+gamma)
        factors drop out of both the outer gamma term (left as plain
        gamma**0.3) and the geometry bracket (left as radii_ratio**0.41,
        with no ((1-gamma)/(1+gamma))**1.72 multiplier) -- this is Formula
        (25) as given, not an approximation of Formula (20).

        Same 0,089*A1 = 98,0665 constant as _fc_nonzero_alpha() -- see that
        docstring for the 1000x transcription error this was fixed from.
        """
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma = Dw/Dpw must be in (0, 1); got {gamma}.")
        if 2.0 * ri <= Dw:
            raise ValueError(f"2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")
        if 2.0 * re <= Dw:
            raise ValueError(f"2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) -- invalid groove geometry.")

        groove_conformity = (2.0 * ri / (2.0 * ri - Dw)) ** 0.41
        radii_ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))
        bracket = radii_ratio ** 0.41
        correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

        return cls._A1_0089_N * lam * eta * groove_conformity * (gamma ** 0.3) * correction

    @classmethod
    def dynamic_90deg(cls, Z: int, Dw: float, ri: float, re: float, gamma: float,
                       lam: float, eta: float) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (23) (Dw <= 25,4 mm) / Formula (24)
        (Dw > 25,4 mm), fc from Formula (25).

        Single row only (Sec 6.3.2) -- no alpha_0 argument (fixed at 90deg
        by definition, so no cos/tan(alpha) term exists in Formula (21),
        unlike Formula (16) for the alpha != 90deg case) and no i
        parameter, same reasoning as dynamic_nonzero_alpha().
        """
        if Z <= 0 or Dw <= 0.0:
            raise ValueError(f"Z and Dw must both be positive; got Z={Z}, Dw={Dw}.")

        fc = cls._fc_90deg(ri, re, Dw, gamma, lam, eta)

        if Dw <= cls._DW_THRESHOLD_MM:
            return fc * Z ** (2.0 / 3.0) * Dw ** 1.8
        return 3.647 * fc * Z ** (2.0 / 3.0) * Dw ** 1.4

    # ------------------------------------------------------------------
    # Sec 6.4 -- thrust ball bearings with two or more rows of balls
    # ------------------------------------------------------------------
    @staticmethod
    def dynamic_multirow(Z_rows: Sequence[int], Ca_rows: Sequence[float]) -> float:
        """
        Ca [N] -- ISO 281:2007 Formula (29) (practical form of (26)-(28)
        after assuming load on a row is proportional to its ball count).

            Ca = (Z1+Z2+...+Zn) * [sum_j (Zj/Caj)**(10/3)]**(-3/10)

        Ca_rows[j] is the SINGLE-ROW Ca of row j -- obtained by calling
        dynamic_nonzero_alpha() or dynamic_90deg() once per row, with that
        row's own Z/Dw/geometry (Sec 6.3, "the appropriate single row
        thrust ball bearing formula in 6.3"). This method does not resolve
        Ca itself, same "given already-known inputs" pattern as
        RollingElementCapacity below -- the caller loops over rows, this
        just combines the results. Rows need not be identical.

        Only valid for n >= 2 rows (i > 1); a single row is
        dynamic_nonzero_alpha()/dynamic_90deg() directly, not this method.
        """
        if len(Z_rows) != len(Ca_rows):
            raise ValueError(
                f"Z_rows and Ca_rows must be the same length; got {len(Z_rows)} and {len(Ca_rows)}."
            )
        if len(Z_rows) < 2:
            raise ValueError(
                f"dynamic_multirow() needs >= 2 rows (i > 1); got {len(Z_rows)}. "
                "Use dynamic_nonzero_alpha()/dynamic_90deg() directly for a single row."
            )
        if any(Zj <= 0 for Zj in Z_rows):
            raise ValueError(f"All Z_rows must be positive; got {list(Z_rows)}.")
        if any(Caj <= 0.0 for Caj in Ca_rows):
            raise ValueError(f"All Ca_rows must be positive; got {list(Ca_rows)}.")

        total_Z = sum(Z_rows)
        bracket = sum((Zj / Caj) ** (10.0 / 3.0) for Zj, Caj in zip(Z_rows, Ca_rows))
        return total_Z * bracket ** (-3.0 / 10.0)

    @staticmethod
    def static(Z, Dw, alpha_0, reduction_factor, i=1) -> float:
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
        itself (see BearingCapacity.dynamic_nonzero_alpha() above).
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