# =====================================================================
# families/capacity.py
# =====================================================================
"""
core/machine_elements/bearings/families/capacity.py

One capacity calculator per (contact type, duty) combination -- each
bundles dynamic (Cr), static (C0), and per-element (Q_ci, Q_ce)
together, since they share the same geometry inputs and often the same
intermediate terms. Contract enforced via CapacityCalculator(ABC).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
import math


class CapacityCalculator(ABC):
    """Contract every capacity calculator must satisfy. Tagged by
    CONTACT_TYPE/DUTY for identification -- lets a caller filter/assert
    on which physical regime a given instance covers."""

    @property
    @abstractmethod
    def Cr(self) -> float:
        """Overall dynamic load rating [N] -- ISO 281."""

    @property
    @abstractmethod
    def C0(self) -> float:
        """Overall static load rating [N] -- ISO 76."""

    @property
    @abstractmethod
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- per-rolling-element capacity, ISO/TS 16281."""


class _MultirowCombinablePointContact:
    """Mixin for POINT-contact (ball) capacity classes that support
    multi-row combination -- ISO 1281-1:2021 Formula (29), Sec 6.4."""

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        if len(rows) < 2:
            raise ValueError("combine_multirow requires at least 2 rows; use .Cr directly for a single row.")

        instances = [cls(**row) for row in rows]
        Cr_rows = [inst.Cr for inst in instances]
        Z_rows = [row["Z"] for row in rows]

        total_Z = sum(Z_rows)
        bracket = sum(cr ** (-10.0 / 3.0) for cr in Cr_rows)
        return total_Z ** (2.0 / 3.0) * bracket ** (-3.0 / 10.0)


class _MultirowCombinableLineContact:
    """Mixin for LINE-contact (roller) capacity classes that support
    multi-row combination -- expoentes diferentes do ponto (ISO 281
    Sec 6.4, contacto linear). Fórmula ainda não derivada."""

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        raise NotImplementedError("line-contact multi-row combination not derived yet")


# ---------------------------------------------------------------------
# ---- point contact / radial duty -------------------------------------
# ---------------------------------------------------------------------

class PointContactCapacityRadial(CapacityCalculator):
    """Ball bearings, radial duty. Instantiate once with geometry;
    read .Cr / .C0 / .Q_elements as needed -- shared terms (_fc,
    _bracket) are computed once each, not duplicated per property."""

    _A1_0089_N = 98.0665
    _DW_THRESHOLD_MM = 25.4

    def __init__(self, Z: int, Dw: float, alpha_0: float, ri: float, re: float,
                 gamma: float, reduction_factor: float, i: int = 1):
        if Z <= 0 or Dw <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dw and i must all be positive; got Z={Z}, Dw={Dw}, i={i}.")
        if 2.0 * ri <= Dw or 2.0 * re <= Dw:
            raise ValueError("groove radius too small relative to Dw.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dw, self.alpha_0 = Z, Dw, alpha_0
        self.ri, self.re, self.gamma = ri, re, gamma
        self.reduction_factor, self.i = reduction_factor, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Cr and (indirectly) C0 --
    @property
    def _fc(self) -> float:
        if "_fc" not in self._cache:
            groove_conformity = (2.0 * self.ri / (2.0 * self.ri - self.Dw)) ** 0.41
            gamma_term = self.gamma ** 0.3 * (1.0 - self.gamma) ** 1.39 / (1.0 + self.gamma) ** (1.0 / 3.0)

            if math.isinf(self.re):
                radii_ratio = 2.0 * self.ri / (2.0 * self.ri - self.Dw)
            else:
                radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))

            bracket = 1.04 * radii_ratio ** 0.41 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
            correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

            self._cache["_fc"] = (self._A1_0089_N * 0.41 * self.reduction_factor
                                   * groove_conformity * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def _q_bracket(self) -> float:
        """Distinct geometry bracket used by Q_ci/Q_ce -- ISO/TS 16281
        eq.(19)-(20), not the same numeric expression as _fc's bracket
        even though it looks similar (different exponent context)."""
        if "_q_bracket" not in self._cache:
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))
            self._cache["_q_bracket"] = (1.044 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
                                          * radii_ratio ** 0.41)
        return self._cache["_q_bracket"]

    @property
    def Cr(self) -> float:
        """Formula (13)/(14) -- switches on Dw <= 25.4 mm."""
        cos_term = (self.i * math.cos(self.alpha_0)) ** 0.7
        if self.Dw <= self._DW_THRESHOLD_MM:
            return self._fc * cos_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * cos_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.4

    @property
    def C0(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) -- needs Cr, so this reuses .Cr rather than
        taking it as a separate constructor argument; pass an
        override via a subclass/param if a caller ever needs a Cr
        other than this instance's own computed value."""
        Cr = self.Cr
        cos_alpha_07 = np.cos(self.alpha_0) * self.i ** 0.7
        Q_ci = (Cr / (0.407 * self.Z * cos_alpha_07)) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Cr / (0.389 * self.Z * cos_alpha_07)) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce


# ---------------------------------------------------------------------
# ---- point contact / thrust duty ---------------------
# ---------------------------------------------------------------------

class PointContactCapacityThrust_Non_90deg(CapacityCalculator, _MultirowCombinablePointContact):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Thrust-duty point-contact capacity not derived yet.")

    @property
    def Cr(self) -> float: raise NotImplementedError

    @property
    def C0(self) -> float: raise NotImplementedError

    @property
    def Q_elements(self) -> tuple[float, float]: raise NotImplementedError


class PointContactCapacityThrust_90deg(CapacityCalculator, _MultirowCombinablePointContact):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Thrust-duty point-contact capacity not derived yet.")

    @property
    def Cr(self) -> float: raise NotImplementedError

    @property
    def C0(self) -> float: raise NotImplementedError

    @property
    def Q_elements(self) -> tuple[float, float]: raise NotImplementedError




class LineContactCapacityRadial(CapacityCalculator):
    CONTACT_TYPE, DUTY = "line", "radial"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Line-contact (radial) capacity not derived yet.")

    @property
    def Cr(self) -> float: raise NotImplementedError

    @property
    def C0(self) -> float: raise NotImplementedError

    @property
    def Q_elements(self) -> tuple[float, float]: raise NotImplementedError


class LineContactCapacityThrust(CapacityCalculator):
    CONTACT_TYPE, DUTY = "line", "thrust"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Line-contact (thrust) capacity not derived yet.")

    @property
    def Cr(self) -> float: raise NotImplementedError

    @property
    def C0(self) -> float: raise NotImplementedError

    @property
    def Q_elements(self) -> tuple[float, float]: raise NotImplementedError