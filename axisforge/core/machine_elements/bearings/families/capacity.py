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


# =====================================================================
# ---- contracts -----------------------------------------------------
# =====================================================================

# ---------------------------------------------------------------------
# ---- radial duty ----------------------------------------------------
# ---------------------------------------------------------------------

class RadialCapacityCalculator(ABC):
    """Contract every capacity calculator must satisfy. Tagged by
    CONTACT_TYPE/DUTY for identification -- lets a caller filter/assert
    on which physical regime a given instance covers."""

    @property
    @abstractmethod
    def Cr(self) -> float:
        """Overall dynamic load rating [N] -- ISO 281."""

    @property
    def dynamic_rating(self) -> float:
        return self.Cr
    
    @property
    @abstractmethod
    def C0(self) -> float:
        """Overall static load rating [N] -- ISO 76."""

    @property
    @abstractmethod
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- per-rolling-element capacity, ISO/TS 16281."""

# ---------------------------------------------------------------------
# ---- thrust duty ----------------------------------------------------
# ---------------------------------------------------------------------

class ThrustCapacityCalculator(ABC):
    """Contract every capacity calculator must satisfy. Tagged by
    CONTACT_TYPE/DUTY for identification -- lets a caller filter/assert
    on which physical regime a given instance covers."""

    @property
    @abstractmethod
    def Ca(self) -> float:
        """Overall dynamic load rating [N] -- ISO 281."""

    @property
    def dynamic_rating(self) -> float:
        """Alias duty-agnóstico para Ca -- permite a combine_multirow
        (e a qualquer código genérico) ler 'a capacidade dinâmica'
        sem saber se é radial (Cr) ou thrust (Ca)."""
        return self.Ca

    @property
    @abstractmethod
    def C0(self) -> float:
        """Overall static load rating [N] -- ISO 76."""

    @property
    @abstractmethod
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- per-rolling-element capacity, ISO/TS 16281."""

# ---------------------------------------------------------------------
# ---- Multi row helper -----------------------------------------------
# ---------------------------------------------------------------------


class _MultirowCombinablePointContact:
    """Mixin for POINT-contact (ball) capacity classes that support
    multi-row combination -- ISO 1281-1:2021 Formula (29), Sec 6.4."""

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        if len(rows) < 2:
            raise ValueError(...)
        instances = [cls(**row) for row in rows]
        Ca_rows = [inst.dynamic_rating for inst in instances]
        Z_rows = [row["Z"] for row in rows]
        total_Z = sum(Z_rows)
        bracket = sum((z / ca) ** (10.0 / 3.0) for z, ca in zip(Z_rows, Ca_rows))
        return total_Z * bracket ** (-3.0 / 10.0)


class _MultirowCombinableLineContact:
    """Mixin for LINE-contact (roller) capacity classes that support
    multi-row combination -- expoentes diferentes do ponto (ISO 281
    Sec 6.4, contacto linear). Fórmula ainda não derivada."""

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        if len(rows) < 2:
            raise ValueError(...)
        instances = [cls(**row) for row in rows]
        Ca_rows = [inst.dynamic_rating for inst in instances]
        Z_rows = [row["Z"] for row in rows]
        Lwe_rows = [row["Lwe"] for row in rows]

        weighted = [z * lwe for z, lwe in zip(Z_rows, Lwe_rows)]
        total_weighted = sum(weighted)
        bracket = sum((w / ca) ** (9.0 / 2.0) for w, ca in zip(weighted, Ca_rows))
        return total_weighted * bracket ** (-2.0 / 9.0)
    
# =====================================================================
# ---- point contact / radial duty-------------------------------------
# =====================================================================

class PointContactCapacityRadial(RadialCapacityCalculator):
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


# =====================================================================
# ---- point contact / thrust duty-------------------------------------
# =====================================================================

class PointContactCapacityThrust_Non_90deg(ThrustCapacityCalculator, _MultirowCombinablePointContact):

    _A1_0089_N = 98.0665      # "0,089*A1", Formula (20)/(25) note -- Ca in N, Dw in mm
    _DW_THRESHOLD_MM = 25.4   # Formula (18)/(19) and (23)/(24) switch point

    def __init__(self, Z: int, Dw: float, alpha_0: float, ri: float, re: float,
                 gamma: float, lam: float, eta: float, i: int = 1):
        if Z <= 0 or Dw <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dw and i must all be positive; got Z={Z}, Dw={Dw}, i={i}.")
        if 2.0 * ri <= Dw or 2.0 * re <= Dw:
            raise ValueError("groove radius too small relative to Dw.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dw, self.alpha_0 = Z, Dw, alpha_0
        self.ri, self.re, self.gamma = ri, re, gamma
        self.lam, self.eta, self.i = lam, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        if "_fc" not in self._cache:
            groove_conformity = (2.0 * self.ri / (2.0 * self.ri - self.Dw)) ** 0.41
            gamma_term = self.gamma ** 0.3 * (1.0 - self.gamma) ** 1.39 / (1.0 + self.gamma) ** (1.0 / 3.0)

            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))

            bracket = radii_ratio ** 0.41 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
            correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

            self._cache["_fc"] = (self._A1_0089_N * self.lam * self.eta
                                   * groove_conformity * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        cos_term = np.cos(self.alpha_0) ** 0.7
        tan_term = np.tan(self.alpha_0)
        if self.Dw <= self._DW_THRESHOLD_MM:
            return self._fc * cos_term * tan_term * self.Z ** (2.0/3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * cos_term * tan_term * self.Z ** (2.0/3.0) * self.Dw ** 1.4 

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
    def C0(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) -- needs Cr, so this reuses .Cr rather than
        taking it as a separate constructor argument; pass an
        override via a subclass/param if a caller ever needs a Cr
        other than this instance's own computed value."""
        Ca = self.Ca
        sin_alpha_0 = np.sin(self.alpha_0)
        Q_ci = (Ca / (self.Z * sin_alpha_0)) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / (self.Z * sin_alpha_0)) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce


class PointContactCapacityThrust_90deg(ThrustCapacityCalculator, _MultirowCombinablePointContact):

    _A1_0089_N = 98.0665      # "0,089*A1", Formula (20)/(25) note -- Ca in N, Dw in mm
    _DW_THRESHOLD_MM = 25.4   # Formula (18)/(19) and (23)/(24) switch point

    def __init__(self, Z: int, Dw: float, alpha_0: float, ri: float, re: float,
                 gamma: float, lam: float, eta: float, i: int = 1):
        if Z <= 0 or Dw <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dw and i must all be positive; got Z={Z}, Dw={Dw}, i={i}.")
        if 2.0 * ri <= Dw or 2.0 * re <= Dw:
            raise ValueError("groove radius too small relative to Dw.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dw, self.alpha_0 = Z, Dw, alpha_0
        self.ri, self.re, self.gamma = ri, re, gamma
        self.lam, self.eta, self.i = lam, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        if "_fc" not in self._cache:
            groove_conformity = (2.0 * self.ri / (2.0 * self.ri - self.Dw)) ** 0.41
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))

            bracket = radii_ratio ** 0.41
            correction = (1.0 + bracket ** (10.0 / 3.0)) ** (-3.0 / 10.0)

            self._cache["_fc"] = (self._A1_0089_N * self.lam * self.eta
                                   * groove_conformity * (self.gamma ** 0.3) * correction)
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        if self.Dw <= self._DW_THRESHOLD_MM:
            return self._fc * self.Z ** (2.0/3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * self.Z ** (2.0/3.0) * self.Dw ** 1.4 

    @property
    def _q_bracket(self) -> float:
        """Distinct geometry bracket used by Q_ci/Q_ce -- ISO/TS 16281
        eq.(19)-(20), not the same numeric expression as _fc's bracket
        even though it looks similar (different exponent context)."""
        if "_q_bracket" not in self._cache:
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))
            self._cache["_q_bracket"] = radii_ratio ** 0.41
        return self._cache["_q_bracket"]

    @property
    def C0(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) -- needs Cr, so this reuses .Cr rather than
        taking it as a separate constructor argument; pass an
        override via a subclass/param if a caller ever needs a Cr
        other than this instance's own computed value."""
        Ca = self.Ca
        Q_ci = (Ca / self.Z) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / self.Z) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce

# =====================================================================
# ---- line contact / radial duty--------------------------------------
# =====================================================================


class LineContactCapacityRadial(RadialCapacityCalculator):

    _B1_0483_N = 551.13373

    def __init__(self, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                 gamma: float, lambda_v: float, n_s: float, i: int = 1):
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0.0 or i <= 0:
            raise ValueError(f"Z, Dwe, Lwe and i must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dwe, self.Lwe, self.alpha_0 = Z, Dwe, Lwe, alpha_0
        self.gamma, self.lambda_v, self.n_s, self.i = gamma, lambda_v, n_s, i
        self._cache: dict[str, float] = {}

    @property
    def _fc(self) -> float:
        if "_fc" not in self._cache:
            gamma_term = (self.gamma ** (2.0 / 9.0) * (1.0 - self.gamma) ** (29.0 / 27.0)
                        / (1.0 + self.gamma) ** (1.0 / 4.0))
            bracket = 1.04 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)
            correction = (1.0 + bracket ** (9.0 / 2.0)) ** (-2.0 / 9.0)

            self._cache["_fc"] = (self._B1_0483_N * 0.377 * self.lambda_v * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def Cr(self) -> float:
        cos_term = (self.i * self.Lwe * np.cos(self.alpha_0)) ** (7.0 / 9.0)
        return self._fc * cos_term * self.Z ** (3.0/4.0) * self.Dwe ** (29.0 / 27.0)

    @property
    def C0(self) -> float: raise NotImplementedError

    @property
    def _numer(self) -> float:
        return 1.038 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)

    @property
    def _denom(self) -> float:
        return np.cos(self.alpha_0) * (self.i ** (7.0 / 9.0))

    @property
    def Q_elements(self) -> tuple[float, float]:
        Q_ci = (1.0 / self.lambda_v) * (self.Cr / (0.378 * self.Z * self._denom)) * (
            1.0 + self._numer ** (9.0 / 2.0)) ** (2.0 / 9.0)
        Q_ce = (1.0 / self.lambda_v) * (self.Cr / (0.364 * self.Z * self._denom)) * (
            1.0 + self._numer ** (-9.0 / 2.0)) ** (2.0 / 9.0)
        return Q_ci, Q_ce

    @property
    def per_lamina(self) -> tuple[float, float]:
        Q_ci, Q_ce = self.Q_elements
        q_ci = Q_ci * (1.0 / self.n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / self.n_s) ** (7.0 / 9.0)
        return q_ci, q_ce

# =====================================================================
# ---- line contact / thrust duty--------------------------------------
# =====================================================================

class LineContactCapacityThrust_Non_90deg(ThrustCapacityCalculator, _MultirowCombinableLineContact):

    _B1_0483_N = 551.13373     

    def __init__(self, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                 gamma: float, lambda_v: float, eta: float, i: int = 1):
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0 or i <= 0:
            raise ValueError(f"Z, Dwe, Lwe and i must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dwe, self.Lwe, self.alpha_0 = Z, Dwe, Lwe, alpha_0
        self.gamma, self.lambda_v, self.eta, self.i = gamma, lambda_v, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        if "_fc" not in self._cache:
            gamma_term = (self.gamma ** (2.0 / 9.0) * (1.0 - self.gamma) ** (29.0 / 27.0)
                          / (1.0 + self.gamma) ** (1.0 / 4.0))

            bracket = ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)
            correction = (1.0 + bracket ** (9.0 / 2.0)) ** (-2.0 / 9.0)

            self._cache["_fc"] = (self._B1_0483_N * self.lambda_v * self.eta
                                   * gamma_term * correction)
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        cos_term = (self.Lwe * np.cos(self.alpha_0)) ** (7.0 / 9.0)
        tan_term = np.tan(self.alpha_0)
        return self._fc * cos_term * tan_term * self.Z ** (3.0 / 4.0) * self.Dwe ** (29.0 / 27.0) 

    @property
    def _numer(self) -> float:
        return ((1.0 - self.gamma) / (1.0 + self.gamma)) ** (143.0 / 108.0)

    @property
    def _denom(self) -> float:
        return (self.Z * np.sin(self.alpha_0))

    @property
    def C0(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) -- needs Ca, so this reuses .Ca rather than
        taking it as a separate constructor argument; pass an
        override via a subclass/param if a caller ever needs a Ca
        other than this instance's own computed value."""
        Q_ci = (1.0 / self.lambda_v) * (self.Ca / self._denom) * (
            1.0 + self._numer ** (9.0 / 2.0)) ** (2.0 / 9.0)
        
        Q_ce = (1.0 / self.lambda_v) * (self.Ca / self._denom) * (
            1.0 + self._numer ** (-9.0 / 2.0)) ** (2.0 / 9.0)
        return Q_ci, Q_ce

    @property
    def per_lamina(self) -> tuple[float, float]:
        Q_ci, Q_ce = self.Q_elements
        q_ci = Q_ci * (1.0 / self.n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / self.n_s) ** (7.0 / 9.0)
        return q_ci, q_ce


class LineContactCapacityThrust_90deg(ThrustCapacityCalculator, _MultirowCombinableLineContact):

    _B1_041_N  = 472.45388   

    def __init__(self, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                 gamma: float, lambda_v: float, eta: float, i: int = 1):
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0 or i <= 0:
            raise ValueError(f"Z, Dwe, Lwe and i must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dwe, self.Lwe, self.alpha_0 = Z, Dwe, Lwe, alpha_0
        self.gamma, self.lambda_v, self.eta, self.i = gamma, lambda_v, eta, i
        self._cache: dict[str, float] = {}

    # -- shared intermediate terms, cached, reused by Ca and (indirectly) C0 --

    @property
    def _fc(self) -> float:
        if "_fc" not in self._cache:

            self._cache["_fc"] = (self._B1_041_N * self.lambda_v * self.eta
                                   * self.gamma ** (2.0 / 9.0))
        return self._cache["_fc"]

    @property
    def Ca(self) -> float:
        return self._fc * self.Lwe ** (7.0 / 9.0) * self.Z ** (3.0 / 4.0) * self.Dwe ** (29.0 / 27.0) 

    @property
    def C0(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        Q = (1.0 / self.lambda_v) * (self.Ca / self.Z) * 2.0 ** (2.0 / 9.0)
        Q_ci, Q_ce = Q 
        return Q_ci, Q_ce

    @property
    def per_lamina(self) -> tuple[float, float]:
        Q_ci, Q_ce = self.Q_elements
        q_ci = Q_ci * (1.0 / self.n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / self.n_s) ** (7.0 / 9.0)
        return q_ci, q_ce