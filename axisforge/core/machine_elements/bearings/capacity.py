# axisforge/core/machine_elements/bearings/capacity.py
"""
core/machine_elements/bearings/capacity.py

ISO 281 (dinamica, Cr/Ca) + ISO 76 (estatica, C0) + ISO/TS 16281
(Q_ci/Q_ce por elemento rolante). Fica em `core` -- ao contrario de
iso16281_contact.py (que moveu para solvers/), esta e formula fechada,
sem estado iterativo (ver ADR-001, decisao explicita "C e C0 vem do
ficheiro capacity para os niveis 3, sem mudanca").

CONTEUDO IDENTICO ao antigo families/capacity.py -- so mudou de sitio
no pacote (bearings/families/capacity.py -> bearings/capacity.py).
Nenhuma formula foi tocada.

Um calculador por (contact type, duty): cada um agrupa Cr/Ca, C0 e
Q_elements porque partilham a mesma geometria de entrada e, muitas
vezes, os mesmos termos intermedios. Contrato imposto via
RadialCapacityCalculator(ABC) / ThrustCapacityCalculator(ABC).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
import math


# =====================================================================
# ---- contratos -----------------------------------------------------
# =====================================================================

class RadialCapacityCalculator(ABC):
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


class ThrustCapacityCalculator(ABC):
    @property
    @abstractmethod
    def Ca(self) -> float:
        """Overall dynamic load rating [N] -- ISO 281."""

    @property
    def dynamic_rating(self) -> float:
        """Alias duty-agnostico para Ca -- permite a combine_multirow
        (e qualquer codigo generico) ler 'a capacidade dinamica' sem
        saber se e radial (Cr) ou thrust (Ca)."""
        return self.Ca

    @property
    @abstractmethod
    def C0(self) -> float:
        """Overall static load rating [N] -- ISO 76."""

    @property
    @abstractmethod
    def Q_elements(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- per-rolling-element capacity, ISO/TS 16281."""


# =====================================================================
# ---- multi-row helpers -----------------------------------------------
# =====================================================================

class _MultirowCombinablePointContact:
    """Mixin para calculadores de contacto PONTUAL (esferas) que
    suportam combinacao multi-fila -- ISO 1281-1:2021 Formula (29), Sec 6.4."""

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        if len(rows) < 2:
            raise ValueError(
                f"combine_multirow requires at least 2 rows; got {len(rows)}. "
                f"Use .Ca directly for a single row."
            )
        instances = [cls(**row) for row in rows]
        Ca_rows = [inst.dynamic_rating for inst in instances]
        Z_rows = [row["Z"] for row in rows]

        total_Z = sum(Z_rows)
        bracket = sum((z / ca) ** (10.0 / 3.0) for z, ca in zip(Z_rows, Ca_rows))
        return total_Z * bracket ** (-3.0 / 10.0)


class _MultirowCombinableLineContact:
    """Mixin para calculadores de contacto LINEAR (rolos) que suportam
    combinacao multi-fila -- expoentes diferentes do ponto (ISO 281
    Sec 6.4, contacto linear). Formula ainda nao derivada/confirmada."""

    @classmethod
    def combine_multirow(cls, rows: Sequence[dict]) -> float:
        if len(rows) < 2:
            raise ValueError(
                f"combine_multirow requires at least 2 rows; got {len(rows)}. "
                f"Use .Ca directly for a single row."
            )
        instances = [cls(**row) for row in rows]
        Ca_rows = [inst.dynamic_rating for inst in instances]
        Z_rows = [row["Z"] for row in rows]
        Lwe_rows = [row["Lwe"] for row in rows]

        weighted = [z * lwe for z, lwe in zip(Z_rows, Lwe_rows)]
        total_weighted = sum(weighted)
        bracket = sum((w / ca) ** (9.0 / 2.0) for w, ca in zip(weighted, Ca_rows))
        return total_weighted * bracket ** (-2.0 / 9.0)


# =====================================================================
# ---- point contact / radial duty --------------------------------------
# =====================================================================

class PointContactCapacityRadial(RadialCapacityCalculator):
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
        if "_q_bracket" not in self._cache:
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))
            self._cache["_q_bracket"] = (1.044 * ((1.0 - self.gamma) / (1.0 + self.gamma)) ** 1.72
                                          * radii_ratio ** 0.41)
        return self._cache["_q_bracket"]

    @property
    def Cr(self) -> float:
        cos_term = (self.i * math.cos(self.alpha_0)) ** 0.7
        if self.Dw <= self._DW_THRESHOLD_MM:
            return self._fc * cos_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * cos_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.4

    @property
    def C0(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        Cr = self.Cr
        cos_alpha_07 = np.cos(self.alpha_0) * self.i ** 0.7
        Q_ci = (Cr / (0.407 * self.Z * cos_alpha_07)) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Cr / (0.389 * self.Z * cos_alpha_07)) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce


# =====================================================================
# ---- point contact / thrust duty --------------------------------------
# =====================================================================

class PointContactCapacityThrust_Non_90deg(ThrustCapacityCalculator, _MultirowCombinablePointContact):
    _A1_0089_N = 98.0665
    _DW_THRESHOLD_MM = 25.4

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
            return self._fc * cos_term * tan_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * cos_term * tan_term * self.Z ** (2.0 / 3.0) * self.Dw ** 1.4

    @property
    def _q_bracket(self) -> float:
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
        Ca = self.Ca
        sin_alpha_0 = np.sin(self.alpha_0)
        Q_ci = (Ca / (self.Z * sin_alpha_0)) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / (self.Z * sin_alpha_0)) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce


class PointContactCapacityThrust_90deg(ThrustCapacityCalculator, _MultirowCombinablePointContact):
    _A1_0089_N = 98.0665
    _DW_THRESHOLD_MM = 25.4

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
            return self._fc * self.Z ** (2.0 / 3.0) * self.Dw ** 1.8
        return 3.647 * self._fc * self.Z ** (2.0 / 3.0) * self.Dw ** 1.4

    @property
    def _q_bracket(self) -> float:
        if "_q_bracket" not in self._cache:
            radii_ratio = (self.ri / self.re) * ((2.0 * self.re - self.Dw) / (2.0 * self.ri - self.Dw))
            self._cache["_q_bracket"] = radii_ratio ** 0.41
        return self._cache["_q_bracket"]

    @property
    def C0(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    @property
    def Q_elements(self) -> tuple[float, float]:
        Ca = self.Ca
        Q_ci = (Ca / self.Z) * (1.0 + self._q_bracket ** (10.0 / 3.0)) ** 0.3
        Q_ce = (Ca / self.Z) * (1.0 + self._q_bracket ** (-10.0 / 3.0)) ** 0.3
        return Q_ci, Q_ce


# =====================================================================
# ---- line contact / radial duty ---------------------------------------
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
        return self._fc * cos_term * self.Z ** (3.0 / 4.0) * self.Dwe ** (29.0 / 27.0)

    @property
    def C0(self) -> float:
        raise NotImplementedError

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
# ---- line contact / thrust duty ---------------------------------------
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
    _B1_041_N = 472.45388

    def __init__(self, Z: int, Dwe: float, Lwe: float, alpha_0: float,
                 gamma: float, lambda_v: float, eta: float, i: int = 1):
        if Z <= 0 or Dwe <= 0.0 or Lwe <= 0 or i <= 0:
            raise ValueError(f"Z, Dwe, Lwe and i must all be positive; got Z={Z}, Dwe={Dwe}, Lwe={Lwe}, i={i}.")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}.")

        self.Z, self.Dwe, self.Lwe, self.alpha_0 = Z, Dwe, Lwe, alpha_0
        self.gamma, self.lambda_v, self.eta, self.i = gamma, lambda_v, eta, i
        self._cache: dict[str, float] = {}

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
        return Q, Q

    @property
    def per_lamina(self) -> tuple[float, float]:
        Q_ci, Q_ce = self.Q_elements
        q_ci = Q_ci * (1.0 / self.n_s) ** (7.0 / 9.0)
        q_ce = Q_ce * (1.0 / self.n_s) ** (7.0 / 9.0)
        return q_ci, q_ce