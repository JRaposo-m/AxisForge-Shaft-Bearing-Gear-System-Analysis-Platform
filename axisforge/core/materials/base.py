"""
core/materials/base.py  -- ESBOCO (base geral de definicao de materiais)

Nao conhece elementos de maquinas (veios, engrenagens, rolamentos): so
define o que e um material. Cada elemento le daqui o que precisa; os dados
especificos de uma norma de um elemento (ex. sigma_Hlim da ISO 6336-5)
vivem com esse elemento e referenciam o material por id.

Um Material compoe blocos de propriedades independentes:

    Material
      |- density
      |- elastic   : ElasticBehavior (abstrata)     <- isotropico ou nao
      |     |- IsotropicElastic        E, poisson_ratio
      |     |- OrthotropicElastic      E1..E3, G12..G23, nu12..nu23
      |- strength  : StrengthProperties  (opcional)  Sut, Sy, Se
      `- thermal   : ThermalProperties   (opcional)  cp, k_thermal, alpha

Crescer = acrescentar uma subclasse de ElasticBehavior (ex.
TransverselyIsotropic) ou um novo bloco opcional (fadiga, fluencia,
temperatura, ...), sem tocar no que ja existe.

Unidades: E, G, tensoes [MPa]; density [kg/m^3]; cp [J/(kg.K)];
k_thermal [W/(m.K)]; alpha [1/K].
Requer Python >= 3.10 (kw_only).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


def _raise_if(errors: list[str]) -> None:
    if errors:
        raise ValueError("\n".join(errors))


# =====================================================================
# ---- Comportamento elastico ----------------------------------------
# =====================================================================

@dataclass(frozen=True)
class ElasticBehavior(ABC):
    """Contrato: cada comportamento elastico diz se e isotropico e valida-se."""

    def __post_init__(self) -> None:
        _raise_if(self.errors())

    @property
    @abstractmethod
    def is_isotropic(self) -> bool: ...

    @abstractmethod
    def errors(self) -> list[str]: ...


@dataclass(frozen=True)
class IsotropicElastic(ElasticBehavior):
    """E [MPa], poisson_ratio [-] (0 < nu < 0.5). Sem default para nu."""
    E: float
    poisson_ratio: float

    is_isotropic = True

    @property
    def G(self) -> float:
        """Modulo de corte [MPa]."""
        return self.E / (2.0 * (1.0 + self.poisson_ratio))

    def errors(self) -> list[str]:
        errs = []
        if self.E <= 0:
            errs.append(f"IsotropicElastic: E must be > 0, got {self.E}")
        if not 0.0 < self.poisson_ratio < 0.5:
            errs.append(f"IsotropicElastic: poisson_ratio must be in (0, 0.5), "
                        f"got {self.poisson_ratio}")
        return errs


@dataclass(frozen=True)
class OrthotropicElastic(ElasticBehavior):
    """Ortotropico: 3 moduli, 3 de corte, 3 coeficientes de Poisson.
    Por agora so valida positividade; a verificacao de estabilidade da
    matriz de flexibilidade fica para quando for usado."""
    E1: float
    E2: float
    E3: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float

    is_isotropic = False

    def errors(self) -> list[str]:
        errs = []
        for name in ("E1", "E2", "E3", "G12", "G13", "G23"):
            if getattr(self, name) <= 0:
                errs.append(f"OrthotropicElastic: {name} must be > 0, got {getattr(self, name)}")
        return errs


# =====================================================================
# ---- Blocos opcionais de propriedades -------------------------------
# =====================================================================

@dataclass(frozen=True)
class StrengthProperties:
    """Resistencia [MPa]. Sut rotura, Sy cedencia, Se limite de fadiga
    (valor guardado; regras para o estimar vivem no elemento que as usa)."""
    Sut: Optional[float] = None
    Sy: Optional[float] = None
    Se: Optional[float] = None

    def __post_init__(self) -> None:
        errs = [f"StrengthProperties: {n} must be > 0, got {v}"
                for n, v in (("Sut", self.Sut), ("Sy", self.Sy), ("Se", self.Se))
                if v is not None and v <= 0]
        if self.Sut is not None and self.Sy is not None and self.Sy > self.Sut:
            errs.append(f"StrengthProperties: Sy ({self.Sy}) cannot exceed Sut ({self.Sut})")
        _raise_if(errs)


@dataclass(frozen=True)
class ThermalProperties:
    """cp [J/(kg.K)], k_thermal [W/(m.K)], alpha [1/K]."""
    cp: Optional[float] = None
    k_thermal: Optional[float] = None
    alpha: Optional[float] = None

    def __post_init__(self) -> None:
        errs = [f"ThermalProperties: {n} must be > 0, got {v}"
                for n, v in (("cp", self.cp), ("k_thermal", self.k_thermal), ("alpha", self.alpha))
                if v is not None and v <= 0]
        _raise_if(errs)


# =====================================================================
# ---- Material --------------------------------------------------------
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class Material:
    material_id: str
    density: float
    elastic: ElasticBehavior
    strength: Optional[StrengthProperties] = None
    thermal: Optional[ThermalProperties] = None
    description: str = ""

    def __post_init__(self) -> None:
        errs = []
        if not self.material_id:
            errs.append("Material: material_id must be non-empty")
        if self.density <= 0:
            errs.append(f"Material({self.material_id!r}): density must be > 0, got {self.density}")
        _raise_if(errs)

    # --- acesso pelo que cada analise exige --------------------------
    def require_isotropic(self) -> IsotropicElastic:
        if not isinstance(self.elastic, IsotropicElastic):
            raise TypeError(f"Material '{self.material_id}' is not isotropic "
                            f"({type(self.elastic).__name__}).")
        return self.elastic

    @property
    def E(self) -> float:
        return self.require_isotropic().E

    @property
    def poisson_ratio(self) -> float:
        return self.require_isotropic().poisson_ratio

    def require_strength(self, *fields: str) -> StrengthProperties:
        if self.strength is None or any(getattr(self.strength, f) is None for f in fields):
            raise ValueError(f"Material '{self.material_id}': missing strength "
                             f"data {fields or ''}")
        return self.strength

    def require_thermal(self, *fields: str) -> ThermalProperties:
        if self.thermal is None or any(getattr(self.thermal, f) is None for f in fields):
            raise ValueError(f"Material '{self.material_id}': missing thermal "
                             f"data {fields or ''}")
        return self.thermal


# =====================================================================
# ---- Registo / lookup ------------------------------------------------
# =====================================================================
# As bibliotecas (steels.py, cast_iron.py, polymers.py, ...) criam as
# instancias e chamam register(); o __init__.py importa-as.

_REGISTRY: dict[str, Material] = {}


def register(*materials: Material) -> None:
    for m in materials:
        if m.material_id in _REGISTRY:
            raise ValueError(f"Material '{m.material_id}' already registered")
        _REGISTRY[m.material_id] = m


def get_material(material_id: str) -> Material:
    if material_id not in _REGISTRY:
        raise KeyError(f"Material '{material_id}' not found. "
                       f"Available: {', '.join(_REGISTRY)}")
    return _REGISTRY[material_id]


def available_materials() -> list[str]:
    return list(_REGISTRY)


__all__ = [
    "ElasticBehavior", "IsotropicElastic", "OrthotropicElastic",
    "StrengthProperties", "ThermalProperties", "Material",
    "register", "get_material", "available_materials",
]