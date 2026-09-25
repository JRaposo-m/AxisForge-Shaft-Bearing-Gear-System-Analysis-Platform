# axisforge/core/machine_elements/bearings/family.py
"""
core/machine_elements/bearings/family.py

BearingFamily(Bearing, ABC) -- nivel 2 da hierarquia (ADR-001): funde o
antigo BearingCatalog (d, D, b, C, C0) com o antigo contrato
BearingFamily(ABC) (assemble_geometry / dynamic_capacity / ...). Cada
subtipo concreto (types/*.py) subclassa esta classe diretamente e
constroi + valida tudo em __init__ -- ja nao ha passo .assemble()
separado nem um Bearing.assemble(family, catalog, geometry) externo.

`isinstance(bearing, BearingFamily)` e a forma sancionada de um
chamador que recebe um `Bearing` generico (nivel 1) verificar/aceder
aos dados completos deste nivel.

O registo (_FAMILY_REGISTRY / register_family) sobrevive do ficheiro
antigo, inalterado em espirito -- dispatch dinamico por nome de classe
sem cada chamador ter de importar cada subtipo.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Optional

from axisforge.core.machine_elements.bearings.base import Bearing, BearingArrangement, BearingType
from axisforge.core.machine_elements.bearings.bearing_properties import SurfacePair


# =====================================================================
# ---- registo -----------------------------------------------------
# =====================================================================

_FAMILY_REGISTRY: dict[str, type["BearingFamily"]] = {}


def register_family(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    _FAMILY_REGISTRY[cls.__name__] = cls
    return cls


def get_family(name: str) -> type["BearingFamily"]:
    return _FAMILY_REGISTRY[name]


# =====================================================================
# ---- contrato -----------------------------------------------------
# =====================================================================

class BearingFamily(Bearing, ABC):
    """Nivel 2 -- acrescenta dimensoes de catalogo (d, D, b), override
    de capacidade (C, C0), os materiais (surfaces) e o contrato
    geometrico/fisico que cada subtipo concreto tem de cumprir.

    C / C0: se fornecidos diretamente no construtor (ex.: valor de
    catalogo do fabricante), sao usados tal-e-qual; se None, sao
    calculados sob demanda via .dynamic_capacity()/.static_capacity()
    (ISO 281 / ISO 76, capacity.py) -- nunca cacheados aqui, para nunca
    desincronizar de geometria que so existe depois de __init__ acabar.
    """

    BEARING_TYPE: ClassVar[BearingType | None] = None
    DUTY: ClassVar[Literal["radial", "thrust"] | None] = None

    def __init__(self, *, position: float, arrangement: BearingArrangement,
                 d: float, D: float, b: float, surfaces: SurfacePair,
                 C: Optional[float] = None, C0: Optional[float] = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement,
                          label=label, designation=designation)

        tag = label or designation or self.__class__.__name__
        if d <= 0:
            raise ValueError(f"{tag}: d must be > 0, got {d}")
        if D <= d:
            raise ValueError(f"{tag}: D must be > d, got D={D}, d={d}")
        if b < 0:
            raise ValueError(f"{tag}: b must be >= 0, got {b}")

        self.d, self.D, self.b = d, D, b
        self.dm = 0.5 * (d + D)
        self.surfaces = surfaces
        self._C_override, self._C0_override = C, C0

    # -- C/C0: override de catalogo OU ISO 281/76 sob demanda ----------

    @property
    def C(self) -> float:
        return self._C_override if self._C_override is not None else self.dynamic_capacity()

    @property
    def C0(self) -> float:
        return self._C0_override if self._C0_override is not None else self.static_capacity()

    # -- contrato geometrico (pura geometria, sem material) -------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'deep_groove_ball'."""

    @abstractmethod
    def curvature_rolling_element(self) -> tuple[float, float]:
        """r1 -- (Rx, Ry) do elemento rolante, pronto para Hertz
        (slippy.hertz_full)."""

    @abstractmethod
    def curvature_inner_raceway(self) -> tuple[float, float]:
        """r2 (via interior) -- (Rx, Ry), pronto para Hertz."""

    @abstractmethod
    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        """r2 (via exterior) -- (Rx, Ry), ou None para tipos sem via
        exterior (ex.: aplicacoes magneticas)."""

    # -- contrato de capacidade (ISO 281/76 -- capacity.py) -------------

    @abstractmethod
    def dynamic_capacity(self) -> float:
        """Cr/Ca [N] -- ISO 281. So invocada quando C=None."""

    @abstractmethod
    def static_capacity(self) -> float:
        """C0 [N] -- ISO 76. So invocada quando C0=None."""

    @abstractmethod
    def per_element_dynamic_capacity(self) -> tuple[float, float]:
        """(Q_ci, Q_ce) [N] -- ISO/TS 16281, por elemento rolante."""