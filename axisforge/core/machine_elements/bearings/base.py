# axisforge/core/machine_elements/bearings/base.py
"""
axisforge/core/machine_elements/bearings/base.py

BearingType, BearingCatalog, BearingFamily -- os tres primitivos de base
de que tudo em bearings/ depende. Vivem juntos porque sao pequenos,
estaveis, e mutuamente acoplados (BearingFamily.assemble_geometry le
BearingCatalog; BearingFamily.BEARING_TYPE e um BearingType) -- nao
porque o pacote precise de um ficheiro so para eles por regra.

bearings/__init__.py reexporta os tres diretamente daqui (import
simples, sem __getattr__/lazy -- sao baratos, nao ha ganho em adiar).
Bearing (bearing.py) continua lazy la, por ser mais pesado.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal


# =====================================================================
# ---- BearingType -----------------------------------------------------
# =====================================================================

class BearingType(str, Enum):
    """Informational label + fonte única de verdade sobre a duty
    (radial/thrust) de cada tipo -- ver `.duty` abaixo. Herdar de `str`
    em vez de usar `auto()`: o projeto persiste em SQLite, e um inteiro
    de `auto()` desloca-se silenciosamente se algum dia inseres um
    membro no meio da lista; uma string é estável para sempre.

    Dispatch no lado core/ já não é feito por este enum (ver
    BearingFamily) -- sobrevive como label para relatórios/GUI
    agruparem por família sem importar cada classe, e para as tabelas de
    dispatch do lado dos solvers (ISO_16281/rolling_bearing_solver.py)
    continuarem a funcionar sem alteração.
    """
    DEEP_GROOVE_BALL          = "deep_groove_ball"
    ANGULAR_CONTACT           = "angular_contact"
    SELF_ALIGNING_BALL        = "self_aligning_ball"
    THRUST_BALL               = "thrust_ball"
    CYLINDRICAL_ROLLER        = "cylindrical_roller"
    TAPERED_ROLLER            = "tapered_roller"
    SPHERICAL_ROLLER          = "spherical_roller"
    THRUST_CYLINDRICAL_ROLLER = "thrust_cylindrical_roller"
    THRUST_NEEDLE_ROLLER      = "thrust_needle_roller"

    @property
    def duty(self) -> str:
        """'radial' | 'thrust'. Levanta NotImplementedError para
        TAPERED_ROLLER/SPHERICAL_ROLLER de propósito -- rolamentos
        cónicos e autocompensadores de rolos tipicamente levam carga
        combinada radial+axial, não é um dos dois de forma limpa; a tua
        chamada quando os implementares, não uma que eu deva adivinhar
        aqui."""
        try:
            return _DUTY_BY_TYPE[self]
        except KeyError:
            raise NotImplementedError(
                f"BearingType.{self.name}: duty not yet decided -- este "
                f"tipo pode levar carga combinada, ver docstring."
            ) from None


_DUTY_BY_TYPE: dict[BearingType, str] = {
    BearingType.DEEP_GROOVE_BALL: "radial",
    BearingType.ANGULAR_CONTACT: "radial",
    BearingType.SELF_ALIGNING_BALL: "radial",
    BearingType.THRUST_BALL: "thrust",
    BearingType.CYLINDRICAL_ROLLER: "radial",
    BearingType.THRUST_CYLINDRICAL_ROLLER: "thrust",
    BearingType.THRUST_NEEDLE_ROLLER: "thrust",
    # TAPERED_ROLLER, SPHERICAL_ROLLER -- de propósito não mapeados
}


# =====================================================================
# ---- BearingCatalog -----------------------------------------------
# =====================================================================

@dataclass(frozen=True)
class BearingCatalog:
    """Generic, family-agnostic catalogue data. Valida-se sozinho no
    momento da construção (__post_init__ chama validate_or_raise()) --
    antes, validate()/validate_or_raise() existiam mas eram opt-in,
    dava para construir um catalog fisicamente impossível (D < d, b
    negativo) e ele circular pelo resto do código sem nada reparar até
    alguém se lembrar de chamar .validate_or_raise() manualmente.

    d, D, b       : bore / outer diameter / width [mm]
    C, C0         : dynamic (ISO 281) / static (ISO 76) load rating [N]
    designation   : manufacturer designation, e.g. "6208"
    label         : identifier for reporting/traceability
    position      : axial coordinate along the shaft [mm]
    arrangement   : "locating" | "floating" | "non-locating"
    """
    d: float
    D: float
    b: float = 0.0
    C: float = 0.0
    C0: float = 0.0
    designation: str = ""
    label: str = ""
    position: float = 0.0
    arrangement: Literal["locating", "floating", "non-locating"] = "locating"

    def __post_init__(self) -> None:
        self.validate_or_raise()

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.designation or "BearingCatalog"

        if self.d <= 0:
            errors.append(f"{tag}: d must be > 0, got {self.d}")
        if self.D <= 0:
            errors.append(f"{tag}: D must be > 0, got {self.D}")
        if self.D <= self.d:
            errors.append(f"{tag}: D must be > d, got D={self.D}, d={self.d}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")
        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.C < 0:
            errors.append(f"{tag}: C must be >= 0, got {self.C}")
        if self.C0 < 0:
            errors.append(f"{tag}: C0 must be >= 0, got {self.C0}")
        if self.arrangement not in ("locating", "floating", "non-locating"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', or "
                f"'non-locating', got '{self.arrangement}'"
            )
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


# =====================================================================
# ---- BearingFamily -----------------------------------------------
# =====================================================================

class BearingFamily(ABC):
    """Contrato que cada família/subtipo tem de implementar para ser
    pluggable em Bearing.assemble(). Instância simples, sem registo
    obrigatório a este nível (families/family.py mantém o seu próprio
    @register_family, mas esse mecanismo é à parte deste contrato).

    __init_subclass__ (abaixo) impõe em tempo de definição da classe --
    não só em teste -- duas consistências que antes só um teste
    apanhava: DUTY tem de bater com BEARING_TYPE.duty, e as chaves de
    REQUIRED_FOR têm de ser exatamente CAPABILITIES. Uma família mal
    declarada agora nem importa.
    """

    CAPABILITIES: ClassVar[frozenset[str]] = frozenset()
    REQUIRED_FOR: ClassVar[dict[str, frozenset[str]]] = {}
    BEARING_TYPE: ClassVar[BearingType | None] = None

    #: "radial" | "thrust". Mantido como atributo explícito (não
    #: derivado só de BEARING_TYPE.duty) de propósito -- é lido ao
    #: nível da classe em vários sítios (testes, relatórios) sem
    #: instanciar a família; __init_subclass__ garante que nunca
    #: diverge do que BEARING_TYPE.duty diria.
    DUTY: ClassVar[Literal["radial", "thrust"] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if set(cls.REQUIRED_FOR) != set(cls.CAPABILITIES):
            raise TypeError(
                f"{cls.__name__}: REQUIRED_FOR keys {set(cls.REQUIRED_FOR)} "
                f"must exactly match CAPABILITIES {set(cls.CAPABILITIES)}"
            )

        if cls.BEARING_TYPE is not None and cls.DUTY is not None:
            expected = cls.BEARING_TYPE.duty
            if cls.DUTY != expected:
                raise TypeError(
                    f"{cls.__name__}: DUTY={cls.DUTY!r} does not match "
                    f"BEARING_TYPE.duty={expected!r} for {cls.BEARING_TYPE!r}"
                )

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'deep_groove_ball'."""

    @abstractmethod
    def assemble_geometry(self, catalog: BearingCatalog, **geometry_kwargs: Any) -> dict[str, Any]:
        """
        Pure function: raw geometry inputs -> flat attribute dict mirrored
        onto the Bearing. No side effects, no cached state on self.

        catalog : read-only, for families that need to cross-check
                  catalogue fields (e.g. reject arrangement="locating")
        Returns : must include every field referenced in this family's
                  own REQUIRED_FOR.
        """