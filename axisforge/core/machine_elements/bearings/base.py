# axisforge/core/machine_elements/bearings/base.py
"""
core/machine_elements/bearings/base.py

Bearing -- nivel 1 da hierarquia (ver ADR-001, vault/40_ADR/): o minimo
necessario para simulacao FEM e posicionamento no veio/engrenagens
quando o rolamento em si nao e o foco da analise. Concreta,
instanciavel diretamente -- "modo simples".

BearingType fica aqui tambem -- pequeno, estavel, mutuamente acoplado
com o resto do pacote (BearingFamily.BEARING_TYPE e um BearingType).

BearingFamily (nivel 2 -- fusao do antigo BearingCatalog +
BearingFamily(ABC)) vive em family.py, nao aqui.
`isinstance(bearing, BearingFamily)` e a forma sancionada de um chamador
que recebe um Bearing (nivel 1) verificar/aceder aos dados completos.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

BearingArrangement = Literal["locating", "floating", "non-locating"]
_VALID_ARRANGEMENTS = ("locating", "floating", "non-locating")


# =====================================================================
# ---- BearingType ------------------------------------------------------
# =====================================================================

class BearingType(str, Enum):
    """Informational label + fonte unica de verdade sobre a duty
    (radial/thrust) de cada tipo -- ver `.duty` abaixo. Herda de `str`
    (nao `auto()`): o projeto persiste em SQLite/relatorios, uma string
    e estavel para sempre, um inteiro de `auto()` desloca-se se um
    membro for inserido no meio da lista."""
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
        """'radial' | 'thrust'. Levanta NotImplementedError de proposito
        para TAPERED_ROLLER/SPHERICAL_ROLLER -- tipicamente levam carga
        combinada radial+axial, nao e um dos dois de forma limpa; a tua
        chamada quando os implementares, nao uma que se deva adivinhar
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
    # TAPERED_ROLLER, SPHERICAL_ROLLER -- de proposito nao mapeados
}


# =====================================================================
# ---- Bearing (nivel 1) ------------------------------------------------
# =====================================================================

class Bearing:
    """Nivel 1 -- posicao e arranjo, mais nada. Suficiente para FEM /
    obter dados para veio ou engrenagens sem detalhar o rolamento.

    Validacao acontece em __init__ (fail-fast), sem passo .validate()
    separado nem .assemble() -- ao contrario do design antigo."""

    def __init__(self, *, position: float, arrangement: BearingArrangement,
                 label: str = "", designation: str = ""):
        if position < 0:
            raise ValueError(f"Bearing({label or designation!r}): position must be >= 0, got {position}")
        if arrangement not in _VALID_ARRANGEMENTS:
            raise ValueError(
                f"Bearing({label or designation!r}): arrangement must be one of "
                f"{_VALID_ARRANGEMENTS}, got {arrangement!r}")

        self.position = position
        self.arrangement = arrangement
        self.label = label
        self.designation = designation

    def is_locating(self) -> bool:
        return self.arrangement == "locating"

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"label={self.label!r}, position={self.position:.2f} mm, "
                f"arrangement={self.arrangement!r})")