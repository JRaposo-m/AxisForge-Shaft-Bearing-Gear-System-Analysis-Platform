# axisforge/core/machine_elements/Bearings/__init__.py
"""
axisforge/core/machine_elements/Bearings/__init__.py

Public surface para os quatro primitivos de orquestracao: o container
imutavel (Bearing), os dados de catalogo (BearingCatalog), o contrato
que uma familia pluggable implementa (BearingFamily), e o enum-label
para dispatch do lado dos solvers (BearingType).

As familias concretas (DeepGrooveBallFamily, AngularContactFamily, ...)
NAO sao reexportadas aqui de proposito -- import de
axisforge.core.machine_elements.Bearings.families diretamente. Segundo
o principio do proprio core/README.md ("adding a new bearing family
means writing a new BearingFamily subclass anywhere -- nothing in
core/ needs to change"), se as familias fossem reexportadas aqui, cada
familia nova obrigaria a editar este ficheiro tambem -- o que anularia
esse principio.

NOTA (migracao em curso): make_bearing, DeepGrooveBallBearing,
AngularContactBallBearing, CylindricalRollerBearing existiam aqui antes
-- residuos da arquitetura antiga (subclasses concretas de Bearing).
DeepGrooveBallBearing ainda existe fisicamente em Bearings/subtypes/ e
e usada por fixtures/bearings/dgbb_generic.py; a via atual e
Bearing.assemble(family=DeepGrooveBallFamily(), ...) via families/.
Migrar dgbb_generic.py e retirar esta nota quando isso acontecer.

Lazy-loaded -- ver families/__init__.py para o mecanismo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "Bearing",
    "BearingCatalog",
    "BearingFamily",
    "BearingType",
]

_LAZY = {
    "Bearing": ".bearing",
    "BearingCatalog": ".catalog",
    "BearingFamily": ".family",
    "BearingType": ".bearing_types",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


if TYPE_CHECKING:  # pragma: no cover
    from .bearing import Bearing
    from .catalog import BearingCatalog
    from .family import BearingFamily
    from .bearing_types import BearingType