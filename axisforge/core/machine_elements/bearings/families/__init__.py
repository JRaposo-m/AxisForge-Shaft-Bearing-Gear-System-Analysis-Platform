# families/__init__.py  (rollup — fonte de verdade é o _FAMILY_REGISTRY em family.py)
"""
axisforge/core/machine_elements/bearings/families/__init__.py

Rollup de todas as BearingFamily concretas registadas via @register_family
em family.py. E' a partir daqui que bearing.py lista/seleciona familias, e
de onde um capabilities.py para bearings vai ler.

Lazy no primeiro acesso: nada e' importado ate' o primeiro
`families.<Nome>` (ou `available_families()`); a partir dai o registo
(_FAMILY_REGISTRY, populado pelo decorator @register_family a correr) e'
a fonte de verdade -- nao ha' lista nome->submodulo a manter a mao, logo
nao ha' como desalinhar (ver nota se separares em ball_bearing.py /
roller_bearing.py: atualiza _SUBMODULES).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

_SUBMODULES = (".family",)  # TODO: trocar para (".ball_bearing", ".roller_bearing") se/quando separares o ficheiro
_loaded = False


def _ensure_loaded() -> dict[str, type]:
    global _loaded
    if not _loaded:
        import importlib
        for mod in _SUBMODULES:
            importlib.import_module(mod, __name__)
        _loaded = True
    from .family import _FAMILY_REGISTRY
    return _FAMILY_REGISTRY


def __getattr__(name: str):
    registry = _ensure_loaded()
    if name in registry:
        value = registry[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_ensure_loaded()))


def available_families() -> dict[str, type]:
    """Nome -> classe BearingFamily, para quem precisar de listar/selecionar
    sem já saber os nomes de antemão (ex.: capabilities.py, uma UI de
    seleção de catálogo). Dispara o import na primeira chamada."""
    return dict(_ensure_loaded())


if TYPE_CHECKING:  # pragma: no cover -- só para autocomplete/type checkers, não corre
    from .family import (
        DeepGrooveBallFamily, AngularContactFamily, SelfAligningBallFamily,
        ThrustBallSingleRowFamily, ThrustBallMultiRowFamily,
        CylindricalRollerFamily, ThrustCylindricalRollerFamily,
        RollerThrustMultiRowFamily,
    )