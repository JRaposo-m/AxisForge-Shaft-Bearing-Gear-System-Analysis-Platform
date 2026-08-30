# axisforge/core/machine_elements/shaft/__init__.py
"""
axisforge/core/machine_elements/shaft/__init__.py

Public surface para a geometria do shaft, convencao (r, theta, z):
KeywayType (enum), Keyway (stress raiser localizado), Shoulder
(transicao de fillet entre seccoes), ShaftSection (segmento cilindrico
uniforme), Shaft (sequencia ordenada de ShaftSection). Tudo definido em
shaft.py.

Nota: Keyway.from_standard() depende de
axisforge.database.shaft.keyway.Parallel.parallel_keyway e
.Woodruff_key.iso3912 -- essas nao sao reexportadas aqui (nao sao
geometria do shaft, sao dados de catalogo), mas fazem parte da mesma
cadeia de import e vao precisar do mesmo tratamento (__init__.py
proprio) quando chegares a essa pasta.

Lazy-loaded -- ver core/machine_elements/Gears/Parallel_Axis_gears/__init__.py
para o mecanismo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "Shaft",
    "ShaftSection",
    "Shoulder",
    "Keyway",
    "KeywayType",
]

_LAZY = {
    "Shaft": ".shaft",
    "ShaftSection": ".shaft",
    "Shoulder": ".shaft",
    "Keyway": ".shaft",
    "KeywayType": ".shaft",
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
    from .shaft import Shaft, ShaftSection, Shoulder, Keyway, KeywayType