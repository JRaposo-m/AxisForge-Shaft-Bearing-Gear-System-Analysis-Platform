# axisforge/core/machine_elements/shaft/__init__.py
"""
axisforge/core/machine_elements/shaft/__init__.py

Public surface para a geometria do shaft, convencao (r, theta, z):
KeywayType (enum), Keyway (stress raiser localizado), Shoulder
(transicao de fillet entre seccoes), ShaftSection (segmento cilindrico
uniforme), Shaft (sequencia ordenada de ShaftSection). Tudo definido em
shaft.py -- estilo bearings/, eager, sem lazy loading.

Nota: Keyway.from_standard() depende de
axisforge.database.shaft.keyway.Parallel.parallel_keyway e
.Woodruff_key.iso3912 -- essas nao sao reexportadas aqui (nao sao
geometria do shaft, sao dados de catalogo), mas fazem parte da mesma
cadeia de import e vao precisar do mesmo tratamento (__init__.py
proprio, mesmo padrao) quando chegares a essa pasta.
"""
from __future__ import annotations

from .shaft import Shaft, ShaftSection, Shoulder, Keyway, KeywayType

__all__ = [
    "Shaft",
    "ShaftSection",
    "Shoulder",
    "Keyway",
    "KeywayType",
]