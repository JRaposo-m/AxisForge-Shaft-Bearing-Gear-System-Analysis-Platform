# axisforge/core/machine_elements/gears/__init__.py
"""
axisforge/core/machine_elements/gears/__init__.py

Ponto único de import para o subsistema gears -- estilo bearings/,
eager, sem lazy loading. Por agora só existe parallel_axis/ (spur/
helical, internal, planetary), por isso reexporta só o que lá está;
quando houver outras famílias de engrenagens (bevel, worm, ...) cada
uma ganha a sua própria subpasta com o mesmo padrão, e uma linha aqui a
mais -- nada no resto do código precisa de mudar.
"""
from __future__ import annotations

from .parallel_axis import *  # noqa: F401,F403 -- __all__ de parallel_axis/ já é a lista curada
from .parallel_axis import __all__ as _parallel_axis_names

__all__ = list(_parallel_axis_names)