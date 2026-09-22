# axisforge/core/mechanical_system/__init__.py
"""
axisforge/core/mechanical_system/__init__.py

Ponto único de import para mechanical_system -- eager, sem lazy
loading. Por agora só existe parallel_axis/; cada família nova (ex.
perpendicular_axis/) ganha a sua própria subpasta com o mesmo padrão, e
uma linha aqui a mais.
"""
from __future__ import annotations

from .parallel_axis import *  # noqa: F401,F403
from .parallel_axis import __all__ as _parallel_axis_names

__all__ = list(_parallel_axis_names)