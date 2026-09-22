# axisforge/core/machine_elements/__init__.py
"""
axisforge/core/machine_elements/__init__.py

Ponto único de import para todos os machine elements: bearings/,
gears/, shaft/ -- eager, sem lazy loading, mesmo estilo do resto da
árvore.
"""
from __future__ import annotations

from .bearings import *  # noqa: F401,F403
from .bearings import __all__ as _bearings_names
from .gears import *  # noqa: F401,F403
from .gears import __all__ as _gears_names
from .shaft import *  # noqa: F401,F403
from .shaft import __all__ as _shaft_names

__all__ = list(_bearings_names) + list(_gears_names) + list(_shaft_names)