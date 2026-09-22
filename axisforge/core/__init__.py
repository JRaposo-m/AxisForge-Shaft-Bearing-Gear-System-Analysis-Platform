# axisforge/core/__init__.py
"""
axisforge/core/__init__.py

Ponto único de import para core -- eager, sem lazy loading, mesmo
padrão do resto da árvore. Agrega materials.py, loads.py e as
subpastas mechanical_system/ e machine_elements/; cada família nova
ganha a sua própria linha aqui.
"""
from __future__ import annotations

from .materials import *  # noqa: F401,F403
from .materials import __all__ as _materials_names
from .loads import *  # noqa: F401,F403
from .loads import __all__ as _loads_names
from .mechanical_system import *  # noqa: F401,F403
from .mechanical_system import __all__ as _mechanical_system_names
from .machine_elements import *  # noqa: F401,F403
from .machine_elements import __all__ as _machine_elements_names

__all__ = (
    list(_materials_names)
    + list(_loads_names)
    + list(_mechanical_system_names)
    + list(_machine_elements_names)
)