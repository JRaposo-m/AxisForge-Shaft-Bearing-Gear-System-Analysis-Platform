"""
Domain objects of AxisForge: materials, loads, mechanical systems and machine elements.

Flattens the ``__all__`` of the four sub-packages into one namespace.
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