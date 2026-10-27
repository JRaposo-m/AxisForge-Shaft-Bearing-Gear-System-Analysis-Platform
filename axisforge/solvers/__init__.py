"""Top-level import point of the solvers: shaft and bearing solvers (machine_elements) and the mesh
convergence solver (mesh), and generic numerical methods (numerics). Eager re-exports, no lazy loading.
"""
from __future__ import annotations

from .machine_elements import *  # noqa: F401,F403 -- __all__ comes from machine_elements/__init__.py
from .machine_elements import __all__ as _machine_elements_names

from .mesh import *  # noqa: F401,F403 -- __all__ comes from mesh/__init__.py
from .mesh import __all__ as _mesh_names

from .numerics import *  # noqa: F401,F403 -- __all__ comes from numerics/__init__.py
from .numerics import __all__ as _numerics_names

__all__ = list(_machine_elements_names) + list(_mesh_names) + list(_numerics_names)