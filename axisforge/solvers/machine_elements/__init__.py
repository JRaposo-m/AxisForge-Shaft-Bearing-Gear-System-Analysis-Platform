"""Import point of the machine-element solvers: re-exports the public names of ``shaft`` (FEM) and
``bearings`` (ISO 281, ISO/TS 16281). Eager re-exports, no lazy loading.
"""
from __future__ import annotations

from .shaft import *  # noqa: F401,F403 -- __all__ comes from shaft/__init__.py
from .shaft import __all__ as _shaft_names

from .bearings import *  # noqa: F401,F403 -- __all__ comes from bearings/__init__.py
from .bearings import __all__ as _bearings_names

__all__ = list(_shaft_names) + list(_bearings_names)