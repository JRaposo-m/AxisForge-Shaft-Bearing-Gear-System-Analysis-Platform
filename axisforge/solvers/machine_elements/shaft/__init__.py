"""Import point of the shaft solvers: re-exports the public names of ``fem_solvers`` unchanged.
"""
from __future__ import annotations

from .fem_solvers import *  # noqa: F401,F403 -- __all__ comes from fem_solvers/__init__.py
from .fem_solvers import __all__ as _fem_solvers_names

__all__ = list(_fem_solvers_names)