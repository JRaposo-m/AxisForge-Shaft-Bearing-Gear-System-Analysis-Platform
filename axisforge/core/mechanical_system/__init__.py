"""
Mechanical systems: assemblies of shafts, bearings and gears.

Re-exports the parallel-axis (spur and helical) system classes. Planetary
and perpendicular-axis systems do not exist (the
``parallel_axis/planetary/`` folder is empty).
"""
from __future__ import annotations

from .parallel_axis import *  # noqa: F401,F403
from .parallel_axis import __all__ as _parallel_axis_names

__all__ = list(_parallel_axis_names)