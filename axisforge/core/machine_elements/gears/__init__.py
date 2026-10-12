# axisforge/core/machine_elements/gears/__init__.py
"""
Single import point of the gear subsystem (eager re-export, as in bearings).

Only the parallel-axis family exists (external and internal spur/helical
gears, their pairs and the planetary train), so only its names are
re-exported. Another family (bevel, worm, ...) would get its own sub-package
and one more import line here.
"""
from __future__ import annotations

from .parallel_axis import *  # noqa: F401,F403 -- parallel_axis.__all__ is already the curated list
from .parallel_axis import __all__ as _parallel_axis_names

__all__ = list(_parallel_axis_names)