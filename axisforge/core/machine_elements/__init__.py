# axisforge/core/machine_elements/__init__.py
"""Public interface of the machine elements: bearings, gears and shaft.

Single import point; the public names of the three subpackages are re-exported
eagerly, without lazy loading.
"""
from __future__ import annotations

from .bearings import *  # noqa: F401,F403
from .bearings import __all__ as _bearings_names
from .gears import *  # noqa: F401,F403
from .gears import __all__ as _gears_names
from .shaft import *  # noqa: F401,F403
from .shaft import __all__ as _shaft_names

__all__ = list(_bearings_names) + list(_gears_names) + list(_shaft_names)