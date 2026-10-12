# axisforge/core/machine_elements/shaft/__init__.py
"""
Public names of the shaft geometry model: Shaft, ShaftSection, Shoulder, Keyway and KeywayType, all defined in shaft.py.
"""
from __future__ import annotations

from .shaft import Shaft, ShaftSection, Shoulder, Keyway, KeywayType

__all__ = [
    "Shaft",
    "ShaftSection",
    "Shoulder",
    "Keyway",
    "KeywayType",
]