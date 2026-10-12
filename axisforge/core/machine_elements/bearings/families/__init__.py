# axisforge/core/machine_elements/bearings/families/__init__.py
"""Eager roll-up of all concrete bearing families.

Importing ``family`` defines every family class, and the ``@register_family``
decorators fill the registry as each class is defined. The registry is exposed
as module attributes and ``__all__`` is built from it, so it always matches the
actual class names. No lazy loading and no ``__getattr__``.
"""
from __future__ import annotations

from .family import _FAMILY_REGISTRY

globals().update(_FAMILY_REGISTRY)

__all__ = sorted(_FAMILY_REGISTRY)


def available_families() -> dict[str, type]:
    """Registered bearing families by class name.

    Lets a caller list or select a family without knowing the names in
    advance.

    Returns
    -------
    dict[str, type]
        Class name -> ``BearingFamily`` subclass; a copy of the registry.
    """
    return dict(_FAMILY_REGISTRY)