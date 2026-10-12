# axisforge/core/machine_elements/bearings/__init__.py
"""Public interface of the bearings subsystem.

Single import point for ``Bearing``, ``BearingCatalog``, ``BearingFamily``,
the concrete bearing families and the contact models. All imports are eager.
The family names come from the roll-up in ``families/__init__.py``, which is
built from the ``@register_family`` registry.
"""
from __future__ import annotations


from .base import BearingCatalog, BearingFamily
from .bearing import Bearing
from .families import *  # noqa: F401,F403 -- __all__ of families/ comes from the registry
from .families import __all__ as _family_names
from .contact_models import *  # noqa: F401,F403 -- __all__ of contact_models/
from .contact_models import __all__ as _contact_names

__all__ = (["Bearing", "BearingCatalog", "BearingFamily"]
           + list(_family_names) + list(_contact_names))