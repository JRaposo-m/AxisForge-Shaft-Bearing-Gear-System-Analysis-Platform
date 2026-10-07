# axisforge/core/machine_elements/bearings/__init__.py
"""
axisforge/core/machine_elements/bearings/__init__.py

Ponto único de import para todo o subsistema -- estilo slippy, tudo
eager. As famílias concretas vêm do rollup em families/__init__.py
(gerado a partir do registo @register_family, não à mão).
"""
from __future__ import annotations


from .base import BearingCatalog, BearingFamily
from .bearing import Bearing
from .families import *  # noqa: F401,F403 -- __all__ de families/ já vem do registo
from .families import __all__ as _family_names
from .contact_models import *  # noqa: F401,F403 -- __all__ de contact_models/
from .contact_models import __all__ as _contact_names

__all__ = (["Bearing", "BearingCatalog", "BearingFamily"]
           + list(_family_names) + list(_contact_names))