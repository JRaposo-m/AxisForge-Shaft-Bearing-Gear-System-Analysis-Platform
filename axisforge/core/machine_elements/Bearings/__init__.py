# axisforge/core/machine_elements/bearings/__init__.py
"""
axisforge/core/machine_elements/bearings/__init__.py

Ponto único de import para todo o subsistema -- estilo slippy, tudo
eager. As famílias concretas vêm do rollup em families/__init__.py
(gerado a partir do registo @register_family, não à mão).
"""
from __future__ import annotations

from .base import BearingType, BearingCatalog, BearingFamily
from .bearing import Bearing
from .types import *  # noqa: F401,F403 -- __all__ de families/ já vem do registo
from .types import __all__ as _family_names

__all__ = ["Bearing", "BearingCatalog", "BearingFamily", "BearingType"] + list(_family_names)