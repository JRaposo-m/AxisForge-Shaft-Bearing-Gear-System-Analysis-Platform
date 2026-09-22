# axisforge/solvers/machine_elements/bearings/load_distribution/__init__.py
"""
axisforge/solvers/machine_elements/bearings/load_distribution/__init__.py

Ponto único de import para load_distribution/ -- de momento só tem o
subpacote ISO/TS 16281 (iso_16281/), por isso reexporta o seu __all__
tal e qual, estilo slippy (eager, sem lazy loading) -- mesmo padrão de
core/machine_elements/bearings/__init__.py com families/. Um segundo
modelo de contacto/solver (se/quando existir) junta-se aqui do mesmo
modo, ao lado deste import.
"""
from __future__ import annotations

from .iso_16281 import *  # noqa: F401,F403 -- __all__ já vem de iso_16281/__init__.py
from .iso_16281 import __all__ as _iso16281_names

__all__ = list(_iso16281_names)
