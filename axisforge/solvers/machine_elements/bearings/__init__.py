# axisforge/solvers/machine_elements/bearings/__init__.py
"""
axisforge/solvers/machine_elements/bearings/__init__.py

Ponto único de import para solvers/machine_elements/bearings/ -- de
momento só tem o subpacote load_distribution/ (que por sua vez só tem
iso_16281/), por isso reexporta o seu __all__ tal e qual, estilo slippy
(eager, sem lazy loading) -- mesmo padrão em cascata de
core/machine_elements/bearings/__init__.py -> families/__init__.py.
Um segundo tipo de solver (ex.: fatigue/, stiffness/) junta-se aqui do
mesmo modo, ao lado deste import.
"""
from __future__ import annotations

from .load_distribution import *  # noqa: F401,F403 -- __all__ já vem de load_distribution/__init__.py
from .load_distribution import __all__ as _load_distribution_names

__all__ = list(_load_distribution_names)
