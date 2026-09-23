"""
axisforge/solvers/__init__.py

Ponto único de import para solvers/ -- nível de topo. Junta os dois
subpacotes diretos com conteúdo real hoje: machine_elements/ (que já
cascateia shaft/ + bearings/) e mesh/ (convergence_solver.py), estilo
slippy (eager, sem lazy loading) -- mesma cascata usada nos níveis
abaixo. Entre os dois, isto cobre de facto os "3" módulos reais
(shaft, bearings, mesh) que existem no projeto até agora.

Um novo subpacote de topo (ex.: fatigue/, lubrication/, failure_risk/,
conforme forem sendo escritos) junta-se aqui do mesmo modo, ao lado
destes dois imports.
"""
from __future__ import annotations

from .machine_elements import *  # noqa: F401,F403 -- __all__ já vem de machine_elements/__init__.py
from .machine_elements import __all__ as _machine_elements_names

from .mesh import *  # noqa: F401,F403 -- __all__ já vem de mesh/__init__.py
from .mesh import __all__ as _mesh_names

__all__ = list(_machine_elements_names) + list(_mesh_names)