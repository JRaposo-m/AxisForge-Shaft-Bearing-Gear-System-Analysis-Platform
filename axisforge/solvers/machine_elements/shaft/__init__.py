"""
axisforge/solvers/machine_elements/shaft/__init__.py

Ponto único de import para solvers/machine_elements/shaft/ -- de momento
só tem o subpacote fem_solvers/ (nenhum outro solver de shaft existe
ainda), por isso reexporta o seu __all__ tal e qual, estilo slippy
(eager, sem lazy loading) -- mesmo padrão em cascata usado em
solvers/machine_elements/bearings/__init__.py -> load_distribution/__init__.py.
Um segundo tipo de solver ao nível de shaft (ex.: um solver alternativo,
ou um módulo de pré-processamento próprio deste nível) junta-se aqui do
mesmo modo, ao lado deste import.
"""
from __future__ import annotations

from .fem_solvers import *  # noqa: F401,F403 -- __all__ já vem de fem_solvers/__init__.py
from .fem_solvers import __all__ as _fem_solvers_names

__all__ = list(_fem_solvers_names)