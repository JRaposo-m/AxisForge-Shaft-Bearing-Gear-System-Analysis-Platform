"""
axisforge/mesh/shaft/mesh_generation/__init__.py

Eager rollup, sem lazy loading, sem __getattr__ -- mesma lógica do
element_type/__init__.py: Mesh1D e Grader são Python/NumPy puro,
importá-los directamente não tem custo que justifique lazy loading, e
mantém `from axisforge.mesh.shaft.mesh_generation import Mesh1D` a
funcionar sem quem importa precisar de saber a disposição interna dos
módulos.
"""
from __future__ import annotations

from .mesh_1D import Mesh1D
from .mesh_grade import Grader

__all__ = ["Mesh1D", "Grader"]