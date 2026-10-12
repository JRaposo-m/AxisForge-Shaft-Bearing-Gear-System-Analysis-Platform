"""
Rollup of the 1D mesh generation package: Mesh1D (node grid of a shaft system) and Grader (refinement grades of a subdomain).
"""
from __future__ import annotations

from .mesh_1D import Mesh1D
from .mesh_grade import Grader

__all__ = ["Mesh1D", "Grader"]