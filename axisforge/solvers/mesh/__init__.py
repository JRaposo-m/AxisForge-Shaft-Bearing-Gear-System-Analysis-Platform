"""Mesh-convergence solver of the shaft FEM: Richardson extrapolation and Grid Convergence Index
(GCI) over successive mesh refinements.
"""
from __future__ import annotations

from .convergence_solver import (
    register_eval_form,
    resolve_eval_form,
    RichardsonGCI,
    MeshConvergenceStudy,
    build_mesh_refinement_result,
)

__all__ = [
    "register_eval_form",
    "resolve_eval_form",
    "RichardsonGCI",
    "MeshConvergenceStudy",
    "build_mesh_refinement_result",
]
