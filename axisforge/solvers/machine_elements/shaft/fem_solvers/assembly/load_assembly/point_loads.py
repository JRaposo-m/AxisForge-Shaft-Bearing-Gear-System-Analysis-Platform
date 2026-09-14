"""
axisforge/solvers/machine_elements/shaft/fem_solvers/assembly/load_assembly/point_loads.py

MOCKUP / SKELETON -- extracted from
RigidSupportFEMSolver._assemble_load_vector() exactly as it was.
Beam-theory-agnostic: a point load is injected directly into the
nearest node's DOF -- it does not use shape functions, does not touch
the stiffness matrix, and does not depend on Timoshenko vs.
Euler-Bernoulli. That is why it is kept separate from the distributed
side (load_vector_distributed_*.py), which DOES depend on theory (see
those modules).

Called by rigid_support.py::RigidSupportFEMSolver.solve() -- see that
module for the orchestrating entry point.
"""

from __future__ import annotations

import numpy as np

from axisforge.mesh.shaft.element_type.elem import Elem


def assemble_point_load_vector(
    x_nodes: list[float],
    radial: list[tuple[float, float]],
    axial: list[tuple[float, float]],
    moments: list[tuple[float, float]],
) -> np.ndarray:
    """
    Extracted from RigidSupportFEMSolver._assemble_load_vector() --
    logic unchanged, only stopped being a method (it never read any
    attribute off the solver, only x_nodes and the load lists already
    decomposed by load_cases.build_load_cases()).

    radial/axial/moments : each a list[(x_position, magnitude)] -- see
        load_cases.py for how they are built from the ShaftSystem.
    """
    f = np.zeros(3 * len(x_nodes))
    for x, mag in axial:
        i = Elem.find_node_index(x_nodes, x)
        f[3 * i] += mag
    for x, mag in radial:
        i = Elem.find_node_index(x_nodes, x)
        f[3 * i + 1] += mag
    for x, mag in moments:
        i = Elem.find_node_index(x_nodes, x)
        f[3 * i + 2] += mag
    return f