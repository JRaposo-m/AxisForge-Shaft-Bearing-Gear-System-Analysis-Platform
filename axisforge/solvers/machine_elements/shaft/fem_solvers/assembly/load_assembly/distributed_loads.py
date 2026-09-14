"""
axisforge/solvers/machine_elements/shaft/fem_solvers/assembly/load_assembly/distributed_loads.py

Theory-agnostic distributed-load vector assembly. Works for both
Euler-Bernoulli and Timoshenko elements without branching on theory
anywhere in the integration loop -- Elem already dispatches its own
shape_functions()/natural_coordenates(), and QuadratureOrderEstimator
already knows the shape-function degree per theory (see
fem_solvers/numerics/gauss_quadrature.py). This replaces what would
otherwise have been two separate files
(load_vector_distributed_euler.py / load_vector_distributed_timoshenko.py)
under element_theories/ -- consolidated here because neither the
integration loop nor the scatter step actually needs to branch on
theory once Elem does its own dispatch.

What IS theory-specific, and lives here as an explicit, named lookup
rather than being inferred, is which shape-function value weights each
of the 4 conceptual local DOF slots [v_a, th_a, v_b, th_b]:

  - euler_bernoulli: shape_functions() returns 4 entries, one per DOF
    (v_a, th_a, v_b, th_b) -- the mapping is the identity: slot i uses
    N[i] directly. A distributed load therefore produces a consistent
    nodal force AND a consistent nodal moment at each end (the classic
    "fixed-end moment" of a UDL on a Hermite beam element).
  - timoshenko: shape_functions() returns only 2 entries, N[0]/N[1] =
    1/2(1 -+ zeta) -- these are plain NODAL weights (one per node),
    not per-DOF-type weights. N[0] therefore weights BOTH local DOFs
    of node a (v_a and th_a alike); N[1] weights both DOFs of node b.
    That is why there are 2 functions for 4 slots: each function
    serves both DOFs at its own node, not one specific DOF type.

Local DOF slots follow the [v_a, th_a, v_b, th_b] layout used
throughout element_theories/, mapped at scatter time onto the global
3-DOF/node layout [axial, radial, moment] already used by
load_vector_point.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.gauss_quadrature import (
    QuadratureOrderEstimator, gauss_points_weights,
)

# Maps local DOF slot [v_a, th_a, v_b, th_b] -> index into
# elem.shape_functions()'s return value. euler_bernoulli is the
# identity (4 slots, 4 functions, one each). timoshenko reuses N[0]
# for both of node a's slots and N[1] for both of node b's slots --
# see module docstring.
_SLOT_SHAPE_FUNCTION_INDEX: dict[str, list[int]] = {
    "euler_bernoulli": [0, 1, 2, 3],
    "timoshenko": [0, 0, 1, 1],
}


def element_distributed_force_vector(
    elem: Elem,
    x_lo: float, x_hi: float,
    q: Callable[[float], float],
    estimator: QuadratureOrderEstimator,
    theta_fn: Callable[[float], float] | None = None,
) -> np.ndarray | None:
    """
    Consistent nodal force vector, in the local 4-slot
    [v_a, th_a, v_b, th_b] layout, for the portion of a distributed
    load q(x) (already projected onto one plane -- see load_cases.py's
    "q") that overlaps elem's domain [elem.x_a, elem.x_b].

    Returns None if there is no overlap -- caller should skip scatter
    for this element entirely in that case.
    """
    a = max(x_lo, elem.x_a)
    b = min(x_hi, elem.x_b)

    le = elem.length
    if b <= a:
        return None   # elem's domain does not intersect [x_lo, x_hi]

    n = estimator.gauss_order(q, a, b, theta_fn)
    pts, wts = gauss_points_weights(n)

    f_local = np.zeros(4)
    jacobian = le / 2.0   # maps Gauss points on [-1, 1] to [a, b], J = dx/dzeta = le/2.0
    slot_map = _SLOT_SHAPE_FUNCTION_INDEX[elem.beam_theory]

    for pt, wt in zip(pts, wts):
        x = (a + b) / 2.0 + jacobian * pt
        zeta = elem.natural_coordenates(x)
        N = elem.shape_functions(zeta)
        q_val = q(x)
        for slot in range(4):
            f_local[slot] += wt * jacobian * N[slot_map[slot]] * q_val

    return f_local


def assemble_distributed_load_vector(
    x_nodes: list[float],
    elements: list[Elem],
    distributed_cases: list[dict],
    estimator: QuadratureOrderEstimator,
) -> np.ndarray:
    """
    Scatter every distributed-load case's per-element contribution
    into one plane's global force vector.

    distributed_cases: the "distributed_xz" or "distributed_xy" list
    from one load_cases.py case dict -- each entry is
    {"x_lo", "x_hi", "q", "theta_fn"}.

    DOF layout: 3 per node [axial, radial, moment] -- same convention
    as load_vector_point.py. Local slot 0/2 (v_a/v_b) -> radial;
    local slot 1/3 (th_a/th_b) -> moment. For timoshenko, slots 0/1
    (and 2/3) come out equal per element -- both DOFs of a node share
    the same nodal weight, per _SLOT_SHAPE_FUNCTION_INDEX. Axial is
    never touched here.
    """
    f = np.zeros(3 * len(x_nodes))

    for case in distributed_cases:
        x_lo, x_hi = case["x_lo"], case["x_hi"]
        q = case["q"]
        theta_fn = case.get("theta_fn")

        for elem in elements:
            f_local = element_distributed_force_vector(
                elem, x_lo, x_hi, q, estimator, theta_fn,
            )
            if f_local is None:
                continue

            i, j = elem.idx_node_1, elem.idx_node_2
            f[3 * i + 1] += f_local[0]   # v_a  -> node i, radial
            f[3 * i + 2] += f_local[1]   # th_a -> node i, moment
            f[3 * j + 1] += f_local[2]   # v_b  -> node j, radial
            f[3 * j + 2] += f_local[3]   # th_b -> node j, moment

    return f