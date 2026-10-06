"""
axisforge/solvers/machine_elements/shaft/fem_solvers/assembly/load_assembly/distributed_loads.py

Consistent nodal load vector for distributed TRANSVERSE loads.

Assembles the element-level consistent load vector of a transverse
distributed load q(x) [N/mm], already projected onto one bending plane,
and scatters it into the global force vector. A single integration loop
serves both beam theories: ``Elem`` dispatches its own
``shape_functions()`` / ``natural_coordenates()``, and
``QuadratureOrderEstimator`` selects a Gauss order sufficient for the
shape-function degree of each theory (``assembly/numerics/gauss_quadrature.py``).

Theory
------
The consistent load vector follows from equating the virtual work of the
distributed load with that of the nodal forces,

.. math::

    \\delta W = \\int_{x_a}^{x_b} q(x)\\,\\delta v(x)\\,dx
    \\quad\\Rightarrow\\quad
    f_i = \\int_{x_a}^{x_b} q(x)\\,N_i^{v}(x)\\,dx ,

where :math:`N_i^{v}` is the shape function that relates nodal DOF *i* to
the transverse displacement v(x). Only DOFs that appear in the
interpolation of v therefore receive a contribution. The local DOF
layout is ``[v_a, th_a, v_b, th_b]``.

* **Euler-Bernoulli** (cubic Hermite element). The rotation is not
  independent (theta = dv/dx), and v is interpolated with the four
  Hermite functions ``N[0..3]``, one per DOF. A distributed load thus
  produces nodal forces *and* nodal moments; for uniform q over a whole
  element these are q*le/2 and +/- q*le**2/12 (the classical fixed-end
  forces and moments).
* **Timoshenko** (two-node linear element, independent interpolation).
  v and theta are interpolated separately with the same linear functions,
  v(x) = N[0] v_a + N[1] v_b and theta(x) = N[0] th_a + N[1] th_b. The
  rotations do not appear in v(x), so a transverse load does no work on
  them: the theta slots are zero. For uniform q over a whole element the
  nodal forces are q*le/2 and the nodal moments are 0.

Assumptions
-----------
* Transverse loads only, in one bending plane per call (the caller
  projects q onto the XY or XZ plane beforehand).
* Small displacements and linear elasticity (the load vector is
  independent of the solution).
* The axial DOF is never loaded here.

Limitations
-----------
Distributed COUPLES m(x) [N.mm/mm] are not supported. A distributed
couple does work on the rotation, delta W = int m(x) delta theta(x) dx,
and its consistent vector differs from the one above:

* Timoshenko: m loads the theta slots through N[0], N[1]; the v slots are
  zero (the mirror of the transverse case);
* Euler-Bernoulli: since theta = dv/dx, m is integrated against the
  DERIVATIVES dN_i/dx of the Hermite functions, which loads both v and
  theta DOFs.

A single slot-to-shape-function map, as used here, cannot represent the
Euler-Bernoulli case. Distributed couples are rare in shaft analysis and
no AxisForge load type produces them (``ExternalMoment`` is a point
moment and is assembled by the point-load path). Supporting them would
require a dedicated load type, its own entry in the load cases and a
separate assembly routine; ``_SLOT_SHAPE_FUNCTION_INDEX`` must not be
reused for that purpose.

References
----------
.. [1] Reddy, J. N. (2019). *An Introduction to the Finite Element
       Method*, 4th ed. McGraw-Hill. Euler-Bernoulli and Timoshenko beam
       elements, consistent load vectors (chapter and equation numbers to
       be confirmed).
.. [2] Cook, R. D., Malkus, D. S., Plesha, M. E. & Witt, R. J. (2002).
       *Concepts and Applications of Finite Element Analysis*, 4th ed.
       Wiley. Consistent nodal loads (section to be confirmed).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.gauss_quadrature import (
    QuadratureOrderEstimator, gauss_points_weights,
)

# Local DOF slot [v_a, th_a, v_b, th_b] -> index into elem.shape_functions(),
# or None when the slot receives no contribution. Valid for TRANSVERSE loads
# only (see module docstring, "Limitations"):
#   euler_bernoulli: identity -- the Hermite functions couple v and theta,
#                    hence the consistent nodal moments.
#   timoshenko:      v and theta are interpolated independently; a transverse
#                    load does work on v only, so the theta slots are None.
_SLOT_SHAPE_FUNCTION_INDEX: dict[str, list[int | None]] = {
    "euler_bernoulli": [0, 1, 2, 3],
    "timoshenko": [0, None, 1, None],
}


def element_distributed_force_vector(
    elem: Elem,
    x_lo: float, x_hi: float,
    q: Callable[[float], float],
    estimator: QuadratureOrderEstimator,
    theta_fn: Callable[[float], float] | None = None,
) -> np.ndarray | None:
    """Consistent nodal load vector of a transverse load on one element.

    Integrates q(x) against the transverse-displacement shape functions
    over the overlap of [x_lo, x_hi] with the element domain
    [elem.x_a, elem.x_b], by Gauss-Legendre quadrature.

    Parameters
    ----------
    elem : Elem
        Beam element; ``elem.beam_theory`` selects the slot map.
    x_lo, x_hi : float
        Extent of the distributed load [mm].
    q : callable
        Signed load intensity q(x) [N/mm], already projected onto the
        bending plane being assembled.
    estimator : QuadratureOrderEstimator
        Chooses the Gauss order for q and the element's shape functions.
    theta_fn : callable, optional
        Load direction theta(x) [deg], used only to raise the quadrature
        order when the direction varies along the span.

    Returns
    -------
    numpy.ndarray of shape (4,) or None
        Local vector ``[F_a, M_a, F_b, M_b]`` with forces in N and moments
        in N.mm, in the ``[v_a, th_a, v_b, th_b]`` layout. For Timoshenko
        elements ``M_a = M_b = 0``. None if the load does not overlap the
        element (the caller skips the scatter).

    Notes
    -----
    f_i = int q(x) N_i^v(x) dx over the overlap. Transverse loads only;
    distributed couples are not supported (module docstring,
    "Limitations").

    The Gauss points are mapped onto the overlap [a, b] itself
    (x = (a + b)/2 + (b - a)/2 * xi, Jacobian (b - a)/2), so a load that
    covers only part of the element is integrated over that part only.
    ``Mesh1D`` normally places nodes at x_lo and x_hi, in which case the
    overlap is always a whole element.
    """
    a = max(x_lo, elem.x_a)
    b = min(x_hi, elem.x_b)

    if b <= a:
        return None   # elem's domain does not intersect [x_lo, x_hi]

    n = estimator.gauss_order(q, a, b, theta_fn)
    pts, wts = gauss_points_weights(n)

    f_local = np.zeros(4)
    jacobian = (b - a) / 2.0   # maps Gauss points on [-1, 1] onto the overlap [a, b]
    slot_map = _SLOT_SHAPE_FUNCTION_INDEX[elem.beam_theory]

    for pt, wt in zip(pts, wts):
        x = (a + b) / 2.0 + jacobian * pt
        zeta = elem.natural_coordenates(x)
        N = elem.shape_functions(zeta)
        q_val = q(x)
        for slot in range(4):
            k = slot_map[slot]
            if k is None:
                continue
            f_local[slot] += wt * jacobian * N[k] * q_val

    return f_local


def assemble_distributed_load_vector(
    x_nodes: list[float],
    elements: list[Elem],
    distributed_cases: list[dict],
    estimator: QuadratureOrderEstimator,
) -> np.ndarray:
    """Assemble the global load vector of all distributed transverse loads in one plane.

    Parameters
    ----------
    x_nodes : list of float
        Global node positions [mm].
    elements : list of Elem
        Elements of the mesh, in node order.
    distributed_cases : list of dict
        The ``"distributed_xz"`` or ``"distributed_xy"`` list of one load
        case built by ``vector_external_forces.py``; each entry holds
        ``"x_lo"``, ``"x_hi"`` [mm], ``"q"`` (callable, N/mm) and
        ``"theta_fn"`` (callable or None).
    estimator : QuadratureOrderEstimator
        Gauss-order selector passed to
        :func:`element_distributed_force_vector`.

    Returns
    -------
    numpy.ndarray of shape (3 * len(x_nodes),)
        Global force vector with 3 DOFs per node ``[axial, radial,
        moment]`` (N, N, N.mm), the layout used by the point-load
        assembly. The axial DOF is never loaded; for Timoshenko meshes the
        moment DOFs receive no contribution from transverse loads.
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