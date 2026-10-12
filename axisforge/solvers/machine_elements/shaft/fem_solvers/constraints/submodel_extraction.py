"""Extraction of the nodal displacements and nodal forces of a global shaft solution inside an
interval, used as the cut-boundary data of the sub-model solve. It only reads the solution arrays
of ``RigidSupportFEMSolver.solve`` and does not depend on the beam theory.
"""

from __future__ import annotations

import numpy as np

from axisforge.config import SOLVER_TOLERANCE


def extract_submodel_values(
    x_nodes: list[float],
    x: list[float],
    d_total_xz: np.ndarray | None,
    d_total_xy: np.ndarray | None,
    f_xz_total: np.ndarray | None,
    f_xy_total: np.ndarray | None,
) -> dict[float, dict[str, float]]:
    """Nodal solution values of every node inside the interval ``[x[0], x[1]]``.

    Parameters
    ----------
    x_nodes : list of float
        Node positions of the global solution [mm].
    x : list of float
        ``[x_lo, x_hi]``, interval bounds [mm]; they must coincide with nodes (a tolerance of
        ``SOLVER_TOLERANCE`` is applied at the ends).
    d_total_xz, d_total_xy : ndarray or None
        Nodal displacements of the XZ and XY planes, DOF order ``[u, v, theta]`` per node [mm, mm,
        rad].
    f_xz_total, f_xy_total : ndarray or None
        Nodal forces ``K d`` of the XZ and XY planes, same DOF order [N, N, N mm].

    Returns
    -------
    dict
        Maps each node position in the interval to a dict with the keys ``"u"`` [mm] (axial
        displacement, read from the XZ solution), ``"v_xz"`` [mm], ``"theta_xz"`` [rad], ``"f_u"`` [N],
        ``"f_v_xz"`` [N], ``"f_theta_xz"`` [N mm], ``"v_xy"`` [mm], ``"theta_xy"`` [rad], ``"f1_xy"``
        [N] (transverse force) and ``"f2_xy"`` [N mm] (moment). The two planes use different names
        for the force keys.

    Raises
    ------
    RuntimeError
        If ``d_total_xz`` is ``None`` (``solve`` has not been called).
    ValueError
        If ``x`` does not hold exactly two values.

    Notes
    -----
    The forces are rows of ``K d``, that is the applied nodal load plus the support reaction (not
    ``K d - f_ext``).

    Sign: ``K d`` at a node is the sum of the external forces on that node, positive along the
    positive DOF directions (+Z for ``_xz``, +Y for ``_xy``, +X axial). At a free node it is the
    applied nodal load (+1000 N for 1000 N along +Y on the node; 0 at an unloaded node). At a
    bearing node with no load on it, it is the force of the support on the shaft, opposite to the
    load (-666.67 N and -333.33 N for that load between two bearings).
    """
    if d_total_xz is None:
        raise RuntimeError(
            "extract_submodel_values() called before solve(). "
            "Call solver.solve(shaft_system) first."
        )

    if len(x) != 2:
        raise ValueError(
            f"x must contain exactly [x_lo, x_hi], got {len(x)} values."
        )

    x_lo, x_hi = x[0], x[1]
    result: dict[float, dict[str, float]] = {}

    for i, xi in enumerate(x_nodes):
        if x_lo - SOLVER_TOLERANCE <= xi <= x_hi + SOLVER_TOLERANCE:
            result[xi] = {
                "u":            float(d_total_xz[3 * i]),
                "v_xz":         float(d_total_xz[3 * i + 1]),
                "theta_xz":     float(d_total_xz[3 * i + 2]),
                "f_u":          float(f_xz_total[3 * i]),
                "f_v_xz":       float(f_xz_total[3 * i + 1]),
                "f_theta_xz":   float(f_xz_total[3 * i + 2]),
                "v_xy":         float(d_total_xy[3 * i + 1]),
                "theta_xy":     float(d_total_xy[3 * i + 2]),
                "f1_xy":        float(f_xy_total[3 * i + 1]),
                "f2_xy":        float(f_xy_total[3 * i + 2]),
            }

    return result