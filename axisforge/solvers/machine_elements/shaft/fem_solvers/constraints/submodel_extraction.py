"""
axisforge/solvers/machine_elements/shaft/fem_solvers/submodel_extraction.py

MOCKUP / SKELETON -- extracted from RigidSupportFEMSolver.return_values()
exactly as it was. Beam-theory-agnostic: it only reads the solver's
already-resolved public attributes (d_total_xz, d_total_xy, f_xz_total,
f_xy_total) -- it never looks at elements, beam, or theory. It only
needs solve() to have already run.

TODO (you): this function currently takes the four arrays explicitly
-- decide whether you would rather keep calling it as
solver.return_values(x) (a method) or as the free function
extract_submodel_values(solver, x) taking the whole solver. Left as a
free function taking the arrays explicitly to avoid a circular
dependency with rigid_support.py; adjust to whatever you find cleanest.
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
    """
    Extracted from RigidSupportFEMSolver.return_values() -- logic
    unchanged (it only stopped reading self.*, and now takes the
    already-resolved arrays as parameters instead).

    Extracts nodal solution quantities for every node within the
    interval [x[0], x[1]].

    Parameters
    ----------
    x_nodes : mesh node positions of the current FEM solution
    x       : [x_lo, x_hi] -- interval bounds (must coincide with nodes)
    d_total_xz, d_total_xy, f_xz_total, f_xy_total :
        attributes already published by RigidSupportFEMSolver.solve()

    Returns
    -------
    dict mapping each node position within [x_lo, x_hi] to its
    quantities:
        {
            x_i: {"v_xz": ..., "theta_xz": ..., "f1_xz": ..., "f2_xz": ...,
                   "v_xy": ..., "theta_xy": ..., "f1_xy": ..., "f2_xy": ...},
            ...
        }

    Notes
    -----
    v     : transverse displacement   [mm]
    theta : rotation                  [rad]
    f1    : nodal transverse force    [N]
    f2    : nodal moment              [N*mm]

    Raises
    ------
    RuntimeError if solve() has not been called yet.
    ValueError   if x does not contain exactly two positions.
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