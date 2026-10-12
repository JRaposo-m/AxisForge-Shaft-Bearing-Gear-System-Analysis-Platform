"""Result container of one sub-model solve (one mesh grade, one interval), after post-processing.
Built by ``build_submodel_result`` from a ``SubmodelSolution``; it holds the per-node arrays of
the sub-model, restricted to [x_lo, x_hi], so that the convergence study can pick any field or
point from them. No torsion and no stress concentration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SubmodelResult"]


def _empty() -> np.ndarray:
    """Empty float array, the default of the per-node fields of ``SubmodelResult``."""
    return np.array([])


@dataclass
class SubmodelResult:
    """Post-processed FEM data of one mesh grade for one interval.

    Parameters
    ----------
    x_lo, x_hi : float
        Interval bounds [mm].
    grade : str
        Mesh grade of this solve.
    x_nodes : list of float
        Node positions of the sub-model mesh at this grade [mm]; the number of nodes gives the
        refinement ratio used by ``RichardsonGCI``.
    u : ndarray
        Axial displacement per node [mm] (from the XZ solution).
    v_xz, v_xy : ndarray
        Transverse deflection per node in the XZ and XY planes [mm].
    theta_xz, theta_xy : ndarray
        Nodal rotation DOF per node [rad].
    M_xz, M_xy : ndarray
        Bending moment per node [N mm], recovered by the element post-processor.
    V_xz, V_xy : ndarray
        Shear force per node [N], recovered by the element post-processor.
    lam_xz, lam_xy : ndarray
        Lagrange multipliers at the two cut nodes (``[u, v, theta]`` at ``x_lo``, then at ``x_hi``)
        [N, N, N mm]; not physical shaft quantities, they check the coupling of the sub-model to the
        global solution. Sign as stored by the solver (see ``SubmodelSolution``): the transverse and
        rotation entries are ``[+V, +M]`` at ``x_lo`` and ``[-V, -M]`` at ``x_hi`` in the shear
        force and bending moment convention of this package.
    """
    x_lo: float
    x_hi: float
    grade: str
    x_nodes: list[float] = field(default_factory=list)

    u: np.ndarray = field(default_factory=_empty)
    v_xz: np.ndarray = field(default_factory=_empty)
    v_xy: np.ndarray = field(default_factory=_empty)
    theta_xz: np.ndarray = field(default_factory=_empty)
    theta_xy: np.ndarray = field(default_factory=_empty)

    M_xz: np.ndarray = field(default_factory=_empty)
    M_xy: np.ndarray = field(default_factory=_empty)
    V_xz: np.ndarray = field(default_factory=_empty)
    V_xy: np.ndarray = field(default_factory=_empty)

    lam_xz: np.ndarray = field(default_factory=_empty)
    lam_xy: np.ndarray = field(default_factory=_empty)