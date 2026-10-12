"""Nodal vector of the point loads of the shaft: each load is added directly to the DOF of the
nearest node, with no shape functions, so it is independent of the beam theory.
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
    """Global load vector of the point loads, injected at the node nearest to each load position.

    Parameters
    ----------
    x_nodes : list of float
        Node positions along the shaft axis [mm].
    radial : list of (float, float)
        Transverse point forces ``(x, F)``: position [mm], force [N] in the plane being assembled.
    axial : list of (float, float)
        Axial point forces ``(x, F)``: position [mm], force [N].
    moments : list of (float, float)
        Point moments ``(x, M)``: position [mm], moment [N mm] in the plane being assembled.

    Returns
    -------
    f : ndarray, shape (3 len(x_nodes),)
        Load vector [N, N, N mm] in the DOF order ``[u, v, theta]`` per node.

    Raises
    ------
    ValueError
        If a load position has no node within the tolerance of ``Elem.find_node_index``.

    Notes
    -----
    ``f[3i] += F_axial``, ``f[3i+1] += F_radial``, ``f[3i+2] += M``, with ``i`` the node nearest to
    the load. Point loads are added directly to the nodal DOF of the assembled system,
    Onate2013_vol2, Eq. (1.21d). Taking the nearest node is a choice of this code; every load position
    is a mandatory mesh node, so the snap is exact within the tolerance. The lists are built by
    ``build_load_cases``.
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