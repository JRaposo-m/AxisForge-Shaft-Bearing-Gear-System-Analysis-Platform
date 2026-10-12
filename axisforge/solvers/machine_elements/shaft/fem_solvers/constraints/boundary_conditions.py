"""Rigid-support boundary conditions of the shaft FEM: v = 0 at every bearing except thrust bearings,
u = 0 at locating and thrust bearings, rotation never constrained. The module only classifies DOFs
as free or constrained; it does not reduce K or f and does not depend on the beam theory or on
how K was assembled.
"""

from __future__ import annotations

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


def boundary_dofs(x_nodes: list[float], shaft_system: ShaftSystem) -> tuple[list[int], list[int]]:
    """Free and constrained DOFs of the shaft for rigid bearing supports, by elimination.

    Parameters
    ----------
    x_nodes : list of float
        Node positions [mm].
    shaft_system : ShaftSystem
        Provides ``bearings`` (each with ``position`` [mm] and ``arrangement``).

    Returns
    -------
    free : list of int
        Global DOF indices that remain in the solve (DOF order ``[u, v, theta]`` per node, index
        ``3*i, 3*i+1, 3*i+2``).
    constrained : list of int
        Global DOF indices set to zero displacement (an index appears twice if two bearings share a
        node).

    Raises
    ------
    ValueError
        If no bearing is ``"locating"`` or ``"thrust"`` (the axial rigid-body mode would remain), or
        if fewer than two bearings give v = 0 at distinct nodes (the bending rigid-body rotation would
        remain in both planes). Also if a bearing position has no node (``Elem.find_node_index``).

    Warnings
    --------
    A thrust bearing is also accepted as the axial fixed point (the check names both ``"locating"``
    and ``"thrust"``). Bearing stiffness is not used here; Cao2004 couples bearings as two-node 10x10
    tangent matrices, which this solver does not do.

    Notes
    -----
    - ``"thrust"``: u = 0 only (no radial support in this model).
    - ``"locating"``: v = 0 and u = 0.
    - Any other arrangement (floating, non-locating): v = 0 only.
    - Rotation theta is never constrained by a bearing.
    """
    constrained: list[int] = []
    has_axial_constraint = False
    radial_constrained_node_indices: set[int] = set()

    for b in shaft_system.bearings:
        i = Elem.find_node_index(x_nodes, b.position)

        if b.arrangement == "thrust":
            constrained.append(3 * i)          # u = 0 only -- no radial support
            has_axial_constraint = True
            continue

        constrained.append(3 * i + 1)          # v = 0
        radial_constrained_node_indices.add(i)
        if b.arrangement == "locating":
            constrained.append(3 * i)          # u = 0, in addition to v = 0
            has_axial_constraint = True

    if not has_axial_constraint:
        raise ValueError(
            "boundary_dofs: shaft_system has no bearing with arrangement "
            "== 'locating' or 'thrust' -- the axial rigid-body mode is "
            "never constrained, so the system is singular in the axial "
            "DOFs regardless of how the stiffness matrix was assembled. "
            "Mark exactly one bearing as locating (or thrust)."
        )

    if len(radial_constrained_node_indices) < 2:
        raise ValueError(
            f"boundary_dofs: only {len(radial_constrained_node_indices)} "
            "bearing(s) provide radial (v=0) support at distinct node "
            "positions -- at least two are required, otherwise nothing "
            "constrains rigid-body rotation about the single constrained "
            "node in either bending plane. A 'thrust' bearing does not "
            "count here -- it provides no radial support in this model."
        )

    n_dofs = 3 * len(x_nodes)
    free = [d for d in range(n_dofs) if d not in constrained]
    return free, constrained

def boundary_dofs_explicit(
    x_nodes: list[float],
    constrained_dofs: list[int],
) -> tuple[list[int], list[int]]:
    """Free and constrained DOFs from a list of DOF indices given directly.

    For verification rigs only (for example a fixed-free cantilever checked against closed-form beam
    theory) where there is no ``ShaftSystem`` or the bearing rules of ``boundary_dofs`` do not apply
    (a clamped node with u = v = theta = 0 is not a bearing). Not used by any production solve path.

    Parameters
    ----------
    x_nodes : list of float
        Node positions [mm].
    constrained_dofs : list of int
        Global DOF indices to constrain, 3 per node in the order ``[u, v, theta]``
        (``3*i, 3*i+1, 3*i+2``). The caller must choose indices that leave K non-singular: the
        locating-bearing and two-bearing checks of ``boundary_dofs`` are not repeated.

    Returns
    -------
    free : list of int
        Global DOF indices that remain in the solve.
    constrained : list of int
        Sorted, de-duplicated ``constrained_dofs``.

    Raises
    ------
    ValueError
        If an index is outside ``0 .. 3*len(x_nodes) - 1``.
    """
    n_dofs = 3 * len(x_nodes)
    constrained = sorted(set(constrained_dofs))

    invalid = [d for d in constrained if d < 0 or d >= n_dofs]
    if invalid:
        raise ValueError(
            f"boundary_dofs_explicit: DOF index/indices {invalid} out of "
            f"range for {len(x_nodes)} nodes (0..{n_dofs - 1})."
        )

    free = [d for d in range(n_dofs) if d not in constrained]
    return free, constrained