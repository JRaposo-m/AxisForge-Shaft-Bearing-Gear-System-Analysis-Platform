"""
axisforge/solvers/machine_elements/shaft/fem_solvers/constraints/boundary_conditions.py

Rigid-support boundary conditions: v=0 always at every bearing
position; u=0 additionally at the "locating" bearing. Beam-theory-
agnostic -- depends only on bearing positions and each bearing's
`arrangement` attribute, never on beam_theory/shear_theory/
integration_method.

This module only classifies DOFs as free/constrained. It does NOT
reduce K or f, and does NOT know or care how the caller assembled K
(in particular, whether StiffnessMatrixBuilder was built with
frame=True or frame=False) -- that pairing lives in the solver's
orchestration (rigid_support.py::RigidSupportFEMSolver.solve()), which
is the only place that has both the `frame` choice and this function's
output at once. See the note in build_stiffness_matrix.py's own
docstring on what frame=False leaves unconstrained -- if the solver
uses frame=False, the axial DOFs constrained here are NOT enough by
themselves to make K solvable, and the solver -- not this module --
is responsible for catching that.

What IS checked here, because it needs nothing beyond shaft_system
itself:

  - at least one bearing must be "locating", or every axial DOF in the
    model is left without a single fixed reference point (an
    unconstrained axial rigid-body mode);
  - at least two bearings at distinct node positions, or bending in
    each plane is left with an unconstrained rigid-body rotation --
    a single v=0 constraint pins translation at that node but nothing
    restrains rotation about it (no bearing ever constrains theta).

Both are properties of shaft_system.bearings alone, independent of
anything else in the assembly (frame, beam_theory, load cases).
Silently returning DOF lists that guarantee a singular system later --
surfacing only as an opaque conditioning warning/error far downstream,
with no indication that the real cause is "not enough bearings" -- is
exactly what "flag, don't silently fix" warns against, so this raises
immediately instead, with a message that says what is actually wrong.
"""

from __future__ import annotations

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


def boundary_dofs(x_nodes: list[float], shaft_system: ShaftSystem) -> tuple[list[int], list[int]]:
    """
    Rigid by construction:
      - v=0 at every bearing EXCEPT "thrust" (a thrust bearing provides
        no radial support in this model -- axial only).
      - u=0 at "locating" (in addition to v=0) and at "thrust" (instead
        of v=0) -- both are axial fixed points; "floating"/"non-locating"
        get v=0 only.

    Raises ValueError if no bearing is "locating" or "thrust" (axial
    rigid-body mode never removed), or if fewer than two bearings
    provide v=0 at distinct node positions (bending rigid-body rotation
    never removed in either plane) -- regardless of how K was assembled.
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
    """
    Direct DOF-index boundary condition, bypassing ShaftSystem/bearing
    inference entirely. For verification/test rigs only (e.g. a pure
    cantilever fixed-free case to check element-level convergence
    against closed-form beam theory) where there is no ShaftSystem, or
    where the rigid-bearing rules in boundary_dofs() above don't apply
    (a single "wall" node with u=v=theta=0 has no bearing at all --
    no bearing arrangement ever constrains theta).

    `constrained_dofs` are raw global DOF indices, 3 per node in the
    same [u, v, theta] order boundary_dofs() uses (3*i, 3*i+1, 3*i+2
    for node i). Caller is responsible for choosing indices that leave
    K non-singular -- this function does not replicate boundary_dofs()'s
    locating-bearing / two-bearing sanity checks, since those are
    specific to the bearing-based rigid support model and don't
    generalize to an arbitrary DOF list.

    NOT used by RigidSupportFEMSolver or any production solve path --
    boundary_dofs() (bearing-driven) remains the only BC source there.
    This exists purely so ad-hoc verification scripts can drive the
    same assembly with a boundary condition that isn't expressible as
    a bearing arrangement.
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