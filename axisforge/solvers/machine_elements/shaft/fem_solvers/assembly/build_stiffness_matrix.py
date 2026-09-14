"""
axisforge/solvers/machine_elements/shaft/fem_solvers/assembly/build_stiffness_matrix.py

Assembles the global shaft stiffness matrix from a mesh's elements,
plus optional stiffness injected at chosen axial positions (bearings,
or any other nodal spring). Fully theory-agnostic: each Elem already
knows its own beam_theory (set once via BeamModelSettings at
construction) and dispatches its own stiffness_element() accordingly
-- this builder never branches on theory, never holds a "beam" object,
and never re-validates shear_theory (Elem/BeamModelSettings already
guarantee a valid, consistent element before it ever reaches here).

DOF layout: 3 per node [axial, transverse, rotation] -- same
convention as load_vector_point.py / distributed_loads.py.

`frame` (required, no default -- see BeamModelSettings-style "no hidden
decisions" convention) decides, at the ASSEMBLY level, whether axial
behaviour is actually wired into the global matrix:

  - frame=True: each element contributes ONE 6x6 block, local order
    [u_a, v_a, th_a, u_b, v_b, th_b] -- deliberately matching the
    global per-node order [axial, transverse, rotation], so the scatter
    is a single direct mapping ([3i, 3i+1, 3i+2, 3j, 3j+1, 3j+2]), no
    separate bend_dofs/axial_dofs bookkeeping. The 6x6 itself is built
    here from elem.stiffness_element() (4x4 bending) +
    elem.axial_stiffness_element() (2x2 axial) -- still two independent,
    uncoupled blocks physically, just assembled into one array before
    scatter.
  - frame=False: only the 4x4 bending block is scattered
    ([3i+1, 3i+2, 3j+1, 3j+2]); elem.axial_stiffness_element() is never
    called and the axial DOFs (3*i, 3*j for every node) are left at
    zero. FLAG: this makes K singular in the axial DOFs unless every
    axial DOF is separately constrained (e.g. by boundary conditions
    removing them from the solve) -- this builder does not check that;
    it is the caller's responsibility, same as any other unconstrained
    DOF.

This is a property of what you want assembled, not of the element or
the beam theory -- Elem still always has both stiffness_element() and
axial_stiffness_element() available regardless of `frame`, exactly as
before.
"""

from __future__ import annotations

import numpy as np

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D


class StiffnessMatrixBuilder:
    """
    Usage:
        builder = StiffnessMatrixBuilder(mesh, elements, frame=True)
        builder.add_nodal_stiffness(x=120.0, K_local=K_bearing, dof_slots=[1, 2])
        K = builder.build()

    add_nodal_stiffness() is generic on purpose -- it adds any local
    stiffness block at the node nearest a given x, into whichever DOF
    slots the caller names (0=axial, 1=transverse, 2=rotation). It does
    not know what a bearing is or how its stiffness is computed; the
    caller supplies the numbers.
    """

    def __init__(self, mesh: Mesh1D, elements: list[Elem], frame: bool):
        self.mesh = mesh
        self.elements = elements
        self.frame = frame
        self.x_nodes = mesh.x_nodes
        self._n_nodes = mesh.n_nodes
        self._n_dofs = 3 * self._n_nodes
        self._K = np.zeros((self._n_dofs, self._n_dofs))
        self._beam_assembled = False

    # ------------------------------------------------------------------
    # Beam stiffness -- from the elements themselves
    # ------------------------------------------------------------------

    @staticmethod
    def _element_stiffness_6x6(elem: Elem, *, kGA_override: float | None = None) -> np.ndarray:
        """
        Combines elem's bending block (4x4, [v_a, th_a, v_b, th_b],
        theory-dispatched by elem itself) and axial block (2x2,
        [u_a, u_b]) into one 6x6, local order
        [u_a, v_a, th_a, u_b, v_b, th_b]. Off-diagonal cross terms
        between axial and bending stay zero -- the two are physically
        uncoupled, this is bookkeeping, not new physics.
        """
        if elem.beam_theory == "timoshenko":
            K_bend = elem.stiffness_element(kGA_override=kGA_override)
        else:
            K_bend = elem.stiffness_element()
        K_ax = elem.axial_stiffness_element()

        K = np.zeros((6, 6))
        bend_idx = (1, 2, 4, 5)   # v_a, th_a, v_b, th_b -> slots in the 6x6
        axial_idx = (0, 3)        # u_a, u_b -> slots in the 6x6

        for a, ga in enumerate(bend_idx):
            for b, gb in enumerate(bend_idx):
                K[ga, gb] += K_bend[a, b]
        for a, ga in enumerate(axial_idx):
            for b, gb in enumerate(axial_idx):
                K[ga, gb] += K_ax[a, b]
        return K

    def _assemble_beam_stiffness(self, *, kGA_override: float | None = None) -> None:
        """
        Scatters every element's stiffness into self._K. Called once,
        lazily, from build().

        kGA_override is forwarded only to elements whose beam_theory is
        "timoshenko" -- silently ignored for "euler_bernoulli" elements,
        which have no shear term to override (same "has no effect for
        euler" convention already used for shear_theory elsewhere in
        this codebase).
        """
        for elem in self.elements:
            i, j = elem.idx_node_1, elem.idx_node_2

            if self.frame:
                K_elem = self._element_stiffness_6x6(elem, kGA_override=kGA_override)
                dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
            else:
                if elem.beam_theory == "timoshenko":
                    K_elem = elem.stiffness_element(kGA_override=kGA_override)
                else:
                    K_elem = elem.stiffness_element()
                dofs = [3 * i + 1, 3 * i + 2, 3 * j + 1, 3 * j + 2]  # v_a, th_a, v_b, th_b

            for a, ga in enumerate(dofs):
                for b, gb in enumerate(dofs):
                    self._K[ga, gb] += K_elem[a, b]

        self._beam_assembled = True

    # ------------------------------------------------------------------
    # Nodal stiffness -- bearings, or any other spring at a position
    # ------------------------------------------------------------------

    def add_nodal_stiffness(self, x: float, K_local: np.ndarray,
                             dof_slots: list[int],
                             node_tol: float | None = None) -> None:
        """
        Adds K_local (an len(dof_slots) x len(dof_slots) matrix) to the
        global stiffness at the node nearest x, at the DOF slots named
        by dof_slots (0=axial, 1=transverse, 2=rotation, in the node's
        local 3-DOF block).

        Example -- a bearing with only radial stiffness K_r at x=120:
            builder.add_nodal_stiffness(120.0, np.array([[K_r]]), dof_slots=[1])

        Example -- a bearing with coupled radial/rotational stiffness:
            builder.add_nodal_stiffness(120.0, K_2x2, dof_slots=[1, 2])

        Raises if K_local's shape doesn't match len(dof_slots), or if
        no node exists at x within tolerance (see Elem.find_node_index).
        """
        K_local = np.asarray(K_local)
        if K_local.shape != (len(dof_slots), len(dof_slots)):
            raise ValueError(
                f"add_nodal_stiffness: K_local shape {K_local.shape} does not "
                f"match dof_slots length {len(dof_slots)}"
            )

        kwargs = {} if node_tol is None else {"tol": node_tol}
        node_idx = Elem.find_node_index(self.x_nodes, x, **kwargs)

        global_dofs = [3 * node_idx + slot for slot in dof_slots]
        for a, ga in enumerate(global_dofs):
            for b, gb in enumerate(global_dofs):
                self._K[ga, gb] += K_local[a, b]

    # ------------------------------------------------------------------

    def build(self, *, kGA_override: float | None = None) -> np.ndarray:
        if not self._beam_assembled:
            self._assemble_beam_stiffness(kGA_override=kGA_override)
        return self._K