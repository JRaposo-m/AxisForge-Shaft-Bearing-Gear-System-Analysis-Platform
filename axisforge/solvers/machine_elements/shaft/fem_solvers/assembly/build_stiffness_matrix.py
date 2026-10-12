"""Assembly of the global shaft stiffness matrix from the stiffness matrices of the beam elements,
with optional nodal springs added at chosen axial positions.
"""

from __future__ import annotations

import numpy as np

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D


class StiffnessMatrixBuilder:
    """Global stiffness matrix K of the shaft, assembled element by element (direct stiffness method).

    Each element ``Elem`` knows its own beam theory (Euler-Bernoulli or Timoshenko) and returns its own
    stiffness matrix; the builder never branches on the theory except to forward ``kGA_override`` to
    Timoshenko elements. Axial and bending stiffness are uncoupled, so each element contributes two
    independent blocks.

    Parameters
    ----------
    mesh : Mesh1D
        Mesh providing ``x_nodes`` [mm] and ``n_nodes``.
    elements : list of Elem
        Beam elements of the mesh (element stiffness in N/mm, N and N mm/rad).
    frame : bool
        ``True``: axial DOFs are assembled (6x6 block per element). ``False``: only the 4x4 bending
        block is assembled and the axial DOFs are left at zero.

    Notes
    -----
    DOF layout: 3 per node, ``[u, v, theta]`` = ``[axial displacement [mm], transverse displacement
    [mm], rotation [rad]]``, global index ``3*i, 3*i+1, 3*i+2`` for node ``i``. Element local order
    ``[u_a, v_a, th_a, u_b, v_b, th_b]`` matches the global per-node order, so the scatter is a direct
    mapping. The 6x6 block is the 4x4 bending block in slots (1, 2, 4, 5) plus the 2x2 axial block in
    slots (0, 3); cross terms are zero.

    With ``frame=False`` the matrix is singular in the axial DOFs unless the caller removes them from
    the solve (this class does not check it).

    Examples
    --------
    >>> builder = StiffnessMatrixBuilder(mesh, elements, frame=True)  # doctest: +SKIP
    >>> K = builder.build()  # doctest: +SKIP
    """

    def __init__(self, mesh: Mesh1D, elements: list[Elem], frame: bool):
        """Store the arguments and allocate the zero global matrix; see the class docstring."""
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
        """Element stiffness matrix in the 6-DOF frame layout (bending plus axial, uncoupled).

        Parameters
        ----------
        elem : Elem
            Beam element.
        kGA_override : float or None, optional
            Shear rigidity k G A [N] replacing the element value; used only by Timoshenko elements.

        Returns
        -------
        K : ndarray, shape (6, 6)
            Stiffness in local order ``[u_a, v_a, th_a, u_b, v_b, th_b]`` [N/mm, N, N mm/rad]. The bending
            4x4 block (``[v_a, th_a, v_b, th_b]``) sits in slots (1, 2, 4, 5), the axial 2x2 block
            (``[u_a, u_b]``) in slots (0, 3); the axial-bending cross terms are zero.
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
        """Scatter the stiffness matrix of every element into the global matrix.

        Called once, lazily, from ``build``. ``kGA_override`` is forwarded only to Timoshenko elements and
        is ignored by Euler-Bernoulli elements (no shear term).

        Parameters
        ----------
        kGA_override : float or None, optional
            Shear rigidity k G A [N] replacing the element value (Timoshenko only).

        Notes
        -----
        ``K[ga, gb] += K_elem[a, b]`` over the global DOFs of the element: 6 DOFs
        ``[3i, 3i+1, 3i+2, 3j, 3j+1, 3j+2]`` if ``frame``, otherwise the 4 bending DOFs
        ``[3i+1, 3i+2, 3j+1, 3j+2]``. Assembly of the element equations ``K_e a_e - f_e = q_e``
        (Onate2013_vol2, Eq. (1.19b)) into the global system, as described without an equation number
        in Onate2013_vol2, 1.3.4 (PDF p. 43).
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
        """Add a local stiffness block (for example a bearing spring) at the node nearest to ``x``.

        The method does not know what a bearing is: the caller supplies the numbers. It is not called by
        any solver in ``fem_solvers`` (bearings there are rigid supports, by elimination of DOFs).

        Parameters
        ----------
        x : float
            Axial position of the spring [mm]. A node must exist there, within the tolerance of
            ``Elem.find_node_index``.
        K_local : ndarray, shape (m, m)
            Stiffness block with ``m = len(dof_slots)`` [N/mm for the axial and transverse slots,
            N mm/rad for the rotation slot, coupling terms in the matching units].
        dof_slots : list of int
            DOF slots of the node on which ``K_local`` acts: 0 = axial, 1 = transverse, 2 = rotation.
        node_tol : float or None, optional
            Maximum distance between ``x`` and the node [mm]; ``None`` keeps the default of
            ``Elem.find_node_index``.

        Raises
        ------
        ValueError
            If the shape of ``K_local`` is not ``(m, m)``, or if no node lies within the tolerance.

        Notes
        -----
        ``K[3*node + slot_a, 3*node + slot_b] += K_local[a, b]``. Cao2004 couples a bearing to the shaft
        as a two-node 10x10 tangent matrix instead; this method does not do that.

        Examples
        --------
        Bearing with radial stiffness ``K_r`` only, at x = 120 mm::

            builder.add_nodal_stiffness(120.0, np.array([[K_r]]), dof_slots=[1])

        Bearing with coupled radial and rotational stiffness::

            builder.add_nodal_stiffness(120.0, K_2x2, dof_slots=[1, 2])
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
        """Return the global stiffness matrix, assembling the elements on the first call.

        Parameters
        ----------
        kGA_override : float or None, optional
            Shear rigidity k G A [N] replacing the element value (Timoshenko elements only). Used only on
            the first call; later calls return the already assembled matrix.

        Returns
        -------
        K : ndarray, shape (3 n_nodes, 3 n_nodes)
            Global stiffness matrix [N/mm, N, N mm/rad], DOF order ``[u, v, theta]`` per node. This is the
            internal array (not a copy), including any nodal stiffness added before the call.
        """
        if not self._beam_assembled:
            self._assemble_beam_stiffness(kGA_override=kGA_override)
        return self._K