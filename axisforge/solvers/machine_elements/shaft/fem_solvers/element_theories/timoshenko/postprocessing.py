"""axisforge/solvers/machine_elements/shaft/element_theories/timoshenko/postprocessing.py

Internal-effort (M, V) recovery for the Timoshenko beam theory, split out
of ShaftResultsReader (results_reader.py). Everything here is about
turning solved nodal DOFs into bending-moment / shear-force diagrams --
deflections, axial/rotation readback, section properties, bearing
reactions and bearing-node data all stay in ShaftResultsReader.

Same shape as element_theories/euler_bernoulli/postprocessing.py
(TimoshenkoPostProcessing mirrors EulerBernoulliPostProcessing):
recover_element_displacements / _element_nodal_displacements /
bending_moment / shear_force / recover_internal_forces / _sweep_plane.

bending_moment() reuses the same code shape as the Euler-Bernoulli version:
elem.bending_strain_matrix(zeta) dispatches internally to
TimoshenkoBeam().bending_strain_matrix(zeta, elem) per elem.py, so
M = D_b * B_b(zeta) @ a^(e), D_b = E*I, is correct for whichever theory
`elem` was built with.

shear_force() is now the real Timoshenko expression: Q = D_s * B_s(zeta) @
a^(e), where B_s comes from elem.shear_strain_matrix(zeta) and D_s (the
shear rigidity, correction factor folded in) from elem.shear_rigidity(
kGA_override=...). This assumes elem.py has been extended with
shear_strain_matrix()/shear_rigidity() dispatch methods mirroring
bending_strain_matrix()/stiffness_element() -- not shown to me yet as
actual code, only discussed; if they don't exist on Elem yet, this file
won't run until they're added there.

TODO(owner): wire up the actual dispatch that decides "this ShaftSystem/
RigidSupportFEMSolver uses Timoshenko -> use this class" vs the
euler_bernoulli sibling. That selection didn't exist in the pasted
results_reader.py (elem.stiffness_element() dispatches per-element
internally, but nothing at the ShaftResultsReader level chooses between
this and element_theories/euler_bernoulli/postprocessing.py yet). Once you
tell me the attribute/flag to key off of (e.g. something on ShaftSystem,
on BeamModelSettings, or on the solver itself), I'll wire it into
results_reader.py.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from axisforge.core.loads import LoadPlane

if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import RigidSupportFEMSolver
    from axisforge.mesh.shaft.element_type.elem import Elem


class TimoshenkoPostProcessing:
    """
    Internal-effort recovery for Timoshenko shaft elements, mirroring
    EulerBernoulliPostProcessing's shape. M and V both go through
    elem-dispatched strain matrices (B_b, B_s) and rigidities (D_b, D_s)
    -- see module docstring.
    """

    def recover_element_displacements(self, solver: "RigidSupportFEMSolver", x_nodes: list[float], n: int):
        """
        Returns a list of (elem, a_xz, a_xy), one entry per element in
        solver.elements order, each a_* a length-4 array [v_a, th_a, v_b, th_b].
        """
        out = []
        for elem in solver.elements:
            a_xz = self._element_nodal_displacements(solver, elem, LoadPlane.XZ)
            a_xy = self._element_nodal_displacements(solver, elem, LoadPlane.XY)
            out.append((elem, a_xz, a_xy))
        return out

    def _element_nodal_displacements(self, solver: "RigidSupportFEMSolver", elem, plane: LoadPlane) -> np.ndarray:
        """
        a^(e) = [v_a, th_a, v_b, th_b] -- the local 4-dof nodal displacement
        vector for one element in one plane, pulled straight from that
        plane's global displacement vector. Bending-only: axial is
        decoupled and not part of a^(e) here.
        """
        d_total = solver.d_total_xz if plane == LoadPlane.XZ else solver.d_total_xy
        i, j = elem.idx_node_1, elem.idx_node_2
        dof = [3 * i + 1, 3 * i + 2, 3 * j + 1, 3 * j + 2]
        return d_total[dof]

    def bending_moment(self, elem: "Elem", a_e: np.ndarray, x: float) -> tuple[float, float]:
        """
        Returns (x, M(x)), per Eq.(1.16c): M(x) = D_b * B_b(zeta) @ a^(e),
        D_b = E*I. Same code as EulerBernoulliPostProcessing.bending_moment()
        -- elem.bending_strain_matrix(zeta) already dispatches to
        TimoshenkoBeam().bending_strain_matrix(zeta, elem) for a Timoshenko
        elem, so the formula itself doesn't change; only elem differs.
        """
        zeta = elem.natural_coordenates(x)
        B_b  = elem.bending_strain_matrix(zeta)
        D_b  = elem.E * elem.I
        M    = float(D_b * (B_b @ a_e))
        return (x, M)

    def shear_force(self, elem: "Elem", a_e: np.ndarray, x: float, *, kGA_override: float | None = None) -> tuple[float, float]:
        """
        Returns (x, Q(x)) = D_s * B_s(zeta) @ a^(e), D_s the shear
        rigidity (shear correction factor folded in via
        elem.shear_rigidity()). Unlike Euler-Bernoulli's V, this is not
        constant within the element in general (B_s depends on zeta for
        the "exact" integration variant -- see TimoshenkoBeam.shear_strain_matrix),
        so x is genuinely used here, not just carried through for shape.

        kGA_override is the solver's per-solve override (solver._kGA_override)
        -- it lives on the solver, not on elem, so it has to be threaded
        through here explicitly by the caller (_sweep_plane).
        """
        zeta = elem.natural_coordenates(x)
        B_s  = elem.shear_strain_matrix(zeta)
        D_s  = elem.shear_rigidity(kGA_override=kGA_override)
        Q    = float(D_s * (B_s @ a_e))
        return (x, Q)

    def recover_internal_forces(self, solver: "RigidSupportFEMSolver", x_nodes: list[float], n: int):
        """
        Returns (M_xz, M_xy, V_xz, V_xy), each a length-n np.ndarray
        aligned with x_nodes -- same shape as
        EulerBernoulliPostProcessing.recover_internal_forces().
        """
        M_xz, V_xz = self._sweep_plane(solver, x_nodes, n, LoadPlane.XZ)
        M_xy, V_xy = self._sweep_plane(solver, x_nodes, n, LoadPlane.XY)
        return M_xz, M_xy, V_xz, V_xy

    def _sweep_plane(self, solver: "RigidSupportFEMSolver", x_nodes: list[float], n: int, plane: LoadPlane):
        """
        Builds M and V over every mesh node, for one plane, by evaluating
        bending_moment()/shear_force() at each element's two end nodes
        (elem.x_a, elem.x_b) using that element's own a^(e). Same
        node-assignment pattern as the Euler-Bernoulli sweep: the first
        element also writes its 'a' node, every element writes its 'b'
        node.
        """
        M = np.zeros(n)
        V = np.zeros(n)

        kGA_override = getattr(solver, "_kGA_override", None)

        for elem_idx, elem in enumerate(solver.elements):
            a_e = self._element_nodal_displacements(solver, elem, plane)
            i, j = elem.idx_node_1, elem.idx_node_2

            _, M1 = self.bending_moment(elem, a_e, elem.x_a)
            _, V1 = self.shear_force(elem, a_e, elem.x_a, kGA_override=kGA_override)
            _, M2 = self.bending_moment(elem, a_e, elem.x_b)
            _, V2 = self.shear_force(elem, a_e, elem.x_b, kGA_override=kGA_override)

            if elem_idx == 0:
                M[i] = M1
                V[i] = V1
            M[j] = M2
            V[j] = V2

        return M, V

    def shear_strain(self, elem: Elem, a_e: np.ndarray, x: float) -> float:

        zeta = elem.natural_coordenates(x)
        B_s  = elem.shear_strain_matrix(zeta)

        shear_strain = B_s @ a_e
        return shear_strain