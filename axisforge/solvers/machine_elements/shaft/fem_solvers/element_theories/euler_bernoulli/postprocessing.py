"""axisforge/solvers/machine_elements/shaft/element_theories/euler_bernoulli/postprocessing.py

Internal-effort (M, V) recovery for the Euler-Bernoulli beam theory, split out
of ShaftResultsReader (results_reader.py). Everything here is about
turning solved nodal DOFs into bending-moment / shear-force diagrams --
deflections, axial/rotation readback, section properties, bearing
reactions and bearing-node data all stay in ShaftResultsReader.

The old k_e @ d_e sweep (recover_internal_forces/_sweep_plane_from_elements,
carried over from results_reader.py) has been dropped -- it's no longer the
approach used here. This file is rebuilt around Eq.(1.16c):
M = (E*I_y) * B_b * a^(e), and Eq.(1.24) for V, where a^(e) is the element's
local nodal displacement vector and B_b is the curvature-displacement row
vector (linear in x within the element, per the reference text; V is
constant within the element).

IMPORTANT for results_reader.py: recover_internal_forces() below is a
bound method on EulerBernoulliPostProcessing, not a module-level function.
The caller needs to instantiate the class:
    EulerBernoulliPostProcessing().recover_internal_forces(solver, x_nodes, n)
rather than importing a bare `recover_internal_forces` from this module.

TODO(owner): wire up the actual dispatch that decides "this ShaftSystem/
RigidSupportFEMSolver uses Euler-Bernoulli -> use this class" vs the
timoshenko sibling. That selection didn't exist in the pasted
results_reader.py (elem.stiffness_element() dispatches per-element
internally, but nothing at the ShaftResultsReader level chooses between
this and element_theories/timoshenko/postprocessing.py yet). Once you tell
me the attribute/flag to key off of (e.g. something on ShaftSystem, on
BeamModelSettings, or on the solver itself), I'll wire it into
results_reader.py.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from axisforge.core.loads import LoadPlane

if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import RigidSupportFEMSolver
    from axisforge.mesh.shaft.element_type.elem import Elem


class EulerBernoulliPostProcessing:
    """
    Internal-effort recovery for Euler-Bernoulli shaft elements, built up
    from Eq.(1.16c) for M and Eq.(1.24) for V.
    """

    def recover_element_displacements(self, solver: RigidSupportFEMSolver, x_nodes: list[float], n: int):
        """
        Returns a list of (elem, a_xz, a_xy), one entry per element in
        solver.elements order, each a_* a length-4 array [v_a, th_a, v_b, th_b]
        per Eq.(1.16a)/(1.16c).
        """
        out = []
        for elem in solver.elements:
            a_xz = self._element_nodal_displacements(solver, elem, LoadPlane.XZ)
            a_xy = self._element_nodal_displacements(solver, elem, LoadPlane.XY)
            out.append((elem, a_xz, a_xy))
        return out

    def _element_nodal_displacements(self, solver: RigidSupportFEMSolver, elem, plane: LoadPlane) -> np.ndarray:
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

    def bending_moment(self, elem: Elem, a_e: np.ndarray, x: float) -> tuple[float, float]:
        """
        Returns (x, M(x)) -- not just M(x) -- so each call already comes
        out as an (x_coord, value) pair ready to drop into the eventual
        x-vs-M map across the whole shaft, per Eq.(1.16c):
        M(x) = (E*I) * B_b(zeta) @ a^(e).

        zeta is elem's own natural coordinate for x -- elem.bending_strain_matrix(zeta)
        already dispatches to EulerBernoulliBeam().bending_strain_matrix(zeta, elem)
        (confirmed: it returns a flat (4,) array, so a plain B_b @ a_e is
        a true scalar here -- no ravel needed).
        """
        zeta = elem.natural_coordenates(x)
        B_b  = elem.bending_strain_matrix(zeta)
        M    = float(elem.E * elem.I * (B_b @ a_e))
        return (x, M)

    def shear_force(self, elem: Elem, a_e: np.ndarray, x: float) -> tuple[float, float]:
        """
        Returns (x, Q(x)) -- not just Q(x) -- so each call already comes
        out as an (x_coord, value) pair ready to drop into the eventual
        x-vs-Q map across the whole shaft, per Eq.(1.24):
        Q(x) = (E*I)/(l_e**3) * [12 , 6*l_e, -12, 6*l_e] @ a_e.

        Q doesn't actually depend on x (shear is constant within an
        Euler-Bernoulli element -- third derivative of a cubic Hermite
        shape function is constant), but x is still taken/returned to
        match bending_moment()'s (x, value) shape for the eventual map.
        """
        E  = elem.E
        I  = elem.I
        le = elem.length
        Q  = float(E * I / (le ** 3) * ([12, 6 * le, -12, 6 * le] @ a_e))
        return (x, Q)

    def recover_internal_forces(self, solver: RigidSupportFEMSolver, x_nodes: list[float], n: int):
        """
        Returns (M_xz, M_xy, V_xz, V_xy), each a length-n np.ndarray
        aligned with x_nodes -- the shape ShaftResultsReader.read() /
        ShaftResults expect (one M and V value per mesh node, matching
        `x_nodes`/`x` exactly).

        This is the entry point results_reader.py should call for the
        Euler-Bernoulli theory. Being a bound method (not a module-level
        function like the old sweep), the caller instantiates this class
        first -- see the module docstring.
        """
        M_xz, V_xz = self._sweep_plane(solver, x_nodes, n, LoadPlane.XZ)
        M_xy, V_xy = self._sweep_plane(solver, x_nodes, n, LoadPlane.XY)
        return M_xz, M_xy, V_xz, V_xy

    def _sweep_plane(self, solver: RigidSupportFEMSolver, x_nodes: list[float], n: int, plane: LoadPlane):
        """
        Builds M and V over every mesh node, for one plane, by evaluating
        bending_moment()/shear_force() at each element's two end nodes
        (elem.x_a, elem.x_b) using that element's own a^(e). Mirrors the
        node-assignment pattern of the old k_e @ d_e sweep: the first
        element also writes its 'a' node, every element writes its 'b'
        node -- M/V are taken as continuous at shared nodes for this
        element formulation (no applied point moments/point shear jumps
        assumed exactly at a shared node).
        """
        M = np.zeros(n)
        V = np.zeros(n)

        for elem_idx, elem in enumerate(solver.elements):
            a_e = self._element_nodal_displacements(solver, elem, plane)
            i, j = elem.idx_node_1, elem.idx_node_2

            _, M1 = self.bending_moment(elem, a_e, elem.x_a)
            _, V1 = self.shear_force(elem, a_e, elem.x_a)
            _, M2 = self.bending_moment(elem, a_e, elem.x_b)
            _, V2 = self.shear_force(elem, a_e, elem.x_b)

            if elem_idx == 0:
                M[i] = M1
                V[i] = V1
            M[j] = M2
            V[j] = V2

        return M, V