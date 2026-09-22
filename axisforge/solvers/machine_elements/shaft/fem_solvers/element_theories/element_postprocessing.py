"""axisforge/solvers/machine_elements/shaft/element_theories/element_postprocessing.py

Internal-effort (M, V) recovery for the shaft's beam element theories,
split out of ShaftResultsReader (results_reader.py). Everything here is
about turning solved nodal DOFs into bending-moment / shear-force
diagrams -- deflections, axial/rotation readback, section properties,
bearing reactions and bearing-node data all stay in ShaftResultsReader.

Merged from two previously separate, near-identical files:
    element_theories/euler_bernoulli/postprocessing.py -> EulerBernoulliPostProcessing
    element_theories/timoshenko/postprocessing.py       -> TimoshenkoPostProcessing
into this single module, with the shared shape pulled up into an ABC
(ElementTheoryPostProcessor) and only the genuinely theory-specific part
(shear_force) left abstract. See class docstrings for what changed and
why relative to the two source files.

NOT changed relative to the two source files, only flagged:
    - Sign convention mismatch between the two shear_force()
      implementations (Euler-Bernoulli negates the closed-form
      expression, Timoshenko does not). Preserved as-is from both
      originals -- see the TODO(owner) on ElementTheoryPostProcessor.
      Needs an explicit decision before this is trusted for a case that
      compares/mixes V_xz, V_xy across theories.
    - TODO(owner) about theory dispatch (which class ShaftResultsReader/
      RigidSupportFEMSolver should instantiate for a given ShaftSystem)
      still open -- carried over unchanged from both source files, not
      resolved by this merge.

TODO(owner): wire up the actual dispatch that decides "this ShaftSystem/
RigidSupportFEMSolver uses Euler-Bernoulli -> EulerBernoulliPostProcessing"
vs "-> TimoshenkoPostProcessing". That selection didn't exist in either
source file (elem.stiffness_element() dispatches per-element internally,
but nothing at the ShaftResultsReader level chooses between the two
classes here yet). Once you tell me the attribute/flag to key off of
(e.g. something on ShaftSystem, on BeamModelSettings, or on the solver
itself), this can be wired into results_reader.py -- and would be a
natural place for the same @register_family-style pattern already used
elsewhere in the project (see BearingFamily(ABC)), if you want the two
subclasses to self-register against a theory key instead of being
imported and branched on by name.

TODO(owner): sign convention. EulerBernoulliPostProcessing.shear_force()
returns a negated value; TimoshenkoPostProcessing.shear_force() does not.
Confirm which is the intended sign before this is relied on anywhere that
compares or mixes V_xz/V_xy across the two theories (e.g. a mesh-theory
convergence study, or any diagram that overlays both).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from typing import TYPE_CHECKING

from axisforge.core.loads import LoadPlane

if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import RigidSupportFEMSolver
    from axisforge.mesh.shaft.element_type.elem import Elem


class ElementTheoryPostProcessor(ABC):
    """
    Common contract for recovering M(x)/V(x) from solved nodal DOFs, per
    beam element theory (Euler-Bernoulli, Timoshenko, ...).

    Everything except shear_force() turned out to be identical code
    between the two original classes:

    - recover_element_displacements / _element_nodal_displacements: pull
      a^(e) = [v_a, th_a, v_b, th_b] out of the solver's global
      displacement vector for one element/plane. Bending-only -- axial is
      decoupled and not part of a^(e) here. Theory-agnostic by
      construction (just indexing into d_total_xz/d_total_xy).

    - bending_moment: M(x) = D_b * B_b(zeta) @ a^(e), D_b = E*I. Also
      theory-agnostic at this level -- elem.bending_strain_matrix(zeta)
      already dispatches internally to the right theory's B_b for
      whichever Elem instance it's called on, so the two original
      subclasses had byte-identical bodies here. Kept concrete on the
      base class rather than duplicated or made abstract.

    - recover_internal_forces / _sweep_plane: build M/V over every mesh
      node by evaluating bending_moment()/shear_force() at each
      element's two end nodes (elem.x_a, elem.x_b), using that element's
      own a^(e). Node-assignment pattern: the first element also writes
      its 'a' node, every element writes its 'b' node -- M/V are taken as
      continuous at shared nodes for this element formulation (no applied
      point moments/point shear jumps assumed exactly at a shared node).
      kGA_override is read off the solver (solver._kGA_override) and
      threaded into shear_force() unconditionally; Euler-Bernoulli's
      implementation simply ignores it (see below).

    shear_force() is the one method each subclass must supply: Euler-
    Bernoulli uses a closed-form cubic-Hermite shear expression that does
    not depend on elem's own dispatch methods; Timoshenko goes through
    elem.shear_strain_matrix()/elem.shear_rigidity(). See the TODO(owner)
    above the sign-convention note in the module docstring before trusting
    both sides against each other.
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
        Returns (x, M(x)) -- not just M(x) -- so each call already comes
        out as an (x_coord, value) pair ready to drop into the eventual
        x-vs-M map across the whole shaft, per Eq.(1.16c):
        M(x) = D_b * B_b(zeta) @ a^(e), D_b = E*I.

        zeta is elem's own natural coordinate for x -- elem.bending_strain_matrix(zeta)
        already dispatches to the right theory's B_b for whichever Elem
        instance this is (confirmed for Euler-Bernoulli: returns a flat
        (4,) array, so a plain B_b @ a_e is a true scalar -- no ravel
        needed). Identical for both theories, so this lives on the base
        class rather than being duplicated per subclass.
        """
        zeta = elem.natural_coordenates(x)
        B_b  = elem.bending_strain_matrix(zeta)
        D_b  = elem.E * elem.I
        M    = float(D_b * (B_b @ a_e))
        return (x, M)

    @abstractmethod
    def shear_force(self, elem: "Elem", a_e: np.ndarray, x: float, *, kGA_override: float | None = None) -> tuple[float, float]:
        """
        Returns (x, Q(x)). The one method that genuinely differs by
        theory -- see each subclass. kGA_override is accepted by every
        implementation for a uniform call shape from _sweep_plane(), even
        where a subclass has no use for it (Euler-Bernoulli).
        """
        raise NotImplementedError

    def recover_internal_forces(self, solver: "RigidSupportFEMSolver", x_nodes: list[float], n: int):
        """
        Returns (M_xz, M_xy, V_xz, V_xy), each a length-n np.ndarray
        aligned with x_nodes -- the shape ShaftResultsReader.read() /
        ShaftResults expect (one M and V value per mesh node, matching
        x_nodes/x exactly).

        This is the entry point results_reader.py should call. Being a
        bound method (not a module-level function like the old sweep),
        the caller instantiates the concrete subclass first, e.g.:
            EulerBernoulliPostProcessing().recover_internal_forces(solver, x_nodes, n)
        """
        M_xz, V_xz = self._sweep_plane(solver, x_nodes, n, LoadPlane.XZ)
        M_xy, V_xy = self._sweep_plane(solver, x_nodes, n, LoadPlane.XY)
        return M_xz, M_xy, V_xz, V_xy

    def _sweep_plane(self, solver: "RigidSupportFEMSolver", x_nodes: list[float], n: int, plane: LoadPlane):
        """
        Builds M and V over every mesh node, for one plane. kGA_override
        is read off the solver once per sweep and passed to every
        shear_force() call; subclasses that don't need it (Euler-
        Bernoulli) simply ignore the keyword.
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


class EulerBernoulliPostProcessing(ElementTheoryPostProcessor):
    """
    Euler-Bernoulli shear recovery: closed-form cubic-Hermite expression,
    per Eq.(1.24): Q = (E*I)/(l_e**3) * [12, 6*l_e, -12, 6*l_e] @ a_e.
    Unchanged from the original euler_bernoulli/postprocessing.py,
    including the leading minus sign -- see module docstring TODO(owner)
    on sign convention before trusting this against the Timoshenko side.
    """

    def shear_force(self, elem: "Elem", a_e: np.ndarray, x: float, *, kGA_override: float | None = None) -> tuple[float, float]:
        """
        Returns (x, Q(x)). Q doesn't actually depend on x (shear is
        constant within an Euler-Bernoulli element -- third derivative of
        a cubic Hermite shape function is constant), but x is still
        taken/returned to match bending_moment()'s (x, value) shape.
        kGA_override has no meaning for this theory and is accepted only
        for call-shape parity with TimoshenkoPostProcessing.
        """
        E  = elem.E
        I  = elem.I
        le = elem.length
        Q  = -float(E * I / (le ** 3) * ([12, 6 * le, -12, 6 * le] @ a_e))
        return (x, Q)


class TimoshenkoPostProcessing(ElementTheoryPostProcessor):
    """
    Timoshenko shear recovery: Q = D_s * B_s(zeta) @ a^(e), where B_s
    comes from elem.shear_strain_matrix(zeta) and D_s (the shear
    rigidity, correction factor folded in) from
    elem.shear_rigidity(kGA_override=...). This assumes elem.py has been
    extended with shear_strain_matrix()/shear_rigidity() dispatch methods
    mirroring bending_strain_matrix()/stiffness_element() -- not shown as
    actual code, only discussed; if they don't exist on Elem yet, this
    class won't run until they're added there. Unlike Euler-Bernoulli's Q,
    this is not constant within the element in general (B_s depends on
    zeta for the "exact" integration variant), so x is genuinely used
    here, not just carried through for shape.

    Unchanged from the original timoshenko/postprocessing.py, including
    the absence of a leading minus sign -- see module docstring
    TODO(owner) on sign convention.
    """

    def shear_force(self, elem: "Elem", a_e: np.ndarray, x: float, *, kGA_override: float | None = None) -> tuple[float, float]:
        zeta = elem.natural_coordenates(x)
        B_s  = elem.shear_strain_matrix(zeta)
        D_s  = elem.shear_rigidity(kGA_override=kGA_override)
        Q    = float(D_s * (B_s @ a_e))
        return (x, Q)

    def shear_strain(self, elem: "Elem", a_e: np.ndarray, x: float) -> float:
        """Timoshenko-only: not part of the shared contract. Carried over
        unchanged from the original file."""
        zeta = elem.natural_coordenates(x)
        B_s  = elem.shear_strain_matrix(zeta)
        return B_s @ a_e