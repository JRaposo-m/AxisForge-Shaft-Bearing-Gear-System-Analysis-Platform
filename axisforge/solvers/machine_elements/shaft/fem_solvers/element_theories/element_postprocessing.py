"""Recovery of the internal bending moment M and shear force V of the shaft beam elements from the
solved nodal DOFs (Euler-Bernoulli and Timoshenko theories), as values at the mesh nodes. Axial
force, deflections, section properties and bearing reactions are handled in ``global_postprocessing``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from typing import Any, TYPE_CHECKING

from axisforge.core.loads import LoadPlane

if TYPE_CHECKING:
    from axisforge.mesh.shaft.element_type.elem import Elem



class ElementTheoryPostProcessor(ABC):
    """Common contract for recovering M(x) and V(x) from solved nodal DOFs, per beam element theory.

    Everything except ``shear_force`` is identical for the two theories and lives here: extraction of
    the element displacement vector, the bending moment, and the sweep over the elements that writes
    M and V at the mesh nodes. ``shear_force`` is the one theory-specific method.

    Warnings
    --------
    Onate2013_vol2 samples the bending moment at the Gauss points for the Euler-Bernoulli element
    (1.3.4, Fig. 1.11) and M and Q at the element mid-point for the 2-noded Timoshenko element
    (2.3.2); this code writes the values at the element ends instead.

    Notes
    -----
    Element displacement vector (one plane): ``a_e = [v_a, th_a, v_b, th_b]`` with ``v`` in mm and
    ``th`` in rad; the axial DOF is uncoupled and not part of it.

    Bending moment (both theories): ``M(x) = D_b * B_b(zeta) @ a_e`` with ``D_b = E I`` [N mm^2] and
    ``B_b`` the bending-strain row of the element (``elem.bending_strain_matrix``). ``B_b @ a_e`` is
    the bending strain [1/mm]: ``d2v/dx2`` for Euler-Bernoulli, so ``M = +E I d2v/dx2`` (Onate2013_vol2,
    Eq. (1.5), (1.16c)), and ``dtheta/dx`` for Timoshenko, so ``M = +E I dtheta/dx`` (Eq. (2.4b),
    (2.17)). Harris2007_vol2, 9.2.2, Eq. (9.8) prints ``E I d2y/dx2 = -M``, the opposite sign.

    Sign of the shear force (both theories): Q is the force on the cross-section whose outward normal
    is +x, positive along +v (Onate2013_vol2, Fig. 1.2); with ``M = +E I d2v/dx2`` this is
    ``dM/dx = -Q``. The Euler-Bernoulli and the Timoshenko ``shear_force`` return the same sign
    (simply supported beam, 1000 N along +v at one third of the span: +666.67 N left of the load,
    -333.33 N right of it, for both).

    Node assignment: M and V are evaluated at the two ends of each element with that element's own
    ``a_e``; element 0 writes its node ``a`` and every element writes its node ``b``. An interior node
    therefore holds the value at the end of the element on its left; the value at the start of the
    element on its right is not stored. At a point load the stored V is the left-hand value (the jump
    P is not represented); under a distributed load the two element values differ by ``q le``.
    """

    def recover_element_displacements(self, solver: Any, x_nodes: list[float], n: int):
        """Element displacement vectors of both bending planes, for every element.

        Parameters
        ----------
        solver : object
            Exposes ``elements``, ``d_total_xz`` and ``d_total_xy`` (global displacements, DOF order
            ``[u, v, theta]`` per node [mm, mm, rad]).
        x_nodes : list of float
            Node positions [mm] (not used here).
        n : int
            Number of nodes (not used here).

        Returns
        -------
        list of tuple
            ``(elem, a_xz, a_xy)`` per element in ``solver.elements`` order; each ``a_*`` is the length-4
            array ``[v_a, th_a, v_b, th_b]`` [mm, rad, mm, rad].
        """
        out = []
        for elem in solver.elements:
            a_xz = self._element_nodal_displacements(solver, elem, LoadPlane.XZ)
            a_xy = self._element_nodal_displacements(solver, elem, LoadPlane.XY)
            out.append((elem, a_xz, a_xy))
        return out

    def _element_nodal_displacements(self, solver: Any, elem, plane: LoadPlane) -> np.ndarray:
        """Local displacement vector of one element in one bending plane.

        Parameters
        ----------
        solver : object
            Exposes ``d_total_xz`` and ``d_total_xy`` [mm, rad], DOF order ``[u, v, theta]`` per node.
        elem : Elem
            Beam element (nodes ``idx_node_1``, ``idx_node_2``).
        plane : LoadPlane
            ``LoadPlane.XZ`` or ``LoadPlane.XY``.

        Returns
        -------
        ndarray, shape (4,)
            ``a_e = [v_a, th_a, v_b, th_b]`` [mm, rad, mm, rad]; bending only.
        """
        d_total = solver.d_total_xz if plane == LoadPlane.XZ else solver.d_total_xy
        i, j = elem.idx_node_1, elem.idx_node_2
        dof = [3 * i + 1, 3 * i + 2, 3 * j + 1, 3 * j + 2]
        return d_total[dof]

    def bending_moment(self, elem: "Elem", a_e: np.ndarray, x: float) -> tuple[float, float]:
        """Bending moment at position ``x`` of an element.

        Parameters
        ----------
        elem : Elem
            Beam element; ``elem.E`` [MPa] and ``elem.I`` [mm^4] give ``D_b = E I``.
        a_e : ndarray, shape (4,)
            Element displacement vector ``[v_a, th_a, v_b, th_b]`` [mm, rad, mm, rad].
        x : float
            Axial position within the element [mm].

        Returns
        -------
        x : float
            The input position [mm] (returned so that each call is an ``(x, value)`` pair).
        M : float
            Bending moment [N mm], ``M = (E I) B_b(zeta) a_e`` (Onate2013_vol2, Eq. (1.16c)); sign as in
            the class notes.

        Notes
        -----
        ``zeta = elem.natural_coordenates(x)`` is the natural coordinate of ``x``;
        ``elem.bending_strain_matrix(zeta)`` already dispatches to the curvature row of the element
        theory and returns a flat array, so the product with ``a_e`` is a scalar.
        """
        zeta = elem.natural_coordenates(x)
        B_b  = elem.bending_strain_matrix(zeta)
        D_b  = elem.E * elem.I
        M    = float(D_b * (B_b @ a_e))
        return (x, M)

    @abstractmethod
    def shear_force(self, elem: "Elem", a_e: np.ndarray, x: float, *, kGA_override: float | None = None) -> tuple[float, float]:
        """Transverse shear force at position ``x`` of an element (theory specific).

        Parameters
        ----------
        elem : Elem
            Beam element.
        a_e : ndarray, shape (4,)
            Element displacement vector ``[v_a, th_a, v_b, th_b]`` [mm, rad, mm, rad].
        x : float
            Axial position within the element [mm].
        kGA_override : float or None, optional
            Shear rigidity k G A [N] replacing the element value; accepted by every implementation for a
            uniform call, used only by Timoshenko.

        Returns
        -------
        x : float
            The input position [mm].
        Q : float
            Shear force [N].
        """
        raise NotImplementedError

    def recover_internal_forces(self, solver: Any, x_nodes: list[float], n: int):
        """Bending moment and shear force at every mesh node, for both bending planes.

        Parameters
        ----------
        solver : object
            Exposes ``elements``, ``d_total_xz``, ``d_total_xy`` and optionally ``_kGA_override`` [N]
            (structural typing, not a class). The global and sub-model post-processors override this
            method to accept their own solution type.
        x_nodes : list of float
            Node positions [mm].
        n : int
            Number of nodes.

        Returns
        -------
        M_xz, M_xy : ndarray, shape (n,)
            Bending moment of the XZ and XY planes at each node [N mm].
        V_xz, V_xy : ndarray, shape (n,)
            Shear force of the XZ and XY planes at each node [N].

        Notes
        -----
        Node assignment: see the class notes (an interior node holds the value of the element on its left).
        """
        M_xz, V_xz = self._sweep_plane(solver, x_nodes, n, LoadPlane.XZ)
        M_xy, V_xy = self._sweep_plane(solver, x_nodes, n, LoadPlane.XY)
        return M_xz, M_xy, V_xz, V_xy

    def _sweep_plane(self, solver: Any, x_nodes: list[float], n: int, plane: LoadPlane):
        """Moment and shear force at every mesh node for one bending plane.

        Parameters
        ----------
        solver : object
            As in ``recover_internal_forces``; ``_kGA_override`` is read once and passed to every
            ``shear_force`` call (ignored by Euler-Bernoulli).
        x_nodes : list of float
            Node positions [mm] (not used here).
        n : int
            Number of nodes.
        plane : LoadPlane
            ``LoadPlane.XZ`` or ``LoadPlane.XY``.

        Returns
        -------
        M, V : ndarray, shape (n,)
            Bending moment [N mm] and shear force [N] at each node.
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
    """Euler-Bernoulli shear recovery: closed-form shear of the cubic Hermite element.

    Notes
    -----
        Q = -(E I / l_e^3) * ([12, 6 l_e, -12, 6 l_e] . a_e)

    with ``E I`` = ``elem.E * elem.I``, ``l_e`` = ``elem.length`` and ``a_e = [v_a, th_a, v_b, th_b]``.
    This is ``Q = -E I d3v/dx3`` (Onate2013_vol2, 1.3.4, Eq. (1.24)); the third derivative of a cubic
    is constant, so Q is constant within the element.

    The last member of Onate2013_vol2, Eq. (1.24) is printed without the minus sign; since
    ``[12, 6 l_e, -12, 6 l_e] . a_e / l_e^3 = +d3v/dx3`` for the Hermite element, that member has a
    sign slip. The code follows ``Q = -E I d3v/dx3`` (first two members; consistent with Example 1.2
    of the book). Sign of Q: see the base-class notes (same sign as the Timoshenko ``shear_force``).
    """

    def shear_force(self, elem: "Elem", a_e: np.ndarray, x: float, *, kGA_override: float | None = None) -> tuple[float, float]:
        """Shear force of the Euler-Bernoulli element (constant along the element).

        Parameters
        ----------
        elem : Elem
            Beam element (``E`` [MPa], ``I`` [mm^4], ``length`` [mm]).
        a_e : ndarray, shape (4,)
            ``[v_a, th_a, v_b, th_b]`` [mm, rad, mm, rad].
        x : float
            Position [mm]; returned unchanged, Q does not depend on it.
        kGA_override : float or None, optional
            Accepted for call-shape parity with the Timoshenko class; has no meaning here.

        Returns
        -------
        x : float
            The input position [mm].
        Q : float
            Shear force [N]; see the class notes.
        """
        E  = elem.E
        I  = elem.I
        le = elem.length
        Q  = -float(E * I / (le ** 3) * ([12, 6 * le, -12, 6 * le] @ a_e))
        return (x, Q)


class TimoshenkoPostProcessing(ElementTheoryPostProcessor):
    """Timoshenko shear recovery: shear force from the shear strain of the element.

    Warnings
    --------
    The bending moment of this element is constant along the element (Onate2013_vol2, 2.3.2), so the
    value written at a node is the element-constant value of the element on its left: it is off by
    ``V le/2`` from the moment at the node, so the stored M at the supports is not zero and ``M_max``
    is underestimated on a coarse mesh. With ``integration_method="exact"`` the shear strain varies
    linearly along the element, so the end value stored at the node is not a usable shear force (the
    element mid-point value is; the book samples Q there: Onate2013_vol2, Eq. (2.17) and
    section 2.4). The fully integrated element is also over-stiff in shear on a coarse mesh (shear
    locking, Onate2013_vol2, 2.4), so the rotations, the deflections and the stored M are too
    small; they converge only with mesh refinement.

    Notes
    -----
    .. math::

        Q = D_s B_s(zeta) a_e = k G A (dv/dx - theta)

    with ``B_s = elem.shear_strain_matrix(zeta)`` (``B_s @ a_e`` is the shear strain gamma
    [-]) and ``D_s = elem.shear_rigidity(kGA_override=...)`` = k G A [N], correction factor k
    included (Onate2013_vol2, Eq. (2.4b), (2.6c), (2.17)). The book takes Q at the element mid-point;
    here Q is evaluated at the requested position, and for the exact integration variant
    ``B_s`` depends on ``zeta``, so ``x`` is genuinely used. Sign of Q: see the base-class notes (same
    sign as the Euler-Bernoulli ``shear_force``).
    """

    def shear_force(self, elem: "Elem", a_e: np.ndarray, x: float, *, kGA_override: float | None = None) -> tuple[float, float]:
        """Shear force of the Timoshenko element at position ``x``.

        Parameters
        ----------
        elem : Elem
            Beam element.
        a_e : ndarray, shape (4,)
            ``[v_a, th_a, v_b, th_b]`` [mm, rad, mm, rad].
        x : float
            Position within the element [mm].
        kGA_override : float or None, optional
            Shear rigidity k G A [N] replacing the element value.

        Returns
        -------
        x : float
            The input position [mm].
        Q : float
            Shear force [N], ``Q = k G A gamma``; see the class notes.
        """
        zeta = elem.natural_coordenates(x)
        B_s  = elem.shear_strain_matrix(zeta)
        D_s  = elem.shear_rigidity(kGA_override=kGA_override)
        Q    = float(D_s * (B_s @ a_e))
        return (x, Q)

    def shear_strain(self, elem: "Elem", a_e: np.ndarray, x: float) -> float:
        """Shear strain gamma = dv/dx - theta of the Timoshenko element at position ``x``.

        Specific to this theory; not part of the shared contract.

        Parameters
        ----------
        elem : Elem
            Beam element.
        a_e : ndarray, shape (4,)
            ``[v_a, th_a, v_b, th_b]`` [mm, rad, mm, rad].
        x : float
            Position within the element [mm].

        Returns
        -------
        float
            Shear strain ``B_s(zeta) @ a_e`` [-].
        """
        zeta = elem.natural_coordenates(x)
        B_s  = elem.shear_strain_matrix(zeta)
        return B_s @ a_e