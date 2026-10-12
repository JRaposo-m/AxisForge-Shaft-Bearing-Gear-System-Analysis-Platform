"""
Beam elements of the shaft FEM model: the Euler-Bernoulli and Timoshenko 2-node bending formulations, the axial bar, and Elem, which builds the element list from a Mesh1D. The shear correction factor is in shear_factor.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from axisforge.core.materials import get_material
from axisforge.config import NODE_LOOKUP_TOL_MM
from axisforge.mesh.shaft.beam_model_settings import (
    BeamModelSettings, VALID_BEAM_THEORIES, VALID_SHEAR_THEORIES,
    VALID_INTEGRATION_METHODS,
)
import axisforge.mesh.shaft.element_type.shear_factor as sf

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
        ShaftSystem,
    )
    from mesh.shaft.mesh_generation.mesh_1D import Mesh1D


# =====================================================================
# ---- registration ----------------------------------------------------
# =====================================================================

_FORMULATION_REGISTRY: dict[tuple[str, str], "BeamFormulation"] = {}


def register_formulation(beam_theory: str, element_order: str = "linear"):
    """Class decorator that registers a beam formulation under (beam_theory, element_order).

    The decorated BeamFormulation is instantiated once; it is stateless (all
    physical data live on the Elem passed to each call). ElemBase resolves
    the tuple, so no other code needs the mapping.

    Parameters
    ----------
    beam_theory : str
        Beam theory name, e.g. ``"euler_bernoulli"`` or ``"timoshenko"`` [-].
    element_order : str, optional
        Element order, ``"linear"`` (2 nodes) by default [-].

    Returns
    -------
    callable
        Decorator returning the class unchanged.
    """
    def _decorator(cls: type["BeamFormulation"]) -> type["BeamFormulation"]:
        """Register one instance of ``cls`` under (beam_theory, element_order) and return ``cls``."""
        _FORMULATION_REGISTRY[(beam_theory, element_order)] = cls()
        return cls
    return _decorator


def _linear_shape_functions_2node(zeta: float) -> np.ndarray:
    """Linear 2-node shape functions N_1 = 1/2 (1 - zeta), N_2 = 1/2 (1 + zeta) [-].

    Shared by TimoshenkoBeam (interpolates v and theta separately) and
    FrameElement (interpolates u): same interpolation, different field.

    Parameters
    ----------
    zeta : float
        Natural coordinate in [-1, 1] [-].
    """
    return np.array([0.5 * (1.0 - zeta), 0.5 * (1.0 + zeta)])


# =====================================================================
# ---- contract: BeamFormulation ---------------------------------------
# =====================================================================

class BeamFormulation(ABC):
    """Contract every bending theory must satisfy (shape functions, strain matrix, stiffness).

    Concrete subclasses are stateless (see register_formulation(), which
    stores singletons); all physical data come from the ``elem`` argument.
    The local bending DOF order of the 2-node element is
    ``[v_a, theta_a, v_b, theta_b]`` (transverse displacement [mm] and
    rotation [rad] at the two nodes).
    """

    @abstractmethod
    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        """Shape functions at natural coordinate ``zeta``.

        The shape depends on the theory: Euler-Bernoulli returns 4 entries
        over ``[v_a, theta_a, v_b, theta_b]`` (``v = N . a``); Timoshenko
        returns the 2 nodal functions ``[N_1, N_2]``, applied separately to
        ``v`` and to ``theta``.
        """
        ...

    @abstractmethod
    def bending_strain_matrix(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        """Matrix B_b that maps the local DOFs to the bending strain at ``zeta``."""
        ...

    @abstractmethod
    def stiffness_element(self, elem: "ElemBase", **kwargs) -> np.ndarray:
        """Local 4x4 stiffness matrix of the bending plane of ``elem`` over ``[v_a, theta_a, v_b, theta_b]`` (bending, plus shear for Timoshenko)."""
        ...

    def natural_coordenates(self, x: float, elem: "ElemBase") -> float:
        """Natural coordinate zeta of the global axial position ``x``.

        Same for every formulation that spans [x_a, x_b], hence defined here:
        ``zeta = 2 (x - x_c) / length``, ``x_c = (x_a + x_b) / 2``.

        Parameters
        ----------
        x : float
            Axial coordinate [mm].
        elem : ElemBase
            Element providing ``x_a``, ``x_b`` and ``length`` [mm].

        Returns
        -------
        float
            zeta, in [-1, 1] inside the element [-].

        References
        ----------
        .. [1] Onate2013_vol2, 1.3.1, Eq. (1.11b).
        """
        x_c = (elem.x_a + elem.x_b) / 2.0
        return 2.0 * (x - x_c) / elem.length


class ShearDeformableBeamFormulation(BeamFormulation):
    """
    Specialization for theories with an independent transverse-shear
    field (currently: the linear Timoshenko element). A subclass, not a second independent ABC mixed in --
    shear deformation is what makes a theory Timoshenko rather than
    Euler-Bernoulli, not an optional add-on. Elem checks
    isinstance(formulation, ShearDeformableBeamFormulation) instead of
    comparing beam_theory strings.
    """

    @abstractmethod
    def shear_rigidity(self, elem: "ElemBase", *, kGA_override: float | None = None) -> float:
        """Transverse shear rigidity D_s = k G A of ``elem`` [N]."""
        ...

    @abstractmethod
    def shear_strain_matrix(self, zeta: float, elem: "ElemBase", integration: str) -> np.ndarray:
        """Matrix B_s that maps the local DOFs to the shear strain gamma at ``zeta``."""
        ...


# =====================================================================
# ---- concrete: Euler-Bernoulli -----------------------------------------
# =====================================================================

@register_formulation("euler_bernoulli")
class EulerBernoulliBeam(BeamFormulation):
    """Euler-Bernoulli 2-node bending element with cubic Hermite interpolation.

    No shear deformation: the rotation is the slope of the beam axis,
    ``theta = dv/dx`` (positive anticlockwise, seen with x to the right and
    v upward), and the bending stiffness is
    the closed-form matrix in ``D_b = E I``. Used for one bending plane at a
    time; see the Warnings below.

    Warnings
    --------
    Differs from Onate2013_vol2 Eq. (1.10)-(1.11): the book keeps the factor
    l/2 outside the rotation functions; here it is folded in
    (``le/8 (...) = (le/2) * 1/4 (...)``), so the rotation shape functions
    have units of [mm].

    Each bending plane (XY and XZ) is solved as an independent 2-D problem
    with this same element and its own in-plane rotation
    ``theta = dv/dx``. In the 3-D elements of Onate2013_vol2 Box 4.3 and
    Cao2004 Appendix A the deflection-rotation coupling terms have opposite
    signs in the two planes (``+6 D l`` on (v, theta_z), ``-6 D l`` on
    (w, theta_y); right-handed rotations). Here the same matrix is used in
    both planes, so the in-plane rotation is the right-handed rotation about
    +Z in XY and minus the right-handed rotation about +Y in XZ. Neither this
    module nor the README Conventions states which sign the solver uses for
    XZ rotations and moments (review item).

    Notes
    -----
    Local DOFs ``[v_a, theta_a, v_b, theta_b]`` (``v`` [mm], ``theta``
    [rad]). Natural coordinate ``zeta = 2 (x - x_c) / le``, ``le`` the element
    length [mm]. Code names ``le``, ``E``, ``I`` map to ``l(e)``, ``E``,
    ``I_y`` of the reference.

    Assumptions: small deflections, isotropic homogeneous material, constant
    ``E I`` along the element.

    References
    ----------
    .. [1] Onate2013_vol2, 1.2.2, Eq. (1.2) (rotation theta = dw/dx).
    .. [2] Onate2013_vol2, 1.3.1, Eq. (1.10)-(1.11) (Hermite functions) and
       Eq. (1.16a) (B_b).
    .. [3] Onate2013_vol2, 1.3.2, Eq. (1.20) (stiffness matrix).
    """

    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        """Hermite shape functions at ``zeta``.

        Returns
        -------
        numpy.ndarray, shape (4,)
            ``[N_va, N_tha, N_vb, N_thb]`` so that ``v = N . [v_a, theta_a,
            v_b, theta_b]``; entries multiplying ``v`` are [-], those
            multiplying ``theta`` are [mm].
        """
        le = elem.length
        N = np.zeros(4)
        N[0] = 1/4 * (2 - 3*zeta + zeta**3)            # transverse v
        N[1] = le/8 * (1 - zeta - zeta**2 + zeta**3)    # rotation theta
        N[2] = 1/4 * (2 + 3*zeta - zeta**3)            # transverse v
        N[3] = le/8 * (-1 - zeta + zeta**2 + zeta**3)   # rotation theta
        return N

    def bending_strain_matrix(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        """Curvature matrix B_b at ``zeta``, with ``kappa = d2v/dx2 = B_b . a``.

        Returns
        -------
        numpy.ndarray, shape (4,)
            Row over ``[v_a, theta_a, v_b, theta_b]``; entries on ``v`` are
            [1/mm^2], on ``theta`` [1/mm]. Matches Onate2013_vol2
            Eq. (1.16a).
        """
        le = elem.length
        B = np.zeros(4)
        B[0] = 6 * zeta / le**2
        B[1] = (-1 + 3 * zeta) / le
        B[2] = -6 * zeta / le**2
        B[3] = (1 + 3 * zeta) / le
        return B

    def stiffness_element(self, elem: "ElemBase") -> np.ndarray:
        """Local bending stiffness matrix E I / le^3 [[12, 6le, -12, 6le], ...] (closed form).

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Symmetric matrix over ``[v_a, theta_a, v_b, theta_b]``: v-v
            entries [N/mm], v-theta [N], theta-theta [N mm/rad]. Same entries
            as Onate2013_vol2 Eq. (1.20).
        """
        le = elem.length
        c = elem.E * elem.I / le**3
        k = np.zeros((4, 4))
        k[0, 0] = k[2, 2] = 12 * c
        k[0, 1] = k[0, 3] = k[1, 0] = k[3, 0] = 6 * c * le
        k[1, 1] = k[3, 3] = 4 * c * le**2
        k[0, 2] = k[2, 0] = -12 * c
        k[1, 2] = k[2, 1] = k[2, 3] = k[3, 2] = -6 * c * le
        k[1, 3] = k[3, 1] = 2 * c * le**2
        return k


# =====================================================================
# ---- concrete: Timoshenko (2-node, linear) ------------------------------
# =====================================================================

@register_formulation("timoshenko")  # -> ("timoshenko", "linear")
class TimoshenkoBeam(ShearDeformableBeamFormulation):
    """Timoshenko 2-node linear bending element with selectable shear integration.

    Shear deformation is included: displacement ``v`` and rotation ``theta``
    are interpolated independently with the linear functions
    ``N = [(1 - zeta)/2, (1 + zeta)/2]``, and the shear strain is
    ``gamma = dv/dx - theta`` (rotation positive anticlockwise, seen with x
    to the right and v upward). The
    stiffness is the sum of a bending part (exact, one point) and a shear
    part integrated with ``"single_point"`` (reduced, one Gauss point at the
    element centre, against shear locking) or ``"exact"`` (two points).

    Warnings
    --------
    ``"exact"`` means exact integration of K_s, not an exact element: it is
    too stiff at every element length, not only for slender elements
    (Onate2013_vol2, 2.4). For a cantilever with an end load and equal
    elements the tip deflection is ``beta / (1 + beta)`` times the
    closed-form Timoshenko value, with ``beta = 12 E I / (k G A le^2)``
    (Onate2013_vol2, Eq. (2.32), (2.34a)); for a solid steel section ``beta``
    is about ``2.2 (d/le)^2``: 35 % at le = 2 d, 90 % at le = d/2.
    ``"single_point"`` does not lock but is not exact at the nodes: for a
    cantilever with an end load the bending part of the tip deflection is
    75 %, 93.8 %, 98.4 % and 99.6 % of ``P L^3 / (3 E I)`` with 1, 2, 4 and 8
    equal elements (Onate2013_vol2, Table 2.1), whereas the Euler-Bernoulli
    element is exact at the nodes for point loads. Mesh1D does not subdivide
    between mandatory nodes, so the result depends on the refinement nodes.

    Each bending plane (XY and XZ) is solved as an independent 2-D problem
    with this same element and its own in-plane rotation, with
    ``gamma = dv/dx - theta`` in both. Onate2013_vol2 Eq. (4.77) writes
    ``dv/dx - theta_z`` in one plane and ``dw/dx + theta_y`` in the other;
    Cao2004 Eq. (14)-(15), (17), (19) has the same pattern (3-D right-handed
    rotations). With ``theta = dv/dx`` in each plane, the in-plane rotation
    equals the right-handed rotation about +Z in the XY plane and MINUS the
    right-handed rotation about +Y in the XZ plane. Neither this module nor
    the README Conventions states which sign the solver uses for XZ rotations
    and moments (review item).

    The shear factor ``k`` differs from the library sources, see ShearFactor.
    Elem.from_mesh and Elem.from_x_nodes leave ``radius_ratio`` at 0.0, so a
    hollow section uses the solid-section factor.

    Notes
    -----
    Local DOFs ``[v_a, theta_a, v_b, theta_b]`` (``v`` [mm], ``theta``
    [rad]). Code names map to the source as ``D_b = E I`` (bending rigidity
    [N mm^2]) and ``D_s = k G A`` (shear rigidity [N]), with
    ``G = E / (2 (1 + nu))`` and ``k`` from ShearFactor.

    ``K = K_b + K_s``; ``K_b = (D_b/le) [[0,0,0,0],[0,1,0,-1],[0,0,0,0],
    [0,-1,0,1]]``; ``K_s`` has v-v entries ``D_s/le``, v-theta entries
    ``+-D_s/2`` and theta-theta entries ``D_s le/4`` (``"single_point"``) or
    ``D_s le/3`` and ``D_s le/6`` (``"exact"``).

    Not in the code: the exact 2-node Timoshenko element with the shear
    parameter ``12 E I / (k G A le^2)`` (beta in Onate2013_vol2 Eq. (2.101c);
    Phi in Cao2004, Appendix A, which uses that element), a mass matrix, and
    a quadratic (3-node) element.

    References
    ----------
    .. [1] Onate2013_vol2, 2.2.1-2.2.2, Eq. (2.1)-(2.2) (kinematics, gamma)
       and 2.2.3, Eq. (2.5b), (2.6b-c) (D_b = E I, D_s = k G A).
    .. [2] Onate2013_vol2, 2.3.1, Eq. (2.10) (linear N) and 2.3.2,
       Eq. (2.15) (B_b, B_s).
    .. [3] Onate2013_vol2, 2.4, Eq. (2.27a) (K_b), Eq. (2.27b) (K_s, exact),
       Eq. (2.38) (K_s, one-point reduced integration); shear locking and
       reduced integration: same section; constant shear strain matrix:
       2.4.1, Eq. (2.44)-(2.45).
    """

    _shear_factor = sf.ShearFactor()

    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        """Linear shape functions ``[(1 - zeta)/2, (1 + zeta)/2]`` [-], two entries (one per node; v and theta share them)."""
        return _linear_shape_functions_2node(zeta)

    def bending_strain_matrix(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        """Bending strain matrix B_b = [0, -1/le, 0, 1/le], with ``kappa = d theta/dx = B_b . a``.

        Returns
        -------
        numpy.ndarray, shape (4,)
            Row over ``[v_a, theta_a, v_b, theta_b]``; entries on ``theta``
            are [1/mm], on ``v`` zero. Onate2013_vol2 Eq. (2.15).
        """
        le = elem.length
        return np.array([0.0, -1/le, 0.0, 1/le])

    def shear_strain_matrix(self, zeta: float, elem: "ElemBase", integration: str) -> np.ndarray:
        """Shear strain matrix B_s, with ``gamma = dv/dx - theta = B_s . a``.

        Parameters
        ----------
        zeta : float
            Natural coordinate [-]; used only by ``"exact"``.
        elem : ElemBase
            Element providing ``length`` [mm].
        integration : str
            ``"single_point"``: ``[-1/le, -1/2, 1/le, -1/2]`` (value at the
            element centre, Onate2013_vol2 Eq. (2.45)); ``"exact"``:
            ``[-1/le, -(1 - zeta)/2, 1/le, -(1 + zeta)/2]`` (Eq. (2.15)).

        Returns
        -------
        numpy.ndarray, shape (4,)
            Row over ``[v_a, theta_a, v_b, theta_b]``; entries on ``v`` are
            [1/mm], on ``theta`` [-].

        Raises
        ------
        ValueError
            For any other ``integration``.
        """
        le = elem.length
        if integration == "single_point":
            return np.array([-1/le, -0.5, 1/le, -0.5])
        elif integration == "exact":
            return np.array([-1/le, -(1 - zeta)/2, 1/le, -(1 + zeta)/2])
        raise ValueError(
            f"TimoshenkoBeam: unknown integration method '{integration}'. "
            f"Expected 'single_point' or 'exact'."
        )

    def shear_rigidity(self, elem: "ElemBase", *, kGA_override: float | None = None) -> float:
        """Transverse shear rigidity D_s = k G A, with ``G = E / (2 (1 + nu))``.

        Parameters
        ----------
        elem : ElemBase
            Element providing ``E`` [MPa], ``v`` [-], ``A`` [mm^2],
            ``radius_ratio`` [-] and ``shear_theory``.
        kGA_override : float or None, optional
            If given, the effective ``k`` is back-computed so that the result
            equals this value [N] (see ShearFactor.shear_correction_factor).

        Returns
        -------
        float
            D_s [N]. Onate2013_vol2 Eq. (2.6b-c).
        """
        G = elem.E / (2 * (1 + elem.v))
        k = self._shear_factor.shear_correction_factor(
            elem.v, elem.radius_ratio, elem.E, elem.A,
            elem.shear_theory, kGA_override=kGA_override,
        )
        return k * G * elem.A

    def stiffness_element(self, elem: "ElemBase", *, integration: str = "single_point",
                           kGA_override: float | None = None) -> np.ndarray:
        """Local 4x4 stiffness matrix ``K_b + K_s`` of the Timoshenko element.

        Parameters
        ----------
        elem : ElemBase
            Element providing ``length`` [mm], ``E`` [MPa], ``I`` [mm^4] and
            what :meth:`shear_rigidity` needs.
        integration : str, optional
            Shear integration: ``"single_point"`` (reduced) or ``"exact"``.
        kGA_override : float or None, optional
            Shear rigidity ``k G A`` imposed instead of the computed one [N].

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Symmetric matrix over ``[v_a, theta_a, v_b, theta_b]``: v-v
            entries [N/mm], v-theta [N], theta-theta [N mm/rad].
            Onate2013_vol2 Eq. (2.27a) + (2.38) (or (2.27b) for ``"exact"``).

        Raises
        ------
        ValueError
            If ``integration`` is not ``"single_point"`` or ``"exact"``.
        """
        le  = elem.length
        D_b = elem.E * elem.I
        D_s = self.shear_rigidity(elem, kGA_override=kGA_override)

        k_b = np.zeros((4, 4))
        k_b[1, 1] = k_b[3, 3] = D_b / le
        k_b[1, 3] = k_b[3, 1] = -D_b / le

        k_s = np.zeros((4, 4))
        k_s[0, 0] = k_s[2, 2] = D_s / le
        k_s[0, 2] = k_s[2, 0] = -D_s / le
        k_s[0, 1] = k_s[1, 0] = k_s[0, 3] = k_s[3, 0] = 0.5 * D_s
        k_s[1, 2] = k_s[2, 1] = k_s[2, 3] = k_s[3, 2] = -0.5 * D_s

        if integration == "single_point":
            k_s[1, 1] = k_s[3, 3] = k_s[1, 3] = k_s[3, 1] = 0.25 * D_s * le
        elif integration == "exact":
            k_s[1, 1] = k_s[3, 3] = D_s * le / 3.0
            k_s[1, 3] = k_s[3, 1] = D_s * le / 6.0
        else:
            raise ValueError(
                f"TimoshenkoBeam: unknown integration method '{integration}'. "
                f"Expected 'single_point' or 'exact'."
            )
        return k_b + k_s


# =====================================================================
# ---- Timoshenko (3-node, quadratic): not implemented -------------------
# =====================================================================
#
# No quadratic beam formulation exists in this file or in
# _FORMULATION_REGISTRY; QuadraticTimoshenkoElem below is only a placeholder.


# =====================================================================
# ---- axial ("frame") behaviour -- NOT part of BeamFormulation --------
# =====================================================================

class FrameElement:
    """Axial bar behaviour (DOFs u_a, u_b) shared by every element, whatever the beam theory.

    Linear 2-node bar: ``B = [-1/le, 1/le]`` and ``K = (E A / le) [[1, -1],
    [-1, 1]]``. It is not a BeamFormulation and is not registered in
    _FORMULATION_REGISTRY: there is no beam-theory-dependent variant to
    select, this is unconditional composition.

    Notes
    -----
    Local DOFs ``[u_a, u_b]`` (axial displacement [mm]); stiffness entries
    [N/mm]. Torsion is not handled here (separate statics in the solver).

    The class reads only ``length``, ``E`` and ``A``, so it would also run
    from a 3-node element, assuming axial behaviour stays 2-DOF (end nodes
    only); that assumption has not been reviewed (see
    QuadraticTimoshenkoElem).

    References
    ----------
    .. [1] Onate2013_vol2, 4.4.1, Eq. (4.78) (axial row of B) and Box 4.3
       with Eq. (4.37a) (D_a = E A).
    """

    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        """Linear axial shape functions at ``zeta``, two entries [-]."""
        return _linear_shape_functions_2node(zeta)

    def axial_strain_matrix(self, elem: "ElemBase") -> np.ndarray:
        """Axial strain matrix ``[-1/le, 1/le]``, with ``du/dx = B . [u_a, u_b]`` [1/mm]."""
        le = elem.length
        return np.array([-1.0 / le, 1.0 / le])

    def stiffness_element(self, elem: "ElemBase") -> np.ndarray:
        """Axial stiffness ``(E A / le) [[1, -1], [-1, 1]]`` over ``[u_a, u_b]`` [N/mm]."""
        k = elem.E * elem.A / elem.length
        return np.array([[k, -k], [-k, k]])


# =====================================================================
# ---- ElemBase: node-count-agnostic contract --------------------------
# =====================================================================

class ElemBase(ABC):
    """
    Abstract contract for a 1D element between mesh nodes, independent
    of how many nodes it has.

    Every concrete subclass's __init__ must set: length, E, I, A, v,
    x_a, x_b, beam_theory, shear_theory, integration_method,
    element_order, radius_ratio. These annotations document that
    contract; they are not enforced by Python on their own -- validate()
    (abstract here, required per subclass) is where each subclass
    actually checks its own data, since what needs checking differs by
    topology (e.g. 2 vs 3 node indices).

    Formulation dispatch (shape_functions, bending_strain_matrix,
    stiffness_element, shear_rigidity, shear_strain_matrix,
    natural_coordenates) lives here, concrete and shared, because none
    of it reads node indices directly -- it only reads the scalar
    physical/settings attributes above. Node bookkeeping is the only
    thing that genuinely varies by subclass, so that's the only part
    left abstract.

    Attributes
    ----------
    length : float
        Element length ``x_b - x_a`` [mm].
    E : float
        Young modulus [MPa].
    I : float
        Second moment of area [mm^4].
    A : float
        Cross-section area [mm^2].
    v : float
        Poisson ratio [-].
    x_a, x_b : float
        Axial coordinates of the end nodes [mm].
    beam_theory : str
        ``"euler_bernoulli"`` or ``"timoshenko"``.
    shear_theory : str or None
        ``"cowper"`` or ``"hutchinson"`` for Timoshenko, else None.
    integration_method : str or None
        ``"single_point"`` or ``"exact"`` for Timoshenko, else None.
    element_order : str
        ``"linear"`` (2 nodes); selects the formulation with ``beam_theory``.
    radius_ratio : float
        Inner radius / outer radius of the section [-]; 0.0 for solid.
    """

    length: float
    E: float
    I: float
    A: float
    v: float
    x_a: float
    x_b: float
    beam_theory: str
    shear_theory: str | None
    integration_method: str | None
    element_order: str
    radius_ratio: float

    _frame = FrameElement()  # stateless singleton, shared by every subclass

    # ------------------------------------------------------------------
    # Node bookkeeping -- the one thing that differs by node count
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def node_indices(self) -> tuple[int, ...]:
        """Mesh node indices this element connects, in local order [-]."""
        ...

    @abstractmethod
    def validate(self) -> list[str]:
        """Data errors of this element (empty list if none)."""
        ...

    def validate_or_raise(self) -> None:
        """Raise ValueError with every error returned by :meth:`validate`."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Formulation dispatch -- identical regardless of node count
    # ------------------------------------------------------------------

    def _formulation(self) -> BeamFormulation:
        """Registered formulation for ``(beam_theory, element_order)``; ValueError if none."""
        key = (self.beam_theory, self.element_order)
        if key not in _FORMULATION_REGISTRY:
            raise ValueError(
                f"{type(self).__name__}: no formulation registered for "
                f"(beam_theory='{self.beam_theory}', "
                f"element_order='{self.element_order}'). "
                f"Registered: {sorted(_FORMULATION_REGISTRY)}"
            )
        return _FORMULATION_REGISTRY[key]

    def shape_functions(self, zeta: float) -> np.ndarray:
        """Bending shape functions at natural coordinate ``zeta``.

        The shape depends on the theory: Euler-Bernoulli returns 4 entries
        over ``[v_a, theta_a, v_b, theta_b]`` (``v = N . a``); Timoshenko
        returns the 2 nodal functions ``[N_1, N_2]``, applied separately to
        ``v`` and to ``theta``.
        """
        return self._formulation().shape_functions(zeta, self)

    def bending_strain_matrix(self, zeta: float) -> np.ndarray:
        """Bending strain matrix B_b at ``zeta`` (see the formulation for units)."""
        return self._formulation().bending_strain_matrix(zeta, self)

    def stiffness_element(self, *, kGA_override: float | None = None) -> np.ndarray:
        """Local 4x4 stiffness matrix of the bending plane over ``[v_a, theta_a, v_b, theta_b]`` (bending, plus shear for Timoshenko).

        Parameters
        ----------
        kGA_override : float or None, optional
            Shear rigidity ``k G A`` [N] imposed instead of the computed one;
            used by shear-deformable theories only.

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Entries: v-v [N/mm], v-theta [N], theta-theta [N mm/rad].
        """
        formulation = self._formulation()
        if isinstance(formulation, ShearDeformableBeamFormulation):
            return formulation.stiffness_element(
                self, integration=self.integration_method, kGA_override=kGA_override,
            )
        return formulation.stiffness_element(self)

    def shear_rigidity(self, *, kGA_override: float | None = None) -> float:
        """Transverse shear rigidity D_s = k G A [N]; ValueError for Euler-Bernoulli."""
        formulation = self._formulation()
        if not isinstance(formulation, ShearDeformableBeamFormulation):
            raise ValueError(
                f"{type(self).__name__}: shear_rigidity() only applies to "
                f"shear-deformable theories, got '{self.beam_theory}'"
            )
        return formulation.shear_rigidity(self, kGA_override=kGA_override)

    def shear_strain_matrix(self, zeta: float) -> np.ndarray:
        """Shear strain matrix B_s at ``zeta``; ValueError for Euler-Bernoulli."""
        formulation = self._formulation()
        if not isinstance(formulation, ShearDeformableBeamFormulation):
            raise ValueError(
                f"{type(self).__name__}: shear_strain_matrix() only applies "
                f"to shear-deformable theories, got '{self.beam_theory}'"
            )
        return formulation.shear_strain_matrix(zeta, self, integration=self.integration_method)

    def natural_coordenates(self, x: float) -> float:
        """Natural coordinate zeta in [-1, 1] of the axial position ``x`` [mm] [-]."""
        return self._formulation().natural_coordenates(x, self)

    # ------------------------------------------------------------------
    # Axial ("frame") behaviour -- unconditional, no dispatch
    # ------------------------------------------------------------------

    def axial_shape_functions(self, zeta: float) -> np.ndarray:
        """Axial shape functions at ``zeta`` [-]."""
        return self._frame.shape_functions(zeta, self)

    def axial_strain_matrix(self) -> np.ndarray:
        """Axial strain matrix ``[-1/le, 1/le]`` [1/mm]."""
        return self._frame.axial_strain_matrix(self)

    def axial_stiffness_element(self) -> np.ndarray:
        """Axial stiffness ``(E A / le) [[1, -1], [-1, 1]]`` over ``[u_a, u_b]`` [N/mm]."""
        return self._frame.stiffness_element(self)


# =====================================================================
# ---- Elem: the 2-node element used throughout AxisForge today --------
# =====================================================================

class Elem(ElemBase):
    """2-node beam element between two mesh nodes (Euler-Bernoulli or linear Timoshenko).

    Holds the section and material data of one element of the shaft mesh and
    delegates shape functions, strain matrices and stiffness to the
    formulation registered for ``(beam_theory, element_order)``. The
    element has no plane of its own: the two bending planes are solved
    independently with the same matrices (README Conventions, mesh README).

    Parameters
    ----------
    length : float
        Element length [mm].
    E : float
        Young modulus [MPa].
    I : float
        Second moment of area [mm^4].
    A : float
        Cross-section area [mm^2].
    v : float
        Poisson ratio [-].
    idx_node_1, idx_node_2 : int
        Mesh indices of the end nodes [-].
    x_a, x_b : float
        Axial coordinates of the end nodes [mm].
    settings : BeamModelSettings
        Beam theory, shear theory and integration method.
    radius_ratio : float, optional
        Inner radius / outer radius of the section [-]; 0.0 for a solid
        section.

    Raises
    ------
    ValueError
        If no formulation is registered for the settings.

    Warnings
    --------
    The factories :meth:`from_mesh` and :meth:`from_x_nodes` never pass
    ``radius_ratio`` (it stays 0.0): a hollow shaft section gets the tube
    ``A`` and ``I`` but the solid-section shear factor (review item).
    """

    VALID_BEAM_THEORIES       = VALID_BEAM_THEORIES
    VALID_SHEAR_THEORIES      = VALID_SHEAR_THEORIES
    VALID_INTEGRATION_METHODS = VALID_INTEGRATION_METHODS

    def __init__(self, length: float, E: float, I: float, A: float,
                 v: float, idx_node_1: int, idx_node_2: int,
                 x_a: float, x_b: float,
                 settings: BeamModelSettings,
                 radius_ratio: float = 0.0):
        """Store the section, material and node data; see the class docstring."""
        self.length             = length
        self.E                  = E
        self.I                  = I
        self.A                  = A
        self.v                  = v
        self.idx_node_1         = idx_node_1
        self.idx_node_2         = idx_node_2
        self.x_a                = x_a
        self.x_b                = x_b
        self.beam_theory        = settings.beam_theory
        self.shear_theory       = settings.shear_theory
        self.integration_method = settings.integration_method
        self.radius_ratio       = radius_ratio  # inner/outer radius, 0.0 for solid

        # element_order is not a BeamModelSettings field; "linear" is the only
        # order that is wired in.
        self.element_order = getattr(settings, "element_order", "linear")

        if (self.beam_theory, self.element_order) not in _FORMULATION_REGISTRY:
            raise ValueError(
                f"Elem: no formulation registered for "
                f"(beam_theory='{self.beam_theory}', "
                f"element_order='{self.element_order}'). "
                f"Registered: {sorted(_FORMULATION_REGISTRY)}"
            )

    @property
    def node_indices(self) -> tuple[int, int]:
        """Mesh indices ``(idx_node_1, idx_node_2)`` of the end nodes [-]."""
        return (self.idx_node_1, self.idx_node_2)

    # ------------------------------------------------------------------
    # Factory -- builds the full element list directly from a Mesh1D
    # ------------------------------------------------------------------

    @staticmethod
    def find_node_index(x_nodes: list[float], x: float, tol: float = NODE_LOOKUP_TOL_MM) -> int:
        """Index of the node nearest to ``x``, if within ``tol``.

        Parameters
        ----------
        x_nodes : list of float
            Node positions [mm].
        x : float
            Requested position [mm].
        tol : float, optional
            Maximum accepted distance [mm]. Defaults to
            ``NODE_LOOKUP_TOL_MM`` (same point), not to the merge distance
            ``MESH_MIN_NODE_DIST_MM``: every load, support and section
            boundary is a mandatory node, so it must exist exactly.

        Returns
        -------
        int
            Index of the nearest node.

        Raises
        ------
        ValueError
            If no node lies within ``tol`` of ``x``.
        """
        if not x_nodes:
            raise ValueError(f"No node found at x={x:.6g} mm: empty node list")
        i = min(range(len(x_nodes)), key=lambda k: abs(x_nodes[k] - x))
        if abs(x_nodes[i] - x) > tol:
            raise ValueError(
                f"No node found at x={x:.6g} mm within tolerance {tol} mm "
                f"(nearest node at {x_nodes[i]:.6g} mm)"
            )
        return i

    @classmethod
    def from_mesh(cls, mesh: "Mesh1D", settings: BeamModelSettings,
                  node_tol: float = NODE_LOOKUP_TOL_MM) -> list["Elem"]:
        """Build the element list of the whole shaft from a Mesh1D.

        One element per interval between consecutive mesh nodes; section
        properties and material are those of the section that contains the
        interval. Mesh1D carries no knowledge of Elem: this is the single
        place that bridges mesh positions to element construction.

        Parameters
        ----------
        mesh : Mesh1D
            Provides ``shaft_system`` and ``x_nodes`` [mm].
        settings : BeamModelSettings
            Beam theory, shear theory and integration method for the whole
            shaft; no default on purpose, the caller must decide.
        node_tol : float, optional
            Tolerance for locating section boundaries among the nodes [mm].

        Returns
        -------
        list of Elem
            Elements in ascending axial order, with ``E`` [MPa] and ``v`` from
            the section material, ``I`` [mm^4] and ``A`` [mm^2] from the
            section.

        Raises
        ------
        ValueError
            If a section boundary has no node within ``node_tol``.
        KeyError
            If ``section.material_id`` is not a registered material.

        Warnings
        --------
        ``radius_ratio`` is not passed (stays 0.0): for a hollow section
        (``inner_diameter`` > 0) the element gets the tube ``A`` and ``I`` but
        the solid-section shear factor, and ``validate()`` cannot reject
        ``shear_theory="hutchinson"`` for it. Onate2013_vol2, Fig. 2.3 prints
        a different k for a tube (review item).
        """
        shaft_system = mesh.shaft_system
        x_nodes = mesh.x_nodes
        shaft = shaft_system.shaft

        elements: list[Elem] = []
        for i, section in enumerate(shaft.sections):
            idx_start = cls.find_node_index(x_nodes, shaft.axial_start(i), node_tol)
            idx_end = cls.find_node_index(x_nodes, shaft.axial_end(i), node_tol)

            mat = get_material(section.material_id)

            for j in range(idx_start, idx_end):
                elements.append(cls(
                    length=x_nodes[j + 1] - x_nodes[j],
                    E=mat.E,
                    I=section.second_moment_of_area,
                    A=section.area,
                    v=mat.poisson_ratio,
                    idx_node_1=j,
                    idx_node_2=j + 1,
                    x_a=x_nodes[j],
                    x_b=x_nodes[j + 1],
                    settings=settings,
                    # radius_ratio is not passed: it stays 0.0 (solid shear factor) even for
                    # a hollow section; section.inner_diameter / section.diameter is available.
                ))
        return elements

    @classmethod
    def from_x_nodes(cls, x_nodes: list[float], shaft_system: "ShaftSystem",
                      settings: BeamModelSettings) -> list["Elem"]:
        """Build one element per interval of ``x_nodes``, taking the section at the interval midpoint.

        Parameters
        ----------
        x_nodes : list of float
            Ascending node positions [mm].
        shaft_system : ShaftSystem
            Provides the shaft sections and materials.
        settings : BeamModelSettings
            Beam theory, shear theory and integration method.

        Returns
        -------
        list of Elem
            ``len(x_nodes) - 1`` elements; ``E`` [MPa], ``v`` from the
            section material, ``I`` [mm^4] and ``A`` [mm^2] from the section.

        Raises
        ------
        KeyError
            If ``section.material_id`` is not a registered material.

        Warnings
        --------
        ``radius_ratio`` is not passed (stays 0.0), same consequence as in
        :meth:`from_mesh` for hollow sections (review item).
        """
        shaft = shaft_system.shaft
        elements: list[Elem] = []

        for j in range(len(x_nodes) - 1):
            x_a   = x_nodes[j]
            x_b   = x_nodes[j + 1]
            x_mid = (x_a + x_b) / 2.0

            section, _ = shaft.section_at(x_mid)   # unpack tuple
            mat        = get_material(section.material_id)

            elements.append(cls(
                length     = x_b - x_a,
                E          = mat.E,
                I          = section.second_moment_of_area,
                A          = section.area,
                v          = mat.poisson_ratio,
                idx_node_1 = j,
                idx_node_2 = j + 1,
                x_a        = x_a,
                x_b        = x_b,
                settings   = settings,
            ))

        return elements

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Data errors of this element (empty list if none).

        Returns
        -------
        list of str
            Messages for: ``length`` <= 0 [mm]; ``E`` <= 0 or < 1000 MPa
            (probably given in GPa); ``I`` < 0 (``I`` = 0 is allowed);
            ``A`` <= 0; ``v`` outside (-1, 0.5); equal or negative node
            indices; ``x_b`` <= ``x_a``; no registered formulation; for
            Timoshenko an invalid ``shear_theory`` or ``integration_method``,
            or ``"hutchinson"`` with ``radius_ratio`` != 0; for
            Euler-Bernoulli a ``shear_theory`` or ``integration_method``
            that is not None.
        """
        errors: list[str] = []

        if self.length <= 0:
            errors.append(f"Elem: length must be > 0, got {self.length}")
        if self.E <= 0:
            errors.append(f"Elem: E must be > 0, got {self.E}")
        if self.E < 1000.0:
            errors.append(
                f"Elem: E={self.E} looks like it may be in GPa, not N/mm^2 "
                f"(MPa) — typical structural steel is ~200000 MPa. "
                f"Check material database units."
            )
        if self.I < 0:
            errors.append(f"Elem: I must be >= 0, got {self.I}")
        if self.A <= 0:
            errors.append(f"Elem: A must be > 0, got {self.A}")
        if not (-1.0 < self.v < 0.5):
            errors.append(f"Elem: v out of physical range (-1, 0.5), got {self.v}")
        if self.idx_node_1 == self.idx_node_2:
            errors.append(
                f"Elem: idx_node_1 and idx_node_2 must differ, "
                f"got {self.idx_node_1} == {self.idx_node_2}"
            )
        if self.idx_node_1 < 0 or self.idx_node_2 < 0:
            errors.append(
                f"Elem: node indices must be >= 0, got "
                f"idx_node_1={self.idx_node_1}, idx_node_2={self.idx_node_2}"
            )
        if self.x_b <= self.x_a:
            errors.append(
                f"Elem: x_b must be > x_a, got x_a={self.x_a}, x_b={self.x_b}"
            )

        if (self.beam_theory, self.element_order) not in _FORMULATION_REGISTRY:
            errors.append(
                f"Elem: no formulation registered for "
                f"(beam_theory='{self.beam_theory}', "
                f"element_order='{self.element_order}'). "
                f"Registered: {sorted(_FORMULATION_REGISTRY)}"
            )
            return errors

        formulation = self._formulation()
        if isinstance(formulation, ShearDeformableBeamFormulation):
            if self.shear_theory not in VALID_SHEAR_THEORIES:
                errors.append(
                    f"Elem: shear_theory must be one of {VALID_SHEAR_THEORIES}, "
                    f"got '{self.shear_theory}'"
                )
            if self.shear_theory == "hutchinson" and self.radius_ratio != 0.0:
                errors.append(
                    "Elem: shear_theory='hutchinson' with radius_ratio != 0.0 "
                    "(hollow section) is not yet implemented — see "
                    "ShearFactor.hutchinson_factor(). Use shear_theory='cowper' "
                    "for hollow sections for now."
                )
            if self.integration_method not in VALID_INTEGRATION_METHODS:
                errors.append(
                    f"Elem: integration_method must be one of "
                    f"{VALID_INTEGRATION_METHODS}, got "
                    f"'{self.integration_method}'"
                )
        else:
            if self.shear_theory is not None:
                errors.append(
                    f"Elem: beam_theory='{self.beam_theory}' does not use "
                    f"shear_theory (got '{self.shear_theory}'); set it to None."
                )
            if self.integration_method is not None:
                errors.append(
                    f"Elem: beam_theory='{self.beam_theory}' uses a closed-form "
                    "stiffness matrix and does not use integration_method "
                    f"(got '{self.integration_method}'); set it to None."
                )

        return errors

    def __repr__(self) -> str:
        """Beam theory, element order, section and material data (lengths in mm, E in MPa) and node indices."""
        return (f"Elem(beam_theory={self.beam_theory!r}, "
                f"element_order={self.element_order!r}, "
                f"length={self.length:.3f} mm, E={self.E:.1f} MPa, "
                f"I={self.I:.3f} mm^4, A={self.A:.3f} mm^2, v={self.v:.3f}, "
                f"shear_theory={self.shear_theory!r}, "
                f"integration_method={self.integration_method!r}, "
                f"nodes=({self.idx_node_1}, {self.idx_node_2}))")


# =====================================================================
# ---- QuadraticTimoshenkoElem: DESIGN PLACEHOLDER, NOT FUNCTIONAL ------
# =====================================================================

class QuadraticTimoshenkoElem(ElemBase):
    """Placeholder for a 3-node quadratic Timoshenko element (``element_order="quadratic"``); not functional.

    It reserves the slot in the ElemBase hierarchy. ``__init__``,
    ``node_indices`` and ``validate`` raise NotImplementedError. No quadratic
    formulation class exists and none is registered in
    _FORMULATION_REGISTRY.

    Notes
    -----
    Open points before it can be implemented:

    1. Axial DOFs: whether the mid-node carries an axial DOF. The inherited
       ``axial_*`` methods read only ``length``, ``E`` and ``A`` and would
       silently keep the axial behaviour 2-DOF (end nodes only).
    2. Three ordered node indices, with ``validate()`` checking them, and an
       integration option for the quadratic shear term that the shared
       ``VALID_INTEGRATION_METHODS`` (``"single_point"``, ``"exact"``) does
       not contain.
    3. Mesh side: Mesh1D emits nodes only at element end points (2-node
       topology); a mid-node position (the element midpoint is the obvious
       choice) is not generated today.
    4. A quadratic formulation must keep no state on the shared singleton
       that ``register_formulation`` creates.
    """

    def __init__(self, *args, **kwargs):
        """Always raises NotImplementedError: the element is a placeholder."""
        raise NotImplementedError(
            "QuadraticTimoshenkoElem is a design placeholder, not yet "
            "implemented -- see the class docstring for what's still open "
            "(axial DOF handling, node validation, mesh mid-node placement, "
            "and QuadraticTimoshenkoBeam registration)."
        )

    @property
    def node_indices(self) -> tuple[int, int, int]:
        """Always raises NotImplementedError: the element is a placeholder."""
        raise NotImplementedError

    def validate(self) -> list[str]:
        """Always raises NotImplementedError: the element is a placeholder."""
        raise NotImplementedError