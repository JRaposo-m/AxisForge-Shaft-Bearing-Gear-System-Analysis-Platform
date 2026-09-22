"""
axisforge/mesh/shaft/element_type/elem.py

Single file holding:
  - the BeamFormulation contract and its shear-deformable specialization,
  - every concrete beam formulation (EulerBernoulliBeam, TimoshenkoBeam),
  - the axial/frame formulation (FrameElement) -- deliberately outside
    the BeamFormulation hierarchy, see its own docstring below,
  - ElemBase, the node-count-agnostic contract for "an element between
    mesh nodes",
  - Elem, the 2-node implementation used throughout AxisForge today,
  - QuadraticTimoshenkoElem, a DESIGN PLACEHOLDER for a 3-node element
    -- not yet functional, see its own docstring for what's open.

Shear correction factors (cowper/hutchinson) stay in the sibling module
shear_factor.py -- pure math with no knowledge of Elem/BeamFormulation,
imported by module (not by symbol), mirroring how
core/machine_elements/bearings/families/family.py imports its sibling
./contact.py and ./capacity.py. Dependency direction is strictly
downward: elem.py -> shear_factor.py, never the reverse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from axisforge.core.materials import get_material
from axisforge.config import MESH_MIN_NODE_DIST_MM
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
    """
    Class decorator: instantiates the decorated BeamFormulation once
    (stateless -- all physical data lives on the Elem passed into each
    call) and registers that singleton under (beam_theory,
    element_order). Elem resolves this tuple; nothing else needs to
    know the mapping exists.
    """
    def _decorator(cls: type["BeamFormulation"]) -> type["BeamFormulation"]:
        _FORMULATION_REGISTRY[(beam_theory, element_order)] = cls()
        return cls
    return _decorator


def _linear_shape_functions_2node(zeta: float) -> np.ndarray:
    """
    N_1 = 1/2 (1 - zeta), N_2 = 1/2 (1 + zeta).

    Shared by TimoshenkoBeam (interpolates v) and FrameElement
    (interpolates u) -- same interpolation, different physical field.
    """
    return np.array([0.5 * (1.0 - zeta), 0.5 * (1.0 + zeta)])


# =====================================================================
# ---- contract: BeamFormulation ---------------------------------------
# =====================================================================

class BeamFormulation(ABC):
    """
    Contract every bending theory must satisfy. Concrete subclasses are
    stateless -- see register_formulation(), which stores singletons.
    """

    @abstractmethod
    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray: ...

    @abstractmethod
    def bending_strain_matrix(self, zeta: float, elem: "ElemBase") -> np.ndarray: ...

    @abstractmethod
    def stiffness_element(self, elem: "ElemBase", **kwargs) -> np.ndarray: ...

    def natural_coordenates(self, x: float, elem: "ElemBase") -> float:
        """
        Global x -> natural coordinate zeta in [-1, 1]. Identical for
        every formulation that spans [x_a, x_b] -- concrete here so
        concrete formulations don't each redefine it.
        """
        x_c = (elem.x_a + elem.x_b) / 2.0
        return 2.0 * (x - x_c) / elem.length


class ShearDeformableBeamFormulation(BeamFormulation):
    """
    Specialization for theories with an independent transverse-shear
    field (currently: Timoshenko, both linear and -- once wired in --
    quadratic). A subclass, not a second independent ABC mixed in --
    shear deformation is what makes a theory Timoshenko rather than
    Euler-Bernoulli, not an optional add-on. Elem checks
    isinstance(formulation, ShearDeformableBeamFormulation) instead of
    comparing beam_theory strings.
    """

    @abstractmethod
    def shear_rigidity(self, elem: "ElemBase", *, kGA_override: float | None = None) -> float: ...

    @abstractmethod
    def shear_strain_matrix(self, zeta: float, elem: "ElemBase", integration: str) -> np.ndarray: ...


# =====================================================================
# ---- concrete: Euler-Bernoulli -----------------------------------------
# =====================================================================

@register_formulation("euler_bernoulli")
class EulerBernoulliBeam(BeamFormulation):

    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        le = elem.length
        N = np.zeros(4)
        N[0] = 1/4 * (2 - 3*zeta + zeta**3)            # transverse v
        N[1] = le/8 * (1 - zeta - zeta**2 + zeta**3)    # rotation theta
        N[2] = 1/4 * (2 + 3*zeta - zeta**3)            # transverse v
        N[3] = le/8 * (-1 - zeta + zeta**2 + zeta**3)   # rotation theta
        return N

    def bending_strain_matrix(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        le = elem.length
        B = np.zeros(4)
        B[0] = 6 * zeta / le**2
        B[1] = (-1 + 3 * zeta) / le
        B[2] = -6 * zeta / le**2
        B[3] = (1 + 3 * zeta) / le
        return B

    def stiffness_element(self, elem: "ElemBase") -> np.ndarray:
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

    _shear_factor = sf.ShearFactor()

    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        return _linear_shape_functions_2node(zeta)

    def bending_strain_matrix(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        le = elem.length
        return np.array([0.0, -1/le, 0.0, 1/le])

    def shear_strain_matrix(self, zeta: float, elem: "ElemBase", integration: str) -> np.ndarray:
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
        G = elem.E / (2 * (1 + elem.v))
        k = self._shear_factor.shear_correction_factor(
            elem.v, elem.radius_ratio, elem.E, elem.A,
            elem.shear_theory, kGA_override=kGA_override,
        )
        return k * G * elem.A

    def stiffness_element(self, elem: "ElemBase", *, integration: str = "single_point",
                           kGA_override: float | None = None) -> np.ndarray:
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
# ---- concrete: Timoshenko (3-node, quadratic) -- NOT REGISTERED YET ----
# =====================================================================
#
# QuadraticTimoshenkoBeam (from timoshenko/three_noded.py) belongs here
# too once it's reviewed -- deliberately left out of this file and out
# of _FORMULATION_REGISTRY for now. Reasons: (1) you asked me to fix
# only the scalar*list bug there, not validate the B_s scaling; (2) it
# still writes to self.shear_correction_parameter, which would corrupt
# state on the shared singleton this registry pattern assumes; (3) its
# stiffness_element() only accepts integration="two_point", a value
# ElemBase's shared dispatch doesn't know to pass (see
# QuadraticTimoshenkoElem docstring below). Once those are resolved:
# add `@register_formulation("timoshenko", "quadratic")` to that class,
# fix self.shear_correction_parameter -> a local variable, and import
# it at the top of this file.


# =====================================================================
# ---- axial ("frame") behaviour -- NOT part of BeamFormulation --------
# =====================================================================

class FrameElement:
    """
    Axial DOF behaviour (u_a, u_b), present on every element regardless
    of beam_theory. Not a BeamFormulation and not registered in
    _FORMULATION_REGISTRY: there is no beam-theory-dependent variant to
    select between -- this is unconditional composition, not strategy
    selection.

    NOTE for 3-node elements (see QuadraticTimoshenkoElem below): this
    only ever reads elem.length/E/A, so it happens to "work" (produce a
    2x2 matrix) even when called from a 3-node element -- but that
    silently assumes axial behaviour stays 2-DOF (end nodes only, mid-
    node ignored for axial) even when bending/shear go quadratic. That
    assumption has not been reviewed; see QuadraticTimoshenkoElem.
    """

    def shape_functions(self, zeta: float, elem: "ElemBase") -> np.ndarray:
        return _linear_shape_functions_2node(zeta)

    def axial_strain_matrix(self, elem: "ElemBase") -> np.ndarray:
        le = elem.length
        return np.array([-1.0 / le, 1.0 / le])

    def stiffness_element(self, elem: "ElemBase") -> np.ndarray:
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
        """Mesh node indices this element connects, in local order."""
        ...

    @abstractmethod
    def validate(self) -> list[str]: ...

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Formulation dispatch -- identical regardless of node count
    # ------------------------------------------------------------------

    def _formulation(self) -> BeamFormulation:
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
        return self._formulation().shape_functions(zeta, self)

    def bending_strain_matrix(self, zeta: float) -> np.ndarray:
        return self._formulation().bending_strain_matrix(zeta, self)

    def stiffness_element(self, *, kGA_override: float | None = None) -> np.ndarray:
        formulation = self._formulation()
        if isinstance(formulation, ShearDeformableBeamFormulation):
            return formulation.stiffness_element(
                self, integration=self.integration_method, kGA_override=kGA_override,
            )
        return formulation.stiffness_element(self)

    def shear_rigidity(self, *, kGA_override: float | None = None) -> float:
        formulation = self._formulation()
        if not isinstance(formulation, ShearDeformableBeamFormulation):
            raise ValueError(
                f"{type(self).__name__}: shear_rigidity() only applies to "
                f"shear-deformable theories, got '{self.beam_theory}'"
            )
        return formulation.shear_rigidity(self, kGA_override=kGA_override)

    def shear_strain_matrix(self, zeta: float) -> np.ndarray:
        formulation = self._formulation()
        if not isinstance(formulation, ShearDeformableBeamFormulation):
            raise ValueError(
                f"{type(self).__name__}: shear_strain_matrix() only applies "
                f"to shear-deformable theories, got '{self.beam_theory}'"
            )
        return formulation.shear_strain_matrix(zeta, self, integration=self.integration_method)

    def natural_coordenates(self, x: float) -> float:
        return self._formulation().natural_coordenates(x, self)

    # ------------------------------------------------------------------
    # Axial ("frame") behaviour -- unconditional, no dispatch
    # ------------------------------------------------------------------

    def axial_shape_functions(self, zeta: float) -> np.ndarray:
        return self._frame.shape_functions(zeta, self)

    def axial_strain_matrix(self) -> np.ndarray:
        return self._frame.axial_strain_matrix(self)

    def axial_stiffness_element(self) -> np.ndarray:
        return self._frame.stiffness_element(self)


# =====================================================================
# ---- Elem: the 2-node element used throughout AxisForge today --------
# =====================================================================

class Elem(ElemBase):
    """
    2-node beam element between two mesh nodes -- euler_bernoulli or
    timoshenko ("linear" element_order). Unchanged behaviour from
    before this file was split into ElemBase/Elem; only the formulation
    dispatch it used to implement directly now lives in ElemBase.
    """

    VALID_BEAM_THEORIES       = VALID_BEAM_THEORIES
    VALID_SHEAR_THEORIES      = VALID_SHEAR_THEORIES
    VALID_INTEGRATION_METHODS = VALID_INTEGRATION_METHODS

    def __init__(self, length: float, E: float, I: float, A: float,
                 v: float, idx_node_1: int, idx_node_2: int,
                 x_a: float, x_b: float,
                 settings: BeamModelSettings,
                 radius_ratio: float = 0.0):
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

        # element_order isn't a BeamModelSettings field yet -- defaults
        # to "linear" (today's only wired-in case) so this stays
        # backward compatible until 3-node support is designed.
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
        return (self.idx_node_1, self.idx_node_2)

    # ------------------------------------------------------------------
    # Factory -- builds the full element list directly from a Mesh1D
    # ------------------------------------------------------------------

    @staticmethod
    def find_node_index(x_nodes: list[float], x: float, tol: float = MESH_MIN_NODE_DIST_MM) -> int:
        for i, xn in enumerate(x_nodes):
            if abs(xn - x) <= tol:
                return i
        raise ValueError(f"No node found at x={x:.4f} mm within tolerance {tol} mm")

    @classmethod
    def from_mesh(cls, mesh: "Mesh1D", settings: BeamModelSettings,
                  node_tol: float = MESH_MIN_NODE_DIST_MM) -> list["Elem"]:
        """
        Build the full element list from a Mesh1D -- reads
        mesh.shaft_system and mesh.x_nodes directly. Mesh1D itself
        carries no knowledge of Elem; this is the single place that
        bridges mesh positions to element construction.

        `settings` decides explicitly which beam theory (and, for
        Timoshenko, which shear correction theory) is used for the
        whole shaft. There is no default here on purpose -- the caller
        must decide.
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
                    # radius_ratio: wire this up once 'section' can tell
                    # solid vs hollow apart, e.g. section.inner_diameter /
                    # section.diameter
                ))
        return elements

    @classmethod
    def from_x_nodes(cls, x_nodes: list[float], shaft_system: "ShaftSystem",
                      settings: BeamModelSettings) -> list["Elem"]:
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
    """
    Placeholder for a 3-node quadratic Timoshenko element
    (element_order="quadratic"), reserving its shape in the hierarchy
    without pretending the design is finished. __init__ raises
    NotImplementedError on purpose -- this is not meant to be usable
    yet, only present so the intended slot in ElemBase exists and is
    documented.

    Open questions to resolve before this is real, found while sketching it:

    1. Axial ("frame") DOFs -- does the mid-node carry an axial DOF, or
       does axial stay 2-DOF (end nodes only)? ElemBase.axial_* methods
       are inherited unchanged and would silently do the latter (they
       only ever read elem.length/E/A) -- that may or may not be the
       intended physics; not decided here.

    2. node_indices / validate() -- needs 3 distinct, correctly ordered
       indices (e.g. idx_node_1 < idx_node_mid < idx_node_2), and
       validate() needs to check that ordering plus whatever
       QuadraticTimoshenkoBeam-specific constraints apply (it currently
       hard-requires integration="two_point" and raises
       NotImplementedError for anything else -- ElemBase.stiffness_element()
       passes self.integration_method straight through, so
       VALID_INTEGRATION_METHODS / self.integration_method would need a
       "two_point" option, not shared with the 2-node element's
       single_point/exact).

    3. from_mesh()/from_x_nodes() -- Mesh1D currently only emits x_nodes
       at element ENDPOINTS (2-node topology). A mid-node needs a
       mesh-side decision on placement (element midpoint is the obvious
       default) that Mesh1D does not make today -- this is the "terá
       influência na mesh também" you flagged; not addressed in this
       file.

    4. QuadraticTimoshenkoBeam itself is not registered in
       _FORMULATION_REGISTRY yet (see the comment block above
       FrameElement) and still writes to
       self.shear_correction_parameter, which would corrupt state on a
       shared singleton -- fix that before registering it.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "QuadraticTimoshenkoElem is a design placeholder, not yet "
            "implemented -- see the class docstring for what's still open "
            "(axial DOF handling, node validation, mesh mid-node placement, "
            "and QuadraticTimoshenkoBeam registration)."
        )

    @property
    def node_indices(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def validate(self) -> list[str]:
        raise NotImplementedError