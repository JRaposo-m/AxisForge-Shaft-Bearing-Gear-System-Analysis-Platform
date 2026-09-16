"""
mesh/shaft/element_type/elem.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axisforge.core.materials import get_material
from axisforge.config import MESH_MIN_NODE_DIST_MM
from axisforge.mesh.shaft.beam_model_settings import (
    BeamModelSettings, VALID_BEAM_THEORIES, VALID_SHEAR_THEORIES,
    VALID_INTEGRATION_METHODS,
)
from axisforge.mesh.shaft.element_type.euler_bernoulli.two_noded import EulerBernoulliBeam
from axisforge.mesh.shaft.element_type.timoshenko.two_noded import TimoshenkoBeam
from axisforge.mesh.shaft.element_type.frame_element import FrameElement

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
        ShaftSystem,
    )
    from mesh.shaft.mesh_generation.mesh_1D import Mesh1D


class Elem:
    """
    Single 1D beam element between two mesh nodes.

    Dispatches shape function / strain / stiffness calls explicitly to
    EulerBernoulliBeam or TimoshenkoBeam according to self.beam_theory.
    The beam theory (and, for Timoshenko, the shear correction theory) is
    not decided here — it comes from a BeamModelSettings instance passed
    in at construction time, so the choice is made once, explicitly, by
    whatever caller builds the mesh, and Elem only carries it forward.
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

    # ------------------------------------------------------------------
    # Explicit per-element-type dispatch
    # ------------------------------------------------------------------

    def _beam(self):
        if self.beam_theory == "euler_bernoulli":
            return EulerBernoulliBeam()
        elif self.beam_theory == "timoshenko":
            return TimoshenkoBeam()
        raise ValueError(f"Elem: unknown beam_theory '{self.beam_theory}'")

    def shape_functions(self, zeta: float) -> "np.ndarray":
        if self.beam_theory == "euler_bernoulli":
            return EulerBernoulliBeam().shape_functions(zeta, self)
        elif self.beam_theory == "timoshenko":
            return TimoshenkoBeam().shape_functions(zeta, self)
        raise ValueError(f"Elem: unknown beam_theory '{self.beam_theory}'")

    def bending_strain_matrix(self, zeta: float) -> "np.ndarray":
        if self.beam_theory == "euler_bernoulli":
            return EulerBernoulliBeam().bending_strain_matrix(zeta, self)
        elif self.beam_theory == "timoshenko":
            return TimoshenkoBeam().bending_strain_matrix(zeta, self)
        raise ValueError(f"Elem: unknown beam_theory '{self.beam_theory}'")

    def shear_rigidity(self, *, kGA_override: float | None = None) -> float:
        if self.beam_theory == "timoshenko":
            return TimoshenkoBeam().shear_rigidity(self, kGA_override=kGA_override)
        raise ValueError(f"Elem: shear_rigidity() only applies to 'timoshenko', got '{self.beam_theory}'")

    def shear_strain_matrix(self, zeta: float) -> "np.ndarray":
        if self.beam_theory == "timoshenko":
            return TimoshenkoBeam().shear_strain_matrix(
                zeta, self, integration=self.integration_method,
            )
        else:
            raise ValueError(f"Elem: '{self.beam_theory}' is not valid"
                            f" Only timoshenko beam_theory is valid.")           

    def stiffness_element(self, *, kGA_override: float | None = None) -> "np.ndarray":
        if self.beam_theory == "euler_bernoulli":
            return EulerBernoulliBeam().stiffness_element(self)
        elif self.beam_theory == "timoshenko":
            return TimoshenkoBeam().stiffness_element(
                self, integration=self.integration_method,
                shear_theory=self.shear_theory,
                kGA_override=kGA_override,
            )
        raise ValueError(f"Elem: unknown beam_theory '{self.beam_theory}'")

    def natural_coordenates(self, x: float) -> float:
        return self._beam().natural_coordenates(x, self)

    # ------------------------------------------------------------------
    # Axial ("frame") behaviour -- theory-independent, no dispatch:
    # every Elem has this regardless of beam_theory. See
    # frame_element.py for why this is not itself a "theory".
    # ------------------------------------------------------------------

    def axial_shape_functions(self, zeta: float) -> "np.ndarray":
        return FrameElement().shape_functions(zeta, self)

    def axial_strain_matrix(self) -> "np.ndarray":
        return FrameElement().axial_strain_matrix(self)

    def axial_stiffness_element(self) -> "np.ndarray":
        return FrameElement().stiffness_element(self)

    # ------------------------------------------------------------------
    # Factory — builds the full element list directly from a Mesh1D
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
        Build the full element list from a Mesh1D — reads mesh.shaft_system
        and mesh.x_nodes directly. Mesh1D itself carries no knowledge of
        Elem; this is the single place that bridges mesh positions to
        element construction.

        `settings` decides explicitly which beam theory (and, for
        Timoshenko, which shear correction theory) is used for the whole
        shaft. There is no default here on purpose — the caller must
        decide.
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
        if self.beam_theory not in VALID_BEAM_THEORIES:
            errors.append(
                f"Elem: beam_theory must be one of {VALID_BEAM_THEORIES}, "
                f"got '{self.beam_theory}'"
            )
        if self.beam_theory == "euler_bernoulli":
            if self.shear_theory is not None:
                errors.append(
                    "Elem: beam_theory='euler_bernoulli' does not use "
                    f"shear_theory (got '{self.shear_theory}'); set it to None."
                )
            if self.integration_method is not None:
                errors.append(
                    "Elem: beam_theory='euler_bernoulli' uses a closed-form "
                    "stiffness matrix and does not use integration_method "
                    f"(got '{self.integration_method}'); set it to None."
                )
        elif self.beam_theory == "timoshenko":
            if self.shear_theory not in VALID_SHEAR_THEORIES:
                errors.append(
                    f"Elem: shear_theory must be one of {VALID_SHEAR_THEORIES}, "
                    f"got '{self.shear_theory}'"
                )
            if self.shear_theory == "hutchinson" and self.radius_ratio != 0.0:
                errors.append(
                    "Elem: shear_theory='hutchinson' with radius_ratio != 0.0 "
                    "(hollow section) is not yet implemented — see "
                    "TimoshenkoBeam._hutchinson_factor(). Use shear_theory='cowper' "
                    "for hollow sections for now."
                )
            if self.integration_method not in VALID_INTEGRATION_METHODS:
                errors.append(
                    f"Elem: integration_method must be one of "
                    f"{VALID_INTEGRATION_METHODS}, got "
                    f"'{self.integration_method}'"
                )
        # Note: beam_theory/shear_theory/integration_method pairing is already
        # validated by BeamModelSettings at construction time, so this branch
        # is normally unreachable from cls(...) — it stays here as a guard
        # against an Elem instance being mutated after construction (Elem is
        # not frozen).

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    def __repr__(self) -> str:
        return (f"Elem(beam_theory={self.beam_theory!r}, "
                f"length={self.length:.3f} mm, E={self.E:.1f} MPa, "
                f"I={self.I:.3f} mm^4, A={self.A:.3f} mm^2, v={self.v:.3f}, "
                f"shear_theory={self.shear_theory!r}, "
                f"integration_method={self.integration_method!r}, "
                f"nodes=({self.idx_node_1}, {self.idx_node_2}))")