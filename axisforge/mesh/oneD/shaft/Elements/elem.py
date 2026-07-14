"""
mesh/oneD/shaft/Elements/elem.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axisforge.core.materials import get_material  
from axisforge.config import MESH_MIN_NODE_DIST_MM

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
        ShaftSystem,
    )
    from mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D


class Elem:
    """
    Single 1D beam element between two mesh nodes.
    """

    def __init__(self, length: float, E: float, I: float, A: float,
                 v: float, idx_node_1: int, idx_node_2: int):
        self.length     = length
        self.E          = E
        self.I          = I
        self.A          = A
        self.v          = v
        self.idx_node_1 = idx_node_1
        self.idx_node_2 = idx_node_2

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
    def from_mesh(cls, mesh: "Mesh1D", node_tol: float = MESH_MIN_NODE_DIST_MM) -> list["Elem"]:
        """
        Build the full element list from a Mesh1D — reads mesh.shaft_system
        and mesh.x_nodes directly. Mesh1D itself carries no knowledge of
        Elem; this is the single place that bridges mesh positions to
        element construction.
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

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    def __repr__(self) -> str:
        return (f"Elem(length={self.length:.3f} mm, E={self.E:.1f} MPa, "
                f"I={self.I:.3f} mm^4, A={self.A:.3f} mm^2, v={self.v:.3f}, "
                f"nodes=({self.idx_node_1}, {self.idx_node_2}))")