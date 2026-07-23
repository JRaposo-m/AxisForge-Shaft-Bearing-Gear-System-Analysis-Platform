"""
mesh/oneD/shaft/mesh_generation/mesh_1D.py
"""

from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import GearElement, ShaftSystem
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalGearSystem
from axisforge.core.materials import get_material
from axisforge.config import MESH_MIN_NODE_DIST_MM


class Mesh1D:
    """
    Generates the 1D FEM node grid for a single ShaftSystem.

    The mesh is built from MANDATORY positions only (no adaptive
    refinement yet) — every section boundary, bearing, gear, and load
    position must coincide with a node so no load/support is applied
    mid-element.

    Parameters
    ----------
    shaft_system : ShaftSystem
        Single-shaft container — bearings, gears, and loads already
        placed via add_bearing/add_gear/add_load, and (if applicable)
        gear-mesh loads already injected via SpurHelicalGearSystem.resolve().
    """

    def __init__(
        self,
        shaft_system: ShaftSystem,
        extra_mandatory: list[float] | None = None,
    ):
        self.shaft_system = shaft_system
        self._x_nodes: list[float] | None = None
        self._extra_mandatory: list[float] = extra_mandatory or []

    # ------------------------------------------------------------------
    # Mandatory node positions
    # ------------------------------------------------------------------

    def _mandatory_positions(self) -> list[float]:
        """
        Collect every axial position that MUST land on a node.

        """
        shaft_system = self.shaft_system
        shaft = shaft_system.shaft

        x_mandatory: list[float] = []

        # section boundaries
        x_mandatory += [shaft.axial_start(i) for i in range(shaft.n_sections)]
        x_mandatory.append(shaft.axial_end(shaft.n_sections - 1))

        # bearings
        x_mandatory += [b.position for b in shaft_system.bearings]
        for bearing in shaft_system.bearings:
            lo_bearing, hi_bearing = shaft_system.bearing_extent(bearing)
            x_mandatory += [lo_bearing, hi_bearing]

        # gears (GearElement.position delegates to gear.position)
        x_mandatory += [g.position for g in shaft_system.gears]
        for gear in shaft_system.gears:
            lo_gear, hi_gear = shaft_system.gear_extent(gear)
            x_mandatory += [lo_gear, hi_gear]

        # loads — one RadialLoad/ExternalMoment already covers both XY/XZ
        # planes via theta_deg, no separate xz/xy lists needed here
        x_mandatory += [ld.position for ld in shaft_system.radial_loads]
        x_mandatory += [ld.position for ld in shaft_system.axial_loads]
        x_mandatory += [ld.position for ld in shaft_system.torque_loads]
        x_mandatory += [m.position for m in shaft_system.external_moments]

        # distributed radial loads — x_lo and x_hi are discontinuities in V(x)
        for ld in shaft_system.distributed_radial_loads:
            x_mandatory += [ld.x_lo, ld.x_hi]

        # extra positions injected externally (e.g. mesh convergence study)
        x_mandatory += self._extra_mandatory

        return x_mandatory

    def _create_mesh(self) -> list[float]:
        """
        Sort mandatory positions, dedupe/merge points closer than
        MESH_MIN_NODE_DIST_MM, return the final node grid.
        """
        x_mandatory = self._mandatory_positions()

        x_nodes = sorted(set(x_mandatory))
        x_nodes = [
            x for i, x in enumerate(x_nodes)
            if i == 0 or x - x_nodes[i - 1] > MESH_MIN_NODE_DIST_MM
        ]
        return x_nodes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> list[float]:
        """Compute (or return cached) node positions."""
        if self._x_nodes is None:
            self._x_nodes = self._create_mesh()
        return self._x_nodes

    @property
    def x_nodes(self) -> list[float]:
        return self.build()

    @property
    def n_nodes(self) -> int:
        return len(self.x_nodes)

    def show_nodes(self, print_output: bool = True) -> list[tuple[int, float]]:
        """
        Return a numbered list of all node positions in the current mesh.

        Parameters
        ----------
        print_output : if True, prints the node table to stdout

        Returns
        -------
        list of (node_index, x_position_mm) tuples
        """
        nodes = self.x_nodes
        result = [(i, x) for i, x in enumerate(nodes)]

        if print_output:
            print(f"Mesh1D — {len(nodes)} nodes")
            print("-" * 35)
            for i, x in result:
                print(f"  node {i:>3d}  :  {x:.4f} mm")
            print("-" * 35)

        return result