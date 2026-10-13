"""
1D FEM node grid for a single ShaftSystem: the axial positions [mm] of the beam element nodes.

Two classes of node
-------------------
* **Mandatory** nodes come from the physical model and must always be
  present: section boundaries (shoulders), shaft ends, bearing centres
  and extents, gear centres and face-width limits, point-load positions
  (radial, axial and torque loads, external moments),
  distributed-load limits and centroids, and any ``protected`` position
  supplied by the caller (e.g. the cut nodes of a submodel). They are
  never removed.
* **Extra** nodes are refinement or user input: ``extra_mandatory`` and
  the nodes injected by graders (mesh-convergence studies). They are
  merged away when they fall within ``MESH_MIN_NODE_DIST_MM`` of another
  node.

Merging rules
-------------
* Positions within ``NODE_LOOKUP_TOL_MM`` of each other are the same
  point and collapse silently (e.g. a gear centre and a load applied at
  the same x).
* extra close to mandatory  -> the extra node is absorbed by the mandatory
  one (recorded in :meth:`Mesh1D.merge_report`);
* extra close to extra       -> the first is kept, the second dropped
  (recorded);
* mandatory close to mandatory (gap <= ``MESH_MIN_NODE_DIST_MM`` and
  > ``NODE_LOOKUP_TOL_MM``) -> **error**: the model itself places two physical
  features less than the minimum element length apart. Reported by
  :meth:`Mesh1D.validate` and raised by :meth:`Mesh1D.build`.

``MESH_MIN_NODE_DIST_MM`` (0.1 mm) and ``NODE_LOOKUP_TOL_MM`` (1e-6 mm) come
from ``axisforge.config``; they are values of this code, not from a source.
There is no maximum element length and no automatic subdivision between
mandatory nodes: element size is set by the model and by the refinement
nodes only.
"""

from __future__ import annotations

from dataclasses import dataclass

from axisforge.core.mechanical_system.shaft_system import GearElement, ShaftSystem
from axisforge.core.mechanical_system.gear_system import SpurHelicalGearSystem
from axisforge.core.materials import get_material
from axisforge.core.loads import LoadPlane
from axisforge.mesh.shaft.mesh_generation.mesh_grade import Grader
from axisforge.config import MESH_MIN_NODE_DIST_MM, NODE_LOOKUP_TOL_MM


@dataclass(frozen=True)
class MergeEvent:
    """One node removed while building the mesh.

    Attributes
    ----------
    kept_x, dropped_x : float
        Position of the surviving node and of the removed one [mm].
    kept_origin, dropped_origin : str
        What each node represents (e.g. ``"bearing 'b1' extent_hi"``,
        ``"extra_mandatory"``, ``"grader grade_3"``).
    distance : float
        |dropped_x - kept_x| [mm].
    """

    kept_x: float
    kept_origin: str
    dropped_x: float
    dropped_origin: str
    distance: float


class Mesh1D:
    """
    Generates the 1D FEM node grid for a single ShaftSystem.

    Every physical position (section boundary, bearing, gear, load) is a
    mandatory node, so no load or support is ever applied mid-element.
    No adaptive refinement: extra nodes come from ``extra_mandatory`` or
    from graders.

    Parameters
    ----------
    shaft_system : ShaftSystem
        Single-shaft container -- bearings, gears and loads already placed;
        for the gear-mesh loads, the loaded copy returned by
        SpurHelicalGearSystem.resolve().
    extra_mandatory : list of float, optional
        Refinement or user-supplied positions [mm]. Low priority: merged
        into a nearby node when closer than ``MESH_MIN_NODE_DIST_MM``.
    protected : list of float, optional
        Additional positions [mm] that must survive like the physical ones
        (e.g. submodel cut nodes). Same priority as mandatory nodes.

    Notes
    -----
    :meth:`merge_report` lists every node removed and why;
    :meth:`validate` lists conflicts between mandatory nodes, which make
    :meth:`build` raise.
    """

    def __init__(
        self,
        shaft_system: ShaftSystem,
        extra_mandatory: list[float] | None = None,
        protected: list[float] | None = None,
    ):
        """Store the arguments; see the class docstring."""
        self.shaft_system = shaft_system
        self._x_nodes: list[float] | None = None
        self._extra_mandatory: list[float] = extra_mandatory or []
        self._protected: list[float] = protected or []
        self._graders: list[tuple["Grader", str]] = []
        self._merge_log: list[MergeEvent] = []
        self._conflicts: list[str] = []

    # ------------------------------------------------------------------
    # Node positions with their origin
    # ------------------------------------------------------------------

    @staticmethod
    def _tag(obj, default: str) -> str:
        """Origin text ``"<default> <label>"`` used in merge reports."""
        label = getattr(obj, "label", "") or getattr(obj, "designation", "") or ""
        return f"{default} '{label}'" if label else default

    def _mandatory_tagged(self) -> list[tuple[float, str]]:
        """Every position that MUST land on a node, with its origin."""
        shaft_system = self.shaft_system
        shaft = shaft_system.shaft
        out: list[tuple[float, str]] = []

        # section boundaries (shoulders) and shaft ends
        out += [(shaft.axial_start(i), f"section {i} start") for i in range(shaft.n_sections)]
        out.append((shaft.axial_end(shaft.n_sections - 1), "shaft end"))

        # bearings
        for b in shaft_system.bearings:
            tag = self._tag(b, "bearing")
            lo, hi = shaft_system.bearing_extent(b)
            out += [(b.position, f"{tag} centre"), (lo, f"{tag} extent_lo"), (hi, f"{tag} extent_hi")]

        # gears (GearElement.position delegates to gear.position)
        for g in shaft_system.gears:
            tag = self._tag(g, "gear")
            lo, hi = shaft_system.gear_extent(g)
            out += [(g.position, f"{tag} centre"), (lo, f"{tag} face_lo"), (hi, f"{tag} face_hi")]

        # point loads -- one RadialLoad/ExternalMoment covers both planes
        out += [(ld.position, self._tag(ld, "radial load")) for ld in shaft_system.radial_loads]
        out += [(ld.position, self._tag(ld, "axial load")) for ld in shaft_system.axial_loads]
        out += [(ld.position, self._tag(ld, "torque load")) for ld in shaft_system.torque_loads]
        out += [(m.position, self._tag(m, "external moment")) for m in shaft_system.external_moments]

        # distributed radial loads -- x_lo and x_hi are discontinuities in V(x)
        for ld in shaft_system.distributed_radial_loads:
            tag = self._tag(ld, "distributed load")
            out += [(ld.x_lo, f"{tag} x_lo"), (ld.x_hi, f"{tag} x_hi"),
                    (ld.centroid(LoadPlane.XY), f"{tag} centroid_xy"),
                    (ld.centroid(LoadPlane.XZ), f"{tag} centroid_xz")]

        # caller-protected positions (e.g. submodel cut nodes)
        out += [(x, "protected") for x in self._protected]
        return out

    def _extra_tagged(self) -> list[tuple[float, str]]:
        """Refinement / user positions, with their origin."""
        out = [(x, "extra_mandatory") for x in self._extra_mandatory]
        for grader, grade in self._graders:
            out += [(x, f"grader {grade}") for x in grader.get_grade(grade)]
        return out

    def _mandatory_positions(self) -> list[float]:
        """All candidate positions (mandatory then extra) [mm], untagged.

        Used only by the tests; :meth:`_create_mesh` works on the tagged lists.
        """
        return [x for x, _ in self._mandatory_tagged()] + [x for x, _ in self._extra_tagged()]

    # ------------------------------------------------------------------
    # Node grid
    # ------------------------------------------------------------------

    def _create_mesh(self) -> list[float]:
        """Merge the candidate positions into the final node grid.

        Mandatory nodes are kept; extra nodes are absorbed or dropped when
        closer than ``MESH_MIN_NODE_DIST_MM`` to a node already kept. Fills
        ``self._merge_log`` and ``self._conflicts``.
        """
        self._merge_log = []
        self._conflicts = []

        # 1. mandatory: collapse coincident points, flag close-but-distinct pairs
        mandatory: list[tuple[float, str]] = []
        for x, origin in sorted(self._mandatory_tagged(), key=lambda t: t[0]):
            if mandatory and x - mandatory[-1][0] <= NODE_LOOKUP_TOL_MM:
                continue                                    # same physical point
            if mandatory and x - mandatory[-1][0] <= MESH_MIN_NODE_DIST_MM:
                px, porigin = mandatory[-1]
                self._conflicts.append(
                    f"mandatory nodes {px:.6g} mm ({porigin}) and {x:.6g} mm ({origin}) "
                    f"are {x - px:.3g} mm apart, closer than MESH_MIN_NODE_DIST_MM = "
                    f"{MESH_MIN_NODE_DIST_MM} mm -- adjust the model geometry/loads"
                )
            mandatory.append((x, origin))

        # 2. extra: absorbed by a mandatory node, or by an extra node already kept
        mand_x = [x for x, _ in mandatory]
        kept_extra: list[tuple[float, str]] = []
        for x, origin in sorted(self._extra_tagged(), key=lambda t: t[0]):
            j = min(range(len(mandatory)), key=lambda k: abs(mand_x[k] - x))
            d = abs(mand_x[j] - x)
            if d <= MESH_MIN_NODE_DIST_MM:
                if d > NODE_LOOKUP_TOL_MM:
                    self._merge_log.append(MergeEvent(mand_x[j], mandatory[j][1], x, origin, d))
                continue
            if kept_extra:
                kx, korigin = kept_extra[-1]
                if x - kx <= MESH_MIN_NODE_DIST_MM:
                    if x - kx > NODE_LOOKUP_TOL_MM:
                        self._merge_log.append(MergeEvent(kx, korigin, x, origin, x - kx))
                    continue
            kept_extra.append((x, origin))

        return sorted(mand_x + [x for x, _ in kept_extra])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> list[float]:
        """Compute (or return cached) node positions.

        Returns
        -------
        list of float
            Ascending node positions [mm]; mandatory nodes plus the extra
            nodes that survived the merge rules.

        Raises
        ------
        ValueError
            If two mandatory nodes are closer than ``MESH_MIN_NODE_DIST_MM``
            (see :meth:`validate`).
        """
        if self._x_nodes is None:
            nodes = self._create_mesh()          # fills _conflicts / _merge_log
            self._raise_on_conflicts(self._conflicts)
            self._x_nodes = nodes
        return self._x_nodes

    @property
    def x_nodes(self) -> list[float]:
        """Ascending node positions [mm] (same list as :meth:`build`)."""
        return self.build()

    @property
    def n_nodes(self) -> int:
        """Number of nodes [-]."""
        return len(self.x_nodes)

    def merge_report(self) -> list[MergeEvent]:
        """Nodes removed while building the mesh, in ascending position.

        Returns
        -------
        list of MergeEvent
            One entry per extra node absorbed by a mandatory node or dropped
            in favour of another extra node. Coincident positions (within
            ``NODE_LOOKUP_TOL_MM``) are not reported.
        """
        self.build()
        return list(self._merge_log)

    def validate(self) -> list[str]:
        """Conflicts between mandatory nodes (empty list if none).

        Returns
        -------
        list of str
            One message per pair of mandatory nodes closer than
            ``MESH_MIN_NODE_DIST_MM`` -- these cannot be merged without
            moving a physical feature.
        """
        if self._x_nodes is None:
            self._create_mesh()
        return list(self._conflicts)

    def validate_or_raise(self) -> None:
        """Raise ValueError listing every mandatory-node conflict.

        Raises
        ------
        ValueError
            If :meth:`validate` returns any message.
        """
        self._raise_on_conflicts(self.validate())

    @staticmethod
    def _raise_on_conflicts(errors: list[str]) -> None:
        """Raise ValueError with one line per conflict message, if any."""
        if errors:
            raise ValueError("Mesh1D: mandatory-node conflicts:\n"
                             + "\n".join(f"  - {e}" for e in errors))

    def show_nodes(self, print_output: bool = True) -> list[tuple[int, float]]:
        """Numbered list of all node positions of the current mesh.

        Parameters
        ----------
        print_output : bool, optional
            If True, print the node table to stdout.

        Returns
        -------
        list of (int, float)
            ``(node_index, x)`` with ``x`` the axial position [mm].
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

    def add_grader(self, grader: "Grader", grade: str) -> None:
        """Add the nodes of a Grader grade as extra (refinement) nodes.

        Parameters
        ----------
        grader : Grader
            Built from the base mesh ``x_nodes``, so call this after the base
            mesh exists.
        grade : str
            Grade name, ``"grade_N"`` (see Grader.get_grade).

        Notes
        -----
        Invalidates the node cache; the next ``x_nodes`` call rebuilds.
        Grader nodes closer than ``MESH_MIN_NODE_DIST_MM`` to another node
        are merged away (see :meth:`merge_report`).
        """
        self._graders.append((grader, grade))
        self._x_nodes = None

    def clear_graders(self) -> None:
        """Remove all injected graders. Invalidates cache."""
        self._graders.clear()
        self._x_nodes = None