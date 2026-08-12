"""
mechanical_system/systems/gear_system.py

Multi-shaft parallel-axis transmission assembler.

Two classes:
  - SpurHelicalMeshLink : one directed mesh, shaft_a(driver) -> shaft_b(driven),
                   carrying a meshing model and the global line-of-centres
                   angle phi_deg. Optional torque_split selects fan-out mode.
  - GearSystem   : a single-source DAG of shafts joined by SpurHelicalMeshLink.
                   resolve() walks the DAG in topological order, propagating
                   torque / rotation sense / shaft position, and injects the
                   resulting mesh loads onto each ShaftSystem.

Scope: parallel-axis only (spur / helical / internal). No bevel/worm, no
convergent torque merge, no multi-source graph — see the context pack §5.

IMPORTANT — fan-out is detected by GearElement IDENTITY, not equality:
A shared driver (e.g. one pinion meshing simultaneously with two wheels)
is recognised ONLY if the SAME GearElement Python object is passed as
gear_a to every link sharing that driver. Two GearElement instances that
describe the identical physical gear (same SpurHelicalGear, same z, same
position) but are DIFFERENT objects are treated as two unrelated drivers.
_topology_errors() raises no error in that case; each link silently falls
back to mode B (100% of available torque each) instead of mode A
(torque_split share). See SpurHelicalMeshLink.torque_split and
SpurHelicalGearSystem._driver_groups() below.

Units: torque PROPAGATES and is STORED in N·m end-to-end (that is what
meshing.forces() consumes and returns, and what TorqueLoad now stores).
Force loads are in N and positions in mm; torque/moment is the one
quantity kept in SI (N·m) rather than shop units, since it never needs to
combine directly with a mm-scale lever arm inside this module.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

from axisforge.core.loads import RadialLoad, AxialLoad, TorqueLoad, DistributedRadialLoad
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import GearElement, ShaftSystem
from axisforge.config import TOL_GEOMETRY_mm

# torque agora propaga e é armazenado em N·m de ponta a ponta


# ===========================================================================
# SpurHelicalMeshLink  
# ===========================================================================

class SpurHelicalMeshLink  :
    """
    A single directed mesh from shaft_a (driver) to shaft_b (driven).

    Parameters
    ----------
    shaft_a, shaft_b : ShaftSystem   — driver / driven shaft containers.
    gear_a, gear_b   : GearElement   — the meshing gears on each shaft.
    meshing          : a *GearMeshing object exposing `.al`, `.u` and
                       `.forces(T_in, phi_deg, rotation_dir_in) -> dict`.
    phi_deg          : angular position of the line of centres (a -> b) in the
                       GLOBAL frame [deg], +Y toward +Z. USER INPUT, not
                       derived. Combined with meshing.al to place shaft_b
                       relative to shaft_a.
    torque_split     : None       -> mode B (this driver takes 100% of T).
                       (0, 1]     -> mode A (simultaneous fan-out share).
                       See GearSystem §4.2.

                       CAUTION: mode A only activates when two or more
                       links share the SAME gear_a object (checked by
                       id(), not by gear geometry). To fan out one
                       physical pinion to several wheels, build ONE
                       GearElement for that pinion and pass it as gear_a
                       to every link:

                           ge_z1  = GearElement(z1, role="driver", ...)
                           link_a = SpurHelicalMeshLink(sys1, ge_z1, ..., torque_split=0.7)
                           link_b = SpurHelicalMeshLink(sys1, ge_z1, ..., torque_split=0.3)

                       Passing two separately-constructed GearElement
                       instances for the same physical z1 (even with
                       identical parameters) is NOT detected as fan-out:
                       validate() raises nothing, and each link quietly
                       resolves in mode B, delivering 100% of the
                       available torque through EACH mesh independently
                       — a silent double-counting of torque.
    """

    def __init__(self, shaft_a: ShaftSystem, gear_a: GearElement,
                 shaft_b: ShaftSystem, gear_b: GearElement,
                 meshing, phi_deg: float,
                 torque_split: float | None = None,
                 distribute_loads: bool = False,
                 label: str = ""):
        
        self.shaft_a             = shaft_a
        self.gear_a              = gear_a
        self.shaft_b             = shaft_b
        self.gear_b              = gear_b
        self.meshing             = meshing
        self.phi_deg             = phi_deg % 360.0
        self.torque_split        = torque_split
        self.distribute_loads    = distribute_loads
        self.meshing_load_factor = 1.0    # in this code, there is only 1 load path so it is always equal to 1.0
        self.label               = label

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or "SpurHelicalMeshLink"
        if self.shaft_a is self.shaft_b:
            errors.append(f"{tag}: a mesh cannot connect a shaft to itself")
        if self.torque_split is not None and not (0.0 < self.torque_split <= 1.0):
            errors.append(
                f"{tag}: torque_split must be in (0, 1] when set, "
                f"got {self.torque_split}"
            )
        for attr in ("al", "forces"):
            if not hasattr(self.meshing, attr):
                errors.append(f"{tag}: meshing object lacks required '{attr}'")
        return errors

    def __repr__(self) -> str:
        return (f"SpurHelicalMeshLink({self.shaft_a.name} -> {self.shaft_b.name}, "
                f"phi={self.phi_deg:.1f}°, "
                f"torque_split={self.torque_split!r}, label={self.label!r})")


# ===========================================================================
# SpurHelicalGearSystem
# ===========================================================================

class SpurHelicalGearSystem:
    """
    A single-source DAG of shafts connected by SpurHelicalMeshLink.

    Enforced topology:
      - exactly one source shaft (zero incoming links),
      - no shaft with >1 incoming link (no convergent merge),
      - acyclic (DAG),
      - any GearElement used as driver in >=2 links is a simultaneous fan-out
        and every such link must carry torque_split, summing to 1.0.
    """

    def __init__(self, shafts: list[ShaftSystem], links: list[SpurHelicalMeshLink],
                 label: str = ""):
        self.shafts = shafts
        self.links = links
        self.label = label
        # Populated by resolve() for summary()/inspection.
        self._resolved: dict[int, dict] | None = None

    # ------------------------------------------------------------------
    # Graph helpers (identity-keyed on shaft objects)
    # ------------------------------------------------------------------

    def _incoming_count(self) -> dict[int, int]:
        indeg = {id(s): 0 for s in self.shafts}
        for link in self.links:
            indeg[id(link.shaft_b)] = indeg.get(id(link.shaft_b), 0) + 1
        return indeg

    def _sources(self) -> list[ShaftSystem]:
        indeg = self._incoming_count()
        return [s for s in self.shafts if indeg.get(id(s), 0) == 0]

    def _topological_order(self) -> list[ShaftSystem] | None:
        """Kahn's algorithm over the shaft graph. Returns None on a cycle."""
        shaft_by_id = {id(s): s for s in self.shafts}
        indeg = self._incoming_count()
        adj: dict[int, list[SpurHelicalMeshLink]] = defaultdict(list)
        for link in self.links:
            adj[id(link.shaft_a)].append(link)

        queue = deque(sid for sid, d in indeg.items() if d == 0)
        order: list[ShaftSystem] = []
        indeg_work = dict(indeg)
        while queue:
            sid = queue.popleft()
            order.append(shaft_by_id[sid])
            for link in adj[sid]:
                bid = id(link.shaft_b)
                indeg_work[bid] -= 1
                if indeg_work[bid] == 0:
                    queue.append(bid)
        if len(order) != len(self.shafts):
            return None  # cycle
        return order

    def _driver_groups(self) -> dict[int, list[SpurHelicalMeshLink]]:

        """
        Group links by the *identity* of their driver GearElement.

        Uses id(link.gear_a), i.e. Python object identity — NOT geometric
        or logical equality. Two GearElement instances wrapping the same
        physical SpurHelicalGear are two different keys here unless they
        are literally the same object. This is the single mechanism that
        decides mode A (fan-out, torque_split applies) vs mode B
        (sequential/independent, 100% torque each) in resolve(). See the
        module docstring and SpurHelicalMeshLink.torque_split for the
        consequences of getting this wrong.
        """
        
        groups: dict[int, list[SpurHelicalMeshLink]] = defaultdict(list)
        for link in self.links:
            groups[id(link.gear_a)].append(link)
        return groups

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _axial_alignment_errors(self, tol: float | None = None) -> list[str]:
        """
        Verify gear_a and gear_b of every SpurHelicalMeshLink occupy the same
        global axial (X) position: shaft.shaft_origin_x + gear.position.

        tol : max allowed centre-to-centre offset [mm].
              - explicit value  -> used as-is
              - None + both face widths (b) known (>0) -> half the smaller
                face width (symmetric overlap requirement)
              - None + either b == 0.0 (not yet defined) -> falls back to
                MIN_BEARING_SEPARATION_mm as a purely numerical tolerance,
                NOT an engineering guarantee of real overlap.

        NOTE: shaft_origin_x is NOT propagated automatically by resolve()
        (see SpurHelicalGearSystem module docstring / session notes) — the caller is
        responsible for setting it consistently on every ShaftSystem
        before validate() is meaningful for multi-shaft systems.
        """
        errors: list[str] = []
        tag = self.label or "SpurHelicalGearSystem"

        for link in self.links:
            xa = link.shaft_a.shaft_origin_x + link.gear_a.position
            xb = link.shaft_b.shaft_origin_x + link.gear_b.position

            ba = getattr(link.gear_a.gear, "b", 0.0)
            bb = getattr(link.gear_b.gear, "b", 0.0)

            if tol is not None:
                allowed = tol
            else:
                allowed = TOL_GEOMETRY_mm

            offset = abs(xa - xb)
            if offset > allowed:
                errors.append(
                    f"{tag}: {link.label or 'mesh'} axial misalignment — "
                    f"gear_a global X={xa:.3f} mm ('{link.shaft_a.name}'), "
                    f"gear_b global X={xb:.3f} mm ('{link.shaft_b.name}'), "
                    f"offset={offset:.3f} mm > allowed {allowed:.3f} mm. "
                    f"Check shaft_origin_x and gear.position on both shafts."
                )
        return errors

    def _topology_errors(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or "SpurHelicalGearSystem"

        for link in self.links:
            errors.extend(f"{tag}: {e}" for e in link.validate())
            if link.shaft_a not in self.shafts:
                errors.append(f"{tag}: link driver shaft not in shafts list")
            if link.shaft_b not in self.shafts:
                errors.append(f"{tag}: link driven shaft not in shafts list")

        # exactly one source
        sources = self._sources()
        if len(sources) == 0:
            errors.append(
                f"{tag}: no source shaft (every shaft has an incoming link — "
                f"likely a cycle)"
            )
        elif len(sources) > 1:
            names = ", ".join(s.name for s in sources)
            errors.append(
                f"{tag}: {len(sources)} source shafts ({names}); exactly one "
                f"required. Disconnected chains belong in separate SpurHelicalGearSystems."
            )

        # no merge
        indeg = self._incoming_count()
        for s in self.shafts:
            if indeg.get(id(s), 0) > 1:
                errors.append(
                    f"{tag}: shaft '{s.name}' has {indeg[id(s)]} incoming links "
                    f"(convergent torque merge is out of scope)"
                )

        # no cycle
        if self._topological_order() is None:
            errors.append(f"{tag}: gear graph contains a cycle (DAG required)")

        # fan-out torque_split consistency
        for group in self._driver_groups().values():
            if len(group) <= 1:
                continue
            driver_name = group[0].gear_a.label or "driver gear"
            if any(link.torque_split is None for link in group):
                errors.append(
                    f"{tag}: '{driver_name}' drives {len(group)} meshes "
                    f"simultaneously — every such link must set torque_split "
                    f"(mode A). Mixed/None splits are inconsistent."
                )
            else:
                total = sum(link.torque_split for link in group)
                if abs(total - 1.0) > 1e-6:
                    errors.append(
                        f"{tag}: torque_split for '{driver_name}' fan-out sums "
                        f"to {total:.6f}, must be 1.0 (±1e-6)"
                    )

        # axial alignment between mesh pairs (global X)
        errors.extend(self._axial_alignment_errors())

        return errors

    def validate(self) -> list[str]:
        errors = list(self._topology_errors())
        for s in self.shafts:
            errors.extend(s.validate())
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError(
                f"SpurHelicalGearSystem '{self.label}' validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Link resolution — two physically distinct fan-out modes
    # ------------------------------------------------------------------

    def _resolve_link_mode_B(self, link: SpurHelicalMeshLink, T_in: float,
                             rotation_dir_in: int) -> dict:
        """Sequential / reuse: the driver delivers the FULL available torque."""
        return link.meshing.forces(T_in, link.phi_deg, rotation_dir_in)

    def _resolve_link_mode_A(self, link: SpurHelicalMeshLink, T_total: float,
                             rotation_dir_in: int) -> dict:
        """Simultaneous fan-out: this mesh takes torque_split of the total."""
        T_in = T_total * link.torque_split
        return link.meshing.forces(T_in, link.phi_deg, rotation_dir_in)

    # ------------------------------------------------------------------
    # Force dict -> Load objects
    # ------------------------------------------------------------------

    def _forces_to_loads(self, F: dict, gear: GearElement, side: str,
                        T_in_nm: float, distribute: bool = False) -> list:
        """
        Build the mesh loads for one side of a mesh.

        distribute=False (default): Ft and Fr as RadialLoad at gear.position.
        distribute=True           : Ft and Fr as DistributedRadialLoad over
                                    [position - b/2, position + b/2], using
                                    b from the gear geometry. Falls back to
                                    RadialLoad when b == 0.0 (face width not set).
        Fa (axial thrust) and TorqueLoad are always point loads.
        """
        suffix   = "_driver" if side == "driver" else "_driven"
        pos      = gear.position
        b        = getattr(gear.gear, "b", 0.0)
        loads: list = []

        if distribute and b > 0.0:
            loads += self._radial_as_distributed(F, gear, suffix, pos, b)
        else:
            loads += self._radial_as_point(F, gear, suffix, pos)

        # axial thrust — always a point load
        Fa = F.get("Fa", 0.0)
        if Fa != 0.0:
            sign = 1.0 if side == "driver" else -1.0
            loads.append(AxialLoad(pos, sign * Fa,
                                label=f"{gear.label or 'mesh'}:Fa",
                                source="gear_mesh"))

        # torque flow — always a point load at mesh position
        T_flow_nm = +T_in_nm if side == "driver" else -F["T_out"]
        loads.append(TorqueLoad(pos, T_flow_nm,
                                label=f"{gear.label or 'mesh'}:T",
                                source="gear_mesh"))
        return loads


    def _radial_as_point(self, F: dict, gear: GearElement,
                        suffix: str, pos: float) -> list:
        """RadialLoad for Ft and Fr (original behaviour)."""
        return [
            RadialLoad(pos, F["Fr"], F[f"theta_Fr{suffix}"],
                    label=f"{gear.label or 'mesh'}:Fr",
                    source="gear_mesh"),
            RadialLoad(pos, F["Ft"], F[f"theta_Ft{suffix}"],
                    label=f"{gear.label or 'mesh'}:Ft",
                    source="gear_mesh"),
        ]


    def _radial_as_distributed(self, F: dict, gear: GearElement,
                                suffix: str, pos: float, b: float) -> list:
        """
        DistributedRadialLoad for Ft and Fr over [pos - b/2, pos + b/2].

        magnitude is the signed scalar from forces() — constant over the
        face width (uniform distribution). theta is the constant angle from
        forces() for this side. Both can be overridden by the caller by
        constructing DistributedRadialLoad directly with a Callable.
        """
        x_lo = pos - b / 2.0
        x_hi = pos + b / 2.0
        return [
            DistributedRadialLoad(
                x_lo, x_hi,
                magnitude=F["Fr"],
                theta_deg=F[f"theta_Fr{suffix}"],
                label=f"{gear.label or 'mesh'}:Fr",
                source="gear_mesh",
            ),
            DistributedRadialLoad(
                x_lo, x_hi,
                magnitude=F["Ft"],
                theta_deg=F[f"theta_Ft{suffix}"],
                label=f"{gear.label or 'mesh'}:Ft",
                source="gear_mesh",
            ),
        ]

    # ------------------------------------------------------------------
    # resolve
    # ------------------------------------------------------------------

    def resolve(self, P: float, rpm: float, rotation_dir_source: int,
                source_position: tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Propagate power P [W] at speed rpm from the unique source shaft through
        the DAG, injecting the resulting mesh loads onto every ShaftSystem.

        Torque propagates in N·m; positions in mm; loads land via
        ShaftSystem.set_gear_loads (idempotent, so re-resolve is safe).
        """
        topo_errors = self._topology_errors()
        if topo_errors:
            raise ValueError(
                "SpurHelicalGearSystem.resolve: invalid topology:\n"
                + "\n".join(f"  - {e}" for e in topo_errors)
            )
        if rpm <= 0.0:
            raise ValueError(f"resolve: rpm must be > 0, got {rpm}")
        if rotation_dir_source not in (1, -1):
            raise ValueError(
                f"resolve: rotation_dir_source must be +1 or -1, got {rotation_dir_source}"
            )

        omega = rpm * 2.0 * math.pi / 60.0
        T_source = P / omega  # N·m

        (source_shaft,) = self._sources()
        source_shaft.shaft_position = source_position

        order = self._topological_order()
        rank = {id(s): i for i, s in enumerate(order)}
        link_order = sorted(self.links, key=lambda lk: rank[id(lk.shaft_a)])

        driver_groups = self._driver_groups()

        T_out_of: dict[int, float] = {id(source_shaft): T_source}
        rot_of: dict[int, int] = {id(source_shaft): rotation_dir_source}
        loads_by_shaft: dict[int, list] = defaultdict(list)

        for link in link_order:
            a_id, b_id = id(link.shaft_a), id(link.shaft_b)
            T_avail = T_out_of[a_id]            # N·m available at the driver
            rot_in = rot_of[a_id]

            group = driver_groups[id(link.gear_a)]
            if len(group) == 1:
                F = self._resolve_link_mode_B(link, T_avail, rot_in)
                T_in_used = T_avail
            else:
                F = self._resolve_link_mode_A(link, T_avail, rot_in)
                T_in_used = T_avail * link.torque_split

            # place shaft_b relative to shaft_a along the line of centres
            y_a, z_a = link.shaft_a.shaft_position
            al = link.meshing.al
            phi = math.radians(link.phi_deg)
            link.shaft_b.shaft_position = (
                y_a + al * math.cos(phi),
                z_a + al * math.sin(phi),
            )

            T_out_of[b_id] = F["T_out"]
            rot_of[b_id] = F["rotation_dir_out"]

            loads_by_shaft[a_id] += self._forces_to_loads(
                F, link.gear_a, side="driver", T_in_nm=T_in_used,
                distribute=link.distribute_loads)
            loads_by_shaft[b_id] += self._forces_to_loads(
                F, link.gear_b, side="driven", T_in_nm=T_in_used,
                distribute=link.distribute_loads)

        # push loads onto every shaft (empty list clears stale gear_mesh loads)
        for shaft in self.shafts:
            shaft.set_gear_loads(loads_by_shaft.get(id(shaft), []))

        # record the resolved chain for summary()/inspection
        self._resolved = {
            id(s): {
                "name": s.name,
                "T_out_Nm": T_out_of.get(id(s)),
                "rotation_dir": rot_of.get(id(s)),
                "shaft_position": s.shaft_position,
                "rpm": None,  # per-shaft rpm is StaticsSolver's kinematics job
            }
            for s in self.shafts
        }

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or "SpurHelicalGearSystem"
        lines = [f"── {tag} ──────────────────────────────────────────────"]
        sources = self._sources()
        lines.append(
            f"  shafts: {len(self.shafts)}   links: {len(self.links)}   "
            f"source: {sources[0].name if len(sources) == 1 else '??'}"
        )
        if self._resolved is None:
            lines.append("  (not resolved yet — call resolve())")
        else:
            order = self._topological_order() or self.shafts
            for s in order:
                r = self._resolved[id(s)]
                y, z = r["shaft_position"]
                t = r["T_out_Nm"]
                t_str = f"{t:.3f} N·m" if t is not None else "—"
                lines.append(
                    f"    {r['name']:<12} T_out={t_str:<14} "
                    f"rot={r['rotation_dir']!s:<3} "
                    f"pos=(y={y:.2f}, z={z:.2f}) mm"
                )
        lines.append("────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"GearSystem(label={self.label!r}, shafts={len(self.shafts)}, "
                f"links={len(self.links)}, "
                f"resolved={self._resolved is not None})")