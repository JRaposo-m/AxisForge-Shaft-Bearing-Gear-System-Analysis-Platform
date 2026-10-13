"""
Multi-shaft parallel-axis transmission: power flow through a tree of gear meshes and the resulting shaft loads.

SpurHelicalMeshLink is one directed mesh (driver shaft -> driven shaft) with a meshing model and the
angle of the line of centres. SpurHelicalGearSystem is a single-source acyclic graph of such links;
resolve() walks it in topological order, propagating torque, sense of rotation and shaft position,
and returns a loaded copy of each ShaftSystem (ShaftSystem.with_loads) carrying the tooth forces,
the axial thrust and the torques; the shafts given by the user are not modified.

Scope: parallel axes only (spur, helical, internal). No bevel or worm gears, no convergent torque
merge, no multi-source graph.

Fan-out is detected by GearElement IDENTITY, not equality: a shared driver (one pinion meshing with
two wheels) is recognised only if the SAME GearElement object is passed as gear_a to every link
sharing it; see SpurHelicalMeshLink.torque_split.

Units: forces [N], positions [mm], angles [deg] at the interface, power [W], speed [rpm]. Torque
propagates and is stored in N m end to end (what meshing.forces() consumes and returns and what
the TorqueLoad objects created here carry); see the Warnings of core/loads.py (open review item U1).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

from axisforge.core.loads import RadialLoad, AxialLoad, TorqueLoad, DistributedRadialLoad
from axisforge.core.mechanical_system.shaft_system import GearElement, ShaftSystem
from axisforge.config import TOL_GEOMETRY_mm



# ===========================================================================
# SpurHelicalMeshLink  
# ===========================================================================

class SpurHelicalMeshLink  :
    """
    A single directed mesh from shaft_a (driver) to shaft_b (driven).

    Parameters
    ----------
    shaft_a, shaft_b : ShaftSystem
        Driver / driven shaft containers.
    gear_a, gear_b : GearElement
        The meshing gears on each shaft.
    meshing : object
        A *GearMeshing object exposing ``.al`` (centre distance [mm]) and
        ``.forces(T_in [N m], phi_deg [deg], rotation_dir_in) -> dict``. ``gear_a`` must wrap the
        gear on which ``forces()`` takes the torque: ``meshing.gear1`` for SpurHelicalGearMeshing
        (whatever ``driver`` says), the gear named by ``meshing.driver`` for InternalGearMeshing.
        Not checked.
    phi_deg : float
        Angular position of the line of centres (a -> b) in the GLOBAL frame [deg], +Y toward +Z.
        USER INPUT, not derived; stored modulo 360. Combined with meshing.al to place shaft_b
        relative to shaft_a.
    torque_split : float or None
        None -> mode B (this driver takes 100% of T). (0, 1] -> mode A (simultaneous fan-out
        share [-]). Choice of this code (source to find).

        CAUTION: mode A only activates when two or more links share the SAME gear_a object
        (checked by id(), not by gear geometry). To fan out one physical pinion to several
        wheels, build ONE GearElement for that pinion and pass it as gear_a to every link::

            ge_z1  = GearElement(z1, role="driver", ...)
            link_a = SpurHelicalMeshLink(sys1, ge_z1, ..., torque_split=0.7)
            link_b = SpurHelicalMeshLink(sys1, ge_z1, ..., torque_split=0.3)

        Passing two separately-constructed GearElement instances for the same physical z1 (even
        with identical parameters) is NOT detected as fan-out: validate() raises nothing, and
        each link quietly resolves in mode B, delivering 100% of the available torque through
        EACH mesh independently, a silent double-counting of torque.
    distribute_loads : bool
        False: Ft and Fr are point loads at gear.position. True: they are uniform
        DistributedRadialLoad over the face width [mm] of each gear (falls back to point loads
        where the face width is 0).
    label : str
        Reporting tag.
    """

    def __init__(self, shaft_a: ShaftSystem, gear_a: GearElement,
                 shaft_b: ShaftSystem, gear_b: GearElement,
                 meshing, phi_deg: float,
                 torque_split: float | None = None,
                 distribute_loads: bool = False,
                 label: str = ""):
        """Store the arguments; see the class docstring."""

        self.shaft_a             = shaft_a
        self.gear_a              = gear_a
        self.shaft_b             = shaft_b
        self.gear_b              = gear_b
        self.meshing             = meshing
        self.phi_deg             = phi_deg % 360.0
        self.torque_split        = torque_split
        self.distribute_loads    = distribute_loads
        # K_gamma [-]: always 1.0, there is only one load path in this code. Read by the
        # Kv method B/C modules under database/, whose imports are broken.
        self.meshing_load_factor = 1.0
        self.label               = label

    def validate(self) -> list[str]:
        """Return the error messages (distinct shafts, torque_split in (0, 1], meshing interface); empty if valid."""
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
        """
        Text of the link.

        Returns
        -------
        str
            Names of the driver and driven shafts, phi [deg], torque_split [-] and label.
        """
        return (f"SpurHelicalMeshLink({self.shaft_a.name} -> {self.shaft_b.name}, "
                f"phi={self.phi_deg:.1f}°, "
                f"torque_split={self.torque_split!r}, label={self.label!r})")


# ===========================================================================
# SpurHelicalGearSystem
# ===========================================================================

class SpurHelicalGearSystem:
    """
    A single-source DAG of shafts connected by SpurHelicalMeshLink.

    Parameters
    ----------
    shafts : list of ShaftSystem
        Every shaft of the transmission.
    links : list of SpurHelicalMeshLink
        The meshes, each directed driver -> driven.
    label : str
        Reporting tag.

    Enforced topology:
      - exactly one source shaft (zero incoming links),
      - no shaft with >1 incoming link (no convergent merge),
      - acyclic (DAG),
      - any GearElement used as driver in >=2 links is a simultaneous fan-out
        and every such link must carry torque_split, summing to 1.0.
    """

    def __init__(self, shafts: list[ShaftSystem], links: list[SpurHelicalMeshLink],
                 label: str = ""):
        """Store the arguments; see the class docstring."""
        self.shafts = shafts
        self.links = links
        self.label = label
        # Populated by resolve() for summary()/inspection.
        self._resolved: dict[int, dict] | None = None
        # Loaded copies returned by the last resolve(), read by validate().
        self._loaded: tuple[ShaftSystem, ...] | None = None

    # ------------------------------------------------------------------
    # Graph helpers (identity-keyed on shaft objects)
    # ------------------------------------------------------------------

    def _incoming_count(self) -> dict[int, int]:
        """Number of incoming links [-] of each shaft, keyed by ``id(shaft)``."""
        indeg = {id(s): 0 for s in self.shafts}
        for link in self.links:
            indeg[id(link.shaft_b)] = indeg.get(id(link.shaft_b), 0) + 1
        return indeg

    def _sources(self) -> list[ShaftSystem]:
        """Shafts with no incoming link (the candidates for the power source)."""
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

        Parameters
        ----------
        tol : float or None
            Maximum allowed offset between the two gear centres [mm]. An explicit value is used as
            given; None uses TOL_GEOMETRY_mm (1e-9 mm) whatever the face widths are.

        Returns
        -------
        list of str
            One message per link whose offset ``|xa - xb|`` exceeds the tolerance.

        Notes
        -----
        The face widths of both gears are read but not used by the code, so the tolerance does
        not depend on the overlap of the two face widths.

        shaft_origin_x is NOT propagated automatically by resolve(): the caller is responsible for
        setting it consistently on every ShaftSystem before this check is meaningful for
        multi-shaft systems.
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
        """Return the topology error messages: links valid, one source, no merge, no cycle, fan-out splits sum to 1 (+-1e-6), axial alignment."""
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
        """
        Return the topology messages plus those of every ShaftSystem; empty if valid.

        Before resolve() the shafts given by the user are validated; after a resolve(), the
        loaded copies of the LAST resolve(), so that the checks that need the mesh loads still run
        (a helical mesh needs a locating bearing: ShaftSystem.validate()). A load or a bearing
        added to a user's shaft after resolve() is seen only after the next resolve().
        """
        errors = list(self._topology_errors())
        # QUESTION(Q43): after resolve() the loaded copies of the last resolve() are validated, as when resolve() loaded the user's shafts in place; alternatives: resolve() validates the copies itself, or validate() only ever sees the user's shafts.
        for s in (self.shafts if self._loaded is None else self._loaded):
            errors.extend(s.validate())
        return errors

    def validate_or_raise(self) -> None:
        """Raise ValueError listing all messages of validate() if there are any."""
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
        """Mode B (sequential / reuse): the driver delivers the FULL available torque T_in [N m]; returns the forces() dict."""
        # QUESTION(Q03): forces() takes this torque on the gear ITS OWN rule names (SpurHelicalGearMeshing: gear1, the smaller working diameter, whatever `driver` says; InternalGearMeshing: the gear named by `driver`), which is not necessarily gear_a of the link.
        return link.meshing.forces(T_in, link.phi_deg, rotation_dir_in)

    def _resolve_link_mode_A(self, link: SpurHelicalMeshLink, T_total: float,
                             rotation_dir_in: int) -> dict:
        """Mode A (simultaneous fan-out): this mesh takes torque_split of the total torque T_total [N m]; returns the forces() dict."""
        T_in = T_total * link.torque_split
        # QUESTION(Q03): as in mode B, forces() chooses the gear that takes T_in.
        return link.meshing.forces(T_in, link.phi_deg, rotation_dir_in)

    # ------------------------------------------------------------------
    # Force dict -> Load objects
    # ------------------------------------------------------------------

    def _forces_to_loads(self, F: dict, gear: GearElement, side: str,
                        T_in_nm: float, distribute: bool = False) -> list:
        """
        Build the mesh loads (tooth forces, axial thrust, torque) for one side of a mesh.

        distribute=False (default): Ft and Fr as RadialLoad at gear.position.
        distribute=True           : Ft and Fr as DistributedRadialLoad over
                                    [position - b/2, position + b/2], using
                                    b from the gear geometry. Falls back to
                                    RadialLoad when b == 0.0 (face width not set).
        Fa (axial thrust) and TorqueLoad are always point loads.

        Parameters
        ----------
        F : dict
            Result of meshing.forces(): Ft, Fr, Fa [N], T_out [N m] and the angles theta_* [deg].
        gear : GearElement
            Gear of this side; its position [mm] and face width b [mm] place the loads.
        side : str
            "driver" or "driven".
        T_in_nm : float
            Torque entering the mesh on the driver gear [N m].
        distribute : bool
            See above.

        Returns
        -------
        list
            RadialLoad / DistributedRadialLoad for Fr and Ft [N], an AxialLoad for Fa if Fa != 0
            [N], and one TorqueLoad (+T_in on the driver, -T_out on the driven gear) [N m].

        Notes
        -----
        Angles come from meshing.forces(): the radial force of the driver acts at phi, the radial
        force of the driven gear at phi + 180 deg; the tangential forces at 90 deg to them,
        signed by the sense of rotation (choice of this code, source to find).

        Warnings
        --------
        With SpurHelicalGearMeshing (external pair), as coded, Fr of each gear points from its own
        centre toward the mesh point (driver at phi, driven at phi + 180 deg), i.e. opposite to the
        separating force on that gear (Harris2007_vol1, sec. 4.2.2, Fig. 4.4, PDF p. 95), and Ft
        has the same direction on the driver and on the driven gear (phi + 90 deg x rotation_dir of
        the driver shaft of the link, for both), so the two are not an action-reaction pair. With
        InternalGearMeshing the same angle formulas give opposed Ft on the two gears, and the mesh
        point lies at phi + 180 deg from both centres (centre distance = r2 - r1), so only the Fr
        of the gear on shaft_b points toward it. Whether the stored loads are meant as the forces
        of the mesh on each shaft is a question for the owner (open review item).

        The axial force is applied as an AxialLoad on the shaft axis only: +Fa on the driver, -Fa on
        the driven gear, with no dependence on the hand of the helix. No tilting moment Fa x r
        (r = pitch radius) is created, although Harris2007_vol1, sec. 4.3, states that thrust loads
        "create concentrated moments that act on the shaft" (Eq. (4.28)-(4.29); sec. 4.3.2 writes it
        as T = r_p P_a).
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
        # QUESTION(Q07a): Fa as returned by forces(), Ft tan(beta) (the alternative is F_bt tan(beta_b)).
        Fa = F.get("Fa", 0.0)
        if Fa != 0.0:
            # QUESTION(Q08): +Fa on the driver, -Fa on the driven, whatever the hand of the helix and the sense of rotation.
            sign = 1.0 if side == "driver" else -1.0
            # QUESTION(Q06): applied on the shaft axis; no tilting moment Fa x r_w is created.
            loads.append(AxialLoad(pos, sign * Fa,
                                label=f"{gear.label or 'mesh'}:Fa",
                                source="gear_mesh"))

        # torque flow — always a point load at mesh position
        # QUESTION(Q05): power-flow signs (+T_in on the driver, -T_out on the driven), independent of the sense of rotation.
        T_flow_nm = +T_in_nm if side == "driver" else -F["T_out"]
        loads.append(TorqueLoad(pos, T_flow_nm,
                                label=f"{gear.label or 'mesh'}:T",
                                source="gear_mesh"))
        return loads


    def _radial_as_point(self, F: dict, gear: GearElement,
                        suffix: str, pos: float) -> list:
        """Ft and Fr as RadialLoad [N] at the gear position [mm], with the angles of forces()."""
        # QUESTION(Q05): the angles of forces() are stored as they come (external pair: both Fr toward the mesh, both Ft the same way).
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
        face width (uniform distribution), as a total force [N]. theta is the
        constant angle from forces() for this side [deg]. Both can be overridden
        by the caller by constructing DistributedRadialLoad directly with a Callable.
        Uniform distribution over the face width: choice of this code (source to find).
        """
        x_lo = pos - b / 2.0
        x_hi = pos + b / 2.0
        # QUESTION(Q05): same angles of forces() as _radial_as_point (external pair: both Fr toward the mesh, both Ft the same way).
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
                source_position: tuple[float, float] = (0.0, 0.0),
                source_torque_x: float = 0.0,
                sinks: tuple[tuple[ShaftSystem, float], ...] = ()) -> tuple[ShaftSystem, ...]:
        """
        Propagate power P at speed rpm from the unique source shaft through the graph and return
        a loaded copy of every ShaftSystem; the shafts of ``self.shafts`` are not modified.

        Parameters
        ----------
        P : float
            Power entering the source shaft [W]. Not validated.
        rpm : float
            Speed of the source shaft [rpm], > 0.
        rotation_dir_source : int
            Sense of rotation of the source shaft about +X, +1 or -1 [-].
        source_position : tuple of float
            (y, z) global offset of the source shaft axis [mm] (written to the shaft_position of
            its loaded copy); the position of every other shaft follows from it. Unrelated to
            torque.
        source_torque_x : float
            Axial coordinate on the source shaft [mm] (0 = left end, as for gear.position) where
            the external power input (motor, coupling) enters; the TorqueLoad -T_source is placed
            there.
        sinks : tuple of (ShaftSystem, float)
            Output ends: for each ``(shaft, x)``, with ``shaft`` one of ``self.shafts`` (the
            user's shaft, not a returned copy) and x the axial coordinate on it [mm], a TorqueLoad
            equal to minus the sum of ALL the TorqueLoads of that shaft [N m] is added at x [mm]
            on its loaded copy, so that the torques about the shaft axis sum to zero (equilibrium
            of torques about the axis; no library source). The sum includes the TorqueLoads the
            user put on the shaft and those of the mesh; ExternalMoment (bending) is not a torque
            about the axis and is not included. The load is labelled ``<shaft name>:T_out`` with
            source "gear_mesh". Being the residue of the stored torque loads, it inherits their
            signs and magnitudes (see Warnings). On a shaft whose torques already sum to zero
            (the source shaft, a shaft already closed by a user load, a second sink on the same
            shaft) the added load is zero or rounding residue. Default ``()``: no shaft is closed
            (the driven shaft keeps the net torque -T_out, as before this argument existed).

        Returns
        -------
        tuple of ShaftSystem
            The loaded copies (``ShaftSystem.with_loads``), in the order of ``self.shafts``: the
            loads of the user's shaft followed by the mesh loads, and the position of the axis.
            They share shaft, bearings and gear elements with the user's shafts. Resolving the
            SAME system again gives new copies with equal loads. A system built from the returned
            copies and resolved would add the mesh loads a second time (the copies carry them as
            ordinary loads).

        Raises
        ------
        ValueError
            If the topology is invalid, rpm <= 0, rotation_dir_source is not +1 or -1, or a sink
            shaft is not in ``self.shafts``.

        Notes
        -----
        ``omega = rpm * 2 pi / 60`` [rad/s] and ``T_source = P / omega`` [N m]; no efficiency.
        For each link in topological order of the driver shaft: the available torque is T_out of
        the driver shaft (mode B, a driver gear used by one link) or the available torque times
        torque_split (mode A, one GearElement shared by several links); meshing.forces() returns
        the forces; shaft_b is placed at ``(y_a + al cos(phi), z_a + al sin(phi))`` [mm] with
        al = meshing.al and phi = radians(link.phi_deg); the torque and the sense of rotation of
        shaft_b are F["T_out"] and F["rotation_dir_out"]. Loads are tagged source="gear_mesh" and
        go through ShaftSystem.with_loads (a new object per shaft). The record ``_resolved``
        keeps T_out [N m], sense of rotation and position per shaft, keyed by the id of the
        user's shaft.

        The input torque on the source shaft is a TorqueLoad of -T_source at source_torque_x; with
        the +T_source TorqueLoad placed at the first gear it makes the torque diagram of the source
        shaft equal to -T_source (magnitude T_source) from the input up to the gear and zero
        beyond it.

        Sign convention (SpurHelicalGearMeshing): driver gear +T_in, driven gear -T_out, power
        input -T_source (net torque of the source shaft is zero; the driven shaft nets -T_out and is
        closed by a sink or by a load of the user). With InternalGearMeshing T_out = T_in u_eff is negative (u < 0), so
        the TorqueLoad on the driven gear is +abs(T_out) and the T_out recorded in ``_resolved``
        for that shaft, and passed on to the next mesh, is negative.

        Assumptions: ideal mesh (T_out = T1 u, no losses), mesh forces independent of the shaft
        and bearing deflections (uncoupled, unlike Mayeux2002, Eq. (1) and Eq. (5), where the
        load distribution depends on the misalignment), forces applied on the shaft axis.

        Warnings
        --------
        Per-shaft rpm is not computed: ShaftSystem.speed_rpm is not set and the ``rpm`` entry of
        ``_resolved`` is None.

        SpurHelicalGearMeshing.forces() takes the torque as acting on gear1 with the working pitch
        diameter 2 al/(u+1), u = max(z)/min(z), and returns T_out = T1 u (spurhelical_meshing.py);
        it does not read ``meshing.driver``. The results are a consistent power balance only when
        gear_a is the gear on which ``forces()`` takes the torque (see ``gear_a``) AND, for
        SpurHelicalGearMeshing, gear1 is the pinion; this is not checked. With
        SpurHelicalGearMeshing the sense of rotation is reversed at each mesh; InternalGearMeshing
        keeps it.

        The signs of the TorqueLoad objects do not depend on the sense of rotation (power-flow
        signs: + on a driver gear, - at an entry), whereas the Ft directions do. For an external
        pair the moment about the shaft axis of the stored Ft, taken at the mesh point, has the
        sign of the TorqueLoad of that gear only when the driver shaft of that link turns in the +1
        sense; in a train the sense alternates, so no value of rotation_dir_source makes every
        mesh consistent (open review item).

        The axial force of a helical mesh creates no tilting moment (see _forces_to_loads).

        References
        ----------
        .. [1] Harris2007_vol1, Eq. (4.13) and Eq. (10.35) (power = torque x speed;
               ``H [W] = 0.001 M [N mm] omega [rad/s]``).

        (source to find) for: the angles of Fr and Ft on driver and driven, the placement of
        shaft_b from the centre distance, T_out = T1 u, the reversal of the sense of rotation and
        the torque split between meshes.
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
        T_source = P / omega  # [N m]; P in W, omega in rad/s

        (source_shaft,) = self._sources()
        # (y, z) of each shaft axis [mm], written on the copies only
        position_of: dict[int, tuple[float, float]] = {id(source_shaft): source_position}

        order = self._topological_order()
        rank = {id(s): i for i, s in enumerate(order)}
        link_order = sorted(self.links, key=lambda lk: rank[id(lk.shaft_a)])

        driver_groups = self._driver_groups()

        T_out_of: dict[int, float] = {id(source_shaft): T_source}
        rot_of: dict[int, int] = {id(source_shaft): rotation_dir_source}
        loads_by_shaft: dict[int, list] = defaultdict(list)

        # external power input on the source shaft itself: T = -T_source, the same sign as
        # the entry side of a driven gear in _forces_to_loads, so it cancels the +T_source
        # placed at the first gear.
        # QUESTION(Q05): -T_source whatever rotation_dir_source (power-flow sign).
        loads_by_shaft[id(source_shaft)].append(
            TorqueLoad(source_torque_x, -T_source,
                       label=f"{source_shaft.name}:T_in",
                       source="gear_mesh")
        )

        for link in link_order:
            a_id, b_id = id(link.shaft_a), id(link.shaft_b)
            T_avail = T_out_of[a_id]            # [N m] available at the driver
            rot_in = rot_of[a_id]

            group = driver_groups[id(link.gear_a)]
            if len(group) == 1:
                F = self._resolve_link_mode_B(link, T_avail, rot_in)
                T_in_used = T_avail
            else:
                F = self._resolve_link_mode_A(link, T_avail, rot_in)
                T_in_used = T_avail * link.torque_split

            # place shaft_b relative to shaft_a along the line of centres [mm] (phi from +Y toward +Z)
            y_a, z_a = position_of[a_id]
            al = link.meshing.al
            phi = math.radians(link.phi_deg)
            position_of[b_id] = (
                y_a + al * math.cos(phi),
                z_a + al * math.sin(phi),
            )

            # QUESTION(Q03): T_out = T u as returned by forces() becomes the torque available on shaft b.
            T_out_of[b_id] = F["T_out"]
            rot_of[b_id] = F["rotation_dir_out"]

            loads_by_shaft[a_id] += self._forces_to_loads(
                F, link.gear_a, side="driver", T_in_nm=T_in_used,
                distribute=link.distribute_loads)
            loads_by_shaft[b_id] += self._forces_to_loads(
                F, link.gear_b, side="driven", T_in_nm=T_in_used,
                distribute=link.distribute_loads)

        # QUESTION(Q16): the output torque is closed here only where the caller names a sink (shaft, x); default: open end, as before (review question 5; position of a gear on its shaft: question 7). The sink is the residue of ALL stored torque loads, so it inherits Q03 and Q05; its label and source differ from the hand-made "output_coupling" / "user" load of the studies.
        shaft_ids = [id(s) for s in self.shafts]
        for sink_shaft, sink_x in sinks:
            if id(sink_shaft) not in shaft_ids:
                raise ValueError(f"resolve: sink shaft {sink_shaft.name!r} is not in this system")
            # closure of the shaft: minus the sum of its torque loads [N m], user loads included
            T_open = sum(ld.magnitude
                         for ld in sink_shaft.torque_loads + loads_by_shaft[id(sink_shaft)]
                         if isinstance(ld, TorqueLoad))
            loads_by_shaft[id(sink_shaft)].append(
                TorqueLoad(sink_x, -T_open, label=f"{sink_shaft.name}:T_out", source="gear_mesh")
            )

        # one loaded copy per shaft, in the order of self.shafts; the user's shafts keep their
        # own loads and position
        # QUESTION(Q09): speed_rpm is not passed, so every copy keeps the speed of the user's shaft (0.0 by default); the alternative is rpm on the source and rpm z_a/z_b down the chain.
        loaded = tuple(
            s.with_loads(loads_by_shaft.get(id(s), []),
                         shaft_position=position_of.get(id(s), s.shaft_position))
            for s in self.shafts
        )

        # record the resolved chain for summary()/inspection
        self._resolved = {
            id(s): {
                "name": s.name,
                "T_out_Nm": T_out_of.get(id(s)),
                "rotation_dir": rot_of.get(id(s)),
                "shaft_position": copy.shaft_position,
                "rpm": None,  # per-shaft rpm is not computed anywhere
            }
            for s, copy in zip(self.shafts, loaded)
        }
        self._loaded = loaded
        return loaded

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line text summary: shafts, links, source and, once resolved, T_out [N m], rotation sense and position [mm] per shaft."""
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
        """
        Text of the system.

        Returns
        -------
        str
            Starts with the class label "GearSystem(" and gives label, number of shafts, number of
            links and whether resolve() has been run.
        """
        return (f"GearSystem(label={self.label!r}, shafts={len(self.shafts)}, "
                f"links={len(self.links)}, "
                f"resolved={self._resolved is not None})")