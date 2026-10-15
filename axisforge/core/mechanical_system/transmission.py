"""
Power path of a gear transmission, written by the user, and the shaft loads of its meshes.

``PairStage`` is one written link of the path: a cylindrical gear pair, the shaft and the gear
element on each side, and the angle of the line of centres. ``Transmission`` is the list of
shafts, the ordered list of stages (the path, as written: nothing is discovered or reordered),
the source shaft and the output ends. ``Transmission.resolve`` walks the path once, carrying
torque, speed, sense of rotation and axis position from shaft to shaft, and returns a
``ResolvedTransmission`` whose shafts are NEW loaded copies (``ShaftSystem.with_loads``); the
shafts given by the user are not modified. ``mesh_forces_to_shaft_loads`` is the only place where a
mesh force becomes shaft loads (D-0010 item 19).

Scope: parallel axes, cylindrical pairs (spur, helical, internal). Each shaft receives power once:
no recombination of power (the shares of a closed power loop depend on stiffness, not on statics).

Units: forces [N], positions [mm], angles [deg] at the interface, power [W], speed [rpm]. Torque
is carried and stored in N m end to end (``mesh_forces`` takes N m; the ``TorqueLoad`` objects
created here carry N m); a bending moment (``ExternalMoment``) is created in N mm, the unit the
shaft FEM consumes (see the Warnings of ``core/loads.py``).

Signs (D-0010 items 12-16; José, 2026-10-15, Q03, Q05, Q47): a torque is signed about +X, the
sense of rotation of a shaft is rot = sign(omega) about +X, and the torque that enters a shaft
(from the source or from the mesh of its input stage) times its angular velocity is the power
that enters it (> 0). The mesh loads are those of ``MeshForces``: separating radial force,
action = reaction, torque of each gear = moment of its mesh force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from axisforge.config import TOL_GEOMETRY_mm
from axisforge.core.loads import AxialLoad, DistributedRadialLoad, ExternalMoment, RadialLoad, TorqueLoad
from axisforge.core.machine_elements.gears.base import GearPair, MeshForces
from axisforge.core.mechanical_system.shaft_system import GearElement, ShaftSystem

# Relative residual below which a closure check passes [-]: rounding of a sum of a few floats.
_CLOSURE_RTOL = 1e-9


def _closure(check: str, residual: float, scale: float) -> tuple[str, float, bool]:
    """One closure check ``(check, residual, ok)``: ok when ``residual <= 1e-9 scale`` (same unit)."""
    return check, float(residual), bool(residual <= _CLOSURE_RTOL * scale)


def _load_sums(loads: list) -> tuple[float, float, float, float]:
    """Resultant of a list of loads: force (F_x, F_y, F_z) [N] and torque about +X [N m]."""
    Fx = Fy = Fz = T = 0.0
    for ld in loads:
        if isinstance(ld, RadialLoad):
            Fy, Fz = Fy + ld.Fy, Fz + ld.Fz
        elif isinstance(ld, DistributedRadialLoad):
            Fy, Fz = Fy + ld.resultant_Fy, Fz + ld.resultant_Fz
        elif isinstance(ld, AxialLoad):
            Fx += ld.magnitude
        elif isinstance(ld, TorqueLoad):
            T += ld.magnitude
    return Fx, Fy, Fz, T


# ===========================================================================
# The one crossing: mesh force -> shaft loads
# ===========================================================================

def mesh_forces_to_shaft_loads(mesh: MeshForces, side: int, gear: GearElement, theta_c_deg: float,
                               distribute: bool = False) -> list:
    """
    Shaft loads of one gear of a mesh: the force of the mesh on that gear, moved to the shaft axis.

    The mesh pushes on the gear at the pitch point C, at the signed distance rho from its axis
    along the line of centres. Statically, a force at C equals the same force on the shaft axis
    (at the axial station of the gear) plus the moment r x F of the force about the axis, r being
    the radius vector from the axis to C (D-0010 item 19; ``units_and_conventions.md``, C.6). The
    axial part of that moment is the torque of the mesh on the shaft, its transverse part the
    bending moment of the axial force F_a at the radius r_w. This is the only function that builds
    shaft loads from a mesh.

    Parameters
    ----------
    mesh : MeshForces
        Forces of the mesh (``CylindricalFamily.mesh_forces``), in the mesh frame (e_c, e_t, e_x).
    side : int
        1 or 2: which gear of the pair this is (``mesh.on_1`` / ``rho_mm[0]`` or ``on_2`` /
        ``rho_mm[1]``).
    gear : GearElement
        Gear of this side; its position [mm] (centre of the face width, from the left end of the
        shaft) and face width b [mm] place the loads; its label names them.
    theta_c_deg : float
        Direction of e_c (from the axis of gear 1 to the axis of gear 2) in the GLOBAL frame
        [deg], from +Y toward +Z. e_t = e_x x e_c lies at theta_c + 90 deg (e_x = +X).
    distribute : bool
        False: the two transverse components are point loads (``RadialLoad``) at the gear position.
        True: each is a uniform ``DistributedRadialLoad`` over [position - b/2, position + b/2]
        [mm], its magnitude the total force [N] (uniform over the face width: choice of this code,
        no library source); point loads when b = 0.

    Raises
    ------
    ValueError
        If ``side`` is not 1 or 2.

    Returns
    -------
    list
        In this order, all tagged ``source="gear_mesh"``: ``<label>:Fr``, the radial component
        F_c [N] (along e_c: a separating force); ``<label>:Ft``, the tangential component F_t [N]
        (along e_t); an ``AxialLoad`` ``<label>:Fa`` = F_x [N] (signed along +X) if F_x != 0; a
        ``TorqueLoad`` ``<label>:T`` = rho F_t / 1000 [N m]; an ``ExternalMoment`` ``<label>:M``,
        the bending moment of F_x [N mm], if F_x != 0. Every magnitude of a transverse load is
        >= 0: the sign is in its angle.

    Notes
    -----
    With F = (F_c, F_t, F_x) the force on this gear in the mesh frame and r = rho e_c [mm]::

        transverse force on the axis:  F_c at theta_c (theta_c + 180 if F_c < 0)
                                       F_t at theta_c + 90 (theta_c - 90 if F_t < 0)
        axial force:                   F_x along +X
        torque about +X  [N m]:        (r x F) . e_x / 1000 = rho F_t / 1000
        bending moment   [N mm]:       r x (F_x e_x) = -rho F_x e_t, modulus abs(rho F_x) = F_a r_w

    ``ExternalMoment`` as the shaft FEM consumes it (``core/loads.py``, Warnings, and
    ``fem_solvers/assembly/load_assembly/vector_external_forces.py``): a magnitude M at the angle
    theta acts in the bending plane that contains X and theta, its moment VECTOR at theta + 90
    deg. The vector -rho F_x e_t is therefore written as magnitude abs(rho F_x) at theta_c when
    -rho F_x > 0 and at theta_c + 180 when -rho F_x < 0 (its plane is the plane of X and e_c).
    D-0010 rows 21-22 (planes and sign of ``ExternalMoment``) are open: this mapping is the one
    the FEM consumes (José, 2026-10-15, answer 5), checked by the bearing reactions and the
    equilibrium of each shaft in ``fem_solvers/tests/test_two_stage_transmission.py``.

    Conventions (D-0010 items 1-3): global frame (X, Y, Z), the shaft along +X; a transverse
    direction is the angle theta from +Y toward +Z, right-hand about +X (F_y = F cos(theta),
    F_z = F sin(theta)); positions [mm] from the left end of the shaft.

    Assumptions: ideal mesh (no losses), mesh forces independent of the deflections of shafts
    and bearings, one force at C, at mid face width.

    References
    ----------
    .. [1] D-0010 items 1-3, 14-19; vault note ``units_and_conventions.md``, C.4, C.6. No library
           source for the directions (statics); separating radial force: Harris2007_vol1, 4.2.2,
           Fig. 4.4; moment of the axial force: Harris2007_vol1, 4.3, Eq. (4.28)-(4.29) (one
           reader).
    """
    if side not in (1, 2):
        raise ValueError(f"side must be 1 or 2 (gear 1 or gear 2 of the pair), got {side!r}")
    F_c, F_t, F_x = mesh.on_1 if side == 1 else mesh.on_2
    rho = mesh.rho_mm[side - 1]
    pos = gear.position
    b = getattr(gear.gear, "b", 0.0)
    tag = gear.label or "mesh"

    # a transverse component as a magnitude >= 0 and an angle: the sign turns the angle by 180 deg
    theta_c = theta_c_deg % 360.0
    theta_Fc = theta_c if F_c >= 0.0 else (theta_c + 180.0) % 360.0
    theta_Ft = (theta_c + 90.0) % 360.0 if F_t >= 0.0 else (theta_c - 90.0) % 360.0     # e_t = e_x x e_c
    if distribute and b > 0.0:
        # total force [N] spread uniformly over the face width, centred on the gear position
        x_lo, x_hi = pos - b / 2.0, pos + b / 2.0
        loads: list = [
            DistributedRadialLoad(x_lo, x_hi, magnitude=abs(F_c), theta_deg=theta_Fc, label=f"{tag}:Fr", source="gear_mesh"),
            DistributedRadialLoad(x_lo, x_hi, magnitude=abs(F_t), theta_deg=theta_Ft, label=f"{tag}:Ft", source="gear_mesh"),
        ]
    else:
        loads = [
            RadialLoad(pos, abs(F_c), theta_Fc, label=f"{tag}:Fr", source="gear_mesh"),
            RadialLoad(pos, abs(F_t), theta_Ft, label=f"{tag}:Ft", source="gear_mesh"),
        ]
    if F_x != 0.0:
        loads.append(AxialLoad(pos, F_x, label=f"{tag}:Fa", source="gear_mesh"))
    # torque of the mesh on the shaft, (r x F) . e_x: [mm] [N] / 1000 -> [N m]
    loads.append(TorqueLoad(pos, rho * F_t / 1000.0, label=f"{tag}:T", source="gear_mesh"))
    if F_x != 0.0:
        # bending moment -rho F_x e_t [N mm]; as the FEM consumes ExternalMoment its vector lies at theta + 90
        m_t = -rho * F_x
        loads.append(ExternalMoment(pos, abs(m_t), theta_c if m_t > 0.0 else (theta_c + 180.0) % 360.0,
                                    label=f"{tag}:M", source="gear_mesh"))
    return loads


# ===========================================================================
# PairStage
# ===========================================================================

@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class PairStage:
    """
    One written link of the power path: a cylindrical gear pair between two shafts.

    ``shaft_a`` brings the power, ``shaft_b`` receives it; ``gear_a`` (on ``shaft_a``) is the
    driving gear (D-0010 item 11: the driver follows from the written path, not from the pair;
    Q03, José 2026-10-15). It is the only implementation of the stage contract of
    ``Transmission`` (see there), with ``shafts = (shaft_a, shaft_b)``.

    Parameters
    ----------
    pair : GearPair
        The pair (``CylindricalFamily``): its working centre distance ``al`` [mm] places
        ``shaft_b``, its family gives the forces (``mesh_forces``).
    shaft_a, shaft_b : ShaftSystem
        The shaft that brings the power and the shaft that receives it.
    gear_a, gear_b : GearElement
        The gear of the pair that sits on ``shaft_a`` (the driver) and the one on ``shaft_b``:
        their positions [mm] and labels place and name the mesh loads; their signed tooth numbers
        give the torque and speed of ``shaft_b``.
    phi_deg : float
        Angle of the line of centres from the axis of ``shaft_a`` to the axis of ``shaft_b`` in
        the GLOBAL frame [deg], from +Y toward +Z. An input, not derived; stored modulo 360.
    share : float
        Part of the torque entering ``shaft_a`` that this stage takes, in (0, 1] [-]. Default
        1.0: the whole torque (see Q40 in ``Transmission``).
    distribute_loads : bool
        False: F_c and F_t are point loads at the gear position. True: uniform loads over the face
        width of each gear [mm] (see ``mesh_forces_to_shaft_loads``).
    label : str
        Reporting tag.

    Notes
    -----
    Not validated at construction: ``Transmission`` runs ``validate()`` of every stage when it is
    built and again at each ``resolve``.
    """

    pair: GearPair
    shaft_a: ShaftSystem
    gear_a: GearElement
    shaft_b: ShaftSystem
    gear_b: GearElement
    phi_deg: float
    share: float = 1.0
    distribute_loads: bool = False
    label: str = ""

    def __post_init__(self):
        """Store ``phi_deg`` modulo 360, as the angles of the loads are."""
        object.__setattr__(self, "phi_deg", self.phi_deg % 360.0)

    @property
    def shafts(self) -> tuple[ShaftSystem, ShaftSystem]:
        """``(shaft_a, shaft_b)``: the input shaft first (stage contract)."""
        return self.shaft_a, self.shaft_b

    def validate(self) -> list[str]:
        """
        Return the error messages of the stage; empty if valid.

        Checked (validation level L6; Q45, José 2026-10-15): ``gear_a`` and ``gear_b`` wrap the
        two gears of the pair; each gear element is on its shaft; the two gears occupy the same
        global axial position ``shaft_origin_x + gear.position`` within ``TOL_GEOMETRY_mm`` (1e-9 mm,
        whatever the face widths: the overlap of the two face widths is not used).
        ``shaft_origin_x`` is not propagated by ``resolve``: it is the caller's to set on every
        shaft.
        """
        errors: list[str] = []
        tag = self.label or "PairStage"
        pair = self.pair
        gears = (self.gear_a.gear, self.gear_b.gear)
        if not ((gears[0] is pair.gear1 and gears[1] is pair.gear2)
                or (gears[0] is pair.gear2 and gears[1] is pair.gear1)):
            errors.append(f"{tag}: gear_a and gear_b are not the two gears of the pair")
        for shaft, element in ((self.shaft_a, self.gear_a), (self.shaft_b, self.gear_b)):
            if not any(element is ge for ge in shaft.gears):
                errors.append(f"{tag}: gear element '{element.label}' is not on shaft '{shaft.name}'")
        xa = self.shaft_a.shaft_origin_x + self.gear_a.position
        xb = self.shaft_b.shaft_origin_x + self.gear_b.position
        offset = abs(xa - xb)
        if offset > TOL_GEOMETRY_mm:
            errors.append(
                f"{tag}: axial misalignment — "
                f"gear_a global X={xa:.3f} mm ('{self.shaft_a.name}'), "
                f"gear_b global X={xb:.3f} mm ('{self.shaft_b.name}'), "
                f"offset={offset:.3f} mm > allowed {TOL_GEOMETRY_mm:.3f} mm. "
                f"Check shaft_origin_x and gear.position on both shafts."
            )
        return errors

    def resolve(self, T_in_Nm: float, n_in_rpm: float, rot_in: int,
                position_in_mm: tuple[float, float]) -> dict:
        """
        Loads, torque, speed, sense of rotation and axis position of the two shafts of the stage.

        Parameters
        ----------
        T_in_Nm : float
            Torque that ``shaft_a`` applies to ``gear_a`` about +X [N m], signed: the torque that
            enters ``shaft_a`` times the share of this stage. The stage drives when
            T_in_Nm rot_in > 0.
        n_in_rpm : float
            Speed of ``shaft_a`` [rpm], a magnitude.
        rot_in : int
            Sense of rotation of ``shaft_a`` about +X, +1 or -1 [-].
        position_in_mm : tuple of float
            (y, z) of the axis of ``shaft_a`` in the global frame [mm].

        Returns
        -------
        dict
            The stage result of the contract of ``Transmission``; every per-shaft tuple is in
            the order ``(shaft_a, shaft_b)``: ``loads`` (lists of loads, from
            ``mesh_forces_to_shaft_loads``), ``torque_Nm`` (T_in, torque that the mesh delivers
            to ``shaft_b``) [N m], ``rotation`` (+1 or -1), ``speed_rpm`` [rpm], ``position_mm``
            (y, z) [mm]; ``meshes``: one ``MeshForces``; ``checks``: the closure checks of the
            mesh, ``(check, residual, ok)``.

        Notes
        -----
        ``shaft_b`` is placed at the working centre distance a_w along phi:
        ``(y_a + a_w cos(phi), z_a + a_w sin(phi))`` [mm]. Across the mesh, with signed z
        (D-0010 item 13; ISO_21771_2007, 5.2.2, Eq. (53)):

            n_b = n_a abs(z_a / z_b)            rot_b = -rot_a sign(z_a z_b)
            T_b = T_a z_b / z_a                 (torque that shaft_b applies to gear_b)

        and the torque that the mesh delivers to ``shaft_b`` is -T_b, the moment of the mesh force
        on ``gear_b`` (an external stage reverses the sign, an internal stage keeps it). The
        family contract takes the torque on gear 1 (``mesh_forces(pair, T_1, rot_1)``): when
        ``gear_a`` is gear 2, T_1 = T_a z_1 / z_2 and rot_1 = -rot_a sign(z_1 z_2).

        Closure checks (statics on the loads just built; no library source):

        - action = reaction: the forces on the two gears sum to zero [N];
        - torque = moment of the mesh force, per gear: the stored torque equals the moment
          about the shaft axis of the stored transverse force applied at the pitch point [N m];
        - power of the mesh: the torques of the mesh on the two shafts, each times the signed
          angular velocity of its shaft, sum to zero [W] (an ideal mesh neither adds nor removes
          power);
        - separating radial force, per gear: the stored radial force pushes an external gear
          toward its own axis and a ring outward (compressive contact) [N].
        """
        pair = self.pair
        z_a, z_b = self.gear_a.gear.z, self.gear_b.gear.z
        a_is_gear2 = self.gear_a.gear is pair.gear2

        # place shaft_b relative to shaft_a along the line of centres [mm] (phi from +Y toward +Z)
        y_a, z_pos = position_in_mm
        al = pair.al
        phi = math.radians(self.phi_deg)
        position_b = (y_a + al * math.cos(phi), z_pos + al * math.sin(phi))

        # kinematics and torque across the mesh, signed z (D-0010 item 13)
        n_out_rpm = n_in_rpm * abs(z_a / z_b)            # no slip at C (ISO_21771_2007 Eq. (53))
        rot_out = -rot_in if z_a * z_b > 0 else rot_in   # an external mesh reverses the sense
        T_b_Nm = T_in_Nm * z_b / z_a                     # torque of shaft_b on gear_b

        # the family takes the torque on GEAR 1: that of gear_b when gear_a is gear 2
        T1_Nm, rot1 = (T_b_Nm, rot_out) if a_is_gear2 else (T_in_Nm, rot_in)
        mesh = pair.family.mesh_forces(pair, T1_Nm, rot1)

        # e_c runs from gear 1 to gear 2: along phi when gear 1 is on shaft_a, the other way when gear 2 is
        theta_c_deg = self.phi_deg + (180.0 if a_is_gear2 else 0.0)
        side_a, side_b = (2, 1) if a_is_gear2 else (1, 2)
        loads_a = mesh_forces_to_shaft_loads(mesh, side_a, self.gear_a, theta_c_deg, self.distribute_loads)
        loads_b = mesh_forces_to_shaft_loads(mesh, side_b, self.gear_b, theta_c_deg, self.distribute_loads)
        T_out_Nm = -T_b_Nm                               # torque the mesh delivers to shaft_b [N m]

        # closure checks on the loads just built
        rho_a, rho_b = mesh.rho_mm[side_a - 1], mesh.rho_mm[side_b - 1]
        e_y, e_z = math.cos(math.radians(theta_c_deg)), math.sin(math.radians(theta_c_deg))
        (Fx_a, Fy_a, Fz_a, T_a), (Fx_b, Fy_b, Fz_b, T_b) = _load_sums(loads_a), _load_sums(loads_b)
        # moment about +X of a transverse force at C: (r x F)_x = r_y F_z - r_z F_y [N mm] -> [N m]
        M_a = rho_a * (e_y * Fz_a - e_z * Fy_a) / 1000.0
        M_b = rho_b * (e_y * Fz_b - e_z * Fy_b) / 1000.0
        omega_a = rot_in * n_in_rpm * 2.0 * math.pi / 60.0      # signed angular velocity [rad/s]
        omega_b = rot_out * n_out_rpm * 2.0 * math.pi / 60.0
        # radial component along the radius from the gear axis to C: < 0 toward the axis; a ring (z < 0) must get > 0
        out_a = math.copysign(1.0, rho_a) * (e_y * Fy_a + e_z * Fz_a) * math.copysign(1.0, z_a)
        out_b = math.copysign(1.0, rho_b) * (e_y * Fy_b + e_z * Fz_b) * math.copysign(1.0, z_b)
        tag = self.label or "PairStage"
        checks = (
            _closure(f"{tag}: action = reaction, |F on gear_a + F on gear_b| [N]",
                     math.hypot(Fx_a + Fx_b, Fy_a + Fy_b, Fz_a + Fz_b), mesh.F_tw_N),
            _closure(f"{tag}: torque of gear_a = moment of its mesh force [N m]", abs(T_a - M_a), abs(T_in_Nm)),
            _closure(f"{tag}: torque of gear_b = moment of its mesh force [N m]", abs(T_b - M_b), abs(T_out_Nm)),
            _closure(f"{tag}: power of the mesh, T_a omega_a + T_b omega_b [W]",
                     abs(T_a * omega_a + T_b * omega_b), abs(T_in_Nm * omega_a)),
            _closure(f"{tag}: radial force separates the gears, gear_a [N]", max(out_a, 0.0), mesh.F_r_N),
            _closure(f"{tag}: radial force separates the gears, gear_b [N]", max(out_b, 0.0), mesh.F_r_N),
        )
        return {"loads": (loads_a, loads_b), "torque_Nm": (T_in_Nm, T_out_Nm), "rotation": (rot_in, rot_out),
                "speed_rpm": (n_in_rpm, n_out_rpm), "position_mm": (position_in_mm, position_b),
                "meshes": (mesh,), "checks": checks}

    def __repr__(self) -> str:
        """Text of the stage: names of the two shafts, phi [deg], share [-] and label."""
        return (f"PairStage({self.shaft_a.name} -> {self.shaft_b.name}, "
                f"phi={self.phi_deg:.1f} deg, share={self.share!r}, label={self.label!r})")


# ===========================================================================
# ResolvedTransmission
# ===========================================================================

@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class ResolvedTransmission:
    """
    Result of ``Transmission.resolve``: the loaded shafts and what each shaft carries.

    A record of rigid-body statics and kinematics; it is the input of the solvers, not a solver
    result. Every per-shaft tuple is in the order of ``Transmission.shafts``.

    Attributes
    ----------
    transmission : Transmission
        The transmission that was resolved (path, source, sinks, label).
    shafts : tuple of ShaftSystem
        NEW loaded copies (``ShaftSystem.with_loads``): the loads of the user's shaft followed
        by the mesh loads, and the position (y, z) of the axis [mm]. They share shaft, bearings
        and gear elements with the user's shafts.
    torque_Nm : tuple of float
        Torque that ENTERS each shaft about +X [N m], signed (D-0010 item 12; José, 2026-10-15,
        answer 9): on the source shaft the torque the source delivers, rot P / omega; on every
        other shaft the torque the mesh of its input stage delivers to it, the moment of the mesh
        force on its gear. Its product with the signed angular velocity of the shaft is the power
        that enters the shaft, > 0 (case 7a: +60.75 and -91.125 N m, senses +1 and -1).
    rotation : tuple of int
        Sense of rotation of each shaft about +X, +1 or -1 [-]: the one given on the source shaft,
        -rot_a sign(z_a z_b) across each mesh (an external mesh reverses it).
    speed_rpm : tuple of float
        Speed of each shaft [rpm], a magnitude: the speed given on the source shaft, then
        n z_a / z_b down the path with the tooth numbers of the gears on the shafts
        (ISO_21771_2007, 5.2.2, Eq. (53)).
    meshes : tuple of MeshForces
        The meshes of the stages, in path order (D-0010): the forces the stored loads are built
        from (``mesh_forces_to_shaft_loads``).
    checks : tuple of (str, float, bool)
        Closure post-conditions ``(check, residual, ok)``, computed on the stored loads; they
        never raise. Per mesh: action = reaction, torque of each gear = moment of its mesh
        force, power of the mesh, separating radial force on each gear. Per shaft: the torque
        loads sum to zero. For the train: the power leaving the end shafts equals the power
        entering. They are NECESSARY conditions, not sufficient: they do not test the magnitudes
        against an independent calculation.

    Notes
    -----
    The speed of every shaft is in ``speed_rpm`` and is also written on its loaded copy,
    ``ShaftSystem.speed_rpm`` (Q09, José 2026-10-15).
    """

    transmission: "Transmission"
    shafts: tuple[ShaftSystem, ...]
    torque_Nm: tuple[float, ...]
    rotation: tuple
    speed_rpm: tuple[float, ...]
    meshes: tuple[MeshForces, ...]
    checks: tuple[tuple[str, float, bool], ...]

    @property
    def label(self) -> str:
        """Reporting tag of the transmission that was resolved."""
        return self.transmission.label

    def summary(self) -> str:
        """Multi-line text: shafts, stages, source and, per shaft in the order of the power path, the
        signed torque that enters it [N m] (``torque_Nm``), sense of rotation and axis position [mm]."""
        tr = self.transmission
        # Q46 (José, 2026-10-15): text of this class (stages; T_in = the signed torque that ENTERS the shaft). ASCII only.
        header = f"-- {tr.label or 'Transmission'} "
        lines = [header + "-" * max(0, 56 - len(header)),
                 f"  shafts: {len(self.shafts)}   stages: {len(tr.path)}   source: {tr.source.name}"]
        # the shafts in the order they receive power: the source, then the output shafts of each stage
        index = {id(s): i for i, s in enumerate(tr.shafts)}
        for i in [index[id(tr.source)]] + [index[id(s)] for stage in tr.path for s in stage.shafts[1:]]:
            y, z = self.shafts[i].shaft_position
            t_str = f"{self.torque_Nm[i]:.3f} N m"
            lines.append(f"    {self.shafts[i].name:<12} T_in={t_str:<14} "
                         f"rot={self.rotation[i]!s:<3} pos=(y={y:.2f}, z={z:.2f}) mm")
        lines.append("-" * 56)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Short text: label, number of shafts and of stages."""
        tr = self.transmission
        return f"ResolvedTransmission(label={self.label!r}, shafts={len(self.shafts)}, stages={len(tr.path)})"


# ===========================================================================
# Transmission
# ===========================================================================

class Transmission:
    """
    Shafts joined by stages along a power path that the user writes.

    Parameters
    ----------
    shafts : sequence of ShaftSystem
        Every shaft of the transmission. Their order is the order of every per-shaft result.
    path : sequence of stages
        The stages in the order of the power flow (a ``PairStage`` each, or any object that
        meets the stage contract below). The path is used as written: it is never reordered and
        nothing is discovered.
    source : ShaftSystem
        The shaft of ``shafts`` that receives the power.
    sinks : tuple of (ShaftSystem, float)
        Output ends: for each ``(shaft, x)``, with ``shaft`` one of ``shafts`` (the user's shaft,
        not a returned copy) and x the axial coordinate on it [mm], ``resolve`` adds on its
        loaded copy a TorqueLoad equal to minus the sum of ALL the TorqueLoads of that shaft
        [N m] at x, so that the torques about the shaft axis sum to zero (equilibrium; no
        library source). The sum includes the TorqueLoads the user put on the shaft and those of
        the mesh; ExternalMoment (bending) is not a torque about the axis and is not included.
        The load is labelled ``<shaft name>:T_out`` with source "gear_mesh"; it is the reaction
        torque of the output (load, coupling) on the shaft. Every output shaft (a shaft that
        feeds no stage) needs one (Q16, José 2026-10-15).
    label : str
        Reporting tag.

    Raises
    ------
    ValueError
        At construction, if the written path is invalid (validation level L6): a message of a
        stage (``PairStage.validate``); a stage whose shafts are not distinct or not in
        ``shafts``; a share outside (0, 1]; shares of the stages leaving one shaft that do not sum
        to 1 (Q40); a stage whose input shaft has no power yet; a shaft that receives power twice;
        a shaft that receives none; an output shaft without a sink (Q16); the source or a sink
        shaft not in ``shafts``.

    Notes
    -----
    Stage contract. ``Transmission`` reads from a stage only: ``shafts``, a tuple of ANY length
    (two or more) of the ``ShaftSystem`` objects it links, the input shaft first; ``share``
    [-]; ``label``; ``validate()`` (list of str); and
    ``resolve(T_in_Nm, n_in_rpm, rot_in, position_in_mm)``, which returns a dict of tuples with
    one entry per shaft of ``stage.shafts``, in that order: ``loads`` (a list of loads each),
    ``torque_Nm`` [N m], ``rotation``, ``speed_rpm`` [rpm], ``position_mm`` ((y, z) [mm]), plus
    ``meshes`` (tuple of ``MeshForces``) and ``checks`` (tuple of ``(check, residual, ok)``).
    The entries of the input shaft are what the stage received; those of the other shafts
    become the state of those shafts. ``PairStage`` is the only implementation; a stage that
    links more than two shafts (a planetary stage) is accepted by this contract and none
    exists. In the same way the members of a ``PlanetaryArrangement`` are a tuple of any length
    (Q44, José 2026-10-15: keep - the stage contract is a documented duck type until the
    planetary stage exists, then a Stage protocol; that class is not written yet).

    Limits: each shaft receives power once (no recombination); one input shaft per stage.

    Each stage takes ``share`` of the torque entering its input shaft; the shares of the stages
    leaving one shaft sum to 1 (power conservation; Q40, José 2026-10-15). FUTURE: a solver for the
    load split of parallel paths (statically indeterminate).
    """

    def __init__(self, shafts, path, source: ShaftSystem,
                 sinks: tuple[tuple[ShaftSystem, float], ...] = (), label: str = ""):
        """Store the arguments and check the written path; see the class docstring."""
        self.shafts = list(shafts)
        self.path = tuple(path)
        self.source = source
        self.sinks = tuple(sinks)
        self.label = label
        self._raise_on_path_errors("Transmission")

    # ------------------------------------------------------------------
    # Validation (level L6: the written path)
    # ------------------------------------------------------------------

    def _path_errors(self) -> list[str]:
        """Return the error messages of the written path (see Raises of the class); empty if valid."""
        errors: list[str] = []
        tag = self.label or "Transmission"
        mine = {id(s) for s in self.shafts}
        if id(self.source) not in mine:
            errors.append(f"{tag}: the source shaft '{self.source.name}' is not in the shafts")
        powered = {id(self.source)}           # shafts that have power when a stage is reached
        for stage in self.path:
            stag = stage.label or type(stage).__name__
            errors.extend(f"{tag}: {e}" for e in stage.validate())
            ids = [id(s) for s in stage.shafts]
            if len(set(ids)) != len(ids):
                errors.append(f"{tag}: {stag}: a stage cannot connect a shaft to itself")
            errors.extend(f"{tag}: {stag}: shaft '{s.name}' is not in the shafts" for s in stage.shafts
                          if id(s) not in mine)
            if not (0.0 < stage.share <= 1.0):
                errors.append(f"{tag}: {stag}: share must be in (0, 1], got {stage.share}")
            if ids[0] not in powered:
                errors.append(f"{tag}: {stag}: its input shaft '{stage.shafts[0].name}' has no power yet; "
                              f"write the stages in the order of the power flow")
            for s in stage.shafts[1:]:
                if id(s) in powered:
                    errors.append(f"{tag}: {stag}: shaft '{s.name}' already has power "
                                  f"(recombination of power is out of scope)")
                powered.add(id(s))
        errors.extend(f"{tag}: shaft '{s.name}' receives no power (it is neither the source nor an "
                      f"output shaft of a stage)" for s in self.shafts if id(s) not in powered)
        # Q40 (José, 2026-10-15): the stages leaving one shaft share its torque: shares summing to 1 conserve power
        for shaft in self.shafts:
            total = sum(stage.share for stage in self.path if stage.shafts[0] is shaft)
            if total and abs(total - 1.0) > 1e-9:
                errors.append(f"{tag}: the shares of the stages leaving shaft '{shaft.name}' sum to {total:.6f}, "
                              f"not 1 (power conservation)")
        # Q16 (José, 2026-10-15): every output shaft (one that feeds no stage) is closed by a sink
        inputs = {id(stage.shafts[0]) for stage in self.path}
        sinks = {id(s) for s, _ in self.sinks}
        errors.extend(f"{tag}: output shaft '{s.name}' has no sink (give it in sinks=((shaft, x),), x the axial "
                      f"position of its output [mm])" for s in self.shafts
                      if id(s) in powered and id(s) not in inputs and id(s) not in sinks)
        errors.extend(f"{tag}: sink shaft {s.name!r} is not in this system" for s, _ in self.sinks
                      if id(s) not in mine)
        return errors

    def _raise_on_path_errors(self, where: str) -> None:
        """Raise ValueError listing the messages of the written path, if any."""
        errors = self._path_errors()
        if errors:
            raise ValueError(f"{where}: invalid power path:\n" + "\n".join(f"  - {e}" for e in errors))

    def validate(self) -> list[str]:
        """
        Return the messages of the written path plus those of every shaft given by the user;
        empty if valid.

        The loaded copies are validated by ``resolve`` itself (Q43, José 2026-10-15): the checks
        that need the mesh loads (a helical mesh needs a locating bearing) run there.
        """
        errors = self._path_errors()
        for s in self.shafts:
            errors.extend(s.validate())
        return errors

    def validate_or_raise(self) -> None:
        """Raise ValueError listing all messages of validate() if there are any."""
        errors = self.validate()
        if errors:
            raise ValueError(
                f"Transmission '{self.label}' validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # resolve
    # ------------------------------------------------------------------

    def resolve(self, P_W: float, n_rpm: float, rotation: int,
                source_position_mm: tuple[float, float] = (0.0, 0.0),
                source_torque_x_mm: float = 0.0) -> ResolvedTransmission:
        """
        Carry the power P_W at the speed n_rpm from the source shaft along the written path and
        return the loaded copies of the shafts; the shafts of ``self.shafts`` are not modified.

        Parameters
        ----------
        P_W : float
            Power entering the source shaft [W], >= 0 (Q41).
        n_rpm : float
            Speed of the source shaft [rpm], > 0.
        rotation : int
            Sense of rotation of the source shaft about +X, +1 or -1 [-].
        source_position_mm : tuple of float
            (y, z) global offset of the source shaft axis [mm] (written to the shaft_position of
            its loaded copy); the position of every other shaft follows from it.
        source_torque_x_mm : float
            Axial coordinate on the source shaft [mm] (0 = left end, as for gear.position) where
            the external power input (motor, coupling) enters; the TorqueLoad +rotation T_source is
            placed there.

        Returns
        -------
        ResolvedTransmission
            Loaded copies, torque, sense of rotation and speed of every shaft, the meshes and
            the closure checks. Resolving the SAME transmission again gives new copies with equal
            loads. A transmission built from the returned copies and resolved would add the mesh
            loads a second time (the copies carry them as ordinary loads).

        Raises
        ------
        ValueError
            If the written path is invalid (the shafts and stages are mutable, so it is checked
            again here), P_W < 0 (Q41), n_rpm <= 0, rotation is not +1 or -1, or a loaded copy
            fails ``ShaftSystem.validate()`` (one error with every message; Q43).

        Notes
        -----
        ``omega = n_rpm * 2 pi / 60`` [rad/s] and ``T_source = P_W / omega`` [N m]
        (ISO_6336-1_2019, 4.2.2, Formulae (2)-(4), with exact constants); no efficiency. Each
        stage, in the order written, takes ``share`` of the torque entering its input shaft and
        gives the loads of its shafts and the torque, speed, sense of rotation and position of
        its output shafts (``PairStage.resolve``). Loads are tagged source="gear_mesh" and go
        through ``ShaftSystem.with_loads`` (a new object per shaft).

        Signs (D-0010 items 12-13; José, 2026-10-15, Q05, Q47, answer 9): the source delivers to
        its shaft the torque T_s = rotation T_source about +X (T_s omega > 0: power in), stored as
        a TorqueLoad of +T_s at source_torque_x_mm. Each stage takes T_a = share x (torque that
        entered its input shaft) on its driving gear; the mesh delivers to the output shaft
        -T_a z_b / z_a, which is what enters that shaft and is passed on. Every TorqueLoad of a
        gear is the moment of its mesh force about the axis, so the source shaft nets zero; an
        output shaft nets the torque its mesh delivers and is closed by a sink. A stage after an
        internal stage receives a torque of the sign of its rotation (positive power): its loads
        have positive magnitudes and its driving gear is the one on its input shaft.

        Assumptions: ideal meshes (no losses), mesh forces independent of the shaft and bearing
        deflections (uncoupled, unlike Mayeux2002, Eq. (1) and Eq. (5)), forces applied on the
        shaft axis with the moment of the force at the pitch point (torque and bending moment,
        ``mesh_forces_to_shaft_loads``).

        Each loaded copy carries the speed of its shaft (Q09) and is validated before it is
        returned (Q43).

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formulae (2)-(4), p. 19 (checked): T, P and omega.
        .. [2] ISO_21771_2007, 5.2.1-5.2.2, Eq. (52), (53) (checked): ratio and speeds.
        """
        self._raise_on_path_errors("Transmission.resolve")
        # Q41 (José, 2026-10-15): power flows from the source; FUTURE: braking / regenerative power with the D-0010 signs
        if P_W < 0.0:
            raise ValueError(f"resolve: P_W must be >= 0, got {P_W} W (braking / regenerative power is not modelled)")
        if n_rpm <= 0.0:
            raise ValueError(f"resolve: rpm must be > 0, got {n_rpm}")
        if rotation not in (1, -1):
            raise ValueError(f"resolve: rotation must be +1 or -1, got {rotation}")

        omega = n_rpm * 2.0 * math.pi / 60.0      # [rad/s]
        T_source = P_W / omega                    # [N m]; P in W, omega in rad/s

        index = {id(s): i for i, s in enumerate(self.shafts)}
        i_source = index[id(self.source)]
        n = len(self.shafts)
        torque: list = [None] * n                 # [N m] entering each shaft
        rot: list = [None] * n
        speed: list = [None] * n                  # [rpm]
        position = [s.shaft_position for s in self.shafts]     # (y, z) of each axis [mm], written on the copies only
        loads: list[list] = [[] for _ in range(n)]
        # the source delivers rot x T_source about +X: T omega > 0, power enters (D-0010 item 12)
        torque[i_source], rot[i_source], speed[i_source] = rotation * T_source, rotation, n_rpm
        position[i_source] = source_position_mm

        # external power input on the source shaft itself, the torque the source delivers; the mesh
        # torque of the driving gear (the moment of its mesh force) is its opposite
        loads[i_source].append(TorqueLoad(source_torque_x_mm, rotation * T_source,
                                          label=f"{self.source.name}:T_in", source="gear_mesh"))

        meshes: list[MeshForces] = []
        checks: list[tuple[str, float, bool]] = []
        for stage in self.path:
            ids = [index[id(s)] for s in stage.shafts]
            i_in = ids[0]
            # each stage takes `share` of the torque entering its input shaft (the shares of a shaft sum to 1: Q40)
            out = stage.resolve(torque[i_in] * stage.share, speed[i_in], rot[i_in], position[i_in])
            for k, i in enumerate(ids):
                loads[i] += out["loads"][k]
                if k:      # an output shaft of the stage: what the stage gives becomes its state
                    torque[i], rot[i], speed[i] = out["torque_Nm"][k], out["rotation"][k], out["speed_rpm"][k]
                    position[i] = out["position_mm"][k]
            meshes += out["meshes"]
            checks += out["checks"]

        # every output shaft has a sink (Q16, checked in _path_errors): the reaction torque of the output
        for sink_shaft, sink_x in self.sinks:
            i = index[id(sink_shaft)]
            # closure of the shaft: minus the sum of its torque loads [N m], user loads included
            T_open = sum(ld.magnitude for ld in sink_shaft.torque_loads + loads[i]
                         if isinstance(ld, TorqueLoad))
            loads[i].append(TorqueLoad(sink_x, -T_open, label=f"{sink_shaft.name}:T_out", source="gear_mesh"))

        # one loaded copy per shaft, in the order of self.shafts; the user's shafts keep their
        # own loads and position
        # Q09 (José, 2026-10-15): each copy carries the speed of its shaft along the path
        loaded = tuple(s.with_loads(loads[i], speed_rpm=speed[i], shaft_position=position[i])
                       for i, s in enumerate(self.shafts))
        # Q43 (José, 2026-10-15): the loaded copies are validated here, one error with every message
        errors = [e for s in loaded for e in s.validate()]
        if errors:
            raise ValueError(f"{self.label or 'Transmission'}: invalid loaded shafts:\n"
                             + "\n".join(f"  - {e}" for e in errors))

        # closure of each shaft and of the train (statics and power balance; no library source)
        for s in loaded:
            checks.append(_closure(f"shaft '{s.name}': sum of the torque loads [N m]",
                                   abs(_load_sums(s.loads)[3]), abs(T_source)))
        inputs = {index[id(stage.shafts[0])] for stage in self.path}
        # power leaving the shafts that feed no stage: the power entering each, T omega with signed T and omega [W]
        P_out = sum(torque[i] * rot[i] * speed[i] * 2.0 * math.pi / 60.0 for i in range(n) if i not in inputs)
        checks.append(_closure("train: power leaving the end shafts - power entering [W]",
                               abs(P_out - P_W), abs(P_W)))

        return ResolvedTransmission(transmission=self, shafts=loaded, torque_Nm=tuple(torque), rotation=tuple(rot),
                                    speed_rpm=tuple(speed), meshes=tuple(meshes), checks=tuple(checks))

    def __repr__(self) -> str:
        """Short text: label, number of shafts and of stages."""
        return f"Transmission(label={self.label!r}, shafts={len(self.shafts)}, stages={len(self.path)})"
