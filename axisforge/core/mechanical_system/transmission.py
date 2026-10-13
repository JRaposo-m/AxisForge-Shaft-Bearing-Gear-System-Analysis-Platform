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
is carried and stored in N m end to end (what ``CylindricalFamily.forces`` takes and returns and
what the ``TorqueLoad`` objects created here carry); see the Warnings of ``core/loads.py``.

The loads are those of the code before this module was rewritten, bit for bit. Where they differ
from what ``MeshForces`` and D-0010 give, the line carries ``# QUESTION(Qnn)``; the closure checks
stored on the ``ResolvedTransmission`` show which statics they do not satisfy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from axisforge.config import TOL_GEOMETRY_mm
from axisforge.core.loads import AxialLoad, DistributedRadialLoad, RadialLoad, TorqueLoad
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

def mesh_forces_to_shaft_loads(F: dict, gear: GearElement, side: str, T_in_Nm: float,
                               distribute: bool = False) -> list:
    """
    Shaft loads of one gear of a mesh: tooth forces, axial thrust and torque.

    The force of the mesh on a gear acts at the pitch point, at the working pitch radius from the
    shaft axis. Here it is put on the shaft AXIS at the axial station of the gear: the transverse
    force as two radial loads on the axis, the axial component as an axial load, and a torque.
    The stored torque is NOT computed as the moment of the stored transverse force about the
    axis: it is +T_in on the driver gear and -T_out on the driven gear (power-flow signs; see
    Warnings, Q05). This is the only function that builds shaft loads from a mesh (D-0010
    item 19).

    Parameters
    ----------
    F : dict
        Result of ``CylindricalFamily.forces``: Ft, Fr, Fa [N] (magnitudes), T_out [N m] and the
        angles theta_Fr_driver, theta_Ft_driver, theta_Fr_driven, theta_Ft_driven [deg].
    gear : GearElement
        Gear of this side; its position [mm] (centre of the face width, from the left end of the
        shaft) and face width b [mm] place the loads; its label names them.
    side : str
        "driver" (the gear on the shaft that brings the power) or "driven".
    T_in_Nm : float
        Torque entering the mesh on the driver gear [N m].
    distribute : bool
        False: Ft and Fr are point loads (``RadialLoad``) at the gear position. True: each is a
        uniform ``DistributedRadialLoad`` over [position - b/2, position + b/2] [mm], its
        magnitude the total force [N] (uniform over the face width: choice of this code, no
        library source); point loads when b = 0.

    Returns
    -------
    list
        In this order: the load ``<label>:Fr`` and the load ``<label>:Ft`` [N], an ``AxialLoad``
        ``<label>:Fa`` [N] if Fa != 0, and one ``TorqueLoad`` ``<label>:T`` [N m]; all tagged
        ``source="gear_mesh"``.

    Notes
    -----
    Conventions (D-0010 items 1-3): global frame (X, Y, Z), the shaft along +X; a transverse
    direction is the angle theta from +Y toward +Z, right-hand about +X (F_y = F cos(theta),
    F_z = F sin(theta)); positions [mm] from the left end of the shaft; forces [N]; torque [N m],
    positive right-hand about +X.

    What is stored today, with phi the angle of the line of centres from the driver shaft to the
    driven shaft and rot the sense of rotation of the driver shaft::

        driver gear:  Fr at phi          Ft at phi + 90 rot    +Fa    TorqueLoad +T_in
        driven gear:  Fr at phi + 180    Ft at theta_Ft_driven  -Fa    TorqueLoad -T_out

    Assumptions: ideal mesh (no losses), mesh forces independent of the deflections of shafts
    and bearings, one force at mid face width.

    Warnings
    --------
    The stored loads are NOT the force of the mesh on each gear under D-0010 (``MeshForces``);
    each difference is an open question, marked at its line:

    - Q05, directions (external pair only): Fr of each gear points from its own axis TOWARD
      the mesh, opposite to the separating force; Ft has the same direction on both gears, so
      the two are not an action-reaction pair (the Ft on the driver is the reversed one).
    - Q05, torque signs (external AND internal pairs): they follow the power flow (+ on the
      driver gear, - on the driven gear) whatever the sense of rotation, so the stored torque
      is not the moment of the stored Ft in general (external pair: equal only when the driver
      shaft turns in the +1 sense), and the power of the mesh computed from them is 2 P, not 0.
    - Q03: the magnitudes are those of ``forces()``, which puts the torque on gear 1 of an
      external pair whichever gear is on the driving shaft. Q20: for an internal pair whose
      ring drives, the meaning of phi is open.
    - Q06: the axial force is applied on the shaft axis; its bending moment F_a r_w about the
      transverse axis (Harris2007_vol1, 4.3, Eq. (4.28)-(4.29)) is not created.
    - Q07a: Fa = Ft tan(beta) (working force, reference helix angle) instead of
      F_bt tan(beta_b): 444.750 against 449.236 N in baseline case 2.
    - Q08: +Fa on the driver and -Fa on the driven gear, whatever the hand of the helix and the
      sense of rotation.

    References
    ----------
    .. [1] D-0010 items 1-3, 14-19; vault note ``units_and_conventions.md``, C.6. No library
           source for the directions (statics); separating radial force: Harris2007_vol1, 4.2.2,
           Fig. 4.4 (one reader).
    """
    suffix = "_driver" if side == "driver" else "_driven"
    pos = gear.position
    b = getattr(gear.gear, "b", 0.0)
    tag = gear.label or "mesh"

    # QUESTION(Q05): the angles of forces() are stored as they come (external pair: both Fr toward the mesh, both Ft the same way); under D-0010 Fr separates the gears and the Ft on the driver is reversed (case 7a, pinion shaft: 30 and 120 deg against 210 and 300 deg).
    if distribute and b > 0.0:
        # total force [N] spread uniformly over the face width, centred on the gear position
        x_lo, x_hi = pos - b / 2.0, pos + b / 2.0
        loads: list = [
            DistributedRadialLoad(x_lo, x_hi, magnitude=F["Fr"], theta_deg=F[f"theta_Fr{suffix}"],
                                  label=f"{tag}:Fr", source="gear_mesh"),
            DistributedRadialLoad(x_lo, x_hi, magnitude=F["Ft"], theta_deg=F[f"theta_Ft{suffix}"],
                                  label=f"{tag}:Ft", source="gear_mesh"),
        ]
    else:
        loads = [
            RadialLoad(pos, F["Fr"], F[f"theta_Fr{suffix}"], label=f"{tag}:Fr", source="gear_mesh"),
            RadialLoad(pos, F["Ft"], F[f"theta_Ft{suffix}"], label=f"{tag}:Ft", source="gear_mesh"),
        ]

    # axial thrust: always a point load
    # QUESTION(Q07a): Fa as returned by forces(), Ft tan(beta); the alternative is MeshForces.F_a_N = F_bt tan(beta_b) (case 7a: 444.750 against 449.236 N).
    Fa = F.get("Fa", 0.0)
    if Fa != 0.0:
        # QUESTION(Q08): +Fa on the driver, -Fa on the driven, whatever the hand of the helix and the sense of rotation; under D-0010 the driver takes hand x rot x F_a.
        sign = 1.0 if side == "driver" else -1.0
        # QUESTION(Q06): applied on the shaft axis; no bending moment Fa x r_w is created (its mapping to ExternalMoment is not ruled: rows 21, 22).
        loads.append(AxialLoad(pos, sign * Fa, label=f"{tag}:Fa", source="gear_mesh"))

    # torque of the mesh on the shaft: always a point load at the gear position
    # QUESTION(Q05): power-flow signs (+T_in on the driver, -T_out on the driven), independent of the sense of rotation; under D-0010 the torque is the moment of the mesh force, -rot T_in on the driver.
    T_flow_Nm = +T_in_Nm if side == "driver" else -F["T_out"]
    loads.append(TorqueLoad(pos, T_flow_Nm, label=f"{tag}:T", source="gear_mesh"))
    return loads


# ===========================================================================
# PairStage
# ===========================================================================

@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class PairStage:
    """
    One written link of the power path: a cylindrical gear pair between two shafts.

    ``shaft_a`` brings the power, ``shaft_b`` receives it. It is the only implementation of the
    stage contract of ``Transmission`` (see there), with ``shafts = (shaft_a, shaft_b)``.

    Parameters
    ----------
    pair : GearPair
        The pair (``CylindricalFamily``): its working centre distance ``al`` [mm] places
        ``shaft_b``, its family gives the forces.
    shaft_a, shaft_b : ShaftSystem
        The shaft that brings the power and the shaft that receives it.
    gear_a, gear_b : GearElement
        The gear of the pair that sits on ``shaft_a`` and the one on ``shaft_b``: their positions
        [mm] and labels place and name the mesh loads; their tooth numbers give the speed of
        ``shaft_b``.
    phi_deg : float
        Angle of the line of centres from the axis of ``shaft_a`` to the axis of ``shaft_b`` in
        the GLOBAL frame [deg], from +Y toward +Z. An input, not derived; stored modulo 360.
    share : float
        Part of the torque entering ``shaft_a`` that this stage takes, in (0, 1] [-]. Default
        1.0: the whole torque (see Q40 in ``Transmission``).
    distribute_loads : bool
        False: Ft and Fr are point loads at the gear position. True: uniform loads over the face
        width of each gear [mm] (see ``mesh_forces_to_shaft_loads``).
    label : str
        Reporting tag.
    driver : str
        "gear1" (default) or "gear2": the ``driver`` argument of ``CylindricalFamily.forces``,
        the gear of the pair that delivers the torque. Read by an internal pair only; an
        external pair takes the torque on its gear 1 whatever it says. It is NOT derived from
        ``gear_a`` (Q03).

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
    # QUESTION(Q03): the driver is written here as it was given to the link, default "gear1", whatever gear `gear_a` wraps; the alternative derives it from the gear that sits on shaft_a (cases 7e, 7f: the torque would go on the wheel, 40.5 N m on the pinion).
    driver: str = "gear1"

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

        Checked: ``driver`` is "gear1" or "gear2"; the two gears occupy the same global axial
        position ``shaft_origin_x + gear.position`` within ``TOL_GEOMETRY_mm`` (1e-9 mm, whatever
        the face widths: the overlap of the two face widths is not used). ``shaft_origin_x`` is
        not propagated by ``resolve``: it is the caller's to set on every shaft.
        """
        errors: list[str] = []
        tag = self.label or "PairStage"
        if self.driver not in ("gear1", "gear2"):
            errors.append(f"{tag}: driver must be 'gear1' or 'gear2', got {self.driver!r}")
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

    @property
    def warnings(self) -> tuple[str, ...]:
        """
        Advisory messages of the stage; they never raise.

        These are checks that the code before this module did not run (validation level L6:
        the gear of each stage belongs to the pair and sits on its shaft). They are warnings and
        not errors because recorded cases would be refused (Q45).
        """
        # QUESTION(Q45): new checks of the written path kept as warnings: as errors they would refuse baseline cases 7e, 7f (gear_a is the wheel while forces() takes the torque on gear 1, the pinion).
        tag = self.label or "PairStage"
        pair, found = self.pair, []
        gears = (self.gear_a.gear, self.gear_b.gear)
        if not ((gears[0] is pair.gear1 and gears[1] is pair.gear2)
                or (gears[0] is pair.gear2 and gears[1] is pair.gear1)):
            found.append(f"{tag}: gear_a and gear_b are not the two gears of the pair")
        # the gear on which forces() takes the torque: gear 1 of an external pair, the declared driver of an internal pair
        taker = pair.gear2 if (pair.gear2.z < 0 and self.driver == "gear2") else pair.gear1
        if self.gear_a.gear is not taker:
            found.append(f"{tag}: the gear on shaft_a ('{self.shaft_a.name}') is not the gear on which "
                         f"forces() takes the torque; the mesh forces and the torque of "
                         f"'{self.shaft_b.name}' do not follow the power balance")
        for shaft, element in ((self.shaft_a, self.gear_a), (self.shaft_b, self.gear_b)):
            if not any(element is ge for ge in shaft.gears):
                found.append(f"{tag}: gear element '{element.label}' is not on shaft '{shaft.name}'")
        return tuple(found)

    def resolve(self, T_in_Nm: float, n_in_rpm: float, rot_in: int,
                position_in_mm: tuple[float, float]) -> dict:
        """
        Loads, torque, speed, sense of rotation and axis position of the two shafts of the stage.

        Parameters
        ----------
        T_in_Nm : float
            Torque that this stage takes from ``shaft_a`` [N m].
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
            ``mesh_forces_to_shaft_loads``), ``torque_Nm`` (T_in, T_out) [N m], ``rotation``,
            ``speed_rpm`` [rpm], ``position_mm`` (y, z) [mm]; ``meshes``: one ``MeshForces``;
            ``checks``: the closure checks of the mesh, ``(check, residual, ok)``.

        Notes
        -----
        ``shaft_b`` is placed at the working centre distance a_w along phi:
        ``(y_a + a_w cos(phi), z_a + a_w sin(phi))`` [mm]. Its speed is
        ``n_a abs(z_a / z_b)`` with the tooth numbers of the gears on the two shafts
        (ISO_21771_2007, 5.2.2, Eq. (53)). Its torque and sense of rotation are T_out and
        rotation_dir_out of ``CylindricalFamily.forces`` (Q03: T_out = T u with u = max/min for
        an external pair, whichever gear is on ``shaft_a``).

        The ``MeshForces`` is the same mesh under D-0010: the torque ``rot_in T_in_Nm`` is put
        on the gear that sits on ``shaft_a``, with a right-hand GEAR 1 (Q08: the same physical
        mesh gets the opposite axial sign when the other gear is called gear 1). That gear is
        the driving one when ``T_in_Nm > 0``. After an internal stage the carried torque is
        negative (T_out = T u, u < 0), so in the next stage it comes out as the DRIVEN gear and
        the stored loads of that stage have negative magnitudes (Q47). The ``MeshForces`` is
        NOT what the loads are built from.

        Closure checks (statics on the loads just built; no library source):

        - action = reaction: the forces on the two gears sum to zero [N];
        - torque = moment of the mesh force, per gear: the stored torque equals the moment
          about the shaft axis of the stored transverse force applied at the pitch point [N m];
        - power of the mesh: the torques of the mesh on the two shafts, each times the signed
          angular velocity of its shaft, sum to zero [W] (an ideal mesh neither adds nor removes
          power).
        """
        pair = self.pair
        # QUESTION(Q03): forces() takes this torque on the gear ITS OWN rule names (external pair: gear 1, the smaller working diameter, whatever the driver; internal pair: the declared driver), which is not necessarily the gear on shaft_a.
        F = pair.family.forces(pair, T_in_Nm, self.phi_deg, rot_in, driver=self.driver)

        # place shaft_b relative to shaft_a along the line of centres [mm] (phi from +Y toward +Z)
        y_a, z_a = position_in_mm
        al = pair.al
        phi = math.radians(self.phi_deg)
        position_b = (y_a + al * math.cos(phi), z_a + al * math.sin(phi))

        loads_a = mesh_forces_to_shaft_loads(F, self.gear_a, "driver", T_in_Nm, self.distribute_loads)
        loads_b = mesh_forces_to_shaft_loads(F, self.gear_b, "driven", T_in_Nm, self.distribute_loads)

        # no slip at the pitch point: n_b / n_a = z_a / z_b, a magnitude (ISO_21771_2007 Eq. (53))
        n_out_rpm = n_in_rpm * abs(self.gear_a.gear.z / self.gear_b.gear.z)
        # QUESTION(Q03): T_out = T u as returned by forces() becomes the torque entering shaft_b (cases 7c-7f: 91.125 N m where the tooth numbers give 40.5 N m).
        T_out_Nm, rot_out = F["T_out"], F["rotation_dir_out"]

        # the same mesh under D-0010 (new numbers; not used for the loads). The shaft applies to the
        # DRIVING gear a torque in its sense of rotation (power enters the mesh: T omega > 0).
        a_is_gear2 = self.gear_a.gear is pair.gear2
        z1, z2 = pair.gear1.z, pair.gear2.z
        # QUESTION(Q47): the torque on the gear of shaft_a is rot_in x T_in_Nm, T_in_Nm signed as it is carried; after an internal stage it is negative (T_out = T u, u < 0), so this gear is then the DRIVEN one of the MeshForces (tangential force reversed against the physical driver).
        if a_is_gear2:
            # torque and sense on gear 1 from those on gear 2: T_1 = T_2 z_1/z_2, omega_1 = -omega_2 z_2/z_1 (signed z)
            T1_Nm, rot1 = rot_in * T_in_Nm * z1 / z2, (-rot_in if z1 * z2 > 0 else rot_in)
        else:
            T1_Nm, rot1 = rot_in * T_in_Nm, rot_in
        # QUESTION(Q08): the stage has no hand of helix: mesh_forces takes its default, a right-hand GEAR 1 (the axial sign of the MeshForces follows which gear is called gear 1).
        mesh = pair.family.mesh_forces(pair, T1_Nm, rot1)

        # pitch point C seen from each axis [mm]: rho along e_c (gear 1 -> gear 2), which is the
        # direction phi when gear 1 is on shaft_a and the opposite one when gear 2 is
        s = -1.0 if a_is_gear2 else 1.0
        rho_a, rho_b = (mesh.rho_mm[1], mesh.rho_mm[0]) if a_is_gear2 else mesh.rho_mm
        e_y, e_z = s * math.cos(phi), s * math.sin(phi)
        (Fx_a, Fy_a, Fz_a, T_a), (Fx_b, Fy_b, Fz_b, T_b) = _load_sums(loads_a), _load_sums(loads_b)
        # moment about +X of a transverse force at C: (r x F)_x = r_y F_z - r_z F_y [N mm] -> [N m]
        M_a = rho_a * (e_y * Fz_a - e_z * Fy_a) / 1000.0
        M_b = rho_b * (e_y * Fz_b - e_z * Fy_b) / 1000.0
        omega_a = rot_in * n_in_rpm * 2.0 * math.pi / 60.0      # signed angular velocity [rad/s]
        omega_b = rot_out * n_out_rpm * 2.0 * math.pi / 60.0
        tag = self.label or "PairStage"
        checks = (
            _closure(f"{tag}: action = reaction, |F on gear_a + F on gear_b| [N]",
                     math.hypot(Fx_a + Fx_b, Fy_a + Fy_b, Fz_a + Fz_b), abs(F["Ft"])),
            _closure(f"{tag}: torque of gear_a = moment of its mesh force [N m]", abs(T_a - M_a), abs(T_in_Nm)),
            _closure(f"{tag}: torque of gear_b = moment of its mesh force [N m]", abs(T_b - M_b), abs(T_out_Nm)),
            _closure(f"{tag}: power of the mesh, T_a omega_a + T_b omega_b [W]",
                     abs(T_a * omega_a + T_b * omega_b), abs(T_in_Nm * omega_a)),
        )
        return {"loads": (loads_a, loads_b), "torque_Nm": (T_in_Nm, T_out_Nm), "rotation": (rot_in, rot_out),
                "speed_rpm": (n_in_rpm, n_out_rpm), "position_mm": (position_in_mm, position_b),
                "meshes": (mesh,), "checks": checks}

    def __repr__(self) -> str:
        """Text of the stage: names of the two shafts, phi [deg], share [-] and label."""
        return (f"PairStage({self.shaft_a.name} -> {self.shaft_b.name}, "
                f"phi={self.phi_deg:.1f}°, share={self.share!r}, label={self.label!r})")


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
        Torque entering each shaft [N m]: P / omega on the source shaft, the output torque of
        its stage on every other shaft. It inherits Q03 (cases 7c-7f: 91.125 N m where the
        tooth numbers give 40.5 N m) and is negative after an internal stage (Q47).
    rotation : tuple
        Sense of rotation of each shaft about +X, +1 or -1 [-]. The value of the source shaft
        is the integer given; that of a driven shaft is the float returned by
        ``CylindricalFamily.forces`` (-1.0 or 1.0), as recorded.
    speed_rpm : tuple of float
        Speed of each shaft [rpm], a magnitude: the speed given on the source shaft, then
        n z_a / z_b down the path with the tooth numbers of the gears on the shafts
        (ISO_21771_2007, 5.2.2, Eq. (53)). A kinematic value: it does NOT inherit Q03, while
        ``torque_Nm`` does (case 7c: 2175 / 1450 rpm), which is why the power of the train does
        not close in cases 7c-7f.
    meshes : tuple of MeshForces
        The meshes of the stages, in path order, under D-0010 (new numbers, NOT what the stored
        loads are built from: see ``PairStage.resolve``). The sign of ``on_1[2]``, ``on_2[2]``
        assumes a RIGHT-hand GEAR 1 (Q08); see Q47 for a stage that follows an internal stage.
    checks : tuple of (str, float, bool)
        Closure post-conditions ``(check, residual, ok)``, computed on the stored loads; they
        never raise. Per mesh: action = reaction, torque of each gear = moment of its mesh
        force, power of the mesh. Per shaft: the torque loads sum to zero. For the train: the
        power leaving the end shafts equals the power entering. Several fail on today's loads
        (Q03, Q05, Q21 for a helical internal pair, Q47, and Q16 for a shaft with no sink).
        They are NECESSARY conditions, not sufficient: a passing check does not make a load
        right. Case 7a passes "torque of gear = moment" with the stored torque AND the stored
        F_t of the pinion both reversed against D-0010; the radial part of "action = reaction"
        holds while both F_r are reversed; no check tests that the radial force separates the
        gears.
    warnings : tuple of str
        Advisory messages of the transmission and of this resolve; they never raise.

    Notes
    -----
    Where the speed is and is not: the speed of every shaft is HERE, in ``speed_rpm``. It is
    NOT written on the loaded copies: ``ShaftSystem.speed_rpm`` of each copy is the value the
    user set on that shaft (0.0 by default), as before (Q09).
    """

    transmission: "Transmission"
    shafts: tuple[ShaftSystem, ...]
    torque_Nm: tuple[float, ...]
    rotation: tuple
    speed_rpm: tuple[float, ...]
    meshes: tuple[MeshForces, ...]
    checks: tuple[tuple[str, float, bool], ...]
    warnings: tuple[str, ...]

    @property
    def label(self) -> str:
        """Reporting tag of the transmission that was resolved."""
        return self.transmission.label

    def summary(self) -> str:
        """Multi-line text: shafts, stages, source and, per shaft in the order of the power path, torque [N m], sense of rotation and axis position [mm]."""
        tr = self.transmission
        # QUESTION(Q46): the words "links" and "T_out" are those of the recorded text of the class this one replaces.
        lines = [f"── {tr.label or 'Transmission'} ──────────────────────────────────────────────",
                 f"  shafts: {len(self.shafts)}   links: {len(tr.path)}   source: {tr.source.name}"]
        # the shafts in the order they receive power: the source, then the output shafts of each stage
        index = {id(s): i for i, s in enumerate(tr.shafts)}
        for i in [index[id(tr.source)]] + [index[id(s)] for stage in tr.path for s in stage.shafts[1:]]:
            y, z = self.shafts[i].shaft_position
            t_str = f"{self.torque_Nm[i]:.3f} N·m"
            lines.append(f"    {self.shafts[i].name:<12} T_out={t_str:<14} "
                         f"rot={self.rotation[i]!s:<3} pos=(y={y:.2f}, z={z:.2f}) mm")
        lines.append("────────────────────────────────────────────────────────")
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
        The load is labelled ``<shaft name>:T_out`` with source "gear_mesh". Being the residue
        of the stored torque loads, it inherits their signs and magnitudes (Q03, Q05). Default
        ``()``: no shaft is closed (a driven shaft keeps the net torque -T_out).
    label : str
        Reporting tag.

    Attributes
    ----------
    warnings : tuple of str
        Advisory messages (a property, read from the current state of the shafts and stages);
        they never raise (Q40, Q45).

    Raises
    ------
    ValueError
        At construction, if the written path is invalid (validation level L6): a message of a
        stage; a stage whose shafts are not distinct or not in ``shafts``; a share outside
        (0, 1]; a stage whose input shaft has no power yet; a shaft that receives power twice;
        a shaft that receives none; the source or a sink shaft not in ``shafts``.

    Notes
    -----
    Stage contract. ``Transmission`` reads from a stage only: ``shafts``, a tuple of ANY length
    (two or more) of the ``ShaftSystem`` objects it links, the input shaft first; ``share``
    [-]; ``label``; ``warnings`` (tuple of str); ``validate()`` (list of str); and
    ``resolve(T_in_Nm, n_in_rpm, rot_in, position_in_mm)``, which returns a dict of tuples with
    one entry per shaft of ``stage.shafts``, in that order: ``loads`` (a list of loads each),
    ``torque_Nm`` [N m], ``rotation``, ``speed_rpm`` [rpm], ``position_mm`` ((y, z) [mm]), plus
    ``meshes`` (tuple of ``MeshForces``) and ``checks`` (tuple of ``(check, residual, ok)``).
    The entries of the input shaft are what the stage received; those of the other shafts
    become the state of those shafts. ``PairStage`` is the only implementation; a stage that
    links more than two shafts (a planetary stage) is accepted by this contract and none
    exists. In the same way the members of a ``PlanetaryArrangement`` are a tuple of any length
    (Q44: that class is not written yet).

    Limits: each shaft receives power once (no recombination); one input shaft per stage.

    Q40. Each stage takes ``share`` of the torque entering its input shaft. With the default
    1.0 two stages leaving one shaft EACH take the whole torque, as the code before this module
    did for two links with separate driver gears; the sum of the shares of a shaft above 1 is a
    stored warning, not an error.
    """

    def __init__(self, shafts, path, source: ShaftSystem,
                 sinks: tuple[tuple[ShaftSystem, float], ...] = (), label: str = ""):
        """Store the arguments and check the written path; see the class docstring."""
        self.shafts = list(shafts)
        self.path = tuple(path)
        self.source = source
        self.sinks = tuple(sinks)
        self.label = label
        # Loaded copies returned by the last resolve(), read by validate() (Q43).
        self._loaded: tuple[ShaftSystem, ...] | None = None
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
        errors.extend(f"{tag}: sink shaft {s.name!r} is not in this system" for s, _ in self.sinks
                      if id(s) not in mine)
        return errors

    def _raise_on_path_errors(self, where: str) -> None:
        """Raise ValueError listing the messages of the written path, if any."""
        errors = self._path_errors()
        if errors:
            raise ValueError(f"{where}: invalid power path:\n" + "\n".join(f"  - {e}" for e in errors))

    @property
    def warnings(self) -> tuple[str, ...]:
        """Advisory messages, read from the current state: those of the stages, and the shares leaving one shaft summing above 1."""
        tag = self.label or "Transmission"
        found = [f"{tag}: {w}" for stage in self.path for w in stage.warnings]
        for shaft in self.shafts:
            total = sum(stage.share for stage in self.path if stage.shafts[0] is shaft)
            # QUESTION(Q40): shares of one shaft summing above 1 are kept (each stage with the default share takes the FULL torque, as two links with separate driver gears did); the alternative refuses a sum other than 1.
            if total > 1.0 + 1e-6:
                found.append(f"{tag}: the stages leaving shaft '{shaft.name}' take {total:.6f} times its "
                             f"torque (sum of their shares); the power balance does not close")
        return tuple(found)

    def validate(self) -> list[str]:
        """
        Return the messages of the written path plus those of every ShaftSystem; empty if valid.

        Before resolve() the shafts given by the user are validated; after a resolve(), the
        loaded copies of the LAST resolve(), so that the checks that need the mesh loads still run
        (a helical mesh needs a locating bearing: ShaftSystem.validate()). A load or a bearing
        added to a user's shaft after resolve() is seen only after the next resolve().
        """
        errors = self._path_errors()
        # QUESTION(Q43): after resolve() the loaded copies of the last resolve() are validated, as when resolve() loaded the user's shafts in place; alternatives: resolve() validates the copies itself, or validate() only ever sees the user's shafts.
        for s in (self.shafts if self._loaded is None else self._loaded):
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
            Power entering the source shaft [W]. Not validated (Q41).
        n_rpm : float
            Speed of the source shaft [rpm], > 0.
        rotation : int
            Sense of rotation of the source shaft about +X, +1 or -1 [-].
        source_position_mm : tuple of float
            (y, z) global offset of the source shaft axis [mm] (written to the shaft_position of
            its loaded copy); the position of every other shaft follows from it.
        source_torque_x_mm : float
            Axial coordinate on the source shaft [mm] (0 = left end, as for gear.position) where
            the external power input (motor, coupling) enters; the TorqueLoad -T_source is placed
            there.

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
            again here), n_rpm <= 0 or rotation is not +1 or -1.

        Notes
        -----
        ``omega = n_rpm * 2 pi / 60`` [rad/s] and ``T_source = P_W / omega`` [N m]
        (ISO_6336-1_2019, 4.2.2, Formulae (2)-(4), with exact constants); no efficiency. Each
        stage, in the order written, takes ``share`` of the torque entering its input shaft and
        gives the loads of its shafts and the torque, speed, sense of rotation and position of
        its output shafts (``PairStage.resolve``). Loads are tagged source="gear_mesh" and go
        through ``ShaftSystem.with_loads`` (a new object per shaft).

        The input torque on the source shaft is a TorqueLoad of -T_source at source_torque_x_mm;
        with the +T_source TorqueLoad placed at the first gear it makes the torque diagram of
        the source shaft equal to -T_source (magnitude T_source) from the input up to the gear
        and zero beyond it. Sign convention (external pair): driver gear +T_in, driven gear
        -T_out, power input -T_source (the net torque of the source shaft is zero; a driven shaft
        nets -T_out and is closed by a sink or by a load of the user). With an internal pair
        T_out = T_in u_eff is negative (u < 0), so the TorqueLoad on the driven gear is
        +abs(T_out) and the torque entering that shaft, passed on to the next stage, is negative.

        Assumptions: ideal meshes (no losses), mesh forces independent of the shaft and bearing
        deflections (uncoupled, unlike Mayeux2002, Eq. (1) and Eq. (5)), forces applied on the
        shaft axis.

        Warnings
        --------
        The speed of each shaft is in the returned record only; ``ShaftSystem.speed_rpm`` of the
        copies is not set (Q09).

        The signs of the TorqueLoad objects do not depend on the sense of rotation (power-flow
        signs: + on a driver gear, - at an entry), whereas the Ft directions do; in a train the
        sense alternates, so no value of ``rotation`` makes every mesh consistent (Q05). The
        axial force of a helical mesh creates no bending moment (Q06). See
        ``mesh_forces_to_shaft_loads``; the closure checks of the returned record show each one.

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formulae (2)-(4), p. 19 (checked): T, P and omega.
        .. [2] ISO_21771_2007, 5.2.1-5.2.2, Eq. (52), (53) (checked): ratio and speeds.
        """
        self._raise_on_path_errors("Transmission.resolve")
        if n_rpm <= 0.0:
            raise ValueError(f"resolve: rpm must be > 0, got {n_rpm}")
        if rotation not in (1, -1):
            raise ValueError(f"resolve: rotation must be +1 or -1, got {rotation}")

        omega = n_rpm * 2.0 * math.pi / 60.0      # [rad/s]
        # QUESTION(Q41): P_W < 0 is accepted and gives negative RadialLoad magnitudes (the validate() of the loaded copies reports them); the alternative (validation level L6) refuses P_W < 0.
        T_source = P_W / omega                    # [N m]; P in W, omega in rad/s

        index = {id(s): i for i, s in enumerate(self.shafts)}
        i_source = index[id(self.source)]
        n = len(self.shafts)
        torque: list = [None] * n                 # [N m] entering each shaft
        rot: list = [None] * n
        speed: list = [None] * n                  # [rpm]
        position = [s.shaft_position for s in self.shafts]     # (y, z) of each axis [mm], written on the copies only
        loads: list[list] = [[] for _ in range(n)]
        torque[i_source], rot[i_source], speed[i_source] = T_source, rotation, n_rpm
        position[i_source] = source_position_mm

        # external power input on the source shaft itself: T = -T_source, the same sign as
        # the entry side of a driven gear, so it cancels the +T_source placed at the first gear.
        # QUESTION(Q05): -T_source whatever the sense of rotation (power-flow sign); under D-0010 the input torque has the sign of the rotation (case 7a: +60.75 N m).
        loads[i_source].append(TorqueLoad(source_torque_x_mm, -T_source,
                                          label=f"{self.source.name}:T_in", source="gear_mesh"))

        meshes: list[MeshForces] = []
        checks: list[tuple[str, float, bool]] = []
        for stage in self.path:
            ids = [index[id(s)] for s in stage.shafts]
            i_in = ids[0]
            # QUESTION(Q40): each stage takes `share` of the torque entering its input shaft; with the default 1.0 several stages leaving one shaft each take the full torque.
            # QUESTION(Q47): the torque is passed on with the sign forces() gave it; after an internal stage it is negative, and forces() of the next stage then returns negative Ft, Fr, Fa (stored as negative load magnitudes).
            out = stage.resolve(torque[i_in] * stage.share, speed[i_in], rot[i_in], position[i_in])
            for k, i in enumerate(ids):
                loads[i] += out["loads"][k]
                if k:      # an output shaft of the stage: what the stage gives becomes its state
                    torque[i], rot[i], speed[i] = out["torque_Nm"][k], out["rotation"][k], out["speed_rpm"][k]
                    position[i] = out["position_mm"][k]
            meshes += out["meshes"]
            checks += out["checks"]

        # QUESTION(Q16): the output torque is closed here only where a sink (shaft, x) is written; default: open end, as before. The sink is the residue of ALL stored torque loads, so it inherits Q03 and Q05; its label and source differ from the hand-made "output_coupling" / "user" load of the studies.
        for sink_shaft, sink_x in self.sinks:
            i = index[id(sink_shaft)]
            # closure of the shaft: minus the sum of its torque loads [N m], user loads included
            T_open = sum(ld.magnitude for ld in sink_shaft.torque_loads + loads[i]
                         if isinstance(ld, TorqueLoad))
            loads[i].append(TorqueLoad(sink_x, -T_open, label=f"{sink_shaft.name}:T_out", source="gear_mesh"))

        # one loaded copy per shaft, in the order of self.shafts; the user's shafts keep their
        # own loads and position
        # QUESTION(Q09): speed_rpm is not passed, so every copy keeps the speed of the user's shaft (0.0 by default); the alternative is with_loads(..., speed_rpm=speed[i]) (case 7a: 1450 and 966.667 rpm).
        loaded = tuple(s.with_loads(loads[i], shaft_position=position[i]) for i, s in enumerate(self.shafts))

        # closure of each shaft and of the train (statics and power balance; no library source)
        for s in loaded:
            checks.append(_closure(f"shaft '{s.name}': sum of the torque loads [N m]",
                                   abs(_load_sums(s.loads)[3]), abs(T_source)))
        inputs = {index[id(stage.shafts[0])] for stage in self.path}
        # power leaving the shafts that feed no stage: abs(T) omega of each [W]
        P_out = sum(abs(torque[i]) * speed[i] * 2.0 * math.pi / 60.0 for i in range(n) if i not in inputs)
        checks.append(_closure("train: power leaving the end shafts - power entering [W]",
                               abs(P_out - abs(P_W)), abs(P_W)))

        warnings = self.warnings
        if P_W < 0.0:
            warnings += (f"{self.label or 'Transmission'}: P_W = {P_W} W is negative",)
        self._loaded = loaded
        return ResolvedTransmission(transmission=self, shafts=loaded, torque_Nm=tuple(torque), rotation=tuple(rot),
                                    speed_rpm=tuple(speed), meshes=tuple(meshes), checks=tuple(checks),
                                    warnings=warnings)

    def __repr__(self) -> str:
        """Short text: label, number of shafts and of stages."""
        return f"Transmission(label={self.label!r}, shafts={len(self.shafts)}, stages={len(self.path)})"
