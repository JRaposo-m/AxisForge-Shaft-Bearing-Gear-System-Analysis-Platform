"""
Single-stage planetary (epicyclic) gear train: sun, k planets, ring and carrier.

Composition of the two tooth-contact problems (sun-planet external pair, planet-ring
internal pair, both delegated to the pair classes) with what is specific to the train:
structural conditions (coaxiality, assembly, neighbouring), Willis kinematics, the ideal
torque distribution over the members and over the planets, and the placement of the planets.
Load sharing between planets and the real mesh, bearing and pin forces are not covered.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.internal_gear import InternalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.internal_meshing import InternalGearMeshing
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing


# ===========================================================================
# Member tagging — the anchor for torque flow / future PlanetarySystem topology
# ===========================================================================

class PlanetaryMember(Enum):
    """
    Member of the planetary train, used as key in the kinematics and torque calls.

    Willis index notation: 1 = sun (SUN), 2 = planet (PLANET), 3 = ring (RING),
    H = carrier (CARRIER).
    """
    SUN = auto()        # 1
    PLANET = auto()     # 2
    RING = auto()       # 3
    CARRIER = auto()    # H


#: The three central elements that can carry external torque / be shafted.
COAXIAL_MEMBERS: tuple[PlanetaryMember, ...] = (
    PlanetaryMember.SUN,
    PlanetaryMember.RING,
    PlanetaryMember.CARRIER,
)


class MeshTag(Enum):
    """
    Identifies which of the two tooth contacts a force set belongs to.
    Not read by any module of the package.
    """
    SUN_PLANET = auto()     # external pair, meshing_12
    PLANET_RING = auto()    # internal pair, meshing_23


@dataclass(frozen=True)
class PlanetaryKinematics:
    """
    Result of ``solve_kinematics``: angular speeds of the members in the Willis index notation.

    1 = sun, 2 = planet, 3 = ring, H = carrier. Absolute speeds are omega_N;
    speeds relative to the carrier (eq. 7.11) are omega_N_rel = omega_N(H).

    Parameters
    ----------
    omega_1, omega_2, omega_3, omega_H : float
        Absolute angular speed of sun, planet (spin about its own axis), ring
        and carrier. The speeds are used only in linear combinations, so any
        consistent unit works; rad/s about +X is the convention of the module.
    omega_1_rel, omega_2_rel, omega_3_rel : float
        Speeds relative to the carrier, same unit as omega_1. omega_1_rel =
        omega_1 - omega_H, omega_2_rel = -omega_1_rel*z1/z2, omega_3_rel =
        omega_3 - omega_H.
    i_0 : float
        Basic ratio i_13(H) = z_ring / z_sun [-], negative with z_ring < 0.
    F : int
        Degrees of freedom of the OPERATING MODE [-], not of the mechanism. The
        PGT is inherently 2-DoF; prescribing one central element at omega = 0
        reduces it to F = 1. F = 2 is the differential / summation mode.
    fixed : PlanetaryMember or None
        The element prescribed at omega == 0.0, if any.
    willis_residual : float
        Left-hand side of eq. (7.2) at the solution, in the unit of omega.
        Should be ~0; exposed for numerical transparency, not used internally.
    prescribed : tuple of (PlanetaryMember, float)
        The input speeds as given, in the unit of omega.

    References
    ----------
    .. [1] Arnaudov and Karaivanov, "Planetary Gear Trains", eq. 7.1-7.11, as
       named in the code; not in the library; not checked against the source
       (source to find).
    """
    omega_1: float                 # sun,     absolute
    omega_2: float                 # planet,  absolute spin about its own axis
    omega_3: float                 # ring,    absolute
    omega_H: float                 # carrier, absolute
    omega_1_rel: float             # = omega_1 - omega_H          (7.11)
    omega_2_rel: float             # = -omega_1_rel*z1/z2         (7.11)
    omega_3_rel: float             # = omega_3 - omega_H          (7.11)
    i_0: float                     # basic ratio i_13(H) = z3/z1 = -abs(z3)/z1 < 0, z3 signed (7.1)
    F: int                         # 1 or 2 — operating-mode DoF
    fixed: PlanetaryMember | None  # the element held at omega = 0, if any
    willis_residual: float
    prescribed: tuple[tuple[PlanetaryMember, float], ...] = ()

    # --- descriptive aliases (the rest of the module reads these) ---
    @property
    def omega_sun(self) -> float:
        """Absolute angular speed of the sun, ``omega_1`` (unit of omega)."""
        return self.omega_1

    @property
    def omega_planet_abs(self) -> float:
        """Absolute spin speed of the planet about its own axis, ``omega_2`` (unit of omega)."""
        return self.omega_2

    @property
    def omega_planet_rel(self) -> float:
        """Spin speed of the planet relative to the carrier, ``omega_2_rel`` (unit of omega)."""
        return self.omega_2_rel

    @property
    def omega_ring(self) -> float:
        """Absolute angular speed of the ring, ``omega_3`` (unit of omega)."""
        return self.omega_3

    @property
    def omega_carrier(self) -> float:
        """Absolute angular speed of the carrier, ``omega_H`` (unit of omega)."""
        return self.omega_H


@dataclass(frozen=True)
class PlanetaryTorques:
    """
    Ideal external torques on the sun, ring and carrier and the nominal mesh loads (result of ``torques``).

    Valid under three preconditions: stationary load (omega_i = const, no
    inertial forces), equal load sharing between planets (K_gamma = 1) and no
    losses (eta_0 = 1).

    Parameters
    ----------
    T1, T3, TH : float
        External torque on the sun, the ring and the carrier [N m].
    t : float
        Torque ratio T3/T1 = -i_0 [-].
    Ft : float
        Nominal tangential force per mesh, F_t12 = F_t32 [N].
    F_H2 : float
        Ideal resultant force on one planet pin [N].
    T_mesh_sun : float
        Torque through one sun-planet mesh, on the sun [N m]; the quantity
        that can be passed to ``meshing_12.forces`` (see ``torques``, Warnings).
    T_mesh_planet : float
        Torque through one planet-ring mesh, on the planet [N m]; the
        quantity that can be passed to ``meshing_23.forces``.
    k : int
        Number of planets [-].

    Notes
    -----
    As coded (1 = sun, 3 = ring, H = carrier; the numbers are the equation
    numbers of the book named below)::

        T3 = -i_0 * T1 = +(abs(z3)/z1) * T1                         (6.3)
        TH = -(T1 + T3)                                             (6.4)
        sum(Ti) = T1 + T3 + TH = 0                                  (6.5)
        t  = T3 / T1 = -i_0 > +1                                    (6.7)
        Ft = 2000*T1/(k*d1)                                         (6.1)
        F_H2 = F_t12 + F_t32 = 2*Ft                                 (6.2)

    ``Ft`` uses the REFERENCE diameter d1 of the sun [mm], not the working
    pitch diameter dl1 used by ``meshing_12.forces()`` and
    ``meshing_23.forces()``. The values agree when al = a (no profile-shift
    effect on the working diameter, e.g. x1 + x2 = 0) and z_sun <= z_planet
    (see ``torques``, Warnings). F_H2 = 2*Ft is the ideal resultant on one
    planet pin (sum of the two equal tangential forces; pressure-angle
    components not included).

    ``satisfies_torque_ordering`` tests eq. (6.6), |T1| < |T3| < |TH|.

    References
    ----------
    .. [1] ISO_6336-1_2019, 4.2.2, Formula (1): F_t = 2000 T / d with T [N m],
       d [mm] (reference diameter); 6.5.8.4 a) (dynamic factor, ring gear
       fixed to the case): F_t in Formulae (15) to (17) is the whole peripheral
       load of the sun divided by the number of planets p.
       K_gamma is not included here.
    .. [2] Arnaudov and Karaivanov, "Planetary Gear Trains", eq. 6.1-6.7, as
       named in the code; not in the library; not checked against the source
       (source to find). Includes F_H2 = 2 Ft.
    """
    T1: float
    T3: float
    TH: float
    t: float
    Ft: float
    F_H2: float
    T_mesh_sun: float
    T_mesh_planet: float
    k: int

    # --- descriptive aliases of the fields above ---
    @property
    def T_sun(self) -> float:
        """External torque on the sun, ``T1`` [N m]."""
        return self.T1

    @property
    def T_ring(self) -> float:
        """External torque on the ring, ``T3`` [N m]."""
        return self.T3

    @property
    def T_carrier(self) -> float:
        """External torque on the carrier, ``TH`` [N m]."""
        return self.TH

    @property
    def T_sun_planet_mesh(self) -> float:
        """Torque through one sun-planet mesh, on the sun, ``T_mesh_sun`` [N m]."""
        return self.T_mesh_sun

    @property
    def T_planet_ring_mesh(self) -> float:
        """Torque through one planet-ring mesh, on the planet, ``T_mesh_planet`` [N m]."""
        return self.T_mesh_planet

    def satisfies_torque_ordering(self) -> bool:
        """
        Torque ordering of eq. (6.6), |T1| < |T3| < |TH|.

        Returns
        -------
        bool
            True if the ordering holds. A sanity check, not enforced.
        """
        return abs(self.T1) < abs(self.T3) < abs(self.TH)


# ===========================================================================
# PlanetaryGearTrainMeshing
# ===========================================================================

class PlanetaryGearTrainMeshing:
    """
    Single-stage planetary train: sun (1), k equally spaced planets (2), ring (3), carrier (H).

    Built from two pair objects, ``meshing_12`` (``SpurHelicalGearMeshing``,
    sun-planet, external) and ``meshing_23`` (``InternalGearMeshing``,
    planet-ring, internal). The class adds the train-level checks, the Willis
    kinematics, the ideal torques and the planet positions.

    Parameters
    ----------
    sun_gear, planet_gear : SpurHelicalGear
        External gears, z > 0.
    ring_gear : InternalGear
        Ring gear, z < 0 (signed-z convention; required by ``validate``).
    k : int, optional
        Number of planets [-], >= 1, equally spaced. Default 3.
    label : str, optional
        Name used in messages and in the summary.
    phi0_deg : float, optional
        Angle of the line of centres sun -> planet 0 [deg]; stored modulo 360.
    al_12, al_23 : float or None, optional
        Working centre distance of the sun-planet and planet-ring pairs [mm];
        None: derived from the profile shifts by the pair object.
    equalise_gs_12, equalise_gs_23 : bool, optional
        Forwarded to the pair objects (``equalise_gs=True`` fails for the
        planet-ring pair, see ``InternalGearMeshing``).
    addendum_reduction_12, addendum_reduction_23 : bool, optional
        Forwarded to the pair objects.
    load_sharing_factor : float, optional
        Placeholder for K_gamma [-]; 1.0 = equal sharing. It is only checked
        (>= 1.0) by ``validate``; no formula uses it.

    Attributes
    ----------
    meshing_12, meshing_23 : object
        The two pair objects; ``forces`` of each gives the tooth loads on the
        working pitch radius.
    al_12, al_23 : float
        Resolved working centre distances [mm].
    r_carrier : float
        Radius of the planet axes from the train axis, set to al_12 [mm].
    i_0 : float
        Basic ratio i_13(H) = ring.z / sun.z [-] (negative for z_ring < 0).

    Notes
    -----
    Index notation: 1 = sun, 2 = planet, 3 = ring, H = carrier. In a speed
    ratio i_AB(C), A is the input, B the output and C the fixed element; C
    goes in brackets: i_1H(3), never i_1H^3.

    Conventions: all members rotate about the global +X axis (right-hand).
    Torque in N m, forces in N, lengths in mm (as in the pair objects). The
    speeds omega are used only in linear combinations, so any consistent unit
    works (the module speaks of rad/s). phi_deg is the line of centres in the
    global YZ frame, +Y toward +Z; planet j sits at phi_j = phi_0 + j*360/k
    (sun -> planet direction); the planet -> ring line of centres is
    phi_j + 180 (ring centre = sun centre).

    The negative z_ring is structural: it makes the signed tooth-number sum of
    ``InternalGearMeshing`` a difference and makes rotation_dir_out =
    +rotation_dir_in (ring and planet turn the same way). abs(z_ring) is used
    where a tooth count is meant (assembly condition); the signed value where
    a ratio is meant (i_0).

    Equal load sharing over the k planets is assumed (K_gamma = 1, no
    mesh-stiffness or manufacturing-error load-sharing factor). Not covered:
    load distribution between planets and the real mesh, bearing and pin forces
    (``torques`` gives ideal torques only). This class computes no F_r, F_a or
    F_n; ``meshing_12.forces`` and ``meshing_23.forces`` do.

    References
    ----------
    .. [1] ISO_6336-1_2019, 4.2.1 (K_gamma) and 6.5.8.4 a) (dynamic factor,
       ring gear fixed to the case: F_t in Formulae (15) to (17) = peripheral
       load of the sun divided by the number of planets).
    .. [2] Arnaudov and Karaivanov, "Planetary Gear Trains", sections 6.1-6.3,
       7.1; eq. 6.1-6.7 and 7.1-7.11; chapters 8 and 9 for the load
       distribution, as named in the code; not in the library; not checked
       against the source (source to find).
    .. [3] KHK Gear Technical Reference sections 4.2 and 5 (planetary trains);
       Henriot, "Traite theorique et pratique des engrenages", chapter on
       epicyclic trains; MAAG Gear Book; ISO 21771:2007: all named in the code;
       not in the library; not checked against the source (source to find).
    """

    def __init__(self,
                 sun_gear: SpurHelicalGear,
                 planet_gear: SpurHelicalGear,
                 ring_gear: InternalGear,
                 k: int = 3,
                 label: str = "",
                 phi0_deg: float = 0.0,
                 al_12: float | None = None,
                 equalise_gs_12: bool = False,
                 addendum_reduction_12: bool = False,
                 al_23: float | None = None,
                 equalise_gs_23: bool = False,
                 addendum_reduction_23: bool = False,
                 load_sharing_factor: float = 1.0):
        """Build the two pair objects and the train data; see the class docstring."""

        self.label = label
        self.k = int(k)
        self.phi0_deg = phi0_deg % 360.0
        self.load_sharing_factor = load_sharing_factor  # K_gamma placeholder (1.0 = ideal)

        self.sun_gear = sun_gear
        self.planet_gear = planet_gear
        self.ring_gear = ring_gear

        # --- the two tooth-contact problems (they run their own compatibility
        #     checks: mn / alpha_n / beta_n) --------------------------------
        self.meshing_12 = SpurHelicalGearMeshing(
            sun_gear, planet_gear,
            label=f"{label}:sun-planet",
            al=al_12,
            equalise_gs=equalise_gs_12,
            addendum_reduction=addendum_reduction_12,
            driver="gear1",
        )
        self.meshing_23 = InternalGearMeshing(
            planet_gear, ring_gear,
            label=f"{label}:planet-ring",
            al=al_23,
            equalise_gs=equalise_gs_23,
            addendum_reduction=addendum_reduction_23,
            driver="gear1",
        )

        # --- resolved (not requested) centre distances -----------------------
        self.al_12 = self.meshing_12.al
        self.al_23 = self.meshing_23.al

        # planet axis radius from the train axis (carrier pitch radius)
        self.r_carrier = self.al_12

        # basic ratio (7.1):  i_0 = i_13(H) = omega_1rel/omega_3rel = z3/z1 = -abs(z3)/z1 < 0 (z3 signed)
        # With the signed convention (z_ring < 0) this is simply z3/z1 — no
        # explicit negation, the sign rides on z_ring.
        self.i_0 = ring_gear.z / sun_gear.z

    # ------------------------------------------------------------------
    # Geometry / placement
    # ------------------------------------------------------------------

    def planet_phi_deg(self, j: int) -> float:
        """
        Angle of the line of centres sun -> planet j.

        Parameters
        ----------
        j : int
            Planet index, 0 <= j < k.

        Returns
        -------
        float
            (phi0 + 360*j/k) mod 360 [deg], from +Y toward +Z (P11).
        """
        return (self.phi0_deg + 360.0 * j / self.k) % 360.0

    def planet_position(self, j: int) -> tuple[float, float]:
        """
        Position of the axis of planet j in the global YZ frame.

        Parameters
        ----------
        j : int
            Planet index, 0 <= j < k.

        Returns
        -------
        tuple of float
            (y, z) = r_carrier*(cos(phi_j), sin(phi_j)) [mm] with r_carrier =
            al_12 and the sun axis at (0, 0) (P11).
        """
        phi = math.radians(self.planet_phi_deg(j))
        return (self.r_carrier * math.cos(phi), self.r_carrier * math.sin(phi))

    # ------------------------------------------------------------------
    # Kinematics — Willis (§7.1)
    # ------------------------------------------------------------------

    def willis_residual(self, omega_1: float, omega_3: float,
                        omega_H: float) -> float:
        """
        Residual of the Willis equation for given member speeds.

        Parameters
        ----------
        omega_1, omega_3, omega_H : float
            Absolute speeds of the sun, the ring and the carrier, in any
            consistent unit.

        Returns
        -------
        float
            omega_1 - i_0*omega_3 - (1 - i_0)*omega_H, in the unit of the
            speeds; zero for any admissible motion.

        Notes
        -----
        Willis equation in its canonical form, eq. (7.2) of the book named in
        the class docstring (P02). Used for numerical transparency and
        regression tests; not consumed internally except to fill
        ``PlanetaryKinematics.willis_residual``.
        """
        return omega_1 - self.i_0 * omega_3 - (1.0 - self.i_0) * omega_H

    def solve_kinematics(
        self,
        prescribed: dict[PlanetaryMember, float],
    ) -> PlanetaryKinematics:
        """
        Speeds of all members from two prescribed central speeds (Willis).

        Solves the Willis equation (7.2) for the one unknown central velocity.

            i_0 = i_13(H) = omega_1rel/omega_3rel = z3/z1 = -abs(z3)/z1 < 0   (7.1)
            (z3 signed, z_ring < 0 as required by this module)
            omega_1 - i_0*omega_3 - (1 - i_0)*omega_H = 0             (7.2)

        The PGT is a 2-DoF mechanism (three central elements 1/3/H, one
        constraint). Exactly TWO velocities must be prescribed; the third
        follows.

            F = 1  — one central element prescribed at omega = 0.
                     e.g. {RING: 0.0, SUN: w_in}   -> i_1H(3), eq. (7.3)
            F = 2  — differential / summation. Two live drives.
                     e.g. {SUN: w1, RING: w3}      -> omega_H, eq. (7.9)
                          {SUN: w1, CARRIER: wH}   -> omega_3, eq. (7.10)
                          {RING: w3, CARRIER: wH}  -> omega_1, eq. (7.10)

        F is REPORTED, not requested — it is inferred from whether a central
        element sits at zero. Fewer than two prescribed velocities raises:
        Willis alone leaves a family of solutions, not a number.

        Parameters
        ----------
        prescribed : dict
            {PlanetaryMember: omega}; the speeds are in any consistent unit
            (rad/s in the module convention), and the result is in the same
            unit. Only SUN / RING / CARRIER are accepted; the planet spin
            (omega_2) is a dependent coordinate, not a DoF.

        Returns
        -------
        PlanetaryKinematics
            Absolute and carrier-relative speeds, i_0, the mode F, the fixed
            element and the Willis residual.

        Raises
        ------
        ValueError
            If a key is not SUN / RING / CARRIER, or if the number of
            prescribed speeds is not exactly two.

        Notes
        -----
        The missing speed is (P03)::

            omega_1 = i_0*omega_3 + (1 - i_0)*omega_H              (7.10)
            omega_3 = (omega_1 - (1 - i_0)*omega_H) / i_0          (7.10)
            omega_H = (omega_1 - i_0*omega_3) / (1 - i_0)          (7.9)

        and the planet speed follows from the carrier frame, eq. (7.11) (P04)::

            omega_2_rel = -(omega_1 - omega_H)*z1/z2;  omega_2 = omega_2_rel + omega_H

        with z1 and z2 the tooth numbers of sun and planet. F = 1 if a
        prescribed element has speed exactly 0.0, otherwise 2.

        References
        ----------
        .. [1] Arnaudov and Karaivanov, "Planetary Gear Trains", eq. 7.1-7.11,
           as named in the code; not in the library; not checked against the
           source (source to find).
        """
        SUN, RING, CARRIER = (PlanetaryMember.SUN, PlanetaryMember.RING,
                              PlanetaryMember.CARRIER)

        bad = [m for m in prescribed if m not in COAXIAL_MEMBERS]
        if bad:
            raise ValueError(
                f"solve_kinematics: only {[m.name for m in COAXIAL_MEMBERS]} can be "
                f"prescribed, got {[m.name for m in bad]}. The planet spin omega_2 "
                f"is a dependent coordinate, not a degree of freedom."
            )

        dof = 2 - len(prescribed)
        if dof > 0:
            raise ValueError(
                f"solve_kinematics: {len(prescribed)} velocity(ies) prescribed, "
                f"{dof} DoF still open. The PGT is a 2-DoF mechanism — prescribe "
                f"exactly 2 of {[m.name for m in COAXIAL_MEMBERS]}. For the F=1 "
                f"mode fix one element with omega=0.0 and drive "
                f"another; for the F=2 differential give two live "
                f"drive speeds."
            )
        if dof < 0:
            raise ValueError(
                "solve_kinematics: over-constrained — 3 velocities prescribed but "
                "Willis supplies only 1 equation to reconcile them."
            )

        i_0 = self.i_0
        w = dict(prescribed)

        # --- solve for the missing central velocity ---
        if SUN not in w:
            # omega_1 = i_0*omega_3 + (1 - i_0)*omega_H              (7.10)
            w[SUN] = i_0 * w[RING] + (1.0 - i_0) * w[CARRIER]
        elif RING not in w:
            # omega_3 = [omega_1 - (1 - i_0)*omega_H] / i_0          (7.10)
            w[RING] = (w[SUN] - (1.0 - i_0) * w[CARRIER]) / i_0
        else:  # CARRIER not in w
            # omega_H = (omega_1 - i_0*omega_3) / (1 - i_0)          (7.9)
            w[CARRIER] = (w[SUN] - i_0 * w[RING]) / (1.0 - i_0)

        w1, w3, wH = w[SUN], w[RING], w[CARRIER]

        # --- relative velocities, carrier frame (7.11) ---
        z1, z2 = self.sun_gear.z, self.planet_gear.z
        w1_rel = w1 - wH                                    # omega_1(H)
        w3_rel = w3 - wH                                    # omega_3(H)
        w2_rel = -w1_rel * z1 / z2                          # = +w3_rel*|z3|/z2
        w2 = w2_rel + wH                                    # absolute planet spin

        # --- operating-mode DoF: F=1 iff a central element is held at zero ---
        fixed = next((m for m, v in prescribed.items() if v == 0.0), None)
        F = 1 if fixed is not None else 2

        return PlanetaryKinematics(
            omega_1=w1, omega_2=w2, omega_3=w3, omega_H=wH,
            omega_1_rel=w1_rel, omega_2_rel=w2_rel, omega_3_rel=w3_rel,
            i_0=i_0, F=F, fixed=fixed,
            willis_residual=self.willis_residual(w1, w3, wH),
            prescribed=tuple(prescribed.items()),
        )

    # ------------------------------------------------------------------
    # Speed ratios i_AB(C) — closed forms - F=1
    # ------------------------------------------------------------------

    def _ratio_and_equation(self, A: PlanetaryMember, B: PlanetaryMember,
                            C: PlanetaryMember) -> tuple[float, str]:
        """
        Closed form of the speed ratio i_AB(C) and the equation number of the book.

        Single source of truth for ``speed_ratio`` / ``speed_ratio_equation``.
        Written as explicit closed forms rather than derived from
        ``solve_kinematics`` so each branch is traceable to its equation.
        Only valid for F = 1 (one element fixed).

        Parameters
        ----------
        A, B, C : PlanetaryMember
            Input, output and fixed element; a permutation of SUN, RING,
            CARRIER.

        Returns
        -------
        tuple
            (ratio [-], equation number as str, e.g. "7.3").

        Raises
        ------
        ValueError
            If (A, B, C) is not one of the six F = 1 modes.

        Notes
        -----
        The six ratios (P05), with i_0 = z_ring/z_sun < 0: i_1H(3) = 1 - i_0
        (7.3); i_H1(3) = 1/(1 - i_0) (7.4); i_3H(1) = (i_0 - 1)/i_0 (7.5);
        i_H3(1) = i_0/(i_0 - 1) (7.6); i_13(H) = i_0 (7.7); i_31(H) = 1/i_0
        (7.8). Equation numbers are those of Arnaudov and Karaivanov (class
        docstring; source to find).
        """
        SUN, RING, CARRIER = (PlanetaryMember.SUN, PlanetaryMember.RING,
                              PlanetaryMember.CARRIER)
        i_0 = self.i_0
        key = (A, B, C)

        # --- omega_3 = 0 (ring fixed) — the most common case ---
        if key == (SUN, CARRIER, RING):
            return 1.0 - i_0, "7.3"                 # i_1H(3), reducer,    > +1
        if key == (CARRIER, SUN, RING):
            return 1.0 / (1.0 - i_0), "7.4"         # i_H1(3), multiplier, < +1

        # --- omega_1 = 0 (sun fixed) ---
        if key == (RING, CARRIER, SUN):
            return (i_0 - 1.0) / i_0, "7.5"         # i_3H(1), reducer,    > +1
        if key == (CARRIER, RING, SUN):
            return i_0 / (i_0 - 1.0), "7.6"         # i_H3(1), multiplier, < +1

        # --- omega_H = 0 (carrier fixed) — pseudo-PGT, fixed-axis train ---
        if key == (SUN, RING, CARRIER):
            return i_0, "7.7"                       # i_13(H), reducer,    < -1
        if key == (RING, SUN, CARRIER):
            return 1.0 / i_0, "7.8"                 # i_31(H), multiplier, > -1

        raise ValueError(
            f"speed_ratio: (A={A.name}, B={B.name}, C={C.name}) is not one of "
            f"the six F=1 modes. A/B/C must be a permutation of "
            f"{[m.name for m in COAXIAL_MEMBERS]} — the planet can be neither "
            f"input, output nor fixed."
        )

    def speed_ratio(self, A: PlanetaryMember, B: PlanetaryMember,
                    C: PlanetaryMember) -> float:
        """
        Speed ratio i_AB(C) = omega_A / omega_B with element C stationary.

        Book index notation: A input, B output, C fixed; the fixed element goes
        in BRACKETS: i_1H(3), never i_1H^3. Only valid for F = 1 cases.

        Covers all six F=1 modes:

            i_1H(3) = 1 - i_0              reducer,    > +1
            i_H1(3) = 1/(1 - i_0)          multiplier, < +1
            i_3H(1) = (i_0 - 1)/i_0        reducer,    > +1
            i_H3(1) = i_0/(i_0 - 1)        multiplier, < +1
            i_13(H) = i_0                  reducer,    < -1  (pseudo-PGT)
            i_31(H) = 1/i_0                multiplier, > -1  (pseudo-PGT)

        Purely kinematic, no losses. Equations 7.3-7.8 of the book named in
        the class docstring (P05).

        Parameters
        ----------
        A, B, C : PlanetaryMember
            Input, output and fixed element.

        Returns
        -------
        float
            The ratio i_AB(C) [-].
        """
        return self._ratio_and_equation(A, B, C)[0]

    def speed_ratio_equation(self, A: PlanetaryMember, B: PlanetaryMember,
                             C: PlanetaryMember) -> str:
        """
        Equation number of the book behind ``speed_ratio(A, B, C)``.

        Parameters
        ----------
        A, B, C : PlanetaryMember
            Input, output and fixed element (F = 1 cases only).

        Returns
        -------
        str
            Equation number, e.g. "7.3".
        """
        return self._ratio_and_equation(A, B, C)[1]

    @staticmethod
    def mode_of_work(i: float) -> str:
        """
        Classify a speed ratio.

        Parameters
        ----------
        i : float
            Speed ratio [-].

        Returns
        -------
        str
            "reducer" if abs(i) > 1, "multiplier" if abs(i) < 1, else "direct".
        """
        if abs(i) > 1.0:
            return "reducer"
        if abs(i) < 1.0:
            return "multiplier"
        return "direct"

    def is_pseudo_pgt(self, C: PlanetaryMember) -> bool:
        """
        True when the carrier is the fixed element (pseudo-PGT).

        The train then degenerates to a fixed-axis gear train and is not
        planetary at all (the book's "pseudo-PGT"): the planets do not orbit,
        so there is no centrifugal load and the tooth load-cycle count changes.

        Parameters
        ----------
        C : PlanetaryMember
            The fixed element.

        Returns
        -------
        bool
            True if C is CARRIER.
        """
        return C is PlanetaryMember.CARRIER

    def ratio(self, input_member: PlanetaryMember,
              output_member: PlanetaryMember,
              fixed_member: PlanetaryMember) -> float:
        """
        Alias of ``speed_ratio(input_member, output_member, fixed_member)``.

        Returns
        -------
        float
            The ratio i_AB(C) [-].
        """
        return self.speed_ratio(input_member, output_member, fixed_member)

    @property
    def t(self) -> float:
        """
        Torque ratio t = T3/T1 = -i_0 [-] (eq. 6.7, > +1 when abs(z_ring) > z_sun).

        Depends only on i_0, which is fixed once the train is built; unlike
        ``torques``, it needs no input torque (P06).
        """
        return -self.i_0

    # ------------------------------------------------------------------
    # Torque distribution
    # ------------------------------------------------------------------

    def torques(self, T_in: float,
                input_member: PlanetaryMember) -> PlanetaryTorques:
        """
        Ideal external torques on the sun, ring and carrier from one known input torque.

        Independent of which member is fixed: the fixed member simply absorbs
        its share as a reaction.

        Parameters
        ----------
        T_in : float
            Known external torque on ``input_member`` [N m].
        input_member : PlanetaryMember
            SUN, RING or CARRIER.

        Returns
        -------
        PlanetaryTorques
            T1, T3, TH [N m], t [-], Ft and F_H2 [N], T_mesh_sun and
            T_mesh_planet [N m], k.

        Raises
        ------
        ValueError
            If ``input_member`` is not SUN, RING or CARRIER.

        Notes
        -----
        As coded (P06-P08; d1 = reference diameter of the sun [mm], k planets)::

            T1 = T_in (SUN);  T_in/(-i_0) (RING);  T_in/(i_0 - 1) (CARRIER)
            T3 = -i_0 * T1                                          (6.3)
            TH = -(T1 + T3)                                         (6.4)
            t  = -i_0  = T3/T1                                      (6.7)
            T_mesh_sun = T1/k;  T_mesh_planet = T_mesh_sun*z_planet/z_sun
            Ft = 2000*T1/(k*d1)         nominal per-mesh force       (6.1)
            F_H2 = 2*Ft                 ideal resultant planet-pin force (6.2)

        The planet is an idler: what it takes from the sun it hands to the ring,
        scaled by z_planet/z_sun (same Ft, different radius). Equal load sharing
        over the k planets is assumed; ``load_sharing_factor`` is not used here.
        Ft is on the reference diameter of the sun (ISO_6336-1_2019, 4.2.2,
        Formula (1)); ``meshing_12.forces`` and ``meshing_23.forces`` use the
        working pitch radius (see ``PlanetaryTorques``). The planetary
        class computes no F_r, F_a or F_n.

        Warnings
        --------
        Ft uses the reference diameter (ISO_6336-1_2019, 4.2.2, Formula (1)) while
        ``meshing_12.forces`` uses the working pitch radius.
        When z_sun > z_planet, ``meshing_12.forces(T_mesh_sun)`` takes the
        working radius of the planet for the sun (the pair object gives gear1
        the smaller working diameter, see ``SpurHelicalGearMeshing``), so its
        Ft and T_out differ from Ft here and from T_mesh_planet.

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formula (1) and 6.5.8.4 a) (sun peripheral
           load divided by the number of planets; the standard's symbol is p,
           the code's k; K_gamma is not included).
        .. [2] Arnaudov and Karaivanov, "Planetary Gear Trains", eq. 6.1-6.4 and
           6.7, as named in the code; not in the library; not checked against
           the source (source to find). F_H2 = 2 Ft has no other source.
        """
        if input_member not in COAXIAL_MEMBERS:
            raise ValueError(
                f"torques: input_member must be one of "
                f"{[m.name for m in COAXIAL_MEMBERS]}, got {input_member.name}"
            )
        i_0 = self.i_0

        if input_member is PlanetaryMember.SUN:
            T1 = T_in
        elif input_member is PlanetaryMember.RING:
            T1 = T_in / (-i_0)
        else:  # CARRIER
            T1 = T_in / (i_0 - 1.0)

        T3 = -i_0 * T1                          # (6.3)
        TH = -(T1 + T3)                         # (6.4)
        t = -i_0                                # (6.7)

        T_mesh_sun = T1 / self.k
        # planet is an idler: what it takes from the sun it hands to the ring,
        # scaled by z_p/z_s (same Ft, different radius).
        T_mesh_planet = T_mesh_sun * (self.planet_gear.z / self.sun_gear.z)

        Ft = 2000.0 * T1 / (self.k * self.sun_gear.d)      # (6.1)
        F_H2 = 2.0 * Ft                                          # (6.2)

        return PlanetaryTorques(
            T1=T1, T3=T3, TH=TH, t=t, Ft=Ft, F_H2=F_H2,
            T_mesh_sun=T_mesh_sun, T_mesh_planet=T_mesh_planet,
            k=self.k,
        )

    # ------------------------------------------------------------------
    # Forces — NOT IMPLEMENTED YET.
    #
    # torques() (above) covers Ch.6/7: ideal external torques, F=1 kinematics.
    # Actual mesh forces belong after Ch.8 (load distribution/unevenness
    # between planets, K_gamma) and Ch.9 (loading on gear wheels, bearings,
    # planet pins, sun shaft, couplings, carrier) of Arnaudov & Karaivanov —
    # neither is covered yet. Rebuilding from scratch once that ground is
    # covered, rather than patching the earlier draft.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Structural conditions (train-level, not pair-level)
    # ------------------------------------------------------------------

    def coaxiality_error(self) -> float:
        """
        Coaxiality error of the two pairs, |al_12 - al_23|.

        Returns
        -------
        float
            Difference of the working centre distances [mm]; must be ~0 for the
            ring and the sun to be coaxial.
        """
        return abs(self.al_12 - self.al_23)

    def assembly_condition(self) -> tuple[bool, float]:
        """
        Assembly condition for equally spaced planets.

        Returns
        -------
        tuple
            (ok, value): value = (abs(z_ring) + z_sun) / k [-] (P09); ok is
            True when value is an integer within 1e-9.
        """
        val = (abs(self.ring_gear.z) + self.sun_gear.z) / self.k
        return (abs(val - round(val)) < 1e-9, val)

    def neighbouring_condition(self, Safety_Factor: float = 1.1) -> tuple[bool, float, float]:
        """
        Neighbouring condition: adjacent planets must not touch.

        Parameters
        ----------
        Safety_Factor : float, optional
            Factor on the planet tip diameter [-]. Default 1.1.

        Returns
        -------
        tuple
            (ok, available, required) [-, mm, mm]. available = 2*al_12*sin(pi/k)
            is the distance between neighbouring planet axes; required =
            da_planet * Safety_Factor with da_planet the working tip diameter
            of the planet (``meshing_12.gear2_w.da``) (P10). ok is True when
            available > required. For k < 2 the result is (True, inf, 0.0).
        """
        if self.k < 2:
            return (True, math.inf, 0.0)
        available = 2.0 * self.al_12 * math.sin(math.pi / self.k)
        required = self.meshing_12.gear2_w.da * Safety_Factor
        return (available > required, available, required)

# TODO: later add the functions that give i0_max and k_max taking the neighbouring_condition into account

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, coaxiality_tol_mm: float = 1e-4,
                 Safety_Factor: float = 1.1) -> list[str]:
        """
        Check the train: both pairs, signs, coaxiality, assembly, neighbouring.

        Parameters
        ----------
        coaxiality_tol_mm : float, optional
            Allowed |al_12 - al_23| [mm]. Default 1e-4.
        Safety_Factor : float, optional
            Factor of the neighbouring condition [-]. Default 1.1.

        Returns
        -------
        list of str
            One message per violation; empty if the train is valid. Checks:
            the ``validate()`` lists of the two pair objects; k >= 1;
            ring z < 0; sun and planet z > 0; i_0 < 0; coaxiality;
            assembly condition; neighbouring condition; load_sharing_factor
            >= 1.0.

        Warnings
        --------
        The list inherits the messages of ``InternalGearMeshing.validate``, in
        which the trimming test (z - z1) < 2 is always true for z < 0 (as
        coded), so the list is never empty for the signed ring this class
        requires and ``validate_or_raise`` raises.
        """
        errors: list[str] = []
        tag = self.label or self.__class__.__name__

        # 1 — delegate to the two pair objects
        errors.extend(f"{tag}:sun-planet: {e}" for e in self.meshing_12.validate())
        errors.extend(f"{tag}:planet-ring: {e}" for e in self.meshing_23.validate())

        # 2 — planet count and sign convention
        if self.k < 1:
            errors.append(f"{tag}: k must be >= 1, got {self.k}")
        if self.ring_gear.z >= 0:
            errors.append(
                f"{tag}: ring_gear.z must be < 0 (InternalGear signed-z "
                f"convention), got {self.ring_gear.z}. A positive z_ring "
                f"silently flips i_0, the mesh rotation sense and the working "
                f"centre distance."
            )
        if self.sun_gear.z <= 0 or self.planet_gear.z <= 0:
            errors.append(
                f"{tag}: sun/planet are external gears — z must be > 0, got "
                f"z_sun={self.sun_gear.z}, z_planet={self.planet_gear.z}"
            )
        if self.i_0 >= 0:
            errors.append(
                f"{tag}: i_0={self.i_0:.4f} must be < 0 for a sun/ring pair "
                f"(eq. 7.1) — check the z signs."
            )

        # 3 — coaxiality (on the RESOLVED al, not on the requested inputs)
        e_ax = self.coaxiality_error()
        if e_ax > coaxiality_tol_mm:
            errors.append(
                f"{tag}: coaxiality violated — al_12={self.al_12:.4f} mm vs "
                f"al_23={self.al_23:.4f} mm (Δ={e_ax:.4e} mm > {coaxiality_tol_mm:.1e}). "
                f"Adjust x_sun/x_planet/x_ring or impose matching al."
            )

        # 4 — assembly
        ok, val = self.assembly_condition()
        if not ok:
            errors.append(
                f"{tag}: assembly condition violated — (z_ring + z_sun)/k = "
                f"{val:.4f} is not an integer (z_ring={abs(self.ring_gear.z)}, "
                f"z_sun={self.sun_gear.z}, k={self.k}). Equally-spaced planets "
                f"cannot be assembled."
            )

        # 5 — neighbouring
        ok, avail, req = self.neighbouring_condition(Safety_Factor)
        if not ok:
            errors.append(
                f"{tag}: neighbouring condition violated — planet spacing "
                f"{avail:.4f} mm <= da_planet*SF {req:.4f} mm "
                f"(SF={Safety_Factor:.3f}). Reduce k, reduce z_planet, or "
                f"increase al."
            )

        # 6 — load sharing factor sanity
        if self.load_sharing_factor < 1.0:
            errors.append(
                f"{tag}: load_sharing_factor must be >= 1.0 (1.0 = ideal sharing), "
                f"got {self.load_sharing_factor}"
            )

        return errors

    def validate_or_raise(self, **kwargs) -> None:
        """Raise ``ValueError`` with all messages of ``validate(**kwargs)`` if any."""
        errors = self.validate(**kwargs)
        if errors:
            raise ValueError(
                f"PlanetaryGearTrainMeshing '{self.label}' validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line text with the main data of the train (lengths [mm])."""
        tag = self.label or "PlanetaryGearTrainMeshing"
        ok_asm, val_asm = self.assembly_condition()
        ok_nb, avail, req = self.neighbouring_condition()
        i_1H3 = self.speed_ratio(PlanetaryMember.SUN, PlanetaryMember.CARRIER,
                                 PlanetaryMember.RING)
        lines = [
            f"── {tag} ─────────────────────────────────",
            f"  sun    : z={self.sun_gear.z}, mn={self.sun_gear.mn}, x={self.sun_gear.x}",
            f"  planet : z={self.planet_gear.z}, x={self.planet_gear.x}   (k={self.k})",
            f"  ring   : z={abs(self.ring_gear.z)}, x={self.ring_gear.x}",
            f"  i_0    : {self.i_0:.6f}   (7.1, i_13(H))",
            f"  t      : {self.t:.6f}   (6.7, torque ratio, |t|=|i_0|)",
            f"  i_1H(3): {i_1H3:.6f}   (7.3, {self.mode_of_work(i_1H3)})",
            f"  al_12  : {self.al_12:.4f} mm   al_23: {self.al_23:.4f} mm   "
            f"Δ={self.coaxiality_error():.3e} mm",
            f"  assembly    : {'OK' if ok_asm else 'FAIL'}  (z_r+z_s)/k = {val_asm:.4f}",
            f"  neighbouring: {'OK' if ok_nb else 'FAIL'}  {avail:.3f} vs {req:.3f} mm",
            f"  εα (12/23)  : {self.meshing_12.epslon_alpha:.4f} / "
            f"{self.meshing_23.epslon_alpha:.4f}",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """One-line representation: tooth numbers, k, i_0 and al_12."""
        return (f"PlanetaryGearTrainMeshing(z_s={self.sun_gear.z}, "
                f"z_p={self.planet_gear.z}, z_r={abs(self.ring_gear.z)}, "
                f"k={self.k}, i_0={self.i_0:.4f}, al={self.al_12:.4f} mm)")