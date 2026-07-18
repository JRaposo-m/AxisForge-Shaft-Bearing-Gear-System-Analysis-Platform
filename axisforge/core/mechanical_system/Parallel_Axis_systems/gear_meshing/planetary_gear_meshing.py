"""
axisforge/core/mechanical_system/Parallel_Axis_systems/gear_meshing/planetary_gear_meshing.py

Single-stage planetary (epicyclic) train: sun + k planets + ring + carrier.

Composition, not inheritance: the two tooth-contact problems are delegated to
the existing pair objects —
    meshing_12 : SpurHelicalGearMeshing(sun, planet)    external pair
    meshing_23 : InternalGearMeshing(planet, ring)      internal pair

This module owns ONLY what is specific to the train:
  - structural conditions (coaxiality, assembly, neighbouring)
  - kinematics (Willis), F = 1 and F = 2 modes
  - ideal torque distribution over the members and over the k planets (Ch.6)
  - per-planet angular placement (phi_j)

NOT YET covered (Ch.8/9 of Arnaudov & Karaivanov): load distribution /
unevenness between planets (K_gamma), and actual mesh/bearing/pin forces.
`torques()` gives IDEAL torques only (K_gamma=1, eta_0=1, omega=const).

References
----------
  - ISO 21771:2007
  - KHK Gear Technical Reference §4.2 / §5 (planetary trains)
  - Arnaudov & Karaivanov, "Planetary Gear Trains", §6.1-6.3 (forces/torques,
    ideal Ft/F_H2/t) and §7.1 (Willis analytical method); equation numbers
    6.1-6.7 and 7.1-7.11 below refer to this text.
  - Henriot, "Traité théorique et pratique des engrenages", ch. trains épicycloïdaux
  - MAAG Gear Book

Willis index notation (§7.1)
----------------------------
    1 = sun,  2 = planet,  3 = ring,  H = carrier
    A = input element, B = output element, C = fixed element
    i_AB(C) — the FIXED element goes in brackets: i_1H(3), never i_1H^3.

Conventions
-----------
  - All members rotate about the global +X axis; omega [rad/s], +X right-hand.
  - Torque in N·m, forces in N, lengths in mm  (same as the pair objects).
  - phi_deg: line of centres in the global YZ frame, +Y toward +Z.
    Planet j sits at phi_j = phi_0 + j*360/k  (sun -> planet direction).
    The planet -> ring line of centres is phi_j + 180 (ring centre == sun centre).
  - SIGN CONVENTION: z_sun, z_planet > 0 (external); z_ring < 0 (InternalGear
    signed-z convention). The negative z_ring is STRUCTURAL, not cosmetic — it
    is what makes InternalGearMeshing's z1+z2 collapse to the difference form
    and what makes rotation_dir_out = +rotation_dir_in (ring and planet turn the
    same way). Use abs(z_ring) only where a tooth COUNT is meant (assembly
    condition); use the signed value wherever a RATIO is meant (i_0).

Units note: torque propagates in N·m end-to-end, matching *GearMeshing.forces().
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.internal_gear import InternalGear
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.internal_gear_meshing import InternalGearMeshing
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing


# ===========================================================================
# Member tagging — the anchor for torque flow / future PlanetarySystem topology
# ===========================================================================

class PlanetaryMember(Enum):
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
    Currently unused — was consumed only by the removed forces() draft.
    Kept as a likely building block for the Ch.8/9 rebuild.
    """
    SUN_PLANET = auto()     # external pair, meshing_12
    PLANET_RING = auto()    # internal pair, meshing_23


@dataclass(frozen=True)
class PlanetaryKinematics:
    """
    Result of solve_kinematics(), in the Willis index notation:
        1 = sun, 2 = planet, 3 = ring, H = carrier.
    omega in rad/s about +X. Absolute velocities are omega_N; velocities
    relative to the carrier (eq. 7.11) are omega_N_rel = omega_N(H).

    F : degrees of freedom of the OPERATING MODE, not of the mechanism.
        The PGT is inherently 2-DoF; prescribing one central element at
        omega = 0 reduces it to F = 1. F = 2 is the differential /
        summation mode.
    willis_residual : LHS of eq. (7.2) evaluated at the solution. Should be
        ~0; exposed for numerical transparency, not consumed internally.
    """
    omega_1: float                 # sun,     absolute
    omega_2: float                 # planet,  absolute spin about its own axis
    omega_3: float                 # ring,    absolute
    omega_H: float                 # carrier, absolute
    omega_1_rel: float             # = omega_1 - omega_H          (7.11)
    omega_2_rel: float             # = -omega_1_rel*z1/z2         (7.11)
    omega_3_rel: float             # = omega_3 - omega_H          (7.11)
    i_0: float                     # basic ratio i_13(H) = -z3/z1 < 0   (7.1)
    F: int                         # 1 or 2 — operating-mode DoF
    fixed: PlanetaryMember | None  # the element held at omega = 0, if any
    willis_residual: float
    prescribed: tuple[tuple[PlanetaryMember, float], ...] = ()

    # --- descriptive aliases (the rest of the module reads these) ---
    @property
    def omega_sun(self) -> float:        return self.omega_1
    @property
    def omega_planet_abs(self) -> float: return self.omega_2
    @property
    def omega_planet_rel(self) -> float: return self.omega_2_rel
    @property
    def omega_ring(self) -> float:       return self.omega_3
    @property
    def omega_carrier(self) -> float:    return self.omega_H


@dataclass(frozen=True)
class PlanetaryTorques:
    """
    Ideal external torques (1=sun, 3=ring, H=carrier). Valid under three
    preconditions: stationary load (omega_i = const, no inertial forces),
    K_gamma = 1 (equal load sharing — load_sharing_factor is the extension
    point), eta_0 = 1 (no losses).

        T3 = -i_0 * T1 = +(z3/z1) * T1                              (6.3)
        TH = -(T1 + T3)                                             (6.4)
        sum(Ti) = T1 + T3 + TH = 0                                  (6.5)
        t  = T3 / T1 = -i_0 > +1     (torque ratio)                 (6.7)
        Ft = F_t12 = F_t32 = 2000*T1/(k*d1)   [N]  (nominal, hand-calc,
             uses the REFERENCE diameter d1 — NOT the working pitch diameter
             dl1 that meshing_12.forces() uses; the two agree only at x=0.
             meshing_12/meshing_23.forces() remain the precise source of
             truth for stress/bearing loads. Ft here is a cross-check only.)  (6.1)
        F_H2 = F_t12 + F_t32 = 2*Ft   (ideal resultant on ONE planet pin,
             shifted-mesh approximation)                            (6.2)

    T_mesh_sun / T_mesh_planet are the per-mesh torques actually fed to
    meshing_12.forces() / meshing_23.forces().
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

    # --- descriptive aliases (older call sites in this module read these) ---
    @property
    def T_sun(self) -> float: return self.T1
    @property
    def T_ring(self) -> float: return self.T3
    @property
    def T_carrier(self) -> float: return self.TH
    @property
    def T_sun_planet_mesh(self) -> float: return self.T_mesh_sun
    @property
    def T_planet_ring_mesh(self) -> float: return self.T_mesh_planet

    def satisfies_torque_ordering(self) -> bool:
        """eq. (6.6): |T1| < |T3| < |TH|. A cheap sanity check, not enforced."""
        return abs(self.T1) < abs(self.T3) < abs(self.TH)


# ===========================================================================
# PlanetaryGearTrainMeshing
# ===========================================================================

class PlanetaryGearTrainMeshing:
    """
    Parameters
    ----------
    sun_gear, planet_gear : SpurHelicalGear   (external, z > 0)
    ring_gear             : InternalGear      (z < 0, signed-z convention)
    k                     : number of planets, >= 1 (equally spaced)
    phi0_deg              : angular position of planet 0 [deg]
    al_12 / al_23         : working centre distances [mm] (None -> derived from x)
    equalise_gs_*, addendum_reduction_* : forwarded to each pair object.

    Notes
    -----
    Equal load sharing over the k planets is ASSUMED (no mesh-stiffness /
    manufacturing-error load-sharing factor yet). A K_gamma-style factor is the
    natural extension point — see `load_sharing_factor`.
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

        # basic ratio (7.1):  i_0 = i_13(H) = omega_1rel/omega_3rel = -z3/z1 < 0
        # With the signed convention (z_ring < 0) this is simply z3/z1 — no
        # explicit negation, the sign rides on z_ring.
        self.i_0 = ring_gear.z / sun_gear.z

    # ------------------------------------------------------------------
    # Geometry / placement
    # ------------------------------------------------------------------

    def planet_phi_deg(self, j: int) -> float:
        """Line-of-centres angle sun -> planet j [deg], equally spaced."""
        return (self.phi0_deg + 360.0 * j / self.k) % 360.0

    def planet_position(self, j: int) -> tuple[float, float]:
        """(y, z) of planet j's axis in the global frame [mm], sun axis at (0,0)."""
        phi = math.radians(self.planet_phi_deg(j))
        return (self.r_carrier * math.cos(phi), self.r_carrier * math.sin(phi))

    # ------------------------------------------------------------------
    # Kinematics — Willis (§7.1)
    # ------------------------------------------------------------------

    def willis_residual(self, omega_1: float, omega_3: float,
                        omega_H: float) -> float:
        """
        LHS of the Willis equation in its canonical form (eq. 7.2):

            omega_1 - i_0*omega_3 - (1 - i_0)*omega_H = 0

        Zero for any admissible motion. Exposed for numerical transparency /
        regression tests — never consumed internally.
        """
        return omega_1 - self.i_0 * omega_3 - (1.0 - self.i_0) * omega_H

    def solve_kinematics(
        self,
        prescribed: dict[PlanetaryMember, float],
    ) -> PlanetaryKinematics:
        """
        Solve the Willis equation (7.2) for the one unknown central velocity.

            i_0 = i_13(H) = omega_1rel/omega_3rel = -z3/z1 < 0        (7.1)
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
        prescribed : {PlanetaryMember: omega [rad/s]}. Only SUN / RING /
                     CARRIER are accepted — the planet spin (omega_2) is a
                     dependent coordinate, not a DoF.
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
        Closed form of i_AB(C) and the book equation number backing it.
        Single source of truth for speed_ratio() / speed_ratio_equation().

        Written as explicit closed forms rather than derived from
        solve_kinematics() so each branch is traceable to its equation and
        directly regression-testable against it.

        NOTE: Only valid for F=1 cases

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
        Speed ratio i_AB(C) = omega_A / omega_B with element C stationary,
        in the book's index notation ("A—input, B—output, C—fixed").
        Note the fixed element belongs in BRACKETS: i_1H(3), never i_1H^3.
        NOTE: Only valid for F=1 cases

        Covers all six F=1 modes:

            i_1H(3) = 1 - i_0              reducer,    > +1
            i_H1(3) = 1/(1 - i_0)          multiplier, < +1
            i_3H(1) = (i_0 - 1)/i_0        reducer,    > +1
            i_H3(1) = i_0/(i_0 - 1)        multiplier, < +1
            i_13(H) = i_0                  reducer,    < -1  (pseudo-PGT)
            i_31(H) = 1/i_0                multiplier, > -1  (pseudo-PGT)

        Purely kinematic — no losses.
        """
        return self._ratio_and_equation(A, B, C)[0]

    def speed_ratio_equation(self, A: PlanetaryMember, B: PlanetaryMember,
                             C: PlanetaryMember) -> str:
        """Book equation number backing speed_ratio(A, B, C) — traceability.
        NOTE: Only valid for F=1 cases
        """
        return self._ratio_and_equation(A, B, C)[1]

    @staticmethod
    def mode_of_work(i: float) -> str:
        """'reducer' if |i| > 1, 'multiplier' if |i| < 1."""
        if abs(i) > 1.0:
            return "reducer"
        if abs(i) < 1.0:
            return "multiplier"
        return "direct"

    def is_pseudo_pgt(self, C: PlanetaryMember) -> bool:
        """
        True when the CARRIER is the fixed element: the train degenerates to a
        fixed-axis gear train and is not planetary at all — the book's
        'pseudo-PGT'. Downstream consequence: the
        planets do not orbit, so there is NO centrifugal load and the tooth
        load-cycle count changes.
        """
        return C is PlanetaryMember.CARRIER

    def ratio(self, input_member: PlanetaryMember,
              output_member: PlanetaryMember,
              fixed_member: PlanetaryMember) -> float:
        """Backwards-compatible alias for speed_ratio()."""
        return self.speed_ratio(input_member, output_member, fixed_member)

    @property
    def t(self) -> float:
        """
        Torque ratio (6.7): t = T3/T1 = -i_0 > +1 for an AI-PGT. Depends only
        on i_0 (fixed once the train is built) — unlike torques(), which also
        needs T_in.
        """
        return -self.i_0

    # ------------------------------------------------------------------
    # Torque distribution
    # ------------------------------------------------------------------

    def torques(self, T_in: float,
                input_member: PlanetaryMember) -> PlanetaryTorques:
        """
        Ideal external torques on the three coaxial members from one known
        input torque. Independent of which member is fixed — the fixed member
        simply absorbs its share as a reaction.

            T3 = -i_0 * T1                                          (6.3)
            TH = -(T1 + T3)                                         (6.4)
            t  = -i_0  = T3/T1                                      (6.7)
            Ft = 2000*T1/(k*d1)         nominal per-mesh force       (6.1)
            F_H2 = 2*Ft                 ideal resultant planet-pin force (6.2)

        Ft/F_H2 use the book's simplified reference-diameter formula — see
        PlanetaryTorques docstring for why they are cross-checks, not the
        precise mesh forces (those come from forces() below).

        Torque through ONE mesh assumes ideal load sharing over k planets,
        scaled by load_sharing_factor (K_gamma placeholder).
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
        """|al_12 - al_23| [mm]. Must be ~0 for the ring and sun to be coaxial."""
        return abs(self.al_12 - self.al_23)

    def assembly_condition(self) -> tuple[bool, float]:
        """
        Equally-spaced planets require (z_ring + z_sun) / k to be an integer.
        Returns (ok, value).
        """
        val = (abs(self.ring_gear.z) + self.sun_gear.z) / self.k
        return (abs(val - round(val)) < 1e-9, val)

    def neighbouring_condition(self, Safety_Factor: float = 1.1) -> tuple[bool, float, float]:
        """
        Adjacent planets must not touch:
            2 * al_12 * sin(pi/k) > da_planet * Safety_Factor
        Returns (ok, available_spacing, required).
        """
        if self.k < 2:
            return (True, math.inf, 0.0)
        available = 2.0 * self.al_12 * math.sin(math.pi / self.k)
        required = self.meshing_12.gear2_w.da * Safety_Factor
        return (available > required, available, required)

# depois incluir as funções que permitem obter o i0_max e o k_max tendo em conta a neighbouring_condition

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, coaxiality_tol_mm: float = 1e-4,
                 Safety_Factor: float = 1.1) -> list[str]:
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
        return (f"PlanetaryGearTrainMeshing(z_s={self.sun_gear.z}, "
                f"z_p={self.planet_gear.z}, z_r={abs(self.ring_gear.z)}, "
                f"k={self.k}, i_0={self.i_0:.4f}, al={self.al_12:.4f} mm)")