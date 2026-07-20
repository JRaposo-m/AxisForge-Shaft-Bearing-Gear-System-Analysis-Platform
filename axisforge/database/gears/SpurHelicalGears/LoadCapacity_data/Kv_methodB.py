"""
axisforge/database/gears/SpurHelicalGears/LoadCapacity_data/Kv_methodB.py
"""
import numpy as np

from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalMeshLink
from axisforge.core.materials import get_gear_material 


class DynamicFactor:

    def __init__(self, 
                 link: SpurHelicalMeshLink, 
                 T1: float,
                 rpm_driver: float,
                 moment_inertia_1: float,
                 moment_inertia_2: float,
                 stiffness: float,
                 KA: float,
                 f_pb_eff: float,
                 f_fa_eff: float,
                 rotation_dir: float = 1):
        
        self.link    = link
        self.meshing: SpurHelicalGearMeshing = link.meshing
        self.rpm_driver                      = rpm_driver
        self.moment_inertia_1                = moment_inertia_1
        self.moment_inertia_2                = moment_inertia_2
        self.stiffness                       = stiffness
        self.KA                              = KA
        self.f_pb_eff                        = f_pb_eff
        self.f_fa_eff                        = f_fa_eff

        driver = self.meshing.driver

        z1, z2 = self.meshing.gear1_w.z, self.meshing.gear2_w.z
        self.u_iso = max(z1, z2) / min(z1, z2)

        pinion = "gear1" if z1 <= z2 else "gear2"
        self.wheel  = "gear1" if z1 > z2 else "gear2"

        if driver != pinion:
            pinion_rpm = rpm_driver * self.u_iso
        else:
            pinion_rpm = rpm_driver

        self.pinion = pinion
        self.pinion_rpm = pinion_rpm
        self.v = (np.pi * getattr(self.meshing, f"{pinion}_w").dl / 1000) * pinion_rpm / 60.0

        rb1 = self.meshing.gear1_w.rb
        rb2 = self.meshing.gear1_w.rb
        self.m_red = self.moment_inertia_1 * self.moment_inertia_2 / (
            self.moment_inertia_1 * rb1**2 + self.moment_inertia_2 * rb2**2)
        
        self.n_E_pinion = 30000 * np.sqrt(stiffness/self.m_red) / (np.pi * getattr(self.meshing, f"{pinion}_w").z)
        self.N_pinion = self.pinion_rpm / self.n_E_pinion

        self.forces = self.meshing.forces(T1, self.link.phi_deg, rotation_dir=rotation_dir)
        Ft = abs(self.forces["Ft"])
        self.specific_loading = Ft * KA * self.link.meshing_load_factor / self.meshing.gear1_w.b

        if self.specific_loading < 100:
            self.Ns = 0.5 + 0.35 * np.sqrt(self.specific_loading/(100 * self.meshing.gear_w.b))
        else:
            self.Ns = 0.85

        if self.N_pinion < self.Ns:
            self.range = "Subcritical"
        elif self.N_pinion > self.Ns and self.N_pinion <= 1.15:
            self.range = "Main Resonance"
        elif self.N_pinion > 1.15 and self.N_pinion <= 1.5:
            self.range = "Intermediate"
        else:
            self.range = "Supercritical"

        self.material_pinion = get_gear_material(
            getattr(self.meshing, f"{self.pinion}").material_id
        )

        self.material_wheel = get_gear_material(
            getattr(self.meshing, f"{self.wheel}").material_id
        )

        self._C_constants()
        self.single_stiffness()   # popula self.c_prime
        self.Kv = self.Kv()


    # ------------------------------------------------------------------
    # Functions for diferent ranges
    # ------------------------------------------------------------------

    # NOTE: Here it shall be divided with the helpers and principal functions

    def _C_constants(self) -> None:
        """
        Velocity coefficients Cv1–Cv7 and mesh stiffness factor Cay
        (ISO 6336-1 Table 8).
        Called once at construction; results stored as attributes.
        """
        eg = self.meshing.epslon_gamma

        if eg <= 1.0:
            raise ValueError(
                f"epslon_gamma={eg:.4f} <= 1.0 — contact ratio out of range for Method B."
            )

        # --- Cv1 to Cv6 ---
        if eg <= 2.0:
            Cv1 = 0.32
            Cv2 = 0.34
            Cv3 = 0.23
            Cv4 = 0.90
            Cv5 = 0.47
            Cv6 = 0.47
        else:
            Cv1 = 0.32
            Cv2 = 0.57  / (eg - 0.3)
            Cv3 = 0.096 / (eg - 1.56)
            Cv4 = (0.57 - 0.05 * eg) / (eg - 1.44)
            Cv5 = 0.47
            Cv6 = 0.12  / (eg - 1.74)

        # --- Cv7 ---
        if eg <= 1.5:
            Cv7 = 0.75
        elif eg <= 2.5:
            Cv7 = 0.125 * np.sin(np.pi * (eg - 2.0)) + 0.875
        else:
            Cv7 = 1.0

        # --- Cay ---
        Cay_1 = (1/18) * (self.material_pinion.sigma_Hlim / 97 - 18.45) ** 2 + 1.5
        Cay_2 = (1/18) * (self.material_wheel.sigma_Hlim  / 97 - 18.45) ** 2 + 1.5
        Cay   = 0.5 * (Cay_1 + Cay_2)

        self.Cv1 = Cv1
        self.Cv2 = Cv2
        self.Cv3 = Cv3
        self.Cv4 = Cv4
        self.Cv5 = Cv5
        self.Cv6 = Cv6
        self.Cv7 = Cv7
        self.Cay = Cay


    # ------------------------------------------------------------------
    # Single tooth stiffness c' (ISO 6336-1 Clause 9)
    # ------------------------------------------------------------------

    @staticmethod
    def _q_prime(zn1: float, zn2: float, x1: float, x2: float) -> float:
        """
        Minimum flexibility of a tooth pair (ISO 6336-1 eq. 84).
        Table 13 coefficients for standard rack profile (αp=20°, haP=mn, hfP=1.2mn, ρfP=0.2mn).

        Parameters
        ----------
        zn1, zn2 : virtual (equivalent) number of teeth
        x1, x2   : profile shift coefficients
        """
        C1 = 0.04723
        C2 = 0.15551
        C3 = 0.25791
        C4 = -0.00635
        C5 = -0.11654
        C6 = -0.00193
        C7 = -0.24188
        C8 =  0.00529
        C9 =  0.00182

        return (C1
                + C2 / zn1 + C3 / zn2
                + C4 * x1  + C5 * x1 / zn1
                + C6 * x2  + C7 * x2 / zn2
                + C8 * x1**2 + C9 * x2**2)

    @staticmethod
    def _C_R(bs: float | None, b: float, s_R: float, mn: float) -> float:
        """
        Gear blank factor C_R (ISO 6336-1 eq. 86/87).
        For solid disc gears (bs=None or bs>=b): C_R = 1.0.
        For webbed gears: C_R = 1 + ln(bs/b) / (5·e^(sR/(5·mn)))

        Parameters
        ----------
        bs   : rim width [mm] (None → solid disc)
        b    : face width [mm]
        s_R  : web/rim thickness [mm]
        mn   : normal module [mm]
        """
        if bs is None or bs >= b:
            return 1.0
        return 1.0 + np.log(bs / b) / (5.0 * np.exp(s_R / (5.0 * mn)))

    @staticmethod
    def _C_B(hfp: float, mn: float, alpha_pn_deg: float) -> float:
        """
        Basic rack factor C_B (ISO 6336-1 eq. 88).
        For standard rack (hfP=1.25mn, αpn=20°): C_B ≈ 1.0.

        Parameters
        ----------
        hfp         : dedendum of basic rack [mm]
        mn          : normal module [mm]
        alpha_pn_deg: normal pressure angle of basic rack [°]
        """
        return (1.0 + 0.5 * (1.2 - hfp / mn)) * (1.0 - 0.02 * (20.0 - alpha_pn_deg))

    def _C_B_pair(self) -> float:
        """
        C_B for the pair (ISO 6336-1 eq. 89).
        If pinion and wheel have different rack profiles: C_B = 0.5*(C_B1 + C_B2).
        """
        g1 = self.meshing.gear1
        g2 = self.meshing.gear2
        CB1 = self._C_B(g1.hfP * g1.mn, g1.mn, g1.alpha_n_deg)
        CB2 = self._C_B(g2.hfP * g2.mn, g2.mn, g2.alpha_n_deg)
        return 0.5 * (CB1 + CB2)

    def single_stiffness(self,
                         bs: float | None = None,
                         s_R: float = 0.0,
                         Ky: float = 1.0,
                         steel_only: bool = True) -> float:
        """
        Single tooth stiffness c' [N/(mm·µm)] (ISO 6336-1 eq. 82 / 92).

        Parameters
        ----------
        bs        : rim width [mm] — None for solid disc gear (C_R = 1.0)
        s_R       : web/rim thickness [mm] — only used if bs is not None
        Ky        : load distribution factor along face width (= 1.0 for Method B)
        steel_only: if True, uses steel/steel c'. If False, applies eq. 90
                    correction for mixed materials (E from GearMaterial).

        Returns
        -------
        c_prime : float [N/(mm·µm)]
        """
        g1_w = self.meshing.gear1_w
        g2_w = self.meshing.gear2_w

        # --- theoretical single stiffness c'_th (eq. 83) ---
        q_prime = self._q_prime(g1_w.zn, g2_w.zn, g1_w.x, g2_w.x)
        c_th = 1.0 / q_prime                          # [N/(mm·µm)]

        # --- correction factors ---
        C_M = 0.8                                      # eq. 85 — solid disc
        C_R = self._C_R(bs, self.meshing.b, s_R, g1_w.mn)
        C_B = self._C_B_pair()
        beta = self.meshing.gear1.beta                 # helix angle [rad]

        c_prime = c_th * C_M * C_R * C_B * np.cos(beta)   # eq. 82

        # --- low specific load correction (eq. 92) ---
        specific_load = self.specific_loading          # Ft·KA·Ky / b [N/mm]
        if specific_load < 100.0:
            c_prime = c_th * C_M * C_B * C_R * np.cos(beta) * (specific_load / 100.0) ** 0.25

        # --- material correction for non-steel pairs (eq. 90/91) ---
        if not steel_only:
            E1 = self.material_pinion.E
            E2 = self.material_wheel.E
            E_pair  = 2.0 * E1 * E2 / (E1 + E2)      # eq. 91
            E_St    = 206_000.0                        # [MPa] reference steel
            c_prime = c_prime * (E_pair / E_St)        # eq. 90

        self.c_prime = c_prime
        return c_prime


    def _compute_Bp_Bf_Bk(self) -> tuple[float, float, float]:
        Bp = self.c_prime * self.f_pb_eff / self.specific_loading
        Bf = self.c_prime * self.f_fa_eff / self.specific_loading
        Ca_1 = self.meshing.gear1_w.Ca
        Cf_1 = self.meshing.gear1_w.Cf
        Ca_2 = self.meshing.gear2_w.Ca
        Cf_2 = self.meshing.gear2_w.Cf
        Bk = abs(1 - (self.c_prime * min(Ca_1 + Cf_2, Ca_2 + Cf_1)) / self.specific_loading)
        return Bp, Bf, Bk

    # ------------------------------------------------------------------
    # Kv calculations
    # ------------------------------------------------------------------

    def Kv_subcritical(self, Bp: float, Bf: float, Bk: float) -> float:
        k = (self.Cv1 * Bp) + (self.Cv2 * Bf) + (self.Cv3 * Bk)
        return (self.N_pinion * k) + 1

    def Kv_main_resonance(self, Bp: float, Bf: float, Bk: float) -> float:
        return (self.Cv1 * Bp) + (self.Cv2 * Bf) + (self.Cv4 * Bk) + 1

    def Kv_supercritical(self, Bp: float, Bf: float, Bk: float) -> float:
        return (self.Cv5 * Bp) + (self.Cv6 * Bf) + self.Cv7
    
    def Kv_intermediate(self, Bp: float, Bf: float, Bk: float) -> float:
        Kv_at_115 = self.Kv_main_resonance(Bp, Bf, Bk)   # N = 1.15
        Kv_at_150 = self.Kv_supercritical(Bp, Bf, Bk)    # N = 1.50
        # linear interpolation between 1.15 and 1.50
        Kv = Kv_at_150 + (Kv_at_115 - Kv_at_150) * (1.50 - self.N_pinion) / 0.35
        return Kv

    def Kv(self) -> float:
        Bp, Bf, Bk = self._compute_Bp_Bf_Bk()
        if self.range == "Subcritical":
            return self.Kv_subcritical(Bp, Bf, Bk)
        elif self.range == "Main Resonance":
            return self.Kv_main_resonance(Bp, Bf, Bk)
        elif self.range == "Intermediate":
            return self.Kv_intermediate(Bp, Bf, Bk)
        else:
            return self.Kv_supercritical(Bp, Bf, Bk)

    # ------------------------------------------------------------------
    # Verification for validity
    # ------------------------------------------------------------------

    def applicability(self) -> list[str]:
        errors = []

        u = self.u_iso
        condition = (self.v * self.meshing.gear1_w.z / 100) * np.sqrt(u**2 / (1 + u**2))
        if condition < 3:
            errors.append(
                f"KV Method B not recommended: "
                f"(v·z1/100)·√(u²/(1+u²)) = {condition:.3f} m/s < 3 m/s. "
                f"Use Method C instead."
            )
        return errors