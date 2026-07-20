"""
axisforge/database/gears/SpurHelicalGears/LoadCapacity_data/Kv_methodC.py
"""

import numpy as np

from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalMeshLink


# Table 11 — K1 and K2 values (ISO 6336-1 Table 11)
_K1_SPUR    = {3: 2.1, 4: 3.9, 5: 7.5, 6: 14.9, 7: 26.8, 8: 39.1, 9: 52.8, 10: 76.6, 11: 102.6}
_K1_HELICAL = {3: 1.9, 4: 3.5, 5: 6.7, 6: 13.3, 7: 23.9, 8: 34.8, 9: 47.0, 10: 68.2, 11:  91.4}
_K2_SPUR    = 0.0193
_K2_HELICAL = 0.0087


class DynamicFactorC:

    def __init__(self,
                 link: SpurHelicalMeshLink,
                 T1: float,
                 rpm_driver: float,
                 KA: float,
                 gear_tolerance_class: int,
                 rotation_dir: float = 1):

        self.link                            = link
        self.meshing: SpurHelicalGearMeshing = link.meshing
        self.rpm_driver                      = rpm_driver
        self.KA                              = KA
        self.gear_tolerance_class            = gear_tolerance_class

        # --- gear ratio ---
        z1, z2 = self.meshing.gear1_w.z, self.meshing.gear2_w.z
        self.u_iso = max(z1, z2) / min(z1, z2)

        pinion = "gear1" if z1 <= z2 else "gear2"
        if self.meshing.driver != pinion:
            pinion_rpm = rpm_driver * self.u_iso
        else:
            pinion_rpm = rpm_driver

        self.pinion     = pinion
        self.pinion_rpm = pinion_rpm

        # --- pitch line velocity at pinion [m/s] ---
        self.v = (np.pi * getattr(self.meshing, f"{pinion}_w").dl / 1000) * pinion_rpm / 60.0

        # --- forces & specific loading ---
        self.forces = self.meshing.forces(T1, self.link.phi_deg, rotation_dir=rotation_dir)
        Ft = abs(self.forces["Ft"])
        self.specific_loading = Ft * KA * self.link.meshing_load_factor / self.meshing.gear1_w.b

        # --- is helical? ---
        self.is_helical = self.meshing.gear1_w.beta_deg > 0.0

        # --- compute Kv ---
        self.Kv = self._Kv()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _K1_K2(self) -> tuple[float, float]:
        """K1 from Table 11 (worst tolerance class); K2 same for all classes."""
        tc = self.gear_tolerance_class
        if tc not in range(3, 12):
            raise ValueError(f"Tolerance class {tc} out of range 3–11.")
        if self.is_helical:
            return _K1_HELICAL[tc], _K2_HELICAL
        return _K1_SPUR[tc], _K2_SPUR

    def _speed_term(self) -> float:
        """(v·z1/100) · √(u²/(1+u²))"""
        z1 = getattr(self.meshing, f"{self.pinion}_w").z
        u  = self.u_iso
        return (self.v * z1 / 100.0) * np.sqrt(u**2 / (1.0 + u**2))

    def _K3(self, speed_term: float) -> float:
        """K3 per formulae (37) and (38)."""
        if speed_term <= 0.2:
            return 2.0
        return -0.357 * speed_term + 2.071

    # ------------------------------------------------------------------
    # Kv (Method C, eq. 36)
    # ------------------------------------------------------------------

    def _Kv(self) -> float:
        speed_term = self._speed_term()
        K3         = self._K3(speed_term)
        w_t        = max(self.specific_loading, 100.0)

        eps_beta = self.meshing.epslon_beta

        # spur case (a) — always needed
        K1_spur, K2_spur = _K1_SPUR[self.gear_tolerance_class], _K2_SPUR
        Kv_alpha = 1.0 + (K1_spur / w_t + K2_spur) * speed_term * K3

        if not self.is_helical or eps_beta >= 1.0:
            # spur gear OR helical with ε_β ≥ 1 → use eq. 36 directly
            return Kv_alpha

        # helical with ε_β < 1 → interpolate with eq. 35
        K1_hel, K2_hel = _K1_HELICAL[self.gear_tolerance_class], _K2_HELICAL
        Kv_beta = 1.0 + (K1_hel / w_t + K2_hel) * speed_term * K3

        return Kv_alpha - eps_beta * (Kv_alpha - Kv_beta)   # eq. 35

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def applicability(self) -> list[str]:
        errors = []
        speed_term = self._speed_term()
        if speed_term > 0.2 and self._K3(speed_term) < 0:
            errors.append(
                f"K3 = {self._K3(speed_term):.3f} < 0 — speed term too high for Method C; "
                f"consider Method B."
            )
        return errors