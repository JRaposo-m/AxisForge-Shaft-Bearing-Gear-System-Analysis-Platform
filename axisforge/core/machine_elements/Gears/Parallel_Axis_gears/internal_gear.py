"""
core/mechanical_elements/Gears/Parallel_Axis_gears/internal_gear.py

Geometry based on:
  - ISO 21771:2007  (involute gear geometry)
  - KHK Gear Technical Reference, §4.2 (Internal Gear dimensions)
  - Gear Solutions: "Internal ring gears – design and considerations"

Sign convention:
  The internal gear allows z to carry a sign: z > 0 for a conventional
  internal (ring) gear definition, but z may be passed as negative to flip
  the diametral relations for certain mesh/assembly conventions used
  upstream. ALL diametral directions are inverted relative to an external
  gear via the z/|z| factor:
    - addendum shrinks the bore   → da = d - 2·mn·(haP - x)   (z>0 case)
    - dedendum enlarges the bore  → df = d + 2·mn·(hfP + x)   (z>0 case)
"""

from __future__ import annotations
import numpy as np


class InternalGear:

    def __init__(self,
                 mn: float,
                 z: int,
                 x: float = 0.0,
                 b: float = 0.0,
                 alpha_n_deg: float = 20.0,
                 beta_n_deg: float = 0.0,
                 haP: float = 1.0,
                 cP: float = 0.25,
                 rfP: float = 0.38,
                 Ra: float = 0.8,
                 Rq: float = 1.0,
                 Rz: float = 4.0,
                 label: str = "",
                 material_id: str = ""):
        """
        Parameters
        ----------
        mn          : normal module [mm]
        z           : number of teeth (sign carries mesh/assembly convention,
                      see module docstring)
        x           : profile shift coefficient [-]  (positive shifts tool outward)
        b           : face width [mm]
        alpha_n_deg : normal pressure angle [°]
        haP         : addendum coefficient of basic rack  (ISO 53: 1.0)
        cP          : tip clearance coefficient           (ISO 53: 0.25)
        rfP         : fillet radius coefficient           (ISO 53: 0.38)
        Ra          : arithmetic mean roughness   [µm]
        Rq          : root mean square roughness  [µm]
        Rz          : mean peak-to-valley roughness [µm]
        """
        # --- metadata ---
        self.label       = label
        self.material_id = material_id

        # --- input parameters ---
        self.mn          = mn
        self.z           = z
        self.x           = x
        self.b           = b
        self.alpha_n_deg = alpha_n_deg
        self.alpha       = np.radians(alpha_n_deg)
        self.beta_n_deg  = beta_n_deg
        self.beta        = np.radians(beta_n_deg)
        self.haP         = haP
        self.cP          = cP
        self.hfP         = haP + cP          # = 1.25 for ISO 53 standard
        self.rfP         = rfP
        self.Ra          = Ra
        self.Rq          = Rq
        self.Rz          = Rz

        # --- reference geometry (ISO 21771) ---
        self.mt   = mn / np.cos(self.beta)
        self.mx   = self.mt / np.tan(self.beta) if self.beta != 0 else np.inf
        self.pn   = np.pi * self.mn                         # circular pitch, normal plane [mm]
        self.pt   = np.pi * self.mt                          # transverse pitch [mm]
        self.pz   = abs(z) * self.mt * np.pi / np.tan(self.beta) if self.beta != 0 else np.inf
        self.px   = np.pi * self.mx

        self.d    = self.mt * abs(z)                         # reference (pitch) diameter [mm]
        self.r    = self.d / 2.0                              # reference radius [mm]

        self.beta_b = np.arcsin(np.sin(self.beta) / np.cos(self.alpha))

        self.db     = abs(z) * self.mn * np.cos(self.alpha) / np.cos(self.beta_b)
        self.rb     = self.db / 2
        self.alphat = np.arccos(self.db / self.d)
        self.tau    = 2 * np.pi / abs(z)                     # angular pitch [rad]

        self.pbt  = self.pt * np.cos(self.alphat)             # transverse base pitch [mm]
        self.pbn  = self.pn * np.cos(self.alpha)               # base pitch, normal plane [mm]

        self.dv   = self.d + 2 * z / abs(z) * self.x * self.mn
        self.rv   = self.dv / 2
        self.da   = self.d + 2 * self.mn * z / abs(z) * (self.x + self.haP + self.cP)
        self.ra   = self.da / 2
        self.df   = self.d - 2 * self.mn * z / abs(z) * (self.hfP - self.x)
        self.rf   = self.df / 2

    def values_at(self, dy: float) -> dict:
        """
        Compute point-specific (radius-dependent) involute quantities at an
        arbitrary diameter dy on the gear flank.

        Parameters
        ----------
        dy : diameter [mm] at which to evaluate the involute geometry.

        Returns
        -------
        dict with keys: beta_y, alpha_yt, alpha_yn, L_y, inv_alpha_yt, p_yt, p_yn
        """
        beta_y       = np.arctan(np.tan(self.beta) * dy / self.d)               # helix angle at dy
        alpha_yt     = np.arccos(self.db / dy)                                   # transverse pressure angle
        alpha_yn     = np.arctan(np.tan(alpha_yt) * np.cos(beta_y))              # normal pressure angle
        L_y          = (self.z / abs(self.z)) * self.db * np.tan(alpha_yt) / 2   # radius of curvature
        inv_alpha_yt = np.tan(alpha_yt) - alpha_yt
        p_yt         = dy * self.pt / self.d
        p_yn         = p_yt * np.cos(beta_y)

        return {
            "beta_y": beta_y,
            "alpha_yt": alpha_yt,
            "alpha_yn": alpha_yn,
            "L_y": L_y,
            "inv_alpha_yt": inv_alpha_yt,
            "p_yt": p_yt,
            "p_yn": p_yn,
        }

    # ------------------------------------------------------------------ #
    # Interference checks                                                  #
    # ------------------------------------------------------------------ #

    def tip_below_base_circle(self) -> bool:
        """
        Returns True if the addendum (tip) circle falls inside the base circle.
        This makes the involute profile non-existent at the tooth tip — invalid geometry.

        Condition:  da > db  must hold  →  d - 2·mn·(haP - x) > d·cos(α)
        """
        return self.da <= self.db

    def involute_interference(self, z1: int, x1: float = 0.0) -> bool:
        """
        Check involute interference between this internal gear (z2, x2=self.x)
        and a mating external spur gear (z1, x1).

        The condition to AVOID involute interference (KHK §4.2 / Gear Solutions):
            αa2 ≤ αw   where αa2 = arccos(db / da)

        Returns True when involute interference IS present.

        Parameters
        ----------
        z1  : number of teeth on the mating external gear
        x1  : profile shift of the mating external gear
        """
        # working pressure angle from involute function
        inv_alpha = np.tan(self.alpha) - self.alpha
        inv_alphaw = inv_alpha + 2 * np.tan(self.alpha) * (x1 - self.x) / (self.z - z1)
        # solve inv(alphaw) numerically via Newton-Raphson
        alphaw = self._solve_inv(inv_alphaw)
        alphaa2 = np.arccos(self.db / self.da)          # pressure angle at tip of internal gear
        return alphaa2 > alphaw

    def trochoid_interference(self, z1: int, x1: float = 0.0) -> bool:
        """
        Trochoid (flank) interference check.
        Rule of thumb for standard gears (α=20°, x=0):  z2 - z1 > 9
        General condition (KHK): αa1 ≤ αw, where αa1 is tip pressure angle of external gear.

        Returns True when trochoid interference IS present.
        """
        d1  = self.mn * z1
        db1 = d1 * np.cos(self.alpha)

        # working pressure angle
        inv_alpha  = np.tan(self.alpha) - self.alpha
        inv_alphaw = inv_alpha + 2 * np.tan(self.alpha) * (x1 - self.x) / (self.z - z1)
        alphaw     = self._solve_inv(inv_alphaw)

        da1    = d1 + 2 * self.mn * (self.haP + x1)    # tip diameter of external gear
        alphaa1 = np.arccos(db1 / da1)
        return alphaa1 > alphaw

    def trimming_interference(self, z1: int) -> bool:
        """
        Trimming interference occurs when the tooth difference is very small.
        Simplified check: z2 - z1 < minimum safe difference.
        The exact limit depends on pinion-cutter geometry (KHK Table 4.7).
        For a quick conservative check: requires z2 - z1 ≥ 2.
        """
        return (self.z - z1) < 2

    def validate_mesh(self, z1: int, x1: float = 0.0) -> list[str]:
        """
        Validate this internal gear against a specific mating external gear
        (pinion). Separate from validate() because interference checks are
        inherently a mesh-pair property, not a standalone-geometry property.

        Parameters
        ----------
        z1 : number of teeth on the mating external gear
        x1 : profile shift of the mating external gear
        """
        errors: list[str] = []
        tag = self.label or self.__class__.__name__

        if self.involute_interference(z1, x1):
            errors.append(
                f"{tag}: involute interference with mating gear (z1={z1}, x1={x1})."
            )
        if self.trochoid_interference(z1, x1):
            errors.append(
                f"{tag}: trochoid (flank) interference with mating gear (z1={z1}, x1={x1})."
            )
        if self.trimming_interference(z1):
            errors.append(
                f"{tag}: trimming interference — tooth difference z2-z1={self.z - z1} < 2."
            )

        return errors

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.__class__.__name__

        if self.mn <= 0:
            errors.append(f"{tag}: mn must be > 0, got {self.mn}")
        if self.z == 0:
            errors.append(f"{tag}: z must be != 0, got {self.z}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")
        if not (0.0 < self.alpha_n_deg < 45.0):
            errors.append(f"{tag}: alpha_n_deg out of range (0, 45), got {self.alpha_n_deg}")
        if self.Ra < 0:
            errors.append(f"{tag}: Ra must be >= 0, got {self.Ra}")
        if self.Rq < 0:
            errors.append(f"{tag}: Rq must be >= 0, got {self.Rq}")
        if self.Rz < 0:
            errors.append(f"{tag}: Rz must be >= 0, got {self.Rz}")
        if self.da >= self.d:
            errors.append(
                f"{tag}: tip diameter da={self.da:.4f} ≥ pitch diameter d={self.d:.4f}. "
                "Internal gear addendum must reduce da below d."
            )
        if self.tip_below_base_circle():
            errors.append(
                f"{tag}: tip circle (da={self.da:.4f}) ≤ base circle (db={self.db:.4f}). "
                "No valid involute at tooth tip — increase z or x, or reduce haP."
            )

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------ #
    # Helper                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _solve_inv(inv_target: float, alpha0: float = 0.3491) -> float:
        """
        Solve the involute function:  inv(α) = tan(α) - α = inv_target
        via Newton-Raphson (converges in ~5 iterations for typical gear angles).
        """
        alpha = alpha0
        for _ in range(50):
            f  = np.tan(alpha) - alpha - inv_target
            df = np.tan(alpha) ** 2          # d/dα [tan(α) - α] = tan²(α)
            alpha -= f / df
            if abs(f) < 1e-12:
                break
        return alpha