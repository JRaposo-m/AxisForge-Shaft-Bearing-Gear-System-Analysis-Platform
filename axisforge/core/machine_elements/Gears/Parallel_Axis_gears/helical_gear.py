"""
core/mechanical_elements/Gears/Parallel_Axis_gears/helical_gear.py

References:
  - ISO 53:2013    — standard basic rack tooth profile
  - ISO 21771:2007 — cylindrical involute gear geometry
"""

from __future__ import annotations
import numpy as np


class HelicalGear:

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
                 material_id: str = "",
                 position: float = 0.0):
        """
        Parameters
        ----------
        mn          : normal module [mm]
        z           : number of teeth
        x           : profile shift coefficient [-]
        b           : face width [mm]
        alpha_n_deg : normal pressure angle [°]
        beta_n_deg  : helix angle [°]
        haP         : addendum coefficient        (ISO 53: 1.0)
        cP          : tip clearance coefficient   (ISO 53: 0.25)
        rfP         : fillet radius coefficient   (ISO 53: 0.38)
        k           : addendum reduction factor   [-]
        Ra          : arithmetic mean roughness   [µm]
        Rq          : root mean square roughness  [µm]
        Rz          : mean peak-to-valley roughness [µm]
        """
        # --- metadata ---
        self.label       = label
        self.material_id = material_id
        self.position    = position
        
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
        self.hfP         = haP + cP
        self.rfP         = rfP
        self.Ra          = Ra
        self.Rq          = Rq
        self.Rz          = Rz

        # --- reference geometry (ISO 21771) ---
        self.mt     = self.mn / np.cos(self.beta)                     # transverse module [mm]
        self.alphat = np.arctan(np.tan(self.alpha) / np.cos(self.beta))  # transverse pressure angle
        self.betab  = np.arcsin(np.sin(self.beta) * np.cos(self.alpha))  # base helix angle
        self.rhoF   = rfP * mn                                        # fillet radius [mm]
        self.pb     = np.pi * mn * np.cos(self.alpha)                 # base pitch [mm]
        self.pt     = np.pi * self.mt                                 # transverse pitch [mm]
        self.pbt    = self.pt * np.cos(self.alphat)                   # transverse base pitch [mm]
        self.d      = self.mt * z                                     # pitch diameter [mm]
        self.r      = self.d / 2.0                                    # pitch radius [mm]
        self.db     = self.d * np.cos(self.alphat)                    # base diameter [mm]
        self.rb     = self.db / 2.0                                   # base radius [mm]
        self.da     = self.d + 2 * mn * (haP + cP + x)                     # addendum diameter [mm]
        self.ra     = self.da / 2.0                                   # addendum radius [mm]
        self.df     = self.d + 2 * mn * (x - self.hfP)                # dedendum diameter [mm]
        self.rf     = self.df / 2.0                                   # dedendum radius [mm]
        self.zn     = z / (np.cos(self.beta) ** 3)                    # equivalent number of teeth
        self.de     = mn * self.zn                                    # equivalent diameter [mm]

    # ------------------------------------------------------------------
    # Geometry checks
    # ------------------------------------------------------------------

    def undercutting(self) -> bool:
        """
        Returns True if the gear is undercut.

            z_min = 2*cos(β)/sin²(αt) * (haP - x)
        """
        z_min = 2.0 * (np.cos(self.beta) / (np.sin(self.alphat) ** 2)) * (self.haP - self.x)
        return self.z < z_min

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.__class__.__name__

        if self.mn <= 0:
            errors.append(f"{tag}: mn must be > 0, got {self.mn}")
        if self.z < 1:
            errors.append(f"{tag}: z must be >= 1, got {self.z}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")
        if not (0.0 < self.alpha_n_deg < 45.0):
            errors.append(
                f"{tag}: alpha_n_deg out of range (0, 45), got {self.alpha_n_deg}"
            )
        if not (0.0 <= self.beta_n_deg < 45.0):
            errors.append(
                f"{tag}: beta_n_deg out of range [0, 45), got {self.beta_n_deg}"
            )
        if self.Ra < 0:
            errors.append(f"{tag}: Ra must be >= 0, got {self.Ra}")
        if self.Rq < 0:
            errors.append(f"{tag}: Rq must be >= 0, got {self.Rq}")
        if self.Rz < 0:
            errors.append(f"{tag}: Rz must be >= 0, got {self.Rz}")
        if self.undercutting():
            errors.append(
                f"{tag}: gear is undercut — z={self.z}, mn={self.mn}, "
                f"alpha={self.alpha_n_deg}°, beta={self.beta_n_deg}°, x={self.x}"
            )

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))