"""
core/mechanical_elements/Bearings/bearing.py

Rolling bearing: catalog input data for a single bearing position.

References:
  - ISO 281:2007   — dynamic load rating, life calculation, X/Y factors
  - ISO 76:2006    — static load rating
"""

from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq

from axisforge.core.machine_elements.Bearings.bearing_types import BearingType


class Bearing:

    def __init__(self,
                 bearing_type: BearingType = BearingType.DEEP_GROOVE_BALL,
                 designation: str = "",
                 b: float = 0.0,
                 d: float = 0.0,
                 D: float = 0.0,
                 d1: float = 0.0,
                 re: float = 0.0,
                 ri: float = 0.0,
                 C: float = 0.0,
                 C0: float = 0.0,
                 arrangement: str = "locating",
                 single: bool = True,
                 contact_angle_deg: float = 0.0,
                 X: float = 1.0,
                 Y: float = 0.0,
                 label: str = "",
                 position: float = 0.0):
        """
        Parameters
        ----------
        position          : axial coordinate along the shaft [mm]
        bearing_type       : BearingType enum — catalog family
        designation         : manufacturer designation, e.g. "6208" (informational only)
        C                  : dynamic load rating [N]        (ISO 281)
        C0                 : static load rating [N]         (ISO 76)
        arrangement         : "fixed" (locating, restrains axial) | "floating"
        contact_angle_deg   : nominal contact angle [°]      (0 = radial)
        X, Y                : dynamic equivalent load factors (ISO 281 Table)
                              static input for now — later derived from Fa/C0
        Kr                 : radial stiffness [N/mm]         (FEM 1D input)
        Ka                 : axial stiffness [N/mm]          (required if arrangement="fixed")
        label               : identifier for reporting/traceability
        """
        # --- metadata ---
        self.label       = label
        self.designation = designation

        # --- catalog / rating data ---
        self.bearing_type = bearing_type
        self.b            = b
        self.d            = d
        self.D            = D
        self.d1           = d1
        self.re           = re      # external raceway groove curvature radius [mm]
        self.ri           = ri      # internal raceway groove curvature radius [mm]
        self.C            = C
        self.C0           = C0
        self.contact_angle_deg = contact_angle_deg
        self.contact_angle      = np.radians(contact_angle_deg)

        self.dm = 0.5 * (self.d + self.D)  # mean diameter [mm]

        # --- ISO 281 equivalent load factors (static input, fase 1) ---
        self.X = X
        self.Y = Y

        # --- mounting / FEM 1D ---
        self.position    = position
        self.arrangement = arrangement
        self.single      = single

        if not single:
            self.b = 2 * b  # double row bearing width
            self.C = 2 * C  # double row bearing dynamic load rating
            self.C0 = 2 * C0  # double row bearing static load rating
            # verificar e fazer em vez disto ver no ShaftSystem quais as propriedades associadas!!!

        if bearing_type == BearingType.DEEP_GROOVE_BALL:
            self.Rx_e, self.Ry_e, self.R_e, self.T_e = self.curvature_radii(self.re)
            self.Rx_i, self.Ry_i, self.R_i, self.T_i = self.curvature_radii(self.ri)
            
            self.k_e = 1.18 * ((self.Ry_e/self.Rx_e) **0.598) - 0.19
            self.k_i = 1.18 * ((self.Ry_i/self.Rx_i) **0.598) - 0.19
            
            self.F_e, self.S_e = self.ellipse_parameters(self.k_e)
            self.F_i, self.S_i = self.ellipse_parameters(self.k_i)


    # ------------------------------------------------------------------
    # Derived / convenience
    # ------------------------------------------------------------------

    def is_locating(self) -> bool:
        """True if this bearing restrains axial displacement (fixed side)."""
        return self.arrangement == "locating"

    def equivalent_dynamic_load(self, Fr: float, Fa: float) -> float:
        """
        P = X·Fr + Y·Fa   (ISO 281)

        X, Y are taken as fixed inputs for this phase — no lookup on Fa/C0
        is performed yet.
        """
        return self.X * Fr + self.Y * Fa
    
    # ------------------------------------------------------------------
    # Ball Bearing stiffness
    # ------------------------------------------------------------------

    def curvature_radii(self, r: float) -> tuple[float, float, float, float]:
        f = r / self.D
        y = self.D * np.cos(self.contact_angle) / self.dm

        Rx = (1 - y) * self.D / 2
        Ry = f * self.D / (2 * f - 1)
        R = 1 / (1 / Rx + 1 / Ry)
        T = R * (1 / Ry - 1 / Rx)

        return Rx, Ry, R, T

    def ellipse_parameters(self, k: float) -> tuple[float, float]:
        """
        F (1st kind), S (2nd kind) — solved EXACTLY via scipy given kappa.
        """
        m = 1.0 - 1.0 / k**2
        F = ellipk(m)
        S = ellipe(m)
        return F, S

    def deformation(self, Q: float, E: float) -> float:
        """
        Total normal approach delta_n = delta_i + delta_e (eq. 18-19),
        from ball load Q and effective elastic modulus E (E*).
        """
        delta_e = self.F_e / (2 * self.S_e * self.R_e)**(1/3) * ((3 * Q) / (np.pi * self.k_e * E))**(2/3)
        delta_i = self.F_i / (2 * self.S_i * self.R_i)**(1/3) * ((3 * Q) / (np.pi * self.k_i * E))**(2/3)

        return delta_e + delta_i
    
    def contact_stiffness(self, kappa: float, F: float, S: float, R: float, E: float) -> float:
        """
        Single-contact Hertzian stiffness (Ki or Ke, eq. 16):

            K = (pi/3) * kappa * E * sqrt(2*S*R / F**3)
        """
        return (np.pi / 3.0) * kappa * E * np.sqrt(2 * S * R / F**3)

    def ball_stiffness(self, E: float) -> float:
        """
        Ball stiffness along the normal, Kn (eq. 20) — combines the inner
        and outer race contact stiffness in series:

            Kn = (1/Ki^(2/3) + 1/Ke^(2/3))^(-3/2)

        Kn does not depend on load — it is a fixed geometric/material
        property of the ball-race contact pair, computed once.
        """
        Ki = self.contact_stiffness(self.k_i, self.F_i, self.S_i, self.R_i, E)
        Ke = self.contact_stiffness(self.k_e, self.F_e, self.S_e, self.R_e, E)

        return (Ki**(-2.0/3.0) + Ke**(-2.0/3.0)) ** (-1.5)


    # ------------------------------------------------------------------
    # Axial deflection
    # ------------------------------------------------------------------

    def axial_deflection_single(self, Fa:float, Preload:float, Z:float, K_n:float):

        delta_a = (Fa / (Z * K_n * (np.sin(self.contact_angle))**(5/2)))**(2/3)
        axial_stiffness = 1.5 * ((Z * K_n)**(2/3)) * np.sin(self.contact_angle)**(5/3) * Preload **(1/3)

        return delta_a, axial_stiffness
    
    def preload_deflections(self, P: float, Z1: float, Kn1: float, alpha1: float,
                                     Z2: float, Kn2: float, alpha2: float) -> tuple[float, float]:
        """
        Initial axial deflection on each bearing under pure preload P
        (eq. 26, applied individually to each bearing — closed form,
        no iteration needed since Fa=P is directly known here).
        """
        delta_a1_0 = (P / (Z1 * Kn1 * np.sin(alpha1)**2.5)) ** (2.0/3.0)
        delta_a2_0 = (P / (Z2 * Kn2 * np.sin(alpha2)**2.5)) ** (2.0/3.0)
        e = delta_a1_0 + delta_a2_0
        return delta_a1_0, delta_a2_0, e
    
    def axial_deflection_paired(self, A: float, e: float,
                              Z1: float, Kn1: float, alpha1: float,
                              Z2: float, Kn2: float, alpha2: float) -> tuple[float, float, float, float]:
        """
        General (asymmetric) paired angular-contact bearing, back-to-back,
        under external axial load A, with total relative approach e fixed
        by preload (rigid rings, eq. 30: delta_a1 + delta_a2 = e).

        Reduces to Guay eq. 33-36 when bearing 1 == bearing 2.

        Parameters
        ----------
        A  : external axial load [N] (eq. 29)
        e  : total axial deflection under preload (delta_a1_0 + delta_a2_0),
            from preload_deflections()
        """
        def residual(delta_a1: float) -> float:
            delta_a2 = e - delta_a1
            Fa1 = Z1 * Kn1 * np.sin(alpha1)**2.5 * delta_a1**1.5
            Fa2 = Z2 * Kn2 * np.sin(alpha2)**2.5 * max(delta_a2, 0.0)**1.5
            return (Fa1 - Fa2) - A

        # bearing 1 alone can only carry loads with delta_a1 in (0, e)
        # (beyond e, gapping — bearing 2 fully unloaded, see eq. 32)
        delta_a1 = brentq(residual, 1e-12, e - 1e-12)
        delta_a2 = e - delta_a1

        Fa1 = Z1 * Kn1 * np.sin(alpha1)**2.5 * delta_a1**1.5
        Fa2 = Z2 * Kn2 * np.sin(alpha2)**2.5 * delta_a2**1.5

        return delta_a1, delta_a2, Fa1, Fa2
    
    def axial_stiffness_paired(self, delta_a1: float, delta_a2: float,
                             Z1: float, Kn1: float, alpha1: float,
                             Z2: float, Kn2: float, alpha2: float) -> float:
        """
        Tangent axial stiffness of the paired bearing at the current
        equilibrium (delta_a1, delta_a2). Reduces to ka = 6P/e (eq. 36)
        when both bearings are identical and A=0 (delta_a1=delta_a2=e/2).
        """
        ka1 = 1.5 * Z1 * Kn1 * np.sin(alpha1)**2.5 * delta_a1**0.5
        ka2 = 1.5 * Z2 * Kn2 * np.sin(alpha2)**2.5 * delta_a2**0.5
        return ka1 + ka2
    

    # ------------------------------------------------------------------
    # Auxiliary / convenience
    # ------------------------------------------------------------------

    def equivalent_modulus_of_elasticity(self, E1: float, E2: float, v1: float, v2: float) -> float:
        """
        E* = 1 / ( (1-v1^2)/E1 + (1-v2^2)/E2 )   (Hertzian contact)
        """
        return 1.0 / ((1.0 - v1**2) / E1 + (1.0 - v2**2) / E2)


    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.designation or self.__class__.__name__

        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.C < 0:
            errors.append(f"{tag}: C must be >= 0, got {self.C}")
        if self.C0 < 0:
            errors.append(f"{tag}: C0 must be >= 0, got {self.C0}")
        if self.arrangement not in ("locating", "floating", "non-locating"):
            errors.append(
                f"{tag}: arrangement must be 'locating' or 'floating' or 'non-locating', got '{self.arrangement}'"
            )
        if not (0.0 <= self.contact_angle_deg < 90.0):
            errors.append(
                f"{tag}: contact_angle_deg must be in [0, 90), got {self.contact_angle_deg}"
            )
        if not (0.0 < self.X <= 1.0):
            errors.append(f"{tag}: X (radial factor) must be in (0, 1], got {self.X}")
        if self.Y < 0:
            errors.append(f"{tag}: Y (axial factor) must be >= 0, got {self.Y}")


        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or self.designation or "Bearing"
        lines = [
            f"── {tag} ─────────────────────────────────",
            f"  type        : {self.bearing_type.name}",
            f"  position    : {self.position:.2f} mm",
            f"  arrangement : {self.arrangement}",
            f"  C / C0      : {self.C:.0f} N / {self.C0:.0f} N",
            f"  X / Y       : {self.X:.3f} / {self.Y:.3f}",
            f"  Kr / Ka     : {self.Kr:.2e} N/mm / "
            f"{'—' if self.Ka is None else f'{self.Ka:.2e}'} N/mm",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Bearing(type={self.bearing_type.name}, position={self.position:.2f} mm, "
            f"arrangement='{self.arrangement}', Kr={self.Kr:.2e} N/mm)"
        )
    


