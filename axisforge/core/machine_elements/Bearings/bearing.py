"""
core/mechanical_elements/Bearings/bearing.py

Rolling bearing: catalog input data for a single bearing position.

References:
  - ISO 281:2007   — dynamic load rating, life calculation, X/Y factors
  - ISO 76:2006    — static load rating
  - ISO/TS 16281   — internal load distribution, Hertz contact, stiffness
"""

from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import brentq

from axisforge.core.machine_elements.Bearings.bearing_types import BearingType


class Bearing:

    def __init__(self,
                 d: float,
                 D: float,
                 bearing_type: BearingType = BearingType.DEEP_GROOVE_BALL,
                 designation: str = "",
                 b: float = 0.0,
                 C: float = 0.0,
                 C0: float = 0.0,
                 X: float = 1.0,
                 Y: float = 0.0,
                 arrangement: str = "locating",
                 contact_angle_deg: float = 0.0,
                 label: str = "",
                 position: float = 0.0):
        """
        Minimal catalog-level bearing definition.

        Parameters
        ----------
        d                 : bore diameter [mm]
        D                 : outer diameter [mm]
        bearing_type      : BearingType enum — catalog family
        designation       : manufacturer designation, e.g. "6208" (informational only)
        b                 : bearing width [mm]
        C                 : dynamic load rating [N]        (ISO 281)
        C0                : static load rating [N]         (ISO 76)
        X, Y              : dynamic equivalent load factors (ISO 281 Table)
        arrangement       : "locating" (restrains axial) | "floating"
        contact_angle_deg : nominal contact angle [°]      (0 = radial)
        label             : identifier for reporting/traceability
        position          : axial coordinate along the shaft [mm]

        Internal geometry (Dw, Dpw, Z, ri, re, E, nu) is NOT stored here.
        Pass it explicitly to the advanced methods that require it
        (compute_hertz_point_contact, curvature_sum_inner, etc.).
        """
        # --- metadata ---
        self.label        = label
        self.designation  = designation
        self.bearing_type = bearing_type

        # --- catalog / rating data ---
        self.b   = b
        self.d   = d
        self.D   = D
        self.C   = C
        self.C0  = C0
        self.X   = X
        self.Y   = Y
        self.dm  = 0.5 * (d + D)

        # --- contact geometry ---
        self.contact_angle_deg = contact_angle_deg
        self.contact_angle     = np.radians(contact_angle_deg)

        # --- mounting / FEM 1D ---
        self.position    = position
        self.arrangement = arrangement

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
    # Ball Bearing — ISO/TS 16281
    # Internal geometry parameters (Dw, Dpw, ri, re, E, nu) are passed
    # They are passed as attributes and so use in te other solvers 
    # compute_hertz_point_contact must be called
    # ------------------------------------------------------------------

    def curvature_sum_inner(self,
                            Dw: float,
                            Dpw: float,
                            ri: float) -> float:
        """
        Σρᵢ = (2/Dw) · (2 + γ/(1−γ) − Dw/(2rᵢ))    ISO/TS 16281 eq.(5)

        Parameters
        ----------
        Dw  : rolling element diameter [mm]
        Dpw : pitch circle diameter [mm]
        ri  : inner groove radius [mm]
        """
        gamma = Dw * np.cos(self.contact_angle) / Dpw
        return (2.0 / Dw) * (2.0 + gamma / (1.0 - gamma) - Dw / (2.0 * ri))

    def curvature_sum_outer(self,
                            Dw: float,
                            Dpw: float,
                            re: float) -> float:
        """
        Σρₑ = (2/Dw) · (2 − γ/(1+γ) − Dw/(2rₑ))    ISO/TS 16281 eq.(6)

        Parameters
        ----------
        Dw  : rolling element diameter [mm]
        Dpw : pitch circle diameter [mm]
        re  : outer groove radius [mm]
        """
        gamma = Dw * np.cos(self.contact_angle) / Dpw
        return (2.0 / Dw) * (2.0 - gamma / (1.0 + gamma) - Dw / (2.0 * re))

    def curvature_difference_inner(self,
                                   Dw: float,
                                   Dpw: float,
                                   ri: float) -> float:
        """
        Fᵢ(ρ) = (γ/(1−γ) + Dw/(2rᵢ)) / (2 + γ/(1−γ) − Dw/(2rᵢ))    eq.(7)

        Parameters
        ----------
        Dw  : rolling element diameter [mm]
        Dpw : pitch circle diameter [mm]
        ri  : inner groove radius [mm]
        """
        gamma = Dw * np.cos(self.contact_angle) / Dpw
        num = gamma / (1.0 - gamma) + Dw / (2.0 * ri)
        den = 2.0 + gamma / (1.0 - gamma) - Dw / (2.0 * ri)
        return num / den

    def curvature_difference_outer(self,
                                   Dw: float,
                                   Dpw: float,
                                   re: float) -> float:
        """
        Fₑ(ρ) = (−γ/(1+γ) + Dw/(2rₑ)) / (2 − γ/(1+γ) − Dw/(2rₑ))    eq.(8)

        Parameters
        ----------
        Dw  : rolling element diameter [mm]
        Dpw : pitch circle diameter [mm]
        re  : outer groove radius [mm]
        """
        gamma = Dw * np.cos(self.contact_angle) / Dpw
        num = -gamma / (1.0 + gamma) + Dw / (2.0 * re)
        den = 2.0 - gamma / (1.0 + gamma) - Dw / (2.0 * re)
        return num / den

    @staticmethod
    def _chi_equation(chi: float, F_rho: float) -> float:
        """Eq.(2): 1 − 2/(χ²−1)·[K(χ)/E(χ) − 1] − F(ρ) = 0"""
        m = 1.0 - 1.0 / chi**2
        K = ellipk(m)
        E = ellipe(m)
        return 1.0 - (2.0 / (chi**2 - 1.0)) * (K / E - 1.0) - F_rho

    def chi_contact(self, F_rho: float) -> float:
        """
        Resolve eq.(2) para χ via brentq.

        Parameters
        ----------
        F_rho : curvature difference F(ρ) — inner or outer
        """
        return brentq(self._chi_equation, 1.0001, 1000.0, args=(F_rho,))

    def setup_internal_geometry(self,
                                ri: float,
                                re: float,
                                Dw: float,
                                Dpw: float,
                                Z: int,
                                s: float,
                                E: float,
                                nu: float = 0.3) -> None:
        """
        Cache internal geometry as public attributes.
        Must be called before compute_hertz_point_contact() and the
        load distribution solver.

        Parameters
        ----------
        ri, re : inner/outer groove radii [mm]
        Dw     : rolling element diameter [mm]
        Dpw    : pitch circle diameter [mm]
        Z      : number of rolling elements
        s      : diametral operating clearance [mm]
        E      : Young's modulus [MPa]
        nu     : Poisson's ratio
        """
        self.ri      = ri
        self.re      = re
        self.Dw      = Dw
        self.Dpw     = Dpw
        self.Z       = Z
        self.s       = s
        self.E       = E
        self.nu      = nu
        self.A       = ri + re - Dw
        self.alpha_0 = np.arccos(1.0 - s / (2.0 * self.A))
        self.Ri      = (Dpw / 2.0) - (Dw / 2.0) * np.cos(self.contact_angle) 
        self.phi_j   = np.linspace(0, 2 * np.pi, Z, endpoint=False)


    def compute_hertz_point_contact(self) -> float:
        """
        Hertzian spring constant c_p for a DGBB.
        ISO/TS 16281 eq.(5)–(11).
        Requires setup_internal_geometry() to have been called first.

        Returns
        -------
        c_p : Hertzian spring constant [N/mm^(3/2)]
        """
        sum_rho_i = self.curvature_sum_inner(self.Dw, self.Dpw, self.ri)
        sum_rho_e = self.curvature_sum_outer(self.Dw, self.Dpw, self.re)
        F_i       = self.curvature_difference_inner(self.Dw, self.Dpw, self.ri)
        F_e       = self.curvature_difference_outer(self.Dw, self.Dpw, self.re)
        chi_i     = self.chi_contact(F_i)
        chi_e     = self.chi_contact(F_e)

        E_star = self.E / (1.0 - self.nu**2)

        m_i = 1.0 - 1.0 / chi_i**2
        m_e = 1.0 - 1.0 / chi_e**2
        Ki, Ei = ellipk(m_i), ellipe(m_i)
        Ke, Ee = ellipk(m_e), ellipe(m_e)

        bracket_i = Ki * np.cbrt(sum_rho_i / (chi_i**2 * Ei))
        bracket_e = Ke * np.cbrt(sum_rho_e / (chi_e**2 * Ee))

        self.cp = 1.48 * E_star * (bracket_i + bracket_e) ** (-3/2)
        return self.cp
    

    # ------------------------------------------------------------------
    # Auxiliary / convenience
    # ------------------------------------------------------------------

    def equivalent_modulus_of_elasticity(self,
                                         E1: float, E2: float,
                                         v1: float, v2: float) -> float:
        """
        E* = 1 / ( (1-v1²)/E1 + (1-v2²)/E2 )   (Hertzian contact)
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
                f"{tag}: arrangement must be 'locating' or 'floating' or 'non-locating', "
                f"got '{self.arrangement}'"
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
            f"  d / D       : {self.d:.1f} mm / {self.D:.1f} mm",
            f"  b           : {self.b:.1f} mm",
            f"  C / C0      : {self.C:.0f} N / {self.C0:.0f} N",
            f"  X / Y       : {self.X:.3f} / {self.Y:.3f}",
            f"  α₀          : {self.contact_angle_deg:.1f} °",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Bearing(type={self.bearing_type.name}, "
            f"designation='{self.designation}', "
            f"position={self.position:.2f} mm, "
            f"arrangement='{self.arrangement}')"
        )