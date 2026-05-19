# models/static_failure_result.py
"""
StaticFailureResult — output of StaticFailureSolver.

Stores one StaticFailureSection per evaluated critical section (shoulder positions).
Sections are sorted by governing_n ascending (most critical first).

Theory: Shigley MED 10th ed., Ch. 05 — §5-4 (MSS), §5-5 (DE).
Units: MPa for stresses, dimensionless for safety factors.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StaticFailureSection:
    """
    Static failure assessment at a single axial position.

    Stress state (shaft cross-section):
      sigma_x   = 32*M_res / (pi*d^3)   [MPa]  — bending normal stress
      tau_xy    = 16*T / (pi*d^3)        [MPa]  — torsional shear stress
      sigma_y   = 0                              — no transverse normal stress

    Kt = 1.0 for all sections: ductile material under static loading does not
    concentrate stress (Shigley §5-2 — local yielding redistributes load).

    Failure theories implemented (Phase 1):
      DE  — Distortion Energy (von Mises), Eq. 5-15 / 5-19
      MSS — Maximum Shear Stress (Tresca), Eq. 5-3

    DCM (Ductile Coulomb-Mohr, §5-6): planned Phase 2 — requires Syc != Syt.
    """
    x: float                    # Axial position [mm]
    diameter: float             # Section diameter at x [mm]
    sigma_x: float              # Bending normal stress [MPa]
    tau_xy: float               # Torsional shear stress [MPa]
    sigma_prime: float          # Von Mises effective stress [MPa]
    sigma_eff_mss: float        # MSS effective stress = sigma_1 - sigma_3 [MPa]
    n_DE: float                 # Safety factor — Distortion Energy
    n_MSS: float                # Safety factor — Maximum Shear Stress
    governing_n: float          # min(n_DE, n_MSS)
    governing_theory: str       # "DE" | "MSS" | "equal"
    yielded: bool               # governing_n < 1.0
    label: str = ""             # Source shoulder label (optional)

    @property
    def risk_label(self) -> str:
        """Engineering interpretation of governing_n."""
        n = self.governing_n
        if math.isinf(n):
            return "no_stress"
        if n < 1.0:
            return "yielded"
        if n < 1.5:
            return "marginal"
        if n < 2.5:
            return "acceptable"
        return "conservative"


@dataclass
class StaticFailureResult:
    """
    Complete static failure analysis output.

    sections: list of StaticFailureSection, sorted by governing_n ascending
              (most critical first; index 0 = worst section).

    Attributes
    ----------
    sections : list[StaticFailureSection]
        All evaluated sections, sorted most-critical first.
    material_id : str
        Material used for this analysis.
    Sy : float
        Yield strength used [MPa].
    """
    sections: list[StaticFailureSection] = field(default_factory=list)
    material_id: str = ""
    Sy: float = 0.0

    @property
    def critical_section(self) -> Optional[StaticFailureSection]:
        """Section with lowest governing_n (most critical). None if no sections."""
        return self.sections[0] if self.sections else None

    @property
    def min_governing_n(self) -> float:
        """Minimum governing_n across all sections. inf if no sections."""
        if not self.sections:
            return math.inf
        return self.sections[0].governing_n

    @property
    def any_yielded(self) -> bool:
        """True if any section has governing_n < 1.0."""
        return any(s.yielded for s in self.sections)
