"""
models/stress_result.py
StressResult — immutable output of StressSolver.

Phase 1 scope:
  - Critical sections at shaft shoulders only.
  - Rotating shaft: Ma = M_res, Mm = 0, Ta = 0, Tm = T.
  - Goodman (DE-Goodman) and ASME-Elliptic (DE-ASME) safety factors.
  - Yielding check: ny = Sy / σ'_max.

References:
  Shigley 10th ed. §7-4, Eq. 7-7, 7-11, 7-15, 7-16.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class StressRaiserType(Enum):
    """Classification of the geometric stress raiser at a critical section."""
    SHOULDER = auto()      # stepped shaft shoulder (Phase 1)
    KEYWAY = auto()        # keyway — Phase 2+
    PRESS_FIT = auto()     # interference fit — Phase 2+
    GROOVE = auto()        # retaining ring groove — Phase 2+
    OTHER = auto()


@dataclass
class CriticalSection:
    """
    Stress state and fatigue safety factors at one critical cross-section.

    Phase 1 assumption (rotating shaft):
      Ma = M_res  (fully reversed bending)
      Mm = 0
      Ta = 0
      Tm = T      (steady torsion)

    Attributes
    ----------
    x : float
        Axial position [mm] from shaft datum.
    diameter : float
        Shaft outer diameter at this section [mm].
    raiser_type : StressRaiserType
        Type of stress raiser.
    Kt : float
        Theoretical stress concentration factor (bending).
    Kts : float
        Theoretical stress concentration factor (torsion).
    q : float
        Notch sensitivity for bending (Neuber).
    qs : float
        Notch sensitivity for torsion (Neuber).
    Kf : float
        Fatigue stress concentration factor for bending.
    Kfs : float
        Fatigue stress concentration factor for torsion.
    Ma : float
        Alternating bending moment [N·mm]  (= M_res for rotating shaft).
    Mm : float
        Mean bending moment [N·mm]         (= 0 for rotating shaft).
    Ta : float
        Alternating torque [N·mm]          (= 0 for steady torsion).
    Tm : float
        Mean torque [N·mm]                 (= T for steady torsion).
    sigma_a : float
        Alternating von Mises stress σ'_a [MPa]  (Shigley Eq. 7-5).
    sigma_m : float
        Mean von Mises stress σ'_m [MPa]         (Shigley Eq. 7-6).
    Se_prime : float
        Corrected endurance limit at this section [MPa].
    ka : float
        Marin surface finish factor.
    kb : float
        Marin size factor.
    ke : float
        Marin reliability factor.
    nf_goodman : float
        Fatigue safety factor — DE-Goodman (Eq. 7-7).
        float('inf') if Ma < 1 N·mm (no bending).
    nf_asme : float
        Fatigue safety factor — DE-ASME Elliptic (Eq. 7-11).
        float('inf') if Ma < 1 N·mm.
    ny : float
        Yielding safety factor ny = Sy / σ'_max (Eq. 7-16).
    langer_ok : bool
        True if nf_goodman ≤ ny (Langer static yield line not governing).
    """
    x: float
    diameter: float
    raiser_type: StressRaiserType

    # Stress concentration
    Kt: float
    Kts: float
    q: float
    qs: float
    Kf: float
    Kfs: float

    # Load amplitudes
    Ma: float
    Mm: float
    Ta: float
    Tm: float

    # Stresses
    sigma_a: float          # alternating von Mises [MPa]
    sigma_m: float          # mean von Mises [MPa]

    # Endurance limit
    Se_prime: float
    ka: float
    kb: float
    ke: float

    # Safety factors
    nf_goodman: float
    nf_asme: float
    ny: float
    langer_ok: bool

    @property
    def governing_nf(self) -> float:
        """Minimum of nf_goodman and nf_asme — governing fatigue criterion."""
        return min(self.nf_goodman, self.nf_asme)

    @property
    def is_safe(self) -> bool:
        """True if all safety factors ≥ 1.0."""
        return self.nf_goodman >= 1.0 and self.nf_asme >= 1.0 and self.ny >= 1.0


@dataclass
class StressResult:
    """
    Complete output of StressSolver.solve().

    Contains one CriticalSection per shaft shoulder, sorted by
    nf_goodman ascending (most critical first).

    Sections with M_res < 1 N·mm are excluded from the sorted list
    (infinite life — no bending stress).

    Attributes
    ----------
    sections : list[CriticalSection]
        All evaluated sections, sorted by nf_goodman ascending.
        Sections with nf = inf are appended at the end.
    material_id : str
        Material identifier used for this analysis.
    finish : str
        Surface finish key used for ka.
    reliability_percent : float
        Reliability target [%] used for ke.
    """
    sections: list[CriticalSection] = field(default_factory=list)
    material_id: str = ""
    finish: str = "machined"
    reliability_percent: float = 99.0

    @property
    def most_critical(self) -> Optional[CriticalSection]:
        """Section with lowest nf_goodman; None if no sections."""
        finite = [s for s in self.sections if math.isfinite(s.nf_goodman)]
        return finite[0] if finite else (self.sections[0] if self.sections else None)

    @property
    def min_nf_goodman(self) -> float:
        """Minimum Goodman safety factor across all sections."""
        if not self.sections:
            return float("inf")
        return min(s.nf_goodman for s in self.sections)

    @property
    def min_nf_asme(self) -> float:
        """Minimum ASME-Elliptic safety factor across all sections."""
        if not self.sections:
            return float("inf")
        return min(s.nf_asme for s in self.sections)

    @property
    def min_ny(self) -> float:
        """Minimum yielding safety factor across all sections."""
        if not self.sections:
            return float("inf")
        return min(s.ny for s in self.sections)
