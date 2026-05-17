"""
solvers/stress.py
StressSolver — combined stress, Goodman, ASME-Elliptic — Phase 1.

Formulation:
  Euler-Bernoulli beam + Saint-Venant torsion.
  Bending stress:  σ_b = M_res(x) / W(x)          [Shigley §7-1]
  Torsion stress:  τ    = T(x) / Wt(x)             [Shigley §7-1]
  Von Mises:       σ_eq = sqrt(σ_b² + 3τ²)
  Goodman:         σ_a/Se' + σ_m/Sut = 1/nf        [Shigley §6-9]
  ASME-Elliptic:   (σ_a/Se')² + (σ_m/Sy)² = 1/nf² [Shigley §6-11]

Endurance limit correction (Marin factors):
  Se' = Se × ka × kb × kc × kd × ke               [Shigley §6-6]

Stress concentration (Peterson):
  Kf  = 1 + q × (Kt − 1)                          [Shigley §6-10]
  q   = 1 / (1 + sqrt(a) / sqrt(r))   (Neuber)
  Kt  from Peterson Fig. A-15-9 (bending), A-15-10 (torsion)

Phase 1 scope:
  - Critical sections at shaft shoulders only.
  - Fully-reversed bending (σ_m = 0 unless axial load present).
  - Steady torsion (τ contributes to σ_m via von Mises in ASME-Elliptic).
  - StressSolver.solve() implemented in Weeks 6–7.

Restrictions:
  - No GUI imports. No database calls. No global state.
  - All functions are pure — no side effects.

References:
  Shigley 10th ed.: §6-2 (Se), §6-6 (Marin), §6-9 (Goodman),
                    §6-10 (Kf, Neuber), §6-11 (ASME-Elliptic), §7-1
  Peterson 3rd ed.: Fig. A-15-9, A-15-10
"""
from __future__ import annotations

import math
import numpy as np

from scipy.interpolate import interp1d  # used for Peterson Kt interpolation

from core.materials import Material
from core.system import MechanicalSystem
from models.statics_result import StaticsResult


# ===========================================================================
# Marin surface finish factor — ka
# Shigley Tab. 6-2: ka = a × Sut^b  (Sut in MPa)
# ===========================================================================

_KA_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "ground":     (1.58,  -0.085),
    "machined":   (4.51,  -0.265),
    "cold_drawn": (4.51,  -0.265),   # same as machined per Shigley
    "hot_rolled": (57.7,  -0.718),
    "as_forged":  (272.0, -0.995),
}


def ka_surface_finish(Sut_MPa: float, finish: str = "machined") -> float:
    """
    Marin surface finish factor ka.

    ka = a × Sut^b  (Shigley Eq. 6-19, Tab. 6-2)
    Clipped to 1.0 — ground finish on low-Sut material can nominally exceed 1.

    Args:
        Sut_MPa: ultimate tensile strength [MPa].
        finish: 'ground', 'machined', 'cold_drawn', 'hot_rolled', or 'as_forged'.

    Returns:
        ka in (0, 1].
    """
    if finish not in _KA_COEFFICIENTS:
        available = ", ".join(_KA_COEFFICIENTS.keys())
        raise ValueError(f"Unknown finish '{finish}'. Available: {available}")
    a, b = _KA_COEFFICIENTS[finish]
    return float(min(1.0, a * (Sut_MPa ** b)))


# ===========================================================================
# Marin size factor — kb
# Shigley Eq. 6-20 (rotating round shaft in bending)
# ===========================================================================

def kb_size(diameter_mm: float) -> float:
    """
    Marin size factor kb for rotating round shafts in bending.

    Shigley Eq. 6-20 (SI form):
      2.79 ≤ d ≤ 51 mm:   kb = 1.24 × d^(−0.107)
      51 < d ≤ 254 mm:    kb = 1.51 × d^(−0.157)
      d > 254 mm:          kb = 0.70  (conservative, Shigley note)

    For d < 2.79 mm: kb = 1.0 (size effect negligible at small diameters;
    Shigley does not extend the correlation below 2.79 mm).

    For axial loading: kb = 1.0 per Shigley §6-6 — use kc_load('axial') instead.

    Args:
        diameter_mm: shaft diameter at the section of interest [mm].

    Returns:
        kb in [0.70, 1.0].
    """
    if diameter_mm <= 0:
        raise ValueError(f"diameter_mm must be > 0, got {diameter_mm}")
    if diameter_mm < 2.79:
        return 1.0
    elif diameter_mm <= 51.0:
        return float(1.24 * diameter_mm ** (-0.107))
    elif diameter_mm <= 254.0:
        return float(1.51 * diameter_mm ** (-0.157))
    else:
        return 0.70


# ===========================================================================
# Marin load factor — kc
# Shigley Tab. 6-3
# ===========================================================================

def kc_load(load_type: str = "bending") -> float:
    """
    Marin load type factor kc.

    Shigley Eq. 6-26:
      'bending':  kc = 1.0
      'axial':    kc = 0.85
      'torsion':  kc = 0.59

    IMPORTANT: use kc = 'torsion' only for pure torsional fatigue loading.
    For combined bending + torsion (the standard shaft case), set kc = 1.0
    (i.e. load_type = 'bending') and apply the von Mises criterion via
    the combined stress equations (Shigley Eq. 6-55/6-56, §6-14).

    Args:
        load_type: 'bending', 'axial', or 'torsion'.
    """
    mapping = {"bending": 1.0, "axial": 0.85, "torsion": 0.59}
    if load_type not in mapping:
        raise ValueError(
            f"load_type must be one of {list(mapping.keys())}, got '{load_type}'"
        )
    return mapping[load_type]


# ===========================================================================
# Marin temperature factor — kd
# Shigley §6-6
# ===========================================================================

def kd_temperature(T_celsius: float = 20.0) -> float:
    """
    Marin temperature factor kd.

    Phase 1: returns 1.0 for T ≤ 70°C (standard operating assumption).
    For T > 70°C, degradation occurs but is not modelled in Phase 1.

    Args:
        T_celsius: operating temperature [°C].

    Returns:
        kd = 1.0 (Phase 1).
    """
    return 1.0


# ===========================================================================
# Marin reliability factor — ke
# Shigley Tab. 6-6
# ===========================================================================

_KE_TABLE: dict[float, float] = {
    50.0:     1.000,
    90.0:     0.897,
    95.0:     0.868,
    99.0:     0.814,
    99.9:     0.753,
    99.99:    0.702,
    99.999:   0.659,
    99.9999:  0.620,
}


def ke_reliability(reliability_percent: float = 99.0) -> float:
    """
    Marin reliability factor ke.

    Shigley Tab. 6-6 / Eq. 6-29.
    Supported values: 50, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999 [%].

    Args:
        reliability_percent: target reliability [%].

    Returns:
        ke ≤ 1.0.
    """
    if reliability_percent not in _KE_TABLE:
        available = sorted(_KE_TABLE.keys())
        raise ValueError(
            f"reliability_percent must be one of {available}, "
            f"got {reliability_percent}. "
            f"Use ke_reliability_from_z() for arbitrary values."
        )
    return _KE_TABLE[reliability_percent]


def ke_reliability_from_z(z_score: float) -> float:
    """
    Reliability factor from standard normal deviate z.

    Shigley approximate formula §6-6:  ke = 1 − 0.08 × z

    Args:
        z_score: standard normal deviate
                 (e.g. 1.288 → 90%, 1.645 → 95%, 2.326 → 99%).
    """
    return float(1.0 - 0.08 * z_score)


# ===========================================================================
# Corrected endurance limit Se'
# Se' = Se × ka × kb × kc × kd × ke  [Shigley §6-7]
# ===========================================================================

def endurance_limit_corrected(
    Se_base: float,
    Sut_MPa: float,
    diameter_mm: float,
    finish: str = "machined",
    reliability_percent: float = 99.0,
    load_type: str = "bending",
    T_celsius: float = 20.0,
) -> dict[str, float]:
    """
    Corrected endurance limit Se' with all Marin factors.

    Se' = Se × ka × kb × kc × kd × ke

    Args:
        Se_base: specimen endurance limit Se [MPa] — typically 0.5 × Sut,
                 capped at 700 MPa for Sut > 1400 MPa (Shigley §6-2).
        Sut_MPa: ultimate tensile strength [MPa].
        diameter_mm: section diameter [mm].
        finish: surface finish key (see ka_surface_finish).
        reliability_percent: target reliability [%].
        load_type: 'bending', 'axial', or 'torsion'.
        T_celsius: operating temperature [°C].

    Returns:
        dict with keys: 'ka', 'kb', 'kc', 'kd', 'ke', 'Se_prime' [MPa].
    """
    ka = ka_surface_finish(Sut_MPa, finish)
    kb = kb_size(diameter_mm)
    kc = kc_load(load_type)
    kd = kd_temperature(T_celsius)
    ke = ke_reliability(reliability_percent)
    Se_prime = Se_base * ka * kb * kc * kd * ke
    return {
        "ka":       ka,
        "kb":       kb,
        "kc":       kc,
        "kd":       kd,
        "ke":       ke,
        "Se_prime": Se_prime,
    }


# ===========================================================================
# Peterson stress concentration factors — Kt (shoulders)
# Peterson 3rd ed.: Fig. A-15-9 (bending), A-15-10 (torsion)
# Phase 1: D/d ≈ 1.5 approximation
# ===========================================================================

# Shoulder in bending — D/d ≈ 1.5 (conservative for D/d < 3)
_KT_SHOULDER_BENDING_R_OVER_D = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
_KT_SHOULDER_BENDING_KT       = [3.00,  2.70, 2.30, 2.10, 1.85, 1.60, 1.40]

_kt_shoulder_bending_interp = interp1d(
    _KT_SHOULDER_BENDING_R_OVER_D,
    _KT_SHOULDER_BENDING_KT,
    kind="linear",
    bounds_error=False,
    fill_value=(_KT_SHOULDER_BENDING_KT[0], _KT_SHOULDER_BENDING_KT[-1]),
)

# Shoulder in torsion — D/d ≈ 1.5
_KT_SHOULDER_TORSION_R_OVER_D = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
_KT_SHOULDER_TORSION_KT       = [2.00,  1.85, 1.65, 1.55, 1.40, 1.25, 1.15]

_kt_shoulder_torsion_interp = interp1d(
    _KT_SHOULDER_TORSION_R_OVER_D,
    _KT_SHOULDER_TORSION_KT,
    kind="linear",
    bounds_error=False,
    fill_value=(_KT_SHOULDER_TORSION_KT[0], _KT_SHOULDER_TORSION_KT[-1]),
)


def kt_shoulder_bending(r_over_d: float, D_over_d: float = 1.5) -> float:
    """
    Theoretical stress concentration factor Kt for a shoulder in bending.

    Peterson Fig. A-15-9. Phase 1 uses D/d = 1.5 approximation.
    Phase 2+: 2D interpolation over (r/d, D/d).

    Args:
        r_over_d: fillet radius / small diameter.
        D_over_d: large diameter / small diameter (documented; Phase 1 ignores).

    Returns:
        Kt ≥ 1.0.
    """
    return float(_kt_shoulder_bending_interp(r_over_d))


def kt_shoulder_torsion(r_over_d: float, D_over_d: float = 1.5) -> float:
    """
    Theoretical stress concentration factor Kts for a shoulder in torsion.

    Peterson Fig. A-15-10. Phase 1 uses D/d = 1.5 approximation.

    Args:
        r_over_d: fillet radius / small diameter.
        D_over_d: large diameter / small diameter (documented; Phase 1 ignores).

    Returns:
        Kts ≥ 1.0.
    """
    return float(_kt_shoulder_torsion_interp(r_over_d))


# ===========================================================================
# Neuber constant and notch sensitivity
# Shigley §6-10, Eq. 6-33 to 6-35
# ===========================================================================

def neuber_constant_sqrt_a(Sut_MPa: float, loading: str = "bending") -> float:
    """
    Neuber material constant sqrt(a) [mm^0.5] for steels.

    Shigley Eq. 6-35a (bending/axial) and 6-35b (torsion).
    Both equations use Sut in kpsi internally; input is MPa.

    Eq. 6-35a (bending or axial):
      sqrt(a) = 0.246 − 3.08e-3*Sut + 1.51e-5*Sut² − 2.67e-8*Sut³  [Sut in kpsi]

    Eq. 6-35b (torsion):
      sqrt(a) = 0.190 − 2.51e-3*Sut + 1.35e-5*Sut² − 2.67e-8*Sut³  [Sut in kpsi]

    Result is in sqrt(in); converted to sqrt(mm) via × sqrt(25.4).

    Args:
        Sut_MPa: ultimate tensile strength [MPa].
        loading: 'bending' (default, also valid for axial) or 'torsion'.

    Returns:
        sqrt(a) [mm^0.5]. Higher Sut → smaller value → higher notch sensitivity.
    """
    Sut_kpsi = Sut_MPa / 6.895
    if loading == "torsion":
        sqrt_a_in = (
            0.190
            - 2.51e-3 * Sut_kpsi
            + 1.35e-5 * Sut_kpsi ** 2
            - 2.67e-8 * Sut_kpsi ** 3
        )
    else:  # bending or axial
        sqrt_a_in = (
            0.246
            - 3.08e-3 * Sut_kpsi
            + 1.51e-5 * Sut_kpsi ** 2
            - 2.67e-8 * Sut_kpsi ** 3
        )
    # Convert sqrt(in) → sqrt(mm): 1 in = 25.4 mm → sqrt(1 in) = sqrt(25.4 mm)
    return float(sqrt_a_in * math.sqrt(25.4))


def notch_sensitivity(r_mm: float, Sut_MPa: float, loading: str = "bending") -> float:
    """
    Notch sensitivity q.

    Shigley Eq. 6-34:  q = 1 / (1 + sqrt(a) / sqrt(r))

    Args:
        r_mm: fillet or notch radius [mm]. Clamped to MIN_FILLET_RADIUS_mm.
        Sut_MPa: ultimate tensile strength [MPa].
        loading: 'bending' or 'torsion' — selects Neuber equation (6-35a or 6-35b).

    Returns:
        q in [0, 1]. q → 0: insensitive; q → 1: fully sensitive.
    """
    from config import MIN_FILLET_RADIUS_mm
    r_mm = max(r_mm, MIN_FILLET_RADIUS_mm)
    sqrt_a = neuber_constant_sqrt_a(Sut_MPa, loading)
    return float(1.0 / (1.0 + sqrt_a / math.sqrt(r_mm)))


def kf_from_kt(Kt: float, q: float) -> float:
    """
    Fatigue stress concentration factor Kf.

    Shigley Eq. 6-32:  Kf = 1 + q × (Kt − 1)

    Args:
        Kt: theoretical (geometric) concentration factor.
        q: notch sensitivity in [0, 1].

    Returns:
        Kf in [1.0, Kt].
    """
    return float(1.0 + q * (Kt - 1.0))


# ===========================================================================
# StressSolver
# Full implementation: Weeks 6–7 (Semanas 6–7 do Phase 1 Execution Plan)
# ===========================================================================

class StressSolver:
    """
    Computes nominal and corrected stresses at critical cross-sections
    and evaluates fatigue safety factors.

    Interface (Shigley §7-1, §6-9, §6-11):
        solver = StressSolver()
        result = solver.solve(system, statics_result, material)

    Phase 1 scope:
      - Critical sections identified at shaft shoulders.
      - Fully-reversed bending: σ_a = Kf × σ_b, σ_m = 0 (no axial load).
      - Steady torsion: τ_a = 0, τ_m = Kfs × τ.
      - Goodman:        1/nf = σ_a/Se' + σ_m/Sut
      - ASME-Elliptic:  1/nf² = (σ_a/Se')² + (σ_m/Sy)²

    Implementation status: PLANNED (Semanas 6–7).
    Currently provides all helper functions (Marin, Peterson, Neuber).
    """

    def solve(
        self,
        system: MechanicalSystem,
        statics_result: StaticsResult,
        material: Material,
    ):
        """
        Full stress analysis at all critical sections.

        Args:
            system: MechanicalSystem with shaft geometry and shoulder data.
            statics_result: output of StaticsSolver.solve().
            material: shaft material with Sut, Sy, endurance_limit.

        Returns:
            StressResult (models/stress_result.py) — planned Week 6.

        Raises:
            NotImplementedError: until Semana 6 implementation.
        """
        raise NotImplementedError(
            "StressSolver.solve() is planned for Semana 6–7 of Phase 1. "
            "Marin, Peterson, and Neuber helper functions are available."
        )