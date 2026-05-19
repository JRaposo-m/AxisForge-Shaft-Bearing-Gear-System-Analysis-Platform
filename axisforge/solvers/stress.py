"""
solvers/stress.py
StressSolver — combined stress, Goodman, ASME-Elliptic — Phase 1.

Formulation:
  Euler-Bernoulli beam + Saint-Venant torsion.
  Rotating shaft (Phase 1):
    Ma = M_res  (fully reversed bending)
    Mm = 0
    Ta = 0
    Tm = T      (steady torsion)

  Von Mises equivalent stresses (Shigley Eq. 7-5, 7-6):
    σ'_a = sqrt[(32·Kf·Ma / π·d³)²]           = 32·Kf·Ma / π·d³
    σ'_m = sqrt[3·(16·Kfs·Tm / π·d³)²]        = √3 · 16·Kfs·Tm / π·d³

  DE-Goodman (Eq. 7-7, simplified):
    1/n = (16/π·d³) · { 2·Kf·Ma/Se' + √3·Kfs·Tm/Sut }

  DE-ASME Elliptic (Eq. 7-11, simplified):
    1/n² = (16/π·d³)² · [ 4·(Kf·Ma/Se')² + 3·(Kfs·Tm/Sy)² ]

  Yielding (Eq. 7-15, 7-16):
    σ'_max = sqrt[(32·Kf·Ma/π·d³)² + 3·(16·Kfs·Tm/π·d³)²]
    ny = Sy / σ'_max

  Langer static yield check: warn if ny < nf_goodman.

Endurance limit correction (Marin factors):
  Se' = Se × ka × kb × kc × kd × ke               [Shigley §6-6]
  kc = 1.0 always for combined bending+torsion (§6-14).

Stress concentration (Peterson):
  Kf  = 1 + q × (Kt − 1)                          [Shigley §6-10]
  Kfs = 1 + qs × (Kts − 1)
  q from Neuber Eq. 6-34; Kt from Peterson Fig. A-15-9/10.

Critical sections: shaft shoulders only (Phase 1).
Sections with M_res < 1 N·mm → nf = inf (no bending, excluded from ranking).

Phase 1 scope:
  - Shoulders only; no keyways or press fits.
  - D/d = 1.5 approximation for Peterson (one-parameter interpolation).
  - No axial load contribution (Mm = 0).
  - kc = 1.0 (combined bending+torsion mode, Shigley §6-14).

Restrictions:
  - No GUI imports. No database calls. No global state.
  - All functions are pure — no side effects.

References:
  Shigley 10th ed.: §6-2 (Se), §6-6 (Marin), §6-9 (Goodman),
                    §6-10 (Kf, Neuber), §6-11 (ASME-Elliptic),
                    §7-1, §7-4 (Eq. 7-5, 7-6, 7-7, 7-11, 7-15, 7-16)
  Peterson 3rd ed.: Fig. A-15-9, A-15-10
"""
from __future__ import annotations

import math
import warnings
import numpy as np

from scipy.interpolate import interp1d

from core.materials import Material
from core.system import MechanicalSystem
from models.statics_result import StaticsResult
from models.stress_result import CriticalSection, StressRaiserType, StressResult


# ===========================================================================
# Marin surface finish factor — ka
# Shigley Tab. 6-2: ka = a × Sut^b  (Sut in MPa)
# ===========================================================================

_KA_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "ground":     (1.58,  -0.085),
    "machined":   (4.51,  -0.265),
    "cold_drawn": (4.51,  -0.265),
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

_KT_SHOULDER_BENDING_R_OVER_D = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
_KT_SHOULDER_BENDING_KT       = [3.00,  2.70, 2.30, 2.10, 1.85, 1.60, 1.40]

_kt_shoulder_bending_interp = interp1d(
    _KT_SHOULDER_BENDING_R_OVER_D,
    _KT_SHOULDER_BENDING_KT,
    kind="linear",
    bounds_error=False,
    fill_value=(_KT_SHOULDER_BENDING_KT[0], _KT_SHOULDER_BENDING_KT[-1]),
)

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
    else:
        sqrt_a_in = (
            0.246
            - 3.08e-3 * Sut_kpsi
            + 1.51e-5 * Sut_kpsi ** 2
            - 2.67e-8 * Sut_kpsi ** 3
        )
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
# ===========================================================================

_MIN_BENDING_MOMENT_Nmm = 1.0   # Sections with M_res < this → nf = inf


class StressSolver:
    """
    Computes fatigue safety factors at critical cross-sections of a rotating shaft.

    Interface (Shigley §7-4):
        solver = StressSolver()
        result = solver.solve(system, statics_result, material)

    Phase 1 scope:
      - Critical sections at shaft shoulders only.
      - Rotating shaft: Ma = M_res, Mm = 0, Ta = 0, Tm = T.
      - DE-Goodman (Eq. 7-7) and DE-ASME Elliptic (Eq. 7-11).
      - Yielding check ny = Sy / σ'_max (Eq. 7-16).
      - Langer check: warn if ny < nf_goodman.
      - kc = 1.0 always (combined loading mode, §6-14).
    """

    def solve(
        self,
        system: MechanicalSystem,
        statics_result: StaticsResult,
        material: Material,
        finish: str = "machined",
        reliability_percent: float = 99.0,
    ) -> StressResult:
        """
        Full stress analysis at all shoulder critical sections.

        Args:
            system: validated MechanicalSystem with shaft shoulder geometry.
            statics_result: output of StaticsSolver.solve().
            material: shaft material (Sut, Sy, endurance_limit).
            finish: Marin surface finish key for ka.
            reliability_percent: target reliability for ke [%].

        Returns:
            StressResult with CriticalSection list sorted by nf_goodman ascending.
            Sections with M_res < 1 N·mm appended at end with nf = inf.

        Raises:
            ValueError: via system.validate_or_raise() on invalid geometry.
        """
        system.validate_or_raise()

        shoulders = self._identify_critical_sections(system)

        finite_sections: list[CriticalSection] = []
        infinite_sections: list[CriticalSection] = []

        for x_shoulder, shoulder in shoulders:
            section = self._evaluate_section(
                x=x_shoulder,
                shoulder=shoulder,
                system=system,
                statics_result=statics_result,
                material=material,
                finish=finish,
                reliability_percent=reliability_percent,
            )
            if math.isfinite(section.nf_goodman):
                finite_sections.append(section)
            else:
                infinite_sections.append(section)

        # Sort finite sections by nf_goodman ascending (most critical first)
        finite_sections.sort(key=lambda s: s.nf_goodman)
        all_sections = finite_sections + infinite_sections

        return StressResult(
            sections=all_sections,
            material_id=material.material_id,
            finish=finish,
            reliability_percent=reliability_percent,
        )

    # ------------------------------------------------------------------
    # Private: identify shoulders
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_critical_sections(
        system: MechanicalSystem,
    ) -> list[tuple[float, object]]:
        """
        Return list of (x_position, Shoulder) for all shoulders in the shaft.

        shaft.shoulders() returns (x, shoulder_right) at the right-end boundary
        of each section. The diameter relevant for stress calculation is
        shoulder.diameter_small (the smaller section side).
        """
        return system.shaft.shoulders()

    # ------------------------------------------------------------------
    # Private: interpolate M_res and T at a given x
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_at(
        x: float, statics_result: StaticsResult
    ) -> tuple[float, float]:
        """
        Interpolate M_res and T at axial position x.

        Uses np.interp (linear, clamp at boundaries).

        Returns:
            (M_res_x, T_x) in N·mm.
        """
        M_res_x = float(np.interp(x, statics_result.x, statics_result.M_res))
        T_x     = float(np.interp(x, statics_result.x, statics_result.T))
        return M_res_x, T_x

    # ------------------------------------------------------------------
    # Private: compute Kf/Kfs for a shoulder
    # ------------------------------------------------------------------

    @staticmethod
    def _stress_concentration(
        shoulder,
        Sut_MPa: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Compute Kt, Kts, q, qs, Kf, Kfs for a shoulder.

        Args:
            shoulder: Shoulder object with fillet_radius, r_over_d, D_over_d.
            Sut_MPa: material Sut [MPa].

        Returns:
            (Kt, Kts, q, qs, Kf, Kfs)
        """
        r_over_d = shoulder.r_over_d

        Kt  = kt_shoulder_bending(r_over_d)
        Kts = kt_shoulder_torsion(r_over_d)

        r_mm = shoulder.fillet_radius
        q  = notch_sensitivity(r_mm, Sut_MPa, loading="bending")
        qs = notch_sensitivity(r_mm, Sut_MPa, loading="torsion")

        Kf  = kf_from_kt(Kt,  q)
        Kfs = kf_from_kt(Kts, qs)

        return Kt, Kts, q, qs, Kf, Kfs

    # ------------------------------------------------------------------
    # Private: fatigue criteria
    # ------------------------------------------------------------------

    @staticmethod
    def _goodman_factor(
        Kf: float, Ma: float, Kfs: float, Tm: float,
        d: float, Se_prime: float, Sut: float,
    ) -> float:
        """
        DE-Goodman safety factor nf (Shigley Eq. 7-7, rotating shaft).

        Simplified (Mm=0, Ta=0):
          1/n = (16/π·d³) · { 2·Kf·Ma/Se' + √3·Kfs·Tm/Sut }

        Returns inf if Ma < _MIN_BENDING_MOMENT_Nmm.
        """
        if Ma < _MIN_BENDING_MOMENT_Nmm:
            return float("inf")
        factor = (16.0 / (math.pi * d ** 3))
        inv_n = factor * (
            2.0 * Kf * Ma / Se_prime
            + math.sqrt(3.0) * Kfs * Tm / Sut
        )
        if inv_n <= 0.0:
            return float("inf")
        return 1.0 / inv_n

    @staticmethod
    def _asme_factor(
        Kf: float, Ma: float, Kfs: float, Tm: float,
        d: float, Se_prime: float, Sy: float,
    ) -> float:
        """
        DE-ASME Elliptic safety factor nf (Shigley Eq. 7-11, rotating shaft).

        Simplified (Mm=0, Ta=0):
          1/n² = (16/π·d³)² · [ 4·(Kf·Ma/Se')² + 3·(Kfs·Tm/Sy)² ]

        Returns inf if Ma < _MIN_BENDING_MOMENT_Nmm.
        """
        if Ma < _MIN_BENDING_MOMENT_Nmm:
            return float("inf")
        factor = (16.0 / (math.pi * d ** 3))
        inv_n2 = factor ** 2 * (
            4.0 * (Kf * Ma / Se_prime) ** 2
            + 3.0 * (Kfs * Tm / Sy) ** 2
        )
        if inv_n2 <= 0.0:
            return float("inf")
        return 1.0 / math.sqrt(inv_n2)

    @staticmethod
    def _yielding_factor(
        Kf: float, Ma: float, Kfs: float, Tm: float,
        d: float, Sy: float,
    ) -> float:
        """
        Yielding safety factor ny (Shigley Eq. 7-15, 7-16).

        σ'_max = sqrt[(32·Kf·Ma/π·d³)² + 3·(16·Kfs·Tm/π·d³)²]
        ny = Sy / σ'_max

        Returns inf if both Ma and Tm are zero.
        """
        sigma_max = math.sqrt(
            (32.0 * Kf * Ma / (math.pi * d ** 3)) ** 2
            + 3.0 * (16.0 * Kfs * Tm / (math.pi * d ** 3)) ** 2
        )
        if sigma_max < 1e-12:
            return float("inf")
        return Sy / sigma_max

    # ------------------------------------------------------------------
    # Private: evaluate one section end-to-end
    # ------------------------------------------------------------------

    def _evaluate_section(
        self,
        x: float,
        shoulder,
        system: MechanicalSystem,
        statics_result: StaticsResult,
        material: Material,
        finish: str,
        reliability_percent: float,
    ) -> CriticalSection:
        """
        Compute full stress state at one shoulder position.

        The relevant diameter is shoulder.diameter_small — the smaller section
        that experiences the full stress concentration from the step.
        """
        d = shoulder.diameter_small

        # Load amplitudes at this position
        M_res_x, T_x = self._interpolate_at(x, statics_result)
        Ma = M_res_x   # fully reversed bending
        Mm = 0.0
        Ta = 0.0
        Tm = T_x       # steady torsion

        # Stress concentration
        Kt, Kts, q, qs, Kf, Kfs = self._stress_concentration(shoulder, material.Sut)

        # Marin Se' — kc = 1.0 (combined bending+torsion, §6-14)
        Se_base = material.endurance_limit
        marin = endurance_limit_corrected(
            Se_base=Se_base,
            Sut_MPa=material.Sut,
            diameter_mm=d,
            finish=finish,
            reliability_percent=reliability_percent,
            load_type="bending",   # kc = 1.0 per §6-14
        )
        Se_prime = marin["Se_prime"]

        # Von Mises equivalent stresses (Eq. 7-5, 7-6)
        sigma_a = (32.0 * Kf * Ma) / (math.pi * d ** 3) if Ma >= _MIN_BENDING_MOMENT_Nmm else 0.0
        sigma_m = math.sqrt(3.0) * (16.0 * Kfs * Tm) / (math.pi * d ** 3)

        # Safety factors
        nf_goodman = self._goodman_factor(Kf, Ma, Kfs, Tm, d, Se_prime, material.Sut)
        nf_asme    = self._asme_factor(Kf, Ma, Kfs, Tm, d, Se_prime, material.Sy)
        ny         = self._yielding_factor(Kf, Ma, Kfs, Tm, d, material.Sy)

        # Langer check: static yield governs if ny < nf_goodman
        langer_ok = ny >= nf_goodman
        if not langer_ok:
            warnings.warn(
                f"Langer check at x={x:.1f} mm: ny={ny:.3f} < nf_goodman={nf_goodman:.3f}. "
                f"Static yield governs before fatigue. Check Sy vs load.",
                stacklevel=4,
            )

        return CriticalSection(
            x=x,
            diameter=d,
            raiser_type=StressRaiserType.SHOULDER,
            Kt=Kt,
            Kts=Kts,
            q=q,
            qs=qs,
            Kf=Kf,
            Kfs=Kfs,
            Ma=Ma,
            Mm=Mm,
            Ta=Ta,
            Tm=Tm,
            sigma_a=sigma_a,
            sigma_m=sigma_m,
            Se_prime=Se_prime,
            ka=marin["ka"],
            kb=marin["kb"],
            ke=marin["ke"],
            nf_goodman=nf_goodman,
            nf_asme=nf_asme,
            ny=ny,
            langer_ok=langer_ok,
        )
