"""
solvers/shaft/stress.py
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

from core.materials import Material
from core.system import MechanicalSystem
from models.statics_result import StaticsResult
from models.stress_result import CriticalSection, StressRaiserType, StressResult

from solvers.shaft.utils import (
    endurance_limit_corrected,
    kf_from_kt,
    kt_shoulder_bending,
    kt_shoulder_torsion,
    notch_sensitivity,
)


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
        langer_ok = True if not math.isfinite(nf_goodman) else (ny >= nf_goodman)
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
