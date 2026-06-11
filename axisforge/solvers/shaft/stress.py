"""
solvers/shaft/stress.py
StressSolver — combined stress, Goodman, ASME-Elliptic.

Pipeline:
    1. Varredura contínua (SOLVER_RESOLUTION pontos) — tensões nominais em todo o veio.
    2. Nos ombros aplica Kf/Kfs; no resto Kf = Kfs = 1.0.
    3. Rankeia todas as secções por nf_goodman ascending.
    4. A secção mais crítica pode estar num ombro OU entre ombros.

Loading decomposition via LoadingProfile (core/loads.py):
    Ma, Mm = loading_profile.decompose_bending(M_peak)
    Ta, Tm = loading_profile.decompose_torsion(T_peak)
    Default: LoadingProfile.rotating_shaft() → R_bend=-1, R_tors=+1

Formulation:
    Von Mises equivalent stresses (Shigley Eq. 7-5, 7-6):
        σ'_a = sqrt[ (Kf  · 32·Ma / π·d³)² + 3·(Kfs · 16·Ta / π·d³)² ]
        σ'_m = sqrt[ (Kf  · 32·Mm / π·d³)² + 3·(Kfs · 16·Tm / π·d³)² ]

    DE-Goodman (Shigley Eq. 6-46):
        1/n = σ'_a / Se' + σ'_m / Sut

    DE-ASME Elliptic (Shigley Eq. 6-48):
        1/n² = (σ'_a / Se')² + (σ'_m / Sy)²

    Langer static yield (Shigley §6-12):
        ny = Sy / (σ'_a + σ'_m)

Endurance limit correction (Marin factors):
    Se' = Se × ka × kb × kc × kd × ke    [Shigley §6-6]
    kc = 1.0 always for combined bending+torsion (§6-14).

Stress concentration (Peterson):
    Kf  = 1 + q  × (Kt  − 1)    [Shigley §6-10]
    Kfs = 1 + qs × (Kts − 1)
    Applied only at shaft shoulders (Phase 1).
    Kf = Kfs = 1.0 at all other positions.

Phase 1 scope:
    - Shoulders only for stress concentration; no keyways or press fits.
    - D/d = 1.5 approximation for Peterson (one-parameter interpolation).
    - kc = 1.0 (combined bending+torsion mode, Shigley §6-14).

Restrictions:
    - No GUI imports. No database calls. No global state.
    - All functions are pure — no side effects.

References:
    Shigley 10th ed.: §6-2 (Se), §6-6 (Marin), §6-9 (Goodman),
                      §6-10 (Kf, Neuber), §6-11 (ASME-Elliptic),
                      §6-12 (Langer, stress ratio R), §7-1, §7-4
    Peterson 3rd ed.: Fig. A-15-9, A-15-10
"""
from __future__ import annotations

import math
import warnings

from core.loads import LoadingProfile
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


_MIN_BENDING_MOMENT_Nmm = 1.0   # secções com Ma < este valor → sigma_a = 0


class StressSolver:
    """
    Computes fatigue safety factors across the full shaft length.

    Interface:
        solver = StressSolver()
        result = solver.solve(system, statics_result, material)

    Loading profile controls cycle decomposition (Ma/Mm/Ta/Tm).
    Default is the canonical rotating shaft assumption (Shigley §7-1).
    """

    def solve(
        self,
        system: MechanicalSystem,
        statics_result: StaticsResult,
        material: Material,
        loading_profile: LoadingProfile | None = None,
        finish: str = "machined",
        reliability_percent: float = 99.0,
    ) -> StressResult:
        """
        Full stress analysis across the complete shaft length.

        Args:
            system: validated MechanicalSystem with shaft geometry.
            statics_result: output of StaticsSolver.solve().
            material: shaft material (Sut, Sy, endurance_limit).
            loading_profile: cycle decomposition. Default = rotating_shaft().
            finish: Marin surface finish key for ka.
            reliability_percent: target reliability for ke [%].

        Returns:
            StressResult with all sections sorted by nf_goodman ascending.
            Sections without significant bending (nf = inf) appended at end.

        Raises:
            ValueError: via system.validate_or_raise() on invalid geometry.
        """
        system.validate_or_raise()

        if loading_profile is None:
            loading_profile = LoadingProfile.rotating_shaft()

        # Mapa {x_shoulder: Shoulder} para lookup O(1) no loop
        shoulder_map: dict[float, object] = {
            x: sh for x, sh in system.shaft.shoulders()
        }

        # Varredura completa — todos os pontos do StaticsResult
        x_array = statics_result.x
        M_array = statics_result.M_res
        T_array = statics_result.T

        all_sections: list[CriticalSection] = []

        for i in range(len(x_array)):
            x      = float(x_array[i])
            M_peak = float(M_array[i])
            T_peak = float(T_array[i])

            d = system.shaft.diameter_at(x)

            # Decomposição do ciclo
            Ma, Mm = loading_profile.decompose_bending(M_peak)
            Ta, Tm = loading_profile.decompose_torsion(T_peak)

            # Concentração de tensões — só nos ombros
            if x in shoulder_map:
                shoulder = shoulder_map[x]
                Kt, Kts, q, qs, Kf, Kfs = self._stress_concentration(
                    shoulder, material.Sut
                )
                raiser_type = StressRaiserType.SHOULDER
            else:
                Kt = Kts = Kf = Kfs = 1.0
                q  = qs  = 1.0
                raiser_type = StressRaiserType.NONE

            section = self._evaluate_section(
                x=x, d=d,
                Ma=Ma, Mm=Mm, Ta=Ta, Tm=Tm,
                Kt=Kt, Kts=Kts, q=q, qs=qs, Kf=Kf, Kfs=Kfs,
                raiser_type=raiser_type,
                material=material,
                finish=finish,
                reliability_percent=reliability_percent,
            )
            all_sections.append(section)

        # Ordenar por nf_goodman ascending — mais crítico primeiro
        finite   = [s for s in all_sections if math.isfinite(s.nf_goodman)]
        infinite = [s for s in all_sections if not math.isfinite(s.nf_goodman)]
        finite.sort(key=lambda s: s.nf_goodman)

        return StressResult(
            sections=finite + infinite,
            material_id=material.material_id,
            finish=finish,
            reliability_percent=reliability_percent,
        )

    # ------------------------------------------------------------------
    # Private: stress concentration at a shoulder
    # ------------------------------------------------------------------

    @staticmethod
    def _stress_concentration(
        shoulder,
        Sut_MPa: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Compute Kt, Kts, q, qs, Kf, Kfs for a shoulder.

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
    # Private: evaluate one section end-to-end
    # ------------------------------------------------------------------

    def _evaluate_section(
        self,
        x: float,
        d: float,
        Ma: float,
        Mm: float,
        Ta: float,
        Tm: float,
        Kt: float,
        Kts: float,
        q: float,
        qs: float,
        Kf: float,
        Kfs: float,
        raiser_type,
        material: Material,
        finish: str,
        reliability_percent: float,
    ) -> CriticalSection:
        """
        Compute full stress state at one axial position.

        Von Mises equivalent stresses (Shigley Eq. 7-5, 7-6):
            σ'_a = sqrt[ (Kf·32·Ma/πd³)² + 3·(Kfs·16·Ta/πd³)² ]
            σ'_m = sqrt[ (Kf·32·Mm/πd³)² + 3·(Kfs·16·Tm/πd³)² ]
        """
        # Marin Se'
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

        # Tensões nominais na secção
        bend_a = (32.0 * Ma) / (math.pi * d**3) if Ma >= _MIN_BENDING_MOMENT_Nmm else 0.0
        bend_m = (32.0 * Mm) / (math.pi * d**3)
        tors_a = (16.0 * Ta) / (math.pi * d**3)
        tors_m = (16.0 * Tm) / (math.pi * d**3)

        # Von Mises equivalentes com Kf/Kfs
        sigma_a = math.sqrt((Kf * bend_a)**2 + 3.0 * (Kfs * tors_a)**2)
        sigma_m = math.sqrt((Kf * bend_m)**2 + 3.0 * (Kfs * tors_m)**2)

        # Critérios de falha por fadiga
        nf_goodman = self._goodman(sigma_a, sigma_m, Se_prime, material.Sut)
        nf_asme    = self._asme(sigma_a, sigma_m, Se_prime, material.Sy)
        ny         = self._yielding(sigma_a, sigma_m, material.Sy)

        # Langer check
        langer_ok = True if not math.isfinite(nf_goodman) else (ny >= nf_goodman)
        if not langer_ok:
            warnings.warn(
                f"Langer check at x={x:.1f} mm: ny={ny:.3f} < "
                f"nf_goodman={nf_goodman:.3f}. Static yield governs before fatigue.",
                stacklevel=4,
            )

        return CriticalSection(
            x=x,
            diameter=d,
            raiser_type=raiser_type,
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

    # ------------------------------------------------------------------
    # Critérios de falha
    # ------------------------------------------------------------------

    @staticmethod
    def _goodman(
        sigma_a: float,
        sigma_m: float,
        Se_prime: float,
        Sut: float,
    ) -> float:
        """
        DE-Goodman safety factor (Shigley Eq. 6-46).

        1/n = σ'_a / Se' + σ'_m / Sut

        Returns inf if both stress components are negligible.
        """
        if sigma_a < 1e-12 and sigma_m < 1e-12:
            return float("inf")
        inv_n = sigma_a / Se_prime + sigma_m / Sut
        return 1.0 / inv_n if inv_n > 0.0 else float("inf")

    @staticmethod
    def _asme(
        sigma_a: float,
        sigma_m: float,
        Se_prime: float,
        Sy: float,
    ) -> float:
        """
        DE-ASME Elliptic safety factor (Shigley Eq. 6-48).

        1/n² = (σ'_a / Se')² + (σ'_m / Sy)²

        Returns inf if both stress components are negligible.
        """
        if sigma_a < 1e-12 and sigma_m < 1e-12:
            return float("inf")
        inv_n2 = (sigma_a / Se_prime)**2 + (sigma_m / Sy)**2
        return 1.0 / math.sqrt(inv_n2) if inv_n2 > 0.0 else float("inf")

    @staticmethod
    def _yielding(
        sigma_a: float,
        sigma_m: float,
        Sy: float,
    ) -> float:
        """
        Langer static yield safety factor (Shigley §6-12).

        ny = Sy / (σ'_a + σ'_m)

        Returns inf if both stress components are negligible.
        """
        total = sigma_a + sigma_m
        return Sy / total if total > 1e-12 else float("inf")