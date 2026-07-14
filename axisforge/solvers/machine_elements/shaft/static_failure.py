# solvers/shaft/static_failure.py
"""
StaticFailureSolver — static failure analysis for ductile materials.

Theory: Shigley MED 10th ed., Ch. 05.
  §5-4  Maximum-Shear-Stress theory (MSS / Tresca)
  §5-5  Distortion-Energy theory (DE / von Mises)
  §5-6  Ductile Coulomb-Mohr (DCM) — documented but NOT implemented in Phase 1.
           Reason: all embedded materials have Syt = Syc (steels).
           Requires Syc as separate Material attribute (Phase 2).

Key assumptions (all documented in Phase1_Limitations.md):
  - Kt = 1 for all sections (ductile material, static loading — §5-2).
    Local yielding redistributes stress; notch not failure-critical under static load.
  - sigma_y = 0: no transverse normal stress on shaft cross-section.
  - sigma_x from bending: 32*M_res / (pi*d^3)  [MPa]
  - tau_xy from torsion:  16*T / (pi*d^3)       [MPa]
  - Critical sections: shoulder positions only (same as StressSolver).
  - M_res and T values taken at shoulder axial position from StaticsResult.

Design rules:
  - No GUI imports. No database calls. No global state.
  - All methods are pure functions of their arguments.
  - validate_or_raise() called at entry point.
  - Intermediate quantities (sigma_x, tau_xy, sigma_prime, sigma_eff_mss)
    all exposed in StaticFailureSection for debugging/inspection.
"""
from __future__ import annotations
import math
import warnings
import numpy as np

from core.system import MechanicalSystem
from core.materials import Material
from models.statics_result import StaticsResult
from models.static_failure_result import StaticFailureResult, StaticFailureSection


class StaticFailureSolver:
    """
    Evaluates static failure safety factors at all shoulder critical sections.

    Usage:
        statics_result = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics_result, material)

    The solver is stateless — all data flows through arguments and return values.
    """

    def solve(
        self,
        system: MechanicalSystem,
        statics_result: StaticsResult,
        material: Material,
    ) -> StaticFailureResult:
        """
        Perform static failure analysis at all shoulder positions.

        Parameters
        ----------
        system : MechanicalSystem
            Validated system with shaft geometry.
        statics_result : StaticsResult
            Output of StaticsSolver — provides M_res(x) and T(x).
        material : Material
            Shaft material — provides Sy.

        Returns
        -------
        StaticFailureResult
            Sections sorted by governing_n ascending (most critical first).

        Raises
        ------
        ValueError
            If system fails validation.
        """
        system.validate_or_raise()

        Sy = material.Sy
        sections: list[StaticFailureSection] = []

        for x_pos, shoulder, label in self._shoulder_positions(system):
            d = system.shaft.diameter_at(x_pos)
            M = self._interpolate(statics_result.x, statics_result.M_res, x_pos)
            T = self._interpolate(statics_result.x, statics_result.T, x_pos)

            # Stress components — Kt = 1.0 (ductile static, Shigley §5-2)
            sigma_x = self._bending_stress(M, d)
            tau_xy = self._torsion_stress(T, d)

            # Effective stresses
            sigma_prime = self._von_mises(sigma_x, 0.0, tau_xy)
            sigma_eff_mss = self._mss_effective(sigma_x, 0.0, tau_xy)

            # Safety factors
            n_DE, n_MSS = self._safety_factors(sigma_prime, sigma_eff_mss, Sy)

            governing_n, governing_theory = self._governing(n_DE, n_MSS)
            yielded = governing_n < 1.0

            sections.append(StaticFailureSection(
                x=x_pos,
                diameter=d,
                sigma_x=sigma_x,
                tau_xy=tau_xy,
                sigma_prime=sigma_prime,
                sigma_eff_mss=sigma_eff_mss,
                n_DE=n_DE,
                n_MSS=n_MSS,
                governing_n=governing_n,
                governing_theory=governing_theory,
                yielded=yielded,
                label=label,
            ))

        # Sort most critical first (lowest governing_n)
        sections.sort(key=lambda s: s.governing_n)

        if not sections:
            warnings.warn(
                "StaticFailureSolver: no shoulder sections found in system. "
                "No critical sections evaluated. Add Shoulder geometry to ShaftSections.",
                stacklevel=2,
            )

        return StaticFailureResult(
            sections=sections,
            material_id=material.material_id,
            Sy=Sy,
        )

    # ── Core stress calculations ──────────────────────────────────────────────

    def _bending_stress(self, M: float, d: float) -> float:
        """
        Normal stress due to bending moment.
        σx = 32*M / (π*d³)   [MPa]  — Shigley Eq. 7-15a (shaft form).
        Kt = 1.0 for ductile static (§5-2).
        """
        if d <= 0:
            return 0.0
        return 32.0 * M / (math.pi * d ** 3)

    def _torsion_stress(self, T: float, d: float) -> float:
        """
        Shear stress due to torsion.
        τxy = 16*T / (π*d³)   [MPa]  — Shigley Eq. 7-15b (shaft form).
        Kt = 1.0 for ductile static (§5-2).
        """
        if d <= 0:
            return 0.0
        return 16.0 * abs(T) / (math.pi * d ** 3)

    def _von_mises(self, sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """
        Von Mises effective stress for plane stress state.
        σ' = √(σx² - σx·σy + σy² + 3·τxy²)   Shigley Eq. 5-15.

        Returns 0.0 if result is numerically zero (guard for negative radicand
        due to floating-point — physically impossible for real stress states).
        """
        val = sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3.0 * tau_xy**2
        return math.sqrt(max(val, 0.0))

    def _mss_effective(self, sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """
        MSS effective stress = σ1 - σ3 for plane stress.

        Principal stresses (plane stress, σ3 = 0 out-of-plane):
          σ1, σ2 = (σx+σy)/2 ± √[(σx-σy)²/4 + τxy²]
          σ3 = 0

        Order: σ1 ≥ σ2 ≥ σ3, then σ_eff = σ1 - σ3.

        Three cases (Shigley §5-4, Eqs. 5-4 to 5-6):
          Case 1: σA ≥ σB ≥ 0  →  σ_eff = σA
          Case 2: σA ≥ 0 ≥ σB  →  σ_eff = σA - σB
          Case 3: 0 ≥ σA ≥ σB  →  σ_eff = -σB = |σB|

        Shigley Eq. 3-16 for τmax → σ_eff = 2·τmax directly, which is
        equivalent. Used here via principal stress computation for clarity.
        """
        centre = (sigma_x + sigma_y) / 2.0
        radius = math.sqrt(((sigma_x - sigma_y) / 2.0) ** 2 + tau_xy ** 2)
        sigma_A = centre + radius   # larger in-plane principal
        sigma_B = centre - radius   # smaller in-plane principal

        # Order the three principal stresses (σ3 = 0 out-of-plane)
        principals = sorted([sigma_A, sigma_B, 0.0], reverse=True)
        sigma_1, _, sigma_3 = principals
        return sigma_1 - sigma_3

    def _safety_factors(
        self,
        sigma_prime: float,
        sigma_eff_mss: float,
        Sy: float,
    ) -> tuple[float, float]:
        """
        Compute safety factors for DE and MSS theories.

        DE:   n = Sy / σ'          Shigley Eq. 5-19
        MSS:  n = Sy / (σ1−σ3)    Shigley Eq. 5-3

        Returns (n_DE, n_MSS). Returns (inf, inf) if both effective stresses
        are numerically zero (no load on section).
        """
        n_DE = math.inf if sigma_prime < 1e-10 else Sy / sigma_prime
        n_MSS = math.inf if sigma_eff_mss < 1e-10 else Sy / sigma_eff_mss
        return n_DE, n_MSS

    def _governing(self, n_DE: float, n_MSS: float) -> tuple[float, str]:
        """
        Return (governing_n, theory_label).
        MSS is always ≤ DE (more conservative) per Shigley §5-7.
        """
        tol = 1e-9
        if math.isinf(n_DE) and math.isinf(n_MSS):
            return math.inf, "equal"
        if abs(n_DE - n_MSS) < tol:
            return n_MSS, "equal"
        if n_MSS <= n_DE:
            return n_MSS, "MSS"
        # n_DE < n_MSS — theoretically impossible for plane stress but
        # can occur due to floating point; return DE with warning.
        warnings.warn(
            f"n_DE ({n_DE:.4f}) < n_MSS ({n_MSS:.4f}) — unexpected for plane stress. "
            f"Check stress state. Using n_DE as governing.",
            stacklevel=3,
        )
        return n_DE, "DE"

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _shoulder_positions(
        self, system: MechanicalSystem
    ) -> list[tuple[float, object, str]]:
        """
        Return list of (x_position, shoulder_object, label) for all shoulders
        in the shaft. Mirrors StressSolver._identify_critical_sections() approach.
        """
        shaft = system.shaft
        positions = []
        for x_pos, shoulder in shaft.shoulders():
            label = f"shoulder@{x_pos:.1f}mm"
            positions.append((x_pos, shoulder, label))
        return positions

    @staticmethod
    def _interpolate(x_arr: np.ndarray, y_arr: np.ndarray, x_pos: float) -> float:
        """
        Linear interpolation of discretised diagram at position x_pos.
        Returns 0.0 if x_pos is outside array bounds (guard).
        """
        if x_pos < x_arr[0] or x_pos > x_arr[-1]:
            return 0.0
        return float(np.interp(x_pos, x_arr, y_arr))
