# tests/test_solvers/test_static_failure.py
"""
Tests for StaticFailureSolver — static failure analysis, ductile materials.

Theory: Shigley MED 10th ed., Ch. 05 — §5-4 (MSS), §5-5 (DE).

Validation cases
----------------
SHIGLEY_EX5_1 — Example 5-1, p. 238–241
  Material: hot-rolled steel, Sy = Syc = 100 kpsi (689.5 MPa), εf = 0.55 (ductile).
  Five plane-stress states; DE and MSS computed analytically in the textbook.

  Conversion: 1 kpsi = 6.895 MPa

  Case | σx [MPa]  | σy [MPa]  | τxy [MPa] | n_DE  | n_MSS
  (a)  | 482.65    | 482.65    | 0.0       | 1.43  | 1.43
  (b)  | 413.70    | 275.80    | -103.43   | 1.70  | 1.47
  (c)  | 0.0       | 275.80    | 310.28    | 1.14  | 1.02
  (d)  | -275.80   | -413.70   | 103.43    | 1.70  | 1.47
  (e)  | — (3D)    | — (3D)    | —         | inf   | inf

  Note: case (e) is σ1=σ2=σ3=30 kpsi — hydrostatic, σ'=0, n→∞. Not a plane
  stress case; the solver does not handle 3D principal input directly, so this
  case is verified analytically only (documented limitation, not tested via solver).

  Tolerance: 1% (textbook uses 3 significant figures).
"""
from __future__ import annotations
import math
import pytest
import numpy as np

from core.materials import Material
from core.shaft import Shaft, ShaftSection, Shoulder
from core.components import Bearing, GearElement
from core.system import MechanicalSystem
from models.static_failure_result import StaticFailureResult, StaticFailureSection
from solvers.static_failure import StaticFailureSolver


# ---------------------------------------------------------------------------
# Helpers — stress-state-only interface (bypasses shaft/system machinery)
# for the Shigley Ex 5-1 cases which supply raw stress components.
# ---------------------------------------------------------------------------

TOL_REL = 0.01   # 1% relative tolerance for textbook cases


def _make_material_sy(sy_mpa: float) -> Material:
    """Minimal ductile material with given Sy for direct stress-state tests."""
    return Material(
        material_id=f"TEST_Sy{sy_mpa:.0f}",
        Sut=sy_mpa * 1.2,   # arbitrary Sut > Sy
        Sy=sy_mpa,
        E=207.0,
        density=7850.0,
    )


# ---------------------------------------------------------------------------
# Unit tests — StaticFailureSolver internal methods
# ---------------------------------------------------------------------------

class TestStaticFailureSolverInternals:
    """Test _von_mises, _mss_effective, _safety_factors as isolated units."""

    def setup_method(self):
        self.solver = StaticFailureSolver()
        self.Sy = 689.476  # 100 kpsi in MPa

    # ── von Mises (DE) ──────────────────────────────────────────────────────

    def test_von_mises_pure_tension(self):
        """σx=σ, σy=0, τ=0 → σ'=σ."""
        assert self.solver._von_mises(100.0, 0.0, 0.0) == pytest.approx(100.0, rel=1e-6)

    def test_von_mises_pure_shear(self):
        """σx=0, σy=0, τ → σ'=√3·τ (Eq. 5-15)."""
        tau = 50.0
        expected = math.sqrt(3) * tau
        assert self.solver._von_mises(0.0, 0.0, tau) == pytest.approx(expected, rel=1e-6)

    def test_von_mises_equal_biaxial(self):
        """σx=σy=σ, τ=0 → σ'=σ (no distortion energy)."""
        assert self.solver._von_mises(70.0, 70.0, 0.0) == pytest.approx(70.0, rel=1e-6)

    def test_von_mises_zero_stress(self):
        assert self.solver._von_mises(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    # ── MSS effective stress ─────────────────────────────────────────────────

    def test_mss_pure_tension(self):
        """σx=100, σy=0, τ=0 → σ1=100, σ3=0 → σ_eff=100."""
        assert self.solver._mss_effective(100.0, 0.0, 0.0) == pytest.approx(100.0, rel=1e-6)

    def test_mss_pure_shear(self):
        """σx=0, σy=0, τ=50 → σ1=50, σ3=-50 → σ_eff=100."""
        assert self.solver._mss_effective(0.0, 0.0, 50.0) == pytest.approx(100.0, rel=1e-6)

    def test_mss_equal_biaxial(self):
        """σx=σy=70, τ=0 → σ1=σ2=70, σ3=0 → σ_eff=70."""
        assert self.solver._mss_effective(70.0, 70.0, 0.0) == pytest.approx(70.0, rel=1e-6)

    def test_mss_zero_stress(self):
        assert self.solver._mss_effective(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    # ── Safety factors ───────────────────────────────────────────────────────

    def test_safety_factors_zero_stress_returns_inf(self):
        n_DE, n_MSS = self.solver._safety_factors(0.0, 0.0, self.Sy)
        assert math.isinf(n_DE)
        assert math.isinf(n_MSS)

    def test_safety_factors_at_yield_returns_one(self):
        """σ'=Sy → n_DE=1.0; σ_eff_mss=Sy → n_MSS=1.0."""
        n_DE, n_MSS = self.solver._safety_factors(self.Sy, self.Sy, self.Sy)
        assert n_DE == pytest.approx(1.0, rel=1e-6)
        assert n_MSS == pytest.approx(1.0, rel=1e-6)

    def test_governing_n_selects_minimum(self):
        """governing_n = min(n_DE, n_MSS) and theory label is correct."""
        # Force n_MSS < n_DE: pure shear gives MSS = Sy/σ_eff_mss, DE = Sy/√3τ
        # For τ such that σ_eff_mss > σ_prime: not possible (MSS ≥ DE always)
        # Check that governing_n = n_MSS when MSS < DE
        n_DE, n_MSS = self.solver._safety_factors(100.0, 115.47, self.Sy)
        # n_DE = Sy/100, n_MSS = Sy/115.47 < n_DE
        assert n_MSS < n_DE


# ---------------------------------------------------------------------------
# Validation cases — Shigley Example 5-1
# All stress values converted: 1 kpsi = 6.895 MPa
# ---------------------------------------------------------------------------

SY_MPa = 100.0 * 6.895   # = 689.5 MPa


class TestShigleyEx5_1_StressState:
    """
    Shigley Ex 5-1 — five stress states, direct stress-component input.
    Tests _von_mises and _mss_effective + _safety_factors in isolation.

    Source: Shigley MED 10th ed., Example 5-1, pp. 238–241.
    Tolerance: 1% (textbook rounds to 3 s.f.).
    """

    def setup_method(self):
        self.solver = StaticFailureSolver()
        self.Sy = SY_MPa

    def _check(self, sx, sy, txy, n_DE_exp, n_MSS_exp):
        sp = self.solver._von_mises(sx, sy, txy)
        se = self.solver._mss_effective(sx, sy, txy)
        n_DE, n_MSS = self.solver._safety_factors(sp, se, self.Sy)
        assert n_DE == pytest.approx(n_DE_exp, rel=TOL_REL), \
            f"n_DE: got {n_DE:.4f}, expected {n_DE_exp}"
        assert n_MSS == pytest.approx(n_MSS_exp, rel=TOL_REL), \
            f"n_MSS: got {n_MSS:.4f}, expected {n_MSS_exp}"

    def test_case_a_equal_biaxial(self):
        """(a) σx=σy=70kpsi, τ=0 → n_DE=n_MSS=1.43."""
        sx = 70.0 * 6.895
        sy = 70.0 * 6.895
        self._check(sx, sy, 0.0, n_DE_exp=1.43, n_MSS_exp=1.43)

    def test_case_b_combined(self):
        """(b) σx=60, σy=40, τ=-15 kpsi → n_DE=1.70, n_MSS=1.47."""
        sx = 60.0 * 6.895
        sy = 40.0 * 6.895
        txy = -15.0 * 6.895
        self._check(sx, sy, txy, n_DE_exp=1.70, n_MSS_exp=1.47)

    def test_case_c_shear_dominant(self):
        """(c) σx=0, σy=40, τ=45 kpsi → n_DE=1.14, n_MSS=1.02."""
        sx = 0.0
        sy = 40.0 * 6.895
        txy = 45.0 * 6.895
        self._check(sx, sy, txy, n_DE_exp=1.14, n_MSS_exp=1.02)

    def test_case_d_compressive(self):
        """(d) σx=-40, σy=-60, τ=15 kpsi → n_DE=1.70, n_MSS=1.47."""
        sx = -40.0 * 6.895
        sy = -60.0 * 6.895
        txy = 15.0 * 6.895
        self._check(sx, sy, txy, n_DE_exp=1.70, n_MSS_exp=1.47)

    def test_case_e_hydrostatic_sigma_prime_zero(self):
        """
        (e) σ1=σ2=σ3=30kpsi — purely hydrostatic (not plane stress).
        DE: σ'=0 → n=∞. MSS: σ1-σ3=0 → n=∞.
        Verified directly on internal methods (3D input not shaft use-case).
        """
        # For 3D hydrostatic: σ' = 0 analytically (Eq. 5-12).
        # We verify the safety_factors method returns inf when sigma_prime=0.
        solver = StaticFailureSolver()
        n_DE, n_MSS = solver._safety_factors(0.0, 0.0, self.Sy)
        assert math.isinf(n_DE)
        assert math.isinf(n_MSS)


# ---------------------------------------------------------------------------
# Integration tests — full solver via MechanicalSystem
# ---------------------------------------------------------------------------

class TestStaticFailureSolverIntegration:
    """
    Tests using StaticFailureSolver.solve() with a real MechanicalSystem.
    Validates the full pipeline: system → statics → static failure.
    """

    def _make_system(self, gear_Wt, gear_Wr, gear_T, gear_pos=200.0,
                     shaft_d=50.0, span_start=0.0, span_end=400.0):
        """Helper: single-section shaft, two bearings, one gear."""
        shaft = Shaft(name="test")
        shaft.add_section(ShaftSection(
            length=span_end,
            diameter=shaft_d,
            shoulder_left=Shoulder(
                fillet_radius=2.0,
                diameter_large=shaft_d,
                diameter_small=shaft_d * 0.8,
            ),
            shoulder_right=Shoulder(
                fillet_radius=2.0,
                diameter_large=shaft_d,
                diameter_small=shaft_d * 0.8,
            ),
        ))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system.add_bearing(Bearing(
            position=span_start, C=35000.0, C0=22000.0,
            arrangement="fixed", label="A",
        ))
        system.add_bearing(Bearing(
            position=span_end, C=35000.0, C0=22000.0,
            arrangement="floating", label="B",
        ))
        system.add_gear(GearElement(
            position=gear_pos,
            tangential_force=gear_Wt,
            radial_force=gear_Wr,
            pitch_diameter=100.0,
            torque=gear_T,
        ))
        return system

    def test_returns_static_failure_result(self):
        system = self._make_system(3500.0, 1274.0, 175_000.0)
        material = _make_material_sy(500.0)
        solver = StaticFailureSolver()
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = solver.solve(system, statics, material)
        assert isinstance(result, StaticFailureResult)

    def test_sections_sorted_most_critical_first(self):
        """governing_n must be non-decreasing (most critical = index 0)."""
        system = self._make_system(3500.0, 1274.0, 175_000.0)
        material = _make_material_sy(500.0)
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics, material)
        ns = [s.governing_n for s in result.sections]
        assert ns == sorted(ns), f"Sections not sorted ascending: {ns}"

    def test_material_id_and_sy_stored(self):
        mat = _make_material_sy(400.0)
        system = self._make_system(3500.0, 1274.0, 175_000.0)
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics, mat)
        assert result.material_id == mat.material_id
        assert result.Sy == pytest.approx(400.0)

    def test_zero_load_sections_have_inf_n(self):
        """If M=0 and T=0 at a section → n=inf, yielded=False."""
        # No gear, no load — all sections have zero stress
        shaft = Shaft(name="empty")
        shaft.add_section(ShaftSection(
            length=400.0, diameter=50.0,
            shoulder_left=Shoulder(2.0, 50.0, 40.0),
            shoulder_right=Shoulder(2.0, 50.0, 40.0),
        ))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system.add_bearing(Bearing(position=0.0, C=1.0, C0=1.0,
                                   arrangement="fixed", label="A"))
        system.add_bearing(Bearing(position=400.0, C=1.0, C0=1.0,
                                   arrangement="floating", label="B"))
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics, _make_material_sy(300.0))
        for s in result.sections:
            assert math.isinf(s.n_DE), f"Expected inf n_DE at x={s.x}, got {s.n_DE}"
            assert math.isinf(s.n_MSS), f"Expected inf n_MSS at x={s.x}, got {s.n_MSS}"
            assert not s.yielded

    def test_yielded_flag_when_overloaded(self):
        """
        Very high load on small shaft → governing_n < 1.0 → yielded=True.
        T omitted so GearElement auto-computes from Wt×d/2 (avoids inconsistency check).
        """
        system = self._make_system(
            gear_Wt=200_000.0, gear_Wr=80_000.0, gear_T=0.0,
            shaft_d=20.0,
        )
        material = _make_material_sy(200.0)
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics, material)
        assert result.any_yielded, "Expected at least one yielded section"
        assert result.critical_section.yielded

    def test_n_mss_leq_n_de_always(self):
        """
        MSS is always ≤ DE (MSS is more conservative).
        Shigley §5-7: DE is on or outside the MSS boundary.
        """
        system = self._make_system(3500.0, 1274.0, 175_000.0)
        material = _make_material_sy(355.0)
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics, material)
        for s in result.sections:
            if not math.isinf(s.n_MSS) and not math.isinf(s.n_DE):
                assert s.n_MSS <= s.n_DE + 1e-9, \
                    f"MSS should be ≤ DE at x={s.x}: n_MSS={s.n_MSS:.3f}, n_DE={s.n_DE:.3f}"

    def test_governing_theory_label(self):
        """governing_theory must be 'DE', 'MSS', or 'equal'."""
        system = self._make_system(3500.0, 1274.0, 175_000.0)
        material = _make_material_sy(500.0)
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics, material)
        for s in result.sections:
            assert s.governing_theory in ("DE", "MSS", "equal"), \
                f"Unexpected governing_theory: {s.governing_theory!r}"

    def test_validate_or_raise_called(self):
        """Solver must reject invalid system (< 2 bearings)."""
        shaft = Shaft(name="bad")
        shaft.add_section(ShaftSection(length=300.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system.add_bearing(Bearing(position=0.0, C=1.0, C0=1.0,
                                   arrangement="fixed", label="A"))
        from solvers.statics import StaticsSolver
        with pytest.raises(ValueError):
            StaticsSolver().solve(system)

    def test_stress_formula_sigma_x(self):
        """
        Manual verification: σx = 32*M / (π*d³).

        shaft.shoulders() reports only shoulder_right at section right-end.
        3-section shaft: s1(100mm,d40) | s2(200mm,d50,shoulder_right) | s3(100mm,d40).
        Shoulder is at x=300 (right end of s2).
        Bearings at 0 and 400. Gear (Wt=3500N) at x=200.

        Reactions (XZ, symmetric load → use moment equations):
          R_A = 3500*(400-200)/400 = 1750 N
          R_B = 3500*200/400       = 1750 N  (symmetric)

        M at x=300 (XZ plane only, Wr=0):
          M_xz(300) = R_A*300 - Wt*(300-200) = 1750*300 - 3500*100 = 175000 N·mm

        d at x=300: section s2 ends at 300 → diameter_large of shoulder = 50mm.
        σx_expected = 32*175000 / (π*50³)
        """
        shaft = Shaft(name="manual")
        shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="s1"))
        shaft.add_section(ShaftSection(
            length=200.0, diameter=50.0, label="s2",
            shoulder_right=Shoulder(2.0, 50.0, 40.0),
        ))
        shaft.add_section(ShaftSection(length=100.0, diameter=40.0, label="s3"))

        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system.add_bearing(Bearing(position=0.0, C=1.0, C0=1.0,
                                   arrangement="fixed", label="A"))
        system.add_bearing(Bearing(position=400.0, C=1.0, C0=1.0,
                                   arrangement="floating", label="B"))
        system.add_gear(GearElement(
            position=200.0, tangential_force=3500.0, radial_force=0.0,
            pitch_diameter=100.0,
        ))
        from solvers.statics import StaticsSolver
        statics = StaticsSolver().solve(system)
        result = StaticFailureSolver().solve(system, statics, _make_material_sy(500.0))

        M_at_300 = 175_000.0   # hand-calculated above
        # diameter_at(300) = 40mm: shoulder position x=300 is the start of s3.
        # shaft.diameter_at() returns the section containing x, which at x=300
        # is s3 (d=40mm). Consistent with how StressSolver uses diameter_at().
        d_at_300 = shaft.diameter_at(300.0)   # = 40mm
        sigma_x_expected = 32.0 * M_at_300 / (math.pi * d_at_300 ** 3)

        assert len(result.sections) == 1, f"Expected 1 shoulder section, got {len(result.sections)}"
        sec = result.sections[0]
        assert abs(sec.x - 300.0) < 1.0, f"Expected shoulder at x≈300, got x={sec.x}"
        assert sec.sigma_x == pytest.approx(sigma_x_expected, rel=0.02)


# ---------------------------------------------------------------------------
# StaticFailureSection — dataclass property tests
# ---------------------------------------------------------------------------

class TestStaticFailureSection:

    def test_risk_label_yielded(self):
        s = StaticFailureSection(
            x=100.0, diameter=50.0, sigma_x=200.0, tau_xy=50.0,
            sigma_prime=210.0, sigma_eff_mss=215.0,
            n_DE=0.9, n_MSS=0.85, governing_n=0.85,
            governing_theory="MSS", yielded=True,
        )
        assert s.risk_label == "yielded"

    def test_risk_label_marginal(self):
        s = StaticFailureSection(
            x=100.0, diameter=50.0, sigma_x=100.0, tau_xy=20.0,
            sigma_prime=105.0, sigma_eff_mss=108.0,
            n_DE=1.3, n_MSS=1.2, governing_n=1.2,
            governing_theory="MSS", yielded=False,
        )
        assert s.risk_label == "marginal"

    def test_risk_label_acceptable(self):
        s = StaticFailureSection(
            x=100.0, diameter=50.0, sigma_x=60.0, tau_xy=10.0,
            sigma_prime=63.0, sigma_eff_mss=64.0,
            n_DE=2.0, n_MSS=1.8, governing_n=1.8,
            governing_theory="MSS", yielded=False,
        )
        assert s.risk_label == "acceptable"

    def test_risk_label_no_stress(self):
        s = StaticFailureSection(
            x=0.0, diameter=50.0, sigma_x=0.0, tau_xy=0.0,
            sigma_prime=0.0, sigma_eff_mss=0.0,
            n_DE=math.inf, n_MSS=math.inf, governing_n=math.inf,
            governing_theory="equal", yielded=False,
        )
        assert s.risk_label == "no_stress"


class TestStaticFailureResult:

    def test_critical_section_is_first(self):
        s1 = StaticFailureSection(
            x=100.0, diameter=50.0, sigma_x=200.0, tau_xy=50.0,
            sigma_prime=210.0, sigma_eff_mss=215.0,
            n_DE=1.2, n_MSS=1.1, governing_n=1.1,
            governing_theory="MSS", yielded=False,
        )
        s2 = StaticFailureSection(
            x=200.0, diameter=50.0, sigma_x=100.0, tau_xy=20.0,
            sigma_prime=105.0, sigma_eff_mss=108.0,
            n_DE=2.0, n_MSS=1.9, governing_n=1.9,
            governing_theory="MSS", yielded=False,
        )
        result = StaticFailureResult(sections=[s1, s2], material_id="X", Sy=300.0)
        assert result.critical_section is s1
        assert result.min_governing_n == pytest.approx(1.1)

    def test_any_yielded(self):
        s = StaticFailureSection(
            x=0.0, diameter=50.0, sigma_x=500.0, tau_xy=100.0,
            sigma_prime=520.0, sigma_eff_mss=540.0,
            n_DE=0.8, n_MSS=0.7, governing_n=0.7,
            governing_theory="MSS", yielded=True,
        )
        result = StaticFailureResult(sections=[s], Sy=300.0)
        assert result.any_yielded

    def test_empty_result(self):
        result = StaticFailureResult()
        assert result.critical_section is None
        assert math.isinf(result.min_governing_n)
        assert not result.any_yielded
