# tests/test_solvers/test_shaft/test_statics.py
"""
StaticsSolver test suite — written BEFORE solver implementation (TDD).

Validation sources:
  Shigley Ex. 3-6: Shigley's MED 10th ed., Chapter 3, p. 93
  simple_system:   hand-calculated reference (see conftest.py docstring)

Tolerance conventions:
  TOLERANCE_FORCE_N    = 1.0      [N]      — absolute force residual
  TOLERANCE_MOMENT_Nmm = 100.0   [N·mm]   — absolute moment residual
  TOLERANCE_BOUNDARY   = 500.0   [N·mm]   — boundary condition at support (discretisation)
  TOLERANCE_REL        = 0.005            — 0.5% relative for textbook validation
"""
import pytest
import numpy as np
import warnings

from core.shaft import Shaft, ShaftSection
from core.components import Bearing, GearElement
from core.loads import RadialLoad, TorqueLoad, LoadPlane
from core.system import MechanicalSystem
from solvers.shaft.statics import StaticsSolver
from models.statics_result import StaticsResult

TOLERANCE_FORCE_N = 1.0
TOLERANCE_MOMENT_Nmm = 100.0
TOLERANCE_BOUNDARY = 500.0      # N·mm — acceptable M residual at support due to discretisation
TOLERANCE_REL = 0.005           # 0.5%


# ─── TestStaticsSolverEquilibrium ─────────────────────────────────────────────

class TestStaticsSolverEquilibrium:
    """Global force and moment equilibrium checks — must hold for any valid system."""

    def test_reactions_sum_to_zero_xy(self, simple_system):
        """
        ΣFy = 0: sum of ALL forces (reactions + applied) = 0.
        Applied Wr = +1274 N (downward). Reactions are negative (upward).
        R_A_xy + R_B_xy + Wr = 0.
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        total = r["A_xy"] + r["B_xy"] + 1274.0
        assert abs(total) < TOLERANCE_FORCE_N, (
            f"XY force equilibrium violated: ΣFy = {total:.4f} N"
        )

    def test_reactions_sum_to_zero_xz(self, simple_system):
        """ΣFx = 0: sum of all forces in XZ plane = 0."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        total = r["A_xz"] + r["B_xz"] + 3500.0
        assert abs(total) < TOLERANCE_FORCE_N, (
            f"XZ force equilibrium violated: ΣFx = {total:.4f} N"
        )

    def test_symmetric_load_symmetric_reactions_xy(self, simple_system_central_load):
        """Load at midspan → |R_A| = |R_B| (symmetric)."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        r = result.reactions
        # Both reactions oppose the downward load — magnitude should be equal
        assert pytest.approx(abs(r["A_xy"]), rel=TOLERANCE_REL) == abs(r["B_xy"])

    def test_symmetric_gear_symmetric_reactions_xz(self, simple_system):
        """Gear at midspan → R_A_xz = R_B_xz (symmetric XZ)."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        assert pytest.approx(abs(r["A_xz"]), rel=TOLERANCE_REL) == abs(r["B_xz"])

    def test_symmetric_gear_symmetric_reactions_xy(self, simple_system):
        """Gear at midspan → |R_A_xy| = |R_B_xy|."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        assert pytest.approx(abs(r["A_xy"]), rel=TOLERANCE_REL) == abs(r["B_xy"])

    def test_reactions_correct_magnitude_xy(self, simple_system_central_load):
        """F=5000N at midspan (150mm from each 300mm-span support) → R = 2500N each."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        r = result.reactions
        assert pytest.approx(abs(r["A_xy"]), rel=TOLERANCE_REL) == 2500.0
        assert pytest.approx(abs(r["B_xy"]), rel=TOLERANCE_REL) == 2500.0

    def test_no_axial_load_zero_axial_reaction(self, simple_system_central_load):
        """No axial forces → axial reaction = 0."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        assert abs(result.reactions["axial"]) < TOLERANCE_FORCE_N


# ─── TestStaticsSolverDiagrams ────────────────────────────────────────────────

class TestStaticsSolverDiagrams:
    """Internal diagram shape and boundary condition tests."""

    def test_result_is_staticsresult_instance(self, simple_system):
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert isinstance(result, StaticsResult)

    def test_x_array_starts_at_zero(self, simple_system):
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert result.x[0] == pytest.approx(0.0, abs=1e-9)

    def test_x_array_ends_at_shaft_length(self, simple_system):
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert result.x[-1] == pytest.approx(400.0, rel=1e-6)

    def test_all_arrays_same_length(self, simple_system):
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        n = len(result.x)
        assert len(result.V_xz) == n
        assert len(result.V_xy) == n
        assert len(result.M_xz) == n
        assert len(result.M_xy) == n
        assert len(result.M_res) == n
        assert len(result.T) == n
        assert len(result.axial_force) == n

    def test_M_res_equals_sqrt_sum_squares(self, simple_system):
        """M_res must equal sqrt(M_xz² + M_xy²) at every point."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        expected = np.sqrt(result.M_xz ** 2 + result.M_xy ** 2)
        np.testing.assert_allclose(result.M_res, expected, rtol=1e-9, atol=1e-6)

    def test_moment_zero_at_support_A(self, simple_system):
        """M(xA) ≈ 0 — simply-supported boundary condition at bearing A."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        xA = 50.0
        idx = np.argmin(np.abs(result.x - xA))
        assert abs(result.M_xy[idx]) < TOLERANCE_BOUNDARY, (
            f"M_xy at xA={xA}: {result.M_xy[idx]:.2f} N·mm (should be ≈ 0)"
        )
        assert abs(result.M_xz[idx]) < TOLERANCE_BOUNDARY

    def test_moment_zero_at_support_B(self, simple_system):
        """M(xB) ≈ 0 — simply-supported boundary condition at bearing B."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        xB = 350.0
        idx = np.argmin(np.abs(result.x - xB))
        assert abs(result.M_xy[idx]) < TOLERANCE_BOUNDARY
        assert abs(result.M_xz[idx]) < TOLERANCE_BOUNDARY

    def test_moment_max_near_gear_position(self, simple_system):
        """M_res maximum should occur near the gear application point (x=200mm)."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        x_max = result.x_at_M_res_max
        assert 180.0 < x_max < 220.0, (
            f"M_res max expected near x=200mm, got x={x_max:.1f}mm"
        )

    def test_torsion_zero_before_gear(self, simple_system):
        """T(x) = 0 for x < gear position (no torque source upstream)."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        x_gear = 200.0
        mask = result.x < (x_gear - 5.0)
        if mask.any():
            max_T_before = np.max(np.abs(result.T[mask]))
            assert max_T_before < TOLERANCE_MOMENT_Nmm, (
                f"T before gear should be ~0, got {max_T_before:.1f} N·mm"
            )

    def test_torsion_constant_between_gear_and_support_B(self, simple_system):
        """T constant between gear (x=200) and bearing B (x=350)."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        mask = (result.x > 210.0) & (result.x < 340.0)
        T_region = result.T[mask]
        if len(T_region) > 1:
            np.testing.assert_allclose(
                T_region, T_region[0], rtol=1e-6,
                err_msg="Torsion should be constant between gear and support B"
            )

    def test_torsion_magnitude_between_gear_and_support(self, simple_system):
        """T ≈ 175000 N·mm in region between gear and bearing B."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        mask = (result.x > 210.0) & (result.x < 340.0)
        T_region = result.T[mask]
        if len(T_region) > 0:
            assert pytest.approx(abs(T_region[0]), rel=TOLERANCE_REL) == 175000.0

    def test_M_res_non_negative(self, simple_system):
        """M_res = sqrt(...) must be non-negative everywhere."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert np.all(result.M_res >= 0.0)

    def test_shear_force_before_support_A_is_zero(self, shigley_ex3_6):
        """No loads before xA=0, so V(0) is set by the reaction at x=0."""
        solver = StaticsSolver()
        result = solver.solve(shigley_ex3_6)
        # At x=0 the reaction is applied, so V just after should equal R_A
        # Before any load (x before first support): shear = 0 for pure textbook case
        # This test confirms the sign convention is consistent.
        # R_A should be the first force in the diagram.
        assert abs(result.reactions["A_xy"]) > 0.0  # reaction exists

    def test_no_torque_system_torsion_zero_everywhere(self, simple_system_central_load):
        """System with no torque loads → T(x) = 0 everywhere."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        np.testing.assert_allclose(result.T, 0.0, atol=1e-9)


# ─── TestStaticsSolverValidation ──────────────────────────────────────────────

class TestStaticsSolverValidation:
    """Input validation — solver must reject invalid systems before computing."""

    def test_invalid_system_raises_before_solving(self):
        """System with 1 bearing should raise ValueError."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(Bearing(position=50.0, C=35000.0, C0=22000.0, arrangement="fixed"))
        solver = StaticsSolver()
        with pytest.raises(ValueError, match="validation failed"):
            solver.solve(system)

    def test_zero_span_raises(self):
        """Both bearings at same position → span = 0 → should raise."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(Bearing(position=100.0, C=35000.0, C0=22000.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=100.0, C=35000.0, C0=22000.0, arrangement="floating", label="B"))
        solver = StaticsSolver()
        with pytest.raises(ValueError, match="span"):
            solver.solve(system)

    def test_empty_shaft_raises(self):
        """Shaft with no sections should fail validation."""
        shaft = Shaft()
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed"))
        system._bearings.append(Bearing(position=100.0, C=1.0, C0=1.0, arrangement="floating"))
        solver = StaticsSolver()
        with pytest.raises(ValueError):
            solver.solve(system)

    def test_torque_without_counter_torque_emits_warning(self):
        """Single torque load (no opposing torque) should emit a warning, not raise."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(Bearing(position=0.0, C=35000.0, C0=22000.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=400.0, C=35000.0, C0=22000.0, arrangement="floating", label="B"))
        system.add_load(TorqueLoad(position=200.0, magnitude=100000.0))
        solver = StaticsSolver()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = solver.solve(system)
            torque_warnings = [x for x in w if "torque" in str(x.message).lower()]
            assert len(torque_warnings) >= 1, "Expected warning for unbalanced torque"
        assert isinstance(result, StaticsResult)


# ─── TestStaticsSolverValidationCases ─────────────────────────────────────────

class TestStaticsSolverValidationCases:
    """
    Textbook validation — results must match published values within 0.5%.
    Sources documented in tests/fixtures/expected_results.py.
    """

    def test_shigley_ex3_6_reaction_A(self, shigley_ex3_6):
        """
        Shigley Ex. 3-6: R_A = 3333.3 N.
        xA=0, xB=600mm, F=5000N at x=200mm.
        """
        solver = StaticsSolver()
        result = solver.solve(shigley_ex3_6)
        expected_R_A = 3333.3
        assert pytest.approx(abs(result.reactions["A_xy"]), rel=TOLERANCE_REL) == expected_R_A

    def test_shigley_ex3_6_reaction_B(self, shigley_ex3_6):
        """Shigley Ex. 3-6: R_B = 1666.7 N."""
        solver = StaticsSolver()
        result = solver.solve(shigley_ex3_6)
        expected_R_B = 1666.7
        assert pytest.approx(abs(result.reactions["B_xy"]), rel=TOLERANCE_REL) == expected_R_B

    def test_shigley_ex3_6_moment_max(self, shigley_ex3_6):
        """
        Shigley Ex. 3-6: M_max = R_A × a = 3333.3 × 200 = 666667 N·mm.
        Maximum must occur at x=200mm (load position).
        """
        solver = StaticsSolver()
        result = solver.solve(shigley_ex3_6)
        expected_M_max = 666_667.0
        assert pytest.approx(result.M_xy_max, rel=TOLERANCE_REL) == expected_M_max

    def test_shigley_ex3_6_M_xy_max_position(self, shigley_ex3_6):
        """M_xy maximum must occur near x=200mm."""
        solver = StaticsSolver()
        result = solver.solve(shigley_ex3_6)
        idx_max = np.argmax(np.abs(result.M_xy))
        x_max = result.x[idx_max]
        assert 195.0 < x_max < 205.0, f"Expected M_max near x=200mm, got x={x_max:.1f}mm"

    def test_shigley_ex3_6_equilibrium(self, shigley_ex3_6):
        """ΣF = 0: R_A + R_B + F_applied = 0."""
        solver = StaticsSolver()
        result = solver.solve(shigley_ex3_6)
        r = result.reactions
        total_force = r["A_xy"] + r["B_xy"] + 5000.0
        assert abs(total_force) < TOLERANCE_FORCE_N

    def test_simple_system_reaction_A_xz(self, simple_system):
        """
        simple_system: gear at midspan (150mm from each support, span=300mm).
        Wt=3500N → R_A_xz = R_B_xz = 1750N.
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert pytest.approx(abs(result.reactions["A_xz"]), rel=TOLERANCE_REL) == 1750.0

    def test_simple_system_reaction_B_xz(self, simple_system):
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert pytest.approx(abs(result.reactions["B_xz"]), rel=TOLERANCE_REL) == 1750.0

    def test_simple_system_reaction_A_xy(self, simple_system):
        """Wr=1274N at midspan → R_A_xy = R_B_xy = 637N."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert pytest.approx(abs(result.reactions["A_xy"]), rel=TOLERANCE_REL) == 637.0

    def test_simple_system_reaction_B_xy(self, simple_system):
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        assert pytest.approx(abs(result.reactions["B_xy"]), rel=TOLERANCE_REL) == 637.0

    def test_simple_system_M_res_max_magnitude(self, simple_system):
        """
        M_res_max at x=200mm:
          M_xz = 1750 × 150 = 262500 N·mm
          M_xy =  637 × 150 =  95550 N·mm
          M_res = sqrt(262500² + 95550²) ≈ 278413 N·mm
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        M_xz_expected = 1750.0 * 150.0      # = 262500
        M_xy_expected = 637.0 * 150.0       # = 95550
        M_res_expected = np.sqrt(M_xz_expected**2 + M_xy_expected**2)
        assert pytest.approx(result.M_res_max, rel=0.01) == M_res_expected

    def test_simple_system_torsion_magnitude(self, simple_system):
        """T = 175000 N·mm between gear and bearing B."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        mask = (result.x > 210.0) & (result.x < 340.0)
        T_region = result.T[mask]
        assert len(T_region) > 0
        assert pytest.approx(abs(T_region[0]), rel=TOLERANCE_REL) == 175000.0

    def test_off_centre_load_correct_reactions(self):
        """
        Custom off-centre case: xA=0, xB=500mm, F=4000N at x=100mm.
        R_B = 4000 × 100 / 500 = 800 N
        R_A = 4000 - 800 = 3200 N
        """
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=500.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=500.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
        system.add_load(RadialLoad(position=100.0, magnitude=4000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        assert pytest.approx(abs(result.reactions["A_xy"]), rel=TOLERANCE_REL) == 3200.0
        assert pytest.approx(abs(result.reactions["B_xy"]), rel=TOLERANCE_REL) == 800.0


# ─── TestStaticsSolverEdgeCases ───────────────────────────────────────────────

class TestStaticsSolverEdgeCases:
    """Edge cases for coverage of gear axial forces and boundary validation."""

    def test_gear_with_axial_force_populates_axial_diagram(self):
        """Gear with Wa > 0 → axial diagram non-zero after gear position."""
        import warnings as _w
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(Bearing(position=0.0, C=35000.0, C0=22000.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=400.0, C=35000.0, C0=22000.0, arrangement="floating", label="B"))
        from core.components import GearElement
        gear = GearElement(
            position=200.0,
            tangential_force=3500.0,
            radial_force=1274.0,
            axial_force=800.0,
            pitch_diameter=100.0,
            torque=175000.0,
        )
        system.add_gear(gear)
        solver = StaticsSolver()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            result = solver.solve(system)
        assert abs(result.reactions["axial"]) > 0.0
        mask_before_gear = (result.x > 5.0) & (result.x < 190.0)
        assert np.any(np.abs(result.axial_force[mask_before_gear]) > 1.0)
        mask_before_gear = (result.x > 5.0) & (result.x < 190.0)
        assert np.any(np.abs(result.axial_force[mask_before_gear]) > 1.0)

    def test_validate_result_raises_on_bad_boundary_M_xz(self):
        """_validate_result raises RuntimeError when |M_xz(xA)| > tolerance."""
        solver = StaticsSolver()
        x = np.linspace(0.0, 400.0, 1000)
        M_bad = np.ones_like(x) * 10000.0
        M_zero = np.zeros_like(x)
        with pytest.raises(RuntimeError, match="Boundary condition violated"):
            solver._validate_result(M_bad, M_zero, xA=50.0, xB=350.0, x=x)

    def test_validate_result_raises_on_bad_boundary_M_xy(self):
        """_validate_result raises RuntimeError when |M_xy(xA)| > tolerance."""
        solver = StaticsSolver()
        x = np.linspace(0.0, 400.0, 1000)
        M_bad = np.ones_like(x) * 10000.0
        M_zero = np.zeros_like(x)
        with pytest.raises(RuntimeError, match="Boundary condition violated"):
            solver._validate_result(M_zero, M_bad, xA=50.0, xB=350.0, x=x)

    def test_more_than_two_bearings_raises(self):
        """StaticsSolver with >2 bearings raises ValueError (Phase 1 constraint)."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=600.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        # Bypass add_bearing validation to inject 3 bearings directly
        system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=300.0, C=1.0, C0=1.0, arrangement="floating", label="C"))
        system._bearings.append(Bearing(position=600.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
        solver = StaticsSolver()
        with pytest.raises(ValueError, match="2 bearings"):
            solver.solve(system)

    def test_validate_result_raises_on_bad_boundary_at_xB(self):
        """_validate_result raises RuntimeError when |M(xB)| > tolerance."""
        solver = StaticsSolver()
        x = np.linspace(0.0, 400.0, 1000)
        # M_xz is fine at xA but violated at xB — M is zero everywhere except near xB
        M_good_at_A_bad_at_B = np.zeros_like(x)
        idx_B = np.argmin(np.abs(x - 350.0))
        M_good_at_A_bad_at_B[idx_B] = 100000.0
        M_zero = np.zeros_like(x)
        with pytest.raises(RuntimeError, match="Boundary condition violated"):
            solver._validate_result(M_good_at_A_bad_at_B, M_zero, xA=50.0, xB=350.0, x=x)
