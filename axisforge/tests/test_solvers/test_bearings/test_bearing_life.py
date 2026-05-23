"""
tests/test_bearing_life.py
Validation tests for BearingLifeSolver.

Primary reference:
  SKF General Catalogue pub. 10000 EN, §2.1
  ISO 281:2007 §6, §7

Hand-calculated reference case (SKF 6210):
  C = 35 000 N, C0 = 22 000 N, type = DGBB → p = 3.0
  Fr = 5 000 N, Fa = 0 N, n = 1 450 rpm
  Phase 1: X=1, Y=0 → P = 5 000 N
  C/P = 7.0
  L10 = 7.0³ = 343.0 × 10⁶ rev
  L10h = 343×10⁶ / (60×1450) = 3 942.53 h
  X0=0.6, Y0=0.5 → P0 = 0.6×5000 + 0.5×0 = 3 000 N
  S0 = 22 000 / 3 000 = 7.333

Roller reference case (cylindrical roller, same loads):
  p = 10/3
  L10 = 7.0^(10/3) ≈ 573.1 × 10⁶ rev
  L10h = 573.1×10⁶ / (60×1450) ≈ 6 588 h
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from core.components import Bearing, BearingType
from core.loads import LoadPlane
from core.system import MechanicalSystem
from models.bearing_result import BearingLifeResult
from solvers.bearings.bearing_life import BearingLifeSolver


# ── Helpers ─────────────────────────────────────────────────────────────────

RTOL = 0.005   # 0.5% tolerance for textbook reference cases


def _skf6210_fixed(position: float = 50.0) -> Bearing:
    return Bearing(
        position=position,
        designation="6210",
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        C=35_000.0,
        C0=22_000.0,
        arrangement="fixed",
        label="A",
    )


def _skf6210_floating(position: float = 350.0) -> Bearing:
    return Bearing(
        position=position,
        designation="6210",
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        C=35_000.0,
        C0=22_000.0,
        arrangement="floating",
        label="B",
    )


def _roller_bearing(position: float = 50.0) -> Bearing:
    """Cylindrical roller — p = 10/3."""
    return Bearing(
        position=position,
        designation="NU210",
        bearing_type=BearingType.CYLINDRICAL_ROLLER,
        C=45_000.0,
        C0=32_000.0,
        arrangement="fixed",
        label="A",
    )


@pytest.fixture
def solver() -> BearingLifeSolver:
    return BearingLifeSolver()


# ── BearingLifeResult dataclass tests ────────────────────────────────────────

class TestBearingLifeResult:
    def _make(self, **kwargs) -> BearingLifeResult:
        defaults = dict(
            bearing_label="A", bearing_position=50.0,
            Fr=5000.0, Fa=0.0, P=5000.0, P0=3000.0,
            L10=343.0, L10h=3942.5, S0=7.33,
            C_over_P=7.0, life_target_hours=20_000.0,
            meets_life_target=False, meets_static_safety=True,
            X=1.0, Y=0.0, X0=0.6, Y0=0.5,
            life_exponent=3.0, speed_rpm=1450.0,
        )
        defaults.update(kwargs)
        return BearingLifeResult(**defaults)

    def test_is_safe_both_true(self):
        r = self._make(meets_life_target=True, meets_static_safety=True)
        assert r.is_safe is True

    def test_is_safe_life_fails(self):
        r = self._make(meets_life_target=False, meets_static_safety=True)
        assert r.is_safe is False

    def test_is_safe_static_fails(self):
        r = self._make(meets_life_target=True, meets_static_safety=False)
        assert r.is_safe is False

    def test_life_ratio(self):
        r = self._make(L10h=40_000.0, life_target_hours=20_000.0)
        assert math.isclose(r.life_ratio, 2.0, rel_tol=1e-6)

    def test_life_ratio_zero_target(self):
        r = self._make(life_target_hours=0.0)
        assert math.isinf(r.life_ratio)

    def test_repr_contains_label(self):
        r = self._make()
        assert "A" in repr(r)


# ── solve_bearing — primary reference case (SKF 6210) ───────────────────────

class TestSolveBearingSkf6210:
    """
    SKF 6210, DGBB, Fr=5000N, Fa=0, n=1450rpm.
    Source: SKF General Catalogue pub. 10000 EN, §2.1.
    """

    def test_C_over_P(self, solver):
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.C_over_P, 7.0, rel_tol=RTOL)

    def test_L10_million_rev(self, solver):
        """L10 = 7.0³ = 343.0 × 10⁶ rev."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.L10, 343.0, rel_tol=RTOL)

    def test_L10h(self, solver):
        """L10h = 343×10⁶ / (60×1450) ≈ 3942.5 h."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        expected = 343.0e6 / (60.0 * 1450.0)
        assert math.isclose(result.L10h, expected, rel_tol=RTOL)

    def test_P_phase1_radial_only(self, solver):
        """Phase 1: X=1, Y=0 → P = Fr."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.P, 5000.0, rel_tol=RTOL)

    def test_P0(self, solver):
        """P0 = 0.6*Fr + 0.5*Fa = 0.6*5000 + 0 = 3000 N."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.P0, 3000.0, rel_tol=RTOL)

    def test_S0(self, solver):
        """S0 = C0/P0 = 22000/3000 = 7.333."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.S0, 22000.0 / 3000.0, rel_tol=RTOL)

    def test_meets_life_target_false(self, solver):
        """L10h ≈ 3942h < 20000h default target."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0,
                                      design_life_hours=20_000.0)
        assert result.meets_life_target is False

    def test_meets_life_target_true(self, solver):
        """With target = 3000h, L10h ≈ 3942h → meets target."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0,
                                      design_life_hours=3_000.0)
        assert result.meets_life_target is True

    def test_meets_static_safety_true(self, solver):
        """S0 = 7.33 >> 1.0."""
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert result.meets_static_safety is True

    def test_life_exponent_ball(self, solver):
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.life_exponent, 3.0, rel_tol=1e-9)

    def test_X_Y_phase1(self, solver):
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert result.X == 1.0
        assert result.Y == 0.0

    def test_result_fields_populated(self, solver):
        bearing = _skf6210_fixed()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert result.bearing_label == "A"
        assert math.isclose(result.bearing_position, 50.0)
        assert math.isclose(result.Fr, 5000.0)
        assert math.isclose(result.Fa, 0.0)
        assert math.isclose(result.speed_rpm, 1450.0)


# ── solve_bearing — roller bearing (p = 10/3) ────────────────────────────────

class TestSolveBearingRoller:
    """
    Cylindrical roller bearing, same loads. Verifies p = 10/3 vs. p = 3.
    """

    def test_life_exponent_roller(self, solver):
        bearing = _roller_bearing()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.life_exponent, 10.0 / 3.0, rel_tol=1e-9)

    def test_L10_roller_greater_than_ball(self, solver):
        """
        Same C/P=9, p=10/3 gives higher L10 than p=3 for C/P > 1.
        9^(10/3) > 9^3.
        """
        b_ball = Bearing(position=50.0, bearing_type=BearingType.DEEP_GROOVE_BALL,
                         C=45_000.0, C0=32_000.0, arrangement="fixed", label="ball")
        b_roller = _roller_bearing()
        r_ball = solver.solve_bearing(b_ball, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        r_roller = solver.solve_bearing(b_roller, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert r_roller.L10 > r_ball.L10

    def test_L10_roller_value(self, solver):
        """
        NU210: C=45000, Fr=5000 → C/P=9.0
        L10 = 9^(10/3) = 9^3.333... 
        9^3 = 729; 9^(1/3) ≈ 2.0801 → L10 ≈ 729 * 2.0801 ≈ 1516.4
        """
        bearing = _roller_bearing()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        expected = 9.0 ** (10.0 / 3.0)
        assert math.isclose(result.L10, expected, rel_tol=RTOL)

    def test_taper_roller_exponent(self, solver):
        b = Bearing(position=50.0, bearing_type=BearingType.TAPER_ROLLER,
                    C=40_000.0, C0=28_000.0, arrangement="fixed", label="T")
        result = solver.solve_bearing(b, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.life_exponent, 10.0 / 3.0, rel_tol=1e-9)

    def test_spherical_roller_exponent(self, solver):
        b = Bearing(position=50.0, bearing_type=BearingType.SPHERICAL_ROLLER,
                    C=60_000.0, C0=45_000.0, arrangement="fixed", label="S")
        result = solver.solve_bearing(b, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.life_exponent, 10.0 / 3.0, rel_tol=1e-9)

    def test_angular_contact_ball_exponent(self, solver):
        b = Bearing(position=50.0, bearing_type=BearingType.ANGULAR_CONTACT_BALL,
                    C=35_000.0, C0=22_000.0, arrangement="fixed", label="AC")
        result = solver.solve_bearing(b, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)
        assert math.isclose(result.life_exponent, 3.0, rel_tol=1e-9)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_speed_returns_inf_L10h(self, solver):
        bearing = _skf6210_fixed()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=0.0)
        assert math.isinf(result.L10h)
        assert any("speed_rpm" in str(warning.message).lower() or
                   "speed" in str(warning.message).lower() for warning in w)

    def test_C_over_P_less_than_1_warns(self, solver):
        """C=10000, Fr=15000 → C/P < 1."""
        bearing = Bearing(position=50.0, C=10_000.0, C0=8_000.0,
                          arrangement="fixed", label="weak")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = solver.solve_bearing(bearing, Fr=15_000.0, Fa=0.0, speed_rpm=1450.0)
        assert result.C_over_P < 1.0
        assert any("C/P" in str(warning.message) or "c/p" in str(warning.message).lower()
                   for warning in w)

    def test_S0_below_minimum_warns(self, solver):
        """Very high load → S0 < 1.0 → warning."""
        bearing = Bearing(position=50.0, C=35_000.0, C0=5_000.0,
                          arrangement="fixed", label="low_c0")
        # P0 = 0.6 * Fr; need 0.6*Fr > C0=5000 → Fr > 8333
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = solver.solve_bearing(bearing, Fr=10_000.0, Fa=0.0, speed_rpm=1450.0)
        assert result.meets_static_safety is False
        assert any("S0" in str(warning.message) or "static" in str(warning.message).lower()
                   for warning in w)

    def test_zero_bearing_C_raises(self, solver):
        bearing = Bearing(position=50.0, C=0.0, C0=22_000.0,
                          arrangement="fixed", label="no_C")
        with pytest.raises(ValueError, match="catalogue"):
            solver.solve_bearing(bearing, Fr=5000.0, Fa=0.0, speed_rpm=1450.0)

    def test_zero_Fr_zero_Fa_uses_guard(self, solver):
        """Fr=Fa=0 → P clamped to 1.0 N minimum (no division by zero)."""
        bearing = _skf6210_fixed()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = solver.solve_bearing(bearing, Fr=0.0, Fa=0.0, speed_rpm=1450.0)
        assert result.P >= 1.0
        # C/P should be very large → very long life
        assert result.L10h > 1e9

    def test_axial_force_with_floating_arrangement(self, solver):
        """Floating bearing: Fa contribution still included in P0 if passed directly.
        extract_bearing_forces() zeroes Fa — but solve_bearing() itself accepts Fa as given.
        """
        bearing = _skf6210_floating()
        result = solver.solve_bearing(bearing, Fr=5000.0, Fa=500.0, speed_rpm=1450.0)
        # Phase 1: X=1, Y=0 → P = Fr regardless of Fa
        assert math.isclose(result.P, 5000.0, rel_tol=RTOL)
        # P0 includes Fa: 0.6*5000 + 0.5*500 = 3250
        assert math.isclose(result.P0, 0.6 * 5000.0 + 0.5 * 500.0, rel_tol=RTOL)


# ── extract_bearing_forces ────────────────────────────────────────────────────

class TestExtractBearingForces:
    """
    Uses simple_system fixture (from conftest):
      Bearings: A at x=50 (fixed), B at x=350 (floating)
      Gear: Wt=3500N (XZ), Wr=1274N (XY), symmetric between bearings
      Expected reactions (midspan): R_A_xz=1750N, R_B_xz=1750N,
                                    R_A_xy=637N, R_B_xy=637N
      Fr_A = sqrt(1750²+637²) ≈ 1863.6 N
      Fr_B = sqrt(1750²+637²) ≈ 1863.6 N
      Axial: gear.axial_force=0 → reactions["axial"]=0 → Fa_A=Fa_B=0
    """

    def test_returns_dict_with_two_keys(self, solver, simple_system):
        from solvers.shaft.statics import StaticsSolver
        sr = StaticsSolver().solve(simple_system)
        forces = solver.extract_bearing_forces(sr, simple_system)
        assert len(forces) == 2

    def test_keys_are_bearing_labels(self, solver, simple_system):
        from solvers.shaft.statics import StaticsSolver
        sr = StaticsSolver().solve(simple_system)
        forces = solver.extract_bearing_forces(sr, simple_system)
        assert "A" in forces
        assert "B" in forces

    def test_Fr_A_symmetric(self, solver, simple_system):
        from solvers.shaft.statics import StaticsSolver
        sr = StaticsSolver().solve(simple_system)
        forces = solver.extract_bearing_forces(sr, simple_system)
        Fr_A, _ = forces["A"]
        expected = math.sqrt(1750.0**2 + 637.0**2)
        assert math.isclose(Fr_A, expected, rel_tol=0.01)  # 1% — reactions are symmetric

    def test_Fr_B_symmetric(self, solver, simple_system):
        from solvers.shaft.statics import StaticsSolver
        sr = StaticsSolver().solve(simple_system)
        forces = solver.extract_bearing_forces(sr, simple_system)
        _, _ = forces["A"]
        Fr_B, _ = forces["B"]
        expected = math.sqrt(1750.0**2 + 637.0**2)
        assert math.isclose(Fr_B, expected, rel_tol=0.01)

    def test_Fa_zero_no_axial_gear(self, solver, simple_system):
        """Gear has axial_force=0 → Fa_A=Fa_B=0."""
        from solvers.shaft.statics import StaticsSolver
        sr = StaticsSolver().solve(simple_system)
        forces = solver.extract_bearing_forces(sr, simple_system)
        _, Fa_A = forces["A"]
        _, Fa_B = forces["B"]
        assert math.isclose(Fa_A, 0.0, abs_tol=1e-6)
        assert math.isclose(Fa_B, 0.0, abs_tol=1e-6)

    def test_Fa_fixed_bearing_gets_axial(self, solver, three_section_shaft):
        """
        System with axial load: fixed bearing A gets full Fa, floating B gets 0.
        """
        from core.components import Bearing, BearingType
        from core.loads import AxialLoad
        from core.system import MechanicalSystem
        from solvers.shaft.statics import StaticsSolver

        system = MechanicalSystem(shaft=three_section_shaft, speed_rpm=1450.0)
        system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                                   arrangement="fixed", label="A"))
        system.add_bearing(Bearing(position=350.0, C=35_000.0, C0=22_000.0,
                                   arrangement="floating", label="B"))
        system.add_load(AxialLoad(position=200.0, magnitude=2000.0, label="Fa_ext"))
        sr = StaticsSolver().solve(system)
        forces = solver.extract_bearing_forces(sr, system)
        _, Fa_A = forces["A"]
        _, Fa_B = forces["B"]
        assert math.isclose(abs(Fa_A), 2000.0, rel_tol=0.01)
        assert math.isclose(Fa_B, 0.0, abs_tol=1e-6)

    def test_Fa_two_fixed_bearings_split(self, solver, three_section_shaft):
        """
        Both bearings fixed: Fa split equally — each gets Fa_total/2.
        """
        from core.components import Bearing
        from core.loads import AxialLoad
        from core.system import MechanicalSystem
        from solvers.shaft.statics import StaticsSolver

        system = MechanicalSystem(shaft=three_section_shaft, speed_rpm=1450.0)
        system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                                   arrangement="fixed", label="A"))
        system.add_bearing(Bearing(position=350.0, C=35_000.0, C0=22_000.0,
                                   arrangement="fixed", label="B"))
        system.add_load(AxialLoad(position=200.0, magnitude=3000.0, label="Fa_ext"))
        sr = StaticsSolver().solve(system)
        forces = solver.extract_bearing_forces(sr, system)
        _, Fa_A = forces["A"]
        _, Fa_B = forces["B"]
        assert math.isclose(abs(Fa_A), 1500.0, rel_tol=0.01)
        assert math.isclose(abs(Fa_B), 1500.0, rel_tol=0.01)

    def test_no_fixed_bearings_warns(self, solver, three_section_shaft):
        """Zero fixed bearings → warning, Fa=0 for all."""
        from core.components import Bearing
        from core.loads import AxialLoad
        from core.system import MechanicalSystem
        from solvers.shaft.statics import StaticsSolver

        system = MechanicalSystem(shaft=three_section_shaft, speed_rpm=1450.0)
        system.add_bearing(Bearing(position=50.0, C=35_000.0, C0=22_000.0,
                                   arrangement="floating", label="A"))
        system.add_bearing(Bearing(position=350.0, C=35_000.0, C0=22_000.0,
                                   arrangement="floating", label="B"))
        system.add_load(AxialLoad(position=200.0, magnitude=1000.0))
        sr = StaticsSolver().solve(system)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            forces = solver.extract_bearing_forces(sr, system)
        assert any("fixed" in str(warning.message).lower() or
                   "axial" in str(warning.message).lower() for warning in w)
        for label, (_, Fa) in forces.items():
            assert math.isclose(Fa, 0.0, abs_tol=1e-6)

    def test_validate_or_raise_called(self, solver, three_section_shaft):
        """extract_bearing_forces must call system.validate_or_raise()."""
        from core.system import MechanicalSystem
        from models.statics_result import StaticsResult

        # System with only 1 bearing → fails validate_or_raise
        bad_system = MechanicalSystem(shaft=three_section_shaft, speed_rpm=0.0)
        bad_system.add_bearing(_skf6210_fixed())
        import numpy as np
        dummy_sr = StaticsResult(
            x=np.zeros(10), V_xz=np.zeros(10), V_xy=np.zeros(10),
            M_xz=np.zeros(10), M_xy=np.zeros(10), M_res=np.zeros(10),
            T=np.zeros(10), axial_force=np.zeros(10),
            reactions={"A_xz": 0, "A_xy": 0, "B_xz": 0, "B_xy": 0, "axial": 0},
        )
        with pytest.raises(ValueError):
            solver.extract_bearing_forces(dummy_sr, bad_system)


# ── Integration: full pipeline ────────────────────────────────────────────────

class TestIntegration:
    """
    Full pipeline: StaticsSolver → extract_bearing_forces → solve_bearing.
    Uses simple_system (symmetric midspan gear).
    """

    def test_full_pipeline_meets_static_safety(self, solver, simple_system):
        from solvers.shaft.statics import StaticsSolver
        sr = StaticsSolver().solve(simple_system)
        forces = solver.extract_bearing_forces(sr, simple_system)
        bearings = {b.label: b for b in simple_system.bearings}
        for label, (Fr, Fa) in forces.items():
            result = solver.solve_bearing(
                bearings[label], Fr=Fr, Fa=Fa,
                speed_rpm=simple_system.speed_rpm,
                design_life_hours=simple_system.design_life_hours,
            )
            assert result.meets_static_safety is True

    def test_full_pipeline_result_type(self, solver, simple_system):
        from solvers.shaft.statics import StaticsSolver
        sr = StaticsSolver().solve(simple_system)
        forces = solver.extract_bearing_forces(sr, simple_system)
        bearings = {b.label: b for b in simple_system.bearings}
        for label, (Fr, Fa) in forces.items():
            result = solver.solve_bearing(
                bearings[label], Fr=Fr, Fa=Fa,
                speed_rpm=simple_system.speed_rpm,
            )
            assert isinstance(result, BearingLifeResult)
