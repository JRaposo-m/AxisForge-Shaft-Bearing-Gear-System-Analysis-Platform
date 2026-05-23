# tests/test_solvers/test_shaft/test_statics_external_moment.py
"""
Tests for ExternalMoment handling in StaticsSolver.

Mathematical derivation (documented in solvers/shaft/statics.py):
  For a CCW couple M0 at position pos (xA=0, xB=L, no forces):
    R_B = +M0/L  (upward),  R_A = -M0/L  (downward)
    M(x<pos)  = R_A * x           = -(M0/L) * x
    M(x>=pos) = R_A * x + M0      = -(M0/L)*x + M0
    M(L)      = -(M0/L)*L + M0    = -M0 + M0 = 0  ✓

Reference case (from Session Brief):
  Shaft: 0→600mm, xA=0, xB=600mm
  ExternalMoment M0=120000 N·mm (XY) at x=200mm, no radial forces
  R_B = +120000/600 = +200 N
  R_A = -200 N
  M_xy(100) = -200*100 = -20000 N·mm
  M_xy(200) = -200*200 + 120000 = +80000 N·mm  (after step)
  M_xy(400) = -200*400 + 120000 = +40000 N·mm  [Note: sign consistent]
  M_xy(600) = -200*600 + 120000 = 0 ✓

  Wait — let me recheck with the actual implementation:
  _equilibrium: moment_about_A = 0 - M0 = -120000
  R_B = -(-120000)/600 = +200
  R_A = -0 - 200 = -200

  all_xy includes (0, -200) and (600, +200)
  _moment_diagram at x=400:
    From R_A at x=0: -200 * (400-0) = -80000
    From R_B at x=600: +200 * (400-600) = -80000 [x=400 < 600, so no contribution]
    Subtotal from forces: -80000
  _moment_external_diagram at x=400: +120000 (x=400 >= 200)
  Total M(400) = -80000 + 120000 = +40000 ✓

  At x=300:
    Forces: -200*300 = -60000
    External: +120000
    Total: +60000 ✓

  At x=600:
    Forces: -200*600 + 200*(600-600) = -120000 + 0 = -120000
    External: +120000
    Total: 0 ✓
"""
import pytest
import numpy as np
import warnings

from core.shaft import Shaft, ShaftSection
from core.components import Bearing
from core.loads import ExternalMoment, RadialLoad, LoadPlane
from core.system import MechanicalSystem
from solvers.shaft.statics import StaticsSolver
from models.statics_result import StaticsResult

TOLERANCE_REL = 0.005
TOLERANCE_ABS_M = 200.0   # N·mm — discretisation tolerance
TOLERANCE_ABS_F = 1.0     # N
TOLERANCE_BOUNDARY = 500.0  # N·mm


def build_system_600mm(speed_rpm: float = 0.0) -> MechanicalSystem:
    """Shaft 0→600mm, supports at xA=0, xB=600mm."""
    shaft = Shaft()
    shaft.add_section(ShaftSection(length=600.0, diameter=50.0))
    system = MechanicalSystem(shaft=shaft, speed_rpm=speed_rpm)
    system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
    system._bearings.append(Bearing(position=600.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
    return system


def value_at(array: np.ndarray, x_array: np.ndarray, x_target: float) -> float:
    idx = int(np.argmin(np.abs(x_array - x_target)))
    return float(array[idx])


class TestExternalMomentEquilibrium:
    """Equilibrium verification for systems with ExternalMoment loads."""

    def test_external_moment_reactions_force_equilibrium(self):
        """ΣF = 0: R_A + R_B = 0 (couples don't contribute to net force)."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        total_F = result.reactions["A_xy"] + result.reactions["B_xy"]
        assert abs(total_F) < TOLERANCE_ABS_F, f"ΣF = {total_F:.4f} N, expected 0"

    def test_external_moment_reaction_B_magnitude(self):
        """R_B = M0/L = 120000/600 = +200 N."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == 200.0

    def test_external_moment_reaction_A_magnitude(self):
        """R_A = -M0/L = -200 N."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == -200.0

    def test_external_moment_boundary_conditions(self):
        """M(xA) = M(xB) = 0 — simply supported boundary conditions."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        M_at_A = value_at(result.M_xy, result.x, 0.0)
        M_at_B = value_at(result.M_xy, result.x, 600.0)
        assert abs(M_at_A) < TOLERANCE_BOUNDARY, f"M(xA) = {M_at_A:.2f} N·mm"
        assert abs(M_at_B) < TOLERANCE_BOUNDARY, f"M(xB) = {M_at_B:.2f} N·mm"

    def test_external_moment_xz_plane_no_effect_on_xy(self):
        """XZ-plane moment does not affect XY reactions."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XZ))
        solver = StaticsSolver()
        result = solver.solve(system)
        assert abs(result.reactions["A_xy"]) < TOLERANCE_ABS_F
        assert abs(result.reactions["B_xy"]) < TOLERANCE_ABS_F

    def test_external_moment_xz_plane_affects_xz_reactions(self):
        """XZ-plane moment affects XZ reactions: R_B_xz = M0/L."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XZ))
        solver = StaticsSolver()
        result = solver.solve(system)
        assert pytest.approx(result.reactions["B_xz"], rel=TOLERANCE_REL) == 200.0
        assert pytest.approx(result.reactions["A_xz"], rel=TOLERANCE_REL) == -200.0


class TestExternalMomentDiagram:
    """Moment diagram shape verification for ExternalMoment loads."""

    def test_moment_step_before_application(self):
        """
        M_xy(100) = R_A * 100 = -200 * 100 = -20000 N·mm.
        No step from M0 yet (x=100 < pos=200).
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        M_100 = value_at(result.M_xy, result.x, 100.0)
        assert pytest.approx(M_100, abs=TOLERANCE_ABS_M) == -20_000.0

    def test_moment_step_at_application(self):
        """
        M_xy just after x=200: R_A*200 + M0 = -40000 + 120000 = +80000 N·mm.
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        M_210 = value_at(result.M_xy, result.x, 210.0)
        # M(210) = -200*210 + 120000 = -42000 + 120000 = +78000 (approx, 210 not 200)
        expected = -200.0 * 210.0 + 120_000.0
        assert pytest.approx(M_210, abs=TOLERANCE_ABS_M) == expected

    def test_moment_step_at_midspan_after_moment(self):
        """
        M_xy(400) = R_A*400 + M0 = -80000 + 120000 = +40000 N·mm.
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        M_400 = value_at(result.M_xy, result.x, 400.0)
        assert pytest.approx(M_400, abs=TOLERANCE_ABS_M) == 40_000.0

    def test_shear_force_zero_with_moment_only(self):
        """ExternalMoment only → no net force → V_xy = 0 everywhere. Wait —
        reactions R_A=-200 and R_B=+200 ARE forces, so V is non-zero.
        V(x<0)   = 0
        V(0<=x<600) = R_A = -200 N
        V(x>=600) = R_A + R_B = 0
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        V_at_300 = value_at(result.V_xy, result.x, 300.0)
        # Between A and B: V = R_A = -200 N
        assert pytest.approx(V_at_300, abs=1.0) == -200.0

    def test_external_moment_combined_with_radial_load(self):
        """
        Combined: RadialLoad F=5000N at x=300 + ExternalMoment M0=120000 at x=200.
        R_B = -(F*(300-0) + (-M0)) / 600 = -(1500000 - 120000)/600 = -1380000/600 = -2300 N
        R_A = -5000 - (-2300) = -2700 N
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        system.add_load(RadialLoad(position=300.0, magnitude=5000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)
        r = result.reactions
        total_F = r["A_xy"] + r["B_xy"] + 5000.0
        assert abs(total_F) < TOLERANCE_ABS_F, f"ΣF violated: {total_F:.3f}"
        assert pytest.approx(r["B_xy"], abs=1.0) == -2300.0
        assert pytest.approx(r["A_xy"], abs=1.0) == -2700.0

    def test_boundary_check_still_passes_with_external_moment(self):
        """_validate_result should not raise for a properly balanced system with M0."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=300.0, magnitude=60_000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        result = solver.solve(system)  # must not raise
        assert isinstance(result, StaticsResult)


class TestStaticsResultVMaxProperties:
    """Tests for V_xz_max and V_xy_max properties added to StaticsResult."""

    def test_V_xz_max_type_float(self, simple_system):
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        assert isinstance(result.V_xz_max, float)

    def test_V_xy_max_type_float(self, simple_system):
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        assert isinstance(result.V_xy_max, float)

    def test_V_xz_max_equals_max_abs_V_xz(self, simple_system):
        """V_xz_max must equal max(|V_xz(x)|)."""
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        expected = float(np.max(np.abs(result.V_xz)))
        assert pytest.approx(result.V_xz_max) == expected

    def test_V_xy_max_equals_max_abs_V_xy(self, simple_system):
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        expected = float(np.max(np.abs(result.V_xy)))
        assert pytest.approx(result.V_xy_max) == expected

    def test_V_xz_max_known_value(self, simple_system):
        """
        simple_system: Wt=3500N, span=300mm, gear at midspan.
        V_xz in region [50,200] = R_A_xz = 1750 N.
        V_xz in region [200,350] = 1750 N (symmetric).
        V_xz_max = 1750 N.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        assert pytest.approx(result.V_xz_max, rel=0.005) == 1750.0

    def test_V_xy_max_known_value(self, simple_system):
        """Wr=1274N, symmetric → V_xy_max = 637 N."""
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        assert pytest.approx(result.V_xy_max, rel=0.005) == 637.0

    def test_V_xz_max_non_negative(self, simple_system_central_load):
        """V_max must always be non-negative (it's max of absolute values)."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        assert result.V_xz_max >= 0.0
        assert result.V_xy_max >= 0.0

    def test_V_xy_max_central_load(self, simple_system_central_load):
        """Central load F=5000N, symmetric → V_xy_max = 2500 N."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        assert pytest.approx(result.V_xy_max, rel=0.005) == 2500.0

    def test_overhang_warning_contains_position(self):
        """Overhang warning message includes the overhang position."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(Bearing(position=100.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=350.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
        system.add_load(RadialLoad(position=0.0, magnitude=5000.0, plane=LoadPlane.XY))
        solver = StaticsSolver()
        with pytest.warns(UserWarning, match="0.0"):
            with pytest.raises(RuntimeError):
                solver.solve(system)
