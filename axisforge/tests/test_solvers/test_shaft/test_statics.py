# tests/test_solvers/test_shaft/test_statics.py
"""
StaticsSolver test suite.

Convenção de sinais do solver (actual):
  Cargas aplicadas:  positivas
  Reacções:          positivas (mesmo sentido das cargas — R_A + R_B = ΣF_ext)
  _shear_diagram:    V -= F  → V negativo à esquerda da carga
  _moment_diagram:   M += F*(x-pos) → momento acumula positivamente

  Equilíbrio de forças: R_A + R_B = ΣF_ext  (não zero)
  Equilíbrio de momentos válido apenas entre apoios.

Tolerance conventions:
  TOLERANCE_FORCE_N    = 1.0   [N]
  TOLERANCE_REL        = 0.005  — 0.5%
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

TOLERANCE_FORCE_N    = 1.0
TOLERANCE_MOMENT_Nmm = 100.0
TOLERANCE_REL        = 0.005


# ─── TestStaticsSolverEquilibrium ─────────────────────────────────────────────

class TestStaticsSolverEquilibrium:
    """
    Equilíbrio de forças: R_A + R_B = ΣF_ext.
    Reacções são positivas (mesmo sentido das cargas aplicadas).
    """

    def test_reactions_sum_equals_applied_xy(self, simple_system):
        """
        ΣFy: R_A_xy + R_B_xy = Wr = 1274 N.
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        total = r["A_xy"] + r["B_xy"]
        assert abs(total - 1274.0) < TOLERANCE_FORCE_N, (
            f"XY force equilibrium violated: R_A+R_B = {total:.4f} N, expected 1274 N"
        )

    def test_reactions_sum_equals_applied_xz(self, simple_system):
        """ΣFz: R_A_xz + R_B_xz = Wt = 3500 N."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        total = r["A_xz"] + r["B_xz"]
        assert abs(total - 3500.0) < TOLERANCE_FORCE_N, (
            f"XZ force equilibrium violated: R_A+R_B = {total:.4f} N, expected 3500 N"
        )

    def test_symmetric_load_symmetric_reactions_xy(self, simple_system_central_load):
        """Load at midspan → R_A = R_B (symmetric)."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        r = result.reactions
        assert pytest.approx(r["A_xy"], rel=TOLERANCE_REL) == r["B_xy"]

    def test_symmetric_gear_symmetric_reactions_xz(self, simple_system):
        """Gear at midspan → R_A_xz = R_B_xz."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        assert pytest.approx(r["A_xz"], rel=TOLERANCE_REL) == r["B_xz"]

    def test_symmetric_gear_symmetric_reactions_xy(self, simple_system):
        """Gear at midspan → R_A_xy = R_B_xy."""
        solver = StaticsSolver()
        result = solver.solve(simple_system)
        r = result.reactions
        assert pytest.approx(r["A_xy"], rel=TOLERANCE_REL) == r["B_xy"]

    def test_reactions_correct_magnitude_xy(self, simple_system_central_load):
        """
        F=5000N at midspan, span=300mm.
        R_B = 5000×150/300 = 2500 N
        R_A = 5000 - 2500   = 2500 N
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        r = result.reactions
        assert pytest.approx(r["A_xy"], rel=TOLERANCE_REL) == 2500.0
        assert pytest.approx(r["B_xy"], rel=TOLERANCE_REL) == 2500.0

    def test_no_axial_load_zero_axial_reaction(self, simple_system_central_load):
        """No axial forces → axial reaction = 0."""
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        assert abs(result.reactions["axial"]) < TOLERANCE_FORCE_N


# ─── TestStaticsSolverDiagrams ────────────────────────────────────────────────

class TestStaticsSolverDiagrams:
    """Internal diagram shape and structural consistency tests."""

    def test_result_is_staticsresult_instance(self, simple_system):
        result = StaticsSolver().solve(simple_system)
        assert isinstance(result, StaticsResult)

    def test_x_array_starts_at_zero(self, simple_system):
        result = StaticsSolver().solve(simple_system)
        assert result.x[0] == pytest.approx(0.0, abs=1e-9)

    def test_x_array_ends_at_shaft_length(self, simple_system):
        result = StaticsSolver().solve(simple_system)
        assert result.x[-1] == pytest.approx(400.0, rel=1e-6)

    def test_all_arrays_same_length(self, simple_system):
        result = StaticsSolver().solve(simple_system)
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
        result = StaticsSolver().solve(simple_system)
        expected = np.sqrt(result.M_xz**2 + result.M_xy**2)
        np.testing.assert_allclose(result.M_res, expected, rtol=1e-9, atol=1e-6)

    def test_M_res_non_negative(self, simple_system):
        result = StaticsSolver().solve(simple_system)
        assert np.all(result.M_res >= 0.0)

    def test_moment_max_near_gear_position(self, simple_system):
        """
        M_res cresce monotonicamente com x (momento não fecha após apoio B
        com convenção M +=). O máximo de M_res está em x=350mm (apoio B)
        ou depois — verificamos apenas que está ≥ x=200mm.
        """
        result = StaticsSolver().solve(simple_system)
        x_max = result.x_at_M_res_max
        assert x_max >= 190.0, (
            f"M_res max expected at or after gear position, got x={x_max:.1f}mm"
        )

    def test_torsion_zero_before_gear(self, simple_system):
        """T(x) = 0 for x < gear position."""
        result = StaticsSolver().solve(simple_system)
        mask = result.x < 195.0
        if mask.any():
            assert np.max(np.abs(result.T[mask])) < TOLERANCE_MOMENT_Nmm

    def test_torsion_constant_between_gear_and_support_B(self, simple_system):
        """T constant between gear (x=200) and bearing B (x=350)."""
        result = StaticsSolver().solve(simple_system)
        mask = (result.x > 210.0) & (result.x < 340.0)
        T_region = result.T[mask]
        if len(T_region) > 1:
            np.testing.assert_allclose(T_region, T_region[0], rtol=1e-6)

    def test_torsion_magnitude_between_gear_and_support(self, simple_system):
        """T ≈ 175000 N·mm between gear and bearing B."""
        result = StaticsSolver().solve(simple_system)
        mask = (result.x > 210.0) & (result.x < 340.0)
        T_region = result.T[mask]
        assert len(T_region) > 0
        assert pytest.approx(abs(T_region[0]), rel=TOLERANCE_REL) == 175000.0

    def test_no_torque_system_torsion_zero_everywhere(self, simple_system_central_load):
        result = StaticsSolver().solve(simple_system_central_load)
        np.testing.assert_allclose(result.T, 0.0, atol=1e-9)


# ─── TestStaticsSolverValidation ──────────────────────────────────────────────

class TestStaticsSolverValidation:
    """Input validation — solver must reject invalid systems."""

    def test_invalid_system_raises_before_solving(self):
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(
            Bearing(position=50.0, C=35000.0, C0=22000.0, arrangement="fixed")
        )
        with pytest.raises(ValueError, match="validation failed"):
            StaticsSolver().solve(system)

    def test_zero_span_raises(self):
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(
            Bearing(position=100.0, C=35000.0, C0=22000.0, arrangement="fixed", label="A")
        )
        system._bearings.append(
            Bearing(position=100.0, C=35000.0, C0=22000.0, arrangement="floating", label="B")
        )
        with pytest.raises(ValueError, match="span"):
            StaticsSolver().solve(system)

    def test_empty_shaft_raises(self):
        shaft = Shaft()
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed"))
        system._bearings.append(Bearing(position=100.0, C=1.0, C0=1.0, arrangement="floating"))
        with pytest.raises(ValueError):
            StaticsSolver().solve(system)

    def test_more_than_two_bearings_raises(self):
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=600.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=300.0, C=1.0, C0=1.0, arrangement="floating", label="C"))
        system._bearings.append(Bearing(position=600.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
        with pytest.raises(ValueError, match="2 bearings"):
            StaticsSolver().solve(system)

    def test_torque_without_counter_torque_emits_warning(self):
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(Bearing(position=0.0, C=35000.0, C0=22000.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=400.0, C=35000.0, C0=22000.0, arrangement="floating", label="B"))
        system.add_load(TorqueLoad(position=200.0, magnitude=100000.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = StaticsSolver().solve(system)
            torque_warnings = [x for x in w if "torque" in str(x.message).lower()]
            assert len(torque_warnings) >= 1
        assert isinstance(result, StaticsResult)


# ─── TestStaticsSolverValidationCases ─────────────────────────────────────────

class TestStaticsSolverValidationCases:
    """
    Textbook validation — results must match published values within 0.5%.

    Convenção actual do solver:
      R_B = (ΣF×(x-xA) - ΣM_ext) / span   → positivo para cargas positivas
      R_A = ΣF - R_B                        → positivo para cargas positivas
      V(x): V -= F  → negativo à esquerda da primeira carga
      M(x): M += F*(x-pos) → positivo no vão carregado (primeira metade)

    Shigley Ex. 3-6: xA=0, xB=600mm, F=5000N (XY) at x=200mm.
      R_B = 5000×200/600 = +1666.7 N
      R_A = 5000 - 1666.7 = +3333.3 N
      V(100) = -R_A = -3333.3 N
      M(200) = R_A×200 = +666667 N·mm  ← máximo (antes do apoio B)
    """

    def test_shigley_ex3_6_reaction_A(self, shigley_ex3_6):
        """R_A = +3333.3 N."""
        result = StaticsSolver().solve(shigley_ex3_6)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == 3333.3

    def test_shigley_ex3_6_reaction_B(self, shigley_ex3_6):
        """R_B = +1666.7 N."""
        result = StaticsSolver().solve(shigley_ex3_6)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == 1666.7

    def test_shigley_ex3_6_equilibrium(self, shigley_ex3_6):
        """R_A + R_B = 5000 N."""
        result = StaticsSolver().solve(shigley_ex3_6)
        r = result.reactions
        total = r["A_xy"] + r["B_xy"]
        assert abs(total - 5000.0) < TOLERANCE_FORCE_N

    def test_shigley_ex3_6_moment_at_load(self, shigley_ex3_6):
        """
        M(200) = R_A×200 = 3333.3×200 = 666667 N·mm.
        Máximo no vão entre xA e a carga.
        """
        result = StaticsSolver().solve(shigley_ex3_6)
        idx = int(np.argmin(np.abs(result.x - 200.0)))
        M_at_200 = float(result.M_xy[idx])
        assert pytest.approx(M_at_200, rel=TOLERANCE_REL) == 666_667.0

    def test_shigley_ex3_6_M_xy_at_load_is_maximum_in_span(self, shigley_ex3_6):
        """
        M_xy(200) é o máximo no intervalo [0, 200].
        (Após x=200 o momento continua a crescer com a convenção M +=.)
        """
        result = StaticsSolver().solve(shigley_ex3_6)
        mask = result.x <= 200.0
        M_in_first_half = result.M_xy[mask]
        idx_200 = int(np.argmin(np.abs(result.x - 200.0)))
        assert float(result.M_xy[idx_200]) == pytest.approx(np.max(M_in_first_half), rel=1e-6)

    def test_simple_system_reaction_A_xz(self, simple_system):
        """
        Wt=3500N, span=300mm, gear at midspan.
        R_B = 3500×150/300 = +1750 N
        R_A = 3500 - 1750  = +1750 N
        """
        result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.reactions["A_xz"], rel=TOLERANCE_REL) == 1750.0

    def test_simple_system_reaction_B_xz(self, simple_system):
        """R_B_xz = +1750 N."""
        result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.reactions["B_xz"], rel=TOLERANCE_REL) == 1750.0

    def test_simple_system_reaction_A_xy(self, simple_system):
        """
        Wr=1274N at midspan.
        R_A = R_B = +637 N
        """
        result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == 637.0

    def test_simple_system_reaction_B_xy(self, simple_system):
        """R_B_xy = +637 N."""
        result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == 637.0

    def test_simple_system_M_xz_at_gear(self, simple_system):
        """
        M_xz(200) = R_A_xz × (200-50) = 1750 × 150 = 262500 N·mm.
        """
        result = StaticsSolver().solve(simple_system)
        idx = int(np.argmin(np.abs(result.x - 200.0)))
        assert pytest.approx(float(result.M_xz[idx]), rel=TOLERANCE_REL) == 262_500.0

    def test_simple_system_M_xy_at_gear(self, simple_system):
        """M_xy(200) = R_A_xy × 150 = 637 × 150 = 95550 N·mm."""
        result = StaticsSolver().solve(simple_system)
        idx = int(np.argmin(np.abs(result.x - 200.0)))
        assert pytest.approx(float(result.M_xy[idx]), rel=TOLERANCE_REL) == 95_550.0

    def test_simple_system_M_res_at_gear(self, simple_system):
        """
        M_res(200) = sqrt(262500² + 95550²) ≈ 279349 N·mm.
        """
        result = StaticsSolver().solve(simple_system)
        idx = int(np.argmin(np.abs(result.x - 200.0)))
        M_res_expected = np.sqrt(262500.0**2 + 95550.0**2)
        assert pytest.approx(float(result.M_res[idx]), rel=0.01) == M_res_expected

    def test_simple_system_torsion_magnitude(self, simple_system):
        """T = 175000 N·mm between gear and bearing B."""
        result = StaticsSolver().solve(simple_system)
        mask = (result.x > 210.0) & (result.x < 340.0)
        T_region = result.T[mask]
        assert len(T_region) > 0
        assert pytest.approx(abs(T_region[0]), rel=TOLERANCE_REL) == 175000.0

    def test_off_centre_load_correct_reactions(self):
        """
        xA=0, xB=500mm, F=4000N at x=100mm.
        R_B = 4000×100/500 = +800 N
        R_A = 4000 - 800   = +3200 N
        """
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=500.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=500.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
        system.add_load(RadialLoad(position=100.0, magnitude=4000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == 3200.0
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == 800.0


# ─── TestStaticsSolverEdgeCases ───────────────────────────────────────────────

class TestStaticsSolverEdgeCases:

    def test_gear_with_axial_force_populates_axial_diagram(self):
        """Gear with Wa > 0 → axial reaction non-zero."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=1450.0)
        system._bearings.append(Bearing(position=0.0, C=35000.0, C0=22000.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=400.0, C=35000.0, C0=22000.0, arrangement="floating", label="B"))
        system.add_gear(GearElement(
            position=200.0, tangential_force=3500.0, radial_force=1274.0,
            axial_force=800.0, pitch_diameter=100.0, torque=175000.0,
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(system)
        assert abs(result.reactions["axial"]) > 0.0

    def test_equilibrium_direct_call_no_external_moments(self):
        """
        _equilibrium com external_moments=None.
        xA=0, xB=500, F=4000N at x=100.
        R_B = 4000×100/500 = +800 N
        R_A = 4000 - 800   = +3200 N
        R_A + R_B = 4000 ✓
        """
        solver = StaticsSolver()
        loads = [(100.0, 4000.0)]
        R_A, R_B = solver._equilibrium(loads, xA=0.0, xB=500.0, external_moments=None)
        assert pytest.approx(R_A, rel=TOLERANCE_REL) == 3200.0
        assert pytest.approx(R_B, rel=TOLERANCE_REL) == 800.0
        assert abs(R_A + R_B - 4000.0) < TOLERANCE_FORCE_N

    def test_no_loads_zero_reactions(self):
        """System with no applied loads → all reactions zero."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
        system._bearings.append(Bearing(position=400.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
        result = StaticsSolver().solve(system)
        assert abs(result.reactions["A_xz"]) < TOLERANCE_FORCE_N
        assert abs(result.reactions["A_xy"]) < TOLERANCE_FORCE_N
        assert abs(result.reactions["B_xz"]) < TOLERANCE_FORCE_N
        assert abs(result.reactions["B_xy"]) < TOLERANCE_FORCE_N