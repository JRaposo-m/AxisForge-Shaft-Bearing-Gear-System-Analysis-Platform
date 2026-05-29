# tests/test_solvers/test_shaft/test_statics_external_moment.py
"""
Tests for ExternalMoment handling in StaticsSolver.

Convenção do solver (actual):
  R_B = (ΣF×(x-xA) - ΣM_ext) / span
  R_A = ΣF - R_B
  V -= F  |  M += F*(x-pos)

Caso de referência: xA=0, xB=600, M0=120000 N·mm (XY) a x=200, sem forças radiais.
  R_B = (0 - 120000) / 600 = -200 N
  R_A = 0 - (-200) = +200 N
  R_A + R_B = 0 ✓ (momento não contribui para ΣF)

V_xy (V -= F, all_xy = [(0, +200), (600, -200)]):
  0 <= x < 600: V = -(+200) = -200 N
  x >= 600:     V = -(+200 + (-200)) = 0 N

M_xy (M += F*(x-pos) + M_ext_step):
  M(100) = 200×100 = +20000 N·mm  (antes do step M_ext)
  Após x=200: M += 120000 (step de M_ext)
  M(210) ≈ 200×210 + 120000 = 42000+120000 = +162000 N·mm

Caso combinado: F=5000N (XY) a x=300 + M0=120000 (XY) a x=200, sem forças radiais antes.
  R_B = (5000×300 - (-120000)) / 600 = (1500000+120000)/600 = +2700 N
  R_A = 5000 - 2700 = +2300 N
  R_A + R_B = 5000 ✓
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

TOLERANCE_REL      = 0.005
TOLERANCE_ABS_M    = 200.0
TOLERANCE_ABS_F    = 1.0


def build_system_600mm(speed_rpm=0.0):
    shaft = Shaft()
    shaft.add_section(ShaftSection(length=600.0, diameter=50.0))
    system = MechanicalSystem(shaft=shaft, speed_rpm=speed_rpm)
    system._bearings.append(Bearing(position=0.0, C=1.0, C0=1.0, arrangement="fixed", label="A"))
    system._bearings.append(Bearing(position=600.0, C=1.0, C0=1.0, arrangement="floating", label="B"))
    return system


def value_at(array, x_array, x_target):
    idx = int(np.argmin(np.abs(x_array - x_target)))
    return float(array[idx])


class TestExternalMomentEquilibrium:
    """Verificação de equilíbrio de forças para sistemas com ExternalMoment."""

    def test_external_moment_reactions_force_equilibrium(self):
        """R_A + R_B = 0 (momento não contribui para ΣF)."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        total_F = result.reactions["A_xy"] + result.reactions["B_xy"]
        assert abs(total_F) < TOLERANCE_ABS_F

    def test_external_moment_reaction_B_magnitude(self):
        """R_B = (0 - 120000) / 600 = -200 N."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == -200.0

    def test_external_moment_reaction_A_magnitude(self):
        """R_A = 0 - (-200) = +200 N."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == +200.0

    def test_external_moment_xz_plane_no_effect_on_xy(self):
        """Momento no plano XZ não afecta reacções XY."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XZ))
        result = StaticsSolver().solve(system)
        assert abs(result.reactions["A_xy"]) < TOLERANCE_ABS_F
        assert abs(result.reactions["B_xy"]) < TOLERANCE_ABS_F

    def test_external_moment_xz_plane_affects_xz_reactions(self):
        """Momento XZ: R_B_xz = -200 N, R_A_xz = +200 N."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XZ))
        result = StaticsSolver().solve(system)
        assert pytest.approx(result.reactions["B_xz"], rel=TOLERANCE_REL) == -200.0
        assert pytest.approx(result.reactions["A_xz"], rel=TOLERANCE_REL) == +200.0


class TestExternalMomentDiagram:
    """Verificação do diagrama de momentos com ExternalMoment."""

    def test_moment_before_application(self):
        """
        M_xy(100): x<200, sem step de M_ext.
        M = R_A×100 = 200×100 = +20000 N·mm.
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        assert pytest.approx(value_at(result.M_xy, result.x, 100.0), abs=TOLERANCE_ABS_M) == +20_000.0

    def test_shear_force_between_supports(self):
        """
        V_xy entre A e B = -R_A = -200 N.
        (V -= R_A; momento externo não cria força de corte.)
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        assert pytest.approx(value_at(result.V_xy, result.x, 300.0), abs=TOLERANCE_ABS_F) == -200.0

    def test_external_moment_combined_with_radial_load(self):
        """
        F=5000N (XY) a x=300 + M0=120000 (XY) a x=200.
        R_B = (5000×300 - (-120000)) / 600 = +2700 N
        R_A = 5000 - 2700 = +2300 N
        R_A + R_B = 5000 ✓
        """
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=200.0, magnitude=120_000.0, plane=LoadPlane.XY))
        system.add_load(RadialLoad(position=300.0, magnitude=5000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        r = result.reactions
        assert abs(r["A_xy"] + r["B_xy"] - 5000.0) < TOLERANCE_ABS_F
        assert pytest.approx(r["A_xy"], abs=1.0) == +2700.0
        assert pytest.approx(r["B_xy"], abs=1.0) == +2300.0

    def test_result_is_staticsresult(self):
        """Solver não levanta excepção com ExternalMoment válido."""
        system = build_system_600mm()
        system.add_load(ExternalMoment(position=300.0, magnitude=60_000.0, plane=LoadPlane.XY))
        result = StaticsSolver().solve(system)
        assert isinstance(result, StaticsResult)


class TestStaticsResultVMaxProperties:
    """Testes para as propriedades V_xz_max e V_xy_max do StaticsResult."""

    def test_V_xz_max_type_float(self, simple_system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert isinstance(result.V_xz_max, float)

    def test_V_xy_max_type_float(self, simple_system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert isinstance(result.V_xy_max, float)

    def test_V_xz_max_equals_max_abs_V_xz(self, simple_system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.V_xz_max) == float(np.max(np.abs(result.V_xz)))

    def test_V_xy_max_equals_max_abs_V_xy(self, simple_system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.V_xy_max) == float(np.max(np.abs(result.V_xy)))

    def test_V_xz_max_known_value(self, simple_system):
        """
        V_xz_max = max(|V_xz|).
        V acumula: 0, -1750, -5250, -7000 → max = 7000 N.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.V_xz_max, rel=0.005) == 7000.0

    def test_V_xy_max_known_value(self, simple_system):
        """
        V_xy acumula: 0, -637, -1911, -2548 → max = 2548 N.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(result.V_xy_max, rel=0.005) == 2548.0

    def test_V_xz_max_non_negative(self, simple_system_central_load):
        result = StaticsSolver().solve(simple_system_central_load)
        assert result.V_xz_max >= 0.0
        assert result.V_xy_max >= 0.0

    def test_V_xy_max_central_load(self, simple_system_central_load):
        """
        F=5000N centrada → V acumula: 0, -2500, -7500, -10000 → max = 10000 N.
        """
        result = StaticsSolver().solve(simple_system_central_load)
        assert pytest.approx(result.V_xy_max, rel=0.005) == 10000.0