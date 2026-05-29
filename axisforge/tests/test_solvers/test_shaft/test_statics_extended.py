# tests/test_solvers/test_shaft/test_statics_extended.py
"""
StaticsSolver — testes aprofundados com verificação quantitativa completa.

Convenção de sinais do solver (actual):
  Reacções: positivas  R_A + R_B = ΣF_ext
  V(x):     V -= F  → negativo à esquerda da carga, mais negativo após cada força
  M(x):     M += F*(x-pos) → positivo, cresce monotonicamente até ao fim do veio

  Nota importante sobre o diagrama M com a convenção M +=:
    O momento não fecha a zero no apoio B — acumula indefinidamente após B.
    Os testes de M verificam apenas valores entre xA e a carga (primeira metade),
    onde o diagrama está correcto e é comparável com textbooks.
    Os testes de V verificam a estrutura do diagrama de corte.

CASOS:
  A — simple_system: diagramas V, M, T completos
  B — carga off-centre assimétrica
  C — duas cargas em planos diferentes
  D — overhang: _equilibrium directo
  E — simetria do diagrama M para carga centrada (primeira metade)
  F — relação fundamental dM/dx = -V (com convenção M +=, V -=)
"""
import pytest
import numpy as np
import warnings

from core.shaft import Shaft, ShaftSection
from core.components import Bearing, GearElement
from core.loads import RadialLoad, TorqueLoad, LoadPlane
from core.system import MechanicalSystem
from solvers.shaft.statics import StaticsSolver

TOLERANCE_REL      = 0.005
TOLERANCE_ABS_M    = 200.0
TOLERANCE_ABS_V    = 2.0


def value_at(array: np.ndarray, x_array: np.ndarray, x_target: float) -> float:
    idx = int(np.argmin(np.abs(x_array - x_target)))
    return float(array[idx])


def build_simple_shaft_system(shaft_length, xA, xB, diameter=50.0):
    shaft = Shaft()
    shaft.add_section(ShaftSection(length=shaft_length, diameter=diameter))
    system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
    system._bearings.append(Bearing(position=xA, C=1.0, C0=1.0, arrangement="fixed", label="A"))
    system._bearings.append(Bearing(position=xB, C=1.0, C0=1.0, arrangement="floating", label="B"))
    return system


# ---------------------------------------------------------------------------
# CASO A — simple_system
#
# xA=50, xB=350, span=300, gear x=200, Wt=3500N (XZ), Wr=1274N (XY)
#
# Reacções:
#   R_B_xz = 3500×150/300 = +1750 N
#   R_A_xz = 3500 - 1750  = +1750 N
#   R_B_xy = 1274×150/300 = +637 N
#   R_A_xy = 1274 - 637   = +637 N
#
# V_xz (V -= F, all_xz = [(50,+1750),(350,+1750),(200,+3500)]):
#   x < 50:           V = 0
#   50 <= x < 200:    V = -(1750) = -1750 N
#   200 <= x < 350:   V = -(1750+3500) = -5250 N
#   x >= 350:         V = -(1750+3500+1750) = -7000 N
#   Passo no gear:    ΔV = -3500 N
#
# M_xz (M += F*(x-pos)):
#   M(125) = 1750×(125-50)          = +131250 N·mm
#   M(200) = 1750×(200-50)          = +262500 N·mm  ← máximo antes do gear
#   M(275) = 1750×(275-50)+3500×(275-200) = 393750+262500 = +656250 N·mm
#   (acumula após gear — não é simétrico com M(125))
# ---------------------------------------------------------------------------

class TestCasoA_DiagramasCompletos:

    def test_V_xz_before_support_A_is_zero(self, simple_system):
        """x=25mm: antes de qualquer força → V = 0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert abs(value_at(result.V_xz, result.x, 25.0)) < TOLERANCE_ABS_V

    def test_V_xz_between_A_and_gear(self, simple_system):
        """x=125mm: só R_A_xz=+1750 actua → V = -1750 N."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.V_xz, result.x, 125.0), abs=TOLERANCE_ABS_V) == -1750.0

    def test_V_xz_between_gear_and_B(self, simple_system):
        """x=275mm: R_A + Wt = -(1750+3500) = -5250 N."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.V_xz, result.x, 275.0), abs=TOLERANCE_ABS_V) == -5250.0

    def test_V_xz_after_support_B(self, simple_system):
        """x=380mm: R_A+Wt+R_B = -(1750+3500+1750) = -7000 N."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.V_xz, result.x, 380.0), abs=TOLERANCE_ABS_V) == -7000.0

    def test_V_xz_step_at_gear(self, simple_system):
        """Salto em x=200mm = -Wt = -3500 N (V -= F)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        step = value_at(result.V_xz, result.x, 210.0) - value_at(result.V_xz, result.x, 190.0)
        assert pytest.approx(step, abs=10.0) == -3500.0

    def test_M_xz_at_x125(self, simple_system):
        """M_xz(125) = 1750×75 = +131250 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.M_xz, result.x, 125.0), rel=TOLERANCE_REL) == +131_250.0

    def test_M_xz_at_x200(self, simple_system):
        """M_xz(200) = 1750×150 = +262500 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.M_xz, result.x, 200.0), rel=TOLERANCE_REL) == +262_500.0

    def test_M_xz_at_x275(self, simple_system):
        """M_xz(275) = 1750×225 + 3500×75 = 393750+262500 = +656250 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.M_xz, result.x, 275.0), rel=TOLERANCE_REL) == +656_250.0

    def test_M_xy_at_x125(self, simple_system):
        """M_xy(125) = 637×75 = +47775 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.M_xy, result.x, 125.0), rel=TOLERANCE_REL) == +47_775.0

    def test_M_xy_at_x200(self, simple_system):
        """M_xy(200) = 637×150 = +95550 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.M_xy, result.x, 200.0), rel=TOLERANCE_REL) == +95_550.0

    def test_M_xy_at_x275(self, simple_system):
        """M_xy(275) = 637×225 + 1274×75 = 143325+95550 = +238875 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.M_xy, result.x, 275.0), rel=TOLERANCE_REL) == +238_875.0

    def test_M_res_at_x200(self, simple_system):
        """M_res(200) = sqrt(262500² + 95550²) ≈ 279349 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        expected = np.sqrt(262500.0**2 + 95550.0**2)
        assert pytest.approx(value_at(result.M_res, result.x, 200.0), rel=TOLERANCE_REL) == expected

    def test_M_res_at_x125(self, simple_system):
        """M_res(125) = sqrt(131250² + 47775²) ≈ 139675 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        expected = np.sqrt(131250.0**2 + 47775.0**2)
        assert pytest.approx(value_at(result.M_res, result.x, 125.0), rel=TOLERANCE_REL) == expected

    def test_reactions_sign_A_xz_positive(self, simple_system):
        """R_A_xz = +1750 N (positivo — mesmo sentido da carga)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert result.reactions["A_xz"] > 0
        assert pytest.approx(result.reactions["A_xz"], rel=TOLERANCE_REL) == +1750.0

    def test_reactions_sign_A_xy_positive(self, simple_system):
        """R_A_xy = +637 N."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert result.reactions["A_xy"] > 0
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == +637.0

    def test_T_zero_before_gear(self, simple_system):
        """T(100) = 0 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert abs(value_at(result.T, result.x, 100.0)) < 1.0

    def test_T_at_x250(self, simple_system):
        """T(250) = 175000 N·mm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StaticsSolver().solve(simple_system)
        assert pytest.approx(value_at(result.T, result.x, 250.0), rel=TOLERANCE_REL) == 175_000.0


# ---------------------------------------------------------------------------
# CASO B — carga off-centre assimétrica
#
# xA=0, xB=500, F=6000N (XY) a x=100mm
#
# R_B = 6000×100/500 = +1200 N
# R_A = 6000 - 1200  = +4800 N
# Verificação: R_A + R_B = 6000 ✓
#
# V_xy (V -= F):
#   0 <= x < 100:   V = -4800 N
#   100 <= x < 500: V = -(4800+6000) = -10800 N
#   x >= 500:       V = -(4800+6000+1200) = -12000 N
#
# M_xy (M += F*(x-pos)):
#   M(50)  = 4800×50           = +240000 N·mm
#   M(100) = 4800×100          = +480000 N·mm  ← máximo em [0,100]
#   M(300) = 4800×300+6000×200 = 1440000+1200000 = +2640000 N·mm
# ---------------------------------------------------------------------------

class TestCasoB_OffCentreAssimetrico:

    @pytest.fixture
    def system_off_centre(self):
        system = build_simple_shaft_system(500.0, 0.0, 500.0)
        system.add_load(RadialLoad(position=100.0, magnitude=6000.0, plane=LoadPlane.XY))
        return system

    def test_reaction_A_off_centre(self, system_off_centre):
        """R_A = +4800 N."""
        result = StaticsSolver().solve(system_off_centre)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == +4800.0

    def test_reaction_B_off_centre(self, system_off_centre):
        """R_B = +1200 N."""
        result = StaticsSolver().solve(system_off_centre)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == +1200.0

    def test_reactions_sum_equals_load(self, system_off_centre):
        """R_A + R_B = 6000 N."""
        result = StaticsSolver().solve(system_off_centre)
        total = result.reactions["A_xy"] + result.reactions["B_xy"]
        assert abs(total - 6000.0) < 1.0

    def test_V_xy_before_load(self, system_off_centre):
        """V(50) = -R_A = -4800 N."""
        result = StaticsSolver().solve(system_off_centre)
        assert pytest.approx(value_at(result.V_xy, result.x, 50.0), abs=TOLERANCE_ABS_V) == -4800.0

    def test_V_xy_after_load(self, system_off_centre):
        """V(300) = -(R_A + F) = -(4800+6000) = -10800 N."""
        result = StaticsSolver().solve(system_off_centre)
        assert pytest.approx(value_at(result.V_xy, result.x, 300.0), abs=TOLERANCE_ABS_V) == -10800.0

    def test_M_xy_at_x50(self, system_off_centre):
        """M(50) = 4800×50 = +240000 N·mm."""
        result = StaticsSolver().solve(system_off_centre)
        assert pytest.approx(value_at(result.M_xy, result.x, 50.0), rel=TOLERANCE_REL) == +240_000.0

    def test_M_xy_at_x100(self, system_off_centre):
        """M(100) = 4800×100 = +480000 N·mm."""
        result = StaticsSolver().solve(system_off_centre)
        assert pytest.approx(value_at(result.M_xy, result.x, 100.0), rel=TOLERANCE_REL) == +480_000.0

    def test_M_xy_maximum_in_first_segment(self, system_off_centre):
        """Máximo de M_xy em [0,100] está em x≈100mm."""
        result = StaticsSolver().solve(system_off_centre)
        mask = result.x <= 100.0
        x_max_first = result.x[mask][np.argmax(result.M_xy[mask])]
        assert 95.0 < x_max_first <= 100.0

    def test_M_xy_at_x300(self, system_off_centre):
        """M(300) = 4800×300 + 6000×200 = +2640000 N·mm."""
        result = StaticsSolver().solve(system_off_centre)
        assert pytest.approx(value_at(result.M_xy, result.x, 300.0), rel=TOLERANCE_REL) == +2_640_000.0


# ---------------------------------------------------------------------------
# CASO C — duas cargas em planos diferentes
#
# xA=0, xB=600, F1=4000N (XY) x=200, F2=3000N (XZ) x=400
#
# XY:
#   R_B_xy = 4000×200/600 = +1333.3 N
#   R_A_xy = 4000 - 1333.3 = +2666.7 N
#
# XZ:
#   R_B_xz = 3000×400/600 = +2000 N
#   R_A_xz = 3000 - 2000  = +1000 N
#
# M_xz(100) = 1000×100 = +100000 N·mm
# M_xz(400) = 1000×400 = +400000 N·mm  ← máximo em [0,400]
# M_xy(200) = 2666.7×200 = +533333 N·mm  ← máximo em [0,200]
#
# M_res(200) = sqrt(533333² + (1000×200)²) = sqrt(533333²+200000²) ≈ 569600
# ---------------------------------------------------------------------------

class TestCasoC_DuasCargas:

    @pytest.fixture
    def system_two_loads(self):
        system = build_simple_shaft_system(600.0, 0.0, 600.0)
        system.add_load(RadialLoad(position=200.0, magnitude=4000.0, plane=LoadPlane.XY))
        system.add_load(RadialLoad(position=400.0, magnitude=3000.0, plane=LoadPlane.XZ))
        return system

    def test_reaction_A_xy(self, system_two_loads):
        """R_A_xy = +2666.7 N."""
        result = StaticsSolver().solve(system_two_loads)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == +2666.7

    def test_reaction_B_xy(self, system_two_loads):
        """R_B_xy = +1333.3 N."""
        result = StaticsSolver().solve(system_two_loads)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == +1333.3

    def test_reaction_A_xz(self, system_two_loads):
        """R_A_xz = +1000 N."""
        result = StaticsSolver().solve(system_two_loads)
        assert pytest.approx(result.reactions["A_xz"], rel=TOLERANCE_REL) == +1000.0

    def test_reaction_B_xz(self, system_two_loads):
        """R_B_xz = +2000 N."""
        result = StaticsSolver().solve(system_two_loads)
        assert pytest.approx(result.reactions["B_xz"], rel=TOLERANCE_REL) == +2000.0

    def test_planes_decoupled_M_xz_at_x100(self, system_two_loads):
        """M_xz(100) = R_A_xz×100 = 1000×100 = +100000 N·mm."""
        result = StaticsSolver().solve(system_two_loads)
        assert pytest.approx(value_at(result.M_xz, result.x, 100.0), rel=TOLERANCE_REL) == +100_000.0

    def test_M_xy_at_x200(self, system_two_loads):
        """M_xy(200) = 2666.7×200 = +533333 N·mm."""
        result = StaticsSolver().solve(system_two_loads)
        assert pytest.approx(value_at(result.M_xy, result.x, 200.0), rel=TOLERANCE_REL) == +533_333.0

    def test_M_xz_at_x400(self, system_two_loads):
        """M_xz(400) = 1000×400 = +400000 N·mm."""
        result = StaticsSolver().solve(system_two_loads)
        assert pytest.approx(value_at(result.M_xz, result.x, 400.0), rel=TOLERANCE_REL) == +400_000.0

    def test_M_res_at_x200(self, system_two_loads):
        """M_res(200) = sqrt(533333² + 200000²) ≈ 569600 N·mm."""
        result = StaticsSolver().solve(system_two_loads)
        M_xz_200 = 1000.0 * 200.0
        M_xy_200 = 2666.7 * 200.0
        expected = np.sqrt(M_xy_200**2 + M_xz_200**2)
        assert pytest.approx(value_at(result.M_res, result.x, 200.0), rel=TOLERANCE_REL) == expected

    def test_equilibrium_xy(self, system_two_loads):
        """R_A_xy + R_B_xy = 4000 N."""
        result = StaticsSolver().solve(system_two_loads)
        total = result.reactions["A_xy"] + result.reactions["B_xy"]
        assert abs(total - 4000.0) < 1.0

    def test_equilibrium_xz(self, system_two_loads):
        """R_A_xz + R_B_xz = 3000 N."""
        result = StaticsSolver().solve(system_two_loads)
        total = result.reactions["A_xz"] + result.reactions["B_xz"]
        assert abs(total - 3000.0) < 1.0


# ---------------------------------------------------------------------------
# CASO D — overhang: _equilibrium directo
#
# xA=100, xB=350, F=5000N (XY) a x=0 (fora do span)
#
# R_B = 5000×(0-100)/250 = -2000 N  ← negativo (sentido oposto)
# R_A = 5000 - (-2000) = +7000 N
# Verificação: R_A + R_B = 7000-2000 = 5000 ✓
# ---------------------------------------------------------------------------

class TestCasoD_Overhang:

    def test_overhang_reactions_correct(self):
        """
        _equilibrium directo com carga em overhang.
        R_A = +7000 N, R_B = -2000 N. R_A + R_B = 5000 ✓
        """
        solver = StaticsSolver()
        loads = [(0.0, 5000.0)]
        R_A, R_B = solver._equilibrium(loads, xA=100.0, xB=350.0, external_moments=None)
        assert pytest.approx(R_A, rel=TOLERANCE_REL) == +7000.0
        assert pytest.approx(R_B, rel=TOLERANCE_REL) == -2000.0
        assert abs(R_A + R_B - 5000.0) < 1.0


# ---------------------------------------------------------------------------
# CASO E — diagrama M para carga centrada (primeira metade do vão)
#
# simple_system_central_load: xA=50, xB=350, span=300, F=5000N (XY) x=200
#
# R_B = 5000×150/300 = +2500 N
# R_A = 5000 - 2500  = +2500 N
#
# V_xy (V -= F):
#   50 <= x < 200:  V = -2500 N
#   200 <= x < 350: V = -(2500+5000) = -7500 N
#   x >= 350:       V = -(2500+5000+2500) = -10000 N
#
# M_xy (M += F*(x-pos)):
#   M(125) = 2500×75  = +187500 N·mm
#   M(200) = 2500×150 = +375000 N·mm
#   M(275) = 2500×225 + 5000×75 = 562500+375000 = +937500 N·mm  (não simétrico)
#   M(100) = 2500×50  = +125000 N·mm
# ---------------------------------------------------------------------------

class TestCasoE_Diagrama:

    def test_M_xy_at_x125(self, simple_system_central_load):
        """M(125) = 2500×75 = +187500 N·mm."""
        result = StaticsSolver().solve(simple_system_central_load)
        assert pytest.approx(value_at(result.M_xy, result.x, 125.0), rel=TOLERANCE_REL) == +187_500.0

    def test_M_xy_at_x200(self, simple_system_central_load):
        """M(200) = 2500×150 = +375000 N·mm."""
        result = StaticsSolver().solve(simple_system_central_load)
        assert pytest.approx(value_at(result.M_xy, result.x, 200.0), rel=TOLERANCE_REL) == +375_000.0

    def test_M_xy_at_x100(self, simple_system_central_load):
        """M(100) = 2500×50 = +125000 N·mm."""
        result = StaticsSolver().solve(simple_system_central_load)
        assert pytest.approx(value_at(result.M_xy, result.x, 100.0), rel=TOLERANCE_REL) == +125_000.0

    def test_M_xy_at_x275(self, simple_system_central_load):
        """M(275) = 2500×225 + 5000×75 = +937500 N·mm."""
        result = StaticsSolver().solve(simple_system_central_load)
        assert pytest.approx(value_at(result.M_xy, result.x, 275.0), rel=TOLERANCE_REL) == +937_500.0

    def test_M_xy_monotonically_increasing_in_first_half(self, simple_system_central_load):
        """M_xy cresce monotonicamente de xA até à carga (x=200)."""
        result = StaticsSolver().solve(simple_system_central_load)
        mask = (result.x >= 50.0) & (result.x <= 200.0)
        M_region = result.M_xy[mask]
        assert np.all(np.diff(M_region) >= -TOLERANCE_ABS_M)

    def test_V_xy_before_load(self, simple_system_central_load):
        """V(150) = -R_A = -2500 N."""
        result = StaticsSolver().solve(simple_system_central_load)
        assert pytest.approx(value_at(result.V_xy, result.x, 150.0), abs=TOLERANCE_ABS_V) == -2500.0

    def test_V_xy_after_load(self, simple_system_central_load):
        """V(250) = -(R_A+F) = -(2500+5000) = -7500 N."""
        result = StaticsSolver().solve(simple_system_central_load)
        assert pytest.approx(value_at(result.V_xy, result.x, 250.0), abs=TOLERANCE_ABS_V) == -7500.0


# ---------------------------------------------------------------------------
# CASO F — relação fundamental dM/dx = -V
#
# Com M += F*(x-pos) e V -= F:
#   dM/dx = +F  (força positiva → M cresce)
#   V     = -F  (força negativa)
#   → dM/dx = -V  ✓
#
# Entre x=50 e x=200 (simple_system_central_load):
#   V = -2500, ΔM = M(200)-M(50) = 375000-0 = +375000
#   -V×Δx = +2500×150 = +375000 ✓
#
# Entre x=200 e x=350:
#   V = -7500, ΔM = M(350)-M(200)
#   M(350) = 2500×300 + 5000×150 = 750000+750000 = 1500000
#   ΔM = 1500000-375000 = +1125000
#   -V×Δx = +7500×150 = +1125000 ✓
# ---------------------------------------------------------------------------

class TestCasoF_RelacaoVM:

    def test_dM_dx_equals_minus_V_first_segment(self, simple_system_central_load):
        """
        Entre x=50 e x=200: ΔM = +375000, -V×Δx = +2500×150 = +375000.
        """
        result = StaticsSolver().solve(simple_system_central_load)
        M_50  = value_at(result.M_xy, result.x, 50.0)
        M_200 = value_at(result.M_xy, result.x, 200.0)
        V_avg = value_at(result.V_xy, result.x, 125.0)
        delta_M = M_200 - M_50
        expected = -V_avg * (200.0 - 50.0)
        assert pytest.approx(delta_M, rel=TOLERANCE_REL) == expected

    def test_dM_dx_equals_minus_V_second_segment(self, simple_system_central_load):
        """
        Entre x=200 e x=350: ΔM = +1125000, -V×Δx = +7500×150 = +1125000.
        """
        result = StaticsSolver().solve(simple_system_central_load)
        M_200 = value_at(result.M_xy, result.x, 200.0)
        M_350 = value_at(result.M_xy, result.x, 350.0)
        V_avg = value_at(result.V_xy, result.x, 275.0)
        delta_M = M_350 - M_200
        expected = -V_avg * (350.0 - 200.0)
        assert pytest.approx(delta_M, rel=TOLERANCE_REL) == expected