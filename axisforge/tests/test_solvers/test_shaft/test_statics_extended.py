# tests/test_solvers/test_shaft/test_statics_extended.py
"""
StaticsSolver — testes aprofundados com verificação quantitativa completa.

Todos os valores esperados foram calculados manualmente e estão documentados
nos comentários de cada teste. O objectivo é verificar não só que o solver
"funciona" mas que produz os valores fisicamente correctos em cada ponto.

CASOS COBERTOS:
  A — simple_system: diagramas V, M, T completos ponto a ponto
  B — carga off-centre assimétrica: valores intermédios de M e V
  C — duas cargas em planos diferentes: M_res em pontos específicos
  D — overhang: documentação da limitação conhecida do solver Phase 1
  E — sinal das reacções: verificação explícita (não só magnitude)
  F — simetria do diagrama M para carga centrada

Tolerâncias:
  TOLERANCE_REL     = 0.005   (0.5% — validação textbook)
  TOLERANCE_ABS_M   = 200.0   (N·mm — valores intermédios com discretização)
  TOLERANCE_ABS_V   = 2.0     (N — esforço transverso)
  TOLERANCE_BOUNDARY = 500.0  (N·mm — condição de fronteira nos apoios)
"""
import pytest
import numpy as np
import warnings

from core.shaft import Shaft, ShaftSection
from core.components import Bearing, GearElement
from core.loads import RadialLoad, TorqueLoad, LoadPlane
from core.system import MechanicalSystem
from solvers.shaft.statics import StaticsSolver

TOLERANCE_REL = 0.005
TOLERANCE_ABS_M = 200.0    # N·mm — tolerância absoluta para momentos intermédios
TOLERANCE_ABS_V = 2.0      # N — tolerância absoluta para esforço transverso
TOLERANCE_BOUNDARY = 500.0  # N·mm — condição de fronteira nos apoios


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def value_at(array: np.ndarray, x_array: np.ndarray, x_target: float) -> float:
    """Interpola/extrai o valor do array no ponto x_target mais próximo."""
    idx = int(np.argmin(np.abs(x_array - x_target)))
    return float(array[idx])


def build_simple_shaft_system(
    shaft_length: float,
    xA: float,
    xB: float,
    diameter: float = 50.0,
) -> MechanicalSystem:
    """
    Constrói um sistema mínimo: shaft uniforme + 2 apoios sem cargas.
    Usado como base para adicionar cargas nos testes.
    """
    shaft = Shaft()
    shaft.add_section(ShaftSection(length=shaft_length, diameter=diameter))
    system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
    system._bearings.append(
        Bearing(position=xA, C=1.0, C0=1.0, arrangement="fixed", label="A")
    )
    system._bearings.append(
        Bearing(position=xB, C=1.0, C0=1.0, arrangement="floating", label="B")
    )
    return system


# ---------------------------------------------------------------------------
# CASO A — simple_system: diagramas completos ponto a ponto
# ---------------------------------------------------------------------------
#
# Geometria:
#   Shaft: 0 → 400mm
#   Apoio A: xA=50mm (fixed), Apoio B: xB=350mm (floating), Span=300mm
#   Gear: x=200mm → Wt=3500N (XZ), Wr=1274N (XY), T=175000 N·mm
#
# Reacções (calculadas manualmente, já validadas nos testes base):
#   No solver: R_A_xz = -1750N, R_B_xz = -1750N (negativas — upward)
#              R_A_xy = -637N,  R_B_xy = -637N
#   Nota: F_aplicada positiva (downward), reacção negativa (upward).
#
# Diagrama V_xz:
#   x < 50:          V = 0
#   50 ≤ x < 200:    V = R_A_xz = -1750 N
#   200 ≤ x < 350:   V = R_A_xz + Wt = -1750 + 3500 = +1750 N
#   x ≥ 350:         V = R_A_xz + Wt + R_B_xz = -1750 + 3500 - 1750 = 0 N
#
# Diagrama M_xz:
#   M(x) = R_A_xz × (x - 50) para 50 ≤ x ≤ 200
#   M(50)  =  -1750 × 0   =       0 N·mm
#   M(125) =  -1750 × 75  = -131250 N·mm
#   M(200) =  -1750 × 150 = -262500 N·mm  ← máximo absoluto
#
#   M(x) = R_A_xz×(x-50) + Wt×(x-200) para 200 ≤ x ≤ 350
#   M(275) = -1750×225 + 3500×75 = -393750 + 262500 = -131250 N·mm
#   M(350) = -1750×300 + 3500×150 = -525000 + 525000 = 0 N·mm  ← apoio B
#
# Diagrama M_xy (idêntico com Wr=1274N em vez de Wt=3500N):
#   M(125) = -637 × 75  = -47775 N·mm
#   M(200) = -637 × 150 = -95550 N·mm  ← máximo absoluto
#   M(275) = -637×225 + 1274×75 = -143325 + 95550 = -47775 N·mm
#
# M_res nos pontos críticos:
#   M_res(200) = sqrt(262500² + 95550²) = sqrt(68906250000 + 9129902500)
#              ≈ 279349 N·mm
#   M_res(125) = sqrt(131250² + 47775²) ≈ 139675 N·mm
#   M_res(275) = M_res(125) ≈ 139675 N·mm  (simetria da carga centrada)

class TestCasoA_DiagramasCompletos:
    """Verificação quantitativa de V, M, T em pontos específicos — simple_system."""

    # ── Esforço transverso V_xz ─────────────────────────────────────────

    def test_V_xz_before_support_A_is_zero(self, simple_system):
        """
        Antes do apoio A (x<50mm) não há nenhuma força → V_xz = 0.
        Ponto de teste: x=25mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        V_at_25 = value_at(result.V_xz, result.x, 25.0)
        # Esperado: 0 N (nenhuma carga ou reacção antes de x=50)
        assert abs(V_at_25) < TOLERANCE_ABS_V, (
            f"V_xz(25mm) = {V_at_25:.2f} N, esperado ≈ 0 N"
        )

    def test_V_xz_between_A_and_gear(self, simple_system):
        """
        Entre apoio A (x=50) e gear (x=200): V_xz = R_A_xz = -1750 N.
        Cálculo: só a reacção A actua. V = R_A_xz = -1750 N.
        Ponto de teste: x=125mm (midpoint entre A e gear).
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        V_at_125 = value_at(result.V_xz, result.x, 125.0)
        # Esperado: R_A_xz = -1750 N
        assert pytest.approx(V_at_125, abs=TOLERANCE_ABS_V) == -1750.0, (
            f"V_xz(125mm) = {V_at_125:.2f} N, esperado -1750 N"
        )

    def test_V_xz_between_gear_and_B(self, simple_system):
        """
        Entre gear (x=200) e apoio B (x=350): V_xz = R_A_xz + Wt = -1750 + 3500 = +1750 N.
        Ponto de teste: x=275mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        V_at_275 = value_at(result.V_xz, result.x, 275.0)
        # Esperado: +1750 N
        assert pytest.approx(V_at_275, abs=TOLERANCE_ABS_V) == +1750.0, (
            f"V_xz(275mm) = {V_at_275:.2f} N, esperado +1750 N"
        )

    def test_V_xz_after_support_B_is_zero(self, simple_system):
        """
        Após apoio B (x>350mm): todas as forças equilibradas → V_xz = 0.
        Ponto de teste: x=380mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        V_at_380 = value_at(result.V_xz, result.x, 380.0)
        # Esperado: 0 N
        assert abs(V_at_380) < TOLERANCE_ABS_V, (
            f"V_xz(380mm) = {V_at_380:.2f} N, esperado ≈ 0 N"
        )

    def test_V_xz_step_at_gear(self, simple_system):
        """
        Em x=200mm há um salto de V_xz de -1750 para +1750.
        Verifica que o salto ocorre (diferença ≈ Wt = 3500N).
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        V_before = value_at(result.V_xz, result.x, 190.0)  # antes do gear
        V_after  = value_at(result.V_xz, result.x, 210.0)  # depois do gear
        step = V_after - V_before
        # Esperado: salto de +3500 N (= Wt)
        assert pytest.approx(step, abs=10.0) == 3500.0, (
            f"Salto em V_xz no gear: {step:.1f} N, esperado 3500 N"
        )

    # ── Momento fletor M_xz — valores intermédios ────────────────────────

    def test_M_xz_at_x125(self, simple_system):
        """
        M_xz(125mm) = R_A_xz × (125 - 50) = -1750 × 75 = -131250 N·mm.
        x=125mm está entre apoio A (x=50) e gear (x=200).
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_at_125 = value_at(result.M_xz, result.x, 125.0)
        # Esperado: -131250 N·mm
        assert pytest.approx(M_at_125, rel=TOLERANCE_REL) == -131_250.0, (
            f"M_xz(125mm) = {M_at_125:.1f} N·mm, esperado -131250 N·mm"
        )

    def test_M_xz_at_x200_maximum(self, simple_system):
        """
        M_xz(200mm) = R_A_xz × (200 - 50) = -1750 × 150 = -262500 N·mm.
        Este é o máximo absoluto de M_xz (ponto de aplicação da carga).
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_at_200 = value_at(result.M_xz, result.x, 200.0)
        # Esperado: -262500 N·mm
        assert pytest.approx(M_at_200, rel=TOLERANCE_REL) == -262_500.0, (
            f"M_xz(200mm) = {M_at_200:.1f} N·mm, esperado -262500 N·mm"
        )

    def test_M_xz_at_x275(self, simple_system):
        """
        M_xz(275mm) = R_A_xz×(275-50) + Wt×(275-200)
                    = -1750×225 + 3500×75
                    = -393750 + 262500
                    = -131250 N·mm.
        Simetria: M_xz(275) deve ser igual a M_xz(125) — carga centrada.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_at_275 = value_at(result.M_xz, result.x, 275.0)
        M_at_125 = value_at(result.M_xz, result.x, 125.0)
        # Esperado: -131250 N·mm
        assert pytest.approx(M_at_275, rel=TOLERANCE_REL) == -131_250.0
        # Verificação de simetria: M(275) = M(125) para carga centrada
        assert pytest.approx(M_at_275, rel=TOLERANCE_REL) == M_at_125, (
            f"M_xz não simétrico: M(125)={M_at_125:.1f}, M(275)={M_at_275:.1f}"
        )

    # ── Momento fletor M_xy — valores intermédios ────────────────────────

    def test_M_xy_at_x125(self, simple_system):
        """
        M_xy(125mm) = R_A_xy × (125 - 50) = -637 × 75 = -47775 N·mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_at_125 = value_at(result.M_xy, result.x, 125.0)
        # Esperado: -47775 N·mm
        assert pytest.approx(M_at_125, rel=TOLERANCE_REL) == -47_775.0

    def test_M_xy_at_x200_maximum(self, simple_system):
        """
        M_xy(200mm) = R_A_xy × (200 - 50) = -637 × 150 = -95550 N·mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_at_200 = value_at(result.M_xy, result.x, 200.0)
        # Esperado: -95550 N·mm
        assert pytest.approx(M_at_200, rel=TOLERANCE_REL) == -95_550.0

    def test_M_xy_at_x275(self, simple_system):
        """
        M_xy(275mm) = R_A_xy×(275-50) + Wr×(275-200)
                    = -637×225 + 1274×75
                    = -143325 + 95550
                    = -47775 N·mm.
        Igual a M_xy(125) — simetria confirmada.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_at_275 = value_at(result.M_xy, result.x, 275.0)
        assert pytest.approx(M_at_275, rel=TOLERANCE_REL) == -47_775.0

    # ── M_res em pontos específicos ──────────────────────────────────────

    def test_M_res_at_x200(self, simple_system):
        """
        M_res(200mm) = sqrt(M_xz(200)² + M_xy(200)²)
                     = sqrt((-262500)² + (-95550)²)
                     = sqrt(68906250000 + 9129802500)
                     = sqrt(78036052500)
                     ≈ 279349 N·mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_res_at_200 = value_at(result.M_res, result.x, 200.0)
        expected = np.sqrt(262500.0**2 + 95550.0**2)  # ≈ 279349
        assert pytest.approx(M_res_at_200, rel=TOLERANCE_REL) == expected, (
            f"M_res(200mm) = {M_res_at_200:.1f} N·mm, esperado {expected:.1f}"
        )

    def test_M_res_at_x125_equals_x275(self, simple_system):
        """
        Para carga centrada no span, M_res deve ser simétrico:
        M_res(125) = M_res(275).
        M_res(125) = sqrt(131250² + 47775²) ≈ 139675 N·mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        M_res_125 = value_at(result.M_res, result.x, 125.0)
        M_res_275 = value_at(result.M_res, result.x, 275.0)
        expected = np.sqrt(131250.0**2 + 47775.0**2)  # ≈ 139675
        assert pytest.approx(M_res_125, rel=TOLERANCE_REL) == expected
        assert pytest.approx(M_res_275, rel=TOLERANCE_REL) == expected

    # ── Diagrama T (torção) ──────────────────────────────────────────────

    def test_T_zero_before_gear(self, simple_system):
        """
        Antes do gear (x=200mm) não há torque → T = 0.
        Ponto de teste: x=100mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        T_at_100 = value_at(result.T, result.x, 100.0)
        assert abs(T_at_100) < 1.0, f"T(100mm) = {T_at_100:.1f} N·mm, esperado 0"

    def test_T_at_x250(self, simple_system):
        """
        Entre gear (x=200) e apoio B (x=350): T = 175000 N·mm.
        Ponto de teste: x=250mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        T_at_250 = value_at(result.T, result.x, 250.0)
        # Esperado: 175000 N·mm
        assert pytest.approx(T_at_250, rel=TOLERANCE_REL) == 175_000.0

    def test_T_step_at_gear(self, simple_system):
        """
        O salto de torque em x=200mm deve ser igual ao torque do gear = 175000 N·mm.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        T_before = value_at(result.T, result.x, 190.0)  # antes do gear
        T_after  = value_at(result.T, result.x, 210.0)  # depois do gear
        step = T_after - T_before
        # Esperado: +175000 N·mm
        assert pytest.approx(step, rel=TOLERANCE_REL) == 175_000.0, (
            f"Salto em T no gear: {step:.1f} N·mm, esperado 175000 N·mm"
        )

    # ── Sinal explícito das reacções ─────────────────────────────────────

    def test_reactions_sign_A_xz_negative(self, simple_system):
        """
        Wt=+3500N (downward/forward) → R_A_xz deve ser NEGATIVO (upward/backward).
        R_A_xz = -1750 N.
        O solver usa equilibrio puro: R_A + R_B + Wt = 0.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        assert result.reactions["A_xz"] < 0, (
            f"R_A_xz deveria ser negativo, got {result.reactions['A_xz']:.2f}"
        )
        assert pytest.approx(result.reactions["A_xz"], rel=TOLERANCE_REL) == -1750.0

    def test_reactions_sign_A_xy_negative(self, simple_system):
        """
        Wr=+1274N → R_A_xy deve ser NEGATIVO = -637 N.
        """
        solver = StaticsSolver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(simple_system)
        assert result.reactions["A_xy"] < 0
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == -637.0


# ---------------------------------------------------------------------------
# CASO B — carga off-centre assimétrica
# ---------------------------------------------------------------------------
#
# Shaft: 0 → 500mm (d=50mm, secção única)
# Apoio A: xA=0mm, Apoio B: xB=500mm
# Carga: F=6000N (XY) a x=100mm
#
# Reacções:
#   R_B = -F × a / L = -6000 × 100 / 500 = -1200 N
#   R_A = -F - R_B = -6000 - (-1200) = -4800 N
#   Verificação: -4800 + (-1200) + 6000 = 0 ✓
#   ΣM_A: 6000×100 + (-1200)×500 = 600000 - 600000 = 0 ✓
#
# Diagrama V_xy:
#   0 ≤ x < 100mm:   V = R_A = -4800 N
#   100 ≤ x < 500mm: V = R_A + F = -4800 + 6000 = +1200 N
#   x ≥ 500mm:       V = 0 ✓
#
# Diagrama M_xy:
#   M(0)   =  0 N·mm  ✓ (apoio A)
#   M(50)  =  R_A × 50 = -4800 × 50 = -240000 N·mm
#   M(100) =  R_A × 100 = -4800 × 100 = -480000 N·mm  ← máximo
#   M(300) =  R_A×300 + F×200 = -1440000 + 1200000 = -240000 N·mm
#   M(500) =  R_A×500 + F×400 = -2400000 + 2400000 = 0 N·mm ✓ (apoio B)

class TestCasoB_OffCentreAssimetrico:
    """Carga assimétrica: verifica que o solver não assume simetria."""

    @pytest.fixture
    def system_off_centre(self):
        """Shaft 500mm, apoios nos extremos, F=6000N a x=100mm."""
        system = build_simple_shaft_system(
            shaft_length=500.0, xA=0.0, xB=500.0
        )
        system.add_load(RadialLoad(
            position=100.0, magnitude=6000.0, plane=LoadPlane.XY
        ))
        return system

    def test_reaction_A_off_centre(self, system_off_centre):
        """
        R_A = -4800 N.
        Cálculo: ΣM_B=0 → R_A×500 = -6000×400 → R_A = -4800 N.
        Nota: carga mais próxima de A → R_A maior em magnitude.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == -4800.0

    def test_reaction_B_off_centre(self, system_off_centre):
        """
        R_B = -1200 N.
        Carga mais distante de B → R_B menor em magnitude.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == -1200.0

    def test_reactions_are_not_equal(self, system_off_centre):
        """
        Carga assimétrica → R_A ≠ R_B. Confirmar que o solver não força simetria.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        assert abs(result.reactions["A_xy"]) != pytest.approx(
            abs(result.reactions["B_xy"]), rel=0.01
        )

    def test_V_xy_before_load(self, system_off_centre):
        """
        Entre xA=0 e x=100mm: V_xy = R_A = -4800 N.
        Ponto de teste: x=50mm.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        V_at_50 = value_at(result.V_xy, result.x, 50.0)
        assert pytest.approx(V_at_50, abs=TOLERANCE_ABS_V) == -4800.0

    def test_V_xy_after_load(self, system_off_centre):
        """
        Entre x=100mm e xB=500mm: V_xy = R_A + F = -4800 + 6000 = +1200 N.
        Ponto de teste: x=300mm.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        V_at_300 = value_at(result.V_xy, result.x, 300.0)
        assert pytest.approx(V_at_300, abs=TOLERANCE_ABS_V) == +1200.0

    def test_M_xy_at_x50(self, system_off_centre):
        """
        M_xy(50) = R_A × 50 = -4800 × 50 = -240000 N·mm.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        M_at_50 = value_at(result.M_xy, result.x, 50.0)
        assert pytest.approx(M_at_50, rel=TOLERANCE_REL) == -240_000.0

    def test_M_xy_maximum_at_x100(self, system_off_centre):
        """
        M_xy(100) = R_A × 100 = -4800 × 100 = -480000 N·mm ← máximo absoluto.
        Está em x=100mm (ponto de aplicação da carga).
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        M_at_100 = value_at(result.M_xy, result.x, 100.0)
        assert pytest.approx(M_at_100, rel=TOLERANCE_REL) == -480_000.0

    def test_M_xy_maximum_position_at_load(self, system_off_centre):
        """
        O máximo de |M_xy| deve ocorrer em x≈100mm (posição da carga).
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        idx_max = np.argmax(np.abs(result.M_xy))
        x_max = result.x[idx_max]
        assert 95.0 < x_max < 105.0, (
            f"Máximo de |M_xy| em x={x_max:.1f}mm, esperado perto de x=100mm"
        )

    def test_M_xy_at_x300(self, system_off_centre):
        """
        M_xy(300) = R_A×300 + F×200 = -1440000 + 1200000 = -240000 N·mm.
        Igual a M(50) — ponto simétrico em relação ao máximo? Não exactamente,
        mas a fórmula dá -240000 em ambos os pontos neste caso específico.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        M_at_300 = value_at(result.M_xy, result.x, 300.0)
        assert pytest.approx(M_at_300, rel=TOLERANCE_REL) == -240_000.0

    def test_M_xy_at_support_B_zero(self, system_off_centre):
        """
        M_xy(500) = R_A×500 + F×400 = -2400000 + 2400000 = 0 N·mm.
        Condição de fronteira do apoio B.
        """
        solver = StaticsSolver()
        result = solver.solve(system_off_centre)
        M_at_500 = value_at(result.M_xy, result.x, 500.0)
        assert abs(M_at_500) < TOLERANCE_BOUNDARY


# ---------------------------------------------------------------------------
# CASO C — duas cargas em planos diferentes
# ---------------------------------------------------------------------------
#
# Shaft: 0 → 600mm (d=50mm)
# Apoio A: xA=0mm, Apoio B: xB=600mm
# Carga 1: F1=4000N (XY) a x=200mm
# Carga 2: F2=3000N (XZ) a x=400mm
#
# Plano XY:
#   R_B_xy = -4000 × 200 / 600 = -1333.3 N
#   R_A_xy = -4000 + 1333.3 = -2666.7 N
#
# Plano XZ:
#   R_B_xz = -3000 × 400 / 600 = -2000 N
#   R_A_xz = -3000 + 2000 = -1000 N
#
# M_xy:
#   M_xy(200) = R_A_xy × 200 = -2666.7 × 200 = -533333 N·mm  ← max XY
#   M_xy(400) = R_A_xy×400 + 4000×200 = -1066667 + 800000 = -266667 N·mm
#
# M_xz:
#   M_xz(200) = R_A_xz × 200 = -1000 × 200 = -200000 N·mm
#   M_xz(400) = R_A_xz × 400 = -1000 × 400 = -400000 N·mm  ← max XZ
#
# M_res:
#   M_res(200) = sqrt(533333² + 200000²)
#              = sqrt(284444388889 + 40000000000)
#              = sqrt(324444388889) ≈ 569600 N·mm
#
#   M_res(400) = sqrt(266667² + 400000²)
#              = sqrt(71111555556 + 160000000000)
#              = sqrt(231111555556) ≈ 480741 N·mm
#
#   Máximo global: x=200mm com M_res ≈ 569600 N·mm

class TestCasoC_DuasCargas:
    """Dois cargas em planos diferentes — verificação do desacoplamento de planos e M_res."""

    @pytest.fixture
    def system_two_loads(self):
        """Shaft 600mm, apoios nos extremos, F1=4000N XY a x=200, F2=3000N XZ a x=400."""
        system = build_simple_shaft_system(
            shaft_length=600.0, xA=0.0, xB=600.0
        )
        system.add_load(RadialLoad(
            position=200.0, magnitude=4000.0, plane=LoadPlane.XY
        ))
        system.add_load(RadialLoad(
            position=400.0, magnitude=3000.0, plane=LoadPlane.XZ
        ))
        return system

    def test_reaction_A_xy(self, system_two_loads):
        """R_A_xy = -2666.7 N (só F1 actua no plano XY)."""
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        assert pytest.approx(result.reactions["A_xy"], rel=TOLERANCE_REL) == -2666.7

    def test_reaction_B_xy(self, system_two_loads):
        """R_B_xy = -1333.3 N."""
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        assert pytest.approx(result.reactions["B_xy"], rel=TOLERANCE_REL) == -1333.3

    def test_reaction_A_xz(self, system_two_loads):
        """R_A_xz = -1000 N (só F2 actua no plano XZ)."""
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        assert pytest.approx(result.reactions["A_xz"], rel=TOLERANCE_REL) == -1000.0

    def test_reaction_B_xz(self, system_two_loads):
        """R_B_xz = -2000 N."""
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        assert pytest.approx(result.reactions["B_xz"], rel=TOLERANCE_REL) == -2000.0

    def test_planes_are_decoupled_xy_has_no_xz_contamination(self, system_two_loads):
        """
        F1 está apenas no plano XY. M_xz entre x=0 e x=200 deve ser zero
        (F1 não contribui para M_xz).
        Ponto: x=100mm — só R_A_xz actua, que vem de F2 apenas.
        M_xz(100) = R_A_xz × 100 = -1000 × 100 = -100000 N·mm (não zero).
        Mas M_xy(100) = R_A_xy × 100 = -2666.7 × 100 = -266667 N·mm.
        Verifica desacoplamento: M_xz não é afectado por F1.
        """
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        # M_xz(100) só tem contribuição de R_A_xz (que vem de F2)
        M_xz_100 = value_at(result.M_xz, result.x, 100.0)
        assert pytest.approx(M_xz_100, rel=TOLERANCE_REL) == -100_000.0

    def test_M_xy_maximum_at_x200(self, system_two_loads):
        """
        M_xy(200) = R_A_xy × 200 = -2666.7 × 200 = -533333 N·mm ← máximo XY.
        """
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        M_xy_200 = value_at(result.M_xy, result.x, 200.0)
        assert pytest.approx(M_xy_200, rel=TOLERANCE_REL) == -533_333.0

    def test_M_xz_maximum_at_x400(self, system_two_loads):
        """
        M_xz(400) = R_A_xz × 400 = -1000 × 400 = -400000 N·mm ← máximo XZ.
        """
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        M_xz_400 = value_at(result.M_xz, result.x, 400.0)
        assert pytest.approx(M_xz_400, rel=TOLERANCE_REL) == -400_000.0

    def test_M_res_at_x200(self, system_two_loads):
        """
        M_res(200) = sqrt(533333² + 200000²) ≈ 569600 N·mm.
        """
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        M_res_200 = value_at(result.M_res, result.x, 200.0)
        expected = np.sqrt(533_333.0**2 + 200_000.0**2)
        assert pytest.approx(M_res_200, rel=TOLERANCE_REL) == expected

    def test_M_res_at_x400(self, system_two_loads):
        """
        M_res(400) = sqrt(266667² + 400000²) ≈ 480741 N·mm.
        """
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        M_res_400 = value_at(result.M_res, result.x, 400.0)
        expected = np.sqrt(266_667.0**2 + 400_000.0**2)
        assert pytest.approx(M_res_400, rel=TOLERANCE_REL) == expected

    def test_M_res_maximum_is_at_x200(self, system_two_loads):
        """
        M_res(200) ≈ 569600 > M_res(400) ≈ 480741 → máximo global em x=200mm.
        """
        solver = StaticsSolver()
        result = solver.solve(system_two_loads)
        x_max = result.x_at_M_res_max
        assert 190.0 < x_max < 210.0, (
            f"M_res máximo em x={x_max:.1f}mm, esperado perto de x=200mm"
        )


# ---------------------------------------------------------------------------
# CASO D — overhang: documentação da limitação conhecida
# ---------------------------------------------------------------------------
#
# Um veio com carga fora do span (overhang) é fisicamente válido mas o
# solver Phase 1 tem _validate_result que verifica M(xA) ≈ 0.
# Com overhang, M(xA) ≠ 0 — o solver vai lançar RuntimeError.
#
# Geometria:
#   Shaft: 0 → 400mm
#   Apoio A: xA=100mm, Apoio B: xB=350mm, Span=250mm
#   Carga: F=5000N (XY) a x=0mm (fora do span, à esquerda de A)
#
#   Reacções:
#   R_B = -F×(0-100)/250 = -5000×(-100)/250 = +2000 N
#   R_A = -F - R_B = -5000 - 2000 = -7000 N
#   Verificação ΣF: -7000 + 2000 + 5000 = 0 ✓
#   ΣM_A: 5000×(0-100) + 2000×250 = -500000 + 500000 = 0 ✓
#
#   M_xy(xA=100) = F×(100-0) = 5000×100 = 500000 N·mm ≠ 0
#   Isto é CORRECTO fisicamente mas viola a suposição de viga simplesmente
#   apoiada que o solver verifica em _validate_result.

class TestCasoD_Overhang:
    """
    Documenta o comportamento do solver para carga em overhang.
    Phase 1 não suporta overhang — o teste confirma e documenta isto.
    """

    @pytest.fixture
    def system_overhang(self):
        """Shaft 400mm, apoios em x=100 e x=350, carga em x=0 (overhang)."""
        shaft = Shaft()
        shaft.add_section(ShaftSection(length=400.0, diameter=50.0))
        system = MechanicalSystem(shaft=shaft, speed_rpm=0.0)
        system._bearings.append(
            Bearing(position=100.0, C=1.0, C0=1.0, arrangement="fixed", label="A")
        )
        system._bearings.append(
            Bearing(position=350.0, C=1.0, C0=1.0, arrangement="floating", label="B")
        )
        system.add_load(RadialLoad(
            position=0.0, magnitude=5000.0, plane=LoadPlane.XY
        ))
        return system

    def test_overhang_raises_runtime_error(self, system_overhang):
        """
        LIMITAÇÃO CONHECIDA Phase 1: carga em overhang causa M(xA) ≠ 0,
        o que viola a verificação de fronteira do solver e lança RuntimeError.

        Fisicamente: R_A=-7000N, R_B=+2000N, M(xA=100)=500000 N·mm ≠ 0.
        O solver detecta a violação e lança RuntimeError com mensagem descritiva.

        Phase 2+: implementar verificação de overhang e relaxar a condição
        de fronteira quando existem cargas fora do span.
        """
        solver = StaticsSolver()
        with pytest.raises(RuntimeError, match="Boundary condition violated"):
            solver.solve(system_overhang)

    def test_overhang_reactions_are_correct_before_validation(self, system_overhang):
        """
        Verifica que as reacções calculadas estão correctas antes da validação falhar.
        Usa _equilibrium directamente (método interno) para confirmar a física.

        R_A = -7000 N, R_B = +2000 N.
        """
        solver = StaticsSolver()
        loads = [(0.0, 5000.0)]   # carga em x=0
        xA, xB = 100.0, 350.0
        R_A, R_B = solver._equilibrium(loads, xA, xB)
        # Cálculo manual: R_B = -5000×(0-100)/250 = +2000, R_A = -5000-2000 = -7000
        assert pytest.approx(R_A, rel=TOLERANCE_REL) == -7000.0
        assert pytest.approx(R_B, rel=TOLERANCE_REL) == +2000.0


# ---------------------------------------------------------------------------
# CASO E — simetria de M para carga exactamente centrada
# ---------------------------------------------------------------------------
#
# Shaft: 0 → 400mm, apoios em x=50 e x=350, carga F=5000N XY a x=200mm
# (midpoint do span de 300mm)
#
# Simetria: M(xA + d) = M(xB - d) para qualquer d.
# M(125) = M(275), M(100) = M(300), etc.
#
# M(125) = R_A × 75 = -2500 × 75 = -187500 N·mm
# M(275) = R_A×225 + F×75 = -2500×225 + 5000×75 = -562500 + 375000 = -187500 ✓

class TestCasoE_SimetriaDiagrama:
    """Verifica simetria do diagrama M para carga centrada (simple_system_central_load)."""

    def test_M_xy_symmetric_about_midspan(self, simple_system_central_load):
        """
        Carga centrada em x=200mm (midpoint do span 50-350).
        M_xy deve ser simétrico: M(xA+d) = M(xB-d).

        Pares testados:
          M(125) = M(275)   → d=75mm
          M(100) = M(300)   → d=50mm

        M(125) = R_A×75 = -2500×75 = -187500 N·mm
        M(100) = R_A×50 = -2500×50 = -125000 N·mm
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)

        M_125 = value_at(result.M_xy, result.x, 125.0)
        M_275 = value_at(result.M_xy, result.x, 275.0)
        M_100 = value_at(result.M_xy, result.x, 100.0)
        M_300 = value_at(result.M_xy, result.x, 300.0)

        assert pytest.approx(M_125, rel=TOLERANCE_REL) == -187_500.0
        assert pytest.approx(M_275, rel=TOLERANCE_REL) == -187_500.0
        assert pytest.approx(M_125, rel=TOLERANCE_REL) == M_275

        assert pytest.approx(M_100, rel=TOLERANCE_REL) == -125_000.0
        assert pytest.approx(M_300, rel=TOLERANCE_REL) == -125_000.0
        assert pytest.approx(M_100, rel=TOLERANCE_REL) == M_300

    def test_M_xy_maximum_at_midspan(self, simple_system_central_load):
        """
        Carga centrada → M_xy máximo exactamente no meio do span = x=200mm.
        M_xy(200) = R_A×150 = -2500×150 = -375000 N·mm.
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        M_at_200 = value_at(result.M_xy, result.x, 200.0)
        assert pytest.approx(M_at_200, rel=TOLERANCE_REL) == -375_000.0

    def test_V_xy_antisymmetric(self, simple_system_central_load):
        """
        Para carga centrada, V_xy é anti-simétrico em relação ao ponto de carga:
          V antes da carga = R_A = -2500 N
          V após a carga   = R_A + F = -2500 + 5000 = +2500 N
        |V antes| = |V após| (mesma magnitude, sinal oposto).
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)
        V_before = value_at(result.V_xy, result.x, 150.0)  # entre A e carga
        V_after  = value_at(result.V_xy, result.x, 250.0)  # entre carga e B
        assert pytest.approx(V_before, abs=TOLERANCE_ABS_V) == -2500.0
        assert pytest.approx(V_after,  abs=TOLERANCE_ABS_V) == +2500.0
        assert pytest.approx(abs(V_before), rel=TOLERANCE_REL) == abs(V_after)


# ---------------------------------------------------------------------------
# CASO F — relação fundamental V(x) e M(x)
# ---------------------------------------------------------------------------
#
# Para uma viga com cargas pontuais, M(x) é a integral de V(x).
# Entre dois pontos sem carga aplicada (x1, x2):
#   M(x2) - M(x1) = integral de V(x) dx ≈ V × (x2 - x1)
#   (V é constante entre cargas pontuais)
#
# Usando simple_system_central_load:
#   Entre x=50 (apoio A) e x=200 (carga): V = -2500 N
#   M(200) - M(50) = -2500 × (200 - 50) = -2500 × 150 = -375000 N·mm
#   M(50) = 0 → M(200) = -375000 N·mm ✓

class TestCasoF_RelacaoVM:
    """Verifica a relação fundamental dM/dx = V entre pontos sem carga."""

    def test_integral_V_equals_delta_M(self, simple_system_central_load):
        """
        Entre x=50mm (apoio A) e x=200mm (carga):
          V_xy = -2500 N (constante)
          ΔM = M(200) - M(50) = V × Δx = -2500 × 150 = -375000 N·mm.

        Verifica a relação fundamental dM/dx = V numericamente.
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)

        M_50  = value_at(result.M_xy, result.x, 50.0)
        M_200 = value_at(result.M_xy, result.x, 200.0)
        V_avg = value_at(result.V_xy, result.x, 125.0)  # V constante neste intervalo

        delta_M = M_200 - M_50
        expected_delta_M = V_avg * (200.0 - 50.0)

        assert pytest.approx(delta_M, rel=TOLERANCE_REL) == expected_delta_M, (
            f"ΔM = {delta_M:.1f}, V×Δx = {expected_delta_M:.1f}"
        )

    def test_integral_V_equals_delta_M_second_segment(self, simple_system_central_load):
        """
        Entre x=200mm (carga) e x=350mm (apoio B):
          V_xy = +2500 N (constante após a carga)
          M(350) - M(200) = 2500 × 150 = +375000 N·mm
          M(200) = -375000, M(350) = 0 → ΔM = +375000 ✓
        """
        solver = StaticsSolver()
        result = solver.solve(simple_system_central_load)

        M_200 = value_at(result.M_xy, result.x, 200.0)
        M_350 = value_at(result.M_xy, result.x, 350.0)
        V_avg = value_at(result.V_xy, result.x, 275.0)  # V constante neste intervalo

        delta_M = M_350 - M_200
        expected_delta_M = V_avg * (350.0 - 200.0)

        assert pytest.approx(delta_M, rel=TOLERANCE_REL) == expected_delta_M
