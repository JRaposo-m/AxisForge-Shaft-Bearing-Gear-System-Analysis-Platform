# conftest.py
"""Fixtures partilhadas pelos testes de fem_solvers/.

Âmbito desta primeira passagem (a pedido): verificar que o código *em
si* funciona e segue a sequência de chamadas que deve -- não validar a
física de Elem/ShaftSystem/Shaft/BeamModelSettings/materiais/loads
contra os standards, porque não vi o código-fonte de nenhuma dessas
classes nesta conversa. Onde este conftest precisa de algo desse tipo,
usa um duplo mínimo (Fake*) que implementa só o contrato que os
ficheiros de fem_solvers/ realmente chamam -- documentado em cada
classe. Troca por fixtures reais quando esses ficheiros estiverem
disponíveis (ver PENDENTE no final).

FakeElem usa as fórmulas-padrão de viga (Hermite cúbica para
Euler-Bernoulli, interpolação linear para Timoshenko) -- fórmulas de
livro, não copiadas do Elem real (nunca visto nesta conversa). Servem
para dar aos testes de element_postprocessing.py/build_stiffness_matrix.py/
distributed_loads.py um "elemento" fisicamente coerente com que
trabalhar, mas se o Elem real usar uma convenção de sinal/ordem de nó
diferente, estes testes não o vão apanhar -- são testes do NOSSO
código (o que foi escrito nesta conversa), não do Elem.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest


# ---------------------------------------------------------------------
# FakeElem -- duplo controlável de axisforge.mesh.shaft.element_type.Elem
# ---------------------------------------------------------------------

@dataclass
class FakeElem:
    """
    Duplo mínimo de Elem. Contrato coberto (o que
    element_postprocessing.py / build_stiffness_matrix.py /
    distributed_loads.py realmente chamam):

        atributos: E, I, length, x_a, x_b, idx_node_1, idx_node_2, beam_theory
        métodos:   natural_coordenates(x) -> zeta em [-1, 1]
                   bending_strain_matrix(zeta) -> B_b, shape (4,)
                   shape_functions(zeta) -> N, shape (4,) [euler] ou (2,) [timoshenko]
                   shear_strain_matrix(zeta) -> B_s (só timoshenko)
                   shear_rigidity(kGA_override=None) -> D_s (só timoshenko)
                   stiffness_element(kGA_override=None) -> K_bend 4x4
                   axial_stiffness_element() -> K_ax 2x2

    Fórmulas usadas (Hermite cúbica padrão, graus de liberdade locais
    [v_a, th_a, v_b, th_b], zeta em [-1,1], le = length):

        B_b(zeta) = [6*zeta/le**2, (3*zeta-1)/le, -6*zeta/le**2, (3*zeta+1)/le]
        K_bend = (E*I/le**3) * [[12, 6le, -12, 6le],
                                 [6le, 4le^2, -6le, 2le^2],
                                 [-12, -6le, 12, -6le],
                                 [6le, 2le^2, -6le, 4le^2]]

    Para timoshenko, shear_strain_matrix/shear_rigidity/shape_functions
    usam uma forma simplificada, linear em zeta, só para exercitar a
    sequência de chamadas -- não é uma dedução de viga de Timoshenko
    completa (fator de correção de corte incluído), o que precisaria do
    Elem real.
    """
    x_a: float
    x_b: float
    idx_node_1: int
    idx_node_2: int
    E: float = 210_000.0       # MPa
    I: float = 5_000.0         # mm^4
    beam_theory: str = "euler_bernoulli"
    kGA: float = 5.0e6         # N -- rigidez de corte "placeholder" p/ timoshenko

    @property
    def length(self) -> float:
        return self.x_b - self.x_a

    def natural_coordenates(self, x: float) -> float:
        return 2.0 * (x - self.x_a) / self.length - 1.0

    def bending_strain_matrix(self, zeta: float) -> np.ndarray:
        le = self.length
        return np.array([
            6.0 * zeta / le**2,
            (3.0 * zeta - 1.0) / le,
            -6.0 * zeta / le**2,
            (3.0 * zeta + 1.0) / le,
        ])

    def shape_functions(self, zeta: float) -> np.ndarray:
        if self.beam_theory == "euler_bernoulli":
            le = self.length
            N1 = 0.25 * (1 - zeta) ** 2 * (2 + zeta)
            N2 = le / 8.0 * (1 - zeta) ** 2 * (1 + zeta)
            N3 = 0.25 * (1 + zeta) ** 2 * (2 - zeta)
            N4 = le / 8.0 * (1 + zeta) ** 2 * (zeta - 1)
            return np.array([N1, N2, N3, N4])
        # timoshenko -- 2 funções nodais lineares (ver docstring da classe)
        return np.array([0.5 * (1 - zeta), 0.5 * (1 + zeta)])

    def shear_strain_matrix(self, zeta: float) -> np.ndarray:
        le = self.length
        # placeholder linear em zeta -- não é a dedução completa de Timoshenko
        return np.array([1.0 / le, 0.5, -1.0 / le, 0.5])

    def shear_rigidity(self, kGA_override: float | None = None) -> float:
        return kGA_override if kGA_override is not None else self.kGA

    def stiffness_element(self, kGA_override: float | None = None) -> np.ndarray:
        le, E, I = self.length, self.E, self.I
        k = E * I / le**3
        return k * np.array([
            [12.0,      6 * le,     -12.0,      6 * le],
            [6 * le,    4 * le**2,  -6 * le,    2 * le**2],
            [-12.0,     -6 * le,    12.0,       -6 * le],
            [6 * le,    2 * le**2,  -6 * le,    4 * le**2],
        ])

    def axial_stiffness_element(self) -> np.ndarray:
        le, E = self.length, self.E
        A = 100.0  # mm^2 -- placeholder, só para o bloco axial não ser nulo
        k = E * A / le
        return k * np.array([[1.0, -1.0], [-1.0, 1.0]])


# ---------------------------------------------------------------------
# malha de 2 elementos / 3 nós -- usada em quase todos os testes que
# precisam de "algum" FakeElem
# ---------------------------------------------------------------------

@pytest.fixture
def x_nodes_2elem():
    return [0.0, 50.0, 100.0]


@pytest.fixture
def euler_elements(x_nodes_2elem):
    xs = x_nodes_2elem
    return [
        FakeElem(x_a=xs[0], x_b=xs[1], idx_node_1=0, idx_node_2=1, beam_theory="euler_bernoulli"),
        FakeElem(x_a=xs[1], x_b=xs[2], idx_node_1=1, idx_node_2=2, beam_theory="euler_bernoulli"),
    ]


@pytest.fixture
def timoshenko_elements(x_nodes_2elem):
    xs = x_nodes_2elem
    return [
        FakeElem(x_a=xs[0], x_b=xs[1], idx_node_1=0, idx_node_2=1, beam_theory="timoshenko"),
        FakeElem(x_a=xs[1], x_b=xs[2], idx_node_1=1, idx_node_2=2, beam_theory="timoshenko"),
    ]


# ---------------------------------------------------------------------
# FakeSolver -- shape mínima que ElementTheoryPostProcessor espera
# (.elements, .d_total_xz, .d_total_xy, ._kGA_override opcional)
# ---------------------------------------------------------------------

@dataclass
class FakeSolver:
    elements: list
    d_total_xz: np.ndarray
    d_total_xy: np.ndarray
    _kGA_override: float | None = None


@pytest.fixture
def fake_solver_euler(euler_elements, x_nodes_2elem):
    n = len(x_nodes_2elem)
    rng = np.random.default_rng(0)
    d_xz = rng.normal(scale=0.01, size=3 * n)
    d_xy = rng.normal(scale=0.01, size=3 * n)
    return FakeSolver(elements=euler_elements, d_total_xz=d_xz, d_total_xy=d_xy)


@pytest.fixture
def fake_solver_timoshenko(timoshenko_elements, x_nodes_2elem):
    n = len(x_nodes_2elem)
    rng = np.random.default_rng(1)
    d_xz = rng.normal(scale=0.01, size=3 * n)
    d_xy = rng.normal(scale=0.01, size=3 * n)
    return FakeSolver(elements=timoshenko_elements, d_total_xz=d_xz, d_total_xy=d_xy, _kGA_override=None)


# ---------------------------------------------------------------------
# bearings/shaft_system fakes -- para boundary_conditions.py e
# global_postprocessing.py, sem precisar do ShaftSystem real
# ---------------------------------------------------------------------

@dataclass
class FakeBearing:
    position: float
    arrangement: str = "floating"   # "locating" ou "floating"
    label: str | None = None
    designation: str = "TEST-BRG"


@dataclass
class FakeShaft:
    """Duplo de Shaft -- só as 3 leituras que global_postprocessing.py e
    torsion.py fazem (diameter_at/W_at/Wt_at), constantes ao longo de x.
    NÃO implementa J_at -- ver test_torsion.py (KNOWN_GAP)."""
    d: float = 40.0
    W: float = 6_000.0
    Wt: float = 12_000.0

    def diameter_at(self, x: float) -> float:
        return self.d

    def W_at(self, x: float) -> float:
        return self.W

    def Wt_at(self, x: float) -> float:
        return self.Wt


@dataclass
class FakeShaftSystem:
    name: str = "test-shaft"
    shaft: FakeShaft = field(default_factory=FakeShaft)
    bearings: list = field(default_factory=list)
    torque_loads: list = field(default_factory=list)


@pytest.fixture
def fake_shaft_system_2brg(x_nodes_2elem):
    """2 chumaceiras, exatamente sobre nós da malha (obrigatório --
    _find_node()/Elem.find_node_index() exigem coincidência exata)."""
    xs = x_nodes_2elem
    return FakeShaftSystem(bearings=[
        FakeBearing(position=xs[0], arrangement="locating", label="brg-A"),
        FakeBearing(position=xs[-1], arrangement="floating", label="brg-B"),
    ])
