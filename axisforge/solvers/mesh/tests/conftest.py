# conftest.py
"""Fixtures/helpers partilhados pelos testes de solvers/mesh/convergence_solver.py.

Não precisa de ShaftSystem/SubmodelSolver/Elem nenhum -- convergence_solver.py
não corre solves (essa decisão foi tomada nesta conversa: o loop de
grades e a resolução de pontos nomeados ficam num dispatcher fora
deste módulo). Os helpers aqui constroem SubmodelResult sintéticos
diretamente (a dataclass real) com valores desenhados para dar números
exatos e fáceis de verificar à mão contra as fórmulas do próprio
convergence_solver.py."""
from __future__ import annotations

import numpy as np

from axisforge.results.fem_results.submodel_results import SubmodelResult


def zeros(n: int) -> np.ndarray:
    return np.zeros(n)


def make_result(x_nodes, *, M_xz=None, M_xy=None, v_xz=None) -> SubmodelResult:
    """SubmodelResult sintético -- só os campos passados têm conteúdo
    não-nulo; o resto fica a zeros. lam_xz/lam_xy fixos em 6 elementos
    (tamanho do vetor de multiplicadores de Lagrange nos dois nós de
    corte, ver lagrange_multipliers.py::_build_constraint_matrix)."""
    n = len(x_nodes)
    return SubmodelResult(
        x_lo=x_nodes[0], x_hi=x_nodes[-1], grade="grade_x",
        x_nodes=list(x_nodes),
        u=zeros(n),
        v_xz=v_xz if v_xz is not None else zeros(n),
        v_xy=zeros(n),
        theta_xz=zeros(n), theta_xy=zeros(n),
        M_xz=M_xz if M_xz is not None else zeros(n),
        M_xy=M_xy if M_xy is not None else zeros(n),
        V_xz=zeros(n), V_xy=zeros(n),
        lam_xz=zeros(6), lam_xy=zeros(6),
    )


def array_with_exact_max_minus_mean(n: int, target: float) -> np.ndarray:
    """Constrói um array de n valores tal que
    max(abs(values)) - mean(values) == target exatamente, para n >= 3
    e target > 0: um valor = target, o resto = -target/(n-1) (soma
    zero -> mean=0 -> max_minus_mean = target - 0 = target, desde que
    target/(n-1) < target, verdade para n > 2)."""
    assert n >= 3, "array_with_exact_max_minus_mean requer n >= 3"
    values = np.full(n, -target / (n - 1))
    values[0] = target
    return values
