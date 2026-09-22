# test_distributed_loads.py
"""Testes de assembly/load_assembly/distributed_loads.py, com FakeElem
(conftest.py) -- exercita a sequência real (estimador de ordem de
Gauss -> integração -> scatter para o vetor global), com uma fórmula
fisicamente simples (carga uniforme) para poder verificar o resultado
contra um valor fechado conhecido para viga de Euler-Bernoulli:
carga distribuída uniforme q sobre um elemento inteiro produz forças
nodais consistentes q*le/2 em cada nó e momentos +-q*le^2/12 (fixed-end
moments clássicos)."""
import math
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.load_assembly.distributed_loads import (
    element_distributed_force_vector, assemble_distributed_load_vector,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.gauss_quadrature import (
    QuadratureOrderEstimator,
)


class TestElementDistributedForceVectorEuler:
    def test_no_overlap_returns_none(self, euler_elements):
        elem = euler_elements[0]  # [0, 50]
        estimator = QuadratureOrderEstimator("euler_bernoulli")
        result = element_distributed_force_vector(elem, 60.0, 80.0, lambda x: 1.0, estimator)
        assert result is None

    def test_uniform_load_matches_fixed_end_forces(self, euler_elements):
        """q(x) = q0 constante sobre TODO o elemento -> f_local deve
        bater com os "fixed-end forces/moments" clássicos de uma viga
        engastada-engastada sob carga uniforme:
            F_a = F_b = q0*le/2
            M_a = +q0*le**2/12 , M_b = -q0*le**2/12
        (sinal depende da convenção de N2/N4 usada no FakeElem -- ver
        conftest.py; verificado aqui contra essa mesma convenção)."""
        elem = euler_elements[0]  # le = 50
        q0 = 2.0  # N/mm
        estimator = QuadratureOrderEstimator("euler_bernoulli")
        f_local = element_distributed_force_vector(elem, elem.x_a, elem.x_b, lambda x: q0, estimator)

        le = elem.length
        F_expected = q0 * le / 2.0
        M_expected = q0 * le**2 / 12.0

        assert math.isclose(f_local[0], F_expected, rel_tol=1e-6)   # v_a
        assert math.isclose(f_local[2], F_expected, rel_tol=1e-6)   # v_b
        assert math.isclose(f_local[1], M_expected, rel_tol=1e-6)   # th_a
        assert math.isclose(f_local[3], -M_expected, rel_tol=1e-6)  # th_b

    def test_partial_overlap_clips_to_intersection(self, euler_elements):
        elem = euler_elements[0]  # [0, 50]
        estimator = QuadratureOrderEstimator("euler_bernoulli")
        # carga só definida em [25, 100] -> só a metade direita do elemento conta
        f_local = element_distributed_force_vector(elem, 25.0, 100.0, lambda x: 1.0, estimator)
        assert f_local is not None
        assert f_local.shape == (4,)


class TestElementDistributedForceVectorTimoshenko:
    def test_returns_shape_4_and_finite(self, timoshenko_elements):
        elem = timoshenko_elements[0]
        estimator = QuadratureOrderEstimator("timoshenko")
        f_local = element_distributed_force_vector(elem, elem.x_a, elem.x_b, lambda x: 3.0, estimator)
        assert f_local.shape == (4,)
        assert np.all(np.isfinite(f_local))

    def test_timoshenko_slot_pairs_share_same_weight(self, timoshenko_elements):
        """Por _SLOT_SHAPE_FUNCTION_INDEX, os slots 0/1 (nó a) usam o
        mesmo N[0], e os slots 2/3 (nó b) usam o mesmo N[1] -- para uma
        carga q constante isso implica f_local[0]==f_local[1] e
        f_local[2]==f_local[3]."""
        elem = timoshenko_elements[0]
        estimator = QuadratureOrderEstimator("timoshenko")
        f_local = element_distributed_force_vector(elem, elem.x_a, elem.x_b, lambda x: 5.0, estimator)
        assert math.isclose(f_local[0], f_local[1], rel_tol=1e-6)
        assert math.isclose(f_local[2], f_local[3], rel_tol=1e-6)


class TestAssembleDistributedLoadVector:
    def test_scatters_into_correct_global_dofs(self, euler_elements, x_nodes_2elem):
        estimator = QuadratureOrderEstimator("euler_bernoulli")
        cases = [{"x_lo": 0.0, "x_hi": 100.0, "q": lambda x: 1.0, "theta_fn": None}]
        f = assemble_distributed_load_vector(x_nodes_2elem, euler_elements, cases, estimator)
        n_dofs = 3 * len(x_nodes_2elem)
        assert f.shape == (n_dofs,)
        # axial nunca é tocado por carga distribuída
        assert f[0] == 0.0 and f[3] == 0.0 and f[6] == 0.0
        # nó do meio (partilhado pelos 2 elementos) deve acumular contribuições de ambos
        assert f[4] != 0.0  # v no nó 1

    def test_no_cases_returns_zero_vector(self, euler_elements, x_nodes_2elem):
        estimator = QuadratureOrderEstimator("euler_bernoulli")
        f = assemble_distributed_load_vector(x_nodes_2elem, euler_elements, [], estimator)
        assert np.allclose(f, 0.0)
