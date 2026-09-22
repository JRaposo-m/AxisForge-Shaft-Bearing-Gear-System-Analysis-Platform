# test_numerical_guards.py
"""Testes de assembly/numerics/numerical_guards.py -- puro numpy, sem
dependências de Elem/ShaftSystem."""
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.numerical_guards import (
    check_conditioning,
)


class TestCheckConditioning:
    def test_well_conditioned_matrix_does_not_raise(self):
        K = np.eye(4) * 1000.0
        check_conditioning(K)  # não deve levantar

    def test_singular_matrix_raises(self):
        K = np.zeros((3, 3))
        with pytest.raises(np.linalg.LinAlgError, match="near-\\)singular"):
            check_conditioning(K)

    def test_near_singular_above_threshold_raises(self):
        # matriz diagonal com um valor 1e15x mais pequeno que os outros
        K = np.diag([1.0, 1.0, 1e-15])
        with pytest.raises(np.linalg.LinAlgError):
            check_conditioning(K)

    def test_moderately_ill_conditioned_below_threshold_does_not_raise(self):
        # condição ~1e6, bem abaixo do limite de 1e14
        K = np.diag([1.0, 1.0, 1e-6])
        check_conditioning(K)
