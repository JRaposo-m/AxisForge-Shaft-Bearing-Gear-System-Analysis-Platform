# test_gauss_quadrature.py
"""Testes de assembly/numerics/gauss_quadrature.py -- zero dependências
de Elem/ShaftSystem, testável de forma completa e verificada (não só
contrato): pontos/pesos de Gauss-Legendre são fórmulas fechadas
conhecidas, e q_degree/theta_degree/QuadratureOrderEstimator só
dependem de numpy."""
import math
import pytest
import numpy as np

from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.gauss_quadrature import (
    gauss_points_weights, q_degree, theta_degree, QuadratureOrderEstimator,
)


class TestGaussPointsWeights:
    def test_n1_midpoint_rule(self):
        pts, wts = gauss_points_weights(1)
        assert np.allclose(pts, [0.0])
        assert np.allclose(wts, [2.0])

    def test_n2_matches_closed_form(self):
        pts, wts = gauss_points_weights(2)
        s = 1.0 / math.sqrt(3)
        assert np.allclose(sorted(pts), [-s, s])
        assert np.allclose(wts, [1.0, 1.0])

    def test_n3_matches_closed_form(self):
        pts, wts = gauss_points_weights(3)
        s = math.sqrt(3 / 5)
        assert np.allclose(sorted(pts), [-s, 0.0, s])

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
    def test_weights_sum_to_2(self, n):
        """Integral de 1 sobre [-1, 1] é 2, para qualquer ordem."""
        _, wts = gauss_points_weights(n)
        assert math.isclose(wts.sum(), 2.0, abs_tol=1e-10)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
    def test_exact_for_polynomial_up_to_degree_2n_minus_1(self, n):
        """Regra de Gauss de ordem n integra exatamente qualquer
        polinómio até grau 2n-1. Verifica com x**(2n-1)."""
        pts, wts = gauss_points_weights(n)
        degree = 2 * n - 1
        numeric = float(np.sum(wts * pts**degree))
        # integral analítica de x**degree em [-1,1]: 0 se degree ímpar,
        # 2/(degree+1) se par
        exact = 0.0 if degree % 2 == 1 else 2.0 / (degree + 1)
        assert math.isclose(numeric, exact, abs_tol=1e-8)

    def test_n0_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            gauss_points_weights(0)


class TestDegreeEstimators:
    def test_q_degree_constant(self):
        assert q_degree(lambda x: 5.0, 0.0, 10.0) == 0

    def test_q_degree_linear(self):
        assert q_degree(lambda x: 2.0 * x + 1.0, 0.0, 10.0) == 1

    def test_q_degree_quadratic(self):
        assert q_degree(lambda x: x**2, -5.0, 5.0) == 2

    def test_theta_degree_constant_angle(self):
        assert theta_degree(lambda x: 30.0, 0.0, 10.0) == 0

    def test_theta_degree_varying_angle(self):
        # cos(theta(x)) varia de forma não-trivial -- só verifica que
        # devolve um grau plausível (0..3), não um valor exato
        d = theta_degree(lambda x: 90.0 * x / 10.0, 0.0, 10.0)
        assert 0 <= d <= 3


class TestQuadratureOrderEstimator:
    def test_unknown_beam_theory_raises(self):
        with pytest.raises(ValueError, match="unknown beam_theory"):
            QuadratureOrderEstimator("something_else")

    @pytest.mark.parametrize("theory,expected_deg_N", [
        ("euler_bernoulli", 3),
        ("timoshenko", 1),
    ])
    def test_known_shape_function_degree(self, theory, expected_deg_N):
        est = QuadratureOrderEstimator(theory)
        assert est._deg_N == expected_deg_N

    def test_gauss_order_constant_load_euler(self):
        """q constante (grau 0) + N cúbica (grau 3) -> p=3 -> order = ceil(4/2) = 2."""
        est = QuadratureOrderEstimator("euler_bernoulli")
        order = est.gauss_order(lambda x: 1.0, 0.0, 10.0)
        assert order == 2

    def test_gauss_order_never_below_1(self):
        est = QuadratureOrderEstimator("timoshenko")
        order = est.gauss_order(lambda x: 0.0, 0.0, 1.0)
        assert order >= 1

    def test_gauss_order_increases_with_theta_fn(self):
        est = QuadratureOrderEstimator("euler_bernoulli")
        order_no_theta = est.gauss_order(lambda x: x**2, 0.0, 10.0)
        order_with_theta = est.gauss_order(
            lambda x: x**2, 0.0, 10.0, theta_fn=lambda x: 45.0 * x / 10.0,
        )
        assert order_with_theta >= order_no_theta
