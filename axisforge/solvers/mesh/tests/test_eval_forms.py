# test_eval_forms.py
"""Testes do registo de formas de avaliar (register_eval_form/
resolve_eval_form) e das 5 formas concretas já embutidas em
convergence_solver.py. Nenhuma forma sabe de que variável veio o
array -- verificado aqui contra arrays sintéticos, não contra
M_xz/v_xz reais."""
import math
import numpy as np
import pytest

from axisforge.solvers.mesh.convergence_solver import (
    register_eval_form, resolve_eval_form,
    _eval_max, _eval_mean, _eval_max_minus_mean,
    _eval_max_minus_mean_relative, _eval_at_point,
)


class TestEvalFormRegistry:
    def test_resolve_known_form(self):
        assert resolve_eval_form("max") is _eval_max
        assert resolve_eval_form("mean") is _eval_mean
        assert resolve_eval_form("max_minus_mean") is _eval_max_minus_mean
        assert resolve_eval_form("max_minus_mean_relative") is _eval_max_minus_mean_relative
        assert resolve_eval_form("at_point") is _eval_at_point

    def test_resolve_unknown_form_raises(self):
        with pytest.raises(ValueError, match="unknown eval form"):
            resolve_eval_form("does_not_exist")

    def test_register_duplicate_name_raises(self):
        @register_eval_form("test_only_form_xyz")
        def _f(values):
            return 0.0
        with pytest.raises(ValueError, match="already registered"):
            @register_eval_form("test_only_form_xyz")
            def _g(values):
                return 0.0


class TestBuiltinEvalForms:
    def test_max_takes_absolute_value(self):
        assert _eval_max(np.array([-5.0, 3.0, 1.0])) == 5.0

    def test_mean(self):
        assert math.isclose(_eval_mean(np.array([1.0, 2.0, 3.0])), 2.0)

    def test_max_minus_mean(self):
        values = np.array([10.0, -5.0, -5.0])  # mean=0, max(abs)=10
        assert math.isclose(_eval_max_minus_mean(values), 10.0)

    def test_max_minus_mean_relative(self):
        values = np.array([10.0, 5.0, 3.0])  # mean=6, max=10 -> (10-6)/6
        expected = (10.0 - 6.0) / 6.0
        assert math.isclose(_eval_max_minus_mean_relative(values), expected)

    def test_max_minus_mean_relative_zero_mean_raises(self):
        values = np.array([5.0, -5.0])  # mean = 0
        with pytest.raises(ValueError, match="mean is 0.0"):
            _eval_max_minus_mean_relative(values)

    def test_at_point_reads_matching_node(self):
        values = np.array([1.0, 2.0, 3.0])
        x_nodes = [0.0, 50.0, 100.0]
        assert _eval_at_point(values, x_nodes, 50.0) == 2.0

    def test_at_point_respects_tolerance(self):
        values = np.array([1.0, 2.0])
        x_nodes = [0.0, 10.0]
        assert _eval_at_point(values, x_nodes, 10.0 + 1e-9) == 2.0

    def test_at_point_missing_x_raises(self):
        values = np.array([1.0, 2.0])
        x_nodes = [0.0, 10.0]
        with pytest.raises(ValueError, match="not found in x_nodes"):
            _eval_at_point(values, x_nodes, 5.0)
