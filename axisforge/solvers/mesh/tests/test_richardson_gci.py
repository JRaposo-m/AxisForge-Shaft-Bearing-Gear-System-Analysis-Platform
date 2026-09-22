# test_richardson_gci.py
"""Testes de RichardsonGCI/_DummyGCI/_compute_gci -- verificados contra
as próprias fórmulas do módulo (não uma validação física
independente), com um caso de ordem 2 exata (p=2, r=2) escolhido para
dar números redondos."""
import math
import pytest

from axisforge.solvers.mesh.convergence_solver import RichardsonGCI, _DummyGCI, _compute_gci


class TestRichardsonGCI:
    def test_second_order_convergence_exact_p(self):
        """f(h) = f_exact + C*h**2, refinamento uniforme r=2 -- p
        calculado tem de sair exatamente 2.0."""
        f_exact, C = 100.0, 10.0
        f_coarse = f_exact + C * 1.0**2     # 110.0
        f_medium = f_exact + C * 0.5**2     # 102.5
        f_fine   = f_exact + C * 0.25**2    # 100.625

        x_c = list(range(3))   # n=2
        x_m = list(range(5))   # n=4
        x_f = list(range(9))   # n=8

        gci = RichardsonGCI(f_coarse, f_medium, f_fine, x_c, x_m, x_f)

        assert math.isclose(gci.r, 2.0)
        assert math.isclose(gci.p, 2.0, rel_tol=1e-9)

        e_m_c_expected = (f_medium - f_coarse) / f_coarse
        e_f_m_expected = (f_fine - f_medium) / f_medium
        assert math.isclose(gci.e_m_c, e_m_c_expected)
        assert math.isclose(gci.e_f_m, e_f_m_expected)

        GCI_m_c_expected = 1.25 * abs(e_m_c_expected) / (2.0**2.0 - 1)
        GCI_f_m_expected = 1.25 * abs(e_f_m_expected) / (2.0**2.0 - 1)
        assert math.isclose(gci.GCI_m_c, GCI_m_c_expected)
        assert math.isclose(gci.GCI_f_m, GCI_f_m_expected)

    def test_flat_metric_short_circuits_converged(self):
        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(9))
        gci = RichardsonGCI(50.0, 50.0, 50.0, x_c, x_m, x_f)
        assert gci.converged is True
        assert gci.GCI_m_c == 0.0 and gci.GCI_f_m == 0.0
        assert math.isnan(gci.p)

    def test_non_uniform_refinement_ratio_raises(self):
        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(11))  # r_m_c=2, r_f_m=2.5
        with pytest.raises(ValueError, match="Non-uniform refinement ratio"):
            RichardsonGCI(110.0, 102.5, 100.0, x_c, x_m, x_f)

    def test_refinement_ratio_not_greater_than_1_raises(self):
        x_c, x_m, x_f = list(range(5)), list(range(3)), list(range(9))  # medium mais grosseiro que coarse
        with pytest.raises(ValueError, match="Refinement ratio r_m_c"):
            RichardsonGCI(110.0, 102.5, 100.0, x_c, x_m, x_f)

    def test_non_monotonic_convergence_raises(self):
        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(9))
        with pytest.raises(ValueError, match="Non-monotonic convergence"):
            RichardsonGCI(100.0, 110.0, 105.0, x_c, x_m, x_f)

    def test_f_medium_equals_f_fine_raises(self):
        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(9))
        with pytest.raises(ValueError, match="f_medium == f_fine"):
            RichardsonGCI(110.0, 100.0, 100.0, x_c, x_m, x_f)

    def test_single_node_level_raises(self):
        x_c, x_m, x_f = [0.0], list(range(5)), list(range(9))
        with pytest.raises(ValueError, match="fewer than 2 nodes"):
            RichardsonGCI(110.0, 102.5, 100.0, x_c, x_m, x_f)

    def test_min_p_max_p_clamp(self):
        f_exact, C = 100.0, 10.0
        f_coarse = f_exact + C * 1.0**2
        f_medium = f_exact + C * 0.5**2
        f_fine   = f_exact + C * 0.25**2
        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(9))

        gci_clamped = RichardsonGCI(f_coarse, f_medium, f_fine, x_c, x_m, x_f, max_p=1.5)
        assert gci_clamped.p == 1.5

        gci_clamped_min = RichardsonGCI(f_coarse, f_medium, f_fine, x_c, x_m, x_f, min_p=3.0)
        assert gci_clamped_min.p == 3.0


class TestComputeGciFallback:
    def test_valid_inputs_return_real_gci(self):
        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(9))
        result = _compute_gci(110.0, 102.5, 100.625, x_c, x_m, x_f,
                               gci_threshold=0.01, safety_factor=1.25, min_p=None, max_p=None)
        assert isinstance(result, RichardsonGCI)

    def test_degenerate_inputs_fall_back_to_dummy(self):
        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(9))
        result = _compute_gci(100.0, 110.0, 105.0, x_c, x_m, x_f,
                               gci_threshold=0.01, safety_factor=1.25, min_p=None, max_p=None)
        assert isinstance(result, _DummyGCI)
        assert result.converged is False
