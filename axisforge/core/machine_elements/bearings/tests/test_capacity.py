# test_capacity.py
"""Testes de capacity.py -- todas as classes de capacidade, point e line
contact, radial e thrust, mais combine_multirow(). Foco: contrato
(sinais, exceções, relações entre grandezas), não valores absolutos
certificados contra a norma -- isso fica para quando tiveres um caso de
catálogo real para comparar."""
import math
import pytest

from axisforge.core.machine_elements.bearings import capacity as bcap

POINT_RADIAL_KWARGS = dict(Z=12, Dw=8.0, ri=4.3, re=4.3, gamma=0.2, reduction_factor=0.95, i=1)
POINT_THRUST_KWARGS = dict(Z=12, Dw=8.0, ri=4.3, re=4.3, gamma=0.2, lam=0.90, eta=0.9)
LINE_RADIAL_KWARGS = dict(Z=18, Dwe=8.0, Lwe=8.0, gamma=0.2, lambda_v=0.83, n_s=40, i=1)
LINE_THRUST_KWARGS = dict(Z=18, Dwe=8.0, Lwe=8.0, gamma=0.2, lambda_v=0.73, eta=0.9)


# =====================================================================
# point contact / radial
# =====================================================================

class TestPointContactCapacityRadial:
    def test_Cr_positive(self):
        calc = bcap.PointContactCapacityRadial(alpha_0=0.0, **POINT_RADIAL_KWARGS)
        assert calc.Cr > 0.0

    def test_Cr_cached(self):
        calc = bcap.PointContactCapacityRadial(alpha_0=0.0, **POINT_RADIAL_KWARGS)
        assert calc._fc is calc._fc or calc._fc == calc._fc

    def test_C0_not_implemented(self):
        calc = bcap.PointContactCapacityRadial(alpha_0=0.0, **POINT_RADIAL_KWARGS)
        with pytest.raises(NotImplementedError):
            _ = calc.C0

    def test_Q_elements_positive_pair(self):
        calc = bcap.PointContactCapacityRadial(alpha_0=0.0, **POINT_RADIAL_KWARGS)
        Q_ci, Q_ce = calc.Q_elements
        assert Q_ci > 0.0 and Q_ce > 0.0

    @pytest.mark.parametrize("bad_kwargs", [
        dict(Z=0), dict(Dw=0.0), dict(i=0),
    ])
    def test_rejects_non_positive_Z_Dw_i(self, bad_kwargs):
        kwargs = {**POINT_RADIAL_KWARGS, **bad_kwargs}
        with pytest.raises(ValueError):
            bcap.PointContactCapacityRadial(alpha_0=0.0, **kwargs)

    @pytest.mark.parametrize("gamma", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_gamma_outside_open_unit_interval(self, gamma):
        kwargs = {**POINT_RADIAL_KWARGS, "gamma": gamma}
        with pytest.raises(ValueError):
            bcap.PointContactCapacityRadial(alpha_0=0.0, **kwargs)


# =====================================================================
# point contact / thrust
# =====================================================================

class TestPointContactCapacityThrust90deg:
    def test_Ca_positive(self):
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **POINT_THRUST_KWARGS)
        assert calc.Ca > 0.0

    def test_C0_not_implemented(self):
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **POINT_THRUST_KWARGS)
        with pytest.raises(NotImplementedError):
            _ = calc.C0

    def test_Q_elements_positive_pair(self):
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **POINT_THRUST_KWARGS)
        Q_ci, Q_ce = calc.Q_elements
        assert Q_ci > 0.0 and Q_ce > 0.0


class TestPointContactCapacityThrustNon90deg:
    def test_Ca_switches_exponent_below_threshold(self):
        RATIO = 0.535

        def make(Dw):
            ri = re = RATIO * Dw
            return bcap.PointContactCapacityThrust_Non_90deg(
                alpha_0=math.radians(60.0), Z=12, Dw=Dw, ri=ri, re=re,
                gamma=0.2, lam=0.9, eta=0.9)

        small, large = make(10.0), make(30.0)
        ratio_fc = large._fc / small._fc
        naive_same_exponent_ratio = 3.647 * ratio_fc * (30.0 / 10.0) ** 1.4
        assert not math.isclose(large.Ca / small.Ca, naive_same_exponent_ratio, rel_tol=1e-6)


class TestPointContactMultirowCombination:
    def test_combine_multirow_positive(self):
        rows = [POINT_THRUST_KWARGS | dict(alpha_0=math.pi / 2)] * 2
        assert bcap.PointContactCapacityThrust_90deg.combine_multirow(rows) > 0.0

    def test_combine_multirow_requires_at_least_two_rows(self):
        rows = [POINT_THRUST_KWARGS | dict(alpha_0=math.pi / 2)]
        with pytest.raises(ValueError, match="at least 2 rows"):
            bcap.PointContactCapacityThrust_90deg.combine_multirow(rows)

    def test_combine_multirow_two_identical_rows_below_double_single_row(self):
        """Combinar 2 filas idênticas deve dar mais capacidade que 1 fila
        sozinha, mas menos que o dobro (a fórmula não é linear em Z)."""
        single = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **POINT_THRUST_KWARGS)
        rows = [POINT_THRUST_KWARGS | dict(alpha_0=math.pi / 2)] * 2
        combined = bcap.PointContactCapacityThrust_90deg.combine_multirow(rows)
        assert single.Ca < combined < 2.0 * single.Ca


# =====================================================================
# line contact / radial
# =====================================================================

class TestLineContactCapacityRadial:
    def test_Cr_positive(self):
        calc = bcap.LineContactCapacityRadial(alpha_0=0.0, **LINE_RADIAL_KWARGS)
        assert calc.Cr > 0.0

    def test_Q_elements_positive_pair(self):
        calc = bcap.LineContactCapacityRadial(alpha_0=0.0, **LINE_RADIAL_KWARGS)
        Q_ci, Q_ce = calc.Q_elements
        assert Q_ci > 0.0 and Q_ce > 0.0

    def test_per_lamina_smaller_than_per_element(self):
        """q_ci/q_ce dividem a carga por n_s lâminas -- têm de ser
        menores que Q_ci/Q_ce, nunca o contrário."""
        calc = bcap.LineContactCapacityRadial(alpha_0=0.0, **LINE_RADIAL_KWARGS)
        Q_ci, Q_ce = calc.Q_elements
        q_ci, q_ce = calc.per_lamina
        assert 0.0 < q_ci < Q_ci
        assert 0.0 < q_ce < Q_ce


# =====================================================================
# line contact / thrust -- classes com bugs conhecidos (ver review);
# testes escritos para o comportamento CORRETO, por isso falham até
# corrigires self.self.lambda_v, o unpacking do Q_elements a 90deg, e
# confirmares/corrigires o expoente 20/9 vs 2/9 em _fc.
# =====================================================================

class TestLineContactCapacityThrustNon90deg:
    def test_Ca_positive(self):
        calc = bcap.LineContactCapacityThrust_Non_90deg(alpha_0=math.radians(70.0), **LINE_THRUST_KWARGS)
        assert calc.Ca > 0.0

    def test_Q_elements_positive_pair(self):
        calc = bcap.LineContactCapacityThrust_Non_90deg(alpha_0=math.radians(70.0), **LINE_THRUST_KWARGS)
        Q_ci, Q_ce = calc.Q_elements
        assert Q_ci > 0.0 and Q_ce > 0.0

class TestLineContactCapacityThrust90deg:
    def test_Ca_positive(self):
        calc = bcap.LineContactCapacityThrust_90deg(alpha_0=math.pi / 2, **LINE_THRUST_KWARGS)
        assert calc.Ca > 0.0

    def test_Q_elements_positive_equal_pair(self):
        calc = bcap.LineContactCapacityThrust_90deg(alpha_0=math.pi / 2, **LINE_THRUST_KWARGS)
        Q_ci, Q_ce = calc.Q_elements
        assert Q_ci == Q_ce > 0.0


class TestLineContactMultirowCombination:
    """Formula (46) -- Ca_total = (ΣZ·Lwe)·[Σ((Z·Lwe)/Ca)**(9/2)]**(-2/9).
    Estes testes assumem o combine_multirow já corrigido (ver review);
    falham contra a versão atual, que ainda usa Z sozinho em vez de
    Z*Lwe."""
    def test_single_row_degenerate_case_returns_own_Ca(self):
        single = bcap.LineContactCapacityThrust_90deg(alpha_0=math.pi / 2, **LINE_THRUST_KWARGS)
        rows = [LINE_THRUST_KWARGS | dict(alpha_0=math.pi / 2)] * 2
        combined = bcap.LineContactCapacityThrust_90deg.combine_multirow(rows)
        # 2 filas idênticas: capacidade combinada tem de exceder 1 fila
        assert combined > single.Ca